"""SAC off-policy agent(与 src/rl/agent.py 的 DrQV2Agent 结构对齐,换成随机策略 + 最大熵)。

为什么用 SAC 而不是 DrQ-v2 的确定性 DDPG:确定性 μ 在 stop zone 里会 tanh 打满饱和,
探索噪声(noise_clip=0.3)拉不到 0,永远采不到 a_mag<STOP_EPS 的低动作 → 采不到停车样本
→ critic 学不出"停=高价值" → 冷启动死锁(见 spr_coadapt 2M 负结果)。SAC 的熵正则让策略
保持随机,zone 内天然能采到低动作;配合 HER(把 goal relabel 到实际到达位置)把这些低动作
样本组合成 in_zone & a<eps = success 样本喂给 critic,正面破死锁。

两种 obs_mode:
- "spr_z":冻结 SPR encoder 的 z 直接进 trunk(encoder 不在本模块);
- "pixels":存的 224 图 GPU resize → RandomShiftsAug → 小 CNN(critic 梯度训练,DrQ-v1 风格)。
不含 spr_coadapt(表征漂移是另一个问题,SAC+HER 先在稳定表征上验证机制)。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.agent import Critic, PixelEncoder, RandomShiftsAug, soft_update

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class StochasticActor(nn.Module):
    """tanh-squashed 对角高斯策略。trunk 与 DrQV2Agent.Actor 一致,输出头改成 mean+log_std。"""

    def __init__(self, repr_dim: int, goal_dim: int, act_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.net = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * act_dim),
        )
        self.act_dim = int(act_dim)

    def forward(self, repr_: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(repr_)
        mu, log_std = self.net(torch.cat((h, goal), dim=-1)).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, repr_: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (action, log_prob[B], mean_action)。action 已 tanh squash 到 [-1,1]。"""
        mu, log_std = self.forward(repr_, goal)
        std = log_std.exp()
        noise = torch.randn_like(mu)
        x = mu + noise * std  # reparameterized
        action = torch.tanh(x)
        # 对角高斯 log_prob + tanh 雅可比修正
        log_prob = (-0.5 * (noise ** 2) - log_std - 0.5 * math.log(2 * math.pi)).sum(dim=-1)
        log_prob = log_prob - torch.log(1.0 - action ** 2 + 1e-6).sum(dim=-1)
        return action, log_prob, torch.tanh(mu)


class SACAgent:
    def __init__(
        self,
        obs_mode: str,
        repr_dim: int,
        goal_dim: int,
        act_dim: int,
        feature_dim: int,
        hidden_dim: int,
        lr: float,
        tau: float,
        gamma: float,
        alpha_init: float,
        alpha_lr: float,
        target_entropy: float,
        noise_clip: float = 0.0,  # 兼容 agent_cfg 字段,SAC 不用
        pixels_res: int = 0,
        aug_pad: int = 0,
        spr_z_dim: int = 0,
        device: torch.device | str = "cuda",
    ):
        if obs_mode not in ("spr_z", "pixels"):
            raise ValueError(f"SACAgent 只支持 spr_z / pixels,收到:{obs_mode}")
        self.obs_mode = obs_mode
        self.device = torch.device(device)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.pixels_res = int(pixels_res)
        self.target_entropy = float(target_entropy)

        self.encoder = None
        self.encoder_opt = None
        self.aug = None
        if obs_mode == "pixels":
            self.encoder = PixelEncoder(3, self.pixels_res).to(self.device)
            self.aug = RandomShiftsAug(int(aug_pad))
            repr_dim = self.encoder.repr_dim
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.repr_dim = int(repr_dim)

        self.actor = StochasticActor(self.repr_dim, goal_dim, act_dim, feature_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.repr_dim, goal_dim, act_dim, feature_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.repr_dim, goal_dim, act_dim, feature_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha = torch.tensor(math.log(float(alpha_init)), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(alpha_lr))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _pixels_to_float(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device).permute(0, 3, 1, 2).float()
        if x.shape[-1] != self.pixels_res:
            x = F.interpolate(x, size=(self.pixels_res, self.pixels_res), mode="bilinear", align_corners=False)
        return x

    def encode_pixels(self, images: torch.Tensor, augment: bool) -> torch.Tensor:
        x = self._pixels_to_float(images)
        if augment:
            x = self.aug(x)
        return self.encoder(x)

    def obs_repr(self, batch_or_obs, augment: bool) -> torch.Tensor:
        if self.obs_mode == "spr_z":
            return batch_or_obs.to(self.device)
        return self.encode_pixels(batch_or_obs, augment=augment)

    @torch.no_grad()
    def act(self, obs, goal: torch.Tensor, eval_mode: bool) -> torch.Tensor:
        repr_ = self.obs_repr(obs, augment=False)
        action, _logp, mean_a = self.actor.sample(repr_, goal.to(self.device))
        return mean_a if eval_mode else action

    def update(self, batch: dict[str, torch.Tensor], sigma: float = 0.0) -> dict[str, float]:
        goal = batch["goal"]
        next_goal = batch["next_goal"]
        action = batch["action"]
        reward = batch["reward"]
        discount = batch["discount"]

        if self.obs_mode == "spr_z":
            repr_ = batch["z"]
            with torch.no_grad():
                next_repr = batch["next_z"]
        else:
            repr_ = self.encode_pixels(batch["obs"], augment=True)
            with torch.no_grad():
                next_repr = self.encode_pixels(batch["next_obs"], augment=True)

        alpha = self.alpha.detach()
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_repr, next_goal)
            tq1, tq2 = self.critic_target(next_repr, next_goal, next_a)
            target_v = torch.min(tq1, tq2) - alpha * next_logp
            target_q = reward + discount * target_v

        q1, q2 = self.critic(repr_, goal, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        repr_d = repr_.detach()
        a, logp, _ = self.actor.sample(repr_d, goal)
        aq1, aq2 = self.critic(repr_d, goal, a)
        actor_loss = (alpha * logp - torch.min(aq1, aq2)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        soft_update(self.critic_target, self.critic, self.tau)
        with torch.no_grad():
            a_mag = a.abs().max(dim=-1).values.mean()
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "q1_mean": float(q1.detach().mean().cpu()),
            "q_target_mean": float(target_q.detach().mean().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "entropy": float((-logp).detach().mean().cpu()),
            "a_mag_batch": float(a_mag.cpu()),
        }

    def save_dict(self) -> dict:
        out = {
            "obs_mode": self.obs_mode,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.encoder is not None:
            out["encoder"] = self.encoder.state_dict()
        return out

    def load_dict(self, ckpt: dict) -> None:
        self.actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
        if "critic_target" in ckpt:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        if "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        if self.encoder is not None:
            self.encoder.load_state_dict(ckpt["encoder"])
