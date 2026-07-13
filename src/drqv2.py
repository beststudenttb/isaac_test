"""DrQ-v2 style off-policy agent: DDPG + n-step TD + twin critic + scheduled noise.

不依赖 Isaac,可离线单测。两种 obs 模式:
- "spr_z": 冻结 SPR encoder 的 z 直接进 trunk(encoder 不在本模块内);
- "pixels": 存储的 224 图 GPU 上 resize -> RandomShiftsAug -> 小 CNN(critic 梯度训练)。
σ 是外部调度传入的(探索与 target smoothing 共用),不是可学参数——这是与 PPO 线的关键区别。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    if duration <= 0:
        return float(end)
    frac = min(max(step / duration, 0.0), 1.0)
    return float(start + (end - start) * frac)


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


class RandomShiftsAug(nn.Module):
    """DrQ-v2 原版随机平移增广(replicate pad + bilinear grid sample)。"""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.size()
        assert h == w
        padding = (self.pad,) * 4
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        return F.grid_sample(x, base_grid + shift, padding_mode="zeros", align_corners=False)


class PixelEncoder(nn.Module):
    """DrQ-v2 原版 4 层卷积。输入 float32 [0,255] 的 NCHW,内部归一化。"""

    def __init__(self, in_ch: int, res: int):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, res, res)
            self.repr_dim = int(self.convnet(dummy).flatten(1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 - 0.5
        return self.convnet(x).flatten(1)


class Actor(nn.Module):
    def __init__(self, repr_dim: int, goal_dim: int, act_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.policy = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, repr_: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        h = self.trunk(repr_)
        return torch.tanh(self.policy(torch.cat((h, goal), dim=-1)))


class Critic(nn.Module):
    def __init__(self, repr_dim: int, goal_dim: int, act_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        in_dim = feature_dim + goal_dim + act_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, repr_: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(repr_)
        x = torch.cat((h, goal, action), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class DrQV2Agent:
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
        noise_clip: float,
        pixels_res: int = 0,
        aug_pad: int = 0,
        device: torch.device | str = "cuda",
    ):
        if obs_mode not in ("spr_z", "pixels"):
            raise ValueError(f"unknown obs_mode: {obs_mode}")
        self.obs_mode = obs_mode
        self.device = torch.device(device)
        self.tau = float(tau)
        self.noise_clip = float(noise_clip)
        self.pixels_res = int(pixels_res)

        self.encoder = None
        self.encoder_opt = None
        self.aug = None
        if obs_mode == "pixels":
            self.encoder = PixelEncoder(3, self.pixels_res).to(self.device)
            self.aug = RandomShiftsAug(int(aug_pad))
            repr_dim = self.encoder.repr_dim
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.repr_dim = int(repr_dim)

        self.actor = Actor(self.repr_dim, goal_dim, act_dim, feature_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.repr_dim, goal_dim, act_dim, feature_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.repr_dim, goal_dim, act_dim, feature_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

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
        """采集/评估用:z 直接返回,像素过 encoder(不增广)。"""
        if self.obs_mode == "spr_z":
            return batch_or_obs.to(self.device)
        return self.encode_pixels(batch_or_obs, augment=augment)

    @torch.no_grad()
    def act(self, obs, goal: torch.Tensor, sigma: float, eval_mode: bool) -> torch.Tensor:
        repr_ = self.obs_repr(obs, augment=False)
        mu = self.actor(repr_, goal.to(self.device))
        if eval_mode:
            return mu
        return (mu + float(sigma) * torch.randn_like(mu)).clamp(-1.0, 1.0)

    def update(self, batch: dict[str, torch.Tensor], sigma: float) -> dict[str, float]:
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

        with torch.no_grad():
            next_mu = self.actor(next_repr, next_goal)
            noise = (torch.randn_like(next_mu) * float(sigma)).clamp(-self.noise_clip, self.noise_clip)
            next_a = (next_mu + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(next_repr, next_goal, next_a)
            target_q = reward + discount * torch.min(tq1, tq2)

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
        mu = self.actor(repr_d, goal)
        noise = (torch.randn_like(mu) * float(sigma)).clamp(-self.noise_clip, self.noise_clip)
        a = (mu + noise).clamp(-1.0, 1.0)
        aq1, aq2 = self.critic(repr_d, goal, a)
        actor_loss = -torch.min(aq1, aq2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.critic_target, self.critic, self.tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "q1_mean": float(q1.detach().mean().cpu()),
            "q_target_mean": float(target_q.detach().mean().cpu()),
        }

    def save_dict(self) -> dict:
        out = {
            "obs_mode": self.obs_mode,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        if self.encoder is not None:
            out["encoder"] = self.encoder.state_dict()
        return out

    def load_dict(self, ckpt: dict) -> None:
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        if self.encoder is not None:
            self.encoder.load_state_dict(ckpt["encoder"])
