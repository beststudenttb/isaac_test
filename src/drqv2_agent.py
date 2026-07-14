"""Standalone DrQ-v2 agent for goal-conditioned visual continuous control.

Core algorithm adapted from facebookresearch/drqv2 under the MIT license.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    if duration <= 0:
        return float(end)
    mix = min(max(float(step) / float(duration), 0.0), 1.0)
    return float((1.0 - mix) * start + mix * end)


def weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.lerp_(online_param.data, float(tau))


class TruncatedNormal(pyd.Normal):
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__(loc, scale, validate_args=False)
        self.low = float(low)
        self.high = float(high)
        self.eps = float(eps)

    def _clamp(self, value: torch.Tensor) -> torch.Tensor:
        clamped = value.clamp(self.low + self.eps, self.high - self.eps)
        return value - value.detach() + clamped.detach()

    def sample(
        self,
        clip: float | None = None,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        noise = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        noise = noise * self.scale
        if clip is not None:
            noise = noise.clamp(-float(clip), float(clip))
        return self._clamp(self.loc + noise)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = int(pad)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = obs.shape
        if height != width:
            raise ValueError(f"random shift requires square observations: {obs.shape}")

        obs = F.pad(obs, (self.pad,) * 4, mode="replicate")
        padded = height + 2 * self.pad
        eps = 1.0 / padded
        coords = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            padded,
            device=obs.device,
            dtype=obs.dtype,
        )[:height]
        coords = coords.unsqueeze(0).repeat(height, 1).unsqueeze(2)
        base_grid = torch.cat((coords, coords.transpose(1, 0)), dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(batch, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(batch, 1, 1, 2),
            device=obs.device,
        ).to(obs.dtype)
        shift *= 2.0 / padded
        return F.grid_sample(
            obs,
            base_grid + shift,
            padding_mode="zeros",
            align_corners=False,
        )


class Encoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int]):
        super().__init__()
        channels, height, width = obs_shape
        if height != width:
            raise ValueError(f"encoder requires square observations: {obs_shape}")

        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.apply(weight_init)
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            self.repr_dim = int(self.convnet(dummy).flatten(1).shape[1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs / 255.0 - 0.5
        return self.convnet(obs).flatten(1)


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        goal_dim: int,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(weight_init)

    def forward(
        self,
        encoded_obs: torch.Tensor,
        goal: torch.Tensor,
        stddev: float,
    ) -> TruncatedNormal:
        features = self.trunk(encoded_obs)
        mean = torch.tanh(self.policy(torch.cat((features, goal), dim=-1)))
        std = torch.ones_like(mean) * float(stddev)
        return TruncatedNormal(mean, std)


class Critic(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        goal_dim: int,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        input_dim = feature_dim + goal_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(
        self,
        encoded_obs: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(encoded_obs)
        inputs = torch.cat((features, goal, action), dim=-1)
        return self.q1(inputs), self.q2(inputs)


class DrQV2Agent:
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        goal_dim: int,
        action_dim: int,
        lr: float,
        feature_dim: int,
        hidden_dim: int,
        critic_target_tau: float,
        num_expl_steps: int,
        stddev_start: float,
        stddev_end: float,
        stddev_duration: int,
        stddev_clip: float,
        aug_pad: int,
        device: torch.device | str,
    ):
        self.device = torch.device(device)
        self.critic_target_tau = float(critic_target_tau)
        self.num_expl_steps = int(num_expl_steps)
        self.stddev_start = float(stddev_start)
        self.stddev_end = float(stddev_end)
        self.stddev_duration = int(stddev_duration)
        self.stddev_clip = float(stddev_clip)

        self.encoder = Encoder(obs_shape).to(self.device)
        self.actor = Actor(
            self.encoder.repr_dim,
            goal_dim,
            action_dim,
            feature_dim,
            hidden_dim,
        ).to(self.device)
        self.critic = Critic(
            self.encoder.repr_dim,
            goal_dim,
            action_dim,
            feature_dim,
            hidden_dim,
        ).to(self.device)
        self.critic_target = Critic(
            self.encoder.repr_dim,
            goal_dim,
            action_dim,
            feature_dim,
            hidden_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.aug = RandomShiftsAug(aug_pad)

    def stddev(self, step: int) -> float:
        return linear_schedule(
            self.stddev_start,
            self.stddev_end,
            self.stddev_duration,
            step,
        )

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        step: int,
        eval_mode: bool,
    ) -> torch.Tensor:
        obs = obs.to(self.device, non_blocking=True).float()
        goal = goal.to(self.device, non_blocking=True).float()
        encoded = self.encoder(obs)
        dist = self.actor(encoded, goal, self.stddev(step))
        if eval_mode:
            return dist.mean
        action = dist.sample(clip=None)
        if step < self.num_expl_steps:
            action.uniform_(-1.0, 1.0)
        return action

    def update(
        self,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> dict[str, float]:
        obs = self.aug(batch["obs"].float())
        next_obs = self.aug(batch["next_obs"].float())
        goal = batch["goal"].float()
        next_goal = batch["next_goal"].float()
        action = batch["action"].float()
        reward = batch["reward"].float()
        discount = batch["discount"].float()

        encoded = self.encoder(obs)
        with torch.no_grad():
            next_encoded = self.encoder(next_obs)
            next_dist = self.actor(next_encoded, next_goal, self.stddev(step))
            next_action = next_dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(
                next_encoded,
                next_goal,
                next_action,
            )
            target_q = reward + discount * torch.min(target_q1, target_q2)

        q1, q2 = self.critic(encoded, goal, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        dist = self.actor(encoded.detach(), goal, self.stddev(step))
        sampled_action = dist.sample(clip=self.stddev_clip)
        actor_q1, actor_q2 = self.critic(encoded.detach(), goal, sampled_action)
        actor_loss = -torch.min(actor_q1, actor_q2).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        soft_update(self.critic_target, self.critic, self.critic_target_tau)

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "q1": float(q1.detach().mean()),
            "q2": float(q2.detach().mean()),
            "target_q": float(target_q.detach().mean()),
            "stddev": self.stddev(step),
        }

    def checkpoint(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }

    def load_checkpoint(self, checkpoint: dict) -> None:
        """critic/critic_target 缺失时只加载策略(归档用的 eval-only checkpoint,见 approved_models)。"""
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        if "critic_target" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target"])

    def set_training(self, training: bool) -> None:
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)
