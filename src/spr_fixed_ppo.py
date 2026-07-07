"""Fixed SPR-style encoder actor-critic for PPO baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.cv_extractor.spr_state import SPRStateNet
from src.ppo import mlp


class SPRFixedActorCritic(nn.Module):
    def __init__(
        self,
        encoder: SPRStateNet,
        goal_dim: int,
        act_dim: int,
        pi_hidden: list[int],
        vf_hidden: list[int],
        activation: type[nn.Module],
        init_std: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.goal_dim = int(goal_dim)
        self.act_dim = int(act_dim)
        obs_dim = int(encoder.z_dim) + self.goal_dim
        self.actor = mlp(obs_dim, pi_hidden, self.act_dim, activation)
        self.critic = mlp(obs_dim, vf_hidden, 1, activation)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), torch.log(torch.tensor(float(init_std)))))

    def obs(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder.encode(image).float()
        goal = goal.to(device=z.device, dtype=z.dtype)
        return torch.cat((z, goal), dim=1)

    def dist_from_obs(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, image: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self.obs(image, goal)
        dist = self.dist_from_obs(obs)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, logp, value

    def evaluate(self, image: torch.Tensor, goal: torch.Tensor, action: torch.Tensor):
        obs = self.obs(image, goal)
        dist = self.dist_from_obs(obs)
        logp = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        mean = self.actor(obs)
        return logp, entropy, value, mean

    def value(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.critic(self.obs(image, goal)).squeeze(-1)

    def predict(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.actor(self.obs(image, goal))
