"""Small CNN actor-critic for visual PPO baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.ppo import mlp


class CNNFeature(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, int(z_dim)),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(next(self.parameters()).device)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            image = image.float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()
        return self.net(image)


class CNNActorCritic(nn.Module):
    def __init__(
        self,
        z_dim: int,
        goal_dim: int,
        act_dim: int,
        pi_hidden: list[int],
        vf_hidden: list[int],
        activation: type[nn.Module],
        init_std: float,
    ):
        super().__init__()
        self.act_dim = int(act_dim)
        obs_dim = int(z_dim) + int(goal_dim)
        self.encoder = CNNFeature(int(z_dim))
        self.actor = mlp(obs_dim, pi_hidden, self.act_dim, activation)
        self.critic = mlp(obs_dim, vf_hidden, 1, activation)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), torch.log(torch.tensor(float(init_std)))))

    def obs(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = self.encoder(image)
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


@dataclass
class CNNBatch:
    images: torch.Tensor
    goals: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    returns: torch.Tensor
    adv: torch.Tensor
    values: torch.Tensor
    teacher_actions: torch.Tensor | None = None


class CNNRollout:
    def __init__(
        self,
        steps: int,
        envs: int,
        image_shape: tuple[int, int, int],
        goal_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        self.steps = int(steps)
        self.envs = int(envs)
        self.images = torch.empty((steps, envs, *image_shape), dtype=torch.uint8, device="cpu")
        self.goals = torch.zeros((steps, envs, goal_dim), device=device)
        self.actions = torch.zeros((steps, envs, act_dim), device=device)
        self.logp = torch.zeros((steps, envs), device=device)
        self.rewards = torch.zeros((steps, envs), device=device)
        self.dones = torch.zeros((steps, envs), device=device)
        self.values = torch.zeros((steps, envs), device=device)

    def image_to_uint8(self, image: torch.Tensor) -> torch.Tensor:
        image = image.detach()
        if image.dtype == torch.uint8:
            return image.cpu()
        if float(image.max().detach().cpu()) <= 1.0:
            image = image * 255.0
        return torch.clamp(image, 0.0, 255.0).to(torch.uint8).cpu()

    def add(self, t: int, image, goal, action, logp, reward, done, value) -> None:
        self.images[t].copy_(self.image_to_uint8(image))
        self.goals[t].copy_(goal.detach())
        self.actions[t].copy_(action.detach())
        self.logp[t].copy_(logp.detach())
        self.rewards[t].copy_(reward.detach())
        self.dones[t].copy_(done.float())
        self.values[t].copy_(value.detach())

    def make_batch(self, last_value: torch.Tensor, gamma: float, lam: float, norm_adv: bool) -> CNNBatch:
        adv = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.envs, device=self.rewards.device)
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_not_done - self.values[t]
            last_gae = delta + gamma * lam * next_not_done * last_gae
            adv[t] = last_gae
        returns = adv + self.values

        images = self.images.reshape(-1, *self.images.shape[2:])
        goals = self.goals.reshape(-1, self.goals.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        logp = self.logp.reshape(-1)
        values = self.values.reshape(-1)
        adv = adv.reshape(-1)
        returns = returns.reshape(-1)
        if norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return CNNBatch(images=images, goals=goals, actions=actions, logp=logp, returns=returns, adv=adv, values=values)


def cnn_ppo_update(
    model: CNNActorCritic,
    opt: torch.optim.Optimizer,
    batch: CNNBatch,
    batch_size: int,
    epochs: int,
    clip: float,
    vf_coef: float,
    ent_coef: float,
    teacher_coef: float,
    max_grad_norm: float,
    clip_vf: float | None,
    log_std_max: float,
) -> dict[str, float]:
    n = batch.actions.shape[0]
    total_pi = 0.0
    total_v = 0.0
    total_ent = 0.0
    total_kl = 0.0
    total_clip = 0.0
    total_teacher = 0.0
    total_gn = 0.0
    updates = 0
    device = batch.actions.device
    for _ in range(int(epochs)):
        ids = torch.randperm(n, device=device)
        for start in range(0, n, int(batch_size)):
            idx = ids[start : start + int(batch_size)]
            idx_cpu = idx.cpu()
            logp, entropy, value, mean = model.evaluate(batch.images[idx_cpu], batch.goals[idx], batch.actions[idx])
            ratio = torch.exp(logp - batch.logp[idx])
            pi_loss = torch.max(
                -batch.adv[idx] * ratio,
                -batch.adv[idx] * torch.clamp(ratio, 1.0 - clip, 1.0 + clip),
            ).mean()
            if clip_vf is None:
                v_loss = 0.5 * (value - batch.returns[idx]).pow(2).mean()
            else:
                v_clip = batch.values[idx] + torch.clamp(value - batch.values[idx], -clip_vf, clip_vf)
                v_loss = 0.5 * torch.max((value - batch.returns[idx]).pow(2), (v_clip - batch.returns[idx]).pow(2)).mean()
            ent = entropy.mean()
            teacher_loss = torch.zeros((), device=device)
            if teacher_coef > 0.0 and batch.teacher_actions is not None:
                teacher_loss = (mean - batch.teacher_actions[idx]).pow(2).mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent + teacher_coef * teacher_loss
            opt.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                model.log_std.clamp_(max=log_std_max)
            with torch.no_grad():
                approx_kl = (batch.logp[idx] - logp).mean()
                clip_frac = ((ratio - 1.0).abs() > clip).float().mean()
            total_pi += float(pi_loss.detach().cpu())
            total_v += float(v_loss.detach().cpu())
            total_ent += float(ent.detach().cpu())
            total_kl += float(approx_kl.detach().cpu())
            total_clip += float(clip_frac.cpu())
            total_teacher += float(teacher_loss.detach().cpu())
            total_gn += float(grad_norm.cpu())
            updates += 1
    return {
        "pi_loss": total_pi / max(updates, 1),
        "v_loss": total_v / max(updates, 1),
        "teacher_loss": total_teacher / max(updates, 1),
        "entropy": total_ent / max(updates, 1),
        "kl": total_kl / max(updates, 1),
        "clip_frac": total_clip / max(updates, 1),
        "grad_norm": total_gn / max(updates, 1),
    }
