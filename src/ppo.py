"""Small PPO implementation for the teacher policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim: int, hidden: list[int], out_dim: int, activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for dim in hidden:
        layers.append(nn.Linear(last, dim))
        layers.append(activation())
        last = dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        pi_hidden: list[int],
        vf_hidden: list[int],
        activation: type[nn.Module],
        init_std: float,
    ):
        super().__init__()
        self.actor = mlp(obs_dim, pi_hidden, act_dim, activation)
        self.critic = mlp(obs_dim, vf_hidden, 1, activation)
        self.log_std = nn.Parameter(torch.full((act_dim,), torch.log(torch.tensor(float(init_std)))))

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.dist(obs)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.dist(obs)
        logp = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return logp, entropy, value

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    returns: torch.Tensor
    adv: torch.Tensor
    values: torch.Tensor
    teacher_actions: torch.Tensor | None = None


class Rollout:
    def __init__(self, steps: int, envs: int, obs_dim: int, act_dim: int, device: torch.device):
        self.steps = int(steps)
        self.envs = int(envs)
        self.obs = torch.zeros((steps, envs, obs_dim), device=device)
        self.actions = torch.zeros((steps, envs, act_dim), device=device)
        self.logp = torch.zeros((steps, envs), device=device)
        self.rewards = torch.zeros((steps, envs), device=device)
        self.dones = torch.zeros((steps, envs), device=device)
        self.values = torch.zeros((steps, envs), device=device)

    def add(
        self,
        t: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        logp: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        self.obs[t].copy_(obs.detach())
        self.actions[t].copy_(action.detach())
        self.logp[t].copy_(logp.detach())
        self.rewards[t].copy_(reward.detach())
        self.dones[t].copy_(done.float())
        self.values[t].copy_(value.detach())

    def make_batch(self, last_value: torch.Tensor, gamma: float, lam: float, norm_adv: bool) -> Batch:
        adv = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.envs, device=self.rewards.device)
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_value = last_value
                next_not_done = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_not_done - self.values[t]
            last_gae = delta + gamma * lam * next_not_done * last_gae
            adv[t] = last_gae
        returns = adv + self.values

        obs = self.obs.reshape(-1, self.obs.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        logp = self.logp.reshape(-1)
        values = self.values.reshape(-1)
        adv = adv.reshape(-1)
        returns = returns.reshape(-1)
        if norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return Batch(obs=obs, actions=actions, logp=logp, returns=returns, adv=adv, values=values)


def ppo_update(
    model: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: Batch,
    batch_size: int,
    epochs: int,
    clip: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    clip_vf: float | None,
    teacher_coef: float = 0.0,
) -> dict[str, float]:
    n = batch.obs.shape[0]
    total_pi = 0.0
    total_v = 0.0
    total_ent = 0.0
    total_kl = 0.0
    total_teacher = 0.0
    updates = 0
    for _ in range(int(epochs)):
        ids = torch.randperm(n, device=batch.obs.device)
        for start in range(0, n, int(batch_size)):
            idx = ids[start : start + int(batch_size)]
            logp, entropy, value = model.evaluate(batch.obs[idx], batch.actions[idx])
            ratio = torch.exp(logp - batch.logp[idx])
            pi_loss = torch.max(
                -batch.adv[idx] * ratio,
                -batch.adv[idx] * torch.clamp(ratio, 1.0 - clip, 1.0 + clip),
            ).mean()

            if clip_vf is None:
                v_loss = 0.5 * (value - batch.returns[idx]).pow(2).mean()
            else:
                v_clip = batch.values[idx] + torch.clamp(value - batch.values[idx], -clip_vf, clip_vf)
                v_loss = 0.5 * torch.max(
                    (value - batch.returns[idx]).pow(2),
                    (v_clip - batch.returns[idx]).pow(2),
                ).mean()

            ent = entropy.mean()
            teacher_loss = torch.zeros((), device=batch.obs.device)
            if teacher_coef > 0.0 and batch.teacher_actions is not None:
                mean = model.predict(batch.obs[idx])
                teacher_loss = (mean - batch.teacher_actions[idx]).pow(2).mean()

            loss = pi_loss + vf_coef * v_loss - ent_coef * ent + teacher_coef * teacher_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            with torch.no_grad():
                approx_kl = (batch.logp[idx] - logp).mean()
            total_pi += float(pi_loss.detach().cpu())
            total_v += float(v_loss.detach().cpu())
            total_ent += float(ent.detach().cpu())
            total_kl += float(approx_kl.detach().cpu())
            total_teacher += float(teacher_loss.detach().cpu())
            updates += 1

    return {
        "pi_loss": total_pi / max(updates, 1),
        "v_loss": total_v / max(updates, 1),
        "entropy": total_ent / max(updates, 1),
        "kl": total_kl / max(updates, 1),
        "teacher_loss": total_teacher / max(updates, 1),
    }
