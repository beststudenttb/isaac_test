"""PPO pieces for SPR visual state training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.cv_extractor.spr_state import SPRStateNet
from src.ppo import mlp


class SPRActorCritic(nn.Module):
    """Actor-critic using image encoder as state.

    Actor uses detached z. Critic uses live z, so value loss updates encoder.
    """

    def __init__(
        self,
        spr: SPRStateNet,
        goal_dim: int,
        act_dim: int,
        pi_hidden: list[int],
        vf_hidden: list[int],
        activation: type[nn.Module],
        init_std: float,
    ):
        super().__init__()
        self.spr = spr
        self.goal_dim = int(goal_dim)
        self.act_dim = int(act_dim)
        obs_dim = int(spr.z_dim) + self.goal_dim
        self.actor = mlp(obs_dim, pi_hidden, self.act_dim, activation)
        self.critic = mlp(obs_dim, vf_hidden, 1, activation)
        self.q_head = mlp(obs_dim + self.act_dim, vf_hidden, 1, activation)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), torch.log(torch.tensor(float(init_std)))))

    def encode(self, image: torch.Tensor, grad: bool = True) -> torch.Tensor:
        if grad:
            return self.spr.encode(image).float()
        with torch.no_grad():
            return self.spr.encode(image).float()

    def actor_obs(self, z: torch.Tensor, goal: torch.Tensor, z_grad_scale: float = 0.0) -> torch.Tensor:
        scale = float(z_grad_scale)
        if scale == 0.0:
            actor_z = z.detach()
        else:
            actor_z = z.detach() + scale * (z - z.detach())
        return torch.cat((actor_z, goal.to(device=z.device, dtype=z.dtype)), dim=1)

    def critic_obs(self, z: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat((z, goal.to(device=z.device, dtype=z.dtype)), dim=1)

    def q_value(self, z: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = self.critic_obs(z, goal)
        action = action.to(device=z.device, dtype=z.dtype)
        return self.q_head(torch.cat((obs, action), dim=1)).squeeze(-1)

    def dist_from_z(self, z: torch.Tensor, goal: torch.Tensor, z_grad_scale: float = 0.0) -> Normal:
        mean = self.actor(self.actor_obs(z, goal, z_grad_scale=z_grad_scale))
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, image: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(image)
        dist = self.dist_from_z(z, goal)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        value = self.critic(self.critic_obs(z, goal)).squeeze(-1)
        return action, logp, value

    def evaluate(
        self,
        image: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        encoder_grad: bool = True,
        actor_z_grad_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(image, grad=encoder_grad)
        dist = self.dist_from_z(z, goal, z_grad_scale=actor_z_grad_scale)
        logp = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(self.critic_obs(z, goal)).squeeze(-1)
        return logp, entropy, value, z

    def value(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = self.encode(image)
        return self.critic(self.critic_obs(z, goal)).squeeze(-1)

    def predict(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = self.encode(image)
        return self.actor(self.actor_obs(z, goal))

    def spr_loss(self, z: torch.Tensor, next_image: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_z = self.spr.encode_target(next_image)
            target = self.spr.spr_target(next_z)
        pred_z = self.spr.predict_next(z, action)
        pred = self.spr.spr_pred(pred_z)
        return F.mse_loss(F.normalize(pred, dim=1), F.normalize(target, dim=1))

    def spr_loss_k(self, z: torch.Tensor, target_image: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_z = self.spr.encode_target(target_image)
            target = self.spr.spr_target(target_z)
        pred_z = z
        for i in range(actions.shape[1]):
            pred_z = self.spr.predict_next(pred_z, actions[:, i])
        pred = self.spr.spr_pred(pred_z)
        return F.mse_loss(F.normalize(pred, dim=1), F.normalize(target, dim=1))


@dataclass
class SPRBatch:
    images: torch.Tensor
    next_images: torch.Tensor
    spr_target_images: torch.Tensor
    spr_actions: torch.Tensor
    spr_mask: torch.Tensor
    goals: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    returns: torch.Tensor
    adv: torch.Tensor
    values: torch.Tensor
    teacher_actions: torch.Tensor | None = None


class SPRRollout:
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
        self.next_images = torch.empty_like(self.images)
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

    def add(
        self,
        t: int,
        image: torch.Tensor,
        next_image: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        logp: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.images[t].copy_(self.image_to_uint8(image))
        self.next_images[t].copy_(self.image_to_uint8(next_image))
        self.goals[t].copy_(goal.detach())
        self.actions[t].copy_(action.detach())
        self.logp[t].copy_(logp.detach())
        self.rewards[t].copy_(reward.detach())
        self.dones[t].copy_(done.float())
        self.values[t].copy_(value.detach())

    def make_batch(self, last_value: torch.Tensor, gamma: float, lam: float, norm_adv: bool, spr_k: int = 1) -> SPRBatch:
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

        spr_k = int(spr_k)
        spr_target_images = torch.empty_like(self.images)
        spr_actions = torch.zeros((self.steps, self.envs, spr_k, self.actions.shape[-1]), device=self.actions.device)
        spr_mask = torch.zeros((self.steps, self.envs), dtype=torch.bool, device=self.actions.device)
        for t in range(self.steps):
            end = t + spr_k - 1
            if end >= self.steps:
                continue
            spr_target_images[t].copy_(self.next_images[end])
            spr_actions[t].copy_(self.actions[t : end + 1].permute(1, 0, 2))
            if spr_k == 1:
                spr_mask[t].fill_(True)
            else:
                spr_mask[t].copy_(self.dones[t:end].sum(dim=0) == 0)

        images = self.images.reshape(-1, *self.images.shape[2:])
        next_images = self.next_images.reshape(-1, *self.next_images.shape[2:])
        spr_target_images = spr_target_images.reshape(-1, *spr_target_images.shape[2:])
        spr_actions = spr_actions.reshape(-1, spr_k, spr_actions.shape[-1])
        spr_mask = spr_mask.reshape(-1)
        goals = self.goals.reshape(-1, self.goals.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        logp = self.logp.reshape(-1)
        values = self.values.reshape(-1)
        adv = adv.reshape(-1)
        returns = returns.reshape(-1)
        if norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return SPRBatch(
            images=images,
            next_images=next_images,
            spr_target_images=spr_target_images,
            spr_actions=spr_actions,
            spr_mask=spr_mask,
            goals=goals,
            actions=actions,
            logp=logp,
            returns=returns,
            adv=adv,
            values=values,
        )


def encode_chunks(fn, images: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    outs = []
    for start in range(0, images.shape[0], int(chunk)):
        outs.append(fn(images[start : start + int(chunk)]))
    return torch.cat(outs, dim=0)


def spr_diagnostics(model: SPRActorCritic, batch: SPRBatch, sample: int = 256, chunk: int = 32) -> dict[str, float]:
    valid = torch.nonzero(batch.spr_mask, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return {"eff_rank": 0.0, "z_std": 0.0, "pred_cos": 0.0, "identity_gap": 0.0, "target_sim": 0.0, "q_corr": 0.0}
    idx = valid[: int(sample)]
    idx_cpu = idx.cpu()
    with torch.no_grad():
        z = encode_chunks(lambda im: model.encode(im, grad=False), batch.images[idx_cpu], chunk)
        zc = z - z.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(zc)
        p = s / s.sum().clamp_min(1e-8)
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()).cpu())
        z_std = float(z.std(dim=0).mean().cpu())
        target_z = encode_chunks(model.spr.encode_target, batch.spr_target_images[idx_cpu], chunk)
        target = F.normalize(model.spr.spr_target(target_z), dim=1)
        pred_z = z
        acts = batch.spr_actions[idx]
        for i in range(acts.shape[1]):
            pred_z = model.spr.predict_next(pred_z, acts[:, i])
        pred = F.normalize(model.spr.spr_pred(pred_z), dim=1)
        ident = F.normalize(model.spr.spr_pred(z), dim=1)
        cur_target_z = encode_chunks(model.spr.encode_target, batch.images[idx_cpu], chunk)
        cur_target = F.normalize(model.spr.spr_target(cur_target_z), dim=1)
        pred_cos = float((pred * target).sum(dim=1).mean().cpu())
        ident_cos = float((ident * target).sum(dim=1).mean().cpu())
        target_sim = float((cur_target * target).sum(dim=1).mean().cpu())
        q_pred = model.q_value(z, batch.goals[idx], batch.actions[idx])
        rank_q = q_pred.argsort().argsort().float()
        rank_r = batch.returns[idx].argsort().argsort().float()
        corr = torch.corrcoef(torch.stack((rank_q, rank_r)))[0, 1]
        q_corr = float(torch.nan_to_num(corr).cpu())
    return {
        "eff_rank": eff_rank,
        "z_std": z_std,
        "pred_cos": pred_cos,
        "identity_gap": pred_cos - ident_cos,
        "target_sim": target_sim,
        "q_corr": q_corr,
    }


def spr_only_update(
    model: SPRActorCritic,
    opt: torch.optim.Optimizer,
    batch: SPRBatch,
    batch_size: int,
    epochs: int,
    spr_coef: float,
    max_grad_norm: float,
    q_coef: float = 0.0,
    return_scale: float = 1.0,
) -> dict[str, float]:
    n = batch.actions.shape[0]
    total_spr = 0.0
    total_q = 0.0
    total_gn = 0.0
    updates = 0
    device = batch.actions.device
    for _ in range(int(epochs)):
        ids = torch.randperm(n, device=device)
        for start in range(0, n, int(batch_size)):
            idx = ids[start : start + int(batch_size)]
            idx_cpu = idx.cpu()
            valid = batch.spr_mask[idx]
            if not torch.any(valid) and q_coef <= 0.0:
                continue
            z = model.encode(batch.images[idx_cpu], grad=True)
            spr_loss = torch.zeros((), device=device)
            if torch.any(valid):
                spr_loss = model.spr_loss_k(z[valid], batch.spr_target_images[idx_cpu][valid.cpu()], batch.spr_actions[idx][valid])
            q_loss = torch.zeros((), device=device)
            if q_coef > 0.0:
                q_pred = model.q_value(z, batch.goals[idx], batch.actions[idx])
                q_loss = 0.5 * (q_pred - batch.returns[idx] / return_scale).pow(2).mean()
            opt.zero_grad()
            (spr_coef * spr_loss + q_coef * q_loss).backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            model.spr.update_target()
            total_spr += float(spr_loss.detach().cpu())
            total_q += float(q_loss.detach().cpu())
            total_gn += float(grad_norm.cpu())
            updates += 1
    with torch.no_grad():
        raw_std = torch.exp(model.log_std.detach())
    return {
        "pi_loss": 0.0,
        "v_loss": 0.0,
        "spr_loss": total_spr / max(updates, 1),
        "q_loss": total_q / max(updates, 1),
        "entropy": 0.0,
        "kl": 0.0,
        "clip_frac": 0.0,
        "grad_norm": total_gn / max(updates, 1),
        "teacher_loss": 0.0,
        "raw_std_mean": float(raw_std.mean().cpu()),
        "raw_std_min": float(raw_std.min().cpu()),
        "raw_std_max": float(raw_std.max().cpu()),
    }


def spr_ppo_update(
    model: SPRActorCritic,
    opt: torch.optim.Optimizer,
    batch: SPRBatch,
    batch_size: int,
    epochs: int,
    clip: float,
    vf_coef: float,
    ent_coef: float,
    spr_coef: float,
    max_grad_norm: float,
    clip_vf: float | None,
    teacher_coef: float = 0.0,
    teacher_loss_type: str = "mse",
    update_spr: bool = True,
    actor_encoder_coef: float = 0.0,
    log_std_min: float | None = None,
    log_std_max: float | None = None,
) -> dict[str, float]:
    n = batch.actions.shape[0]
    total_pi = 0.0
    total_v = 0.0
    total_spr = 0.0
    total_ent = 0.0
    total_kl = 0.0
    total_clip = 0.0
    total_gn = 0.0
    total_teacher = 0.0
    total_raw_std_mean = 0.0
    total_raw_std_min = 0.0
    total_raw_std_max = 0.0
    updates = 0
    device = batch.actions.device
    for _ in range(int(epochs)):
        ids = torch.randperm(n, device=device)
        for start in range(0, n, int(batch_size)):
            idx = ids[start : start + int(batch_size)]
            idx_cpu = idx.cpu()
            image = batch.images[idx_cpu]
            next_image = batch.next_images[idx_cpu]
            goal = batch.goals[idx]
            action = batch.actions[idx]

            actor_z_grad_scale = float(actor_encoder_coef) if update_spr else 0.0
            logp, entropy, value, z = model.evaluate(
                image,
                goal,
                action,
                encoder_grad=update_spr,
                actor_z_grad_scale=actor_z_grad_scale,
            )
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

            spr_loss = torch.zeros((), device=device)
            if update_spr:
                valid = batch.spr_mask[idx]
                if torch.any(valid):
                    spr_loss = model.spr_loss_k(z[valid], batch.spr_target_images[idx_cpu][valid.cpu()], batch.spr_actions[idx][valid])
            ent = entropy.mean()
            teacher_loss = torch.zeros((), device=device)
            if teacher_coef > 0.0 and batch.teacher_actions is not None:
                mean = model.actor(model.actor_obs(z, goal))
                if teacher_loss_type == "nll":
                    var = torch.exp(2.0 * model.log_std)
                    teacher_loss = (model.log_std + (batch.teacher_actions[idx] - mean).pow(2) / (2.0 * var)).mean()
                else:
                    teacher_loss = (mean - batch.teacher_actions[idx]).pow(2).mean()

            loss = pi_loss + vf_coef * v_loss + spr_coef * spr_loss - ent_coef * ent + teacher_coef * teacher_loss
            opt.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                raw_std = torch.exp(model.log_std.detach())
                total_raw_std_mean += float(raw_std.mean().cpu())
                total_raw_std_min += float(raw_std.min().cpu())
                total_raw_std_max += float(raw_std.max().cpu())
            if log_std_min is not None and log_std_max is not None:
                model.log_std.data.clamp_(float(log_std_min), float(log_std_max))
            if update_spr:
                model.spr.update_target()

            with torch.no_grad():
                approx_kl = (batch.logp[idx] - logp).mean()
                clip_frac = ((ratio - 1.0).abs() > clip).float().mean()
            total_pi += float(pi_loss.detach().cpu())
            total_v += float(v_loss.detach().cpu())
            total_spr += float(spr_loss.detach().cpu())
            total_ent += float(ent.detach().cpu())
            total_kl += float(approx_kl.detach().cpu())
            total_clip += float(clip_frac.cpu())
            total_gn += float(grad_norm.cpu())
            total_teacher += float(teacher_loss.detach().cpu())
            updates += 1

    return {
        "pi_loss": total_pi / max(updates, 1),
        "v_loss": total_v / max(updates, 1),
        "spr_loss": total_spr / max(updates, 1),
        "entropy": total_ent / max(updates, 1),
        "kl": total_kl / max(updates, 1),
        "clip_frac": total_clip / max(updates, 1),
        "grad_norm": total_gn / max(updates, 1),
        "teacher_loss": total_teacher / max(updates, 1),
        "raw_std_mean": total_raw_std_mean / max(updates, 1),
        "raw_std_min": total_raw_std_min / max(updates, 1),
        "raw_std_max": total_raw_std_max / max(updates, 1),
    }
