"""HER replay buffer(single-step,列式 (T,N) ring,与 src/rl/buffer.py 布局一致)。

与普通 ReplayBuffer 的差别:
- n_step 固定为 1(SAC 标配),bootstrap 简单:terminated→disc 0,否则 gamma,truncated 用 terminal 帧;
- 每行额外存 cur_px/cur_dist(该步 reward 计算用的球投影 = "achieved goal")与 fail 标志,
  供 HER relabel 时用 src.rl.task_reward.recompute_reward 重算 reward/success/terminated;
- sample() 里对 her_ratio 比例的样本做 future-goal relabel:goal 换成同 episode 未来某步的
  achieved(end_x=px-cx, end_d=dist),relabel 后 obs/z 不变(观测 goal 无关),只改
  goal/next_goal/reward/discount。这样把 SAC 采到的低动作样本组合成 in_zone&a<eps=success 样本。

图像常驻 CPU,pixels 模式按 224x224x3 uint8 估算。spr_z 模式不存图、只存 z。
"""

from __future__ import annotations

import torch

from src.rl.task_reward import achieved_to_end, goal_norm, recompute_reward


class HERReplayBuffer:
    def __init__(
        self,
        capacity_per_env: int,
        num_envs: int,
        image_shape: tuple[int, int, int],
        goal_dim: int,
        act_dim: int,
        z_dim: int = 0,
        gamma: float = 0.99,
        min_episode_len: int = 300,
        store_images: bool = True,
        her_ratio: float = 0.5,
        future_h: int = 20,
    ):
        self.capacity = int(capacity_per_env)
        self.num_envs = int(num_envs)
        self.gamma = float(gamma)
        self.z_dim = int(z_dim)
        self.her_ratio = float(her_ratio)
        self.future_h = int(future_h)
        T, N = self.capacity, self.num_envs

        self.store_images = bool(store_images)
        self.images = torch.zeros((T, N, *image_shape), dtype=torch.uint8) if self.store_images else None
        self.goals = torch.zeros((T, N, goal_dim), dtype=torch.float32)
        self.actions = torch.zeros((T, N, act_dim), dtype=torch.float32)
        self.rewards = torch.zeros((T, N), dtype=torch.float32)
        self.terminated = torch.zeros((T, N), dtype=torch.bool)
        self.truncated = torch.zeros((T, N), dtype=torch.bool)
        self.fail = torch.zeros((T, N), dtype=torch.bool)
        self.cur_px = torch.zeros((T, N), dtype=torch.float32)
        self.cur_dist = torch.zeros((T, N), dtype=torch.float32)
        self.term_slot = torch.full((T, N), -1, dtype=torch.long)
        self.z = torch.zeros((T, N, z_dim), dtype=torch.float32) if z_dim > 0 else None

        pool = N * (T // max(int(min_episode_len), 1) + 2)
        self.term_images = torch.zeros((pool, *image_shape), dtype=torch.uint8) if self.store_images else None
        self.term_z = torch.zeros((pool, z_dim), dtype=torch.float32) if z_dim > 0 else None
        self.slot_owner = torch.full((pool, 2), -1, dtype=torch.long)
        self.pool_size = pool
        self.pool_ptr = 0

        self.ptr = 0
        self.size = 0

    @property
    def frames(self) -> int:
        return self.size * self.num_envs

    def add(
        self,
        images: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        fail: torch.Tensor,
        cur_px: torch.Tensor,
        cur_dist: torch.Tensor,
        z: torch.Tensor | None = None,
        term_images: torch.Tensor | None = None,
        term_z: torch.Tensor | None = None,
    ) -> None:
        p = self.ptr
        if self.images is not None:
            self.images[p].copy_(images.detach().to("cpu"))
        self.goals[p].copy_(goals.detach().to("cpu", torch.float32))
        self.actions[p].copy_(actions.detach().to("cpu", torch.float32))
        self.rewards[p].copy_(rewards.detach().to("cpu", torch.float32))
        self.terminated[p].copy_(terminated.detach().to("cpu"))
        self.truncated[p].copy_(truncated.detach().to("cpu"))
        self.fail[p].copy_(fail.detach().to("cpu"))
        self.cur_px[p].copy_(cur_px.detach().to("cpu", torch.float32))
        self.cur_dist[p].copy_(cur_dist.detach().to("cpu", torch.float32))
        self.term_slot[p].fill_(-1)
        if self.z is not None:
            self.z[p].copy_(z.detach().to("cpu", torch.float32))

        trunc_ids = torch.nonzero(self.truncated[p], as_tuple=False).flatten()
        if trunc_ids.numel() > 0:
            slots = (self.pool_ptr + torch.arange(trunc_ids.numel())) % self.pool_size
            for slot in slots.tolist():
                owner_row, owner_env = self.slot_owner[slot].tolist()
                if owner_row >= 0 and int(self.term_slot[owner_row, owner_env]) == slot:
                    raise RuntimeError(
                        f"terminal pool overflow: slot {slot} still referenced by row ({owner_row},{owner_env}); "
                        "min_episode_len is set too high for the actual episode lengths"
                    )
            if self.term_images is not None:
                if term_images is None or term_images.shape[0] != trunc_ids.numel():
                    raise ValueError("truncated rows require matching terminal images")
                self.term_images[slots] = term_images.detach().to("cpu")
            if self.term_z is not None:
                if term_z is None or term_z.shape[0] != trunc_ids.numel():
                    raise ValueError("truncated rows require matching terminal z")
                self.term_z[slots] = term_z.detach().to("cpu", torch.float32)
            self.term_slot[p, trunc_ids] = slots
            self.slot_owner[slots, 0] = p
            self.slot_owner[slots, 1] = trunc_ids
            self.pool_ptr = int((self.pool_ptr + trunc_ids.numel()) % self.pool_size)

        self.ptr = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device, with_images: bool = True) -> dict[str, torch.Tensor]:
        if self.size <= 2:
            raise ValueError(f"buffer too small to sample: size={self.size}")
        T = self.capacity
        H = self.future_h
        # a 需要 r0-1(prev)与 r0+1(next)都在有效区内。
        max_a = self.size - 2
        base = self.ptr if self.size == T else 0
        a = torch.randint(1, max_a + 1, (batch_size,))
        e = torch.randint(0, self.num_envs, (batch_size,))
        r0 = (base + a) % T
        r1 = (r0 + 1) % T
        rprev = (r0 - 1) % T

        term0 = self.terminated[r0, e]
        trunc0 = self.truncated[r0, e]
        reward = self.rewards[r0, e].clone()
        goal = self.goals[r0, e].clone()
        next_goal = self.goals[r1, e].clone()
        next_goal[trunc0] = self.goals[r0, e][trunc0]  # truncated 用同 episode(r0)的 goal
        discount = torch.where(term0, torch.zeros(batch_size), torch.full((batch_size,), self.gamma))

        slot_sel = self.term_slot[r0, e]  # truncated 行的 terminal 帧槽位

        # ---- HER future relabel ----
        her = (torch.rand(batch_size) < self.her_ratio) & (~term0) & (~trunc0)
        if torch.any(her):
            offs = torch.arange(1, H + 1)
            win_rows = (r0.unsqueeze(1) + offs.unsqueeze(0)) % T           # [B,H]
            win_done = (self.terminated[win_rows, e.unsqueeze(1)]
                        | self.truncated[win_rows, e.unsqueeze(1)])         # [B,H]
            big = H + 1
            done_off = torch.where(win_done, offs.unsqueeze(0).expand_as(win_done),
                                   torch.full_like(win_done, big, dtype=torch.long))
            first_done = done_off.min(dim=1).values                        # H+1 表示窗口内无 done
            max_off = torch.clamp(first_done, max=H).clamp(min=1)          # 未来目标可取到 done 步(含)
            u = 1 + (torch.rand(batch_size) * max_off.float()).floor().long()
            u = torch.clamp(u, min=torch.ones_like(max_off), max=max_off)
            rf = (r0 + u) % T                                              # future achieved 所在行

            end_x, end_d = achieved_to_end(self.cur_px[rf, e], self.cur_dist[rf, e])
            g_new = goal_norm(end_x, end_d)
            new_reward, _success, new_term = recompute_reward(
                self.cur_px[r0, e], self.cur_dist[r0, e], self.actions[r0, e],
                self.cur_px[rprev, e], self.cur_dist[rprev, e], self.actions[rprev, e],
                end_x, end_d, self.fail[r0, e],
            )
            new_disc = torch.where(new_term, torch.zeros(batch_size), torch.full((batch_size,), self.gamma))
            goal[her] = g_new[her]
            next_goal[her] = g_new[her]
            reward[her] = new_reward[her]
            discount[her] = new_disc[her]

        batch = {
            "goal": goal.to(device),
            "next_goal": next_goal.to(device),
            "action": self.actions[r0, e].to(device),
            "reward": reward.to(device),
            "discount": discount.to(device),
        }
        if with_images:
            obs = self.images[r0, e]
            next_obs = self.images[r1, e].clone()
            if torch.any(trunc0):
                next_obs[trunc0] = self.term_images[slot_sel[trunc0]]
            batch["obs"] = obs.to(device)
            batch["next_obs"] = next_obs.to(device)
        if self.z is not None:
            z = self.z[r0, e]
            next_z = self.z[r1, e].clone()
            if torch.any(trunc0):
                next_z[trunc0] = self.term_z[slot_sel[trunc0]]
            batch["z"] = z.to(device)
            batch["next_z"] = next_z.to(device)
        return batch
