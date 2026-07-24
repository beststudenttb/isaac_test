"""Off-policy replay buffer with n-step sampling for the DrQ student line."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """列式 ring buffer,布局 (T, N, ...):同一 env 列内相邻行 = 时间相邻。

    所有 env 每个 tick 同步写一行(向量化采集),n-step 窗口沿列前进。
    窗口语义:
    - 窗口 r..r+n-1 只在同一 env 列内走,遇 done 截断;
    - terminated(success/fail)是吸收态,bootstrap 折扣为 0;
    - truncated(timeout)用该行写入的 terminal 帧 bootstrap(帧存小池子,值由调用方用当前 critic 现算);
    - 存活满 n 步用 r+n 行的 obs bootstrap,折扣 gamma^n。
    图像常驻 CPU 内存(容量按 224x224x3 uint8 估算,默认 8192x32 约 38GB,机器 125GB)。
    """

    def __init__(
        self,
        capacity_per_env: int,
        num_envs: int,
        image_shape: tuple[int, int, int],
        goal_dim: int,
        act_dim: int,
        z_dim: int = 0,
        n_step: int = 3,
        gamma: float = 0.99,
        min_episode_len: int = 300,
        store_images: bool = True,
    ):
        self.capacity = int(capacity_per_env)
        self.num_envs = int(num_envs)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.z_dim = int(z_dim)
        T, N = self.capacity, self.num_envs

        # spr_z 模式训练只用 z,图像纯占 RAM;store_images=False 时不分配/不存,128 env 才放得下。
        self.store_images = bool(store_images)
        self.images = torch.zeros((T, N, *image_shape), dtype=torch.uint8) if self.store_images else None
        self.goals = torch.zeros((T, N, goal_dim), dtype=torch.float32)
        self.actions = torch.zeros((T, N, act_dim), dtype=torch.float32)
        self.rewards = torch.zeros((T, N), dtype=torch.float32)
        self.terminated = torch.zeros((T, N), dtype=torch.bool)
        self.truncated = torch.zeros((T, N), dtype=torch.bool)
        self.term_slot = torch.full((T, N), -1, dtype=torch.long)
        self.z = torch.zeros((T, N, z_dim), dtype=torch.float32) if z_dim > 0 else None

        # truncation(timeout)至少间隔 min_episode_len 步(真实 env 为 375,默认 300 留裕量),
        # 池子按此上限分配;若配置错导致复用还被行引用的 slot,add() 会 raise 而不是静默损坏。
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
        z: torch.Tensor | None = None,
        term_images: torch.Tensor | None = None,
        term_z: torch.Tensor | None = None,
    ) -> None:
        """写入一个 tick。truncated 行必须带 terminal 帧(顺序与 truncated.nonzero() 一致)。

        terminated/truncated 互斥由调用方保证(timeout = truncated & ~terminated)。
        """
        p = self.ptr
        if self.images is not None:
            self.images[p].copy_(images.detach().to("cpu"))
        self.goals[p].copy_(goals.detach().to("cpu", torch.float32))
        self.actions[p].copy_(actions.detach().to("cpu", torch.float32))
        self.rewards[p].copy_(rewards.detach().to("cpu", torch.float32))
        self.terminated[p].copy_(terminated.detach().to("cpu"))
        self.truncated[p].copy_(truncated.detach().to("cpu"))
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

    def sample(self, batch_size: int, device: torch.device, with_images: bool = True, spr_k: int = 0) -> dict[str, torch.Tensor]:
        """采 n-step batch。返回:

        obs/next_obs(uint8 图,with_images=False 时省)、z/next_z(z_dim>0 时)、
        goal/next_goal、action、reward(n-step 折扣和)、discount(bootstrap 折扣,
        terminated 结尾为 0)。next_* 在 truncated 结尾处已替换为 terminal 帧。

        spr_k>0(co-adapt):额外返回 spr_target(起点后第 k 帧图,= K 步转移预测的 target)、
        spr_actions[B,K,act](起点起 K 步动作,喂 transition 迭代)、spr_mask[B](整段 [r0,r0+K)
        无 terminated/truncated 才为 1)。起点图直接复用 obs(with_images 已给),不额外存字段。
        """
        n = self.n_step
        horizon = max(n, spr_k)
        if self.size <= horizon:
            raise ValueError(f"buffer too small to sample: size={self.size} horizon={horizon}")
        T = self.capacity
        max_a = self.size - 1 - horizon
        base = self.ptr if self.size == T else 0
        a = torch.randint(0, max_a + 1, (batch_size,))
        e = torch.randint(0, self.num_envs, (batch_size,))
        r0 = (base + a) % T

        reward = torch.zeros(batch_size)
        discount = torch.ones(batch_size)
        alive = torch.ones(batch_size, dtype=torch.bool)
        boot_disc = torch.zeros(batch_size)
        slot_sel = torch.full((batch_size,), -1, dtype=torch.long)
        trunc_mask = torch.zeros(batch_size, dtype=torch.bool)
        trunc_row = torch.zeros(batch_size, dtype=torch.long)
        for k in range(n):
            row = (r0 + k) % T
            reward = reward + discount * self.rewards[row, e] * alive.float()
            trunc_k = self.truncated[row, e] & alive
            if torch.any(trunc_k):
                slot_sel[trunc_k] = self.term_slot[row[trunc_k], e[trunc_k]]
                boot_disc[trunc_k] = discount[trunc_k] * self.gamma
                trunc_row[trunc_k] = row[trunc_k]
                trunc_mask = trunc_mask | trunc_k
            alive = alive & ~(self.terminated[row, e] | self.truncated[row, e])
            discount = discount * self.gamma
        row_n = (r0 + n) % T
        boot_disc[alive] = discount[alive]

        if torch.any(trunc_mask) and int(slot_sel[trunc_mask].min()) < 0:
            raise RuntimeError("truncated row without terminal slot")

        goal = self.goals[r0, e]
        next_goal = self.goals[row_n, e].clone()
        if torch.any(trunc_mask):
            next_goal[trunc_mask] = self.goals[trunc_row[trunc_mask], e[trunc_mask]]

        batch = {
            "goal": goal.to(device),
            "next_goal": next_goal.to(device),
            "action": self.actions[r0, e].to(device),
            "reward": reward.to(device),
            "discount": boot_disc.to(device),
        }
        if with_images:
            obs = self.images[r0, e]
            next_obs = self.images[row_n, e].clone()
            if torch.any(trunc_mask):
                next_obs[trunc_mask] = self.term_images[slot_sel[trunc_mask]]
            batch["obs"] = obs.to(device)
            batch["next_obs"] = next_obs.to(device)
        if self.z is not None:
            z = self.z[r0, e]
            next_z = self.z[row_n, e].clone()
            if torch.any(trunc_mask):
                next_z[trunc_mask] = self.term_z[slot_sel[trunc_mask]]
            batch["z"] = z.to(device)
            batch["next_z"] = next_z.to(device)
        if spr_k > 0:
            if self.images is None:
                raise RuntimeError("spr_k>0 需要 store_images=True")
            spr_actions = torch.stack([self.actions[(r0 + k) % T, e] for k in range(spr_k)], dim=1)
            spr_target = self.images[(r0 + spr_k) % T, e]
            spr_valid = torch.ones(batch_size, dtype=torch.bool)
            for k in range(spr_k):
                row = (r0 + k) % T
                spr_valid = spr_valid & ~(self.terminated[row, e] | self.truncated[row, e])
            batch["spr_actions"] = spr_actions.to(device)
            batch["spr_target"] = spr_target.to(device)
            batch["spr_mask"] = spr_valid.float().to(device)
        return batch
