"""Frame stacking and vectorized n-step replay for the standalone DrQ-v2 line."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_rgb(images: torch.Tensor, size: int) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"expected NHWC RGB images, got {tuple(images.shape)}")
    rgb = images[..., :3].permute(0, 3, 1, 2).float()
    if rgb.shape[-2:] != (int(size), int(size)):
        rgb = F.interpolate(
            rgb,
            size=(int(size), int(size)),
            mode="bilinear",
            align_corners=False,
        )
    return rgb.round().clamp_(0, 255).to(torch.uint8)


class FrameStack:
    def __init__(self, num_envs: int, frame_stack: int, frame_shape: tuple[int, int, int]):
        channels, height, width = frame_shape
        self.num_envs = int(num_envs)
        self.frame_stack = int(frame_stack)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.obs: torch.Tensor | None = None

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (
            self.channels * self.frame_stack,
            self.height,
            self.width,
        )

    def reset(self, frame: torch.Tensor) -> torch.Tensor:
        self._check_frame(frame, self.num_envs)
        self.obs = frame.repeat(1, self.frame_stack, 1, 1)
        return self.obs

    def terminal_obs(
        self,
        env_ids: torch.Tensor,
        terminal_frame: torch.Tensor,
    ) -> torch.Tensor:
        if self.obs is None:
            raise RuntimeError("frame stack is not initialized")
        self._check_frame(terminal_frame, int(env_ids.numel()))
        return torch.cat(
            (self.obs[env_ids, self.channels :], terminal_frame),
            dim=1,
        )

    def advance(
        self,
        next_frame: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        if self.obs is None:
            raise RuntimeError("frame stack is not initialized")
        self._check_frame(next_frame, self.num_envs)
        shifted = torch.cat((self.obs[:, self.channels :], next_frame), dim=1)
        if torch.any(done):
            shifted[done] = next_frame[done].repeat(1, self.frame_stack, 1, 1)
        self.obs = shifted
        return self.obs

    def _check_frame(self, frame: torch.Tensor, batch: int) -> None:
        expected = (batch, self.channels, self.height, self.width)
        if tuple(frame.shape) != expected:
            raise ValueError(f"expected frame shape {expected}, got {tuple(frame.shape)}")


class VectorNStepReplay:
    def __init__(
        self,
        capacity_per_env: int,
        num_envs: int,
        obs_shape: tuple[int, int, int],
        goal_dim: int,
        action_dim: int,
        nstep: int,
        discount: float,
        min_timeout_interval: int,
    ):
        self.capacity = int(capacity_per_env)
        self.num_envs = int(num_envs)
        self.nstep = int(nstep)
        self.gamma = float(discount)
        if self.capacity <= self.nstep:
            raise ValueError("capacity_per_env must be larger than nstep")

        time_dim = self.capacity
        env_dim = self.num_envs
        self.observations = torch.empty(
            (time_dim, env_dim, *obs_shape),
            dtype=torch.uint8,
        )
        self.goals = torch.empty((time_dim, env_dim, goal_dim), dtype=torch.float32)
        self.actions = torch.empty((time_dim, env_dim, action_dim), dtype=torch.float32)
        self.rewards = torch.empty((time_dim, env_dim), dtype=torch.float32)
        self.terminated = torch.empty((time_dim, env_dim), dtype=torch.bool)
        self.truncated = torch.empty((time_dim, env_dim), dtype=torch.bool)
        self.terminal_slot = torch.full((time_dim, env_dim), -1, dtype=torch.long)

        pool_size = env_dim * (
            time_dim // max(int(min_timeout_interval), 1) + 2
        )
        self.terminal_observations = torch.empty(
            (pool_size, *obs_shape),
            dtype=torch.uint8,
        )
        self.terminal_goals = torch.empty(
            (pool_size, goal_dim),
            dtype=torch.float32,
        )
        self.slot_owner = torch.full((pool_size, 2), -1, dtype=torch.long)
        self.pool_size = int(pool_size)
        self.pool_ptr = 0
        self.ptr = 0
        self.size = 0

    @property
    def num_transitions(self) -> int:
        return self.size * self.num_envs

    def add(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        terminal_obs: torch.Tensor | None = None,
        terminal_goal: torch.Tensor | None = None,
    ) -> None:
        row = self.ptr
        self.observations[row].copy_(obs.detach().to("cpu", non_blocking=False))
        self.goals[row].copy_(goal.detach().to("cpu", torch.float32))
        self.actions[row].copy_(action.detach().to("cpu", torch.float32))
        self.rewards[row].copy_(reward.detach().to("cpu", torch.float32))
        self.terminated[row].copy_(terminated.detach().to("cpu", torch.bool))
        self.truncated[row].copy_(truncated.detach().to("cpu", torch.bool))
        self.terminal_slot[row].fill_(-1)

        timeout_ids = torch.nonzero(self.truncated[row], as_tuple=False).flatten()
        if timeout_ids.numel() > 0:
            count = int(timeout_ids.numel())
            if terminal_obs is None or terminal_obs.shape[0] != count:
                raise ValueError("truncated transitions require matching terminal_obs")
            if terminal_goal is None or terminal_goal.shape[0] != count:
                raise ValueError("truncated transitions require matching terminal_goal")

            slots = (
                self.pool_ptr + torch.arange(count, dtype=torch.long)
            ) % self.pool_size
            for slot in slots.tolist():
                owner_row, owner_env = self.slot_owner[slot].tolist()
                if (
                    owner_row >= 0
                    and int(self.terminal_slot[owner_row, owner_env]) == slot
                ):
                    raise RuntimeError(
                        "terminal pool overflow; min_timeout_interval is too large"
                    )

            self.terminal_observations[slots] = terminal_obs.detach().to("cpu")
            self.terminal_goals[slots] = terminal_goal.detach().to(
                "cpu",
                torch.float32,
            )
            self.terminal_slot[row, timeout_ids] = slots
            self.slot_owner[slots, 0] = row
            self.slot_owner[slots, 1] = timeout_ids
            self.pool_ptr = int((self.pool_ptr + count) % self.pool_size)

        self.ptr = (row + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        if self.size <= self.nstep:
            raise ValueError(
                f"replay has {self.size} rows, needs more than {self.nstep}"
            )

        max_offset = self.size - self.nstep - 1
        base = self.ptr if self.size == self.capacity else 0
        offsets = torch.randint(0, max_offset + 1, (int(batch_size),))
        env_ids = torch.randint(0, self.num_envs, (int(batch_size),))
        start_rows = (base + offsets) % self.capacity

        rewards = torch.zeros(int(batch_size), dtype=torch.float32)
        cumulative_discount = torch.ones(int(batch_size), dtype=torch.float32)
        bootstrap_discount = torch.zeros(int(batch_size), dtype=torch.float32)
        alive = torch.ones(int(batch_size), dtype=torch.bool)
        timeout_mask = torch.zeros(int(batch_size), dtype=torch.bool)
        timeout_rows = torch.zeros(int(batch_size), dtype=torch.long)
        timeout_slots = torch.full((int(batch_size),), -1, dtype=torch.long)

        for offset in range(self.nstep):
            rows = (start_rows + offset) % self.capacity
            rewards += (
                cumulative_discount
                * self.rewards[rows, env_ids]
                * alive.to(torch.float32)
            )
            timed_out = self.truncated[rows, env_ids] & alive
            if torch.any(timed_out):
                timeout_mask |= timed_out
                timeout_rows[timed_out] = rows[timed_out]
                timeout_slots[timed_out] = self.terminal_slot[
                    rows[timed_out],
                    env_ids[timed_out],
                ]
                bootstrap_discount[timed_out] = (
                    cumulative_discount[timed_out] * self.gamma
                )

            alive &= ~(
                self.terminated[rows, env_ids]
                | self.truncated[rows, env_ids]
            )
            cumulative_discount *= self.gamma

        next_rows = (start_rows + self.nstep) % self.capacity
        bootstrap_discount[alive] = cumulative_discount[alive]
        if torch.any(timeout_mask):
            if int(timeout_slots[timeout_mask].min()) < 0:
                raise RuntimeError("truncated transition has no terminal slot")

        next_obs = self.observations[next_rows, env_ids].clone()
        next_goal = self.goals[next_rows, env_ids].clone()
        if torch.any(timeout_mask):
            slots = timeout_slots[timeout_mask]
            next_obs[timeout_mask] = self.terminal_observations[slots]
            next_goal[timeout_mask] = self.terminal_goals[slots]

        target_device = torch.device(device)
        return {
            "obs": self.observations[start_rows, env_ids].to(
                target_device,
                non_blocking=True,
            ),
            "next_obs": next_obs.to(target_device, non_blocking=True),
            "goal": self.goals[start_rows, env_ids].to(
                target_device,
                non_blocking=True,
            ),
            "next_goal": next_goal.to(target_device, non_blocking=True),
            "action": self.actions[start_rows, env_ids].to(
                target_device,
                non_blocking=True,
            ),
            "reward": rewards.unsqueeze(-1).to(target_device, non_blocking=True),
            "discount": bootstrap_discount.unsqueeze(-1).to(
                target_device,
                non_blocking=True,
            ),
        }
