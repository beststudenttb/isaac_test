"""SB3 PPO environment for the ball robot task."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src import task_cfg
from src.env import BallEnv, BallEnvCfg


@configclass
class BallPPOEnvCfg(BallEnvCfg):
    action_space = 3
    observation_space = 2
    episode_length_s = 10.0
    read_camera = False

    xy_speed = task_cfg.XY_SPEED
    head_speed = task_cfg.HEAD_SPEED
    stop_eps = task_cfg.STOP_EPS
    stop_n = task_cfg.STOP_N
    fail_near = task_cfg.FAIL_NEAR
    fail_far = task_cfg.FAIL_FAR

    k_search = task_cfg.K_SEARCH
    k_x = task_cfg.K_X
    k_d = task_cfg.K_D
    k_stop_a = task_cfg.K_STOP_A

    sig_x = task_cfg.SIG_X
    sig_d = task_cfg.SIG_D

    r_find = task_cfg.R_FIND
    r_lost = task_cfg.R_LOST
    r_stop_in = task_cfg.R_STOP_IN
    r_stop_out = task_cfg.R_STOP_OUT
    r_stop_q = task_cfg.R_STOP_Q
    r_success = task_cfg.R_SUCCESS
    r_fail = task_cfg.R_FAIL


class BallPPOEnv(BallEnv):
    cfg: BallPPOEnvCfg

    def __init__(self, cfg: BallPPOEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.stop_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.prev_seen = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_stop = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_xe = torch.zeros(self.num_envs, device=self.device)
        self.prev_de = torch.zeros(self.num_envs, device=self.device)
        self.last_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_timeout = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_px_x = torch.zeros(self.num_envs, device=self.device)
        self.last_dist = torch.zeros(self.num_envs, device=self.device)
        self.last_robot_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_target_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_head_yaw = torch.zeros(self.num_envs, device=self.device)
        self.last_move_actions = torch.zeros((self.num_envs, int(self.cfg.action_space)), device=self.device)
        self.last_policy_obs = torch.zeros((self.num_envs, int(self.cfg.observation_space)), device=self.device)
        self.last_rgb = None

    def _reset_idx(self, env_ids):
        if env_ids is None:
            reset_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            reset_ids = env_ids
        else:
            reset_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(reset_ids)
        if hasattr(self, "stop_steps"):
            self.stop_steps[reset_ids] = 0
            self.prev_seen[reset_ids] = False
            self.prev_stop[reset_ids] = False
            self.prev_xe[reset_ids] = 0.0
            self.prev_de[reset_ids] = 0.0

    def apply_actions(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions
        stop_action = torch.max(torch.abs(actions), dim=1).values < self.cfg.stop_eps
        move_actions = actions.clone()
        move_actions[stop_action] = 0.0
        self.last_move_actions = move_actions.clone()

        yaw = self.robot_yaw + self.head_yaw
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        vx = c * move_actions[:, 0] - s * move_actions[:, 1]
        vy = s * move_actions[:, 0] + c * move_actions[:, 1]
        self.robot_xy[:, 0] += vx * self.cfg.xy_speed * self.step_dt
        self.robot_xy[:, 1] += vy * self.cfg.xy_speed * self.step_dt
        self.head_yaw += move_actions[:, 2] * self.cfg.head_speed * self.step_dt
        self.head_yaw = torch.atan2(torch.sin(self.head_yaw), torch.cos(self.head_yaw))
        self.write_robot_pose()
        self.write_head_pose()

        done_step = self.in_stop_zone() & stop_action
        self.stop_steps = torch.where(done_step, self.stop_steps + 1, torch.zeros_like(self.stop_steps))

    def make_observation(self) -> torch.Tensor:
        if self.cfg.use_camera and self.cfg.read_camera and self.camera is not None:
            _ = self.camera.data.output["rgb"]
        label = self.project_target()
        return torch.stack((self.norm_x(label["px_x"], label["dist"]), self.norm_d(label["dist"])), dim=-1)

    def norm_x(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        seen = d > self.cfg.lost_d
        x_n = x / (self.cfg.image_width * 0.5) - 1.0
        return torch.where(seen, torch.clamp(x_n, -1.0, 1.0), torch.full_like(x_n, -1.0))

    def norm_d(self, d: torch.Tensor) -> torch.Tensor:
        seen = d > self.cfg.lost_d
        d_n = d / self.cfg.fail_far
        return torch.where(seen, torch.clamp(d_n, 0.0, 1.0), torch.zeros_like(d_n))

    def compute_reward(self) -> torch.Tensor:
        label = self.project_target()
        x = label["px_x"]
        d = label["dist"]
        self.last_policy_obs = torch.stack((self.norm_x(x, d), self.norm_d(d)), dim=-1)
        if self.cfg.use_camera and self.cfg.read_camera and self.camera is not None and torch.any(self.reset_time_outs):
            rgb = self.camera.data.output["rgb"]
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            if self.last_rgb is None or self.last_rgb.shape != rgb.shape:
                self.last_rgb = torch.empty_like(rgb)
            self.last_rgb[self.reset_time_outs] = rgb[self.reset_time_outs]
        self.last_px_x = x.clone()
        self.last_dist = d.clone()
        self.last_robot_xy = self.robot_xy.clone()
        self.last_target_xy = self.target_xy.clone()
        self.last_head_yaw = self.head_yaw.clone()
        seen = d > self.cfg.lost_d
        stop = self.in_stop_zone()
        cx = self.cfg.image_width * 0.5
        xe = torch.abs(x - cx)
        de = torch.abs(d - self.cfg.stop_d)

        reward = torch.zeros(self.num_envs, device=self.device)

        reward += torch.where(
            ~seen,
            -self.cfg.k_search * (self.actions[:, 0] ** 2 + self.actions[:, 1] ** 2),
            torch.zeros_like(reward),
        )

        app = seen & self.prev_seen & ~stop
        reward += torch.where(
            app,
            self.cfg.k_x * (self.prev_xe - xe) + self.cfg.k_d * (self.prev_de - de),
            torch.zeros_like(reward),
        )

        reward += torch.where((~self.prev_seen) & seen, torch.full_like(reward, self.cfg.r_find), 0.0)
        reward += torch.where(self.prev_seen & (~seen), torch.full_like(reward, self.cfg.r_lost), 0.0)
        reward += torch.where((~self.prev_stop) & stop, torch.full_like(reward, self.cfg.r_stop_in), 0.0)
        reward += torch.where(self.prev_stop & (~stop), torch.full_like(reward, self.cfg.r_stop_out), 0.0)

        stop_q = torch.exp(
            -0.5 * ((x - cx) / self.cfg.sig_x) ** 2
            -0.5 * ((d - self.cfg.stop_d) / self.cfg.sig_d) ** 2
        )
        act_cost = torch.sum(self.actions * self.actions, dim=1)
        reward += torch.where(
            stop,
            self.cfg.r_stop_q * stop_q - self.cfg.k_stop_a * act_cost,
            torch.zeros_like(reward),
        )

        success = self.stop_steps >= self.cfg.stop_n
        fail = self.out()
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        self.last_success = success
        self.last_fail = fail
        self.last_timeout = timeout & (~success) & (~fail)
        reward += torch.where(success, torch.full_like(reward, self.cfg.r_success), 0.0)
        reward += torch.where(fail & (~success), torch.full_like(reward, self.cfg.r_fail), 0.0)

        self.prev_seen = seen
        self.prev_stop = stop
        self.prev_xe = torch.where(seen, xe, self.prev_xe)
        self.prev_de = torch.where(seen, de, self.prev_de)
        return reward

    def compute_terminated(self) -> torch.Tensor:
        return (self.stop_steps >= self.cfg.stop_n) | self.out()

    def out(self) -> torch.Tensor:
        dist = torch.linalg.norm(self.target_xy - self.robot_xy, dim=1)
        return (dist < self.cfg.fail_near) | (dist > self.cfg.fail_far)


def make_sb3_env(cfg: BallPPOEnvCfg | None = None, render_mode: str | None = None, fast_variant: bool = True):
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = BallPPOEnv(cfg or BallPPOEnvCfg(), render_mode=render_mode)
    return Sb3VecEnvWrapper(env, fast_variant=fast_variant)
