"""IsaacLab environment for rl-games CNN+LSTM actor and privileged critic.

Observation groups:

- policy: RGB image, shape (num_envs, 3, H, W), uint8-like float tensor from TiledCamera.
- critic: privileged vector, shape (num_envs, 8).

Actions are continuous in [-1, 1]:

    [x_speed, y_speed, head_yaw_speed]
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.views import XformPrimView
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from rl_games_ball import task_cfg


@configclass
class RlGamesBallEnvCfg(DirectRLEnvCfg):
    decimation = task_cfg.DECIMATION
    episode_length_s = task_cfg.EPISODE_S
    action_space = 3
    observation_space = [3, task_cfg.IMAGE_HEIGHT, task_cfg.IMAGE_WIDTH]
    state_space = 8

    sim: SimulationCfg = SimulationCfg(
        dt=task_cfg.DT,
        render_interval=decimation,
        render=sim_utils.RenderCfg(
            rendering_mode=task_cfg.RENDERING_MODE,
            antialiasing_mode=task_cfg.ANTIALIASING_MODE,
            enable_dlssg=False,
        ),
    )
    if task_cfg.DLSS_MODE is not None:
        sim.render.dlss_mode = int(task_cfg.DLSS_MODE)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=task_cfg.NUM_ENVS,
        env_spacing=task_cfg.ENV_SPACING,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot_usd = task_cfg.ROBOT_USD
    room_size = task_cfg.ROOM_SIZE
    wall_height = task_cfg.WALL_HEIGHT
    wall_thickness = task_cfg.WALL_THICKNESS
    robot_z = task_cfg.ROBOT_Z
    target_radius = task_cfg.TARGET_RADIUS

    dist_min = task_cfg.DIST_MIN
    dist_max = task_cfg.DIST_MAX
    angle_deg = task_cfg.ANGLE_DEG

    image_width = task_cfg.IMAGE_WIDTH
    image_height = task_cfg.IMAGE_HEIGHT
    focal_length = task_cfg.FOCAL_LENGTH
    fov_x_deg = task_cfg.FOV_X_DEG

    lost_x = task_cfg.LOST_X
    lost_d = task_cfg.LOST_D
    stop_d = task_cfg.STOP_D
    stop_d_tol = task_cfg.STOP_D_TOL
    stop_x_tol = task_cfg.STOP_X_TOL

    stop_eps = task_cfg.STOP_EPS
    stop_n = task_cfg.STOP_N
    fail_near = task_cfg.FAIL_NEAR
    fail_far = task_cfg.FAIL_FAR
    xy_speed = task_cfg.XY_SPEED
    head_speed = task_cfg.HEAD_SPEED

    k_search = task_cfg.K_SEARCH
    k_search_w = task_cfg.K_SEARCH_W
    k_time = task_cfg.K_TIME
    k_x = task_cfg.K_X
    k_d = task_cfg.K_D
    k_stop_a = task_cfg.K_STOP_A
    k_stop_da = task_cfg.K_STOP_DA
    sig_x = task_cfg.SIG_X
    sig_d = task_cfg.SIG_D

    r_find = task_cfg.R_FIND
    r_lost = task_cfg.R_LOST
    r_stop_in = task_cfg.R_STOP_IN
    r_stop_out = task_cfg.R_STOP_OUT
    r_stop_q = task_cfg.R_STOP_Q
    r_stop = task_cfg.R_STOP
    r_success = task_cfg.R_SUCCESS
    r_fail = task_cfg.R_FAIL


class RlGamesBallEnv(DirectRLEnv):
    cfg: RlGamesBallEnvCfg

    def __init__(self, cfg: RlGamesBallEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, int(self.cfg.action_space)), device=self.device)
        self.robot_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.robot_yaw = torch.zeros(self.num_envs, device=self.device)
        self.head_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_xy = torch.zeros((self.num_envs, 2), device=self.device)

        self.stop_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.prev_seen = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_stop = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_xe = torch.zeros(self.num_envs, device=self.device)
        self.prev_de = torch.zeros(self.num_envs, device=self.device)
        self.prev_a_mag = torch.zeros(self.num_envs, device=self.device)

        self.last_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_timeout = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _setup_scene(self):
        env0 = self.scene.env_prim_paths[0]
        self._spawn_env(env0)
        self._standardize_xforms(env0)
        self.camera = TiledCamera(self._camera_cfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.robot_view = XformPrimView(
            f"{self.scene.env_regex_ns}/Robot",
            device=self.device,
            validate_xform_ops=False,
            sync_usd_on_fabric_write=True,
            stage=self.scene.stage,
        )
        self.head_view = XformPrimView(
            f"{self.scene.env_regex_ns}/Robot/head",
            device=self.device,
            validate_xform_ops=False,
            sync_usd_on_fabric_write=True,
            stage=self.scene.stage,
        )
        self.target_view = XformPrimView(
            f"{self.scene.env_regex_ns}/target",
            device=self.device,
            validate_xform_ops=False,
            sync_usd_on_fabric_write=True,
            stage=self.scene.stage,
        )
        self.scene.extras["robot"] = self.robot_view
        self.scene.extras["head"] = self.head_view
        self.scene.extras["target"] = self.target_view
        self.scene.sensors["camera"] = self.camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/light", light_cfg)

    def _standardize_xforms(self, env_path: str):
        for path in (f"{env_path}/Robot", f"{env_path}/Robot/head", f"{env_path}/Robot/head/eye", f"{env_path}/target"):
            prim = self.scene.stage.GetPrimAtPath(path)
            sim_utils.standardize_xform_ops(prim)

    def _spawn_env(self, env_path: str):
        self._spawn_room(env_path)
        sim_utils.create_prim(
            f"{env_path}/Robot",
            "Xform",
            usd_path=self.cfg.robot_usd,
            translation=(0.0, 0.0, self.cfg.robot_z),
        )

        target = sim_utils.SphereCfg(
            radius=self.cfg.target_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
        target.func(f"{env_path}/target", target, translation=(4.0, 0.0, self.cfg.target_radius))

    def _spawn_room(self, env_path: str):
        floor = sim_utils.CuboidCfg(
            size=(self.cfg.room_size, self.cfg.room_size, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.78)),
        )
        floor.func(f"{env_path}/floor", floor, translation=(0.0, 0.0, -0.01))

        wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.65, 0.65))
        wall_x = sim_utils.CuboidCfg(size=(self.cfg.wall_thickness, self.cfg.room_size, self.cfg.wall_height), visual_material=wall_mat)
        wall_y = sim_utils.CuboidCfg(size=(self.cfg.room_size, self.cfg.wall_thickness, self.cfg.wall_height), visual_material=wall_mat)
        half = self.cfg.room_size * 0.5
        z = self.cfg.wall_height * 0.5
        wall_x.func(f"{env_path}/wall_x_pos", wall_x, translation=(half, 0.0, z))
        wall_x.func(f"{env_path}/wall_x_neg", wall_x, translation=(-half, 0.0, z))
        wall_y.func(f"{env_path}/wall_y_pos", wall_y, translation=(0.0, half, z))
        wall_y.func(f"{env_path}/wall_y_neg", wall_y, translation=(0.0, -half, z))

    def _camera_cfg(self) -> TiledCameraCfg:
        aperture = 2.0 * self.cfg.focal_length * math.tan(math.radians(self.cfg.fov_x_deg) * 0.5)
        return TiledCameraCfg(
            prim_path=f"{self.scene.env_regex_ns}/Robot/head/eye/front_camera",
            update_period=0.0,
            height=self.cfg.image_height,
            width=self.cfg.image_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=self.cfg.focal_length,
                focus_distance=400.0,
                horizontal_aperture=aperture,
                clipping_range=(0.1, 100.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        actions = torch.clamp(self.actions, -1.0, 1.0)
        self.actions = actions
        stop_action = torch.max(torch.abs(actions), dim=1).values < self.cfg.stop_eps
        move_actions = actions.clone()
        move_actions[stop_action] = 0.0

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

    def _get_observations(self) -> dict:
        return {"policy": self.camera_rgb_chw(), "critic": self.privileged_obs()}

    def _get_rewards(self) -> torch.Tensor:
        return self.compute_reward()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.compute_terminated()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)
        self.robot_xy[env_ids] = 0.0
        self.robot_yaw[env_ids] = 0.0
        self.head_yaw[env_ids] = 0.0
        self.stop_steps[env_ids] = 0
        self.prev_seen[env_ids] = False
        self.prev_stop[env_ids] = False
        self.prev_xe[env_ids] = 0.0
        self.prev_de[env_ids] = 0.0
        self.prev_a_mag[env_ids] = 0.0
        self.sample_targets(env_ids)
        self.write_robot_pose(env_ids)
        self.write_head_pose(env_ids)
        self.write_target_pose(env_ids)

    def camera_rgb_chw(self) -> torch.Tensor:
        image = self.camera.data.output["rgb"]
        if image.shape[-1] > 3:
            image = image[..., :3]
        is_uint8 = image.dtype == torch.uint8
        image = image.float()
        if is_uint8:
            image = image / 255.0
        return image.permute(0, 3, 1, 2).contiguous()

    def privileged_obs(self) -> torch.Tensor:
        forward, left = self.target_in_head()
        label = self.project_target()
        seen = (label["dist"] > self.cfg.lost_d).float()
        x_n = torch.where(
            label["dist"] > self.cfg.lost_d,
            torch.clamp(label["px_x"] / (self.cfg.image_width * 0.5) - 1.0, -1.0, 1.0),
            torch.full_like(label["px_x"], -1.0),
        )
        d_n = torch.where(
            label["dist"] > self.cfg.lost_d,
            torch.clamp(label["dist"] / self.cfg.fail_far, 0.0, 1.0),
            torch.zeros_like(label["dist"]),
        )
        room_half = self.cfg.room_size * 0.5
        return torch.stack(
            (
                x_n,
                d_n,
                seen,
                torch.clamp(forward / self.cfg.fail_far, -1.0, 1.0),
                torch.clamp(left / self.cfg.fail_far, -1.0, 1.0),
                torch.clamp(self.robot_xy[:, 0] / room_half, -1.0, 1.0),
                torch.clamp(self.robot_xy[:, 1] / room_half, -1.0, 1.0),
                self.head_yaw / math.pi,
            ),
            dim=-1,
        )

    def compute_reward(self) -> torch.Tensor:
        label = self.project_target()
        x = label["px_x"]
        d = label["dist"]
        seen = d > self.cfg.lost_d
        stop = self.in_stop_zone()
        cx = self.cfg.image_width * 0.5
        xe = torch.abs(x - cx)
        de = torch.abs(d - self.cfg.stop_d)

        reward = torch.zeros(self.num_envs, device=self.device)
        reward += torch.where(
            ~seen,
            -self.cfg.k_search * (self.actions[:, 0] ** 2 + self.actions[:, 1] ** 2)
            + self.cfg.k_search_w * torch.abs(self.actions[:, 2]),
            torch.zeros_like(reward),
        )
        reward += torch.where(seen, torch.full_like(reward, -self.cfg.k_time), torch.zeros_like(reward))
        a_mag = torch.max(torch.abs(self.actions), dim=1).values
        app = seen & self.prev_seen & (~stop)
        x_gain = (self.prev_xe - xe) / self.cfg.image_width
        d_gain = (self.prev_de - de) / self.cfg.fail_far
        reward += torch.where(
            app,
            self.cfg.k_x * x_gain + self.cfg.k_d * d_gain,
            torch.zeros_like(reward),
        )
        reward += torch.where((~self.prev_seen) & seen, torch.full_like(reward, self.cfg.r_find), 0.0)
        reward += torch.where(self.prev_seen & (~seen), torch.full_like(reward, self.cfg.r_lost), 0.0)
        reward += torch.where((~self.prev_stop) & stop, torch.full_like(reward, self.cfg.r_stop_in), 0.0)
        reward += torch.where(self.prev_stop & (~stop), torch.full_like(reward, self.cfg.r_stop_out), 0.0)
        reward += torch.where(
            stop & self.prev_stop,
            self.cfg.k_stop_da * (self.prev_a_mag - a_mag),
            torch.zeros_like(reward),
        )

        stop_q = torch.exp(-0.5 * ((x - cx) / self.cfg.sig_x) ** 2 - 0.5 * ((d - self.cfg.stop_d) / self.cfg.sig_d) ** 2)
        stop_action = a_mag < self.cfg.stop_eps
        reward += torch.where(stop & stop_action, self.cfg.r_stop * stop_q, torch.zeros_like(reward))

        success = self.stop_steps >= self.cfg.stop_n
        fail = self.out()
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        self.last_success = success
        self.last_fail = fail
        self.last_timeout = timeout & (~success) & (~fail)
        reward += torch.where(success, self.cfg.r_success * stop_q, 0.0)
        reward += torch.where(fail & (~success), torch.full_like(reward, self.cfg.r_fail), 0.0)

        self.prev_seen = seen
        self.prev_stop = stop
        self.prev_xe = torch.where(seen, xe, self.prev_xe)
        self.prev_de = torch.where(seen, de, self.prev_de)
        self.prev_a_mag = a_mag
        return reward

    def compute_terminated(self) -> torch.Tensor:
        return (self.stop_steps >= self.cfg.stop_n) | self.out()

    def out(self) -> torch.Tensor:
        dist = torch.linalg.norm(self.target_xy - self.robot_xy, dim=1)
        return (dist < self.cfg.fail_near) | (dist > self.cfg.fail_far)

    def sample_targets(self, env_ids: torch.Tensor):
        count = len(env_ids)
        dist = torch.empty(count, device=self.device).uniform_(self.cfg.dist_min, self.cfg.dist_max)
        limit = math.radians(self.cfg.angle_deg)
        angle = torch.empty(count, device=self.device).uniform_(-limit, limit)
        self.target_xy[env_ids, 0] = dist * torch.cos(angle)
        self.target_xy[env_ids, 1] = dist * torch.sin(angle)

    def write_robot_pose(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        origins = self.scene.env_origins[env_ids]
        pos = origins.clone()
        pos[:, 0:2] += self.robot_xy[env_ids]
        pos[:, 2] += self.cfg.robot_z
        self.robot_view.set_world_poses(positions=pos, indices=env_ids.tolist())

    def write_head_pose(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        quat = self.yaw_quat(self.head_yaw[env_ids])
        self.head_view.set_local_poses(orientations=quat, indices=env_ids.tolist())

    def write_target_pose(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        origins = self.scene.env_origins[env_ids]
        pos = origins.clone()
        pos[:, 0:2] += self.target_xy[env_ids]
        pos[:, 2] += self.cfg.target_radius
        self.target_view.set_world_poses(positions=pos, indices=env_ids.tolist())

    def target_in_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.target_xy - self.robot_xy
        yaw = self.robot_yaw + self.head_yaw
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        forward = c * delta[:, 0] + s * delta[:, 1]
        left = -s * delta[:, 0] + c * delta[:, 1]
        return forward, left

    def project_target(self) -> dict[str, torch.Tensor]:
        forward, left = self.target_in_head()
        cx = self.cfg.image_width * 0.5
        fx = self.cfg.image_width / (2.0 * math.tan(math.radians(self.cfg.fov_x_deg) * 0.5))
        px_x = cx - fx * left / torch.clamp(forward, min=1e-6)
        in_view = (forward > 0.0) & (px_x >= 0.0) & (px_x < self.cfg.image_width)
        px_x = torch.where(in_view, px_x, torch.full_like(px_x, self.cfg.lost_x))
        dist = torch.where(in_view, forward, torch.full_like(forward, self.cfg.lost_d))
        return {"px_x": px_x, "dist": dist}

    def in_stop_zone(self) -> torch.Tensor:
        label = self.project_target()
        cx = self.cfg.image_width * 0.5
        dist_ok = torch.abs(label["dist"] - self.cfg.stop_d) <= self.cfg.stop_d_tol
        px_ok = torch.abs(label["px_x"] - cx) <= self.cfg.stop_x_tol
        return dist_ok & px_ok

    def yaw_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(yaw)
        return quat_from_euler_xyz(zeros, zeros, yaw)


def make_env(cfg: RlGamesBallEnvCfg | None = None) -> RlGamesBallEnv:
    return RlGamesBallEnv(cfg or RlGamesBallEnvCfg())
