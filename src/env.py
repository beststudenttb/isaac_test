"""Base IsaacLab environment for the ball robot task.

Subclasses should keep task-specific action, observation and reward logic here,
instead of rebuilding rooms, cameras and target sampling in every script.
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

from src import task_cfg


@configclass
class BallEnvCfg(DirectRLEnvCfg):
    decimation = task_cfg.DECIMATION
    episode_length_s = task_cfg.EPISODE_S
    action_space = 3
    observation_space = 2
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=task_cfg.DT,
        render_interval=decimation,
        render=sim_utils.RenderCfg(rendering_mode=task_cfg.RENDERING_MODE),
    )
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

    use_camera = task_cfg.USE_CAMERA
    image_width = task_cfg.IMAGE_WIDTH
    image_height = task_cfg.IMAGE_HEIGHT
    focal_length = task_cfg.FOCAL_LENGTH
    fov_x_deg = task_cfg.FOV_X_DEG

    lost_x = task_cfg.LOST_X
    lost_d = task_cfg.LOST_D
    stop_d = task_cfg.STOP_D
    stop_d_tol = task_cfg.STOP_D_TOL
    stop_x_tol = task_cfg.STOP_X_TOL


class BallEnv(DirectRLEnv):
    """Shared base env for Isaac ball-robot RL tasks.

    The base class owns the common scene:
    room, robot USD, target ball, optional tiled camera and random target sampling.
    Subclasses normally override these public hooks:

    - ``apply_actions``
    - ``make_observation``
    - ``compute_reward``
    - ``compute_terminated``
    """

    cfg: BallEnvCfg

    def __init__(self, cfg: BallEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, int(self.cfg.action_space)), device=self.device)
        self.robot_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.robot_yaw = torch.zeros(self.num_envs, device=self.device)
        self.head_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_xy = torch.zeros((self.num_envs, 2), device=self.device)

    def _setup_scene(self):
        env0 = self.scene.env_prim_paths[0]
        self._spawn_env(env0)
        self._standardize_xforms(env0)

        if self.cfg.use_camera:
            self.camera = TiledCamera(self._camera_cfg())
        else:
            self.camera = None

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

        if self.camera is not None:
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
        target.func(
            f"{env_path}/target",
            target,
            translation=(4.0, 0.0, self.cfg.target_radius),
        )

    def _spawn_room(self, env_path: str):
        floor = sim_utils.CuboidCfg(
            size=(self.cfg.room_size, self.cfg.room_size, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.78)),
        )
        floor.func(f"{env_path}/floor", floor, translation=(0.0, 0.0, -0.01))

        wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.65, 0.65))
        wall_x = sim_utils.CuboidCfg(
            size=(self.cfg.wall_thickness, self.cfg.room_size, self.cfg.wall_height),
            visual_material=wall_mat,
        )
        wall_y = sim_utils.CuboidCfg(
            size=(self.cfg.room_size, self.cfg.wall_thickness, self.cfg.wall_height),
            visual_material=wall_mat,
        )
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
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.apply_actions(self.actions)

    def _get_observations(self) -> dict:
        return {"policy": self.make_observation()}

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
        self.sample_targets(env_ids)
        self.write_robot_pose(env_ids)
        self.write_head_pose(env_ids)
        self.write_target_pose(env_ids)

    def apply_actions(self, actions: torch.Tensor) -> None:
        """Apply task actions. Subclasses should override this."""

    def make_observation(self) -> torch.Tensor:
        """Return default privileged observation: px_x, dist."""
        label = self.project_target()
        return torch.stack((label["px_x"], label["dist"]), dim=-1)

    def compute_reward(self) -> torch.Tensor:
        """Return zero rewards by default. Subclasses should override this."""
        return torch.zeros(self.num_envs, device=self.device)

    def compute_terminated(self) -> torch.Tensor:
        """Return no task termination by default. Subclasses should override this."""
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

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
