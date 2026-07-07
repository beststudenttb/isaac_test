"""Ball task variants with one blue distractor ball."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim.views import XformPrimView
from isaaclab.utils import configclass

from src.env import BallEnv, BallEnvCfg
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg
from src.sb3_env import BallPPOEnv, BallPPOEnvCfg


NOISE_MIN_DIST = 0.5


@configclass
class NoiseEnvCfg(BallEnvCfg):
    pass


@configclass
class NoisePPOEnvCfg(BallPPOEnvCfg):
    pass


@configclass
class NoiseStudentEnvCfg(MDPStudentEnvCfg):
    pass


class NoiseMixin:
    """Adds a blue distractor ball while leaving red-target labels unchanged."""

    def _setup_scene(self):
        super()._setup_scene()
        self.noise_view = XformPrimView(
            f"{self.scene.env_regex_ns}/noise",
            device=self.device,
            validate_xform_ops=False,
            sync_usd_on_fabric_write=True,
            stage=self.scene.stage,
        )
        self.scene.extras["noise"] = self.noise_view

    def _standardize_xforms(self, env_path: str):
        super()._standardize_xforms(env_path)
        prim = self.scene.stage.GetPrimAtPath(f"{env_path}/noise")
        sim_utils.standardize_xform_ops(prim)

    def _spawn_env(self, env_path: str):
        super()._spawn_env(env_path)
        noise = sim_utils.SphereCfg(
            radius=self.cfg.target_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 1.0)),
        )
        noise.func(
            f"{env_path}/noise",
            noise,
            translation=(4.0, 1.0, self.cfg.target_radius),
        )

    def _reset_idx(self, env_ids):
        ids = self._ids(env_ids)
        super()._reset_idx(ids)
        if hasattr(self, "noise_xy"):
            self.sample_noise(ids)
            self.write_noise_pose(ids)

    def _ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    def sample_noise(self, env_ids: torch.Tensor):
        count = len(env_ids)
        limit = math.radians(self.cfg.angle_deg)
        noise = torch.empty((count, 2), device=self.device)
        target = self.target_xy[env_ids]
        mask = torch.ones(count, device=self.device, dtype=torch.bool)
        while torch.any(mask):
            n = int(mask.sum().item())
            dist = torch.empty(n, device=self.device).uniform_(self.cfg.dist_min, self.cfg.dist_max)
            angle = torch.empty(n, device=self.device).uniform_(-limit, limit)
            sample = torch.stack((dist * torch.cos(angle), dist * torch.sin(angle)), dim=1)
            noise[mask] = sample
            mask = torch.linalg.norm(noise - target, dim=1) < NOISE_MIN_DIST
        self.noise_xy[env_ids] = noise

    def write_noise_pose(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        origins = self.scene.env_origins[env_ids]
        pos = origins.clone()
        pos[:, 0:2] += self.noise_xy[env_ids]
        pos[:, 2] += self.cfg.target_radius
        self.noise_view.set_world_poses(positions=pos, indices=env_ids.tolist())

    def init_noise(self):
        self.noise_xy = torch.zeros((self.num_envs, 2), device=self.device)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.sample_noise(env_ids)
        self.write_noise_pose(env_ids)


class NoiseEnv(NoiseMixin, BallEnv):
    cfg: NoiseEnvCfg

    def __init__(self, cfg: NoiseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.init_noise()


class NoisePPOEnv(NoiseMixin, BallPPOEnv):
    cfg: NoisePPOEnvCfg

    def __init__(self, cfg: NoisePPOEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.init_noise()


class NoiseStudentEnv(NoiseMixin, MDPStudentEnv):
    cfg: NoiseStudentEnvCfg

    def __init__(self, cfg: NoiseStudentEnvCfg):
        super().__init__(cfg)
        self.init_noise()


def make_noise_sb3_env(
    cfg: NoisePPOEnvCfg | None = None,
    render_mode: str | None = None,
    fast_variant: bool = True,
):
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = NoisePPOEnv(cfg or NoisePPOEnvCfg(), render_mode=render_mode)
    return Sb3VecEnvWrapper(env, fast_variant=fast_variant)


def make_noise_student_env(cfg: NoiseStudentEnvCfg | None = None) -> NoiseStudentEnv:
    return NoiseStudentEnv(cfg or NoiseStudentEnvCfg())
