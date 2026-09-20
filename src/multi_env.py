"""4 物体环境(2026-09-19 周末批次)。

槽位固定:**0=红球 1=红方 2=蓝球 3=蓝方**。`cfg.target_slot ∈ {0,1,2,3}` 选谁当目标——
奖励、停车判定、特权标签全部跟着走(覆写 `target_in_head`,和现有 `chase_blue` 用同一个接缝)。

为什么要 4 个:现在的 2 物体环境里蓝球**既是干扰物、又是非红物体**,
所以「A 丢弃干扰物」和「A 丢弃非红物体」在数据上分不开。
红方是一个**能通过颜色滤波器的干扰物**,蓝球是一个**能通过形状滤波器的干扰物**,
两者同时在场就把这个混淆拆开了(conjunction search / 特征绑定)。

方块边长 = 球直径 = 2r(用户 2026-09-18 定)。
**注意**:正投影面积因此比球大 4/π ≈ 1.27 倍。A 的响应是跟红像素面积走的
(红大球 0.765 > 红立方 0.731 > 红球 0.607 > 红锥 0.519),所以分析「追红方」的成绩时
**必须把各物体的红/蓝像素面积作为协变量一起报**,否则说不清是颜色还是面积。

四个物体每回合都出现、位置各自独立采样,所有 6 对间距 ≥ MULTI_MIN_DIST。
干扰物只是视觉:无碰撞、不进 teacher 的特权观测(teacher 训练命令与现在完全一致,不带 --noise)。

本文件自带 Score* 组合类与工厂,**不改 noise_env / score_env / env**,
因为那三个文件正被在跑的机制拆分臂链条使用。
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim.views import XformPrimView
from isaaclab.utils import configclass

from src import task_cfg
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg
from src.sb3_env import BallPPOEnv, BallPPOEnvCfg
from src.score_env import ScoreMixin


MULTI_MIN_DIST = 0.5

RED = (1.0, 0.0, 0.0)      # 与 env.py 的目标球同色
BLUE = (0.0, 0.1, 1.0)     # 与 noise_env.py 的干扰球同色

# (名字, 颜色, 形状);slot 0 由 BallEnv 自己生成,这里只列出来对齐编号
SLOT_SPEC = (
    ("red_sphere", RED, "sphere"),
    ("red_cube", RED, "cube"),
    ("blue_sphere", BLUE, "sphere"),
    ("blue_cube", BLUE, "cube"),
)
SLOT_NAMES = tuple(s[0] for s in SLOT_SPEC)


def _shape_cfg(shape: str, r: float, color):
    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    if shape == "cube":        # 边长 = 球直径 = 2r
        return sim_utils.CuboidCfg(size=(2 * r, 2 * r, 2 * r), visual_material=mat)
    return sim_utils.SphereCfg(radius=r, visual_material=mat)


@configclass
class MultiPPOEnvCfg(BallPPOEnvCfg):
    target_slot = 0


@configclass
class MultiStudentEnvCfg(MDPStudentEnvCfg):
    target_slot = 0


class MultiMixin:
    """在 BallEnv 的红球之外再摆 3 个物体,并让 target_slot 决定谁是目标。"""

    def _setup_scene(self):
        super()._setup_scene()
        self.extra_views = []
        for k in range(1, 4):
            view = XformPrimView(
                f"{self.scene.env_regex_ns}/obj{k}",
                device=self.device,
                validate_xform_ops=False,
                sync_usd_on_fabric_write=True,
                stage=self.scene.stage,
            )
            self.extra_views.append(view)
            self.scene.extras[f"obj{k}"] = view

    def _standardize_xforms(self, env_path: str):
        super()._standardize_xforms(env_path)
        for k in range(1, 4):
            sim_utils.standardize_xform_ops(self.scene.stage.GetPrimAtPath(f"{env_path}/obj{k}"))

    def _spawn_env(self, env_path: str):
        super()._spawn_env(env_path)          # slot 0:红球,由 BallEnv 生成
        r = self.cfg.target_radius
        for k in range(1, 4):
            _, color, shape = SLOT_SPEC[k]
            cfg = _shape_cfg(shape, r, color)
            cfg.func(f"{env_path}/obj{k}", cfg, translation=(4.0, 1.0 * k, r))

    def _ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    def _reset_idx(self, env_ids):
        ids = self._ids(env_ids)
        super()._reset_idx(ids)               # 这里会 sample_targets(slot 0) + write_target_pose
        if hasattr(self, "extra_xy"):
            self.sample_slots(ids)
            self.write_slot_poses(ids)

    def sample_slots(self, env_ids: torch.Tensor):
        """slot 1..3 逐个拒绝采样,保证与已放好的每一个物体都 ≥ MULTI_MIN_DIST。"""
        count = len(env_ids)
        limit = math.radians(self.cfg.angle_deg)
        placed = [self.target_xy[env_ids]]
        for k in range(3):
            xy = torch.empty((count, 2), device=self.device)
            bad = torch.ones(count, device=self.device, dtype=torch.bool)
            while torch.any(bad):
                n = int(bad.sum().item())
                dist = torch.empty(n, device=self.device).uniform_(self.cfg.dist_min, self.cfg.dist_max)
                angle = torch.empty(n, device=self.device).uniform_(-limit, limit)
                xy[bad] = torch.stack((dist * torch.cos(angle), dist * torch.sin(angle)), dim=1)
                bad = torch.zeros(count, device=self.device, dtype=torch.bool)
                for prev in placed:
                    bad |= torch.linalg.norm(xy - prev, dim=1) < MULTI_MIN_DIST
            placed.append(xy)
            self.extra_xy[env_ids, k] = xy

    def write_slot_poses(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        origins = self.scene.env_origins[env_ids]
        for k, view in enumerate(self.extra_views):
            pos = origins.clone()
            pos[:, 0:2] += self.extra_xy[env_ids, k]
            pos[:, 2] += self.cfg.target_radius
            view.set_world_poses(positions=pos, indices=env_ids.tolist())

    def init_slots(self):
        self.extra_xy = torch.zeros((self.num_envs, 3, 2), device=self.device)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.sample_slots(env_ids)
        self.write_slot_poses(env_ids)

    def slot_xy(self) -> torch.Tensor:
        """(N, 4, 2):0=红球 1=红方 2=蓝球 3=蓝方。"""
        return torch.cat((self.target_xy.unsqueeze(1), self.extra_xy), dim=1)

    def target_in_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.slot_xy()[:, int(self.cfg.target_slot)]
        delta = src - self.robot_xy
        yaw = self.robot_yaw + self.head_yaw
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        return c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]


@configclass
class ScoreMultiPPOEnvCfg(MultiPPOEnvCfg):
    score_x_scale = task_cfg.SCORE_X_SCALE
    score_ang_scale = task_cfg.SCORE_ANG_SCALE
    score_diam_scale = task_cfg.SCORE_DIAM_SCALE
    score_d_scale = task_cfg.SCORE_D_SCALE
    score_action_k = task_cfg.SCORE_ACTION_K
    score_action_dead = task_cfg.SCORE_ACTION_DEAD


@configclass
class ScoreMultiStudentEnvCfg(MultiStudentEnvCfg):
    score_x_scale = task_cfg.SCORE_X_SCALE
    score_ang_scale = task_cfg.SCORE_ANG_SCALE
    score_diam_scale = task_cfg.SCORE_DIAM_SCALE
    score_d_scale = task_cfg.SCORE_D_SCALE
    score_action_k = task_cfg.SCORE_ACTION_K
    score_action_dead = task_cfg.SCORE_ACTION_DEAD


class ScoreMultiPPOEnv(ScoreMixin, MultiMixin, BallPPOEnv):
    cfg: ScoreMultiPPOEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.init_slots()


class ScoreMultiStudentEnv(ScoreMixin, MultiMixin, MDPStudentEnv):
    cfg: ScoreMultiStudentEnvCfg

    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_slots()


def make_score_multi_student_env(cfg: ScoreMultiStudentEnvCfg | None = None) -> ScoreMultiStudentEnv:
    return ScoreMultiStudentEnv(cfg or ScoreMultiStudentEnvCfg())
