"""任务 v3「稠密位置分」的 env 变体(2026-09-04)。

与 v1 的差别只有两点,其余(动作、观测、场景、薄记、terminal 帧捕获)全部继承:

- **reward**:丢弃 v1 的全部项,改为纯状态分(2026-09-09 起看不见球时加回 v1 的找球项)
      score = 1 / (1 + xe/SCORE_X_SCALE + de/SCORE_D_SCALE)      看不见球 -> -K_SEARCH·(ax²+ay²) + K_SEARCH_W·|aw|
  2026-09-11 起 SCORE_MODE="angle":xe 换成方位角(deg)/SCORE_ANG_SCALE,de 用真实距离;stop 区 = |方位角|≤3° 且 |距离-1.5|≤0.2
  上限 1、离目标越近越高、处处有梯度。没有 stop 判定、没有事件项、没有终止项。

- **终止**:恒 False。每条 episode 强制跑满 375 步(15s),回报 = 375 步得分之和。

为什么这么设计(都有实测支撑,见 2026-09-04 的分析):
  · v1 的稀疏精确停止让成功率非 0 即 1 -> 回报没方差 -> 优势归一化把噪声放大
    -> 纯 PPO 的 KL 冲到 34.5、BC 撤掉就从 1.000 塌到 0.008。**两个极端同一个病。**
  · 高斯型状态分(hover)在 2.5-4m 处 stop_q ≈ 0.004,接近段没有梯度;
    一阶倒数是多项式衰减,4m 处仍有 0.162。
  · 纯线性(1 - e)虽然处处有梯度,但 1.7m 与 1.5m 只差 0.016,最后一米等于白走。

实现上跑一遍 super().compute_reward() 只为薄记(last_*/prev_* 快照、terminal 帧捕获),
它的返回值直接丢弃。
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src import task_cfg
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg
from src.noise_env import NoiseMixin, NoisePPOEnvCfg, NoiseStudentEnvCfg
from src.sb3_env import BallPPOEnv, BallPPOEnvCfg


@configclass
class ScorePPOEnvCfg(BallPPOEnvCfg):
    score_x_scale = task_cfg.SCORE_X_SCALE
    score_ang_scale = task_cfg.SCORE_ANG_SCALE
    score_diam_scale = task_cfg.SCORE_DIAM_SCALE
    score_d_scale = task_cfg.SCORE_D_SCALE
    score_action_k = task_cfg.SCORE_ACTION_K
    score_action_dead = task_cfg.SCORE_ACTION_DEAD


@configclass
class ScoreNoisePPOEnvCfg(NoisePPOEnvCfg):
    score_x_scale = task_cfg.SCORE_X_SCALE
    score_ang_scale = task_cfg.SCORE_ANG_SCALE
    score_diam_scale = task_cfg.SCORE_DIAM_SCALE
    score_d_scale = task_cfg.SCORE_D_SCALE
    score_action_k = task_cfg.SCORE_ACTION_K
    score_action_dead = task_cfg.SCORE_ACTION_DEAD


@configclass
class ScoreStudentEnvCfg(MDPStudentEnvCfg):
    score_x_scale = task_cfg.SCORE_X_SCALE
    score_ang_scale = task_cfg.SCORE_ANG_SCALE
    score_diam_scale = task_cfg.SCORE_DIAM_SCALE
    score_d_scale = task_cfg.SCORE_D_SCALE
    score_action_k = task_cfg.SCORE_ACTION_K
    score_action_dead = task_cfg.SCORE_ACTION_DEAD


@configclass
class ScoreNoiseStudentEnvCfg(NoiseStudentEnvCfg):
    score_x_scale = task_cfg.SCORE_X_SCALE
    score_ang_scale = task_cfg.SCORE_ANG_SCALE
    score_diam_scale = task_cfg.SCORE_DIAM_SCALE
    score_d_scale = task_cfg.SCORE_D_SCALE
    score_action_k = task_cfg.SCORE_ACTION_K
    score_action_dead = task_cfg.SCORE_ACTION_DEAD


class ScoreMixin:
    def compute_reward(self) -> torch.Tensor:
        super().compute_reward()  # 只为薄记,返回值丢弃
        x = self.last_px_x
        d = self.last_dist
        seen = d > self.cfg.lost_d
        if self.cfg.score_mode == "angle":   # 物理量版:方位角(deg)/8.5 + |真实距离-1.5|/0.5
            lab = self.project_target()
            xe = torch.abs(lab["bearing_deg"])
            de = torch.abs(lab["range"] - self.end_d)
            score = 1.0 / (1.0 + xe / self.cfg.score_ang_scale + de / self.cfg.score_d_scale)
        elif self.cfg.score_mode == "cam":  # 横向用相机坐标(像素),距离与 angle 模式共用物理 range
            lab = self.project_target()
            xe = torch.abs(lab["px_x"] - (self.cfg.image_width * 0.5 + self.end_x))
            de = torch.abs(lab["range"] - self.end_d)
            score = 1.0 / (1.0 + xe / self.cfg.score_x_scale + de / self.cfg.score_d_scale)
        else:
            end_px = self.cfg.image_width * 0.5 + self.end_x
            xe = torch.abs(x - end_px)
            de = torch.abs(d - self.end_d)
            score = 1.0 / (1.0 + xe / self.cfg.score_x_scale + de / self.cfg.score_d_scale)
        a_mag = torch.clamp(self.actions, -1.0, 1.0).abs().max(dim=1).values
        excess = torch.clamp(a_mag - self.cfg.score_action_dead, min=0.0)
        score = score * torch.clamp(1.0 - self.cfg.score_action_k * excess, min=0.0)
        # 找球项(2026-09-09 加回,沿用 v1 的 K_SEARCH / K_SEARCH_W):看不见球时罚平移、奖转头,
        # 去掉"未见 -> 恒 0 -> 零梯度吸收态"(R s2 跑丢就是这个)。无侧移管线里 a_y 恒 0,项自然退化。
        search = (
            -self.cfg.k_search * (self.actions[:, 0] ** 2 + self.actions[:, 1] ** 2)
            + self.cfg.k_search_w * torch.abs(self.actions[:, 2])
        )
        return torch.where(seen, score, search)

    def compute_terminated(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


class ScorePPOEnv(ScoreMixin, BallPPOEnv):
    cfg: ScorePPOEnvCfg


class ScoreNoisePPOEnv(ScoreMixin, NoiseMixin, BallPPOEnv):
    cfg: ScoreNoisePPOEnvCfg

    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_noise()


class ScoreStudentEnv(ScoreMixin, MDPStudentEnv):
    cfg: ScoreStudentEnvCfg


class ScoreNoiseStudentEnv(ScoreMixin, NoiseMixin, MDPStudentEnv):
    cfg: ScoreNoiseStudentEnvCfg

    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_noise()


def make_score_env(cfg: ScorePPOEnvCfg | None = None) -> ScorePPOEnv:
    return ScorePPOEnv(cfg or ScorePPOEnvCfg())


def make_score_sb3_env(cfg: ScorePPOEnvCfg | None = None, render_mode: str | None = None, fast_variant: bool = True):
    """teacher 用:特权观测 + 新奖励,无相机。"""
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    return Sb3VecEnvWrapper(ScorePPOEnv(cfg or ScorePPOEnvCfg(), render_mode=render_mode), fast_variant=fast_variant)


def make_score_student_env(cfg: ScoreStudentEnvCfg | None = None) -> ScoreStudentEnv:
    return ScoreStudentEnv(cfg or ScoreStudentEnvCfg())


def make_score_noise_student_env(cfg: ScoreNoiseStudentEnvCfg | None = None) -> ScoreNoiseStudentEnv:
    return ScoreNoiseStudentEnv(cfg or ScoreNoiseStudentEnvCfg())
