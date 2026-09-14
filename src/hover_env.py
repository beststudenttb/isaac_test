"""任务 v2「悬停」env 变体(见 summary/2026_07_30_hover_v2_plan.md)。

与 v1 的差只有两个任务核心点,其余(动作、观测、场景、快照薄记、terminal 帧捕获)全部继承:
- reward:删掉全部 goal 事件/终止项(R_STOP_IN/OUT、K_STOP_DA、R_STOP、R_SUCCESS、R_FAIL),
  改为 seen 时每步 +R_HOVER·stop_q(现有二维高斯原样复用)。goal 无关项(搜索、K_TIME、
  R_FIND/R_LOST、telescoping 接近 shaping)一字不动。
- 终止:无(success 机制不触发、出界不终止),每条 episode 强制跑满到 timeout。

实现方式是"先跑完整 v1 链再精确撤项":super().compute_reward() 负责全部薄记
(last_* 快照、prev_* 更新、MDPStudentEnv 的 terminal 帧捕获),本 mixin 用快照把
v1 加过的事件项逐项减回去、再加悬停分。v1 的 prev_stop/prev_a_mag 属性在 super 里
是重绑(不是原地改),所以进 super 前留住旧引用即可拿到上一步的值。

评估/部署的切断规则(区内 N 步 → 动作强制 0)不在 env 里,由 val 侧 wrapper 实现,
评估时用 v1 env 保持与 88%/92% 同判据。
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src import task_cfg
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg
from src.noise_env import NoiseStudentEnv, NoiseStudentEnvCfg


@configclass
class HoverStudentEnvCfg(MDPStudentEnvCfg):
    r_hover = task_cfg.R_HOVER


@configclass
class HoverNoiseStudentEnvCfg(NoiseStudentEnvCfg):
    r_hover = task_cfg.R_HOVER


class HoverMixin:
    def compute_reward(self) -> torch.Tensor:
        prev_stop = self.prev_stop      # super 会重绑这两个属性;旧 tensor 即上一步的值
        prev_a_mag = self.prev_a_mag
        reward = super().compute_reward()

        # 与 v1 同源的当前步量:last_px_x/last_dist 就是 v1 reward 里用的 x/d 快照。
        x = self.last_px_x
        d = self.last_dist
        seen = d > self.cfg.lost_d
        end_px = self.cfg.image_width * 0.5 + self.end_x
        stop = (torch.abs(d - self.end_d) <= self.cfg.stop_d_tol) & (
            torch.abs(x - end_px) <= self.cfg.stop_x_tol
        )
        stop_q = torch.exp(
            -0.5 * ((x - end_px) / self.cfg.sig_x) ** 2
            - 0.5 * ((d - self.end_d) / self.cfg.sig_d) ** 2
        )
        a_mag = torch.max(torch.abs(self.actions), dim=1).values

        # 精确撤掉 v1 的 goal 事件/终止项(顺序与 sb3_env.compute_reward 的加法一一对应)。
        zero = torch.zeros_like(reward)
        reward -= torch.where((~prev_stop) & stop, torch.full_like(reward, self.cfg.r_stop_in), zero)
        reward -= torch.where(prev_stop & (~stop), torch.full_like(reward, self.cfg.r_stop_out), zero)
        reward -= torch.where(stop & prev_stop, self.cfg.k_stop_da * (prev_a_mag - a_mag), zero)
        reward -= torch.where(stop & (a_mag < self.cfg.stop_eps), self.cfg.r_stop * stop_q, zero)
        reward -= torch.where(self.last_success, self.cfg.r_success * stop_q, zero)
        reward -= torch.where(self.last_fail & (~self.last_success), torch.full_like(reward, self.cfg.r_fail), zero)

        # v2 主项:悬停状态分。
        reward += torch.where(seen, self.cfg.r_hover * stop_q, zero)
        return reward

    def compute_terminated(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


class HoverStudentEnv(HoverMixin, MDPStudentEnv):
    cfg: HoverStudentEnvCfg


class HoverNoiseStudentEnv(HoverMixin, NoiseStudentEnv):
    cfg: HoverNoiseStudentEnvCfg


def make_hover_student_env(cfg: HoverStudentEnvCfg | None = None) -> HoverStudentEnv:
    return HoverStudentEnv(cfg or HoverStudentEnvCfg())


def make_hover_noise_student_env(cfg: HoverNoiseStudentEnvCfg | None = None) -> HoverNoiseStudentEnv:
    return HoverNoiseStudentEnv(cfg or HoverNoiseStudentEnvCfg())
