"""Goal-conditioned 奖励重算(HER relabel 用)—— 逐行复刻 sb3_env.BallPPOEnv.compute_reward
的 goal 相关逻辑,但做成不依赖 Isaac 的纯 torch 函数,可离线单测。

HER 只改 goal(end_x/end_d),相机观测本身 goal 无关,所以 relabel 一条 transition 就是:
用该步存下来的 (px_x, dist, action) + 前一行的 (px_x, dist, action) + 新 goal,重算 reward 与
success/terminated。前提 STOP_N==1:success ⟺ 单步 in_zone & a_mag<STOP_EPS(不需跨步累积)。

fail(out of bounds)是物理量、与 goal 无关,由调用方按存储的 fail 标志传入。
"""

from __future__ import annotations

import torch

from src import task_cfg as T

CX = T.IMAGE_WIDTH * 0.5


def goal_norm(end_x: torch.Tensor, end_d: torch.Tensor) -> torch.Tensor:
    """与 sb3_env.end_obs() 一致:goal 向量 = [end_d/fail_far, end_x/(W/2)]。"""
    return torch.stack((end_d / T.FAIL_FAR, end_x / (T.IMAGE_WIDTH * 0.5)), dim=-1)


def achieved_to_end(px: torch.Tensor, dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把"机器人实际看到球的位置"反解成 goal 的 raw end_x/end_d(end_px = cx + end_x)。"""
    return px - CX, dist


def in_zone(px: torch.Tensor, dist: torch.Tensor, end_x: torch.Tensor, end_d: torch.Tensor) -> torch.Tensor:
    end_px = CX + end_x
    return (torch.abs(dist - end_d) <= T.STOP_D_TOL) & (torch.abs(px - end_px) <= T.STOP_X_TOL)


def recompute_reward(
    px: torch.Tensor, dist: torch.Tensor, action: torch.Tensor,
    prev_px: torch.Tensor, prev_dist: torch.Tensor, prev_action: torch.Tensor,
    end_x: torch.Tensor, end_d: torch.Tensor, fail: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (reward, success, terminated)。所有输入向量化 [B],action/prev_action [B,3]。

    STOP_N==1 假设写死:success = in_zone & (a_mag<STOP_EPS)。train/sac_cfg.py 的 STOP_N 必须为 1。
    """
    end_px = CX + end_x
    seen = dist > T.LOST_D
    prev_seen = prev_dist > T.LOST_D
    stop = in_zone(px, dist, end_x, end_d)
    prev_stop = in_zone(prev_px, prev_dist, end_x, end_d)
    xe = torch.abs(px - end_px)
    de = torch.abs(dist - end_d)
    prev_xe = torch.abs(prev_px - end_px)
    prev_de = torch.abs(prev_dist - end_d)
    a_mag = action.abs().max(dim=-1).values
    prev_a_mag = prev_action.abs().max(dim=-1).values

    reward = torch.zeros_like(px)
    ax, ay, aw = action[:, 0], action[:, 1], action[:, 2]
    reward = reward + torch.where(
        ~seen, -T.K_SEARCH * (ax ** 2 + ay ** 2) + T.K_SEARCH_W * aw.abs(), torch.zeros_like(reward)
    )
    reward = reward + torch.where(seen, torch.full_like(reward, -T.K_TIME), torch.zeros_like(reward))

    app = seen & prev_seen & (~stop)
    x_gain = (prev_xe - xe) / T.IMAGE_WIDTH
    d_gain = (prev_de - de) / T.FAIL_FAR
    reward = reward + torch.where(app, T.K_X * x_gain + T.K_D * d_gain, torch.zeros_like(reward))

    reward = reward + torch.where((~prev_seen) & seen, torch.full_like(reward, T.R_FIND), torch.zeros_like(reward))
    reward = reward + torch.where(prev_seen & (~seen), torch.full_like(reward, T.R_LOST), torch.zeros_like(reward))
    reward = reward + torch.where((~prev_stop) & stop, torch.full_like(reward, T.R_STOP_IN), torch.zeros_like(reward))
    reward = reward + torch.where(prev_stop & (~stop), torch.full_like(reward, T.R_STOP_OUT), torch.zeros_like(reward))
    reward = reward + torch.where(stop & prev_stop, T.K_STOP_DA * (prev_a_mag - a_mag), torch.zeros_like(reward))

    stop_q = torch.exp(
        -0.5 * ((px - end_px) / T.SIG_X) ** 2
        - 0.5 * ((dist - end_d) / T.SIG_D) ** 2
    )
    stop_action = a_mag < T.STOP_EPS
    reward = reward + torch.where(stop & stop_action, T.R_STOP * stop_q, torch.zeros_like(reward))

    success = stop & stop_action
    reward = reward + torch.where(success, T.R_SUCCESS * stop_q, torch.zeros_like(reward))
    reward = reward + torch.where(fail & (~success), torch.full_like(reward, T.R_FAIL), torch.zeros_like(reward))

    terminated = success | fail
    return reward, success, terminated
