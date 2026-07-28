"""off-policy 训练/评估的日志与轨迹 helper(纯 torch,无 Isaac,可离线测)。

统一 arm A 族(drq_student + spr_coadapt)散在各脚本里的重复:CSV 字段、traj 行、
控制台表格、checkpoint 落盘。teacher 占位列(t_ax/t_ay/t_aw)已删——off-policy 无 teacher,
那三列恒 0 是从 PPO 抄来的死列。
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch


# spr_coadapt 的 superset:spr_z/pixels 模式不产 spr_loss/z_std 时写 0,无害。
LOG_FIELDS = [
    "tick", "step", "fps", "sample_fps", "collect_s", "learn_s",
    "done", "success", "fail", "timeout",
    "success_rate", "fail_rate", "timeout_rate",
    "critic_loss", "actor_loss", "q1_mean", "q_target_mean", "spr_loss", "z_std",
    "sigma", "a_mag_mean", "stop_frac", "zone_frac", "buffer_frames", "updates",
]

TRAJ_FIELDS = [
    "phase", "step", "update", "t", "env_id",
    "px_x", "dist", "end_d", "end_x", "a_x", "a_y", "a_w",
    "reward", "done", "success", "fail", "timeout",
    "robot_x", "robot_y", "head_yaw", "target_x", "target_y",
]

VAL_FIELDS = [
    "checkpoint", "kind", "spr_checkpoint", "update", "step",
    "num_envs", "num_episodes", "episodes",
    "success_rate", "fail_rate", "timeout_rate", "mean_return", "mean_len",
    "final_de_mean", "final_xe_mean",
]


def open_csv(path: Path, fields: list[str], append: bool = False):
    file = path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    if not append:
        writer.writeheader()
    return file, writer


def traj_row(env, phase: str, step: int, update: int, t: int, env_id: int,
             reward: torch.Tensor, done: bool) -> dict:
    action = env.last_move_actions[env_id].detach().cpu().numpy()
    robot_xy = env.last_robot_xy[env_id].detach().cpu().numpy()
    target_xy = env.last_target_xy[env_id].detach().cpu().numpy()
    return {
        "phase": phase, "step": int(step), "update": int(update), "t": int(t), "env_id": int(env_id),
        "px_x": f"{float(env.last_px_x[env_id].item()):.4f}",
        "dist": f"{float(env.last_dist[env_id].item()):.4f}",
        "end_d": f"{float(env.end_d[env_id].item()):.4f}",
        "end_x": f"{float(env.end_x[env_id].item()):.4f}",
        "a_x": f"{float(action[0]):.4f}", "a_y": f"{float(action[1]):.4f}", "a_w": f"{float(action[2]):.4f}",
        "reward": f"{float(reward.item()):.4f}", "done": int(done),
        "success": int(bool(env.last_success[env_id].item())),
        "fail": int(bool(env.last_fail[env_id].item())),
        "timeout": int(bool(env.last_timeout[env_id].item())),
        "robot_x": f"{float(robot_xy[0]):.4f}", "robot_y": f"{float(robot_xy[1]):.4f}",
        "head_yaw": f"{float(env.last_head_yaw[env_id].item()):.4f}",
        "target_x": f"{float(target_xy[0]):.4f}", "target_y": f"{float(target_xy[1]):.4f}",
    }


def fmt_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.3f}"
    return str(value)


def print_log(groups: list[tuple[str, list[tuple[str, object]]]]) -> None:
    rows = []
    for group, items in groups:
        rows.append((f"{group}/", ""))
        rows.extend((f"  {key}", fmt_value(value)) for key, value in items)
    width_l = max(len(left) for left, _ in rows)
    width_r = max(len(str(right)) for _, right in rows)
    line = "-" * (width_l + width_r + 7)
    print(line)
    for left, right in rows:
        print(f"| {left:<{width_l}} | {fmt_value(right):>{width_r}} |")
    print(line)


def save_checkpoint(path: Path, agent, agent_cfg: dict, obs_mode: str,
                    step: int, sigma: float, spr_ckpt) -> None:
    torch.save(
        {
            "agent": agent.save_dict(),
            "agent_cfg": agent_cfg,
            "obs_mode": obs_mode,
            "step": int(step),
            "sigma": float(sigma),
            "spr_ckpt": str(spr_ckpt),
        },
        path,
    )
