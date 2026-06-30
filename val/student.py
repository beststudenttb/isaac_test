"""Offline validate privileged student update checkpoints.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p val/student.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import student_cfg as cfg


parser = argparse.ArgumentParser(description="Offline validate privileged student update checkpoints.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--num-episodes", type=int, default=int(cfg.NUM_EPISODES))
parser.add_argument("--start", type=int, default=int(cfg.START))
parser.add_argument("--stride", type=int, default=int(cfg.STRIDE))
parser.add_argument("--random-stop", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
args_cli.headless = True
args_cli.livestream = 0
args_cli.enable_cameras = bool(cfg.USE_CAMERA)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ppo import ActorCritic
from src.sb3_env import BallPPOEnv, BallPPOEnvCfg


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

VAL_FIELDS = [
    "checkpoint",
    "update",
    "step",
    "num_envs",
    "num_episodes",
    "episodes",
    "success_rate",
    "fail_rate",
    "timeout_rate",
    "mean_return",
    "mean_len",
]

TRAJ_FIELDS = [
    "phase",
    "step",
    "update",
    "t",
    "env_id",
    "px_x",
    "dist",
    "end_d",
    "end_x",
    "a_x",
    "a_y",
    "a_w",
    "reward",
    "done",
    "success",
    "fail",
    "timeout",
    "robot_x",
    "robot_y",
    "head_yaw",
    "target_x",
    "target_y",
]


def run_dir() -> Path:
    return Path(cfg.RANDOM_STOP_OUT_DIR) if args_cli.random_stop else Path(cfg.OUT_DIR)


def end_range() -> tuple[float, float, float, float]:
    if args_cli.random_stop:
        return (
            float(cfg.RANDOM_END_D_MIN),
            float(cfg.RANDOM_END_D_MAX),
            float(cfg.RANDOM_END_X_MIN),
            float(cfg.RANDOM_END_X_MAX),
        )
    return (
        float(cfg.END_D_MIN),
        float(cfg.END_D_MAX),
        float(cfg.END_X_MIN),
        float(cfg.END_X_MAX),
    )


def make_env() -> BallPPOEnv:
    end_d_min, end_d_max, end_x_min, end_x_max = end_range()
    env_cfg = BallPPOEnvCfg()
    env_cfg.seed = int(cfg.SEED)
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else int(cfg.NUM_ENVS)
    env_cfg.sim.device = args_cli.device
    env_cfg.use_camera = bool(cfg.USE_CAMERA)
    env_cfg.read_camera = bool(cfg.READ_CAMERA)
    env_cfg.end_d_min = end_d_min
    env_cfg.end_d_max = end_d_max
    env_cfg.end_x_min = end_x_min
    env_cfg.end_x_max = end_x_max
    return BallPPOEnv(env_cfg)


def load_model(path: Path, device: torch.device, obs_dim: int) -> tuple[ActorCritic, dict]:
    ckpt = torch.load(path, map_location=device)
    model = ActorCritic(
        obs_dim=int(obs_dim),
        act_dim=3,
        pi_hidden=list(cfg.POLICY_NET),
        vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)],
        init_std=float(cfg.STD_INIT),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def traj_row(
    env: BallPPOEnv,
    phase: str,
    step: int,
    update: int,
    t: int,
    env_id: int,
    reward: torch.Tensor,
    done: bool,
) -> dict:
    action = env.last_move_actions[env_id].detach().cpu().numpy()
    robot_xy = env.last_robot_xy[env_id].detach().cpu().numpy()
    target_xy = env.last_target_xy[env_id].detach().cpu().numpy()
    end_d = env.end_d[env_id].detach().cpu().item()
    end_x = env.end_x[env_id].detach().cpu().item()
    return {
        "phase": phase,
        "step": int(step),
        "update": int(update),
        "t": int(t),
        "env_id": int(env_id),
        "px_x": f"{float(env.last_px_x[env_id].item()):.4f}",
        "dist": f"{float(env.last_dist[env_id].item()):.4f}",
        "end_d": f"{float(end_d):.4f}",
        "end_x": f"{float(end_x):.4f}",
        "a_x": f"{float(action[0]):.4f}",
        "a_y": f"{float(action[1]):.4f}",
        "a_w": f"{float(action[2]):.4f}",
        "reward": f"{float(reward.item()):.4f}",
        "done": int(done),
        "success": int(bool(env.last_success[env_id].item())),
        "fail": int(bool(env.last_fail[env_id].item())),
        "timeout": int(bool(env.last_timeout[env_id].item())),
        "robot_x": f"{float(robot_xy[0]):.4f}",
        "robot_y": f"{float(robot_xy[1]):.4f}",
        "head_yaw": f"{float(env.last_head_yaw[env_id].item()):.4f}",
        "target_x": f"{float(target_xy[0]):.4f}",
        "target_y": f"{float(target_xy[1]):.4f}",
    }


def fmt_value(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.3f}"
    return str(value)


def print_log(groups: list[tuple[str, list[tuple[str, object]]]]):
    rows = []
    for group, items in groups:
        rows.append((f"{group}/", ""))
        rows.extend((f"  {key}", fmt_value(value)) for key, value in items)
    width_l = max(len(left) for left, _ in rows)
    width_r = max(len(right) for _, right in rows)
    line = "-" * (width_l + width_r + 7)
    print(line)
    for left, right in rows:
        print(f"| {left:<{width_l}} | {right:>{width_r}} |")
    print(line)


def val(
    env: BallPPOEnv,
    model: ActorCritic,
    num_episodes: int,
    update: int = -1,
    step: int = -1,
    traj_writer: csv.DictWriter | None = None,
    env0_traj_writer: csv.DictWriter | None = None,
) -> dict:
    steps = int(cfg.VAL_STEPS) if int(cfg.VAL_STEPS) > 0 else int(env.max_episode_length)
    episodes = env.num_envs * int(num_episodes)
    success_count = 0
    fail_count = 0
    timeout_count = 0
    return_sum = 0.0
    len_sum = 0.0

    model.eval()
    with torch.no_grad():
        for episode_id in range(int(num_episodes)):
            obs, _ = env.reset()
            obs_t = obs["policy"]
            done_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            returns = torch.zeros(env.num_envs, device=env.device)
            lengths = torch.zeros(env.num_envs, device=env.device)
            success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            fail = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            timeout = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

            for t in range(steps):
                action = model.predict(obs_t)
                obs, reward, terminated, truncated, _info = env.step(action)
                active = ~done_once
                returns[active] += reward[active]
                lengths[active] += 1
                done = (terminated | truncated) & active
                if traj_writer is not None:
                    for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
                        traj_writer.writerow(
                            traj_row(
                                env,
                                "offline_val",
                                step,
                                update,
                                episode_id * steps + t,
                                int(env_id),
                                reward[int(env_id)],
                                bool(done[int(env_id)]),
                            )
                        )
                if env0_traj_writer is not None and bool(active[0].item()):
                    env0_traj_writer.writerow(
                        traj_row(
                            env,
                            "offline_val_env0",
                            step,
                            update,
                            episode_id * steps + t,
                            0,
                            reward[0],
                            bool(done[0]),
                        )
                    )
                if torch.any(done):
                    success[done] = env.last_success[done]
                    fail[done] = env.last_fail[done]
                    timeout[done] = env.last_timeout[done] | truncated[done]
                    done_once[done] = True
                    if bool(done_once.all()):
                        break
                obs_t = obs["policy"]

            timeout[~done_once] = True
            lengths[~done_once] = steps
            success_count += int(success.sum().item())
            fail_count += int(fail.sum().item())
            timeout_count += int(timeout.sum().item())
            return_sum += float(returns.sum().cpu())
            len_sum += float(lengths.sum().cpu())

    return {
        "success_rate": success_count / episodes,
        "fail_rate": fail_count / episodes,
        "timeout_rate": timeout_count / episodes,
        "mean_return": return_sum / episodes,
        "mean_len": len_sum / episodes,
    }


def best_key(row: dict) -> tuple:
    return (
        float(row["success_rate"]),
        -float(row["fail_rate"]),
        -float(row["timeout_rate"]),
        float(row["mean_return"]),
    )


def write_config(path: Path, env: BallPPOEnv):
    end_d_min, end_d_max, end_x_min, end_x_max = end_range()
    lines = [
        f"run_name = 'student_ppo'",
        f"random_stop = {bool(args_cli.random_stop)!r}",
        f"num_envs = {env.num_envs!r}",
        f"num_episodes = {int(args_cli.num_episodes)!r}",
        f"start = {int(args_cli.start)!r}",
        f"stride = {int(args_cli.stride)!r}",
        f"episodes_per_checkpoint = {env.num_envs * int(args_cli.num_episodes)!r}",
        f"val_steps = {int(cfg.VAL_STEPS)!r}",
        f"episode_s = {float(cfg.EPISODE_S)!r}",
        f"stop_n = {int(cfg.STOP_N)!r}",
        f"seed = {int(cfg.SEED)!r}",
        f"device = {env.device!r}",
        f"use_camera = {bool(cfg.USE_CAMERA)!r}",
        f"read_camera = {bool(cfg.READ_CAMERA)!r}",
        f"end_d_min = {end_d_min!r}",
        f"end_d_max = {end_d_max!r}",
        f"end_x_min = {end_x_min!r}",
        f"end_x_max = {end_x_max!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TrajBuffer:
    def __init__(self):
        self.rows = []

    def writerow(self, row: dict):
        self.rows.append(dict(row))


def write_traj(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAJ_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = run_dir()
    update_dir = root / "updates"
    if not update_dir.exists():
        raise FileNotFoundError(f"update dir not found: {update_dir}")
    update_paths = sorted(update_dir.glob("update_*.pt"))
    if not update_paths:
        raise FileNotFoundError(f"no update checkpoints found: {update_dir}")
    start = int(args_cli.start)
    stride = int(args_cli.stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive: {stride}")
    update_paths = [
        path for path in update_paths
        if int(path.stem.split("_")[-1]) >= start
        and (int(path.stem.split("_")[-1]) - start) % stride == 0
    ]
    if not update_paths:
        raise FileNotFoundError(f"no update checkpoints found after start={start}, stride={stride}: {update_dir}")

    env = make_env()
    device = torch.device(env.device)
    write_config(root / "offline_val_config.py", env)
    out_csv = root / "offline_val.csv"
    env0_traj_csv = root / "traj_offline_env0.csv"
    best_row = None
    best_path = None

    print(
        f"[INFO] offline val run=student_ppo updates={len(update_paths)} "
        f"envs={env.num_envs} num_episodes={args_cli.num_episodes} start={args_cli.start} "
        f"stride={args_cli.stride} random_stop={args_cli.random_stop} device={env.device}"
    )
    try:
        mode = "a" if int(args_cli.start) > 0 else "w"
        with out_csv.open(mode, newline="", encoding="utf-8") as file, env0_traj_csv.open(mode, newline="", encoding="utf-8") as env0_file:
            writer = csv.DictWriter(file, fieldnames=VAL_FIELDS)
            env0_writer = csv.DictWriter(env0_file, fieldnames=TRAJ_FIELDS)
            if mode == "w":
                writer.writeheader()
                env0_writer.writeheader()
            for path in update_paths:
                model, ckpt = load_model(path, device, int(env.cfg.observation_space))
                update = int(ckpt.get("update", -1))
                step = int(ckpt.get("step", -1))
                traj_buf = TrajBuffer()
                info = val(
                    env,
                    model,
                    int(args_cli.num_episodes),
                    update,
                    step,
                    traj_buf,
                    env0_traj_writer=env0_writer,
                )
                row = {
                    "checkpoint": path.name,
                    "update": update,
                    "step": step,
                    "num_envs": env.num_envs,
                    "num_episodes": int(args_cli.num_episodes),
                    "episodes": env.num_envs * int(args_cli.num_episodes),
                    **info,
                }
                writer.writerow(row)
                file.flush()
                env0_file.flush()
                if best_row is None or best_key(row) > best_key(best_row):
                    best_row = row
                    best_path = path
                    shutil.copy2(best_path, root / "best_offline.pt")
                    (root / "best_offline_info.txt").write_text(
                        "\n".join(f"{key} = {value}" for key, value in best_row.items()) + "\n",
                        encoding="utf-8",
                    )
                    write_traj(root / "traj_best_offline.csv", traj_buf.rows)
                print_log(
                    [
                        (
                            "model",
                            [
                                ("checkpoint", row["checkpoint"]),
                                ("update", row["update"]),
                                ("step", row["step"]),
                            ],
                        ),
                        (
                            "val",
                            [
                                ("episodes", row["episodes"]),
                                ("success", row["success_rate"]),
                                ("fail", row["fail_rate"]),
                                ("timeout", row["timeout_rate"]),
                                ("return", row["mean_return"]),
                                ("len", row["mean_len"]),
                            ],
                        ),
                    ]
                )

        if best_row is None or best_path is None:
            raise RuntimeError("offline val did not produce best checkpoint")
        print(
            f"[INFO] best checkpoint={best_path.name} success={best_row['success_rate']:.3f} "
            f"fail={best_row['fail_rate']:.3f} timeout={best_row['timeout_rate']:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
