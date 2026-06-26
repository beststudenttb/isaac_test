"""Offline validate teacher update checkpoints.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p val/teacher.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import teacher_cfg as cfg


parser = argparse.ArgumentParser(description="Offline validate teacher update checkpoints.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--num-episodes", type=int, default=int(cfg.NUM_EPISODES))
parser.add_argument("--start", type=int, default=int(cfg.START))
parser.add_argument("--stride", type=int, default=int(cfg.STRIDE))
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
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sb3_env import BallPPOEnvCfg, make_sb3_env
from src.val_callback import TRAJ_FIELDS, make_traj_row


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


def run_dir() -> Path:
    return Path(cfg.OUT_DIR)


def make_env():
    env_cfg = BallPPOEnvCfg()
    env_cfg.seed = int(cfg.SEED)
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else int(cfg.NUM_ENVS)
    env_cfg.sim.device = args_cli.device
    env_cfg.use_camera = bool(cfg.USE_CAMERA)
    env_cfg.read_camera = bool(cfg.READ_CAMERA)
    env_cfg.end_d_min = float(cfg.END_D_MIN)
    env_cfg.end_d_max = float(cfg.END_D_MAX)
    env_cfg.end_x_min = float(cfg.END_X_MIN)
    env_cfg.end_x_max = float(cfg.END_X_MAX)
    return make_sb3_env(env_cfg, fast_variant=True)


def load_model(path: Path) -> tuple[PPO, int]:
    model = PPO.load(path, device=args_cli.device)
    update = int(path.stem.split("_")[-1])
    return model, update


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
    env,
    model: PPO,
    num_episodes: int,
    update: int,
    step: int,
    traj_writer: csv.DictWriter | None = None,
    env0_traj_writer: csv.DictWriter | None = None,
) -> dict:
    base_env = env.unwrapped
    steps = int(cfg.VAL_STEPS) if int(cfg.VAL_STEPS) > 0 else int(base_env.max_episode_length)
    episodes = env.num_envs * int(num_episodes)
    success_count = 0
    fail_count = 0
    timeout_count = 0
    return_sum = 0.0
    len_sum = 0.0

    for episode_id in range(int(num_episodes)):
        obs = env.reset()
        done_once = np.zeros(env.num_envs, dtype=bool)
        returns = np.zeros(env.num_envs, dtype=np.float64)
        lengths = np.zeros(env.num_envs, dtype=np.int64)
        success = np.zeros(env.num_envs, dtype=bool)
        fail = np.zeros(env.num_envs, dtype=bool)
        timeout = np.zeros(env.num_envs, dtype=bool)

        for t in range(steps):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _infos = env.step(actions)
            active = ~done_once
            returns[active] += rewards[active]
            lengths[active] += 1
            done = dones & active
            if traj_writer is not None:
                for env_id in np.flatnonzero(active):
                    traj_writer.writerow(
                        make_traj_row(
                            base_env,
                            phase="offline_val",
                            step=step,
                            rollout=update,
                            t=episode_id * steps + t,
                            env_id=int(env_id),
                            reward=rewards[int(env_id)],
                            done=bool(done[int(env_id)]),
                        )
                    )
            if env0_traj_writer is not None and bool(active[0]):
                env0_traj_writer.writerow(
                    make_traj_row(
                        base_env,
                        phase="offline_val",
                        step=step,
                        rollout=update,
                        t=episode_id * steps + t,
                        env_id=0,
                        reward=rewards[0],
                        done=bool(done[0]),
                    )
                )
            done_ids = np.flatnonzero(done)
            if len(done_ids) > 0:
                success_buf = base_env.last_success.detach().cpu().numpy()
                fail_buf = base_env.last_fail.detach().cpu().numpy()
                timeout_buf = base_env.last_timeout.detach().cpu().numpy()
                success[done_ids] = success_buf[done_ids]
                fail[done_ids] = fail_buf[done_ids]
                timeout[done_ids] = timeout_buf[done_ids]
                done_once[done_ids] = True
                if done_once.all():
                    break

        timeout[~done_once] = True
        lengths[~done_once] = steps
        success_count += int(success.sum())
        fail_count += int(fail.sum())
        timeout_count += int(timeout.sum())
        return_sum += float(returns.sum())
        len_sum += float(lengths.sum())

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


def write_config(path: Path, env):
    base_env = env.unwrapped
    lines = [
        "run_name = 'teacher_ppo'",
        f"num_envs = {env.num_envs!r}",
        f"num_episodes = {int(args_cli.num_episodes)!r}",
        f"start = {int(args_cli.start)!r}",
        f"stride = {int(args_cli.stride)!r}",
        f"episodes_per_checkpoint = {env.num_envs * int(args_cli.num_episodes)!r}",
        f"val_steps = {int(cfg.VAL_STEPS)!r}",
        f"episode_s = {float(cfg.EPISODE_S)!r}",
        f"stop_n = {int(cfg.STOP_N)!r}",
        f"seed = {int(cfg.SEED)!r}",
        f"device = {base_env.device!r}",
        f"use_camera = {bool(cfg.USE_CAMERA)!r}",
        f"read_camera = {bool(cfg.READ_CAMERA)!r}",
        f"end_d_min = {float(cfg.END_D_MIN)!r}",
        f"end_d_max = {float(cfg.END_D_MAX)!r}",
        f"end_x_min = {float(cfg.END_X_MIN)!r}",
        f"end_x_max = {float(cfg.END_X_MAX)!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = run_dir()
    update_dir = root / "updates"
    if not update_dir.exists():
        raise FileNotFoundError(f"update dir not found: {update_dir}")
    update_paths = sorted(update_dir.glob("update_*.zip"))
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
    write_config(root / "offline_val_config.py", env)
    out_csv = root / "offline_val.csv"
    env0_traj_csv = root / "traj_offline_env0.csv"
    best_row = None
    best_path = None

    print(
        f"[INFO] offline val run=teacher_ppo updates={len(update_paths)} "
        f"envs={env.num_envs} num_episodes={args_cli.num_episodes} start={args_cli.start} "
        f"stride={args_cli.stride} device={env.unwrapped.device}"
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
                model, update = load_model(path)
                step = int(model.num_timesteps)
                info = val(env, model, int(args_cli.num_episodes), update, step, env0_traj_writer=env0_writer)
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
                print_log(
                    [
                        ("model", [("checkpoint", row["checkpoint"]), ("update", row["update"]), ("step", row["step"])]),
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
        shutil.copy2(best_path, root / "best_offline.zip")
        shutil.copy2(best_path, root / "best_val.zip")
        (root / "best_offline_info.txt").write_text(
            "\n".join(f"{key} = {value}" for key, value in best_row.items()) + "\n",
            encoding="utf-8",
        )
        best_model, _best_update = load_model(best_path)
        with (root / "traj_best_offline.csv").open("w", newline="", encoding="utf-8") as traj_file:
            traj_writer = csv.DictWriter(traj_file, fieldnames=TRAJ_FIELDS)
            traj_writer.writeheader()
            val(
                env,
                best_model,
                int(args_cli.num_episodes),
                int(best_row["update"]),
                int(best_row["step"]),
                traj_writer,
            )
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
