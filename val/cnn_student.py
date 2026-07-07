"""Offline validate small CNN visual student checkpoints.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p val/cnn_student.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cnn_student_cfg as cfg


parser = argparse.ArgumentParser(description="Offline validate small CNN visual student checkpoints.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--num-episodes", type=int, default=int(cfg.NUM_EPISODES))
parser.add_argument("--start", type=int, default=int(cfg.START))
parser.add_argument("--stride", type=int, default=int(cfg.STRIDE))
parser.add_argument("--random-stop", action="store_true")
parser.add_argument("--noise", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
args_cli.headless = True
args_cli.livestream = 0
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
import isaaclab.sim as sim_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cnn_ppo import CNNActorCritic
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.noise_env import NoiseStudentEnvCfg, make_noise_student_env


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
    "final_de_mean",
    "final_xe_mean",
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


def root_dir() -> Path:
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


def update_num(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def make_env() -> MDPStudentEnv:
    end_d_min, end_d_max, end_x_min, end_x_max = end_range()
    env_cfg = NoiseStudentEnvCfg() if args_cli.noise else MDPStudentEnvCfg()
    env_cfg.seed = int(cfg.SEED)
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else int(cfg.NUM_ENVS)
    env_cfg.sim.device = args_cli.device
    render_kwargs = {
        "rendering_mode": str(cfg.RENDERING_MODE),
        "antialiasing_mode": str(cfg.ANTIALIASING_MODE),
        "enable_dlssg": False,
    }
    if cfg.DLSS_MODE is not None:
        render_kwargs["dlss_mode"] = int(cfg.DLSS_MODE)
    env_cfg.sim.render = sim_utils.RenderCfg(**render_kwargs)
    env_cfg.num_rerenders_on_reset = int(cfg.RERENDER_ON_RESET)
    env_cfg.end_d_min = end_d_min
    env_cfg.end_d_max = end_d_max
    env_cfg.end_x_min = end_x_min
    env_cfg.end_x_max = end_x_max
    make_env_fn = make_noise_student_env if args_cli.noise else make_mdp_student_env
    return make_env_fn(env_cfg)


def camera_rgb(env: MDPStudentEnv) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def build_model(ckpt: dict, device: torch.device) -> CNNActorCritic:
    model_cfg = ckpt["model_cfg"]
    model = CNNActorCritic(
        z_dim=int(model_cfg["z_dim"]),
        goal_dim=int(model_cfg["goal_dim"]),
        act_dim=int(model_cfg["act_dim"]),
        pi_hidden=list(model_cfg["pi_hidden"]),
        vf_hidden=list(model_cfg["vf_hidden"]),
        activation=ACTIVATIONS[str(model_cfg["activation"])],
        init_std=float(model_cfg["init_std"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def traj_row(
    env: MDPStudentEnv,
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
    return {
        "phase": phase,
        "step": int(step),
        "update": int(update),
        "t": int(t),
        "env_id": int(env_id),
        "px_x": f"{float(env.last_px_x[env_id].item()):.4f}",
        "dist": f"{float(env.last_dist[env_id].item()):.4f}",
        "end_d": f"{float(env.end_d[env_id].item()):.4f}",
        "end_x": f"{float(env.end_x[env_id].item()):.4f}",
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


def val(
    env: MDPStudentEnv,
    model: CNNActorCritic,
    num_episodes: int,
    update: int,
    step: int,
    traj_writer,
    env0_traj_writer: csv.DictWriter | None = None,
) -> dict[str, float]:
    steps = int(cfg.VAL_STEPS) if int(cfg.VAL_STEPS) > 0 else int(env.max_episode_length)
    episodes = env.num_envs * int(num_episodes)
    success_count = 0
    fail_count = 0
    timeout_count = 0
    return_sum = 0.0
    len_sum = 0.0
    de_sum = 0.0
    xe_sum = 0.0
    cx = float(env.cfg.image_width) * 0.5

    with torch.no_grad():
        for episode_id in range(int(num_episodes)):
            env.reset()
            image = camera_rgb(env)
            goal = env.end_obs()
            done_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            returns = torch.zeros(env.num_envs, device=env.device)
            lengths = torch.zeros(env.num_envs, device=env.device)
            success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            fail = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            timeout = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            final_de = torch.zeros(env.num_envs, device=env.device)
            final_xe = torch.zeros(env.num_envs, device=env.device)

            for t in range(steps):
                action = model.predict(image, goal)
                _obs, reward, terminated, truncated, _info = env.step(action)
                active = ~done_once
                returns[active] += reward[active]
                lengths[active] += 1
                done = (terminated | truncated) & active
                row_t = episode_id * steps + t
                if traj_writer is not None:
                    for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
                        traj_writer.writerow(
                            traj_row(env, "offline_val", step, update, row_t, int(env_id), reward[int(env_id)], bool(done[int(env_id)]))
                        )
                if env0_traj_writer is not None and bool(active[0].item()):
                    env0_traj_writer.writerow(
                        traj_row(env, "offline_val_env0", step, update, row_t, 0, reward[0], bool(done[0]))
                    )
                if torch.any(done):
                    success[done] = env.last_success[done]
                    fail[done] = env.last_fail[done]
                    timeout[done] = env.last_timeout[done] | truncated[done]
                    final_de[done] = (env.last_dist - env.end_d).abs()[done]
                    final_xe[done] = (env.last_px_x - (cx + env.end_x)).abs()[done]
                    done_once[done] = True
                    if bool(done_once.all()):
                        break
                image = camera_rgb(env)
                goal = env.end_obs()

            rem = ~done_once
            if torch.any(rem):
                final_de[rem] = (env.last_dist - env.end_d).abs()[rem]
                final_xe[rem] = (env.last_px_x - (cx + env.end_x)).abs()[rem]
            timeout[~done_once] = True
            lengths[~done_once] = steps
            success_count += int(success.sum().item())
            fail_count += int(fail.sum().item())
            timeout_count += int(timeout.sum().item())
            return_sum += float(returns.sum().cpu())
            len_sum += float(lengths.sum().cpu())
            de_sum += float(final_de.sum().cpu())
            xe_sum += float(final_xe.sum().cpu())

    return {
        "success_rate": success_count / episodes,
        "fail_rate": fail_count / episodes,
        "timeout_rate": timeout_count / episodes,
        "mean_return": return_sum / episodes,
        "mean_len": len_sum / episodes,
        "final_de_mean": de_sum / episodes,
        "final_xe_mean": xe_sum / episodes,
    }


def best_key(row: dict) -> tuple:
    return (
        float(row["success_rate"]),
        -float(row["fail_rate"]),
        -float(row["timeout_rate"]),
        float(row["mean_return"]),
    )


def write_config(path: Path, env: MDPStudentEnv):
    lines = [
        "run_name = 'cnn_student'",
        f"num_envs = {env.num_envs!r}",
        f"num_episodes = {int(args_cli.num_episodes)!r}",
        f"start = {int(args_cli.start)!r}",
        f"stride = {int(args_cli.stride)!r}",
        f"random_stop = {bool(args_cli.random_stop)!r}",
        f"episodes_per_checkpoint = {env.num_envs * int(args_cli.num_episodes)!r}",
        f"val_steps = {int(cfg.VAL_STEPS)!r}",
        f"episode_s = {float(cfg.EPISODE_S)!r}",
        f"stop_n = {int(cfg.STOP_N)!r}",
        f"seed = {int(cfg.SEED)!r}",
        f"device = {env.device!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    width_r = max(len(str(right)) for _, right in rows)
    line = "-" * (width_l + width_r + 7)
    print(line)
    for left, right in rows:
        print(f"| {left:<{width_l}} | {fmt_value(right):>{width_r}} |")
    print(line)


def main():
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = root_dir()
    update_dir = root / "updates"
    if not update_dir.exists():
        raise FileNotFoundError(f"update dir not found: {update_dir}")
    policy_paths = sorted(update_dir.glob("ppo_*.pt"), key=update_num)
    if not policy_paths:
        raise FileNotFoundError(f"no ppo_* checkpoints found: {update_dir}")
    start = int(args_cli.start)
    stride = int(args_cli.stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive: {stride}")
    policy_paths = [path for path in policy_paths if update_num(path) >= start and (update_num(path) - start) % stride == 0]
    if not policy_paths:
        raise FileNotFoundError(f"no checkpoints found after start={start}, stride={stride}: {update_dir}")

    env = make_env()
    device = torch.device(env.device)
    write_config(root / "offline_val_config.py", env)
    out_csv = root / "offline_val.csv"
    env0_traj_csv = root / "traj_offline_env0.csv"
    best_row = None
    best_path = None

    print(
        f"[INFO] offline val run=cnn_student checkpoints={len(policy_paths)} "
        f"start={start} stride={stride} envs={env.num_envs} episodes={args_cli.num_episodes} device={env.device}"
    )
    try:
        with out_csv.open("w", newline="", encoding="utf-8") as file, env0_traj_csv.open("w", newline="", encoding="utf-8") as env0_file:
            writer = csv.DictWriter(file, fieldnames=VAL_FIELDS)
            env0_writer = csv.DictWriter(env0_file, fieldnames=TRAJ_FIELDS)
            writer.writeheader()
            env0_writer.writeheader()

            for policy_path in policy_paths:
                ckpt = torch.load(policy_path, map_location=device)
                model = build_model(ckpt, device)
                update = int(ckpt.get("update", update_num(policy_path)))
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
                    "checkpoint": policy_path.name,
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
                    best_path = policy_path
                    shutil.copy2(best_path, root / "best_offline.pt")
                    (root / "best_offline_info.txt").write_text(
                        "\n".join(f"{key} = {value}" for key, value in best_row.items()) + "\n",
                        encoding="utf-8",
                    )
                    write_traj(root / "traj_best_offline.csv", traj_buf.rows)
                print_log(
                    [
                        ("model", [("checkpoint", row["checkpoint"]), ("update", row["update"])]),
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
