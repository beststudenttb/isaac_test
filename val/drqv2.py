"""Offline validation and best-checkpoint selection for DrQ-v2."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import drqv2_cfg as cfg


parser = argparse.ArgumentParser(description="Offline validate DrQ-v2 checkpoints.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--num-episodes", type=int, default=int(cfg.NUM_EPISODES))
parser.add_argument("--start", type=int, default=int(cfg.START))
parser.add_argument("--stride", type=int, default=int(cfg.STRIDE))
parser.add_argument("--random-stop", action="store_true")
parser.add_argument("--noise", action="store_true")
parser.add_argument("--show", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
if not args_cli.show:
    args_cli.headless = True
    args_cli.livestream = 0
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.drqv2_agent import DrQV2Agent
from src.drqv2_buffer import FrameStack, resize_rgb
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.noise_env import NoiseStudentEnvCfg, make_noise_student_env


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
    "checkpoint",
    "frame",
    "episode",
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


class TrajBuffer:
    def __init__(self):
        self.rows: list[dict] = []

    def writerow(self, row: dict) -> None:
        self.rows.append(row)


def root_dir() -> Path:
    path = Path(cfg.OUT_DIR)
    if args_cli.random_stop:
        path = path.with_name(path.name + str(cfg.RANDOM_STOP_SUFFIX))
    if args_cli.noise:
        path = path.with_name(path.name + str(cfg.NOISE_SUFFIX))
    return path


def checkpoint_update(path: Path) -> int:
    return int(path.stem.split("_")[-1])


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


def make_env() -> MDPStudentEnv:
    end_d_min, end_d_max, end_x_min, end_x_max = end_range()
    env_cfg = NoiseStudentEnvCfg() if args_cli.noise else MDPStudentEnvCfg()
    env_cfg.seed = int(cfg.SEED)
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = (
        int(args_cli.num_envs)
        if args_cli.num_envs is not None
        else int(cfg.NUM_ENVS)
    )
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
    factory = make_noise_student_env if args_cli.noise else make_mdp_student_env
    return factory(env_cfg)


def camera_rgb(env: MDPStudentEnv) -> torch.Tensor:
    return env.camera.data.output["rgb"][..., :3]


def read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("algorithm") != "drqv2":
        raise ValueError(f"not a DrQ-v2 checkpoint: {path}")
    return checkpoint


def validate_agent_cfg(agent_cfg: dict) -> None:
    expected_shape = (
        3 * int(cfg.FRAME_STACK),
        int(cfg.IMAGE_SIZE),
        int(cfg.IMAGE_SIZE),
    )
    if tuple(agent_cfg["obs_shape"]) != expected_shape:
        raise ValueError(
            f"checkpoint obs_shape={agent_cfg['obs_shape']} != expected {expected_shape}"
        )


def traj_row(
    env: MDPStudentEnv,
    phase: str,
    checkpoint: str,
    frame: int,
    episode: int,
    step: int,
    env_id: int,
    goal: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> dict:
    end_d = goal[env_id, 0] * float(env.cfg.fail_far)
    end_x = goal[env_id, 1] * float(env.cfg.image_width * 0.5)
    robot_xy = env.last_robot_xy[env_id]
    target_xy = env.last_target_xy[env_id]
    return {
        "phase": phase,
        "checkpoint": checkpoint,
        "frame": int(frame),
        "episode": int(episode),
        "t": int(step),
        "env_id": int(env_id),
        "px_x": f"{env.last_px_x[env_id].item():.4f}",
        "dist": f"{env.last_dist[env_id].item():.4f}",
        "end_d": f"{end_d.item():.4f}",
        "end_x": f"{end_x.item():.4f}",
        "a_x": f"{action[env_id, 0].item():.4f}",
        "a_y": f"{action[env_id, 1].item():.4f}",
        "a_w": f"{action[env_id, 2].item():.4f}",
        "reward": f"{reward[env_id].item():.4f}",
        "done": int(done[env_id].item()),
        "success": int((env.last_success[env_id] & done[env_id]).item()),
        "fail": int((env.last_fail[env_id] & done[env_id]).item()),
        "timeout": int((env.last_timeout[env_id] & done[env_id]).item()),
        "robot_x": f"{robot_xy[0].item():.4f}",
        "robot_y": f"{robot_xy[1].item():.4f}",
        "head_yaw": f"{env.last_head_yaw[env_id].item():.4f}",
        "target_x": f"{target_xy[0].item():.4f}",
        "target_y": f"{target_xy[1].item():.4f}",
    }


def evaluate(
    env: MDPStudentEnv,
    agent: DrQV2Agent,
    checkpoint_name: str,
    frame: int,
    num_episodes: int,
    traj_writer,
    env0_traj_writer: csv.DictWriter,
) -> dict[str, float]:
    steps = (
        int(cfg.VAL_STEPS)
        if int(cfg.VAL_STEPS) > 0
        else int(env.max_episode_length)
    )
    episodes = env.num_envs * int(num_episodes)
    success_count = 0
    fail_count = 0
    timeout_count = 0
    return_sum = 0.0
    len_sum = 0.0
    de_sum = 0.0
    xe_sum = 0.0

    with torch.no_grad():
        for episode_id in range(int(num_episodes)):
            env.reset()
            frame_stacker = FrameStack(
                env.num_envs,
                int(cfg.FRAME_STACK),
                (3, int(cfg.IMAGE_SIZE), int(cfg.IMAGE_SIZE)),
            )
            obs = frame_stacker.reset(
                resize_rgb(camera_rgb(env), int(cfg.IMAGE_SIZE))
            )
            goal = env.end_obs().clone()
            done_once = torch.zeros(
                env.num_envs,
                dtype=torch.bool,
                device=env.device,
            )
            returns = torch.zeros(env.num_envs, device=env.device)
            lengths = torch.zeros(env.num_envs, device=env.device)
            success = torch.zeros_like(done_once)
            fail = torch.zeros_like(done_once)
            timeout = torch.zeros_like(done_once)
            final_de = torch.zeros(env.num_envs, device=env.device)
            final_xe = torch.zeros(env.num_envs, device=env.device)

            for step in range(steps):
                transition_goal = goal
                action = agent.act(
                    obs,
                    transition_goal,
                    step=frame,
                    eval_mode=True,
                )
                action[done_once] = 0.0
                _, reward, terminated, truncated, _ = env.step(action)
                active = ~done_once
                returns[active] += reward[active]
                lengths[active] += 1
                done = (terminated | truncated) & active

                for env_id in torch.nonzero(
                    active,
                    as_tuple=False,
                ).flatten().tolist():
                    row = traj_row(
                        env,
                        "offline_val",
                        checkpoint_name,
                        frame,
                        episode_id,
                        step + 1,
                        int(env_id),
                        transition_goal,
                        action,
                        reward,
                        done,
                    )
                    traj_writer.writerow(row)
                    if int(env_id) == 0:
                        env0_row = dict(row)
                        env0_row["phase"] = "offline_val_env0"
                        env0_traj_writer.writerow(env0_row)

                if torch.any(done):
                    success[done] = env.last_success[done]
                    fail[done] = env.last_fail[done]
                    timeout[done] = env.last_timeout[done] | (
                        truncated[done] & (~terminated[done])
                    )
                    end_d = transition_goal[:, 0] * float(env.cfg.fail_far)
                    end_x = transition_goal[:, 1] * float(
                        env.cfg.image_width * 0.5
                    )
                    final_de[done] = torch.abs(env.last_dist - end_d)[done]
                    end_px = float(env.cfg.image_width * 0.5) + end_x
                    final_xe[done] = torch.abs(env.last_px_x - end_px)[done]
                    done_once[done] = True
                    if bool(done_once.all().item()):
                        break

                next_frame = resize_rgb(camera_rgb(env), int(cfg.IMAGE_SIZE))
                obs = frame_stacker.advance(
                    next_frame,
                    terminated | truncated,
                )
                goal = env.end_obs().clone()

            remaining = ~done_once
            if torch.any(remaining):
                end_d = goal[:, 0] * float(env.cfg.fail_far)
                end_x = goal[:, 1] * float(env.cfg.image_width * 0.5)
                final_de[remaining] = torch.abs(env.last_dist - end_d)[remaining]
                end_px = float(env.cfg.image_width * 0.5) + end_x
                final_xe[remaining] = torch.abs(
                    env.last_px_x - end_px
                )[remaining]
                timeout[remaining] = True
                lengths[remaining] = steps

            success_count += int(success.sum().item())
            fail_count += int(fail.sum().item())
            timeout_count += int(timeout.sum().item())
            return_sum += returns.sum().item()
            len_sum += lengths.sum().item()
            de_sum += final_de.sum().item()
            xe_sum += final_xe.sum().item()

    return {
        "success_rate": success_count / episodes,
        "fail_rate": fail_count / episodes,
        "timeout_rate": timeout_count / episodes,
        "mean_return": return_sum / episodes,
        "mean_len": len_sum / episodes,
        "final_de_mean": de_sum / episodes,
        "final_xe_mean": xe_sum / episodes,
    }


def best_key(row: dict) -> tuple[float, float, float, float]:
    return (
        float(row["success_rate"]),
        -float(row["fail_rate"]),
        -float(row["timeout_rate"]),
        float(row["mean_return"]),
    )


def write_config(path: Path, env: MDPStudentEnv) -> None:
    lines = [
        f"{name} = {getattr(cfg, name)!r}"
        for name in dir(cfg)
        if name.isupper()
    ]
    lines.extend(
        (
            f"NUM_ENVS_USED = {env.num_envs!r}",
            f"DEVICE_USED = {env.device!r}",
            f"START_USED = {int(args_cli.start)!r}",
            f"STRIDE_USED = {int(args_cli.stride)!r}",
            f"RANDOM_STOP = {bool(args_cli.random_stop)!r}",
            f"NOISE = {bool(args_cli.noise)!r}",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_traj(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAJ_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = root_dir()
    update_dir = root / "updates"
    if not update_dir.is_dir():
        raise FileNotFoundError(f"update dir not found: {update_dir}")

    checkpoint_paths = sorted(
        update_dir.glob("drqv2_*.pt"),
        key=checkpoint_update,
    )
    start = int(args_cli.start)
    stride = int(args_cli.stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive: {stride}")
    checkpoint_paths = [
        path for path in checkpoint_paths if checkpoint_update(path) >= start
    ][::stride]
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"no DrQ-v2 checkpoints after start={start}, stride={stride}: "
            f"{update_dir}"
        )

    env = make_env()
    device = torch.device(env.device)
    write_config(root / "offline_val_config.py", env)
    out_csv = root / "offline_val.csv"
    env0_traj_csv = root / "traj_offline_env0.csv"
    best_row: dict | None = None
    best_path: Path | None = None
    agent: DrQV2Agent | None = None
    base_agent_cfg: dict | None = None

    print(
        f"[INFO] offline val run=drqv2 checkpoints={len(checkpoint_paths)} "
        f"start={start} stride={stride} envs={env.num_envs} "
        f"episodes={args_cli.num_episodes} device={env.device}"
    )

    try:
        with out_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file, env0_traj_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as env0_file:
            writer = csv.DictWriter(file, fieldnames=VAL_FIELDS)
            env0_writer = csv.DictWriter(env0_file, fieldnames=TRAJ_FIELDS)
            writer.writeheader()
            env0_writer.writeheader()

            for path in checkpoint_paths:
                checkpoint = read_checkpoint(path)
                agent_cfg = dict(checkpoint["agent_cfg"])
                validate_agent_cfg(agent_cfg)
                if agent is None:
                    base_agent_cfg = agent_cfg
                    agent = DrQV2Agent(**agent_cfg, device=device)
                elif agent_cfg != base_agent_cfg:
                    raise ValueError(f"agent config changed at checkpoint: {path}")
                agent.load_checkpoint(checkpoint["agent"])
                agent.set_training(False)

                update = int(checkpoint["update"])
                step = int(checkpoint["step"])
                traj_buffer = TrajBuffer()
                info = evaluate(
                    env,
                    agent,
                    path.name,
                    step,
                    int(args_cli.num_episodes),
                    traj_buffer,
                    env0_writer,
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
                    shutil.copy2(path, root / "best_offline.pt")
                    (root / "best_offline_info.txt").write_text(
                        "\n".join(
                            f"{key} = {value}" for key, value in row.items()
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    write_traj(
                        root / "traj_best_offline.csv",
                        traj_buffer.rows,
                    )

                print(
                    f"[INFO] checkpoint={path.name} update={update} step={step} "
                    f"success={row['success_rate']:.4f} "
                    f"fail={row['fail_rate']:.4f} "
                    f"timeout={row['timeout_rate']:.4f} "
                    f"return={row['mean_return']:.4f}"
                )
    finally:
        env.close()

    assert best_row is not None and best_path is not None
    print(
        f"[INFO] best={best_path.name} update={best_row['update']} step={best_row['step']} "
        f"success={best_row['success_rate']:.4f} "
        f"fail={best_row['fail_rate']:.4f} "
        f"timeout={best_row['timeout_rate']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
