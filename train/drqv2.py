"""Train the standalone DrQ-v2 online off-policy baseline.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p train/drqv2.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import drqv2_cfg as cfg


parser = argparse.ArgumentParser(description="Train DrQ-v2 from stacked RGB.")
parser.add_argument("--num-envs", type=int, default=None)
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
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import task_cfg
from src.drqv2_agent import DrQV2Agent
from src.drqv2_buffer import FrameStack, VectorNStepReplay, resize_rgb
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.noise_env import NoiseStudentEnvCfg, make_noise_student_env


LOG_FIELDS = [
    "tick",
    "frame",
    "fps",
    "collect_s",
    "learn_s",
    "episodes",
    "success_rate",
    "fail_rate",
    "timeout_rate",
    "critic_loss",
    "actor_loss",
    "q1",
    "target_q",
    "stddev",
    "replay_size",
    "updates",
]

TRAJ_FIELDS = [
    "frame",
    "tick",
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
]


def output_dir() -> Path:
    path = Path(cfg.OUT_DIR)
    if args_cli.random_stop:
        path = path.with_name(path.name + str(cfg.RANDOM_STOP_SUFFIX))
    if args_cli.noise:
        path = path.with_name(path.name + str(cfg.NOISE_SUFFIX))
    if bool(cfg.CLEAR_OUT_DIR) and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def agent_config(obs_shape: tuple[int, int, int]) -> dict:
    return {
        "obs_shape": obs_shape,
        "goal_dim": 2,
        "action_dim": 3,
        "lr": float(cfg.LR),
        "feature_dim": int(cfg.FEATURE_DIM),
        "hidden_dim": int(cfg.HIDDEN_DIM),
        "critic_target_tau": float(cfg.CRITIC_TARGET_TAU),
        "num_expl_steps": int(cfg.NUM_EXPL_STEPS),
        "stddev_start": float(cfg.STDDEV_START),
        "stddev_end": float(cfg.STDDEV_END),
        "stddev_duration": int(cfg.STDDEV_DURATION),
        "stddev_clip": float(cfg.STDDEV_CLIP),
        "aug_pad": int(cfg.AUG_PAD),
    }


def write_config(path: Path, env: MDPStudentEnv, obs_shape: tuple[int, int, int]) -> None:
    lines = [
        f"{name} = {getattr(cfg, name)!r}"
        for name in dir(cfg)
        if name.isupper()
    ]
    lines.extend(
        (
            f"NUM_ENVS_USED = {env.num_envs!r}",
            f"DEVICE_USED = {env.device!r}",
            f"OBS_SHAPE = {obs_shape!r}",
            f"RANDOM_STOP = {bool(args_cli.random_stop)!r}",
            f"NOISE = {bool(args_cli.noise)!r}",
        )
    )
    (path / "config.py").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_checkpoint(
    path: Path,
    agent: DrQV2Agent,
    agent_cfg: dict,
    update: int,
    frame: int,
) -> None:
    torch.save(
        {
            "algorithm": "drqv2",
            "update": int(update),
            "step": int(frame),
            "agent_cfg": agent_cfg,
            "agent": agent.checkpoint(),
        },
        path,
    )


def open_csv(path: Path, fields: list[str]):
    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    return file, writer


def main() -> None:
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = output_dir()
    update_dir = out_dir / "updates"
    update_dir.mkdir(parents=True, exist_ok=True)
    log_file, log_writer = open_csv(out_dir / "log.csv", LOG_FIELDS)
    traj_file, traj_writer = open_csv(
        out_dir / "traj_train_env0.csv",
        TRAJ_FIELDS,
    )
    tensorboard = SummaryWriter(str(out_dir / "tb"))

    env = make_env()
    device = torch.device(env.device)
    frame_stacker = FrameStack(
        env.num_envs,
        int(cfg.FRAME_STACK),
        (3, int(cfg.IMAGE_SIZE), int(cfg.IMAGE_SIZE)),
    )
    obs_shape = frame_stacker.obs_shape
    agent_cfg = agent_config(obs_shape)
    agent = DrQV2Agent(**agent_cfg, device=device)
    replay = VectorNStepReplay(
        capacity_per_env=int(cfg.CAPACITY_PER_ENV),
        num_envs=env.num_envs,
        obs_shape=obs_shape,
        goal_dim=2,
        action_dim=3,
        nstep=int(cfg.NSTEP),
        discount=float(cfg.DISCOUNT),
        min_timeout_interval=int(cfg.MIN_TIMEOUT_INTERVAL),
    )
    write_config(out_dir, env, obs_shape)

    env.reset()
    frame = resize_rgb(camera_rgb(env), int(cfg.IMAGE_SIZE))
    current_obs = frame_stacker.reset(frame)
    current_goal = env.end_obs().clone()

    global_frame = 0
    tick = 0
    start_time = time.perf_counter()
    window_collect_s = 0.0
    window_learn_s = 0.0
    window_episodes = 0
    window_success = 0
    window_fail = 0
    window_timeout = 0
    window_updates = 0
    metric_sums: dict[str, float] = {}

    print(
        f"[INFO] DrQ-v2 out={out_dir} envs={env.num_envs} "
        f"frames={cfg.TOTAL_FRAMES} obs={obs_shape} device={env.device}"
    )

    try:
        while global_frame < int(cfg.TOTAL_FRAMES):
            tick += 1
            collect_start = time.perf_counter()
            transition_goal = current_goal
            action = agent.act(
                current_obs,
                transition_goal,
                step=global_frame,
                eval_mode=False,
            )
            _, reward, terminated, truncated, _ = env.step(action)
            timeout = truncated & (~terminated)
            done = terminated | truncated

            terminal_obs = None
            terminal_goal = None
            if torch.any(timeout):
                timeout_ids = torch.nonzero(
                    timeout,
                    as_tuple=False,
                ).flatten()
                terminal_frame = resize_rgb(
                    env.terminal_rgb[timeout_ids],
                    int(cfg.IMAGE_SIZE),
                )
                terminal_obs = frame_stacker.terminal_obs(
                    timeout_ids,
                    terminal_frame,
                )
                terminal_goal = env.terminal_end_obs[timeout_ids].clone()

            replay.add(
                obs=current_obs,
                goal=transition_goal,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=timeout,
                terminal_obs=terminal_obs,
                terminal_goal=terminal_goal,
            )

            global_frame += env.num_envs
            next_frame = resize_rgb(camera_rgb(env), int(cfg.IMAGE_SIZE))
            current_obs = frame_stacker.advance(next_frame, done)
            current_goal = env.end_obs().clone()
            window_collect_s += time.perf_counter() - collect_start

            done_count = int(done.sum().item())
            if done_count:
                window_episodes += done_count
                window_success += int((env.last_success & done).sum().item())
                window_fail += int((env.last_fail & done).sum().item())
                window_timeout += int(timeout.sum().item())

            end_d0 = transition_goal[0, 0] * float(env.cfg.fail_far)
            end_x0 = transition_goal[0, 1] * float(env.cfg.image_width * 0.5)
            traj_writer.writerow(
                {
                    "frame": global_frame,
                    "tick": tick,
                    "px_x": f"{env.last_px_x[0].item():.4f}",
                    "dist": f"{env.last_dist[0].item():.4f}",
                    "end_d": f"{end_d0.item():.4f}",
                    "end_x": f"{end_x0.item():.4f}",
                    "a_x": f"{action[0, 0].item():.4f}",
                    "a_y": f"{action[0, 1].item():.4f}",
                    "a_w": f"{action[0, 2].item():.4f}",
                    "reward": f"{reward[0].item():.4f}",
                    "done": int(done[0].item()),
                    "success": int((env.last_success[0] & done[0]).item()),
                    "fail": int((env.last_fail[0] & done[0]).item()),
                    "timeout": int(timeout[0].item()),
                }
            )

            if (
                global_frame >= int(cfg.SEED_FRAMES)
                and replay.size > int(cfg.NSTEP)
            ):
                learn_start = time.perf_counter()
                for _ in range(int(cfg.UPDATES_PER_TICK)):
                    batch = replay.sample(int(cfg.BATCH_SIZE), device)
                    metrics = agent.update(batch, global_frame)
                    for key, value in metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value
                    window_updates += 1
                window_learn_s += time.perf_counter() - learn_start

            # 每 SAVE_UPDATE_EVERY 个 tick 存一次;文件名用 tick 序号(= update 号),与 SPR 的
            # full_{update:06d}.pt 一致。存的 dict 里带 step(env 帧数),val 报告用它。
            if tick % int(cfg.SAVE_UPDATE_EVERY) == 0:
                save_checkpoint(
                    update_dir / f"drqv2_{tick:06d}.pt",
                    agent,
                    agent_cfg,
                    tick,
                    global_frame,
                )

            if tick % int(cfg.LOG_EVERY_TICKS) == 0:
                elapsed = max(window_collect_s + window_learn_s, 1e-9)
                frames_in_window = int(cfg.LOG_EVERY_TICKS) * env.num_envs
                averaged = {
                    key: metric_sums.get(key, 0.0) / max(window_updates, 1)
                    for key in ("critic_loss", "actor_loss", "q1", "target_q")
                }
                row = {
                    "tick": tick,
                    "frame": global_frame,
                    "fps": f"{frames_in_window / elapsed:.1f}",
                    "collect_s": f"{window_collect_s:.3f}",
                    "learn_s": f"{window_learn_s:.3f}",
                    "episodes": window_episodes,
                    "success_rate": f"{window_success / max(window_episodes, 1):.4f}",
                    "fail_rate": f"{window_fail / max(window_episodes, 1):.4f}",
                    "timeout_rate": f"{window_timeout / max(window_episodes, 1):.4f}",
                    "critic_loss": f"{averaged['critic_loss']:.6f}",
                    "actor_loss": f"{averaged['actor_loss']:.6f}",
                    "q1": f"{averaged['q1']:.5f}",
                    "target_q": f"{averaged['target_q']:.5f}",
                    "stddev": f"{agent.stddev(global_frame):.5f}",
                    "replay_size": replay.num_transitions,
                    "updates": window_updates,
                }
                log_writer.writerow(row)
                log_file.flush()
                traj_file.flush()
                for key in ("critic_loss", "actor_loss", "q1", "target_q"):
                    tensorboard.add_scalar(f"train/{key}", averaged[key], global_frame)
                tensorboard.add_scalar(
                    "train/success_rate",
                    window_success / max(window_episodes, 1),
                    global_frame,
                )
                tensorboard.add_scalar(
                    "train/stddev",
                    agent.stddev(global_frame),
                    global_frame,
                )
                print(
                    f"[INFO] frame={global_frame} fps={row['fps']} "
                    f"replay={replay.num_transitions} updates={window_updates} "
                    f"success={row['success_rate']} std={row['stddev']}"
                )
                window_collect_s = 0.0
                window_learn_s = 0.0
                window_episodes = 0
                window_success = 0
                window_fail = 0
                window_timeout = 0
                window_updates = 0
                metric_sums.clear()

        save_checkpoint(
            out_dir / "last.pt",
            agent,
            agent_cfg,
            tick,
            global_frame,
        )
        duration = time.perf_counter() - start_time
        print(
            f"[INFO] DrQ-v2 training complete frame={global_frame} "
            f"hours={duration / 3600.0:.2f}"
        )
    finally:
        tensorboard.close()
        log_file.close()
        traj_file.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
