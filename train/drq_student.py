"""Train DrQ-v2 style off-policy student (frozen SPR z or pixels).

Run from the project root:

    ./IsaacLab/isaaclab.sh -p train/drq_student.py --obs-mode spr_z
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

import drq_student_cfg as cfg


parser = argparse.ArgumentParser(description="Train DrQ-v2 off-policy student.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--obs-mode", type=str, default=None, choices=["spr_z", "pixels"])
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

import numpy as np
import torch
import isaaclab.sim as sim_utils
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import task_cfg
from src.cv_extractor.spr_state import SPRStateNet
from src.drqv2 import DrQV2Agent, linear_schedule
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.noise_env import NoiseStudentEnvCfg, make_noise_student_env
from src.replay import ReplayBuffer


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
    "t_ax",
    "t_ay",
    "t_aw",
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

LOG_FIELDS = [
    "tick",
    "step",
    "fps",
    "sample_fps",
    "collect_s",
    "learn_s",
    "done",
    "success",
    "fail",
    "timeout",
    "success_rate",
    "fail_rate",
    "timeout_rate",
    "critic_loss",
    "actor_loss",
    "q1_mean",
    "q_target_mean",
    "sigma",
    "a_mag_mean",
    "stop_frac",
    "zone_frac",
    "buffer_frames",
    "updates",
]


def obs_mode() -> str:
    return str(args_cli.obs_mode) if args_cli.obs_mode is not None else str(cfg.OBS_MODE)


def out_dir() -> Path:
    base = Path(cfg.OUT_DIR) if obs_mode() == "spr_z" else Path(cfg.PIXELS_OUT_DIR)
    if args_cli.random_stop:
        base = base.with_name(base.name + str(cfg.RANDOM_STOP_SUFFIX))
    if bool(cfg.CLEAR_OUT_DIR) and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


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


def write_config(path: Path) -> None:
    lines = []
    for name in dir(cfg):
        if name.isupper():
            lines.append(f"{name} = {getattr(cfg, name)!r}")
    lines.append(f"OBS_MODE_USED = {obs_mode()!r}")
    lines.append(f"RANDOM_STOP = {bool(args_cli.random_stop)!r}")
    lines.append(f"NOISE = {bool(args_cli.noise)!r}")
    lines.append(f"DEVICE_USED = {args_cli.device!r}")
    (path / "config.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def open_csv(path: Path, fields: list[str]):
    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    return file, writer


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


def load_spr(device: torch.device) -> SPRStateNet:
    path = Path(cfg.SPR_CKPT)
    if not path.exists():
        raise FileNotFoundError(f"SPR checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    spr = SPRStateNet(**ckpt["spr_cfg"]).to(device)
    spr.load_state_dict({k[len("spr."):]: v for k, v in ckpt["model"].items() if k.startswith("spr.")})
    spr.eval()  # 冻结用:BN running stats 不再更新(07-10 已实证 train() 会漂移)。
    spr.requires_grad_(False)
    return spr


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
        "t_ax": "0.0000",
        "t_ay": "0.0000",
        "t_aw": "0.0000",
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


def print_log(groups: list[tuple[str, list[tuple[str, object]]]]) -> None:
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


def agent_cfg_dict(repr_dim: int) -> dict:
    return {
        "obs_mode": obs_mode(),
        "repr_dim": int(repr_dim),
        "goal_dim": 2,
        "act_dim": 3,
        "feature_dim": int(cfg.FEATURE_DIM),
        "hidden_dim": int(cfg.HIDDEN_DIM),
        "lr": float(cfg.LR),
        "tau": float(cfg.TAU),
        "noise_clip": float(cfg.NOISE_CLIP),
        "pixels_res": int(cfg.PIXELS_RES),
        "aug_pad": int(cfg.AUG_PAD),
    }


def save_checkpoint(path: Path, agent: DrQV2Agent, agent_cfg: dict, step: int, sigma: float) -> None:
    torch.save(
        {
            "agent": agent.save_dict(),
            "agent_cfg": agent_cfg,
            "obs_mode": obs_mode(),
            "step": int(step),
            "sigma": float(sigma),
            "spr_ckpt": str(cfg.SPR_CKPT),
        },
        path,
    )


def main() -> None:
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mode = obs_mode()
    log_dir = out_dir()
    write_config(log_dir)
    log_file, log_writer = open_csv(log_dir / "log.csv", LOG_FIELDS)
    traj_file, traj_writer = open_csv(log_dir / "traj_train_env0.csv", TRAJ_FIELDS)
    update_dir = log_dir / "updates"
    update_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir / "tb"))

    env = make_env()
    device = torch.device(env.device)

    spr = None
    z_dim = 0
    if mode == "spr_z":
        spr = load_spr(device)
        z_dim = int(spr.z_dim)
        shutil.copy2(Path(cfg.SPR_CKPT), log_dir / "spr_encoder.pt")  # run 目录自包含,val/归档直接配对。

    agent_cfg = agent_cfg_dict(z_dim)
    agent = DrQV2Agent(**agent_cfg, device=device)
    image_shape = (int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3)
    buffer = ReplayBuffer(
        capacity_per_env=int(cfg.CAPACITY_PER_ENV),
        num_envs=env.num_envs,
        image_shape=image_shape,
        goal_dim=2,
        act_dim=3,
        z_dim=z_dim,
        n_step=int(cfg.N_STEP),
        gamma=float(cfg.GAMMA),
        store_images=(mode == "pixels"),  # spr_z 只用 z,不存图 → 128 env 才放得下。
    )

    def encode_z(images: torch.Tensor) -> torch.Tensor | None:
        if spr is None:
            return None
        with torch.no_grad():
            return spr.encode(images).float()

    env.reset()
    image_t = camera_rgb(env)
    goal_t = env.end_obs()
    z_t = encode_z(image_t)

    total_steps = int(cfg.TOTAL_ENV_STEPS)
    total_ticks = total_steps // env.num_envs
    print(
        f"[INFO] drq student out={log_dir} mode={mode} envs={env.num_envs} total_steps={total_steps} "
        f"ticks={total_ticks} capacity={cfg.CAPACITY_PER_ENV}/env z_dim={z_dim} device={env.device}"
    )

    global_step = 0
    next_save = int(cfg.SAVE_EVERY_STEPS)
    win = {k: 0.0 for k in ("done", "success", "fail", "timeout", "a_mag", "stop", "zone", "collect_s", "learn_s", "updates")}
    metric_sums: dict[str, float] = {}
    metric_count = 0
    start_time = time.perf_counter()

    try:
        for tick in range(1, total_ticks + 1):
            sigma = linear_schedule(float(cfg.STD_START), float(cfg.STD_END), int(cfg.STD_DECAY_STEPS), global_step)

            collect_start = time.perf_counter()
            with torch.no_grad():
                if global_step < int(cfg.SEED_STEPS):
                    action = torch.empty((env.num_envs, 3), device=device).uniform_(-1.0, 1.0)
                else:
                    obs_in = z_t if mode == "spr_z" else image_t
                    action = agent.act(obs_in, goal_t, sigma, eval_mode=False)
            _obs, reward, terminated, truncated, _ = env.step(action)
            timeout = truncated & (~terminated)
            done = terminated | truncated

            term_images = None
            term_z = None
            if torch.any(timeout):
                term_images = env.terminal_rgb[timeout].clone()
                term_z = encode_z(term_images)
            buffer.add(
                images=image_t,
                goals=goal_t,
                actions=action,
                rewards=reward,
                terminated=terminated,
                truncated=timeout,
                z=z_t,
                term_images=term_images,
                term_z=term_z,
            )

            a_mag = torch.clamp(action, -1.0, 1.0).abs().max(dim=1).values
            win["a_mag"] += float(a_mag.sum().cpu())
            win["stop"] += int((a_mag < float(task_cfg.STOP_EPS)).sum().item())
            win["zone"] += int(env.in_stop_zone().sum().item())
            if torch.any(done):
                win["done"] += int(done.sum().item())
                win["success"] += int((env.last_success & done).sum().item())
                win["fail"] += int((env.last_fail & done).sum().item())
                win["timeout"] += int(((env.last_timeout | truncated) & done).sum().item())

            global_step += env.num_envs
            traj_writer.writerow(traj_row(env, "train", global_step, tick, tick, 0, reward[0], bool(done[0])))

            image_t = camera_rgb(env)
            goal_t = env.end_obs()
            z_t = encode_z(image_t)
            win["collect_s"] += time.perf_counter() - collect_start

            if global_step >= int(cfg.SEED_STEPS) and buffer.size > int(cfg.N_STEP) + 1:
                learn_start = time.perf_counter()
                for _ in range(int(cfg.UPDATES_PER_TICK)):
                    batch = buffer.sample(int(cfg.BATCH_SIZE), device, with_images=(mode == "pixels"))
                    metrics = agent.update(batch, sigma)
                    for key, value in metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value
                    metric_count += 1
                    win["updates"] += 1
                win["learn_s"] += time.perf_counter() - learn_start

            if global_step >= next_save:
                save_checkpoint(update_dir / f"drq_{global_step:08d}.pt", agent, agent_cfg, global_step, sigma)
                next_save += int(cfg.SAVE_EVERY_STEPS)

            if tick % int(cfg.LOG_EVERY_TICKS) == 0:
                steps_window = int(cfg.LOG_EVERY_TICKS) * env.num_envs
                elapsed = max(win["collect_s"] + win["learn_s"], 1e-9)
                m = {k: (metric_sums.get(k, 0.0) / max(metric_count, 1)) for k in ("critic_loss", "actor_loss", "q1_mean", "q_target_mean")}
                row = {
                    "tick": tick,
                    "step": global_step,
                    # fps/sample_fps = 仿真步/秒(每秒 env.step 次数,不乘 env 数)
                    "fps": f"{int(cfg.LOG_EVERY_TICKS) / elapsed:.1f}",
                    "sample_fps": f"{int(cfg.LOG_EVERY_TICKS) / max(win['collect_s'], 1e-9):.1f}",
                    "collect_s": f"{win['collect_s']:.3f}",
                    "learn_s": f"{win['learn_s']:.3f}",
                    "done": int(win["done"]),
                    "success": int(win["success"]),
                    "fail": int(win["fail"]),
                    "timeout": int(win["timeout"]),
                    "success_rate": f"{win['success'] / max(win['done'], 1):.4f}",
                    "fail_rate": f"{win['fail'] / max(win['done'], 1):.4f}",
                    "timeout_rate": f"{win['timeout'] / max(win['done'], 1):.4f}",
                    "critic_loss": f"{m['critic_loss']:.5f}",
                    "actor_loss": f"{m['actor_loss']:.5f}",
                    "q1_mean": f"{m['q1_mean']:.4f}",
                    "q_target_mean": f"{m['q_target_mean']:.4f}",
                    "sigma": f"{sigma:.4f}",
                    "a_mag_mean": f"{win['a_mag'] / steps_window:.4f}",
                    "stop_frac": f"{win['stop'] / steps_window:.4f}",
                    "zone_frac": f"{win['zone'] / steps_window:.4f}",
                    "buffer_frames": buffer.frames,
                    "updates": int(win["updates"]),
                }
                log_writer.writerow(row)
                log_file.flush()
                traj_file.flush()
                for key in ("critic_loss", "actor_loss", "q1_mean", "q_target_mean"):
                    tb.add_scalar(f"train/{key}", m[key], global_step)
                tb.add_scalar("train/success_rate", win["success"] / max(win["done"], 1), global_step)
                tb.add_scalar("train/sigma", sigma, global_step)
                print_log(
                    [
                        (
                            "rollout",
                            [
                                ("tick", tick),
                                ("step", global_step),
                                ("fps", float(row["fps"])),
                                ("sigma", sigma),
                                ("success_rate", win["success"] / max(win["done"], 1)),
                                ("fail_rate", win["fail"] / max(win["done"], 1)),
                                ("zone_frac", win["zone"] / steps_window),
                                ("buffer_frames", buffer.frames),
                            ],
                        ),
                        (
                            "learn",
                            [
                                ("updates", int(win["updates"])),
                                ("critic_loss", m["critic_loss"]),
                                ("actor_loss", m["actor_loss"]),
                                ("q1_mean", m["q1_mean"]),
                                ("q_target_mean", m["q_target_mean"]),
                            ],
                        ),
                    ]
                )
                win = {k: 0.0 for k in win}
                metric_sums = {}
                metric_count = 0

        save_checkpoint(log_dir / "last.pt", agent, agent_cfg, global_step, sigma)
        total_s = time.perf_counter() - start_time
        print(f"[INFO] drq student done steps={global_step} time={total_s / 3600.0:.2f}h out={log_dir}")
    finally:
        log_file.close()
        traj_file.close()
        tb.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
