"""SAC + HER off-policy 训练入口(与 train/offpolicy.py 结构对齐,算法换成 SAC,buffer 换成 HER)。

    ./IsaacLab/isaaclab.sh -p train/sac.py --cfg sac_cfg --obs-mode spr_z --noise
    ./IsaacLab/isaaclab.sh -p train/sac.py --cfg sac_cfg --obs-mode pixels --noise

只支持 spr_z / pixels(不含 spr_coadapt)。采集时额外存每步 cur_px/cur_dist/fail 供 HER relabel。
"""

from __future__ import annotations

import argparse
import importlib
import random
import shutil
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SAC + HER off-policy training.")
parser.add_argument("--cfg", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--obs-mode", type=str, default=None, choices=["spr_z", "pixels"])
parser.add_argument("--random-stop", action="store_true")
parser.add_argument("--noise", action="store_true")
parser.add_argument("--show", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

cfg = importlib.import_module(args_cli.cfg)

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
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import task_cfg
from src.rl.sac_agent import SACAgent
from src.rl.her_buffer import HERReplayBuffer
from src.rl.env import make_env as build_env, camera_rgb
from src.rl.spr import load_spr
from src.rl.logging import TRAJ_FIELDS, open_csv, traj_row, print_log

SAC_LOG_FIELDS = [
    "tick", "step", "fps", "sample_fps", "collect_s", "learn_s",
    "done", "success", "fail", "timeout",
    "success_rate", "fail_rate", "timeout_rate",
    "critic_loss", "actor_loss", "q1_mean", "q_target_mean", "alpha", "entropy",
    "a_mag_mean", "stop_frac", "zone_frac", "buffer_frames", "updates",
]


def obs_mode() -> str:
    return str(args_cli.obs_mode) if args_cli.obs_mode is not None else str(cfg.OBS_MODE)


def out_dir() -> Path:
    base = Path(cfg.OUT_DIR)
    if obs_mode() == "pixels" and hasattr(cfg, "PIXELS_OUT_DIR"):
        base = Path(cfg.PIXELS_OUT_DIR)
    if args_cli.random_stop:
        base = base.with_name(base.name + str(cfg.RANDOM_STOP_SUFFIX))
    if bool(cfg.CLEAR_OUT_DIR) and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_config(path: Path) -> None:
    lines = [f"{name} = {getattr(cfg, name)!r}" for name in dir(cfg) if name.isupper()]
    lines.append(f"OBS_MODE_USED = {obs_mode()!r}")
    lines.append(f"RANDOM_STOP = {bool(args_cli.random_stop)!r}")
    lines.append(f"NOISE = {bool(args_cli.noise)!r}")
    lines.append(f"DEVICE_USED = {args_cli.device!r}")
    (path / "config.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def agent_cfg_dict(repr_dim: int, mode: str) -> dict:
    return {
        "obs_mode": mode, "repr_dim": int(repr_dim), "goal_dim": 2, "act_dim": 3,
        "feature_dim": int(cfg.FEATURE_DIM), "hidden_dim": int(cfg.HIDDEN_DIM),
        "lr": float(cfg.LR), "tau": float(cfg.TAU), "gamma": float(cfg.GAMMA),
        "alpha_init": float(cfg.ALPHA_INIT), "alpha_lr": float(cfg.ALPHA_LR),
        "target_entropy": float(cfg.TARGET_ENTROPY),
        "pixels_res": int(cfg.PIXELS_RES), "aug_pad": int(cfg.AUG_PAD),
    }


def save_ckpt(path: Path, agent, agent_cfg: dict, mode: str, step: int) -> None:
    torch.save({"agent": agent.save_dict(), "agent_cfg": agent_cfg,
                "obs_mode": mode, "step": int(step)}, path)


def main() -> None:
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mode = obs_mode()
    log_dir = out_dir()
    write_config(log_dir)
    log_file, log_writer = open_csv(log_dir / "log.csv", SAC_LOG_FIELDS)
    traj_file, traj_writer = open_csv(log_dir / "traj_train_env0.csv", TRAJ_FIELDS)
    update_dir = log_dir / "updates"
    update_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir / "tb"))

    env = build_env(cfg, noise=args_cli.noise, num_envs=args_cli.num_envs,
                    random_stop=args_cli.random_stop, device=args_cli.device)
    device = torch.device(env.device)

    spr = None
    z_dim = 0
    if mode == "spr_z":
        spr = load_spr(Path(cfg.SPR_CKPT), device)
        z_dim = int(spr.z_dim)
        shutil.copy2(Path(cfg.SPR_CKPT), log_dir / "spr_encoder.pt")

    agent_cfg = agent_cfg_dict(z_dim, mode)
    agent = SACAgent(**agent_cfg, device=device)
    image_shape = (int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3)
    store_images = (mode == "pixels")
    buffer = HERReplayBuffer(
        capacity_per_env=int(cfg.CAPACITY_PER_ENV), num_envs=env.num_envs, image_shape=image_shape,
        goal_dim=2, act_dim=3, z_dim=z_dim, gamma=float(cfg.GAMMA), store_images=store_images,
        her_ratio=float(cfg.HER_RATIO), future_h=int(cfg.HER_FUTURE_H),
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
    print(f"[INFO] sac+her cfg={args_cli.cfg} out={log_dir} mode={mode} envs={env.num_envs} "
          f"total_steps={total_steps} ticks={total_ticks} z_dim={z_dim} device={env.device}")

    global_step = 0
    next_save = int(cfg.SAVE_EVERY_STEPS)
    win = {k: 0.0 for k in ("done", "success", "fail", "timeout", "a_mag", "stop", "zone", "collect_s", "learn_s", "updates")}
    metric_sums: dict[str, float] = {}
    metric_count = 0
    start_time = time.perf_counter()

    try:
        for tick in range(1, total_ticks + 1):
            collect_start = time.perf_counter()
            with torch.no_grad():
                if global_step < int(cfg.SEED_STEPS):
                    action = torch.empty((env.num_envs, 3), device=device).uniform_(-1.0, 1.0)
                else:
                    obs_in = z_t if mode == "spr_z" else image_t
                    action = agent.act(obs_in, goal_t, eval_mode=False)
            _obs, reward, terminated, truncated, _ = env.step(action)
            timeout = truncated & (~terminated)
            done = terminated | truncated

            term_images = None
            term_z = None
            if torch.any(timeout):
                term_images = env.terminal_rgb[timeout].clone()
                term_z = encode_z(term_images)
            buffer.add(
                images=image_t, goals=goal_t, actions=action, rewards=reward,
                terminated=terminated, truncated=timeout, fail=env.last_fail,
                cur_px=env.last_px_x, cur_dist=env.last_dist, z=z_t,
                term_images=(term_images if store_images else None),
                term_z=(term_z if z_dim > 0 else None),
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

            if global_step >= int(cfg.SEED_STEPS) and buffer.size > 2:
                learn_start = time.perf_counter()
                for _ in range(int(cfg.UPDATES_PER_TICK)):
                    batch = buffer.sample(int(cfg.BATCH_SIZE), device, with_images=store_images)
                    metrics = agent.update(batch)
                    for key, value in metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value
                    metric_count += 1
                    win["updates"] += 1
                win["learn_s"] += time.perf_counter() - learn_start

            if global_step >= next_save:
                save_ckpt(update_dir / f"sac_{global_step:08d}.pt", agent, agent_cfg, mode, global_step)
                next_save += int(cfg.SAVE_EVERY_STEPS)

            if tick % int(cfg.LOG_EVERY_TICKS) == 0:
                steps_window = int(cfg.LOG_EVERY_TICKS) * env.num_envs
                elapsed = max(win["collect_s"] + win["learn_s"], 1e-9)
                m = {k: (metric_sums.get(k, 0.0) / max(metric_count, 1))
                     for k in ("critic_loss", "actor_loss", "q1_mean", "q_target_mean", "alpha", "entropy")}
                row = {
                    "tick": tick, "step": global_step,
                    "fps": f"{int(cfg.LOG_EVERY_TICKS) / elapsed:.1f}",
                    "sample_fps": f"{int(cfg.LOG_EVERY_TICKS) / max(win['collect_s'], 1e-9):.1f}",
                    "collect_s": f"{win['collect_s']:.3f}", "learn_s": f"{win['learn_s']:.3f}",
                    "done": int(win["done"]), "success": int(win["success"]),
                    "fail": int(win["fail"]), "timeout": int(win["timeout"]),
                    "success_rate": f"{win['success'] / max(win['done'], 1):.4f}",
                    "fail_rate": f"{win['fail'] / max(win['done'], 1):.4f}",
                    "timeout_rate": f"{win['timeout'] / max(win['done'], 1):.4f}",
                    "critic_loss": f"{m['critic_loss']:.5f}", "actor_loss": f"{m['actor_loss']:.5f}",
                    "q1_mean": f"{m['q1_mean']:.4f}", "q_target_mean": f"{m['q_target_mean']:.4f}",
                    "alpha": f"{m['alpha']:.5f}", "entropy": f"{m['entropy']:.4f}",
                    "a_mag_mean": f"{win['a_mag'] / steps_window:.4f}",
                    "stop_frac": f"{win['stop'] / steps_window:.4f}",
                    "zone_frac": f"{win['zone'] / steps_window:.4f}",
                    "buffer_frames": buffer.frames, "updates": int(win["updates"]),
                }
                log_writer.writerow(row)
                log_file.flush()
                traj_file.flush()
                for key in ("critic_loss", "actor_loss", "q1_mean", "q_target_mean", "alpha", "entropy"):
                    tb.add_scalar(f"train/{key}", m[key], global_step)
                tb.add_scalar("train/success_rate", win["success"] / max(win["done"], 1), global_step)
                tb.add_scalar("train/stop_frac", win["stop"] / steps_window, global_step)
                print_log([
                    ("rollout", [
                        ("tick", tick), ("step", global_step), ("fps", float(row["fps"])),
                        ("success_rate", win["success"] / max(win["done"], 1)),
                        ("fail_rate", win["fail"] / max(win["done"], 1)),
                        ("stop_frac", win["stop"] / steps_window),
                        ("zone_frac", win["zone"] / steps_window), ("buffer_frames", buffer.frames),
                    ]),
                    ("learn", [
                        ("updates", int(win["updates"])), ("critic_loss", m["critic_loss"]),
                        ("actor_loss", m["actor_loss"]), ("q1_mean", m["q1_mean"]),
                        ("alpha", m["alpha"]), ("entropy", m["entropy"]),
                    ]),
                ])
                win = {k: 0.0 for k in win}
                metric_sums = {}
                metric_count = 0

        save_ckpt(log_dir / "last.pt", agent, agent_cfg, mode, global_step)
        total_s = time.perf_counter() - start_time
        print(f"[INFO] sac+her done steps={global_step} time={total_s / 3600.0:.2f}h out={log_dir}")
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
