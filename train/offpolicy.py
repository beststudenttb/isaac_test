"""off-policy(DrQ-v2)训练入口 —— arm A 族统一脚本(spr_z / pixels / spr_coadapt)。

替代 train/drq_student.py + train/spr_coadapt.py。算法与超参由 --cfg 指定的 cfg 模块决定
(cfg 内容不动,只是不再写死 import):

    ./IsaacLab/isaaclab.sh -p train/offpolicy.py --cfg drq_student_cfg --obs-mode spr_z
    ./IsaacLab/isaaclab.sh -p train/offpolicy.py --cfg spr_coadapt_cfg

三个 obs_mode 的差别都收在 agent(src/rl/agent.py)与 buffer(src/rl/buffer.py)里:
spr_z 存 z 不存图、pixels/spr_coadapt 存图,spr_coadapt 额外用 K 步 spr_loss。
env/日志/轨迹/checkpoint 全部走 src/rl/ 的共享 helper。
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

parser = argparse.ArgumentParser(description="Off-policy DrQ-v2 training (arm A family).")
parser.add_argument("--cfg", type=str, required=True, help="cfg 模块名,如 drq_student_cfg / spr_coadapt_cfg")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--obs-mode", type=str, default=None, choices=["spr_z", "pixels", "spr_coadapt"])
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
from src.rl.agent import DrQV2Agent, linear_schedule
from src.rl.buffer import ReplayBuffer
from src.rl.env import make_env as build_env, camera_rgb
from src.rl.spr import load_spr
from src.rl.logging import (
    LOG_FIELDS, TRAJ_FIELDS, open_csv, traj_row, print_log, save_checkpoint,
)


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
    d = {
        "obs_mode": mode, "repr_dim": int(repr_dim), "goal_dim": 2, "act_dim": 3,
        "feature_dim": int(cfg.FEATURE_DIM), "hidden_dim": int(cfg.HIDDEN_DIM),
        "lr": float(cfg.LR), "tau": float(cfg.TAU), "noise_clip": float(cfg.NOISE_CLIP),
        "pixels_res": int(cfg.PIXELS_RES), "aug_pad": int(cfg.AUG_PAD),
    }
    if mode == "spr_coadapt":
        d.update(
            spr_z_dim=int(cfg.SPR_Z_DIM), spr_fpn=int(cfg.SPR_FPN), spr_hidden=int(cfg.SPR_HIDDEN),
            spr_pool=int(cfg.SPR_POOL), spr_tau=float(cfg.SPR_TAU), spr_k=int(cfg.SPR_K),
            spr_coef=float(cfg.SPR_COEF), enc_lr_ratio=float(cfg.ENC_LR_RATIO),
        )
    return d


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

    env = build_env(cfg, noise=args_cli.noise, num_envs=args_cli.num_envs,
                    random_stop=args_cli.random_stop, device=args_cli.device)
    device = torch.device(env.device)

    spr = None
    z_dim = 0
    if mode == "spr_z":
        spr = load_spr(Path(cfg.SPR_CKPT), device)
        z_dim = int(spr.z_dim)
        shutil.copy2(Path(cfg.SPR_CKPT), log_dir / "spr_encoder.pt")  # run 目录自包含。

    agent_cfg = agent_cfg_dict(z_dim, mode)
    agent = DrQV2Agent(**agent_cfg, device=device)
    image_shape = (int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3)
    buffer = ReplayBuffer(
        capacity_per_env=int(cfg.CAPACITY_PER_ENV), num_envs=env.num_envs, image_shape=image_shape,
        goal_dim=2, act_dim=3, z_dim=z_dim, n_step=int(cfg.N_STEP), gamma=float(cfg.GAMMA),
        store_images=(mode in ("pixels", "spr_coadapt")),
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
    print(f"[INFO] offpolicy cfg={args_cli.cfg} out={log_dir} mode={mode} envs={env.num_envs} "
          f"total_steps={total_steps} ticks={total_ticks} z_dim={z_dim} device={env.device}")

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
            buffer.add(images=image_t, goals=goal_t, actions=action, rewards=reward,
                       terminated=terminated, truncated=timeout, z=z_t,
                       term_images=term_images, term_z=term_z)

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
                    batch = buffer.sample(
                        int(cfg.BATCH_SIZE), device,
                        with_images=(mode in ("pixels", "spr_coadapt")),
                        spr_k=(int(cfg.SPR_K) if mode == "spr_coadapt" else 0),
                    )
                    metrics = agent.update(batch, sigma)
                    for key, value in metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value
                    metric_count += 1
                    win["updates"] += 1
                win["learn_s"] += time.perf_counter() - learn_start

            if global_step >= next_save:
                save_checkpoint(update_dir / f"drq_{global_step:08d}.pt", agent, agent_cfg,
                                mode, global_step, sigma, getattr(cfg, "SPR_CKPT", None))
                next_save += int(cfg.SAVE_EVERY_STEPS)

            if tick % int(cfg.LOG_EVERY_TICKS) == 0:
                steps_window = int(cfg.LOG_EVERY_TICKS) * env.num_envs
                elapsed = max(win["collect_s"] + win["learn_s"], 1e-9)
                m = {k: (metric_sums.get(k, 0.0) / max(metric_count, 1))
                     for k in ("critic_loss", "actor_loss", "q1_mean", "q_target_mean", "spr_loss", "z_std")}
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
                    "spr_loss": f"{m['spr_loss']:.5f}", "z_std": f"{m['z_std']:.5f}",
                    "sigma": f"{sigma:.4f}",
                    "a_mag_mean": f"{win['a_mag'] / steps_window:.4f}",
                    "stop_frac": f"{win['stop'] / steps_window:.4f}",
                    "zone_frac": f"{win['zone'] / steps_window:.4f}",
                    "buffer_frames": buffer.frames, "updates": int(win["updates"]),
                }
                log_writer.writerow(row)
                log_file.flush()
                traj_file.flush()
                for key in ("critic_loss", "actor_loss", "q1_mean", "q_target_mean"):
                    tb.add_scalar(f"train/{key}", m[key], global_step)
                tb.add_scalar("train/success_rate", win["success"] / max(win["done"], 1), global_step)
                tb.add_scalar("train/sigma", sigma, global_step)
                print_log([
                    ("rollout", [
                        ("tick", tick), ("step", global_step), ("fps", float(row["fps"])),
                        ("sigma", sigma), ("success_rate", win["success"] / max(win["done"], 1)),
                        ("fail_rate", win["fail"] / max(win["done"], 1)),
                        ("zone_frac", win["zone"] / steps_window), ("buffer_frames", buffer.frames),
                    ]),
                    ("learn", [
                        ("updates", int(win["updates"])), ("critic_loss", m["critic_loss"]),
                        ("actor_loss", m["actor_loss"]), ("q1_mean", m["q1_mean"]),
                        ("q_target_mean", m["q_target_mean"]), ("spr_loss", m["spr_loss"]), ("z_std", m["z_std"]),
                    ]),
                ])
                win = {k: 0.0 for k in win}
                metric_sums = {}
                metric_count = 0

        save_checkpoint(log_dir / "last.pt", agent, agent_cfg, mode, global_step, sigma,
                        getattr(cfg, "SPR_CKPT", None))
        total_s = time.perf_counter() - start_time
        print(f"[INFO] offpolicy done steps={global_step} time={total_s / 3600.0:.2f}h out={log_dir}")
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
