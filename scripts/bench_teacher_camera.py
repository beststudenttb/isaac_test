"""Benchmark teacher env with tiled camera read.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/bench_teacher_camera.py --num-envs 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Benchmark teacher env camera read FPS.")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--show", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if not bool(args_cli.show):
    args_cli.headless = True
    args_cli.livestream = 0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sb3_env import BallPPOEnv, BallPPOEnvCfg


def make_env() -> BallPPOEnv:
    cfg = BallPPOEnvCfg()
    cfg.seed = int(args_cli.seed)
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = 16.0
    cfg.sim.device = args_cli.device
    cfg.use_camera = True
    cfg.read_camera = True
    return BallPPOEnv(cfg)


def main():
    env = make_env()
    obs, _ = env.reset()
    actions = torch.zeros((env.num_envs, int(env.cfg.action_space)), device=env.device)
    frames = 0
    start = time.perf_counter()
    last = start
    last_frames = 0
    print(
        f"[INFO] bench teacher camera envs={env.num_envs} "
        f"res={env.cfg.image_width}x{env.cfg.image_height} dt={env.step_dt:.3f}"
    )

    try:
        while simulation_app.is_running():
            obs, _reward, _terminated, _truncated, _info = env.step(actions)
            frames += 1
            now = time.perf_counter()
            if now - last >= 1.0:
                fps = (frames - last_frames) / (now - last)
                print(f"step_fps={fps:.1f} samples_s={fps * env.num_envs:.1f}")
                last = now
                last_frames = frames
            if now - start >= float(args_cli.duration):
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
