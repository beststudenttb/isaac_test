"""Preview the shared BallEnv scene.

Run from the project directory:

    ./IsaacLab/isaaclab.sh -p scripts/preview_env.py --num-envs 4
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview the BallEnv base scene.")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--spacing", type=float, default=16.0)
parser.add_argument("--duration", type=float, default=0.0)
parser.add_argument("--camera", action="store_true")
parser.add_argument("--camera-view-env", type=int, default=0)
parser.add_argument("--target-interval", type=float, default=10.0)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = bool(args_cli.camera)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import BallEnv, BallEnvCfg


def reset_targets(env: BallEnv):
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.sample_targets(env_ids)
    env.write_target_pose(env_ids)
    label = env.project_target()
    px_x = label["px_x"].detach().cpu().tolist()
    dist = label["dist"].detach().cpu().tolist()
    print("[INFO] target reset")
    for env_id, (x, d) in enumerate(zip(px_x, dist, strict=True)):
        print(f"  env_{env_id}: x={x:.2f} dist={d:.2f}")


def make_env() -> BallEnv:
    cfg = BallEnvCfg()
    cfg.seed = int(args_cli.seed)
    cfg.episode_length_s = 3600.0
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = float(args_cli.spacing)
    cfg.sim.device = args_cli.device
    cfg.use_camera = bool(args_cli.camera)
    return BallEnv(cfg)


def main():
    env = make_env()
    env.sim.set_camera_view(eye=[8.0, -10.0, 6.0], target=[0.0, 0.0, 0.5])
    obs, _ = env.reset()
    reset_targets(env)
    if bool(args_cli.camera) and not bool(args_cli.headless):
        import omni.ui as ui
        from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name

        env_id = max(0, min(int(args_cli.camera_view_env), env.num_envs - 1))
        camera_path = f"/World/envs/env_{env_id}/Robot/head/eye/front_camera"
        viewport = get_viewport_from_window_name("Viewport")
        viewport.set_active_camera("/OmniverseKit_Persp")
        camera_window = create_viewport_window("Env Camera", width=640, height=480)
        camera_window.viewport_api.set_active_camera(camera_path)
        main_window = ui.Workspace.get_window("Viewport")
        dock_window = ui.Workspace.get_window("Env Camera")
        if main_window is not None and dock_window is not None:
            dock_position = getattr(ui.DockPosition, "RIGHT", ui.DockPosition.SAME)
            dock_window.dock_in(main_window, dock_position, 0.35)
            dock_window.focus()
        print(f"[INFO] Camera viewport uses {camera_path}")

    actions = torch.zeros((env.num_envs, int(env.cfg.action_space)), device=env.device)
    frames = 0
    start_time = time.perf_counter()
    last_time = start_time
    last_target_time = start_time

    print(
        f"[INFO] BallEnv preview ready. num_envs={env.num_envs} spacing={env.cfg.scene.env_spacing:.1f} "
        f"camera={int(env.cfg.use_camera)} dt={env.physics_dt:.3f}"
    )

    try:
        while simulation_app.is_running():
            obs, _reward, _terminated, _truncated, _info = env.step(actions)
            frames += 1
            now = time.perf_counter()
            if now - last_target_time >= float(args_cli.target_interval):
                reset_targets(env)
                last_target_time = now
            if now - last_time >= 1.0:
                fps = frames / (now - last_time)
                rtf = fps * env.step_dt
                print(f"fps={fps:.1f} rtf={rtf:.2f}")
                frames = 0
                last_time = now
            if float(args_cli.duration) > 0.0 and now - start_time >= float(args_cli.duration):
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Press Enter to close Isaac Sim...")
    finally:
        simulation_app.close()
