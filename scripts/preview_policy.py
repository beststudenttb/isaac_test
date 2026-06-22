"""Preview one teacher PPO policy in IsaacLab.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/preview_policy.py --model models/rl/teacher_ppo/best_val.zip
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview one deterministic teacher policy.")
parser.add_argument("--model", type=str, default="models/rl/teacher_ppo/best_val.zip")
parser.add_argument("--episode-s", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--duration", type=float, default=0.0)
parser.add_argument("--no-camera", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not bool(args_cli.no_camera)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from src.sb3_env import BallPPOEnvCfg, make_sb3_env


def make_env():
    cfg = BallPPOEnvCfg()
    cfg.seed = int(args_cli.seed)
    cfg.episode_length_s = float(args_cli.episode_s)
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 16.0
    cfg.sim.device = args_cli.device
    cfg.use_camera = not bool(args_cli.no_camera)
    return make_sb3_env(cfg, fast_variant=True)


def main():
    model_path = Path(args_cli.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    env = make_env()
    base_env = env.unwrapped
    base_env.sim.set_camera_view(eye=[8.0, -10.0, 6.0], target=[0.0, 0.0, 0.5])
    if not bool(args_cli.no_camera) and not bool(args_cli.headless):
        import omni.ui as ui
        from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name

        camera_path = "/World/envs/env_0/Robot/head/eye/front_camera"
        viewport = get_viewport_from_window_name("Viewport")
        viewport.set_active_camera("/OmniverseKit_Persp")
        camera_window = create_viewport_window("Policy Camera", width=640, height=480)
        camera_window.viewport_api.set_active_camera(camera_path)
        main_window = ui.Workspace.get_window("Viewport")
        dock_window = ui.Workspace.get_window("Policy Camera")
        if main_window is not None and dock_window is not None:
            dock_position = getattr(ui.DockPosition, "RIGHT", ui.DockPosition.SAME)
            dock_window.dock_in(main_window, dock_position, 0.35)
            dock_window.focus()
        print(f"[INFO] Camera viewport uses {camera_path}")

    model = PPO.load(str(model_path), env=env, device=args_cli.device)
    obs = env.reset()

    frames = 0
    episodes = 0
    start_time = time.perf_counter()
    last_time = start_time
    print(f"[INFO] preview model={model_path} deterministic=1 dt={base_env.step_dt:.3f}")

    try:
        while simulation_app.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            frames += 1

            if bool(done[0]):
                episodes += 1
                print(
                    f"[INFO] done episode={episodes} reward={float(reward[0]):.3f} "
                    f"success={int(base_env.last_success[0].item())} "
                    f"fail={int(base_env.last_fail[0].item())} "
                    f"timeout={int(base_env.last_timeout[0].item())}"
                )

            now = time.perf_counter()
            if now - last_time >= 1.0:
                fps = frames / (now - last_time)
                rtf = fps * base_env.step_dt
                raw = action[0]
                move = base_env.last_move_actions[0].detach().cpu().numpy()
                robot = base_env.last_robot_xy[0].detach().cpu().numpy()
                target = base_env.last_target_xy[0].detach().cpu().numpy()
                delta = target - robot
                body_angle = float(np.arctan2(delta[1], delta[0]))
                head = float(base_env.last_head_yaw[0].item())
                vx = float(np.cos(head) * move[0] - np.sin(head) * move[1])
                vy = float(np.sin(head) * move[0] + np.cos(head) * move[1])
                print(
                    f"fps={fps:.1f} rtf={rtf:.2f} "
                    f"px={float(base_env.last_px_x[0].item()):.1f} "
                    f"d={float(base_env.last_dist[0].item()):.2f} "
                    f"raw=({raw[0]:.2f},{raw[1]:.2f},{raw[2]:.2f}) "
                    f"move=({move[0]:.2f},{move[1]:.2f},{move[2]:.2f}) "
                    f"v_world=({vx:.2f},{vy:.2f}) "
                    f"pos=({robot[0]:.2f},{robot[1]:.2f}) "
                    f"target=({target[0]:.2f},{target[1]:.2f}) "
                    f"head={head:.2f} body_ang={body_angle:.2f}"
                )
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
