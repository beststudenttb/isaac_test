"""Preview one custom student PPO policy.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/preview_student.py --model models/student_ppo/last.pt
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview one deterministic custom student policy.")
parser.add_argument("--model", type=str, default="models/student_ppo/last.pt")
parser.add_argument("--episode-s", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--duration", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ppo import ActorCritic
from src.sb3_env import BallPPOEnv, BallPPOEnvCfg


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


def make_env() -> BallPPOEnv:
    cfg = BallPPOEnvCfg()
    cfg.seed = int(args_cli.seed)
    cfg.episode_length_s = float(args_cli.episode_s)
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 16.0
    cfg.sim.device = args_cli.device
    cfg.use_camera = True
    cfg.read_camera = True
    return BallPPOEnv(cfg)


def load_model(path: Path, env: BallPPOEnv) -> ActorCritic:
    ckpt = torch.load(path, map_location=env.device)
    model = ActorCritic(
        obs_dim=int(env.cfg.observation_space),
        act_dim=int(env.cfg.action_space),
        pi_hidden=[64, 64],
        vf_hidden=[64, 64],
        activation=ACTIVATIONS["tanh"],
        init_std=0.5,
    ).to(env.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def get_rgb(env: BallPPOEnv):
    if env.camera is None:
        return None
    return env.camera.data.output["rgb"]


def main():
    model_path = Path(args_cli.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    env = make_env()
    env.sim.set_camera_view(eye=[8.0, -10.0, 6.0], target=[0.0, 0.0, 0.5])
    if not bool(args_cli.headless):
        import omni.ui as ui
        from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name

        camera_path = "/World/envs/env_0/Robot/head/eye/front_camera"
        viewport = get_viewport_from_window_name("Viewport")
        viewport.set_active_camera("/OmniverseKit_Persp")
        camera_window = create_viewport_window("Student Camera", width=640, height=480)
        camera_window.viewport_api.set_active_camera(camera_path)
        main_window = ui.Workspace.get_window("Viewport")
        dock_window = ui.Workspace.get_window("Student Camera")
        if main_window is not None and dock_window is not None:
            dock_position = getattr(ui.DockPosition, "RIGHT", ui.DockPosition.SAME)
            dock_window.dock_in(main_window, dock_position, 0.35)
            dock_window.focus()
        print(f"[INFO] Camera viewport uses {camera_path}")
    model = load_model(model_path, env)
    obs, _ = env.reset()
    obs_t = obs["policy"]

    frames = 0
    episodes = 0
    start_time = time.perf_counter()
    last_time = start_time
    print(f"[INFO] preview student model={model_path} deterministic=1 camera=1 dt={env.step_dt:.3f}")

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                action = model.predict(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action)
            rgb = get_rgb(env)
            done = terminated | truncated
            obs_t = obs["policy"]
            frames += 1

            if bool(done[0]):
                episodes += 1
                print(
                    f"[INFO] done episode={episodes} reward={float(reward[0].item()):.3f} "
                    f"success={int(env.last_success[0].item())} "
                    f"fail={int(env.last_fail[0].item())} "
                    f"timeout={int(env.last_timeout[0].item())}"
                )

            now = time.perf_counter()
            if now - last_time >= 1.0:
                fps = frames / (now - last_time)
                rtf = fps * env.step_dt
                img_shape = tuple(rgb.shape) if rgb is not None else None
                raw = action[0].detach().cpu().numpy()
                move = env.last_move_actions[0].detach().cpu().numpy()
                robot = env.last_robot_xy[0].detach().cpu().numpy()
                target = env.last_target_xy[0].detach().cpu().numpy()
                delta = target - robot
                body_angle = float(np.arctan2(delta[1], delta[0]))
                head = float(env.last_head_yaw[0].item())
                vx = float(np.cos(head) * move[0] - np.sin(head) * move[1])
                vy = float(np.sin(head) * move[0] + np.cos(head) * move[1])
                print(
                    f"fps={fps:.1f} rtf={rtf:.2f} "
                    f"px={float(env.last_px_x[0].item()):.1f} "
                    f"d={float(env.last_dist[0].item()):.2f} "
                    f"raw=({raw[0]:.2f},{raw[1]:.2f},{raw[2]:.2f}) "
                    f"move=({move[0]:.2f},{move[1]:.2f},{move[2]:.2f}) "
                    f"v_world=({vx:.2f},{vy:.2f}) "
                    f"pos=({robot[0]:.2f},{robot[1]:.2f}) "
                    f"target=({target[0]:.2f},{target[1]:.2f}) "
                    f"head={head:.2f} body_ang={body_angle:.2f} "
                    f"rgb={img_shape}"
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
