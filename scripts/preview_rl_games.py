"""Preview one rl-games CNN+LSTM policy.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/preview_rl_games.py
    ./IsaacLab/isaaclab.sh -p scripts/preview_rl_games.py --model last
"""

from __future__ import annotations

import argparse
import csv
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


parser = argparse.ArgumentParser(description="Preview one rl-games policy.")
parser.add_argument("--cfg", type=Path, default=Path("rl_games_ball/rl_games_ppo_cnn_lstm_asym.yaml"))
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--duration", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--episode-s", type=float, default=None)
parser.add_argument("--csv", type=Path, default=Path("runs/preview_rl_games.csv"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import yaml
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

from rl_games_ball.env import RlGamesBallEnvCfg, make_env


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def model_path(name: str) -> Path:
    if name == "best":
        return Path("rl_games_ball/ball_camera_lstm_asym.pth")
    if name == "last":
        return Path("rl_games_ball/last_ball_camera_lstm_asym_ep_10000_rew__5.747683_.pth")
    return Path(name)


def open_camera_viewport():
    if bool(args_cli.headless):
        return
    import omni.ui as ui
    from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name

    camera_path = "/World/envs/env_0/Robot/head/eye/front_camera"
    viewport = get_viewport_from_window_name("Viewport")
    viewport.set_active_camera("/OmniverseKit_Persp")
    camera_window = create_viewport_window("RL-Games Camera", width=640, height=480)
    camera_window.viewport_api.set_active_camera(camera_path)
    main_window = ui.Workspace.get_window("Viewport")
    dock_window = ui.Workspace.get_window("RL-Games Camera")
    if main_window is not None and dock_window is not None:
        dock_position = getattr(ui.DockPosition, "RIGHT", ui.DockPosition.SAME)
        dock_window.dock_in(main_window, dock_position, 0.35)
        dock_window.focus()
    print(f"[INFO] Camera viewport uses {camera_path}")


def make_wrapped_env(agent_cfg: dict) -> tuple[RlGamesVecEnvWrapper, object]:
    env_cfg = RlGamesBallEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 16.0
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)
    if args_cli.episode_s is not None:
        env_cfg.episode_length_s = float(args_cli.episode_s)

    base_env = make_env(env_cfg)
    params = agent_cfg["params"]
    rl_device = params["config"]["device"]
    clip_obs = params["env"].get("clip_observations", math.inf)
    clip_actions = params["env"].get("clip_actions", math.inf)
    obs_groups = params["env"].get("obs_groups")
    concate_obs_groups = params["env"].get("concate_obs_groups", True)
    env = RlGamesVecEnvWrapper(base_env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    return env, base_env


def build_player(agent_cfg: dict, env: RlGamesVecEnvWrapper, checkpoint: Path):
    runner = Runner()
    runner.load(agent_cfg)
    params = runner.params
    params["config"]["num_actors"] = 1
    params["config"]["vec_env"] = env
    params["config"]["env_info"] = env.get_env_info()
    params["config"]["player"]["games_num"] = 1
    params["config"]["player"]["deterministic"] = True
    params["load_checkpoint"] = True
    params["load_path"] = str(checkpoint)
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.reset()
    return player


def main():
    checkpoint = model_path(args_cli.model)
    if not checkpoint.exists():
        raise FileNotFoundError(f"model not found: {checkpoint}")

    agent_cfg = copy.deepcopy(load_yaml(args_cli.cfg))
    env, base_env = make_wrapped_env(agent_cfg)
    player = build_player(agent_cfg, env, checkpoint)

    base_env.sim.set_camera_view(eye=[8.0, -10.0, 6.0], target=[0.0, 0.0, 0.5])
    open_camera_viewport()

    obs = player.env_reset(env)
    player.get_batch_size(obs, 1)

    args_cli.csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = args_cli.csv.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "step",
            "episode",
            "robot_x",
            "robot_y",
            "robot_yaw",
            "head_yaw",
            "target_x",
            "target_y",
            "px_x",
            "dist",
            "action_x",
            "action_y",
            "action_w",
            "reward",
            "success",
            "fail",
            "timeout",
            "done",
        ]
    )

    frames = 0
    step = 0
    episodes = 0
    start_time = time.perf_counter()
    last_time = start_time
    print(f"[INFO] preview rl-games model={checkpoint} deterministic=1 camera=1 dt={base_env.step_dt:.3f}")

    while simulation_app.is_running():
        action = player.get_action(obs, is_deterministic=True)
        obs, reward, done, _info = player.env_step(env, action)
        frames += 1
        step += 1

        label = base_env.project_target()
        robot = base_env.robot_xy[0].detach().cpu().numpy()
        target = base_env.target_xy[0].detach().cpu().numpy()
        raw = action[0].detach().cpu().numpy()
        writer.writerow(
            [
                step,
                episodes,
                f"{robot[0]:.6f}",
                f"{robot[1]:.6f}",
                f"{float(base_env.robot_yaw[0].item()):.6f}",
                f"{float(base_env.head_yaw[0].item()):.6f}",
                f"{target[0]:.6f}",
                f"{target[1]:.6f}",
                f"{float(label['px_x'][0].item()):.6f}",
                f"{float(label['dist'][0].item()):.6f}",
                f"{float(raw[0]):.6f}",
                f"{float(raw[1]):.6f}",
                f"{float(raw[2]):.6f}",
                f"{float(reward[0].item()):.6f}",
                int(base_env.last_success[0].item()),
                int(base_env.last_fail[0].item()),
                int(base_env.last_timeout[0].item()),
                int(done[0].item()),
            ]
        )

        if bool(done[0]):
            episodes += 1
            if player.is_rnn:
                for state in player.states:
                    state[:, done.nonzero(as_tuple=False), :] = 0.0
            print(
                f"[INFO] done episode={episodes} reward={float(reward[0].item()):.3f} "
                f"success={int(base_env.last_success[0].item())} "
                f"fail={int(base_env.last_fail[0].item())} "
                f"timeout={int(base_env.last_timeout[0].item())}"
            )

        now = time.perf_counter()
        if now - last_time >= 1.0:
            fps = frames / (now - last_time)
            rtf = fps * base_env.step_dt
            delta = target - robot
            body_angle = float(np.arctan2(delta[1], delta[0]))
            head = float(base_env.head_yaw[0].item())
            print(
                f"fps={fps:.1f} rtf={rtf:.2f} "
                f"px={float(label['px_x'][0].item()):.1f} "
                f"d={float(label['dist'][0].item()):.2f} "
                f"a=({raw[0]:.2f},{raw[1]:.2f},{raw[2]:.2f}) "
                f"pos=({robot[0]:.2f},{robot[1]:.2f}) "
                f"target=({target[0]:.2f},{target[1]:.2f}) "
                f"head={head:.2f} body_ang={body_angle:.2f}"
            )
            frames = 0
            last_time = now

        if float(args_cli.duration) > 0.0 and now - start_time >= float(args_cli.duration):
            break

    csv_file.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
