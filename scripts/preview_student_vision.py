"""Preview one vision student PPO policy.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/preview_student_vision.py --cv-model old --state xd --student
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "train"))

import student_vision_cfg as cfg


CV_MODEL_ALIAS = {
    "old": "old",
    "old-m": "old-mobile",
    "resnet": "resnet",
    "mobile": "mobile",
}

STATE_ALIAS = {
    "xd": "pred",
    "shared": "shared_feature",
    "feature": "reserve_feature",
}


parser = argparse.ArgumentParser(description="Preview one deterministic vision student policy.")
parser.add_argument("--state", choices=["xd", "shared", "feature"], default=None)
parser.add_argument("--cv-model", choices=["old", "old-m", "resnet", "mobile"], default=None)
mode_group = parser.add_mutually_exclusive_group()
mode_group.add_argument("--student", action="store_true")
mode_group.add_argument("--teacher", action="store_true")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--episode-s", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--duration", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if (args_cli.cv_model is None) != (args_cli.state is None):
    raise ValueError("--cv-model and --state must be used together")
if args_cli.cv_model is not None:
    cli_model = CV_MODEL_ALIAS[str(args_cli.cv_model)]
    cli_state = STATE_ALIAS[str(args_cli.state)]
    if not any(str(item["model"]) == cli_model and str(item["state"]) == cli_state for item in cfg.VISION_CHOICES.values()):
        raise ValueError(f"state={args_cli.state} is not available for cv-model={args_cli.cv_model}")

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import isaaclab.sim as sim_utils

from src import task_cfg
from src.cv_extractor.simple_cnn import BallVisionNet, MobileBallNet, OldBallVisionNet, OldMobileBallNet
from src.ppo import ActorCritic
from src.sb3_env import BallPPOEnv, BallPPOEnvCfg


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

VISION_MODELS = {
    "resnet": BallVisionNet,
    "mobile": MobileBallNet,
    "old": OldBallVisionNet,
    "old-mobile": OldMobileBallNet,
}


def vision_choice() -> dict:
    if args_cli.cv_model is None and args_cli.state is None:
        name = str(cfg.VISION_CHOICE)
        if name not in cfg.VISION_CHOICES:
            raise KeyError(f"unknown VISION_CHOICE: {name}")
        return cfg.VISION_CHOICES[name]

    model = CV_MODEL_ALIAS[str(args_cli.cv_model)]
    state = STATE_ALIAS[str(args_cli.state)]
    for choice in cfg.VISION_CHOICES.values():
        if str(choice["model"]) == model and str(choice["state"]) == state:
            return choice
    raise ValueError(f"state={args_cli.state} is not available for cv-model={args_cli.cv_model}")


def vision_choice_name() -> str:
    if args_cli.cv_model is None and args_cli.state is None:
        return str(cfg.VISION_CHOICE)
    return f"{args_cli.cv_model}_{args_cli.state}"


def train_mode() -> str:
    if args_cli.student:
        return "student"
    if args_cli.teacher:
        return "teacher"
    return str(cfg.TRAIN_MODE)


def run_name() -> str:
    return f"{vision_choice_name()}_{train_mode()}"


def model_path() -> Path:
    root = Path(cfg.OUT_DIR) / run_name()
    model = str(args_cli.model)
    if model.isdigit():
        return root / "updates" / f"update_{int(model):06d}.pt"
    if model == "best":
        best = root / "best_offline.pt"
        if best.exists():
            return best
        return root / "best.pt"
    if model == "last":
        return root / "last.pt"
    return Path(model)


def make_env() -> BallPPOEnv:
    env_cfg = BallPPOEnvCfg()
    env_cfg.seed = int(cfg.SEED if args_cli.seed is None else args_cli.seed)
    env_cfg.episode_length_s = float(cfg.EPISODE_S if args_cli.episode_s is None else args_cli.episode_s)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 16.0
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.render = sim_utils.RenderCfg(
        rendering_mode=str(cfg.RENDERING_MODE),
        antialiasing_mode=str(cfg.ANTIALIASING_MODE),
        dlss_mode=int(cfg.DLSS_MODE),
        enable_dlssg=False,
    )
    env_cfg.use_camera = True
    env_cfg.read_camera = True
    env_cfg.num_rerenders_on_reset = int(cfg.RERENDER_ON_RESET)
    return BallPPOEnv(env_cfg)


def load_encoder(device: torch.device) -> nn.Module:
    choice = vision_choice()
    path = Path(choice["ckpt"])
    if not path.exists():
        raise FileNotFoundError(f"vision checkpoint not found: {path}")
    model_cls = VISION_MODELS[str(choice["model"])]
    ckpt = torch.load(path, map_location=device)
    model = model_cls(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def select_state(out: dict[str, torch.Tensor]) -> torch.Tensor:
    state = str(vision_choice()["state"])
    if state == "pred":
        dist = torch.clamp(out["distance_pred"] / float(task_cfg.FAIL_FAR), 0.0, 1.0)
        return torch.cat([out["x_pred"], dist], dim=1)
    value = out[state]
    if value.ndim > 2:
        value = value.flatten(1)
    return value


def encode_image(image: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
    if image.shape[-1] > 3:
        image = image[..., :3]
    with torch.no_grad():
        if bool(cfg.ENCODER_FP16) and image.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = encoder(image)
        else:
            out = encoder(image)
    return select_state(out).float().detach()


def vision_obs(env: BallPPOEnv, encoder: nn.Module) -> torch.Tensor:
    if env.camera is None:
        raise RuntimeError("vision preview requires env camera")
    return encode_image(env.camera.data.output["rgb"], encoder)


def load_model(path: Path, device: torch.device) -> ActorCritic:
    ckpt = torch.load(path, map_location=device)
    choice = vision_choice()
    model = ActorCritic(
        obs_dim=int(choice["dim"]),
        act_dim=3,
        pi_hidden=list(cfg.POLICY_NET),
        vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)],
        init_std=float(cfg.STD_INIT),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def open_camera_viewport():
    if bool(args_cli.headless):
        return
    import omni.ui as ui
    from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name

    camera_path = "/World/envs/env_0/Robot/head/eye/front_camera"
    viewport = get_viewport_from_window_name("Viewport")
    viewport.set_active_camera("/OmniverseKit_Persp")
    camera_window = create_viewport_window("Vision Student Camera", width=640, height=480)
    camera_window.viewport_api.set_active_camera(camera_path)
    main_window = ui.Workspace.get_window("Viewport")
    dock_window = ui.Workspace.get_window("Vision Student Camera")
    if main_window is not None and dock_window is not None:
        dock_position = getattr(ui.DockPosition, "RIGHT", ui.DockPosition.SAME)
        dock_window.dock_in(main_window, dock_position, 0.35)
        dock_window.focus()
    print(f"[INFO] Camera viewport uses {camera_path}")


def main():
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")

    env = make_env()
    device = torch.device(env.device)
    encoder = load_encoder(device)
    model = load_model(path, device)

    env.sim.set_camera_view(eye=[8.0, -10.0, 6.0], target=[0.0, 0.0, 0.5])
    open_camera_viewport()

    env.reset()
    obs_t = vision_obs(env, encoder)

    frames = 0
    episodes = 0
    start_time = time.perf_counter()
    last_time = start_time
    print(
        f"[INFO] preview vision student model={path} choice={vision_choice_name()} mode={train_mode()} "
        f"deterministic=1 camera=1 dt={env.step_dt:.3f}"
    )

    while simulation_app.is_running():
        with torch.no_grad():
            action = model.predict(obs_t)
        _obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated | truncated
        obs_t = vision_obs(env, encoder)
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
                f"head={head:.2f} body_ang={body_angle:.2f}"
            )
            frames = 0
            last_time = now

        if float(args_cli.duration) > 0.0 and now - start_time >= float(args_cli.duration):
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
