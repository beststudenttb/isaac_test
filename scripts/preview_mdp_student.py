"""Preview one MDP student PPO policy.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/preview_mdp_student.py
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

import mdp_student_cfg as train_cfg


parser = argparse.ArgumentParser(description="Preview one deterministic MDP student policy.")
parser.add_argument("--model", type=str, default="421")
parser.add_argument("--mdp", type=str, default="auto")
parser.add_argument("--episode-s", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--duration", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if train_cfg.DEVICE is not None:
    args_cli.device = train_cfg.DEVICE
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import isaaclab.sim as sim_utils

from src.cv_extractor.mdp_state import MDPStateNet
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.ppo import ActorCritic


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


def update_num(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def policy_path() -> Path:
    root = Path(train_cfg.OUT_DIR)
    model = str(args_cli.model)
    if model.isdigit():
        return root / "updates" / f"update_{int(model):06d}.pt"
    if model == "best":
        return root / "best_offline.pt"
    if model == "last":
        return root / "last.pt"
    return Path(model)


def pick_mdp(policy_update: int, policy_ckpt: dict) -> Path:
    if str(args_cli.mdp) != "auto":
        return Path(args_cli.mdp)

    root = Path(train_cfg.OUT_DIR)
    mdp_paths = sorted((root / "mdp_updates").glob("update_*.pt"))
    valid = [path for path in mdp_paths if update_num(path) <= int(policy_update)]
    if valid:
        return valid[-1]
    base_path = Path(train_cfg.MDP_PATH)
    if base_path.exists():
        return base_path
    return Path(policy_ckpt["mdp_path"])


def make_env() -> MDPStudentEnv:
    env_cfg = MDPStudentEnvCfg()
    env_cfg.seed = int(train_cfg.SEED if args_cli.seed is None else args_cli.seed)
    env_cfg.episode_length_s = float(train_cfg.EPISODE_S if args_cli.episode_s is None else args_cli.episode_s)
    env_cfg.stop_n = int(train_cfg.STOP_N)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 16.0
    env_cfg.sim.device = args_cli.device
    render_kwargs = {
        "rendering_mode": str(train_cfg.RENDERING_MODE),
        "antialiasing_mode": str(train_cfg.ANTIALIASING_MODE),
        "enable_dlssg": False,
    }
    if train_cfg.DLSS_MODE is not None:
        render_kwargs["dlss_mode"] = int(train_cfg.DLSS_MODE)
    env_cfg.sim.render = sim_utils.RenderCfg(**render_kwargs)
    env_cfg.num_rerenders_on_reset = int(train_cfg.RERENDER_ON_RESET)
    env_cfg.end_d_min = float(train_cfg.END_D_MIN)
    env_cfg.end_d_max = float(train_cfg.END_D_MAX)
    env_cfg.end_x_min = float(train_cfg.END_X_MIN)
    env_cfg.end_x_max = float(train_cfg.END_X_MAX)
    return make_mdp_student_env(env_cfg)


def camera_rgb(env: MDPStudentEnv) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def load_mdp(path: Path, device: torch.device) -> MDPStateNet:
    ckpt = torch.load(path, map_location=device)
    model = MDPStateNet(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if bool(train_cfg.FREEZE_MDP_BACKBONE):
        for module in (model.stem, model.layer1, model.layer2, model.layer3):
            for param in module.parameters():
                param.requires_grad_(False)
    return model


def mdp_observe(
    mdp: MDPStateNet,
    image: torch.Tensor,
    prev_action: torch.Tensor,
    prev_state: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        out = mdp.observe(image, prev_action, prev_state)
    return {
        "belief": out["belief"].detach(),
        "z": out["z"].detach(),
        "state_feature": out["state_feature"].float().detach(),
    }


def next_mdp_state(
    mdp: MDPStateNet,
    env: MDPStudentEnv,
    prev_state: dict[str, torch.Tensor],
    action: torch.Tensor,
    done: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = {
        "belief": prev_state["belief"].clone(),
        "z": prev_state["z"].clone(),
    }
    prev_action = action.clone()
    if torch.any(done):
        state["belief"][done] = 0.0
        state["z"][done] = 0.0
        prev_action[done] = 0.0
    return mdp_observe(mdp, camera_rgb(env), prev_action, state)


def load_policy(path: Path, obs_dim: int, device: torch.device) -> tuple[ActorCritic, dict]:
    ckpt = torch.load(path, map_location=device)
    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=3,
        pi_hidden=list(train_cfg.POLICY_NET),
        vf_hidden=list(train_cfg.VALUE_NET),
        activation=ACTIVATIONS[str(train_cfg.ACTIVATION)],
        init_std=float(train_cfg.STD_INIT),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def open_camera_viewport():
    if bool(args_cli.headless):
        return
    import omni.ui as ui
    from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name

    camera_path = "/World/envs/env_0/Robot/head/eye/front_camera"
    viewport = get_viewport_from_window_name("Viewport")
    viewport.set_active_camera("/OmniverseKit_Persp")
    camera_window = create_viewport_window("MDP Student Camera", width=640, height=480)
    camera_window.viewport_api.set_active_camera(camera_path)
    main_window = ui.Workspace.get_window("Viewport")
    dock_window = ui.Workspace.get_window("MDP Student Camera")
    if main_window is not None and dock_window is not None:
        dock_position = getattr(ui.DockPosition, "RIGHT", ui.DockPosition.SAME)
        dock_window.dock_in(main_window, dock_position, 0.35)
        dock_window.focus()
    print(f"[INFO] Camera viewport uses {camera_path}")


def main():
    path = policy_path()
    if not path.exists():
        raise FileNotFoundError(f"policy not found: {path}")

    policy_ckpt = torch.load(path, map_location="cpu")
    policy_update = int(policy_ckpt["update"]) if "update" in policy_ckpt else update_num(path)
    mdp_path = pick_mdp(policy_update, policy_ckpt)
    if not mdp_path.exists():
        raise FileNotFoundError(f"MDP checkpoint not found: {mdp_path}")

    env = make_env()
    device = torch.device(env.device)
    mdp = load_mdp(mdp_path, device)
    model, _ = load_policy(path, int(mdp.belief_dim + mdp.z_dim + 2), device)

    env.sim.set_camera_view(eye=[8.0, -10.0, 6.0], target=[0.0, 0.0, 0.5])
    open_camera_viewport()

    env.reset()
    zero_action = torch.zeros(env.num_envs, int(env.cfg.action_space), device=env.device)
    mdp_state = mdp_observe(mdp, camera_rgb(env), zero_action, None)
    obs_t = env.mdp_policy_obs(mdp_state)

    frames = 0
    episodes = 0
    start_time = time.perf_counter()
    last_time = start_time
    print(
        f"[INFO] preview MDP student policy={path} mdp={mdp_path} "
        f"update={policy_update} deterministic=1 camera=1 dt={env.step_dt:.3f}"
    )

    while simulation_app.is_running():
        with torch.no_grad():
            action = model.predict(obs_t)
        _obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated | truncated
        mdp_state = next_mdp_state(mdp, env, mdp_state, action, done)
        obs_t = env.mdp_policy_obs(mdp_state)
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
