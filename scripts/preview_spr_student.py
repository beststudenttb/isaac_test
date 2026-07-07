"""Preview one SPR student policy and record one replayable trajectory.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/preview_spr_student.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "val"))

import spr_student_cfg as cfg


parser = argparse.ArgumentParser(description="Preview one deterministic SPR student policy.")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--episode-s", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--duration", type=float, default=0.0)
parser.add_argument("--csv", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import isaaclab.sim as sim_utils
from stable_baselines3 import PPO

from src import task_cfg
from src.cv_extractor.spr_state import SPRStateNet
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.spr_ppo import SPRActorCritic


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

FIELDS = [
    "episode",
    "t",
    "update",
    "step",
    "px_x",
    "dist",
    "end_d",
    "end_x",
    "raw_a_x",
    "raw_a_y",
    "raw_a_w",
    "move_a_x",
    "move_a_y",
    "move_a_w",
    "teacher_a_x",
    "teacher_a_y",
    "teacher_a_w",
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


def update_num(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def root_dir() -> Path:
    return Path(cfg.OUT_DIR)


def policy_path() -> Path:
    root = root_dir()
    model = str(args_cli.model)
    if model == "best":
        return root / "best_offline.pt"
    if model == "last":
        return root / "last.pt"
    if model.isdigit():
        update = int(model)
        ppo_path = root / "updates" / f"ppo_{update:06d}.pt"
        if ppo_path.exists():
            return ppo_path
        return root / "updates" / f"full_{update:06d}.pt"
    return Path(model)


def csv_path() -> Path:
    if args_cli.csv:
        return Path(args_cli.csv)
    return root_dir() / "preview_best.csv"


def make_env() -> MDPStudentEnv:
    env_cfg = MDPStudentEnvCfg()
    env_cfg.seed = int(cfg.SEED if args_cli.seed is None else args_cli.seed)
    env_cfg.episode_length_s = float(cfg.EPISODE_S if args_cli.episode_s is None else args_cli.episode_s)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 16.0
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
    env_cfg.end_d_min = float(cfg.END_D_MIN)
    env_cfg.end_d_max = float(cfg.END_D_MAX)
    env_cfg.end_x_min = float(cfg.END_X_MIN)
    env_cfg.end_x_max = float(cfg.END_X_MAX)
    return make_mdp_student_env(env_cfg)


def camera_rgb(env: MDPStudentEnv) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def build_model(base_ckpt: dict, device: torch.device) -> SPRActorCritic:
    spr = SPRStateNet(**base_ckpt["spr_cfg"]).to(device)
    model = SPRActorCritic(
        spr=spr,
        goal_dim=2,
        act_dim=3,
        pi_hidden=list(cfg.POLICY_NET),
        vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)],
        init_std=float(cfg.STD_INIT),
    ).to(device)
    model.load_state_dict(base_ckpt["model"])
    model.eval()
    return model


def load_policy(path: Path, device: torch.device) -> tuple[SPRActorCritic, dict, Path]:
    ckpt = torch.load(path, map_location=device)
    if "actor" in ckpt:
        spr_path = root_dir() / str(cfg.SPR_CKPT)
        base_ckpt = torch.load(spr_path, map_location=device)
        model = build_model(base_ckpt, device)
        model.actor.load_state_dict(ckpt["actor"])
        model.critic.load_state_dict(ckpt["critic"])
        model.log_std.data.copy_(ckpt["log_std"].to(device))
        model.eval()
        return model, ckpt, spr_path
    model = build_model(ckpt, device)
    return model, ckpt, path


def load_teacher(device: torch.device):
    path = Path("./models/rl/teacher_ppo/best_val.zip")
    if not path.exists():
        raise FileNotFoundError(f"teacher model not found: {path}")
    return PPO.load(str(path), device=device)


def teacher_action(teacher, priv_obs: torch.Tensor) -> torch.Tensor:
    actions, _ = teacher.predict(priv_obs.detach().cpu().numpy(), deterministic=True)
    actions = torch.as_tensor(actions, device=priv_obs.device, dtype=priv_obs.dtype)
    x_ok = torch.abs(priv_obs[:, 0] - priv_obs[:, 3]) <= float(task_cfg.STOP_X_TOL) / (
        float(task_cfg.IMAGE_WIDTH) * 0.5
    )
    d_ok = torch.abs(priv_obs[:, 1] - priv_obs[:, 2]) <= float(task_cfg.STOP_D_TOL) / float(task_cfg.FAIL_FAR)
    actions[x_ok & d_ok] = 0.0
    return actions


def open_camera_viewport():
    if bool(args_cli.headless):
        return
    import omni.ui as ui
    from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name

    camera_path = "/World/envs/env_0/Robot/head/eye/front_camera"
    viewport = get_viewport_from_window_name("Viewport")
    viewport.set_active_camera("/OmniverseKit_Persp")
    camera_window = create_viewport_window("SPR Student Camera", width=640, height=480)
    camera_window.viewport_api.set_active_camera(camera_path)
    main_window = ui.Workspace.get_window("Viewport")
    dock_window = ui.Workspace.get_window("SPR Student Camera")
    if main_window is not None and dock_window is not None:
        dock_position = getattr(ui.DockPosition, "RIGHT", ui.DockPosition.SAME)
        dock_window.dock_in(main_window, dock_position, 0.35)
        dock_window.focus()
    print(f"[INFO] Camera viewport uses {camera_path}")


def row(
    env: MDPStudentEnv,
    episode: int,
    t: int,
    update: int,
    step: int,
    raw_action: torch.Tensor,
    teacher_a: torch.Tensor,
    reward: torch.Tensor,
    done: bool,
) -> dict:
    move = env.last_move_actions[0].detach().cpu().numpy()
    raw = raw_action[0].detach().cpu().numpy()
    teacher = teacher_a[0].detach().cpu().numpy()
    robot = env.last_robot_xy[0].detach().cpu().numpy()
    target = env.last_target_xy[0].detach().cpu().numpy()
    return {
        "episode": int(episode),
        "t": int(t),
        "update": int(update),
        "step": int(step),
        "px_x": f"{float(env.last_px_x[0].item()):.4f}",
        "dist": f"{float(env.last_dist[0].item()):.4f}",
        "end_d": f"{float(env.end_d[0].item()):.4f}",
        "end_x": f"{float(env.end_x[0].item()):.4f}",
        "raw_a_x": f"{float(raw[0]):.4f}",
        "raw_a_y": f"{float(raw[1]):.4f}",
        "raw_a_w": f"{float(raw[2]):.4f}",
        "move_a_x": f"{float(move[0]):.4f}",
        "move_a_y": f"{float(move[1]):.4f}",
        "move_a_w": f"{float(move[2]):.4f}",
        "teacher_a_x": f"{float(teacher[0]):.4f}",
        "teacher_a_y": f"{float(teacher[1]):.4f}",
        "teacher_a_w": f"{float(teacher[2]):.4f}",
        "reward": f"{float(reward[0].item()):.4f}",
        "done": int(done),
        "success": int(bool(env.last_success[0].item())),
        "fail": int(bool(env.last_fail[0].item())),
        "timeout": int(bool(env.last_timeout[0].item())),
        "robot_x": f"{float(robot[0]):.4f}",
        "robot_y": f"{float(robot[1]):.4f}",
        "head_yaw": f"{float(env.last_head_yaw[0].item()):.4f}",
        "target_x": f"{float(target[0]):.4f}",
        "target_y": f"{float(target[1]):.4f}",
    }


def main():
    path = policy_path()
    if not path.exists():
        raise FileNotFoundError(f"policy not found: {path}")

    env = make_env()
    device = torch.device(env.device)
    model, ckpt, spr_path = load_policy(path, device)
    teacher = load_teacher(device)
    update = int(ckpt.get("update", update_num(path) if path.name != "best_offline.pt" else -1))
    step = int(ckpt.get("step", -1))

    env.sim.set_camera_view(eye=[8.0, -10.0, 6.0], target=[0.0, 0.0, 0.5])
    open_camera_viewport()

    out_csv = csv_path()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset()
    priv_t = obs["policy"]
    image = camera_rgb(env)
    goal = env.end_obs()

    frames = 0
    episode = 0
    t = 0
    start_time = time.perf_counter()
    last_time = start_time
    print(
        f"[INFO] preview SPR student policy={path} spr={spr_path} update={update} "
        f"deterministic=1 camera=1 csv={out_csv} dt={env.step_dt:.3f}"
    )

    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        while simulation_app.is_running():
            with torch.no_grad():
                action = model.predict(image, goal)
                teacher_a = teacher_action(teacher, priv_t)
            obs_next, reward, terminated, truncated, _info = env.step(action)
            done = bool((terminated | truncated)[0].item())
            writer.writerow(row(env, episode, t, update, step, action, teacher_a, reward, done))
            file.flush()
            frames += 1
            t += 1

            if done:
                episode += 1
                print(
                    f"[INFO] done episode={episode} reward={float(reward[0].item()):.3f} "
                    f"success={int(env.last_success[0].item())} "
                    f"fail={int(env.last_fail[0].item())} "
                    f"timeout={int(env.last_timeout[0].item())}"
                )
                obs_next, _ = env.reset()
                t = 0

            priv_t = obs_next["policy"]
            image = camera_rgb(env)
            goal = env.end_obs()

            now = time.perf_counter()
            if now - last_time >= 1.0:
                fps = frames / (now - last_time)
                rtf = fps * env.step_dt
                raw = action[0].detach().cpu().numpy()
                move = env.last_move_actions[0].detach().cpu().numpy()
                teacher_np = teacher_a[0].detach().cpu().numpy()
                robot = env.last_robot_xy[0].detach().cpu().numpy()
                target = env.last_target_xy[0].detach().cpu().numpy()
                delta = target - robot
                body_angle = float(np.arctan2(delta[1], delta[0]))
                head = float(env.last_head_yaw[0].item())
                print(
                    f"fps={fps:.1f} rtf={rtf:.2f} "
                    f"px={float(env.last_px_x[0].item()):.1f} "
                    f"d={float(env.last_dist[0].item()):.2f} "
                    f"end=({float(env.end_d[0].item()):.2f},{float(env.end_x[0].item()):.1f}) "
                    f"raw=({raw[0]:.2f},{raw[1]:.2f},{raw[2]:.2f}) "
                    f"move=({move[0]:.2f},{move[1]:.2f},{move[2]:.2f}) "
                    f"teacher=({teacher_np[0]:.2f},{teacher_np[1]:.2f},{teacher_np[2]:.2f}) "
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
    try:
        main()
    finally:
        simulation_app.close()
