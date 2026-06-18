"""Offline validate vision student update checkpoints.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p train/val_student_vision.py --cv-model old --state feature --student
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

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


parser = argparse.ArgumentParser(description="Offline validate vision student update checkpoints.")
parser.add_argument("--state", choices=["xd", "shared", "feature"], required=True)
parser.add_argument("--cv-model", choices=["old", "old-m", "resnet", "mobile"], required=True)
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument("--student", action="store_true")
mode_group.add_argument("--teacher", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

cli_model = CV_MODEL_ALIAS[str(args_cli.cv_model)]
cli_state = STATE_ALIAS[str(args_cli.state)]
if not any(str(item["model"]) == cli_model and str(item["state"]) == cli_state for item in cfg.VISION_CHOICES.values()):
    raise ValueError(f"state={args_cli.state} is not available for cv-model={args_cli.cv_model}")

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
args_cli.headless = True
args_cli.livestream = 0
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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

VAL_FIELDS = [
    "checkpoint",
    "update",
    "step",
    "success_rate",
    "fail_rate",
    "timeout_rate",
    "mean_return",
    "mean_len",
]

TRAJ_FIELDS = [
    "phase",
    "step",
    "update",
    "t",
    "env_id",
    "px_x",
    "dist",
    "a_x",
    "a_y",
    "a_w",
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


def train_mode() -> str:
    if args_cli.student:
        return "student"
    return "teacher"


def vision_choice() -> dict:
    model = CV_MODEL_ALIAS[str(args_cli.cv_model)]
    state = STATE_ALIAS[str(args_cli.state)]
    for choice in cfg.VISION_CHOICES.values():
        if str(choice["model"]) == model and str(choice["state"]) == state:
            return choice
    raise ValueError(f"state={args_cli.state} is not available for cv-model={args_cli.cv_model}")


def run_name() -> str:
    return f"{args_cli.cv_model}_{args_cli.state}_{train_mode()}"


def run_dir() -> Path:
    return Path(cfg.OUT_DIR) / run_name()


def make_env() -> BallPPOEnv:
    env_cfg = BallPPOEnvCfg()
    env_cfg.seed = int(cfg.SEED)
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.scene.num_envs = int(cfg.NUM_ENVS)
    env_cfg.sim.device = args_cli.device
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
        return torch.cat([out["x_pred"], out["distance_pred"]], dim=1)
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
        raise RuntimeError("vision val requires env camera")
    return encode_image(env.camera.data.output["rgb"], encoder)


def load_model(path: Path, device: torch.device) -> tuple[ActorCritic, dict]:
    choice = vision_choice()
    ckpt = torch.load(path, map_location=device)
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
    return model, ckpt


def traj_row(
    env: BallPPOEnv,
    phase: str,
    step: int,
    update: int,
    t: int,
    env_id: int,
    reward: torch.Tensor,
    done: bool,
) -> dict:
    action = env.last_move_actions[env_id].detach().cpu().numpy()
    robot_xy = env.last_robot_xy[env_id].detach().cpu().numpy()
    target_xy = env.last_target_xy[env_id].detach().cpu().numpy()
    return {
        "phase": phase,
        "step": int(step),
        "update": int(update),
        "t": int(t),
        "env_id": int(env_id),
        "px_x": f"{float(env.last_px_x[env_id].item()):.4f}",
        "dist": f"{float(env.last_dist[env_id].item()):.4f}",
        "a_x": f"{float(action[0]):.4f}",
        "a_y": f"{float(action[1]):.4f}",
        "a_w": f"{float(action[2]):.4f}",
        "reward": f"{float(reward.item()):.4f}",
        "done": int(done),
        "success": int(bool(env.last_success[env_id].item())),
        "fail": int(bool(env.last_fail[env_id].item())),
        "timeout": int(bool(env.last_timeout[env_id].item())),
        "robot_x": f"{float(robot_xy[0]):.4f}",
        "robot_y": f"{float(robot_xy[1]):.4f}",
        "head_yaw": f"{float(env.last_head_yaw[env_id].item()):.4f}",
        "target_x": f"{float(target_xy[0]):.4f}",
        "target_y": f"{float(target_xy[1]):.4f}",
    }


def val(
    env: BallPPOEnv,
    encoder: nn.Module,
    model: ActorCritic,
    update: int = -1,
    step: int = -1,
    traj_writer: csv.DictWriter | None = None,
) -> dict:
    env.reset()
    obs_t = vision_obs(env, encoder)
    steps = int(cfg.VAL_STEPS) if int(cfg.VAL_STEPS) > 0 else int(env.max_episode_length)
    done_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros(env.num_envs, device=env.device)
    success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    fail = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    timeout = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    with torch.no_grad():
        for t in range(steps):
            action = model.predict(obs_t)
            _obs, reward, terminated, truncated, _info = env.step(action)
            active = ~done_once
            returns[active] += reward[active]
            lengths[active] += 1
            done = (terminated | truncated) & active
            if traj_writer is not None:
                for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    traj_writer.writerow(
                        traj_row(env, "offline_val", step, update, t, int(env_id), reward[int(env_id)], bool(done[int(env_id)]))
                    )
            if torch.any(done):
                success[done] = env.last_success[done]
                fail[done] = env.last_fail[done]
                timeout[done] = env.last_timeout[done] | truncated[done]
                done_once[done] = True
                if bool(done_once.all()):
                    break
            obs_t = vision_obs(env, encoder)

    timeout[~done_once] = True
    lengths[~done_once] = steps
    return {
        "success_rate": float(success.float().mean().cpu()),
        "fail_rate": float(fail.float().mean().cpu()),
        "timeout_rate": float(timeout.float().mean().cpu()),
        "mean_return": float(returns.mean().cpu()),
        "mean_len": float(lengths.mean().cpu()),
    }


def best_key(row: dict) -> tuple:
    return (
        float(row["success_rate"]),
        -float(row["fail_rate"]),
        -float(row["timeout_rate"]),
        float(row["mean_return"]),
    )


def main():
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = run_dir()
    update_dir = root / "updates"
    if not update_dir.exists():
        raise FileNotFoundError(f"update dir not found: {update_dir}")
    update_paths = sorted(update_dir.glob("update_*.pt"))
    if not update_paths:
        raise FileNotFoundError(f"no update checkpoints found: {update_dir}")

    env = make_env()
    device = torch.device(env.device)
    encoder = load_encoder(device)
    out_csv = root / "offline_val.csv"
    best_row = None
    best_path = None

    print(f"[INFO] offline val run={run_name()} updates={len(update_paths)} envs={env.num_envs} device={env.device}")
    try:
        with out_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=VAL_FIELDS)
            writer.writeheader()
            for path in update_paths:
                model, ckpt = load_model(path, device)
                info = val(env, encoder, model)
                row = {
                    "checkpoint": path.name,
                    "update": int(ckpt.get("update", -1)),
                    "step": int(ckpt.get("step", -1)),
                    **info,
                }
                writer.writerow(row)
                file.flush()
                if best_row is None or best_key(row) > best_key(best_row):
                    best_row = row
                    best_path = path
                print(
                    f"update={row['update']} step={row['step']} "
                    f"succ={row['success_rate']:.3f} fail={row['fail_rate']:.3f} "
                    f"timeout={row['timeout_rate']:.3f} return={row['mean_return']:.2f}"
                )

        if best_row is None or best_path is None:
            raise RuntimeError("offline val did not produce best checkpoint")
        shutil.copy2(best_path, root / "best_offline.pt")
        (root / "best_offline_info.txt").write_text(
            "\n".join(f"{key} = {value}" for key, value in best_row.items()) + "\n",
            encoding="utf-8",
        )
        best_model, _best_ckpt = load_model(best_path, device)
        with (root / "traj_best_offline.csv").open("w", newline="", encoding="utf-8") as traj_file:
            traj_writer = csv.DictWriter(traj_file, fieldnames=TRAJ_FIELDS)
            traj_writer.writeheader()
            val(
                env,
                encoder,
                best_model,
                int(best_row["update"]),
                int(best_row["step"]),
                traj_writer,
            )
        print(
            f"[INFO] best checkpoint={best_path.name} success={best_row['success_rate']:.3f} "
            f"fail={best_row['fail_rate']:.3f} timeout={best_row['timeout_rate']:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
