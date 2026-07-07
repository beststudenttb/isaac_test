"""Offline validate vision student update checkpoints.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p val/student_vision.py --cv-model old --state feature --student
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
parser.add_argument("--cv-ckpt", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--num-episodes", type=int, default=int(cfg.NUM_EPISODES))
parser.add_argument("--start", type=int, default=int(cfg.START))
parser.add_argument("--stride", type=int, default=int(cfg.STRIDE))
parser.add_argument("--random-stop", action="store_true")
parser.add_argument("--noise", action="store_true")
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

from src import task_cfg
from src.cv_extractor.simple_cnn import BallVisionNet, MobileBallNet, OldBallVisionNet, OldMobileBallNet
from src.noise_env import NoisePPOEnv, NoisePPOEnvCfg
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
    "num_envs",
    "num_episodes",
    "episodes",
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
    "end_d",
    "end_x",
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
            selected = dict(choice)
            if args_cli.cv_ckpt is not None:
                selected["ckpt"] = args_cli.cv_ckpt
            return selected
    raise ValueError(f"state={args_cli.state} is not available for cv-model={args_cli.cv_model}")


def run_name() -> str:
    return f"{args_cli.cv_model}_{args_cli.state}_{train_mode()}"


def run_dir() -> Path:
    root = Path(cfg.RANDOM_STOP_OUT_DIR) if args_cli.random_stop else Path(cfg.OUT_DIR)
    return root / run_name()


def end_range() -> tuple[float, float, float, float]:
    if args_cli.random_stop:
        return (
            float(cfg.RANDOM_END_D_MIN),
            float(cfg.RANDOM_END_D_MAX),
            float(cfg.RANDOM_END_X_MIN),
            float(cfg.RANDOM_END_X_MAX),
        )
    return (
        float(cfg.END_D_MIN),
        float(cfg.END_D_MAX),
        float(cfg.END_X_MIN),
        float(cfg.END_X_MAX),
    )


def make_env() -> BallPPOEnv:
    end_d_min, end_d_max, end_x_min, end_x_max = end_range()
    env_cfg = NoisePPOEnvCfg() if args_cli.noise else BallPPOEnvCfg()
    env_cfg.seed = int(cfg.SEED)
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else int(cfg.NUM_ENVS)
    env_cfg.sim.device = args_cli.device
    env_cfg.use_camera = True
    env_cfg.read_camera = True
    env_cfg.num_rerenders_on_reset = int(cfg.RERENDER_ON_RESET)
    env_cfg.end_d_min = end_d_min
    env_cfg.end_d_max = end_d_max
    env_cfg.end_x_min = end_x_min
    env_cfg.end_x_max = end_x_max
    env_cls = NoisePPOEnv if args_cli.noise else BallPPOEnv
    return env_cls(env_cfg)


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


def policy_obs(feature: torch.Tensor, end_obs: torch.Tensor) -> torch.Tensor:
    return torch.cat((feature, end_obs), dim=-1)


def vision_obs(env: BallPPOEnv, encoder: nn.Module) -> torch.Tensor:
    if env.camera is None:
        raise RuntimeError("vision val requires env camera")
    return policy_obs(encode_image(env.camera.data.output["rgb"], encoder), env.end_obs())


def load_model(path: Path, device: torch.device) -> tuple[ActorCritic, dict]:
    choice = vision_choice()
    ckpt = torch.load(path, map_location=device)
    model = ActorCritic(
        obs_dim=int(choice["dim"]) + 2,
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
    end_d = env.end_d[env_id].detach().cpu().item()
    end_x = env.end_x[env_id].detach().cpu().item()
    return {
        "phase": phase,
        "step": int(step),
        "update": int(update),
        "t": int(t),
        "env_id": int(env_id),
        "px_x": f"{float(env.last_px_x[env_id].item()):.4f}",
        "dist": f"{float(env.last_dist[env_id].item()):.4f}",
        "end_d": f"{float(end_d):.4f}",
        "end_x": f"{float(end_x):.4f}",
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


def fmt_value(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.3f}"
    return str(value)


def print_log(groups: list[tuple[str, list[tuple[str, object]]]]):
    rows = []
    for group, items in groups:
        rows.append((f"{group}/", ""))
        rows.extend((f"  {key}", fmt_value(value)) for key, value in items)
    width_l = max(len(left) for left, _ in rows)
    width_r = max(len(right) for _, right in rows)
    line = "-" * (width_l + width_r + 7)
    print(line)
    for left, right in rows:
        print(f"| {left:<{width_l}} | {right:>{width_r}} |")
    print(line)


def val(
    env: BallPPOEnv,
    encoder: nn.Module,
    model: ActorCritic,
    num_episodes: int,
    update: int = -1,
    step: int = -1,
    traj_writer: csv.DictWriter | None = None,
    env0_traj_writer: csv.DictWriter | None = None,
) -> dict:
    steps = int(cfg.VAL_STEPS) if int(cfg.VAL_STEPS) > 0 else int(env.max_episode_length)
    episodes = env.num_envs * int(num_episodes)
    success_count = 0
    fail_count = 0
    timeout_count = 0
    return_sum = 0.0
    len_sum = 0.0

    with torch.no_grad():
        for episode_id in range(int(num_episodes)):
            env.reset()
            obs_t = vision_obs(env, encoder)
            done_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            returns = torch.zeros(env.num_envs, device=env.device)
            lengths = torch.zeros(env.num_envs, device=env.device)
            success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            fail = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            timeout = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

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
                            traj_row(
                                env,
                                "offline_val",
                                step,
                                update,
                                episode_id * steps + t,
                                int(env_id),
                                reward[int(env_id)],
                                bool(done[int(env_id)]),
                            )
                        )
                if env0_traj_writer is not None and bool(active[0].item()):
                    env0_traj_writer.writerow(
                        traj_row(
                            env,
                            "offline_val_env0",
                            step,
                            update,
                            episode_id * steps + t,
                            0,
                            reward[0],
                            bool(done[0]),
                        )
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
            success_count += int(success.sum().item())
            fail_count += int(fail.sum().item())
            timeout_count += int(timeout.sum().item())
            return_sum += float(returns.sum().cpu())
            len_sum += float(lengths.sum().cpu())

    return {
        "success_rate": success_count / episodes,
        "fail_rate": fail_count / episodes,
        "timeout_rate": timeout_count / episodes,
        "mean_return": return_sum / episodes,
        "mean_len": len_sum / episodes,
    }


def best_key(row: dict) -> tuple:
    return (
        float(row["success_rate"]),
        -float(row["fail_rate"]),
        -float(row["timeout_rate"]),
        float(row["mean_return"]),
    )


def write_config(path: Path, env: BallPPOEnv):
    choice = vision_choice()
    lines = [
        f"run_name = {run_name()!r}",
        f"cv_model = {args_cli.cv_model!r}",
        f"state = {args_cli.state!r}",
        f"mode = {train_mode()!r}",
        f"num_envs = {env.num_envs!r}",
        f"num_episodes = {int(args_cli.num_episodes)!r}",
        f"start = {int(args_cli.start)!r}",
        f"stride = {int(args_cli.stride)!r}",
        f"random_stop = {bool(args_cli.random_stop)!r}",
        f"episodes_per_checkpoint = {env.num_envs * int(args_cli.num_episodes)!r}",
        f"val_steps = {int(cfg.VAL_STEPS)!r}",
        f"episode_s = {float(cfg.EPISODE_S)!r}",
        f"seed = {int(cfg.SEED)!r}",
        f"device = {env.device!r}",
        f"vision_model = {str(choice['model'])!r}",
        f"vision_state = {str(choice['state'])!r}",
        f"vision_ckpt = {str(choice['ckpt'])!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TrajBuffer:
    def __init__(self):
        self.rows = []

    def writerow(self, row: dict):
        self.rows.append(dict(row))


def write_traj(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAJ_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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
    start = int(args_cli.start)
    stride = int(args_cli.stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive: {stride}")
    update_paths = [
        path for path in update_paths
        if int(path.stem.split("_")[-1]) >= start
        and (int(path.stem.split("_")[-1]) - start) % stride == 0
    ]
    if not update_paths:
        raise FileNotFoundError(f"no update checkpoints found after start={start}, stride={stride}: {update_dir}")

    env = make_env()
    device = torch.device(env.device)
    encoder = load_encoder(device)
    write_config(root / "offline_val_config.py", env)
    out_csv = root / "offline_val.csv"
    env0_traj_csv = root / "traj_offline_env0.csv"
    best_row = None
    best_path = None

    print(
        f"[INFO] offline val run={run_name()} updates={len(update_paths)} "
        f"envs={env.num_envs} num_episodes={args_cli.num_episodes} start={args_cli.start} "
        f"stride={args_cli.stride} device={env.device}"
    )
    try:
        mode = "a" if int(args_cli.start) > 0 else "w"
        with out_csv.open(mode, newline="", encoding="utf-8") as file, env0_traj_csv.open(mode, newline="", encoding="utf-8") as env0_file:
            writer = csv.DictWriter(file, fieldnames=VAL_FIELDS)
            env0_writer = csv.DictWriter(env0_file, fieldnames=TRAJ_FIELDS)
            if mode == "w":
                writer.writeheader()
                env0_writer.writeheader()
            for path in update_paths:
                model, ckpt = load_model(path, device)
                update = int(ckpt.get("update", -1))
                step = int(ckpt.get("step", -1))
                traj_buf = TrajBuffer()
                info = val(
                    env,
                    encoder,
                    model,
                    int(args_cli.num_episodes),
                    update,
                    step,
                    traj_buf,
                    env0_traj_writer=env0_writer,
                )
                row = {
                    "checkpoint": path.name,
                    "update": update,
                    "step": step,
                    "num_envs": env.num_envs,
                    "num_episodes": int(args_cli.num_episodes),
                    "episodes": env.num_envs * int(args_cli.num_episodes),
                    **info,
                }
                writer.writerow(row)
                file.flush()
                env0_file.flush()
                if best_row is None or best_key(row) > best_key(best_row):
                    best_row = row
                    best_path = path
                    shutil.copy2(best_path, root / "best_offline.pt")
                    (root / "best_offline_info.txt").write_text(
                        "\n".join(f"{key} = {value}" for key, value in best_row.items()) + "\n",
                        encoding="utf-8",
                    )
                    write_traj(root / "traj_best_offline.csv", traj_buf.rows)
                print_log(
                    [
                        (
                            "model",
                            [
                                ("checkpoint", row["checkpoint"]),
                                ("update", row["update"]),
                                ("step", row["step"]),
                            ],
                        ),
                        (
                            "val",
                            [
                                ("episodes", row["episodes"]),
                                ("success", row["success_rate"]),
                                ("fail", row["fail_rate"]),
                                ("timeout", row["timeout_rate"]),
                                ("return", row["mean_return"]),
                                ("len", row["mean_len"]),
                            ],
                        ),
                    ]
                )

        if best_row is None or best_path is None:
            raise RuntimeError("offline val did not produce best checkpoint")
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
