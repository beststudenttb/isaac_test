"""Record old CV outputs during one teacher rollout.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/compare_cv_xd.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record old CV outputs during one teacher rollout.")
parser.add_argument("--teacher", type=str, default="models/rl/teacher_ppo/best_val.zip")
parser.add_argument("--out", type=Path, default=Path("old_cv_features.csv"))
parser.add_argument("--episode-s", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--render-mode", default="balanced")
parser.add_argument("--aa", default="DLSS")
parser.add_argument("--dlss-mode", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cv_extractor.simple_cnn import OldBallVisionNet
from src.sb3_env import BallPPOEnvCfg, make_sb3_env


OLD_CV_PATH = Path("models/vision/cv_old/best.pt")
FEATURE_DIM = 128
SHARED_DIM = 516


def fields() -> list[str]:
    return [
        "step",
        "priv_px_x",
        "priv_dist",
        "a_x",
        "a_y",
        "a_w",
        "reward",
        "done",
        "success",
        "fail",
        "timeout",
        "old_x",
        "old_px_x",
        "old_dist",
    ] + [f"feature_{i:03d}" for i in range(FEATURE_DIM)] + [f"shared_{i:03d}" for i in range(SHARED_DIM)]


def make_env():
    cfg = BallPPOEnvCfg()
    cfg.seed = int(args_cli.seed)
    cfg.episode_length_s = float(args_cli.episode_s)
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 16.0
    cfg.sim.device = args_cli.device
    cfg.sim.render = sim_utils.RenderCfg(
        rendering_mode=str(args_cli.render_mode),
        antialiasing_mode=str(args_cli.aa),
        dlss_mode=int(args_cli.dlss_mode),
        enable_dlssg=False,
    )
    cfg.use_camera = True
    cfg.read_camera = True
    cfg.num_rerenders_on_reset = 1
    return make_sb3_env(cfg, fast_variant=True)


def load_old_cv(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(OLD_CV_PATH, map_location=device)
    model = OldBallVisionNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def cv_out(model: torch.nn.Module, image: torch.Tensor, cx: float) -> dict[str, float | list[float]]:
    if image.shape[-1] > 3:
        image = image[..., :3]
    with torch.no_grad():
        pred = model(image)
    x_norm = float(pred["x_pred"][0, 0].detach().cpu())
    dist = float(pred["distance_pred"][0, 0].detach().cpu())
    feature = pred["reserve_feature"][0].detach().float().cpu().tolist()
    shared = pred["shared_feature"][0].detach().float().cpu().tolist()
    return {
        "old_x": x_norm,
        "old_px_x": (x_norm + 1.0) * cx,
        "old_dist": dist,
        "feature": feature,
        "shared": shared,
    }


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def main():
    teacher_path = Path(args_cli.teacher)
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher model not found: {teacher_path}")

    env = make_env()
    base_env = env.unwrapped
    device = torch.device(base_env.device)
    teacher = PPO.load(str(teacher_path), env=env, device=args_cli.device)
    old_cv = load_old_cv(device)
    cx = float(base_env.cfg.image_width) * 0.5

    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    csv_file = args_cli.out.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fields())
    writer.writeheader()

    steps = int(base_env.max_episode_length)
    start = time.perf_counter()
    total_steps = 0

    obs = env.reset()
    for step in range(steps):
        action, _ = teacher.predict(obs, deterministic=True)
        obs, reward, done, _info = env.step(action)
        image = base_env.camera.data.output["rgb"]

        priv_px = float(base_env.last_px_x[0].detach().cpu())
        priv_dist = float(base_env.last_dist[0].detach().cpu())
        out = cv_out(old_cv, image, cx)
        move_action = base_env.last_move_actions[0].detach().cpu().tolist()

        row = {
            "step": step,
            "priv_px_x": fmt(priv_px),
            "priv_dist": fmt(priv_dist),
            "a_x": fmt(move_action[0]),
            "a_y": fmt(move_action[1]),
            "a_w": fmt(move_action[2]),
            "reward": fmt(float(reward[0])),
            "done": int(bool(done[0])),
            "success": int(bool(base_env.last_success[0].item())),
            "fail": int(bool(base_env.last_fail[0].item())),
            "timeout": int(bool(base_env.last_timeout[0].item())),
            "old_x": fmt(float(out["old_x"])),
            "old_px_x": fmt(float(out["old_px_x"])),
            "old_dist": fmt(float(out["old_dist"])),
        }
        for i, value in enumerate(out["feature"]):
            row[f"feature_{i:03d}"] = fmt(value)
        for i, value in enumerate(out["shared"]):
            row[f"shared_{i:03d}"] = fmt(value)
        writer.writerow(row)
        total_steps += 1

        if bool(done[0]):
            break

    csv_file.close()
    elapsed = time.perf_counter() - start
    print(f"[INFO] wrote {args_cli.out} steps={total_steps} fps={total_steps / elapsed:.2f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
