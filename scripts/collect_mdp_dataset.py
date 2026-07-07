"""Collect teacher rollouts for MDP state learning.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/collect_mdp_dataset.py --num-envs 64 --transitions 50000
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("./data_mdp")
NOISE_OUT_DIR = Path("./data_mdp_noise")
TEACHER_PATH = Path("./models/rl/teacher_ppo/best_val.zip")


parser = argparse.ArgumentParser(description="Collect image transition dataset with teacher PPO.")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--transitions", type=int, default=50000)
parser.add_argument("--out-dir", type=Path, default=None)
parser.add_argument("--teacher", type=Path, default=TEACHER_PATH)
parser.add_argument("--episode-s", type=float, default=15.0)
parser.add_argument("--stop-n", type=int, default=3)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--render-mode", default="balanced")
parser.add_argument("--aa", default="Off")
parser.add_argument("--dlss-mode", type=int, default=None)
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--noise", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.out_dir is None:
    args_cli.out_dir = NOISE_OUT_DIR if args_cli.noise else OUT_DIR

args_cli.headless = True
args_cli.livestream = 0
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(PROJECT_ROOT))

import torch
from stable_baselines3 import PPO
from torchvision.io import write_png

import isaaclab.sim as sim_utils

from src.sb3_env import BallPPOEnv, BallPPOEnvCfg
from src.noise_env import NoiseMixin, NoisePPOEnvCfg


CSV_FIELDS = [
    "index",
    "episode",
    "t",
    "env_id",
    "img",
    "next_img",
    "px_x",
    "dist",
    "next_px_x",
    "next_dist",
    "a_x",
    "a_y",
    "a_w",
    "reward",
    "terminated",
    "truncated",
    "done",
    "success",
    "fail",
    "timeout",
]


class DatasetBallPPOEnv(BallPPOEnv):
    def __init__(self, cfg: BallPPOEnvCfg):
        super().__init__(cfg)
        self.terminal_rgb = None

    def compute_reward(self) -> torch.Tensor:
        reward = super().compute_reward()
        done = self.reset_terminated | self.reset_time_outs
        if torch.any(done):
            image = rgb_images(self)
            if self.terminal_rgb is None or self.terminal_rgb.shape != image.shape:
                self.terminal_rgb = torch.empty_like(image)
            self.terminal_rgb[done] = image[done]
        return reward


class DatasetNoisePPOEnv(NoiseMixin, DatasetBallPPOEnv):
    def __init__(self, cfg: NoisePPOEnvCfg):
        super().__init__(cfg)
        self.init_noise()


def make_env() -> DatasetBallPPOEnv:
    cfg = NoisePPOEnvCfg() if args_cli.noise else BallPPOEnvCfg()
    cfg.seed = int(args_cli.seed)
    cfg.episode_length_s = float(args_cli.episode_s)
    cfg.stop_n = int(args_cli.stop_n)
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = 16.0
    cfg.sim.device = args_cli.device
    render_kwargs = {
        "rendering_mode": str(args_cli.render_mode),
        "antialiasing_mode": args_cli.aa,
        "enable_dlssg": False,
    }
    if args_cli.dlss_mode is not None:
        render_kwargs["dlss_mode"] = args_cli.dlss_mode
    cfg.sim.render = sim_utils.RenderCfg(**render_kwargs)
    cfg.use_camera = True
    cfg.read_camera = True
    cfg.num_rerenders_on_reset = 1
    env_cls = DatasetNoisePPOEnv if args_cli.noise else DatasetBallPPOEnv
    return env_cls(cfg)


def rgb_images(env: BallPPOEnv) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def raw_label(env: BallPPOEnv) -> tuple[torch.Tensor, torch.Tensor]:
    label = env.project_target()
    return label["px_x"].clone(), label["dist"].clone()


def save_image(path: Path, image: torch.Tensor) -> None:
    write_png(image.permute(2, 0, 1).cpu(), str(path))


def frame_name(index: int) -> str:
    return f"{index:08d}.png"


def clear_output(out_dir: Path) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"output dir already exists: {out_dir}")
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True)
    return img_dir


def main() -> None:
    if not args_cli.teacher.exists():
        raise FileNotFoundError(f"teacher not found: {args_cli.teacher}")

    out_dir = Path(args_cli.out_dir)
    img_dir = clear_output(out_dir)
    csv_path = out_dir / "transitions.csv"
    config_path = out_dir / "config.txt"

    env = make_env()
    model = PPO.load(str(args_cli.teacher), device=args_cli.device)
    obs, _ = env.reset()
    obs_t = obs["policy"]

    episode_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    episode_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    frame_index = 0
    current_img = []
    image_t = rgb_images(env).clone()
    current_px_x, current_dist = raw_label(env)
    for env_id in range(env.num_envs):
        name = frame_name(frame_index)
        save_image(img_dir / name, image_t[env_id])
        current_img.append(name)
        frame_index += 1

    saved = 0
    sim_steps = 0
    last_saved = 0
    last_time = time.perf_counter()

    config_path.write_text(
        "\n".join(
            [
                f"teacher = {args_cli.teacher}",
                f"num_envs = {env.num_envs}",
                f"transitions = {int(args_cli.transitions)}",
                f"episode_s = {float(args_cli.episode_s)}",
                f"stop_n = {int(args_cli.stop_n)}",
                f"seed = {int(args_cli.seed)}",
                f"render_mode = {args_cli.render_mode}",
                f"aa = {args_cli.aa}",
                f"dlss_mode = {args_cli.dlss_mode}",
                f"deterministic = {bool(args_cli.deterministic)}",
                f"noise = {bool(args_cli.noise)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[INFO] collect mdp out={out_dir} envs={env.num_envs} transitions={int(args_cli.transitions)} "
        f"teacher={args_cli.teacher} deterministic={int(bool(args_cli.deterministic))} "
        f"noise={int(bool(args_cli.noise))}"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while simulation_app.is_running() and saved < int(args_cli.transitions):
            action_np, _ = model.predict(obs_t.detach().cpu().numpy(), deterministic=bool(args_cli.deterministic))
            action = torch.as_tensor(action_np, device=env.device, dtype=torch.float32)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            image_next = rgb_images(env).clone()
            next_px_x = env.last_px_x.clone()
            next_dist = env.last_dist.clone()
            done = terminated | truncated

            count = min(env.num_envs, int(args_cli.transitions) - saved)
            next_img_names = [None] * env.num_envs
            for env_id in range(count):
                if bool(done[env_id].item()):
                    next_name = frame_name(frame_index)
                    save_image(img_dir / next_name, env.terminal_rgb[env_id])
                    frame_index += 1
                else:
                    next_name = frame_name(frame_index)
                    save_image(img_dir / next_name, image_next[env_id])
                    frame_index += 1
                next_img_names[env_id] = next_name
                writer.writerow(
                    {
                        "index": saved,
                        "episode": int(episode_ids[env_id].item()),
                        "t": int(episode_steps[env_id].item()),
                        "env_id": env_id,
                        "img": current_img[env_id],
                        "next_img": next_name,
                        "px_x": f"{float(current_px_x[env_id].item()):.6f}",
                        "dist": f"{float(current_dist[env_id].item()):.6f}",
                        "next_px_x": f"{float(next_px_x[env_id].item()):.6f}",
                        "next_dist": f"{float(next_dist[env_id].item()):.6f}",
                        "a_x": f"{float(action[env_id, 0].item()):.6f}",
                        "a_y": f"{float(action[env_id, 1].item()):.6f}",
                        "a_w": f"{float(action[env_id, 2].item()):.6f}",
                        "reward": f"{float(reward[env_id].item()):.6f}",
                        "terminated": int(bool(terminated[env_id].item())),
                        "truncated": int(bool(truncated[env_id].item())),
                        "done": int(bool(done[env_id].item())),
                        "success": int(bool(env.last_success[env_id].item())),
                        "fail": int(bool(env.last_fail[env_id].item())),
                        "timeout": int(bool((env.last_timeout[env_id] | truncated[env_id]).item())),
                    }
                )
                saved += 1

            for env_id in range(count):
                if bool(done[env_id].item()):
                    reset_name = frame_name(frame_index)
                    save_image(img_dir / reset_name, image_next[env_id])
                    current_img[env_id] = reset_name
                    frame_index += 1
                else:
                    current_img[env_id] = next_img_names[env_id]
                    current_px_x[env_id] = next_px_x[env_id]
                    current_dist[env_id] = next_dist[env_id]

            if torch.any(done):
                reset_px_x, reset_dist = raw_label(env)
                current_px_x[done] = reset_px_x[done]
                current_dist[done] = reset_dist[done]

            episode_steps += 1
            if torch.any(done):
                episode_ids[done] += 1
                episode_steps[done] = 0

            obs_t = obs_next["policy"]
            sim_steps += 1

            now = time.perf_counter()
            if now - last_time >= 1.0:
                save_fps = (saved - last_saved) / (now - last_time)
                sim_fps = sim_steps / (now - last_time)
                print(f"[INFO] saved={saved}/{int(args_cli.transitions)} save_fps={save_fps:.1f} sim_fps={sim_fps:.1f}")
                last_saved = saved
                sim_steps = 0
                last_time = now

    env.close()
    print(f"[INFO] collect mdp done saved={saved} csv={csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
