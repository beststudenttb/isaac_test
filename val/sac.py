"""SAC + HER 离线评估入口(与 val/offpolicy.py 结构对齐)。评估动作 = 确定性 tanh(mean)。

    ./IsaacLab/isaaclab.sh -p val/sac.py --cfg sac_cfg --obs-mode spr_z --noise
"""

from __future__ import annotations

import argparse
import csv
import importlib
import random
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SAC + HER offline val.")
parser.add_argument("--cfg", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--num-episodes", type=int, default=None)
parser.add_argument("--start", type=int, default=None)
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--obs-mode", type=str, default="spr_z", choices=["spr_z", "pixels"])
parser.add_argument("--random-stop", action="store_true")
parser.add_argument("--noise", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --cfg 指向的 cfg 模块在 train/(与 train/sac.py 同目录);val/sac.py 自身在 val/,
# 需把 train/ 加入 import 路径才能找到 sac_cfg(train 与 val 共用同一份 cfg)。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
cfg = importlib.import_module(args_cli.cfg)

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
args_cli.headless = True
args_cli.livestream = 0
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.sac_agent import SACAgent
from src.rl.env import make_env as build_env, camera_rgb
from src.rl.spr import load_spr
from src.rl.logging import VAL_FIELDS, TRAJ_FIELDS, traj_row, print_log


def arg(name: str, cfg_name: str):
    v = getattr(args_cli, name)
    return v if v is not None else getattr(cfg, cfg_name)


def root_dir() -> Path:
    base = Path(cfg.OUT_DIR)
    if args_cli.obs_mode == "pixels" and hasattr(cfg, "PIXELS_OUT_DIR"):
        base = Path(cfg.PIXELS_OUT_DIR)
    if args_cli.random_stop:
        base = base.with_name(base.name + str(cfg.RANDOM_STOP_SUFFIX))
    return base


def step_num(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_agent(path: Path, device: torch.device) -> SACAgent:
    ckpt = torch.load(path, map_location=device)
    if str(ckpt.get("obs_mode")) != str(args_cli.obs_mode):
        raise ValueError(f"checkpoint obs_mode={ckpt.get('obs_mode')} != cli {args_cli.obs_mode}: {path}")
    agent = SACAgent(**ckpt["agent_cfg"], device=device)
    agent.load_dict(ckpt["agent"])
    agent.actor.eval()
    agent.critic.eval()
    if agent.encoder is not None:
        agent.encoder.eval()
    return agent


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


def best_key(row: dict) -> tuple:
    return (float(row["success_rate"]), -float(row["fail_rate"]),
            -float(row["timeout_rate"]), float(row["mean_return"]))


def val(env, agent, spr, num_episodes, step, traj_writer, env0_traj_writer) -> dict:
    steps = int(cfg.VAL_STEPS) if int(cfg.VAL_STEPS) > 0 else int(env.max_episode_length)
    episodes = env.num_envs * int(num_episodes)
    sc = fc = tc = 0
    return_sum = len_sum = de_sum = xe_sum = 0.0
    cx = float(env.cfg.image_width) * 0.5

    with torch.no_grad():
        for episode_id in range(int(num_episodes)):
            env.reset()
            image = camera_rgb(env)
            goal = env.end_obs()
            done_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            returns = torch.zeros(env.num_envs, device=env.device)
            lengths = torch.zeros(env.num_envs, device=env.device)
            success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            fail = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            timeout = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            final_de = torch.zeros(env.num_envs, device=env.device)
            final_xe = torch.zeros(env.num_envs, device=env.device)

            for t in range(steps):
                obs_in = spr.encode(image).float() if spr is not None else image
                action = agent.act(obs_in, goal, eval_mode=True)
                step_end_d = env.end_d.clone()
                step_end_x = env.end_x.clone()
                _obs, reward, terminated, truncated, _info = env.step(action)
                active = ~done_once
                returns[active] += reward[active]
                lengths[active] += 1
                done = (terminated | truncated) & active
                row_t = episode_id * steps + t
                if traj_writer is not None:
                    for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
                        traj_writer.writerow(traj_row(env, "offline_val", step, step, row_t, int(env_id), reward[int(env_id)], bool(done[int(env_id)])))
                if env0_traj_writer is not None and bool(active[0].item()):
                    env0_traj_writer.writerow(traj_row(env, "offline_val_env0", step, step, row_t, 0, reward[0], bool(done[0])))
                if torch.any(done):
                    success[done] = env.last_success[done]
                    fail[done] = env.last_fail[done]
                    timeout[done] = env.last_timeout[done] | truncated[done]
                    final_de[done] = (env.last_dist - step_end_d).abs()[done]
                    final_xe[done] = (env.last_px_x - (cx + step_end_x)).abs()[done]
                    done_once[done] = True
                    if bool(done_once.all()):
                        break
                image = camera_rgb(env)
                goal = env.end_obs()

            rem = ~done_once
            if torch.any(rem):
                final_de[rem] = (env.last_dist - env.end_d).abs()[rem]
                final_xe[rem] = (env.last_px_x - (cx + env.end_x)).abs()[rem]
            timeout[~done_once] = True
            lengths[~done_once] = steps
            sc += int(success.sum().item())
            fc += int(fail.sum().item())
            tc += int(timeout.sum().item())
            return_sum += float(returns.sum().cpu())
            len_sum += float(lengths.sum().cpu())
            de_sum += float(final_de.sum().cpu())
            xe_sum += float(final_xe.sum().cpu())

    return {
        "success_rate": sc / episodes, "fail_rate": fc / episodes, "timeout_rate": tc / episodes,
        "mean_return": return_sum / episodes, "mean_len": len_sum / episodes,
        "final_de_mean": de_sum / episodes, "final_xe_mean": xe_sum / episodes,
    }


def main():
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = root_dir()
    update_dir = root / "updates"
    if not update_dir.exists():
        raise FileNotFoundError(f"update dir not found: {update_dir}")
    policy_paths = sorted(update_dir.glob("sac_*.pt"), key=step_num)
    if not policy_paths:
        raise FileNotFoundError(f"no sac_* checkpoints: {update_dir}")
    start = int(arg("start", "START"))
    stride = int(arg("stride", "STRIDE"))
    if stride <= 0:
        raise ValueError(f"stride must be positive: {stride}")
    policy_paths = [p for p in policy_paths if step_num(p) >= start][::stride]
    if not policy_paths:
        raise FileNotFoundError(f"no checkpoints after start={start} stride={stride}")

    num_episodes = int(arg("num_episodes", "NUM_EPISODES"))
    env = build_env(cfg, noise=args_cli.noise, num_envs=arg("num_envs", "NUM_ENVS"),
                    random_stop=args_cli.random_stop, device=args_cli.device)
    device = torch.device(env.device)
    spr = load_spr(root / "spr_encoder.pt", device) if args_cli.obs_mode == "spr_z" else None
    spr_ckpt_name = "spr_encoder.pt" if args_cli.obs_mode == "spr_z" else "none"
    out_csv = root / "offline_val.csv"
    env0_traj_csv = root / "traj_offline_env0.csv"
    best_row = best_path = best_traj = None

    print(f"[INFO] sac offline val cfg={args_cli.cfg} mode={args_cli.obs_mode} checkpoints={len(policy_paths)} "
          f"start={start} stride={stride} envs={env.num_envs} episodes={num_episodes} device={env.device}")
    try:
        with out_csv.open("w", newline="", encoding="utf-8") as file, \
             env0_traj_csv.open("w", newline="", encoding="utf-8") as env0_file:
            writer = csv.DictWriter(file, fieldnames=VAL_FIELDS)
            writer.writeheader()
            env0_writer = csv.DictWriter(env0_file, fieldnames=TRAJ_FIELDS)
            env0_writer.writeheader()
            for path in policy_paths:
                agent = load_agent(path, device)
                step = step_num(path)
                traj_buf = TrajBuffer()
                metrics = val(env, agent, spr, num_episodes, step, traj_buf, env0_writer)
                row = {
                    "checkpoint": path.name, "kind": "sac", "spr_checkpoint": spr_ckpt_name,
                    "update": step, "step": step, "num_envs": env.num_envs,
                    "num_episodes": num_episodes, "episodes": env.num_envs * num_episodes,
                    **{k: metrics[k] for k in ("success_rate", "fail_rate", "timeout_rate",
                                               "mean_return", "mean_len", "final_de_mean", "final_xe_mean")},
                }
                writer.writerow(row)
                file.flush()
                print_log([("val", [("checkpoint", path.name), ("step", step),
                                    ("success_rate", metrics["success_rate"]),
                                    ("timeout_rate", metrics["timeout_rate"]),
                                    ("final_de", metrics["final_de_mean"]),
                                    ("final_xe", metrics["final_xe_mean"])])])
                if best_row is None or best_key(row) > best_key(best_row):
                    best_row, best_path, best_traj = row, path, traj_buf.rows

        if best_path is not None:
            shutil.copy2(best_path, root / "best_offline.pt")
            (root / "best_offline_info.txt").write_text(
                "\n".join(f"{k} = {best_row[k]}" for k in VAL_FIELDS) + "\n", encoding="utf-8")
            write_traj(root / "traj_best_offline.csv", best_traj)
            print(f"[INFO] best={best_path.name} success_rate={best_row['success_rate']:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
