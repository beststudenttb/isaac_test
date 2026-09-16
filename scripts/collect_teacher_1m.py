"""用 teacher 采大规模数据集,分单元存放,单元边界不切断 episode。

任务 v3(稠密位置分):env 用 ScoreMixin,所以记录的 reward 就是每步的位置分。

    env -u DISPLAY CUDA_VISIBLE_DEVICES=1 ./IsaacLab/isaaclab.sh -p scripts/collect_teacher_1m.py \
        --transitions 1000000 --units 10 --num-envs 64

与 `collect_mdp_dataset.py` 的三点不同:

1. **分单元**:输出 `unit_00/ .. unit_09/`,各自带 `img/` 和 `transitions.csv`。
   单元在 **episode 开始时**分配,所以一条轨迹永远落在同一个单元里,不会被切断。
   目标数满了之后不再开新 episode,只把在飞的跑完再停 —— 保证每条轨迹完整。

2. **同时记采样动作和 teacher 的确定性 mean**(`a_*` 与 `mu_*`)。
   旧数据只有采样动作,而 teacher 的 log_std ≈ 0.04,导致 BC 目标有不可约噪声
   (a_y 维 26% 的方差是采样噪声,把 R² 天花板压到 0.89)。两个都存就没这个问题。

3. **记 teacher 的 V(s)**,省得事后重算。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("./data_score_k03_1m")
TEACHER_PATH = Path("./models/rl/teacher_score_k0.3/last.zip")

parser = argparse.ArgumentParser(description="Collect a large teacher dataset in trajectory-complete units.")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--transitions", type=int, default=1_000_000)
parser.add_argument("--units", type=int, default=10)
parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
parser.add_argument("--teacher", type=Path, default=TEACHER_PATH)
parser.add_argument("--episode-s", type=float, default=15.0)
parser.add_argument("--stop-n", type=int, default=3)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--angle-deg", type=float, default=45.0)  # 与现有 50k 数据集和任务三一致
parser.add_argument("--render-mode", default="balanced")
parser.add_argument("--aa", default="Off")
parser.add_argument("--dlss-mode", type=int, default=None)
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--clean", action="store_true")  # 默认带蓝色干扰球,与现有数据一致
parser.add_argument("--no-lateral", action="store_true")  # 动作空间 [a_x, a_w];CSV 仍记三列,a_y/mu_y 恒 0
parser.add_argument("--obs-mask", default="")  # 与 teacher 训练一致:"" / x / d / coarse / diam
parser.add_argument("--score-mode", default=None, choices=("angle", "cam", "pixel"), help="奖励与 stop 判定的度量:angle=物理量(方位角/距离),cam=相机坐标(x_px/直径px),pixel=旧版")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.livestream = 0
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from torchvision.io import write_png

import isaaclab.sim as sim_utils

from src.sb3_env import BallPPOEnv
from src.noise_env import NoiseMixin
from src.score_env import ScoreMixin, ScoreNoisePPOEnvCfg, ScorePPOEnvCfg

CSV_FIELDS = [
    "index", "episode", "t", "env_id", "img", "next_img",
    "px_x", "dist", "next_px_x", "next_dist",
    "a_x", "a_y", "a_w",          # 实际执行的采样动作
    "mu_x", "mu_y", "mu_w",       # teacher 的确定性 mean
    "value",                      # teacher 的 V(s)
    "reward", "terminated", "truncated", "done", "success", "fail", "timeout",
]


class DatasetBallPPOEnv(ScoreMixin, BallPPOEnv):
    def __init__(self, cfg):
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
    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_noise()


def make_env():
    cfg = ScorePPOEnvCfg() if args_cli.clean else ScoreNoisePPOEnvCfg()
    cfg.seed = int(args_cli.seed)
    cfg.episode_length_s = float(args_cli.episode_s)
    cfg.stop_n = int(args_cli.stop_n)
    cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.no_lateral:
        cfg.action_space = 2
    cfg.obs_mask = str(args_cli.obs_mask)
    if args_cli.score_mode: cfg.score_mode = str(args_cli.score_mode)
    cfg.scene.env_spacing = 16.0
    cfg.sim.device = args_cli.device
    cfg.angle_deg = float(args_cli.angle_deg)
    render_kwargs = {"rendering_mode": str(args_cli.render_mode), "antialiasing_mode": args_cli.aa, "enable_dlssg": False}
    if args_cli.dlss_mode is not None:
        render_kwargs["dlss_mode"] = args_cli.dlss_mode
    cfg.sim.render = sim_utils.RenderCfg(**render_kwargs)
    cfg.use_camera = True
    cfg.read_camera = True
    cfg.num_rerenders_on_reset = 1
    env_cls = DatasetBallPPOEnv if args_cli.clean else DatasetNoisePPOEnv
    return env_cls(cfg)


def rgb_images(env) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    return image[..., :3] if image.shape[-1] > 3 else image


def raw_label(env):
    label = env.project_target()
    return label["px_x"].clone(), label["dist"].clone()


def save_image(path: Path, image: torch.Tensor) -> None:
    write_png(image.permute(2, 0, 1).cpu(), str(path))


def main() -> None:
    out_root = args_cli.out_dir
    n_units = int(args_cli.units)
    target = int(args_cli.transitions)
    quota = target // n_units

    units = []
    for u in range(n_units):
        d = out_root / f"unit_{u:02d}"
        (d / "img").mkdir(parents=True, exist_ok=True)
        f = (d / "transitions.csv").open("w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        units.append({"dir": d, "file": f, "writer": w, "rows": 0})

    env = make_env()
    model = PPO.load(str(args_cli.teacher), device=env.device)
    obs, _ = env.reset()
    obs_t = obs["policy"]

    n = env.num_envs
    frame_index = 0
    unit_of = [0] * n                      # 每个 env 当前 episode 归属的单元
    episode_ids = torch.zeros(n, dtype=torch.long, device=env.device)
    episode_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    active_unit = 0
    saved = 0
    closing = False                        # 目标已达成,只把在飞的 episode 跑完
    alive = [True] * n

    cur_px, cur_d = raw_label(env)
    image_t = rgb_images(env).clone()
    cur_img = []
    for i in range(n):
        name = f"{frame_index:08d}.png"
        save_image(units[0]["dir"] / "img" / name, image_t[i])
        cur_img.append(name)
        frame_index += 1

    (out_root / "config.txt").write_text(
        "\n".join(
            f"{k} = {v}" for k, v in [
                ("teacher", args_cli.teacher), ("num_envs", n), ("transitions", target),
                ("units", n_units), ("angle_deg", args_cli.angle_deg), ("episode_s", args_cli.episode_s),
                ("stop_n", args_cli.stop_n), ("seed", args_cli.seed), ("noise", not args_cli.clean),
                ("deterministic", bool(args_cli.deterministic)), ("render_mode", args_cli.render_mode),
                ("no_lateral", bool(args_cli.no_lateral)), ("obs_mask", args_cli.obs_mask),
            ]
        ) + "\n", encoding="utf-8")

    print(f"[INFO] out={out_root} envs={n} target={target} units={n_units} quota={quota} "
          f"angle={args_cli.angle_deg} noise={not args_cli.clean}", flush=True)

    t0 = time.perf_counter()
    last_report, last_saved = t0, 0
    while simulation_app.is_running() and any(alive):
        obs_np = obs_t.detach().cpu().numpy()
        action_np, _ = model.predict(obs_np, deterministic=bool(args_cli.deterministic))
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t.detach())
            mu = dist.distribution.mean.cpu().numpy()
            value = model.policy.predict_values(obs_t.detach()).cpu().numpy().ravel()
        action = torch.as_tensor(action_np, device=env.device, dtype=torch.float32)
        if args_cli.no_lateral:  # env 内部补 a_y=0;CSV 也按三列记,与旧数据同 schema
            action_np = np.insert(action_np, 1, 0.0, axis=1)
            mu = np.insert(mu, 1, 0.0, axis=1)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        image_next = rgb_images(env).clone()
        nxt_px, nxt_d = env.last_px_x.clone(), env.last_dist.clone()
        done = terminated | truncated

        next_names = [None] * n
        for i in range(n):
            if not alive[i]:
                continue
            u = units[unit_of[i]]
            name = f"{frame_index:08d}.png"
            save_image(u["dir"] / "img" / name, env.terminal_rgb[i] if bool(done[i]) else image_next[i])
            frame_index += 1
            next_names[i] = name
            u["writer"].writerow({
                "index": u["rows"], "episode": int(episode_ids[i]), "t": int(episode_steps[i]), "env_id": i,
                "img": cur_img[i], "next_img": name,
                "px_x": f"{float(cur_px[i]):.6f}", "dist": f"{float(cur_d[i]):.6f}",
                "next_px_x": f"{float(nxt_px[i]):.6f}", "next_dist": f"{float(nxt_d[i]):.6f}",
                "a_x": f"{action_np[i,0]:.6f}", "a_y": f"{action_np[i,1]:.6f}", "a_w": f"{action_np[i,2]:.6f}",
                "mu_x": f"{mu[i,0]:.6f}", "mu_y": f"{mu[i,1]:.6f}", "mu_w": f"{mu[i,2]:.6f}",
                "value": f"{value[i]:.6f}",
                "reward": f"{float(reward[i]):.6f}",
                "terminated": int(bool(terminated[i])), "truncated": int(bool(truncated[i])), "done": int(bool(done[i])),
                "success": int(bool(env.last_success[i])), "fail": int(bool(env.last_fail[i])),
                "timeout": int(bool((env.last_timeout[i] | truncated[i]))),
            })
            u["rows"] += 1
            saved += 1

        if not closing and saved >= target:
            closing = True
            print(f"[INFO] 目标 {target} 达成,停止开新 episode,等在飞的 {sum(alive)} 条跑完", flush=True)

        # 单元轮换:只在 episode 边界发生,所以轨迹不会被切断。
        while active_unit < n_units - 1 and units[active_unit]["rows"] >= quota:
            active_unit += 1

        for i in range(n):
            if bool(done[i]):
                if closing:
                    alive[i] = False
                    continue
                unit_of[i] = active_unit
                episode_ids[i] += 1
                episode_steps[i] = 0
                name = f"{frame_index:08d}.png"
                save_image(units[active_unit]["dir"] / "img" / name, image_next[i])
                frame_index += 1
                cur_img[i] = name
            else:
                cur_img[i] = next_names[i]
                episode_steps[i] += 1

        cur_px, cur_d = raw_label(env)
        obs_t = obs_next["policy"]

        now = time.perf_counter()
        if now - last_report >= 10.0:
            rate = (saved - last_saved) / (now - last_report)
            eta = (target - saved) / max(rate, 1e-6) / 60
            print(f"[INFO] {saved}/{target} ({saved/target:.1%})  {rate:.0f}/s  剩 {eta:.0f} 分钟  "
                  f"单元 {active_unit}  存活 {sum(alive)}", flush=True)
            last_report, last_saved = now, saved

    for u in units:
        u["file"].close()
    print(f"[INFO] 完成 {saved} transitions,用时 {(time.perf_counter()-t0)/60:.1f} 分钟", flush=True)
    for i, u in enumerate(units):
        print(f"  unit_{i:02d}: {u['rows']} 行")
    env.close()


main()
simulation_app.close()
