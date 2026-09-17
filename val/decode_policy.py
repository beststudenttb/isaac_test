"""解码策略的确定性评估:o -> z -> 解码 teacher 观测 -> teacher 自己的策略:直接用 stage1 的「编码器 + 读出头」当策略,不经过 PPO。

只是把 val/score_student.py 的 actor 换成 head(encoder(img)) * std + mean 再 clamp,
其余(env、起点、375 步、v1 判据、末 100 步统计)完全一致,便于和 PPO 的成绩逐格对比。

原文件头:
任务 v3 的确定性评估:64 env 各跑满一条 375 步 episode,报平均回报。

    ./IsaacLab/isaaclab.sh -p val/score_student.py --arm A

训练日志里的回报是 env0 单条轨迹 + 探索噪声,只能看趋势。这里用确定性动作
(actor 的 mean)、全部 env、完整 episode,给可引用的数字。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Deterministic evaluation for task v3.")
parser.add_argument("--arm", required=True)  # 任意臂名:{ENCODER_ROOT}/{arm}/encoder_final.pt;"init" 特殊 = encoder_init.pt
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--tag", default="")
parser.add_argument("--encoder-root", default=None)
parser.add_argument("--run-tag", default="")  # 训练时的 --tag,拼进 run 目录名
parser.add_argument("--score-mode", default=None, choices=("angle", "cam", "pixel"), help="奖励与 stop 判定的度量,须与该表征对应的 teacher 一致")
parser.add_argument("--teacher", default=None)
parser.add_argument("--chase-blue", action="store_true")
parser.add_argument("--adapter", action="store_true")
parser.add_argument("--adapter-hidden", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
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
import torch.nn as nn
import isaaclab.sim as sim_utils

sys.path.insert(0, str(PROJECT_ROOT / "train"))
import free_student_cfg as cfg
from src import task_cfg
from src.cv_extractor.config import FREE_SPATIAL_CONFIG
from src.cv_extractor.free_spatial import FrozenFreeEncoder
from src.free_ppo import AdapterActorCritic, FreeFeatureActorCritic
from src.score_env import ScoreNoiseStudentEnvCfg, make_score_noise_student_env

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def main() -> None:
    env_cfg = ScoreNoiseStudentEnvCfg()
    env_cfg.seed = 123
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if cfg.NO_LATERAL:
        env_cfg.action_space = 2
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.render = sim_utils.RenderCfg(
        rendering_mode=str(cfg.RENDERING_MODE), antialiasing_mode=str(cfg.ANTIALIASING_MODE), enable_dlssg=False
    )
    env_cfg.num_rerenders_on_reset = int(cfg.RERENDER_ON_RESET)
    env_cfg.angle_deg = float(cfg.ANGLE_DEG)
    if args_cli.score_mode:
        env_cfg.score_mode = str(args_cli.score_mode)
    env_cfg.end_d_min = env_cfg.end_d_max = float(cfg.END_D_MIN)
    env_cfg.end_x_min = env_cfg.end_x_max = float(cfg.END_X_MIN)
    env = make_score_noise_student_env(env_cfg)
    device = torch.device(env.device)

    root = Path(args_cli.encoder_root) if args_cli.encoder_root else Path(cfg.ENCODER_ROOT)
    import json, glob
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
    from stage1_helpers import make_head
    from stable_baselines3 import PPO
    arm = args_cli.arm                      # 通常是 sup:它的头输出的就是 teacher 的那两维观测
    encoder = FreeSpatialFeatureExtractor(**dict(FREE_SPATIAL_CONFIG)).to(device)
    encoder.load_state_dict(torch.load(root / arm / "donor.pt", map_location=device)); encoder.eval()
    st = json.load(open(root / "target_stats.json"))[arm]
    t_mean = torch.tensor(st["mean"], dtype=torch.float32, device=device)
    t_std = torch.tensor(st["std"], dtype=torch.float32, device=device)
    head = make_head(int(FREE_SPATIAL_CONFIG["feature_dim"]), len(st["mean"])).to(device)
    head.load_state_dict(torch.load(root / arm / "head.pt", map_location=device)); head.eval()
    tz = args_cli.teacher or sorted(glob.glob("models/rl/teacher_score_g4_diam_angle_s*/last.zip"))[-1]
    tpol = PPO.load(tz, device=str(device)).policy
    print(f"[DECODE] 表征={arm} 解码头 -> teacher 观测 -> teacher 策略({tz})", flush=True)

    class DecodePolicy:
        """o -> z -> 解码出 teacher 的那两维观测 -> 直接喂 teacher 自己的策略。全链路不含新训练的策略。"""
        def predict(self, image, goal):
            z = encoder(image)["shared_feature"]
            o_hat = head(z) * t_std + t_mean                 # (nx, nd)
            full = torch.stack((o_hat[:, 0], o_hat[:, 1], goal[:, 0], goal[:, 1]), dim=-1)
            a = tpol.action_net(tpol.mlp_extractor.policy_net(full))
            return torch.clamp(a, -1.0, 1.0)
    model = DecodePolicy()
    run_dir = Path(f"{cfg.OUT_ROOT}_decode_{arm}{args_cli.run_tag}")
    run_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    steps = int(round(float(cfg.EPISODE_S) / task_cfg.DT))
    total = torch.zeros(env.num_envs, device=device)
    zone = torch.zeros(env.num_envs, device=device)
    tail_xe = torch.zeros(env.num_envs, device=device)
    tail_de = torch.zeros(env.num_envs, device=device)
    tail_zone = torch.zeros(env.num_envs, device=device)
    tail_amag = torch.zeros(env.num_envs, device=device)
    TAIL = 100
    STOP_N = 3                                                # v1 判据:区内 + max|a|<0.05 连续 3 步
    streak = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    succ_step = torch.full((env.num_envs,), -1, device=device, dtype=torch.long)
    for t in range(steps):
        image = env.camera.data.output["rgb"]
        image = image[..., :3] if image.shape[-1] > 3 else image
        goal = env.end_obs()
        with torch.no_grad():
            action = model.predict(image, goal)          # 确定性:actor 的 mean
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        inz = env.in_stop_zone().float()
        zone += inz
        amag_now = torch.clamp(action, -1.0, 1.0).abs().max(dim=1).values
        hit = env.in_stop_zone() & (amag_now < task_cfg.STOP_EPS)
        streak = torch.where(hit, streak + 1, torch.zeros_like(streak))
        succ_step = torch.where((streak >= STOP_N) & (succ_step < 0), torch.full_like(succ_step, t), succ_step)
        if t >= steps - TAIL:                      # 末 100 步:最终停在哪
            lab = env.project_target()
            seen = lab["dist"] > env.cfg.lost_d
            if env.cfg.score_mode == "angle":   # 单位:deg / m
                tail_xe += torch.where(seen, torch.abs(lab["bearing_deg"]), torch.full_like(lab["px_x"], 40.0))
                tail_de += torch.where(seen, torch.abs(lab["range"] - env.end_d), torch.full_like(lab["dist"], 6.5))
            else:
                tail_xe += torch.where(seen, torch.abs(lab["px_x"] - 112.0), torch.full_like(lab["px_x"], 112.0))
                tail_de += torch.where(seen, torch.abs(lab["dist"] - env.end_d), torch.full_like(lab["dist"], 6.5))
            tail_zone += inz
            tail_amag += torch.clamp(action, -1.0, 1.0).abs().max(dim=1).values

    ret = total.cpu().numpy()
    dwell = (zone / steps).cpu().numpy()
    t_xe = (tail_xe / TAIL).cpu().numpy()
    t_de = (tail_de / TAIL).cpu().numpy()
    t_zone = (tail_zone / TAIL).cpu().numpy()
    t_amag = (tail_amag / TAIL).cpu().numpy()
    succ = (succ_step >= 0).cpu().numpy(); sstep = succ_step.cpu().numpy()
    v1 = float(succ.mean()); first = float(sstep[succ].mean()) if succ.any() else float("nan")
    print(f"[DECODE] arm={args_cli.arm}{args_cli.run_tag}  v1成功率 {v1:.1%} 首次成功步 {first:.0f}  回报 mean={ret.mean():.1f}  std={ret.std():.1f}  "
          f"满分={steps}  区内={dwell.mean():.3f}  末100步: xe={t_xe.mean():.1f}px de={t_de.mean():.3f}m "
          f"区内={t_zone.mean():.3f} |a|={t_amag.mean():.3f}", flush=True)
    out = run_dir / f"eval_decode{args_cli.tag}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["env", "return", "zone_dwell", "tail_xe", "tail_de", "tail_zone", "tail_amag", "v1_success", "succ_step"])
        for i in range(len(ret)):
            w.writerow([i, f"{ret[i]:.4f}", f"{dwell[i]:.4f}", f"{t_xe[i]:.3f}", f"{t_de[i]:.4f}", f"{t_zone[i]:.4f}", f"{t_amag[i]:.4f}", int(succ[i]), int(sstep[i])])
    env.close()


try:
    main()
except Exception:
    import traceback, sys as _s
    traceback.print_exc(file=_s.stdout)
    _s.stdout.flush()
simulation_app.close()
