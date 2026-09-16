"""teacher 在 score env 上的确定性评估。无相机,秒级。

    ./IsaacLab/isaaclab.sh -p val/teacher_score.py --dir models/rl/teacher_score_k0.3 --action-k 0.3

维度(2026-09-08 用户指定):
  stop 位置   首次触发 v1 成功(区内 + max|a|<0.05 连续 3 步)那一刻的 xe/de,max/min/mean
  step 数     首次成功步,max/min/mean
  轨迹表现    按距离 >4m / 1.5-4m / <1.5m 三段:帧占比、|a_x|、max|a|、每步得分
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from isaaclab.app import AppLauncher
PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
parser.add_argument("--action-k", type=float, default=0.0)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--no-lateral", action="store_true", help="与训练一致:动作空间 [a_x, a_w]")
parser.add_argument("--obs-mask", default="")
parser.add_argument("--score-mode", default=None, choices=("angle", "cam", "pixel"), help="奖励与 stop 判定的度量:angle=物理量(方位角/距离),cam=相机坐标(x_px/直径px),pixel=旧版")
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.headless = True; a.livestream = 0
app = AppLauncher(a); sim = app.app
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch
from stable_baselines3 import PPO
from src import task_cfg
from src.score_env import ScorePPOEnvCfg, ScorePPOEnv

cfg = ScorePPOEnvCfg(); cfg.seed = 123; cfg.episode_length_s = 15.0; cfg.stop_n = 1
cfg.scene.num_envs = a.num_envs; cfg.sim.device = a.device; cfg.use_camera = False; cfg.read_camera = False
cfg.end_d_min = cfg.end_d_max = 1.5; cfg.end_x_min = cfg.end_x_max = 0.0; cfg.score_action_k = a.action_k
if a.no_lateral: cfg.action_space = 2
cfg.obs_mask = a.obs_mask
if a.score_mode: cfg.score_mode = str(a.score_mode)
env = ScorePPOEnv(cfg); dev = env.device
m = PPO.load(str(Path(a.dir) / "last.zip"), device=dev)
obs, _ = env.reset(); steps = int(round(15.0 / task_cfg.DT)); N = env.num_envs
STOP_N = 3
streak = torch.zeros(N, device=dev, dtype=torch.long)
succ_step = torch.full((N,), -1, device=dev, dtype=torch.long)
succ_xe = torch.zeros(N, device=dev); succ_de = torch.zeros(N, device=dev)
tot = torch.zeros(N, device=dev)
bands = {">4m": lambda d: d > 4.0, "1.5-4m": lambda d: (d > 1.5) & (d <= 4.0), "<1.5m": lambda d: (d > 0) & (d <= 1.5)}
acc = {b: {"n": 0, "ax": 0.0, "amag": 0.0, "score": 0.0} for b in bands}
for t in range(steps):
    act, _ = m.predict(obs["policy"].detach().cpu().numpy(), deterministic=True)
    act = torch.as_tensor(act, device=dev, dtype=torch.float32)
    obs, r, _, _, _ = env.step(act); tot += r
    lab = env.project_target(); d = lab["dist"]; seen = d > 0
    if cfg.score_mode == "angle":
        xe = torch.abs(lab["bearing_deg"]); de = torch.abs(lab["range"] - 1.5)
    else:
        xe = torch.abs(lab["px_x"] - 112); de = torch.abs(d - 1.5)
    inz = env.in_stop_zone(); ac = torch.clamp(act, -1, 1); amag = ac.abs().max(1).values
    hit = inz & (amag < task_cfg.STOP_EPS)
    streak = torch.where(hit, streak + 1, torch.zeros_like(streak))
    new = (streak >= STOP_N) & (succ_step < 0)
    succ_step = torch.where(new, torch.full_like(succ_step, t), succ_step)
    succ_xe = torch.where(new, xe, succ_xe); succ_de = torch.where(new, de, succ_de)
    for b, f in bands.items():
        msk = seen & f(d); n = int(msk.sum())
        if n:
            acc[b]["n"] += n; acc[b]["ax"] += float(ac[:, 0].abs()[msk].sum())
            acc[b]["amag"] += float(amag[msk].sum()); acc[b]["score"] += float(r[msk].sum())
ok = succ_step >= 0
ss = succ_step[ok].float(); sx = succ_xe[ok]; sd = succ_de[ok]
def mmm(v): return f"max {v.max():.1f} / min {v.min():.1f} / mean {v.mean():.1f}" if len(v) else "—"
def mmm3(v): return f"max {v.max():.3f} / min {v.min():.3f} / mean {v.mean():.3f}" if len(v) else "—"
tot_frames = sum(acc[b]["n"] for b in bands)
print(f"[TEACHER] k={a.action_k} no_lateral={a.no_lateral} obs_mask={a.obs_mask!r}  v1成功率 {ok.float().mean().item():.1%}  回报 {tot.mean().item():.1f}", flush=True)
print(f"  stop 位置  xe({'deg' if cfg.score_mode == 'angle' else 'px'}): {mmm(sx)}   de(m): {mmm3(sd)}", flush=True)
print(f"  step 数    {mmm(ss)}", flush=True)
for b in bands:
    c = acc[b]; n = max(c["n"], 1)
    print(f"  轨迹 {b:7s}  帧占 {c['n']/max(tot_frames,1):5.1%}  |a_x| {c['ax']/n:.3f}  max|a| {c['amag']/n:.3f}  每步得分 {c['score']/n:.3f}", flush=True)
env.close(); sim.close()
