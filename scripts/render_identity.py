"""身份实验渲染:几何完全不变,只改两个球的颜色/存在与否。
问的是"表征凭什么认出目标":是颜色(身份),还是位置/显著性?
槽 1 = target prim(project_target 的标签始终描述它),槽 2 = noise prim。
两个球的 diffuseColor 在运行时直接改,所以同一轮里所有条件的几何逐像素一致。
    ./IsaacLab/isaaclab.sh -p scripts/render_identity.py --rounds 6 --out-dir data_identity
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
from isaaclab.app import AppLauncher
PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=Path("./data_identity"))
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--rounds", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.headless = True; a.livestream = 0; a.enable_cameras = True
app = AppLauncher(a); sim_app = app.app
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch
from torchvision.io import write_png
import isaaclab.sim as sim_utils
from src.score_env import ScoreNoiseStudentEnvCfg, make_score_noise_student_env

cfg = ScoreNoiseStudentEnvCfg()
cfg.seed = 11; cfg.episode_length_s = 1e5; cfg.stop_n = 1
cfg.scene.num_envs = a.num_envs; cfg.sim.device = a.device; cfg.action_space = 2
cfg.sim.render = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="Off", enable_dlssg=False)
cfg.num_rerenders_on_reset = 2
cfg.end_d_min = cfg.end_d_max = 1.5; cfg.end_x_min = cfg.end_x_max = 0.0
env = make_score_noise_student_env(cfg); dev = env.device
env.reset()
stage = env.scene.stage

def color_attrs(suffix):
    out = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if suffix not in p: continue
        for an in ("inputs:diffuseColor", "inputs:diffuse_color"):
            at = prim.GetAttribute(an)
            if at and at.IsValid(): out.append(at)
    return out
SLOT1 = color_attrs("/target"); SLOT2 = color_attrs("/noise")
print(f"[SETUP] 槽1(target)材质 {len(SLOT1)} 个,槽2(noise)材质 {len(SLOT2)} 个", flush=True)
assert SLOT1 and SLOT2, "找不到球的材质属性"

COL = {
    "red":     (1.00, 0.00, 0.00),   # 训练时的目标色
    "blue":    (0.00, 0.10, 1.00),   # 训练时的干扰色
    "orange":  (1.00, 0.45, 0.00),   # 近红,同色系
    "magenta": (1.00, 0.00, 0.55),   # 近红,色相另一侧
    "darkred": (0.42, 0.00, 0.00),   # 同色相,低明度
    "yellow":  (1.00, 0.95, 0.00),   # 高明度,异色相
    "green":   (0.00, 0.72, 0.10),   # 对立色系
    "cyan":    (0.00, 0.85, 1.00),   # 红的补色方向,且接近蓝
    "white":   (0.92, 0.92, 0.92),   # 无彩色,接近墙(0.65)与地板(0.82)
}
# (条件名, 槽1 颜色, 槽2 颜色);None = 把这个球挪到机器人背后(FOV ±40°,x<0 一定看不见)
CONDS = [
    ("base",      "red",     "blue"),    # 基线,探针在这一条上拟合
    ("swap",      "blue",    "red"),     # 几何不动,红蓝互换 —— 读出跟颜色走还是跟位置走
    ("only_red1",  "red",    None),      # 只剩红球,在槽1(对照)
    ("only_red2",  None,     "red"),     # 只剩红球,在槽2 ★ 读出该指向槽2
    ("only_blue1", "blue",   None),      # 只剩蓝球,在槽1 ★ 通用球检测器会指向它
    ("only_blue2", None,     "blue"),    # 只剩蓝球,在槽2 ★ 最关键的一格
    ("two_red",   "red",     "red"),     # 两个都红,看锁哪个
    ("orange",    "orange",  "blue"),
    ("magenta",   "magenta", "blue"),
    ("darkred",   "darkred", "blue"),
    ("yellow",    "yellow",  "blue"),
    ("green",     "green",   "blue"),
    ("cyan",      "cyan",    "blue"),
    ("white",     "white",   "blue"),
]
PARK = (-3.0, 0.0)   # 机器人在原点朝 +x,房间 16m,背后 3m 处安全且在墙内

def set_colors(c1, c2):
    for at in SLOT1: at.Set(tuple(float(x) for x in COL[c1 if c1 else "red"]))
    for at in SLOT2: at.Set(tuple(float(x) for x in COL[c2 if c2 else "blue"]))

def place(r1, b1, r2, b2, c1, c2):
    br = torch.deg2rad(b1)
    x1 = torch.where(torch.tensor(c1 is not None, device=dev), r1*torch.cos(br), torch.full_like(r1, PARK[0]))
    y1 = torch.where(torch.tensor(c1 is not None, device=dev), r1*torch.sin(br), torch.full_like(r1, PARK[1]))
    env.target_xy[:, 0] = x1; env.target_xy[:, 1] = y1; env.write_target_pose()
    bb = torch.deg2rad(b2)
    x2 = torch.where(torch.tensor(c2 is not None, device=dev), r2*torch.cos(bb), torch.full_like(r2, PARK[0]))
    y2 = torch.where(torch.tensor(c2 is not None, device=dev), r2*torch.sin(bb), torch.full_like(r2, PARK[1]-0.6))
    env.noise_xy[:, 0] = x2; env.noise_xy[:, 1] = y2; env.write_noise_pose()

def grab():
    z = torch.zeros((env.num_envs, int(cfg.action_space)), device=dev)
    for _ in range(3): env.step(z)
    img = env.camera.data.output["rgb"]; img = img[..., :3] if img.shape[-1] > 3 else img
    return img.detach().clone()

out = a.out_dir; (out / "img").mkdir(parents=True, exist_ok=True)
g = torch.Generator(device=dev); g.manual_seed(11)
U = lambda lo, hi: torch.rand(a.num_envs, generator=g, device=dev)*(hi-lo)+lo
rows = []
for rd in range(a.rounds):
    r1, b1 = U(1.3, 6.5), U(-35., 35.)
    r2, b2 = U(1.3, 6.5), U(-35., 35.)
    for ci, (name, c1, c2) in enumerate(CONDS):
        set_colors(c1, c2); place(r1, b1, r2, b2, c1, c2)
        img = grab()
        for k in range(a.num_envs):
            fn = f"r{rd:02d}_{name}_{k:03d}.png"
            write_png(img[k].permute(2,0,1).to(torch.uint8).cpu(), str(out/"img"/fn))
            rows.append(dict(img=f"img/{fn}", cond=name, round=rd, env=k,
                             col1=c1 or "none", col2=c2 or "none",
                             rng1=float(r1[k]), bear1=float(b1[k]),
                             rng2=float(r2[k]), bear2=float(b2[k])))
        print(f"[{rd+1}/{a.rounds}] {name}  累计 {len(rows)} 帧", flush=True)
with (out/"meta.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[DONE] {len(rows)} 帧 -> {out}", flush=True)
env.close(); sim_app.close()
