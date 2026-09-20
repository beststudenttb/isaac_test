"""4 物体受控渲染(2026-09-19 周末批次 §3.2)。

四个物体的位置在一轮内**完全固定**,只在运行时改它们的 diffuseColor,
所以同一轮里所有条件的几何**逐像素一致** —— 颜色是唯一变量。

槽位与 prim:0=/target(球) 1=/obj1(方) 2=/obj2(球) 3=/obj3(方)
条件:
  base    0红 1红 2蓝 3蓝   —— 训练时的样子
  swap    0蓝 1蓝 2红 3红   —— 几何不动,颜色翻转。base↔swap 的 ‖Δz‖ = **颜色敏感度**
  allred  四个都红          —— 颜色不再有区分力,**形状是唯一线索**
  allblue 四个都蓝          —— 同上,另一个色相下复核

标签:四个槽位各自的 (range, bearing, px_x, 可见),解析算出,不靠图像检测。
用途:对每个表征线性探针 z -> 各槽 px_x,填 §3 的预测表
      (只做颜色过滤 -> 红球红方都能读;真的区分目标 -> 只有红球能读;不过滤 -> 四个都能读)

    ./IsaacLab/isaaclab.sh -p scripts/render_w4.py --rounds 10 --out-dir data_w4_probe
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=Path("./data_w4_probe"))
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--rounds", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.headless = True; a.livestream = 0; a.enable_cameras = True
app = AppLauncher(a); sim_app = app.app
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.io import write_png
import isaaclab.sim as sim_utils
from src.multi_env import ScoreMultiStudentEnvCfg, make_score_multi_student_env

cfg = ScoreMultiStudentEnvCfg()
cfg.seed = 11; cfg.episode_length_s = 1e5; cfg.stop_n = 1
cfg.scene.num_envs = a.num_envs; cfg.sim.device = a.device; cfg.action_space = 2
cfg.sim.render = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="Off", enable_dlssg=False)
cfg.num_rerenders_on_reset = 2
cfg.end_d_min = cfg.end_d_max = 1.5; cfg.end_x_min = cfg.end_x_max = 0.0
env = make_score_multi_student_env(cfg); dev = env.device
env.reset()
stage = env.scene.stage

RED = (1.0, 0.0, 0.0); BLUE = (0.0, 0.1, 1.0)
PATHS = ["/target", "/obj1", "/obj2", "/obj3"]

def color_attrs(suffix):
    out = []
    for prim in stage.Traverse():
        if suffix not in str(prim.GetPath()): continue
        for an in ("inputs:diffuseColor", "inputs:diffuse_color"):
            at = prim.GetAttribute(an)
            if at and at.IsValid(): out.append(at)
    return out

SLOTS = [color_attrs(p) for p in PATHS]
print("[SETUP] 各槽材质属性数 " + " ".join(f"{p}:{len(s)}" for p, s in zip(PATHS, SLOTS)), flush=True)
assert all(SLOTS), "有槽位找不到材质属性"

CONDS = [("base",    [RED, RED, BLUE, BLUE]),
         ("swap",    [BLUE, BLUE, RED, RED]),
         ("allred",  [RED, RED, RED, RED]),
         ("allblue", [BLUE, BLUE, BLUE, BLUE])]
SHAPE = ["sphere", "cube", "sphere", "cube"]

def set_colors(cols):
    for attrs, c in zip(SLOTS, cols):
        for at in attrs: at.Set(tuple(float(x) for x in c))

def place(rng, bear):
    """rng/bear: (4, N) —— 机器人在原点朝 +x,yaw=0。"""
    br = torch.deg2rad(bear)
    x = rng * torch.cos(br); y = rng * torch.sin(br)
    env.target_xy[:, 0] = x[0]; env.target_xy[:, 1] = y[0]; env.write_target_pose()
    for k in range(3):
        env.extra_xy[:, k, 0] = x[k + 1]; env.extra_xy[:, k, 1] = y[k + 1]
    env.write_slot_poses()

def grab():
    z = torch.zeros((env.num_envs, int(cfg.action_space)), device=dev)
    for _ in range(3): env.step(z)
    img = env.camera.data.output["rgb"]
    return (img[..., :3] if img.shape[-1] > 3 else img).detach().clone()

W = float(cfg.image_width)
FX = W / (2.0 * math.tan(math.radians(float(cfg.fov_x_deg)) * 0.5))

out = a.out_dir; (out / "img").mkdir(parents=True, exist_ok=True)
g = torch.Generator(device=dev); g.manual_seed(11)
U = lambda lo, hi: torch.rand(a.num_envs, generator=g, device=dev) * (hi - lo) + lo

rows = []
for rd in range(a.rounds):
    # 四个槽各自独立采样,最小间距 0.5m(与环境一致),否则会互相遮挡
    while True:
        rng = torch.stack([U(1.3, 6.5) for _ in range(4)])
        bear = torch.stack([U(-35., 35.) for _ in range(4)])
        br = torch.deg2rad(bear); px = rng * torch.cos(br); py = rng * torch.sin(br)
        ok = True
        for i in range(4):
            for j in range(i + 1, 4):
                if float(torch.sqrt((px[i]-px[j])**2 + (py[i]-py[j])**2).min()) < 0.5: ok = False
        if ok: break
    place(rng, bear)
    for name, cols in CONDS:
        set_colors(cols)
        img = grab()
        for k in range(a.num_envs):
            fn = f"r{rd:02d}_{name}_{k:03d}.png"
            write_png(img[k].permute(2, 0, 1).to(torch.uint8).cpu(), str(out / "img" / fn))
            row = dict(img=f"img/{fn}", cond=name, round=rd, env=k)
            for s in range(4):
                r = float(rng[s, k]); b = float(bear[s, k])
                fwd = r * math.cos(math.radians(b)); lft = r * math.sin(math.radians(b))
                x_px = W * 0.5 - FX * lft / max(fwd, 1e-6)
                row[f"rng{s}"] = round(r, 4); row[f"bear{s}"] = round(b, 3)
                row[f"px{s}"] = round(x_px, 2)
                row[f"vis{s}"] = int(0.0 <= x_px < W)
                row[f"col{s}"] = "red" if cols[s] == RED else "blue"
                row[f"shape{s}"] = SHAPE[s]
            rows.append(row)
        print(f"[{rd+1}/{a.rounds}] {name}  累计 {len(rows)} 帧", flush=True)
    # 每轮落一次盘:被 reap 杀掉时最多丢一轮,不是全丢(2026-09-18 栽过一次)
    with (out / "meta.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"[DONE] {len(rows)} 帧 -> {out}", flush=True)
env.close(); sim_app.close()
