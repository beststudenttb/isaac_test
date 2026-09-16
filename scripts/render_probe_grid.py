"""受控扰动渲染:机器人不动,只动小球(距离/角度)、只动干扰球、只换光照,各渲一帧。
之后所有表征的 Δz 分析都可以离线在 CPU 上做。
    ./IsaacLab/isaaclab.sh -p scripts/render_probe_grid.py --out-dir data_probe_grid
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
from isaaclab.app import AppLauncher
PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=Path("./data_probe_grid"))
parser.add_argument("--num-envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.headless = True; a.livestream = 0; a.enable_cameras = True
app = AppLauncher(a); sim_app = app.app
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch
from torchvision.io import write_png
import isaaclab.sim as sim_utils
from src import task_cfg
from src.score_env import ScoreNoiseStudentEnvCfg, make_score_noise_student_env

cfg = ScoreNoiseStudentEnvCfg()
cfg.seed = 123; cfg.episode_length_s = 600.0; cfg.stop_n = 1
cfg.scene.num_envs = a.num_envs; cfg.sim.device = a.device
cfg.action_space = 2
cfg.sim.render = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="Off", enable_dlssg=False)
cfg.num_rerenders_on_reset = 2
cfg.end_d_min = cfg.end_d_max = 1.5; cfg.end_x_min = cfg.end_x_max = 0.0
env = make_score_noise_student_env(cfg); dev = env.device
env.reset()

RANGES = [1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
BEARS  = [-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
gr = torch.tensor([r for r in RANGES for _ in BEARS], device=dev)
gb = torch.tensor([b for _ in RANGES for b in BEARS], device=dev)
def place(rng, bear_deg, blue_shift=0.0):
    br = torch.deg2rad(bear_deg)
    env.target_xy[:, 0] = rng * torch.cos(br); env.target_xy[:, 1] = rng * torch.sin(br)
    env.write_target_pose()
    if hasattr(env, "noise_xy"):
        env.noise_xy[:, 0] = 3.0 + blue_shift; env.noise_xy[:, 1] = 2.0 - blue_shift
        env.write_noise_pose()
def light(intensity, color):
    prim = env.scene.stage.GetPrimAtPath("/World/light")
    prim.GetAttribute("inputs:intensity").Set(float(intensity))
    prim.GetAttribute("inputs:color").Set(tuple(float(c) for c in color))
def grab():
    z = torch.zeros((env.num_envs, int(cfg.action_space)), device=dev)
    for _ in range(3): env.step(z)
    img = env.camera.data.output["rgb"]
    img = img[..., :3] if img.shape[-1] > 3 else img
    lab = env.project_target()
    return img.detach().clone(), {k: v.detach().cpu().numpy().copy() for k, v in lab.items()}

L0, C0 = 2500.0, (0.8, 0.8, 0.8)
CONDS = [
    ("base",        dict(dr=0.0,  db=0.0, blue=0.0, I=L0,    C=C0)),
    ("r+0.05",      dict(dr=0.05, db=0.0, blue=0.0, I=L0,    C=C0)),
    ("r+0.30",      dict(dr=0.30, db=0.0, blue=0.0, I=L0,    C=C0)),
    ("b+1",         dict(dr=0.0,  db=1.0, blue=0.0, I=L0,    C=C0)),
    ("b+5",         dict(dr=0.0,  db=5.0, blue=0.0, I=L0,    C=C0)),
    ("blue_move",   dict(dr=0.0,  db=0.0, blue=1.5, I=L0,    C=C0)),
    ("light_half",  dict(dr=0.0,  db=0.0, blue=0.0, I=L0/2,  C=C0)),
    ("light_x2",    dict(dr=0.0,  db=0.0, blue=0.0, I=L0*2,  C=C0)),
    ("light_warm",  dict(dr=0.0,  db=0.0, blue=0.0, I=L0,    C=(1.0, 0.75, 0.5))),
]
out = a.out_dir; (out / "img").mkdir(parents=True, exist_ok=True)
rows = []
for name, c in CONDS:
    light(c["I"], c["C"])
    place(gr + c["dr"], gb + c["db"], c["blue"])
    img, lab = grab()
    for i in range(env.num_envs):
        fn = f"{name}_{i:03d}.png"
        write_png(img[i].permute(2, 0, 1).to(torch.uint8).cpu(), str(out / "img" / fn))
        rows.append(dict(cond=name, env=i, img=f"img/{fn}",
                         set_range=float(gr[i] + c["dr"]), set_bear=float(gb[i] + c["db"]),
                         px_x=float(lab["px_x"][i]), dist=float(lab["dist"][i]),
                         bearing=float(lab["bearing_deg"][i]), rng=float(lab["range"][i]),
                         intensity=c["I"], blue=c["blue"]))
    print(f"[GRID] {name} 渲染完成", flush=True)
with (out / "grid.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[GRID] 共 {len(rows)} 帧 -> {out}", flush=True)
env.close(); sim_app.close()
