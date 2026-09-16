"""受控析因渲染:7 个变量各自独立采样,机器人不动。
per-env 变量:目标球 距离、方位角;干扰球 距离、方位角
per-round 变量:光照强度、光照颜色、墙颜色、地板颜色
同时渲一份配对扰动集(每个变量单独 +δ),用来测 Δz 的方向敏感度。
    ./IsaacLab/isaaclab.sh -p scripts/render_factorial.py --rounds 60 --out-dir data_factorial
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
from isaaclab.app import AppLauncher
PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=Path("./data_factorial"))
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--rounds", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.headless = True; a.livestream = 0; a.enable_cameras = True
app = AppLauncher(a); sim_app = app.app
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch
from torchvision.io import write_png
import isaaclab.sim as sim_utils
from src.score_env import ScoreNoiseStudentEnvCfg, make_score_noise_student_env

cfg = ScoreNoiseStudentEnvCfg()
cfg.seed = 7; cfg.episode_length_s = 1e5; cfg.stop_n = 1
cfg.scene.num_envs = a.num_envs; cfg.sim.device = a.device; cfg.action_space = 2
cfg.sim.render = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="Off", enable_dlssg=False)
cfg.num_rerenders_on_reset = 2
cfg.end_d_min = cfg.end_d_max = 1.5; cfg.end_x_min = cfg.end_x_max = 0.0
env = make_score_noise_student_env(cfg); dev = env.device
env.reset()
stage = env.scene.stage

def find_color_attrs(keyword):
    """找出路径里含 keyword 的所有带 diffuseColor 的属性。"""
    out = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if keyword not in p: continue
        for an in ("inputs:diffuseColor", "inputs:diffuse_color"):
            at = prim.GetAttribute(an)
            if at and at.IsValid(): out.append(at)
    return out
WALL = find_color_attrs("wall"); FLOOR = find_color_attrs("floor")
LIGHT = stage.GetPrimAtPath("/World/light")
print(f"[SETUP] 墙材质属性 {len(WALL)} 个,地板 {len(FLOOR)} 个,光源 {'有' if LIGHT.IsValid() else '无'}", flush=True)

def set_scene(I, C, wall_c, floor_c):
    if LIGHT.IsValid():
        LIGHT.GetAttribute("inputs:intensity").Set(float(I))
        LIGHT.GetAttribute("inputs:color").Set(tuple(float(x) for x in C))
    for at in WALL: at.Set(tuple(float(x) for x in wall_c))
    for at in FLOOR: at.Set(tuple(float(x) for x in floor_c))
def place(rng_t, bear_t, rng_b, bear_b):
    br = torch.deg2rad(bear_t); env.target_xy[:, 0] = rng_t*torch.cos(br); env.target_xy[:, 1] = rng_t*torch.sin(br)
    env.write_target_pose()
    if hasattr(env, "noise_xy"):
        bb = torch.deg2rad(bear_b); env.noise_xy[:, 0] = rng_b*torch.cos(bb); env.noise_xy[:, 1] = rng_b*torch.sin(bb)
        env.write_noise_pose()
def grab():
    z = torch.zeros((env.num_envs, int(cfg.action_space)), device=dev)
    for _ in range(3): env.step(z)
    img = env.camera.data.output["rgb"]; img = img[..., :3] if img.shape[-1] > 3 else img
    lab = env.project_target()
    return img.detach().clone(), {k: v.detach().cpu().numpy().copy() for k, v in lab.items()}

out = a.out_dir; (out / "img").mkdir(parents=True, exist_ok=True)
g = torch.Generator(device=dev); g.manual_seed(7)
U = lambda lo, hi: torch.rand(a.num_envs, generator=g, device=dev)*(hi-lo)+lo
rows = []
def save(tag, k, img, lab, meta):
    fn = f"{tag}_{k:03d}.png"
    write_png(img[k].permute(2,0,1).to(torch.uint8).cpu(), str(out/"img"/fn))
    rows.append(dict(img=f"img/{fn}", **meta,
                     px_x=float(lab["px_x"][k]), dist=float(lab["dist"][k]),
                     bearing=float(lab["bearing_deg"][k]), rng=float(lab["range"][k])))
np.random.seed(7)
for rd in range(a.rounds):
    I = float(np.exp(np.random.uniform(np.log(600), np.log(6000))))
    C = np.random.uniform(0.55, 1.0, 3)
    wc = np.random.uniform(0.25, 0.9, 3); fc = np.random.uniform(0.25, 0.95, 3)
    rt, bt = U(1.3, 6.5), U(-35., 35.); rb, bb = U(1.0, 7.0), U(-40., 40.)
    set_scene(I, C, wc, fc); place(rt, bt, rb, bb)
    img, lab = grab()
    for k in range(a.num_envs):
        save(f"f{rd:03d}", k, img, lab, dict(kind="factorial", round=rd,
             set_rng=float(rt[k]), set_bear=float(bt[k]), blue_rng=float(rb[k]), blue_bear=float(bb[k]),
             light_I=I, light_r=float(C[0]), light_g=float(C[1]), light_b=float(C[2]),
             wall_r=float(wc[0]), wall_g=float(wc[1]), wall_b=float(wc[2]),
             floor_r=float(fc[0]), floor_g=float(fc[1]), floor_b=float(fc[2])))
    if rd % 10 == 0: print(f"[FACT] round {rd}/{a.rounds}", flush=True)

# --- 配对扰动集:固定基准场景,每次只动一个变量 ---
RANGES=[1.5,1.7,2.0,2.5,3.0,4.0,5.0,6.0]; BEARS=[-30.,-20.,-10.,-5.,0.,5.,10.,20.]
br0=torch.tensor([b for _ in RANGES for b in BEARS], device=dev)
rr0=torch.tensor([r for r in RANGES for _ in BEARS], device=dev)
fx=112/math.tan(math.radians(40))
d_bear_px5 = torch.rad2deg(torch.atan(torch.tan(torch.deg2rad(br0))+5.0/fx))-br0   # 等 5px 横移所需角度
d_rng_diam2 = 62.12/torch.clamp(62.12/rr0-2.0,min=1e-3)-rr0                        # 等 2px 直径变化所需距离
I0,C0,W0,F0 = 2500.0,(0.8,0.8,0.8),(0.65,0.65,0.65),(0.82,0.82,0.78)
PAIR=[("base",       dict()),
      ("bear+1deg",  dict(db=torch.ones_like(br0))),
      ("bear+5px",   dict(db=d_bear_px5)),
      ("rng+0.1m",   dict(dr=torch.full_like(rr0,0.1))),
      ("rng+2pxdiam",dict(dr=d_rng_diam2)),
      ("blue_move",  dict(blue=True)),
      ("light_x0.3", dict(I=I0*0.3)), ("light_x3", dict(I=I0*3)),
      ("light_warm", dict(C=(1.0,0.7,0.45))), ("light_cool", dict(C=(0.5,0.7,1.0))),
      ("wall_dark",  dict(W=(0.25,0.25,0.28))), ("wall_red", dict(W=(0.7,0.3,0.3))),
      ("floor_dark", dict(F=(0.3,0.3,0.3))),   ("floor_blue",dict(F=(0.35,0.45,0.8)))]
for name,c in PAIR:
    set_scene(c.get("I",I0), c.get("C",C0), c.get("W",W0), c.get("F",F0))
    rb2 = torch.full_like(rr0, 4.5 if c.get("blue") else 3.6); bb2 = torch.full_like(br0, -25.0 if c.get("blue") else 25.0)
    place(rr0+c.get("dr",0.0), br0+c.get("db",0.0), rb2, bb2)
    img,lab = grab()
    for k in range(env.num_envs):
        save(f"p_{name}", k, img, lab, dict(kind="pair", cond=name, round=-1,
             set_rng=float((rr0+c.get("dr",0.0))[k]), set_bear=float((br0+c.get("db",0.0))[k]),
             blue_rng=float(rb2[k]), blue_bear=float(bb2[k]), light_I=c.get("I",I0),
             light_r=c.get("C",C0)[0], light_g=c.get("C",C0)[1], light_b=c.get("C",C0)[2],
             wall_r=c.get("W",W0)[0], wall_g=c.get("W",W0)[1], wall_b=c.get("W",W0)[2],
             floor_r=c.get("F",F0)[0], floor_g=c.get("F",F0)[1], floor_b=c.get("F",F0)[2]))
    print(f"[PAIR] {name}", flush=True)
with (out/"meta.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[DONE] {len(rows)} 帧 -> {out}", flush=True)
env.close(); sim_app.close()
