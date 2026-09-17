"""形状实验:颜色已经定死(§22),现在问形状。
除了原有的两个球,再生成 立方体 / 圆锥 / 圆柱 三个视觉体(颜色运行时可改,不用时挪到机器人背后)。
探针仍旧只在 base(红球@槽1,蓝球@槽2)上拟合后冻住。
    ./IsaacLab/isaaclab.sh -p scripts/render_shape.py --rounds 6 --out-dir data_shape
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
from isaaclab.app import AppLauncher
PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=Path("./data_distract"))
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--rounds", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.headless = True; a.livestream = 0; a.enable_cameras = True
app = AppLauncher(a); sim_app = app.app
sys.path.insert(0, str(PROJECT_ROOT))
import torch
from torchvision.io import write_png
import isaaclab.sim as sim_utils
from isaaclab.sim.views import XformPrimView
from src.score_env import ScoreMixin, ScoreNoiseStudentEnvCfg
from src.noise_env import NoiseMixin
from src.mdp_student_env import MDPStudentEnv

EXTRA = ("cube", "cone", "cyl", "ball_s", "ball_l")

class ShapeMixin:
    """再加三个非球形视觉体,和 target/noise 一样用 XformPrimView 摆位。"""
    def _setup_scene(self):
        super()._setup_scene()
        self.shape_views = {}
        for n in EXTRA:
            v = XformPrimView(f"{self.scene.env_regex_ns}/{n}", device=self.device,
                              validate_xform_ops=False, sync_usd_on_fabric_write=True, stage=self.scene.stage)
            self.shape_views[n] = v; self.scene.extras[n] = v
    def _standardize_xforms(self, env_path: str):
        super()._standardize_xforms(env_path)
        for n in EXTRA:
            sim_utils.standardize_xform_ops(self.scene.stage.GetPrimAtPath(f"{env_path}/{n}"))
    def _spawn_env(self, env_path: str):
        super()._spawn_env(env_path)
        r = self.cfg.target_radius
        mat = lambda: sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))
        specs = {
            "cube": sim_utils.CuboidCfg(size=(2*r, 2*r, 2*r), visual_material=mat()),
            "cone": sim_utils.ConeCfg(radius=r, height=2*r, visual_material=mat()),
            "cyl":  sim_utils.CylinderCfg(radius=r, height=2*r, visual_material=mat()),
            "ball_s": sim_utils.SphereCfg(radius=0.5*r, visual_material=mat()),   # 半径 0.5r
            "ball_l": sim_utils.SphereCfg(radius=2.0*r, visual_material=mat()),   # 半径 2r
        }
        for n, c in specs.items():
            c.func(f"{env_path}/{n}", c, translation=(-3.0, 1.5 + list(specs).index(n), r))
    def write_shape_pose(self, name, xy, lift=0.0):
        r = self.cfg.target_radius
        rr = {"ball_s": 0.5*r, "ball_l": 2.0*r}.get(name, r)
        origins = self.scene.env_origins
        pos = origins.clone(); pos[:, 0:2] += xy; pos[:, 2] += rr + lift
        self.shape_views[name].set_world_poses(positions=pos,
                                               indices=list(range(self.num_envs)))

class ShapeEnv(ShapeMixin, ScoreMixin, NoiseMixin, MDPStudentEnv):
    cfg: ScoreNoiseStudentEnvCfg
    def __init__(self, cfg):
        super().__init__(cfg); self.init_noise()

cfg = ScoreNoiseStudentEnvCfg()
cfg.seed = 17; cfg.episode_length_s = 1e5; cfg.stop_n = 1
cfg.scene.num_envs = a.num_envs; cfg.sim.device = a.device; cfg.action_space = 2
cfg.sim.render = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="Off", enable_dlssg=False)
cfg.num_rerenders_on_reset = 2
cfg.end_d_min = cfg.end_d_max = 1.5; cfg.end_x_min = cfg.end_x_max = 0.0
env = ShapeEnv(cfg); dev = env.device
env.reset(); stage = env.scene.stage

def color_attrs(suffix):
    out = []
    for prim in stage.Traverse():
        if suffix not in str(prim.GetPath()): continue
        for an in ("inputs:diffuseColor", "inputs:diffuse_color"):
            at = prim.GetAttribute(an)
            if at and at.IsValid(): out.append(at)
    return out
ATTR = {"ball1": color_attrs("/target"), "ball2": color_attrs("/noise")}
for n in EXTRA: ATTR[n] = color_attrs(f"/{n}")
print("[SETUP] " + "  ".join(f"{k}:{len(v)}" for k, v in ATTR.items()), flush=True)
COL = {"red": (1.0,0.0,0.0), "blue": (0.0,0.1,1.0), "grey": (0.5,0.5,0.5),
       "orange": (1.0,0.45,0.0), "magenta": (1.0,0.0,0.55), "yellow": (1.0,0.95,0.0),
       "green": (0.0,0.72,0.10), "white": (0.92,0.92,0.92)}
PARK = (-3.0, 0.0)

# (条件名, 槽1 = (物体, 颜色) 或 None, 槽2 = 同)
CONDS = [
    # 目标槽(槽1)恒为红球;只换干扰槽(槽2)的属性
    ("base",        ("ball1","red"), ("ball2","blue"),   0.0),  # 基线:异色同形
    ("d_red_ball",  ("ball1","red"), ("ball2","red"),    0.0),  # 同色同形 ★最难
    ("d_orange",    ("ball1","red"), ("ball2","orange"), 0.0),  # 近红同形
    ("d_magenta",   ("ball1","red"), ("ball2","magenta"),0.0),
    ("d_yellow",    ("ball1","red"), ("ball2","yellow"), 0.0),
    ("d_green",     ("ball1","red"), ("ball2","green"),  0.0),  # 对立色同形
    ("d_white",     ("ball1","red"), ("ball2","white"),  0.0),  # 无彩色同形
    ("d_grey",      ("ball1","red"), ("ball2","grey"),   0.0),
    ("d_red_cube",  ("ball1","red"), ("cube","red"),     0.0),  # 同色异形 ★
    ("d_red_cone",  ("ball1","red"), ("cone","red"),     0.0),
    ("d_red_cyl",   ("ball1","red"), ("cyl","red"),      0.0),
    ("d_blue_cube", ("ball1","red"), ("cube","blue"),    0.0),  # 异色异形
    ("d_red_small", ("ball1","red"), ("ball_s","red"),   0.0),  # 同色同形,半径 0.5r
    ("d_red_large", ("ball1","red"), ("ball_l","red"),   0.0),  # 同色同形,半径 2r
    ("d_none",      ("ball1","red"), None,               0.0),  # 无干扰对照
]

OBJS = ["ball1","ball2"] + list(EXTRA)
def place(slot1, slot2, r1, b1, r2, b2, lift=0.0):
    want = {}
    for slot, rr, bb in ((slot1, r1, b1), (slot2, r2, b2)):
        if slot is None: continue
        obj, col = slot
        br = torch.deg2rad(bb)
        want[obj] = (torch.stack((rr*torch.cos(br), rr*torch.sin(br)), dim=1), col)
    for i, o in enumerate(OBJS):
        if o in want:
            xy, col = want[o]
            for at in ATTR[o]: at.Set(tuple(float(x) for x in COL[col]))
        else:
            xy = torch.tensor([[PARK[0], PARK[1] + 0.7*i]], device=dev).repeat(env.num_envs, 1)
        if o == "ball1":
            env.target_xy[:] = xy; env.write_target_pose()
            if lift:      # 把目标球抬离地面:接地点上移,视觉大小不变
                pos = env.scene.env_origins.clone(); pos[:, 0:2] += xy
                pos[:, 2] += env.cfg.target_radius + lift
                env.target_view.set_world_poses(positions=pos, indices=list(range(env.num_envs)))
        elif o == "ball2": env.noise_xy[:] = xy; env.write_noise_pose()
        else: env.write_shape_pose(o, xy, lift if o in want else 0.0)

def grab():
    z = torch.zeros((env.num_envs, int(cfg.action_space)), device=dev)
    for _ in range(3): env.step(z)
    img = env.camera.data.output["rgb"]
    return (img[..., :3] if img.shape[-1] > 3 else img).detach().clone()

out = a.out_dir; (out/"img").mkdir(parents=True, exist_ok=True)
g = torch.Generator(device=dev); g.manual_seed(17)
U = lambda lo, hi: torch.rand(a.num_envs, generator=g, device=dev)*(hi-lo)+lo
rows = []
for rd in range(a.rounds):
    r1, b1 = U(1.3, 6.5), U(-35., 35.)
    r2, b2 = U(1.3, 6.5), U(-35., 35.)
    for name, s1, s2, lift in CONDS:
        place(s1, s2, r1, b1, r2, b2, lift)
        img = grab()
        for k in range(a.num_envs):
            fn = f"r{rd:02d}_{name}_{k:03d}.png"
            write_png(img[k].permute(2,0,1).to(torch.uint8).cpu(), str(out/"img"/fn))
            rows.append(dict(img=f"img/{fn}", cond=name, round=rd, env=k,
                             obj1=(s1[0] if s1 else "none"), col1=(s1[1] if s1 else "none"),
                             obj2=(s2[0] if s2 else "none"), col2=(s2[1] if s2 else "none"),
                             rng1=float(r1[k]), bear1=float(b1[k]),
                             rng2=float(r2[k]), bear2=float(b2[k]),
                             scale=({"ball_s":0.5,"ball_l":2.0}.get(s1[0],1.0) if s1 else 1.0),
                             lift=lift))
        print(f"[{rd+1}/{a.rounds}] {name}  累计 {len(rows)}", flush=True)
with (out/"meta.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[DONE] {len(rows)} 帧 -> {out}", flush=True)
env.close(); sim_app.close()
