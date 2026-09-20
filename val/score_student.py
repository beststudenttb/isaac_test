"""任务 v3 的确定性评估:64 env 各跑满一条 375 步 episode,报平均回报。

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
parser.add_argument("--ckpt", default="last.pt")
parser.add_argument("--tag", default="")
parser.add_argument("--encoder-root", default=None)
parser.add_argument("--run-tag", default="")  # 训练时的 --tag,拼进 run 目录名
parser.add_argument("--score-mode", default=None, choices=("angle", "cam", "pixel"), help="奖励与 stop 判定的度量,须与该表征对应的 teacher 一致")
parser.add_argument("--chase-blue", action="store_true", help="改追蓝球(干扰球),红球留在场上当干扰;表征不变")
parser.add_argument("--noise-shape", default="", choices=("","sphere","cube","cone"), help="把干扰物换成别的形状再评估(外接盒与球相同)。留空=不动。")
parser.add_argument("--noise-color", default="", help="把干扰球改成别的颜色再评估(只改外观,不改标签/奖励)。留空=不动。例:red / green / grey")
parser.add_argument("--multi", action="store_true", help="4 物体环境(红球/红方/蓝球/蓝方),配合 --target-slot")
parser.add_argument("--target-slot", type=int, default=0, choices=(0,1,2,3), help="0=红球 1=红方 2=蓝球 3=蓝方")
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
    if args_cli.multi:      # 4 物体:0=红球 1=红方 2=蓝球 3=蓝方
        from src.multi_env import SLOT_NAMES, ScoreMultiStudentEnvCfg, make_score_multi_student_env
        env_cfg = ScoreMultiStudentEnvCfg()
        env_cfg.target_slot = int(args_cli.target_slot)
        make_env_fn = make_score_multi_student_env
        print(f"[MULTI] 目标槽位 {args_cli.target_slot} = {SLOT_NAMES[int(args_cli.target_slot)]}", flush=True)
    else:
        env_cfg = ScoreNoiseStudentEnvCfg()
        make_env_fn = make_score_noise_student_env
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
    if not args_cli.multi:
        env_cfg.chase_blue = bool(args_cli.chase_blue)
    if args_cli.noise_shape and not args_cli.multi:
        env_cfg.noise_shape = str(args_cli.noise_shape)
        print(f"[NOISE-SHAPE] 干扰物形状 -> {args_cli.noise_shape}", flush=True)
    env_cfg.end_d_min = env_cfg.end_d_max = float(cfg.END_D_MIN)
    env_cfg.end_x_min = env_cfg.end_x_max = float(cfg.END_X_MIN)
    env = make_env_fn(env_cfg)
    if args_cli.noise_color:   # 只改干扰球的漫反射颜色;标签、奖励、停车判定一律不动
        _COL = {"red": (1.0, 0.0, 0.0), "blue": (0.0, 0.1, 1.0), "green": (0.0, 0.72, 0.1),
                "grey": (0.5, 0.5, 0.5), "orange": (1.0, 0.45, 0.0)}
        _c = _COL[str(args_cli.noise_color)]
        _n = 0
        for _prim in env.scene.stage.Traverse():
            if "/noise" not in str(_prim.GetPath()):
                continue
            for _an in ("inputs:diffuseColor", "inputs:diffuse_color"):
                _at = _prim.GetAttribute(_an)
                if _at and _at.IsValid():
                    _at.Set(tuple(float(x) for x in _c)); _n += 1
        print(f"[NOISE-COLOR] 干扰球改成 {args_cli.noise_color}{_c},改了 {_n} 个材质属性", flush=True)
    device = torch.device(env.device)

    root = Path(args_cli.encoder_root) if args_cli.encoder_root else Path(cfg.ENCODER_ROOT)
    enc_path = root / "encoder_init.pt" if args_cli.arm == "init" else root / args_cli.arm / "encoder_final.pt"
    encoder = FrozenFreeEncoder(str(enc_path), **dict(FREE_SPATIAL_CONFIG)).to(device)
    model = (AdapterActorCritic if args_cli.adapter else FreeFeatureActorCritic)(
        encoder=encoder, **({"adapter_hidden": int(args_cli.adapter_hidden)} if args_cli.adapter else {}), goal_dim=2, act_dim=int(env.cfg.action_space),
        pi_hidden=list(cfg.POLICY_NET), vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)], init_std=float(cfg.STD_INIT),
    ).to(device)
    suffix = f"_s{args_cli.seed}" if args_cli.seed is not None else ""
    run_dir = Path(f"{cfg.OUT_ROOT}_{args_cli.arm}{suffix}{args_cli.run_tag}")
    ck = torch.load(run_dir / args_cli.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

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
    print(f"[RESULT] arm={args_cli.arm}{suffix}  v1成功率 {v1:.1%} 首次成功步 {first:.0f}  回报 mean={ret.mean():.1f}  std={ret.std():.1f}  "
          f"满分={steps}  区内={dwell.mean():.3f}  末100步: xe={t_xe.mean():.1f}px de={t_de.mean():.3f}m "
          f"区内={t_zone.mean():.3f} |a|={t_amag.mean():.3f}", flush=True)
    out = run_dir / f"eval_score{args_cli.tag}.csv"
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
