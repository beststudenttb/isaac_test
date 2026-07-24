"""分阶段 SPR-PPO:forward(无teacher,h=0) 与 JSRL(反向课程) 按阶段表串联。

动机:JSRL 里 teacher 替 student 做了初始 ±45° 转头,a_w 的接近信号被吃掉 -> σ_w 顶死。
先跑一段 forward(student 自己从远处开、自己转头对中),用接近段稠密的 x_gain 信号把 σ_w 收下来,
再切 JSRL 学"敢进区+精确停"。切换时把 a_w 的 σ 上限棘轮压住(只降不升,防区内被吹回),
把 a_x 的 σ floor 抬回来(JSRL 段还要靠它探索刹停)。

阶段表 `PHASES` 决定顺序:Combo A = [forward, jsrl],Combo B = [jsrl, forward]。
encoder 模式 `ENCODER_MODE`:frozen(全程冻结,和 arm A 一致)或 cotrain(仅 forward 段训 encoder,
切 JSRL 时冻结 —— JSRL 只收 student 步会断帧,不能全程 co-train)。

    ./IsaacLab/isaaclab.sh -p train/spr_phased.py --noise
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import spr_phased_cfg as cfg


parser = argparse.ArgumentParser(description="Train phased (forward + JSRL) SPR-PPO.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--random-stop", action="store_true")
parser.add_argument("--noise", action="store_true")
parser.add_argument("--show", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
if not args_cli.show:
    args_cli.headless = True
    args_cli.livestream = 0
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
import isaaclab.sim as sim_utils
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import task_cfg
from src.cv_extractor.spr_state import SPRStateNet
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.noise_env import NoiseStudentEnvCfg, make_noise_student_env
from src.spr_ppo import SPRActorCritic, spr_ppo_update
from src.spr_jsrl import JSRLCurriculum, JSRLRollout


ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
A_W = 2  # 动作维顺序 (a_x, a_y, a_w);a_w 是转头/对齐维。
A_X = 0

LOG_FIELDS = [
    "update", "step", "fps", "collect_s", "learn_s", "collect_iters",
    "phase", "mode", "jsrl_h", "student_frac",
    "jsrl_done", "jsrl_success_rate", "jsrl_success_ema",
    "done", "success", "fail", "timeout",
    "pi_loss", "v_loss", "entropy", "kl", "clip_frac", "grad_norm", "spr_loss",
    "a_mag_mean", "stop_frac", "zone_frac",
    "sigma_x", "sigma_y", "sigma_w", "raw_std_mean", "raw_std_min", "raw_std_max",
]

TRAJ_FIELDS = [
    "phase", "step", "update", "t", "env_id", "student",
    "px_x", "dist", "end_d", "end_x", "a_x", "a_y", "a_w",
    "reward", "done", "success", "fail", "timeout",
]


def out_dir() -> Path:
    path = Path(cfg.RANDOM_STOP_OUT_DIR) if args_cli.random_stop else Path(cfg.OUT_DIR)
    if bool(cfg.CLEAR_OUT_DIR) and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def end_range() -> tuple[float, float, float, float]:
    if args_cli.random_stop:
        return (float(cfg.RANDOM_END_D_MIN), float(cfg.RANDOM_END_D_MAX),
                float(cfg.RANDOM_END_X_MIN), float(cfg.RANDOM_END_X_MAX))
    return (float(cfg.END_D_MIN), float(cfg.END_D_MAX),
            float(cfg.END_X_MIN), float(cfg.END_X_MAX))


def teacher_path() -> str:
    return str(cfg.RANDOM_TEACHER_PATH) if args_cli.random_stop else str(cfg.TEACHER_PATH)


def write_config(path: Path) -> None:
    lines = [f"{n} = {getattr(cfg, n)!r}" for n in dir(cfg) if n.isupper()]
    lines.append(f"RANDOM_STOP = {bool(args_cli.random_stop)!r}")
    lines.append(f"NOISE = {bool(args_cli.noise)!r}")
    lines.append(f"TEACHER_PATH_USED = {teacher_path()!r}")
    lines.append(f"DEVICE_USED = {args_cli.device!r}")
    (path / "config.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def open_csv(path: Path, fields: list[str]):
    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    return file, writer


def make_env() -> MDPStudentEnv:
    end_d_min, end_d_max, end_x_min, end_x_max = end_range()
    env_cfg = NoiseStudentEnvCfg() if args_cli.noise else MDPStudentEnvCfg()
    env_cfg.seed = int(cfg.SEED)
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else int(cfg.NUM_ENVS)
    env_cfg.sim.device = args_cli.device
    render_kwargs = {"rendering_mode": str(cfg.RENDERING_MODE),
                     "antialiasing_mode": str(cfg.ANTIALIASING_MODE), "enable_dlssg": False}
    if cfg.DLSS_MODE is not None:
        render_kwargs["dlss_mode"] = int(cfg.DLSS_MODE)
    env_cfg.sim.render = sim_utils.RenderCfg(**render_kwargs)
    env_cfg.num_rerenders_on_reset = int(cfg.RERENDER_ON_RESET)
    env_cfg.end_d_min = end_d_min
    env_cfg.end_d_max = end_d_max
    env_cfg.end_x_min = end_x_min
    env_cfg.end_x_max = end_x_max
    make_env_fn = make_noise_student_env if args_cli.noise else make_mdp_student_env
    return make_env_fn(env_cfg)


def camera_rgb(env: MDPStudentEnv) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    return image[..., :3] if image.shape[-1] > 3 else image


def load_spr(device: torch.device, trainable: bool) -> tuple[SPRStateNet, dict]:
    """frozen: eval + requires_grad_(False)(07-10 实证 train() 会 BN 漂移);cotrain: 可训。"""
    path = Path(cfg.SPR_CKPT)
    if not path.exists():
        raise FileNotFoundError(f"SPR checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    spr = SPRStateNet(**ckpt["spr_cfg"]).to(device)
    spr.load_state_dict({k[len("spr."):]: v for k, v in ckpt["model"].items() if k.startswith("spr.")})
    if trainable:
        spr.train()
        spr.requires_grad_(True)
    else:
        spr.eval()
        spr.requires_grad_(False)
    return spr, ckpt["spr_cfg"]


def freeze_spr(spr: SPRStateNet) -> None:
    spr.eval()
    spr.requires_grad_(False)


def load_teacher(device: torch.device):
    path = Path(teacher_path())
    if not path.exists():
        raise FileNotFoundError(f"teacher model not found: {path}")
    return PPO.load(str(path), device=device)


def teacher_actions(teacher, priv_obs: torch.Tensor) -> torch.Tensor:
    actions, _ = teacher.predict(priv_obs.detach().cpu().numpy(), deterministic=True)
    actions = torch.as_tensor(actions, device=priv_obs.device, dtype=priv_obs.dtype)
    x_ok = torch.abs(priv_obs[:, 0] - priv_obs[:, 3]) <= float(task_cfg.STOP_X_TOL) / (float(task_cfg.IMAGE_WIDTH) * 0.5)
    d_ok = torch.abs(priv_obs[:, 1] - priv_obs[:, 2]) <= float(task_cfg.STOP_D_TOL) / float(task_cfg.FAIL_FAR)
    actions[x_ok & d_ok] = 0.0
    return actions


def explore_action(target: torch.Tensor) -> torch.Tensor:
    action = target.clone()
    if float(cfg.TEACHER_NOISE_STD) > 0.0:
        action = action + torch.randn_like(action) * float(cfg.TEACHER_NOISE_STD)
    return torch.clamp(action, -1.0, 1.0)


def fmt_value(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.3f}"
    return str(value)


def print_log(groups: list[tuple[str, list[tuple[str, object]]]]) -> None:
    rows: list[tuple[str, str]] = []
    for group, items in groups:
        rows.append((f"{group}/", ""))
        rows.extend((f"  {key}", fmt_value(value)) for key, value in items)
    width_l = max(len(left) for left, _ in rows)
    width_r = max(len(right) for _, right in rows)
    line = "-" * (width_l + width_r + 7)
    print(line)
    for left, right in rows:
        print(f"| {left:<{width_l}} | {right:>{width_r}} |")
    print(line)


def save_model(path: Path, model: SPRActorCritic, spr_cfg: dict, update: int, step: int, jsrl_h: float, info: dict):
    torch.save(
        {
            "model": model.state_dict(),
            "spr_cfg": spr_cfg,
            "update": int(update),
            "step": int(step),
            "jsrl_h": float(jsrl_h),
            "info": info,
        },
        path,
    )


def traj_row(env, update, t, t_now, env_id, student, reward, done):
    px = env.last_px_x[env_id].item()
    dist = env.last_dist[env_id].item()
    a = env.last_move_actions[env_id].detach().cpu().numpy()
    return {
        "phase": "train", "step": int(t_now), "update": int(update), "t": int(t), "env_id": int(env_id),
        "student": int(bool(student)),
        "px_x": f"{px:.4f}", "dist": f"{dist:.4f}",
        "end_d": f"{env.end_d[env_id].item():.4f}", "end_x": f"{env.end_x[env_id].item():.4f}",
        "a_x": f"{a[0]:.4f}", "a_y": f"{a[1]:.4f}", "a_w": f"{a[2]:.4f}",
        "reward": f"{reward[env_id].item():.4f}", "done": int(done[env_id].item()),
        "success": int((env.last_success[env_id] & done[env_id]).item()),
        "fail": int((env.last_fail[env_id] & done[env_id]).item()),
        "timeout": int((env.last_timeout[env_id] & done[env_id]).item()),
    }


def sigma_of(model: SPRActorCritic) -> tuple[float, float, float]:
    s = torch.exp(model.log_std.detach()).cpu().tolist()
    return float(s[0]), float(s[1]), float(s[2])


def main() -> None:
    seed = int(cfg.SEED)
    np.random.seed(seed)
    torch.manual_seed(seed)

    phases = list(cfg.PHASES)
    encoder_mode = str(cfg.ENCODER_MODE)
    cotrain = encoder_mode == "cotrain"

    log_dir = out_dir()
    write_config(log_dir)
    log_file, log_writer = open_csv(log_dir / "log.csv", LOG_FIELDS)
    traj_file, traj_writer = open_csv(log_dir / "traj_train_env0.csv", TRAJ_FIELDS)
    update_dir = log_dir / "updates"
    if int(cfg.SAVE_UPDATE_EVERY) > 0:
        update_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir / "tb"))

    env = make_env()
    device = torch.device(env.device)
    # cotrain 先可训;frozen 全程冻结。cotrain 切 JSRL 时会 freeze_spr()。
    spr, spr_cfg = load_spr(device, trainable=cotrain)
    model = SPRActorCritic(
        spr=spr, goal_dim=2, act_dim=int(env.cfg.action_space),
        pi_hidden=list(cfg.POLICY_NET), vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)], init_std=float(cfg.STD_INIT),
    ).to(device)
    # 末层小增益初始化:让初始 μ≈0(标准 PPO 策略头做法)。
    actor_head = [m for m in model.actor if isinstance(m, nn.Linear)][-1]
    actor_head.weight.data.mul_(0.01)
    nn.init.zeros_(actor_head.bias)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.LR), eps=1e-5)
    teacher = load_teacher(device)

    curr = JSRLCurriculum(
        h_base=cfg.JSRL_H_BASE, n_rungs=cfg.JSRL_N_RUNGS,
        up_thresh=cfg.JSRL_UP_THRESH, down_thresh=cfg.JSRL_DOWN_THRESH, ema=cfg.JSRL_EMA, cooldown=cfg.JSRL_COOLDOWN,
    )
    rollout = JSRLRollout(
        steps=int(cfg.N_STEPS), envs=env.num_envs,
        image_shape=(int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3),
        goal_dim=2, act_dim=int(env.cfg.action_space), device=device,
    )

    obs, _ = env.reset()
    priv_t = obs["policy"]
    image_t = camera_rgb(env)
    goal_t = env.end_obs()

    ep_step = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    student_active = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    total_step = 0
    updates = int(cfg.UPDATES)

    # 阶段状态
    phase_idx = 0
    phase = phases[0]
    phase_start_update = 1
    sigma_w_ratchet = float("inf")   # a_w 的单向下棘轮上限(只降不升)
    spr_frozen = not cotrain          # cotrain 在 forward 段不冻;frozen 一直冻

    def enter_phase(idx: int, update: int) -> None:
        nonlocal phase, phase_idx, phase_start_update, sigma_w_ratchet, spr_frozen
        phase_idx = idx
        phase = phases[idx]
        phase_start_update = update
        if phase.get("ratchet_w"):
            sigma_w_ratchet = sigma_of(model)[A_W]  # 从当前(forward 收敛值)起棘轮
        if phase["mode"] == "jsrl":
            curr.idx = curr.n_rungs - 1               # jsrl 从最高档起手
            curr.h = curr.fractions[curr.idx] * curr.h_base
            curr.ema = None
            curr.cooldown = 0
            if cotrain and not spr_frozen:            # 切 JSRL -> 冻结 encoder(断帧,不能 co-train)
                freeze_spr(spr)
                spr_frozen = True
        print(f"[INFO] === enter phase {idx} '{phase.get('name', phase['mode'])}' "
              f"mode={phase['mode']} std_max={phase['std_max']} at update {update} ===")

    print(f"[INFO] phased SPR-PPO out={log_dir} envs={env.num_envs} updates={updates} "
          f"encoder={encoder_mode} phases={[p.get('name', p['mode']) for p in phases]} device={env.device}")
    enter_phase(0, 1)

    start_time = time.perf_counter()
    STOP_EPS = float(task_cfg.STOP_EPS)
    try:
        for update in range(1, updates + 1):
            mode = phase["mode"]
            std_max = float(phase["std_max"])
            std_min = float(phase.get("std_min", cfg.STD_MIN))
            h_eff = 0.0 if mode == "forward" else curr.h
            spr_train_now = cotrain and (not spr_frozen) and mode == "forward"

            rollout.reset_cursors()
            collect_start = time.perf_counter()
            model.train()
            if not spr_train_now:
                spr.eval()  # frozen 或已冻结的 encoder 保持 eval(防 BN 漂移)

            done_count = success_count = fail_count = timeout_count = 0
            jsrl_done = jsrl_success = 0
            student_steps = 0
            a_mag_sum = 0.0
            stop_count = zone_count = collect_slots = 0
            it = 0
            while not rollout.full() and it < int(cfg.MAX_COLLECT_ITERS):
                it += 1
                with torch.no_grad():
                    student_action, logp, value = model.act(image_t, goal_t)
                    target = teacher_actions(teacher, priv_t)
                    t_mag = target.abs().max(dim=1).values
                    handover = (ep_step >= h_eff) | (t_mag < STOP_EPS)
                    student_active |= handover
                    action = torch.where(student_active.unsqueeze(1), student_action, explore_action(target))

                rollout.capture_bootstrap(value)

                obs_next, reward, terminated, truncated, _ = env.step(action)
                timeout = truncated & (~terminated)
                done = terminated | truncated
                next_image = camera_rgb(env)
                next_goal = env.end_obs()
                if torch.any(done):
                    next_image = next_image.clone()
                    next_goal = next_goal.clone()
                    next_image[done] = env.terminal_rgb[done]
                    next_goal[done] = env.terminal_end_obs[done]
                if torch.any(timeout):
                    with torch.no_grad():
                        terminal_value = model.value(env.terminal_rgb[timeout], env.terminal_end_obs[timeout])
                    reward = reward.clone()
                    reward[timeout] += float(cfg.GAMMA) * terminal_value

                rollout.add_student(student_active, image_t, next_image, goal_t, action, logp, reward, done, value)
                student_steps += int(student_active.sum().item())

                traj_writer.writerow(traj_row(env, update, it, total_step + env.num_envs, 0,
                                              bool(student_active[0].item()), reward, done))

                a_mag = torch.clamp(action, -1.0, 1.0).abs().max(dim=1).values
                a_mag_sum += float(a_mag.sum().cpu()); collect_slots += env.num_envs
                stop_count += int((a_mag < STOP_EPS).sum().item())
                zone_count += int(env.in_stop_zone().sum().item())
                if torch.any(done):
                    done_count += int(done.sum().item())
                    success_count += int((env.last_success & done).sum().item())
                    fail_count += int((env.last_fail & done).sum().item())
                    timeout_count += int(((env.last_timeout | truncated) & done).sum().item())
                    acted = done & student_active
                    jsrl_done += int(acted.sum().item())
                    jsrl_success += int((env.last_success & acted).sum().item())
                    ep_step = torch.where(done, torch.zeros_like(ep_step), ep_step)
                    student_active = student_active & ~done

                ep_step = ep_step + 1
                total_step += env.num_envs
                priv_t = obs_next["policy"]
                image_t = camera_rgb(env)
                goal_t = env.end_obs()

            with torch.no_grad():
                last_value = model.value(image_t, goal_t)
            rollout.finalize_bootstrap(last_value)
            collect_s = time.perf_counter() - collect_start

            spr_k = int(cfg.SPR_K) if spr_train_now else 1
            batch = rollout.make_batch(
                last_value=rollout.boot, gamma=float(cfg.GAMMA), lam=float(cfg.GAE_LAMBDA),
                norm_adv=bool(cfg.NORMALIZE_ADVANTAGE), spr_k=spr_k,
            )
            learn_start = time.perf_counter()
            loss = spr_ppo_update(
                model=model, opt=opt, batch=batch,
                batch_size=int(cfg.BATCH_SIZE), epochs=int(cfg.N_EPOCHS),
                clip=float(cfg.CLIP_RANGE), vf_coef=float(cfg.VF_COEF), ent_coef=float(cfg.ENT_COEF),
                spr_coef=float(cfg.SPR_COEF) if spr_train_now else 0.0,
                max_grad_norm=float(cfg.MAX_GRAD_NORM), clip_vf=cfg.CLIP_RANGE_VF,
                teacher_coef=0.0, update_spr=spr_train_now,
                log_std_min=math.log(std_min), log_std_max=math.log(std_max),
            )
            if spr_train_now:
                model.spr.update_target()
            spr_loss_val = float(loss.pop("spr_loss", 0.0) or 0.0)
            loss.pop("teacher_loss", None)
            learn_s = time.perf_counter() - learn_start

            # per-dim σ 处理(spr_ppo_update 只做了标量 clamp,这里做 per-dim):
            #   ratchet_w:a_w 只降不升(防切 JSRL 后区内被吹回);floor_x:a_x 抬回 floor(JSRL 段留探索)。
            with torch.no_grad():
                if phase.get("ratchet_w"):
                    cur_w = math.exp(float(model.log_std.data[A_W]))
                    sigma_w_ratchet = min(sigma_w_ratchet, cur_w)
                    model.log_std.data[A_W].clamp_(max=math.log(sigma_w_ratchet))
                if phase.get("floor_x") is not None:
                    model.log_std.data[A_X].clamp_(min=math.log(float(phase["floor_x"])))

            sx, sy, sw = sigma_of(model)

            if mode == "jsrl":
                curr.observe(jsrl_success, jsrl_done)
                curr.step()

            row = {
                "update": update, "step": total_step,
                "fps": total_step / max(time.perf_counter() - start_time, 1e-6),
                "collect_s": collect_s, "learn_s": learn_s, "collect_iters": it,
                "phase": phase_idx, "mode": mode,
                "jsrl_h": curr.h if mode == "jsrl" else 0.0,
                "student_frac": student_steps / max(collect_slots, 1),
                "jsrl_done": jsrl_done, "jsrl_success_rate": jsrl_success / max(jsrl_done, 1),
                "jsrl_success_ema": curr.ema if curr.ema is not None else 0.0,
                "done": done_count, "success": success_count, "fail": fail_count, "timeout": timeout_count,
                "spr_loss": spr_loss_val,
                "a_mag_mean": a_mag_sum / max(collect_slots, 1),
                "stop_frac": stop_count / max(collect_slots, 1),
                "zone_frac": zone_count / max(collect_slots, 1),
                "sigma_x": sx, "sigma_y": sy, "sigma_w": sw,
                **loss,
            }
            log_writer.writerow(row)
            log_file.flush()
            traj_file.flush()
            for k in ("pi_loss", "v_loss", "kl"):
                tb.add_scalar(f"train/{k}", float(row[k]), total_step)
            tb.add_scalar("train/sigma_w", sw, total_step)
            tb.add_scalar("train/sigma_x", sx, total_step)
            tb.add_scalar("train/jsrl_h", row["jsrl_h"], total_step)

            if update % int(cfg.LOG_EVERY) == 0:
                print_log([
                    ("time", [("update", update), ("phase", f"{phase_idx}:{mode}"),
                              ("step", total_step), ("fps", row["fps"])]),
                    ("sigma", [("a_x", sx), ("a_y", sy), ("a_w", sw)]),
                    ("rollout", [("done", done_count),
                                 ("success", success_count / max(done_count, 1)),
                                 ("zone_frac", row["zone_frac"]),
                                 ("student_frac", row["student_frac"])]),
                    ("curriculum", [("jsrl_h", row["jsrl_h"]),
                                    ("jsrl_success_ema", row["jsrl_success_ema"])]),
                    ("train", [("pi_loss", row["pi_loss"]), ("v_loss", row["v_loss"]),
                               ("kl", row["kl"]), ("spr_loss", spr_loss_val)]),
                ])

            if int(cfg.SAVE_UPDATE_EVERY) > 0 and update % int(cfg.SAVE_UPDATE_EVERY) == 0:
                save_model(update_dir / f"full_{update:06d}.pt", model, spr_cfg, update, total_step, row["jsrl_h"], row)

            # ---- 阶段切换判据 ----
            if phase_idx < len(phases) - 1:
                phase_upd = update - phase_start_update + 1
                switch = False
                if mode == "forward":
                    thr = phase.get("switch_sigma_w")
                    min_u = int(phase.get("min_updates", 0))
                    if thr is not None and phase_upd >= min_u and sw < float(thr):
                        print(f"[INFO] forward σ_w={sw:.3f} < {thr} at update {update} -> switch")
                        switch = True
                if phase_upd >= int(phase.get("max_updates", updates)):
                    if mode == "forward" and phase.get("switch_sigma_w") is not None and not switch:
                        print(f"[WARN] forward σ_w={sw:.3f} 未达阈值,phase 到 max_updates 强制切换(假设可能证伪)")
                    switch = True
                if switch:
                    enter_phase(phase_idx + 1, update + 1)

        save_model(log_dir / "last.pt", model, spr_cfg, updates, total_step, curr.h, {})
        print(f"[INFO] phased done, {(time.perf_counter()-start_time)/3600:.2f} h")
    finally:
        tb.close()
        log_file.close()
        traj_file.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
