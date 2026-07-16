"""JSRL + 反向课程 SPR-PPO。

冻结的 stage1_end.pt encoder(与 arm A 逐字节相同的 z)+ PPO,teacher 带路、student 只学最后
一段(反向课程),只收集 student 步(approach B),base spr_ppo_update 原样训练。

    ./IsaacLab/isaaclab.sh -p train/spr_jsrl.py --noise
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

import spr_jsrl_cfg as cfg


parser = argparse.ArgumentParser(description="Train JSRL + reverse-curriculum SPR-PPO.")
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

LOG_FIELDS = [
    "update", "step", "fps", "collect_s", "learn_s", "collect_iters",
    "jsrl_h", "student_frac",
    "jsrl_done", "jsrl_success_rate", "jsrl_success_ema",
    "done", "success", "fail", "timeout",
    "pi_loss", "v_loss", "entropy", "kl", "clip_frac", "grad_norm",
    "a_mag_mean", "stop_frac", "zone_frac",
    "raw_std_mean", "raw_std_min", "raw_std_max",
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


def load_frozen_spr(device: torch.device) -> tuple[SPRStateNet, dict]:
    """照抄 arm A / val 的冻结加载:BN running stats 不再更新(07-10 已实证 train() 会漂移)。"""
    path = Path(cfg.SPR_CKPT)
    if not path.exists():
        raise FileNotFoundError(f"SPR checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    spr = SPRStateNet(**ckpt["spr_cfg"]).to(device)
    spr.load_state_dict({k[len("spr."):]: v for k, v in ckpt["model"].items() if k.startswith("spr.")})
    spr.eval()
    spr.requires_grad_(False)
    return spr, ckpt["spr_cfg"]


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
    robot = env.last_robot_xy[env_id].detach().cpu().numpy()
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


def main() -> None:
    seed = int(cfg.SEED)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    spr, spr_cfg = load_frozen_spr(device)
    model = SPRActorCritic(
        spr=spr, goal_dim=2, act_dim=int(env.cfg.action_space),
        pi_hidden=list(cfg.POLICY_NET), vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)], init_std=float(cfg.STD_INIT),
    ).to(device)
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

    steps_per_update = env.num_envs * int(cfg.N_STEPS)
    total_step = 0
    updates = int(cfg.UPDATES)
    print(f"[INFO] JSRL SPR-PPO out={log_dir} envs={env.num_envs} updates={updates} "
          f"rungs={cfg.JSRL_N_RUNGS}x{cfg.JSRL_H_BASE} h0={curr.h:.0f} std_max={cfg.STD_MAX} device={env.device}")

    start_time = time.perf_counter()
    STOP_EPS = float(task_cfg.STOP_EPS)
    try:
        for update in range(1, updates + 1):
            rollout.reset_cursors()
            collect_start = time.perf_counter()
            model.train()
            spr.eval()  # encoder 永久 eval,防 BN 漂移

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
                    # 交棒:走满 h 步,或 teacher 想停车(只看它的动作幅度,不碰特权状态)
                    t_mag = target.abs().max(dim=1).values
                    handover = (ep_step >= curr.h) | (t_mag < STOP_EPS)
                    student_active |= handover
                    action = torch.where(student_active.unsqueeze(1), student_action, explore_action(target))

                # bootstrap 捕获必须在 add 之前(用本轮 V(image_t))
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

                # env0 轨迹
                traj_writer.writerow(traj_row(env, update, it, total_step + env.num_envs, 0,
                                              bool(student_active[0].item()), reward, done))

                # 统计:整体 + 只统计 student 接手过的 episode(课程判据)
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

            batch = rollout.make_batch(
                last_value=rollout.boot, gamma=float(cfg.GAMMA), lam=float(cfg.GAE_LAMBDA),
                norm_adv=bool(cfg.NORMALIZE_ADVANTAGE), spr_k=1,
            )
            learn_start = time.perf_counter()
            loss = spr_ppo_update(
                model=model, opt=opt, batch=batch,
                batch_size=int(cfg.BATCH_SIZE), epochs=int(cfg.N_EPOCHS),
                clip=float(cfg.CLIP_RANGE), vf_coef=float(cfg.VF_COEF), ent_coef=float(cfg.ENT_COEF),
                spr_coef=0.0, max_grad_norm=float(cfg.MAX_GRAD_NORM), clip_vf=cfg.CLIP_RANGE_VF,
                teacher_coef=0.0, update_spr=False,
                log_std_min=math.log(float(cfg.STD_MIN)), log_std_max=math.log(float(cfg.STD_MAX)),
            )
            loss.pop("spr_loss", None)
            loss.pop("teacher_loss", None)
            learn_s = time.perf_counter() - learn_start

            curr.observe(jsrl_success, jsrl_done)
            curr.step()

            row = {
                "update": update, "step": total_step,
                "fps": total_step / max(time.perf_counter() - start_time, 1e-6),
                "collect_s": collect_s, "learn_s": learn_s, "collect_iters": it,
                "jsrl_h": curr.h, "student_frac": student_steps / max(collect_slots, 1),
                "jsrl_done": jsrl_done, "jsrl_success_rate": jsrl_success / max(jsrl_done, 1),
                "jsrl_success_ema": curr.ema if curr.ema is not None else 0.0,
                "done": done_count, "success": success_count, "fail": fail_count, "timeout": timeout_count,
                "a_mag_mean": a_mag_sum / max(collect_slots, 1),
                "stop_frac": stop_count / max(collect_slots, 1),
                "zone_frac": zone_count / max(collect_slots, 1),
                **loss,
            }
            log_writer.writerow(row)
            log_file.flush()
            traj_file.flush()
            for k in ("pi_loss", "v_loss", "kl", "raw_std_mean"):
                tb.add_scalar(f"train/{k}", float(row[k]), total_step)
            tb.add_scalar("train/jsrl_h", curr.h, total_step)
            tb.add_scalar("train/jsrl_success_rate", row["jsrl_success_rate"], total_step)

            if update % int(cfg.LOG_EVERY) == 0:
                print_log([
                    ("time", [
                        ("update", update),
                        ("step", total_step),
                        ("fps", row["fps"]),
                        ("collect_iters", it),
                    ]),
                    ("curriculum", [
                        ("jsrl_h", curr.h),
                        ("student_frac", row["student_frac"]),
                        ("jsrl_success", row["jsrl_success_rate"]),
                        ("jsrl_success_ema", row["jsrl_success_ema"]),
                        ("jsrl_done", jsrl_done),
                    ]),
                    ("rollout", [
                        ("done", done_count),
                        ("success", success_count / max(done_count, 1)),
                        ("zone_frac", row["zone_frac"]),
                        ("stop_frac", row["stop_frac"]),
                        ("a_mag", row["a_mag_mean"]),
                    ]),
                    ("policy", [
                        ("raw_std", row["raw_std_mean"]),
                    ]),
                    ("train", [
                        ("pi_loss", row["pi_loss"]),
                        ("v_loss", row["v_loss"]),
                        ("kl", row["kl"]),
                    ]),
                ])

            if int(cfg.SAVE_UPDATE_EVERY) > 0 and update % int(cfg.SAVE_UPDATE_EVERY) == 0:
                save_model(update_dir / f"full_{update:06d}.pt", model, spr_cfg, update, total_step, curr.h, row)

        save_model(log_dir / "last.pt", model, spr_cfg, updates, total_step, curr.h, {})
        print(f"[INFO] JSRL done, {(time.perf_counter()-start_time)/3600:.2f} h")
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
