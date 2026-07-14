"""Train PPO student with basic SPR visual state.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p train/spr_student.py
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import spr_student_cfg as cfg


parser = argparse.ArgumentParser(description="Train PPO student with SPR visual state.")
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
from src.spr_ppo import SPRActorCritic, SPRRollout, spr_diagnostics, spr_only_update, spr_ppo_update


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

TRAJ_FIELDS = [
    "phase",
    "step",
    "update",
    "t",
    "env_id",
    "px_x",
    "dist",
    "end_d",
    "end_x",
    "a_x",
    "a_y",
    "a_w",
    "t_ax",
    "t_ay",
    "t_aw",
    "reward",
    "done",
    "success",
    "fail",
    "timeout",
    "robot_x",
    "robot_y",
    "head_yaw",
    "target_x",
    "target_y",
]

LOG_FIELDS = [
    "update",
    "stage",
    "step",
    "fps",
    "sample_fps",
    "collect_s",
    "learn_s",
    "done",
    "success",
    "fail",
    "timeout",
    "success_rate",
    "fail_rate",
    "timeout_rate",
    "pi_loss",
    "v_loss",
    "teacher_loss",
    "entropy",
    "kl",
    "clip_frac",
    "grad_norm",
    "a_mag_mean",
    "stop_frac",
    "zone_frac",
    "raw_std_mean",
    "raw_std_min",
    "raw_std_max",
    "std_mean",
    "std_min",
    "std_max",
    "teacher_coef",
]

SPR_LOG_FIELDS = [
    "update",
    "stage",
    "step",
    "spr_loss",
    "eff_rank",
    "z_std",
    "pred_cos",
    "identity_gap",
    "target_sim",
]

VAL_FIELDS = [
    "update",
    "step",
    "success_rate",
    "fail_rate",
    "timeout_rate",
    "mean_return",
    "mean_len",
    "std_mean",
    "std_min",
    "std_max",
    "teacher_coef",
]


def out_dir() -> Path:
    path = Path(cfg.RANDOM_STOP_OUT_DIR) if args_cli.random_stop else Path(cfg.OUT_DIR)
    if bool(cfg.CLEAR_OUT_DIR) and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def end_range() -> tuple[float, float, float, float]:
    if args_cli.random_stop:
        return (
            float(cfg.RANDOM_END_D_MIN),
            float(cfg.RANDOM_END_D_MAX),
            float(cfg.RANDOM_END_X_MIN),
            float(cfg.RANDOM_END_X_MAX),
        )
    return (
        float(cfg.END_D_MIN),
        float(cfg.END_D_MAX),
        float(cfg.END_X_MIN),
        float(cfg.END_X_MAX),
    )


def teacher_path() -> str:
    if args_cli.random_stop:
        return str(cfg.RANDOM_TEACHER_PATH)
    return str(cfg.TEACHER_PATH)


def write_config(path: Path) -> None:
    lines = []
    for name in dir(cfg):
        if name.isupper():
            lines.append(f"{name} = {getattr(cfg, name)!r}")
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
    render_kwargs = {
        "rendering_mode": str(cfg.RENDERING_MODE),
        "antialiasing_mode": str(cfg.ANTIALIASING_MODE),
        "enable_dlssg": False,
    }
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
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def load_teacher(device: torch.device):
    if (
        float(cfg.TEACHER_LOSS) <= 0.0
        and float(cfg.TEACHER_UP_OUT) <= 0.0
        and float(cfg.TEACHER_UP_TIMEOUT) <= 0.0
    ):
        return None
    path = Path(teacher_path())
    if not path.exists():
        raise FileNotFoundError(f"teacher model not found: {path}")
    return PPO.load(str(path), device=device)


def teacher_actions(teacher, priv_obs: torch.Tensor) -> torch.Tensor | None:
    if teacher is None:
        return None
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
    if float(cfg.RAND_ACTION_P) > 0.0:
        rand = torch.rand(action.shape[0], device=action.device) < float(cfg.RAND_ACTION_P)
        action[rand] = torch.empty_like(action[rand]).uniform_(-1.0, 1.0)
    return torch.clamp(action, -1.0, 1.0)


def stage_of(step: int) -> int:
    s1 = int(cfg.STAGE1_STEPS)
    s2 = s1 + int(cfg.STAGE2_STEPS)
    s3 = s2 + int(cfg.STAGE3_STEPS)
    if step < s1:
        return 1
    if step < s2:
        return 2
    if step < s3:
        return 3
    return 4


def stage_teacher_coef(step: int, stage: int) -> float:
    if stage in (1, 4):
        return 0.0
    if stage == 2:
        return float(cfg.TEACHER_LOSS)
    s2_end = int(cfg.STAGE1_STEPS) + int(cfg.STAGE2_STEPS)
    frac = (step - s2_end) / max(int(cfg.STAGE3_STEPS), 1)
    return float(cfg.TEACHER_LOSS) * max(1.0 - frac, 0.0)


def scheduled_std(update: int) -> float:
    """外生 σ 调度:stage1 结束后线性 STD_START -> STD_END,之后保持 STD_END。

    消融用:PPO 的 log_std 梯度在本任务 reward 下拿不到信号(SPR 线两次都被钉在 STD_MAX
    的 clamp 上),而成功要求三维动作同时 |a_i| < STOP_EPS,发现概率 ~ [2Φ(STOP_EPS/σ)-1]³。
    两条成功的线(MDP anneal 自由衰减、DrQ-v2 外生调度)都在 σ≈0.12-0.13 处 success 起飞。
    """
    span = max(int(cfg.STD_ANNEAL_UPDATES), 1)
    frac = min(max((update - int(cfg.STAGE1_UPDATES)) / span, 0.0), 1.0)
    return (1.0 - frac) * float(cfg.STD_START) + frac * float(cfg.STD_END)


def std_info(model: SPRActorCritic) -> dict[str, float]:
    std = torch.exp(model.log_std.detach())
    return {
        "std_mean": float(std.mean().cpu()),
        "std_min": float(std.min().cpu()),
        "std_max": float(std.max().cpu()),
    }


def traj_row(
    env: MDPStudentEnv,
    phase: str,
    step: int,
    update: int,
    t: int,
    env_id: int,
    reward: torch.Tensor,
    done: bool,
    teacher_a: torch.Tensor,
) -> dict:
    action = env.last_move_actions[env_id].detach().cpu().numpy()
    robot_xy = env.last_robot_xy[env_id].detach().cpu().numpy()
    target_xy = env.last_target_xy[env_id].detach().cpu().numpy()
    return {
        "phase": phase,
        "step": int(step),
        "update": int(update),
        "t": int(t),
        "env_id": int(env_id),
        "px_x": f"{float(env.last_px_x[env_id].item()):.4f}",
        "dist": f"{float(env.last_dist[env_id].item()):.4f}",
        "end_d": f"{float(env.end_d[env_id].item()):.4f}",
        "end_x": f"{float(env.end_x[env_id].item()):.4f}",
        "a_x": f"{float(action[0]):.4f}",
        "a_y": f"{float(action[1]):.4f}",
        "a_w": f"{float(action[2]):.4f}",
        "t_ax": f"{float(teacher_a[0]):.4f}",
        "t_ay": f"{float(teacher_a[1]):.4f}",
        "t_aw": f"{float(teacher_a[2]):.4f}",
        "reward": f"{float(reward.item()):.4f}",
        "done": int(done),
        "success": int(bool(env.last_success[env_id].item())),
        "fail": int(bool(env.last_fail[env_id].item())),
        "timeout": int(bool(env.last_timeout[env_id].item())),
        "robot_x": f"{float(robot_xy[0]):.4f}",
        "robot_y": f"{float(robot_xy[1]):.4f}",
        "head_yaw": f"{float(env.last_head_yaw[env_id].item()):.4f}",
        "target_x": f"{float(target_xy[0]):.4f}",
        "target_y": f"{float(target_xy[1]):.4f}",
    }


def spr_cfg_dict() -> dict:
    return {
        "input_shape": (int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3),
        "z_dim": int(cfg.Z_DIM),
        "action_dim": 3,
        "fpn_dim": int(cfg.FPN_DIM),
        "hidden_dim": int(cfg.SPR_HIDDEN),
        "pool_size": int(cfg.SPR_POOL),
        "pretrained": False,
        "target_tau": float(cfg.SPR_TAU),
        "freeze_stem_layers": bool(cfg.FREEZE_STEM_LAYERS),
    }


def save_model(path: Path, model: SPRActorCritic, opt: torch.optim.Optimizer, update: int, step: int, best: float, teacher_coef: float, info: dict):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "update": int(update),
            "step": int(step),
            "best": float(best),
            "teacher_coef": float(teacher_coef),
            "spr_cfg": spr_cfg_dict(),
            "info": info,
        },
        path,
    )


def save_spr(path: Path, model: SPRActorCritic, update: int, step: int, info: dict):
    torch.save(
        {
            "spr": model.spr.state_dict(),
            "update": int(update),
            "step": int(step),
            "spr_cfg": spr_cfg_dict(),
            "info": info,
        },
        path,
    )


def save_policy(path: Path, model: SPRActorCritic, update: int, step: int, teacher_coef: float, info: dict):
    torch.save(
        {
            "actor": model.actor.state_dict(),
            "critic": model.critic.state_dict(),
            "log_std": model.log_std.detach().cpu(),
            "update": int(update),
            "step": int(step),
            "teacher_coef": float(teacher_coef),
            "spr_ckpt": "stage1_end.pt",
            "info": info,
        },
        path,
    )


def val(env: MDPStudentEnv, model: SPRActorCritic, max_steps: int, step: int, update: int) -> dict[str, float]:
    env.reset()
    image = camera_rgb(env)
    goal = env.end_obs()
    steps = int(max_steps) if int(max_steps) > 0 else int(env.max_episode_length)
    done_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros(env.num_envs, device=env.device)
    success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    fail = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    timeout = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            action = model.predict(image, goal)
            _obs, reward, terminated, truncated, _ = env.step(action)
            active = ~done_once
            returns[active] += reward[active]
            lengths[active] += 1
            done = (terminated | truncated) & active
            if torch.any(done):
                success[done] = env.last_success[done]
                fail[done] = env.last_fail[done]
                timeout[done] = env.last_timeout[done] | truncated[done]
                done_once[done] = True
                if bool(done_once.all()):
                    break
            image = camera_rgb(env)
            goal = env.end_obs()
    model.train()
    timeout[~done_once] = True
    lengths[~done_once] = steps
    return {
        "success_rate": float(success.float().mean().cpu()),
        "fail_rate": float(fail.float().mean().cpu()),
        "timeout_rate": float(timeout.float().mean().cpu()),
        "mean_return": float(returns.mean().cpu()),
        "mean_len": float(lengths.mean().cpu()),
    }


def fmt_value(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.3f}"
    return str(value)


def print_log(groups: list[tuple[str, list[tuple[str, object]]]]) -> None:
    rows = []
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


def main() -> None:
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = out_dir()
    write_config(log_dir)
    log_file, log_writer = open_csv(log_dir / "log.csv", LOG_FIELDS)
    spr_file, spr_writer = open_csv(log_dir / "spr_log.csv", SPR_LOG_FIELDS)
    val_file, val_writer = open_csv(log_dir / "val.csv", VAL_FIELDS)
    traj_file, traj_writer = open_csv(log_dir / "traj_train_env0.csv", TRAJ_FIELDS)
    update_dir = log_dir / "updates"
    if int(cfg.SAVE_UPDATE_EVERY) > 0:
        update_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir / "tb"))

    env = make_env()
    device = torch.device(env.device)
    spr = SPRStateNet(
        input_shape=(int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3),
        z_dim=int(cfg.Z_DIM),
        action_dim=int(env.cfg.action_space),
        fpn_dim=int(cfg.FPN_DIM),
        hidden_dim=int(cfg.SPR_HIDDEN),
        pool_size=int(cfg.SPR_POOL),
        pretrained=bool(cfg.SPR_PRETRAINED),
        target_tau=float(cfg.SPR_TAU),
        freeze_stem_layers=bool(cfg.FREEZE_STEM_LAYERS),
    ).to(device)
    model = SPRActorCritic(
        spr=spr,
        goal_dim=2,
        act_dim=int(env.cfg.action_space),
        pi_hidden=list(cfg.POLICY_NET),
        vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)],
        init_std=float(cfg.STD_INIT),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.LR), eps=1e-5)
    if bool(cfg.STD_SCHEDULE):
        model.log_std.requires_grad_(False)  # σ 由调度决定,不再由 PPO 的梯度学。
        print(f"[INFO] std schedule ON: {cfg.STD_START} -> {cfg.STD_END} over {cfg.STD_ANNEAL_UPDATES} updates (log_std frozen, clamp disabled)")
    teacher = load_teacher(device)
    teacher_coef = 0.0
    if (int(cfg.STAGE1_STEPS) > 0 or int(cfg.STAGE2_STEPS) > 0 or int(cfg.STAGE3_STEPS) > 0) and teacher is None:
        raise ValueError("stages 1-3 require a teacher policy")

    obs, _ = env.reset()
    priv_t = obs["policy"]
    image_t = camera_rgb(env)
    goal_t = env.end_obs()

    steps_per_update = env.num_envs * int(cfg.N_STEPS)
    total_steps = int(cfg.TOTAL_STEPS)
    updates = int(math.ceil(total_steps / steps_per_update))
    stage4_steps = max(total_steps - int(cfg.STAGE1_STEPS) - int(cfg.STAGE2_STEPS) - int(cfg.STAGE3_STEPS), 0)
    print(
        f"[INFO] spr student PPO out={log_dir} envs={env.num_envs} total_steps={total_steps} "
        f"updates={updates} stage_steps={cfg.STAGE1_STEPS}/{cfg.STAGE2_STEPS}/{cfg.STAGE3_STEPS}/{stage4_steps} "
        f"n_steps={cfg.N_STEPS} z_dim={cfg.Z_DIM} device={env.device}"
    )

    best = -1.0
    last_info = {}
    prev_stage = 0
    start_time = time.perf_counter()
    try:
        for update in range(1, updates + 1):
            step_start = (update - 1) * steps_per_update
            stage = stage_of(step_start)
            teacher_coef = stage_teacher_coef(step_start, stage)
            if bool(cfg.STD_SCHEDULE):
                with torch.no_grad():
                    model.log_std.fill_(math.log(scheduled_std(update)))
            if prev_stage != 0 and stage != prev_stage:
                prev_coef = float(last_info.get("teacher_coef", 0.0))
                save_model(log_dir / f"stage{prev_stage}_end.pt", model, opt, update - 1, step_start, best, prev_coef, last_info)
                obs, _ = env.reset()
                priv_t = obs["policy"]
                image_t = camera_rgb(env)
                goal_t = env.end_obs()
                print(f"[INFO] stage {prev_stage} -> {stage}: env reset, saved stage{prev_stage}_end.pt")
            prev_stage = stage
            rollout = SPRRollout(
                steps=int(cfg.N_STEPS),
                envs=env.num_envs,
                image_shape=(int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3),
                goal_dim=2,
                act_dim=int(env.cfg.action_space),
                device=device,
            )
            priv_buf = torch.zeros((int(cfg.N_STEPS), env.num_envs, int(env.cfg.observation_space)), device=device)
            done_count = 0
            success_count = 0
            fail_count = 0
            timeout_count = 0
            a_mag_sum = 0.0
            stop_count = 0
            zone_count = 0

            collect_start = time.perf_counter()
            model.train()
            for t in range(int(cfg.N_STEPS)):
                with torch.no_grad():
                    if stage == 1:
                        target = teacher_actions(teacher, priv_t)
                        teacher_a0 = target[0]
                        action = explore_action(target)
                        logp = torch.zeros(env.num_envs, device=device)
                        value = torch.zeros(env.num_envs, device=device)
                    else:
                        action, logp, value = model.act(image_t, goal_t)
                        if teacher is not None:
                            teacher_a0 = teacher_actions(teacher, priv_t[0:1])[0]
                        else:
                            teacher_a0 = torch.zeros(int(env.cfg.action_space), device=device)
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
                if torch.any(timeout) and stage != 1:
                    with torch.no_grad():
                        terminal_value = model.value(env.terminal_rgb[timeout], env.terminal_end_obs[timeout])
                    reward = reward.clone()
                    reward[timeout] += float(cfg.GAMMA) * terminal_value
                rollout.add(t, image_t, next_image, goal_t, action, logp, reward, done, value)
                priv_buf[t].copy_(priv_t.detach())
                a_mag = torch.clamp(action, -1.0, 1.0).abs().max(dim=1).values
                a_mag_sum += float(a_mag.sum().cpu())
                stop_count += int((a_mag < float(task_cfg.STOP_EPS)).sum().item())
                zone_count += int(env.in_stop_zone().sum().item())

                step_now = min((update - 1) * steps_per_update + (t + 1) * env.num_envs, total_steps)
                traj_writer.writerow(traj_row(env, "train", step_now, update, t, 0, reward[0], bool(done[0]), teacher_a0))
                if torch.any(done):
                    done_count += int(done.sum().item())
                    success_count += int((env.last_success & done).sum().item())
                    fail_count += int((env.last_fail & done).sum().item())
                    timeout_count += int(((env.last_timeout | truncated) & done).sum().item())
                priv_t = obs_next["policy"]
                image_t = camera_rgb(env)
                goal_t = env.end_obs()

            with torch.no_grad():
                last_value = model.value(image_t, goal_t)
            batch = rollout.make_batch(
                last_value=last_value,
                gamma=float(cfg.GAMMA),
                lam=1.0 if stage == 1 else float(cfg.GAE_LAMBDA),
                norm_adv=bool(cfg.NORMALIZE_ADVANTAGE),
                spr_k=int(cfg.SPR_K),
            )
            priv_batch = priv_buf.reshape(-1, priv_buf.shape[-1])
            batch.teacher_actions = (
                teacher_actions(teacher, priv_batch) if stage in (2, 3) and teacher is not None and teacher_coef > 0.0 else None
            )
            collect_s = time.perf_counter() - collect_start
            learn_start = time.perf_counter()
            update_spr = False
            if stage == 1:
                update_spr = True
                loss = spr_only_update(
                    model=model,
                    opt=opt,
                    batch=batch,
                    batch_size=int(cfg.BATCH_SIZE),
                    epochs=int(cfg.N_EPOCHS),
                    spr_coef=float(cfg.SPR_COEF),
                    max_grad_norm=float(cfg.MAX_GRAD_NORM),
                )
            else:
                update_spr = stage == 4 and int(cfg.SPR_UPDATE_EVERY) > 0 and update % int(cfg.SPR_UPDATE_EVERY) == 0
                loss = spr_ppo_update(
                    model=model,
                    opt=opt,
                    batch=batch,
                    batch_size=int(cfg.BATCH_SIZE),
                    epochs=int(cfg.N_EPOCHS),
                    clip=float(cfg.CLIP_RANGE),
                    vf_coef=float(cfg.VF_COEF),
                    ent_coef=float(cfg.ENT_COEF),
                    spr_coef=float(cfg.SPR_COEF),
                    max_grad_norm=float(cfg.MAX_GRAD_NORM),
                    clip_vf=cfg.CLIP_RANGE_VF,
                    teacher_coef=teacher_coef,
                    teacher_loss_type=str(cfg.TEACHER_LOSS_TYPE),
                    update_spr=update_spr,
                    actor_encoder_coef=float(cfg.ACTOR_ENCODER_COEF),
                    log_std_min=None if bool(cfg.STD_SCHEDULE) else math.log(float(cfg.STD_MIN)),
                    log_std_max=None if bool(cfg.STD_SCHEDULE) else math.log(float(cfg.STD_MAX)),
                )
            learn_s = time.perf_counter() - learn_start
            step_done = min(update * steps_per_update, total_steps)
            spr_loss_val = loss.pop("spr_loss")
            if update_spr:
                diag = spr_diagnostics(model, batch)
                spr_writer.writerow({"update": update, "stage": stage, "step": step_done, "spr_loss": spr_loss_val, **diag})
                spr_file.flush()
            std = std_info(model)
            info = {
                "update": update,
                "stage": stage,
                "step": step_done,
                "fps": step_done / max(time.perf_counter() - start_time, 1e-6),
                "sample_fps": steps_per_update / max(collect_s, 1e-6),
                "collect_s": collect_s,
                "learn_s": learn_s,
                "done": done_count,
                "success": success_count,
                "fail": fail_count,
                "timeout": timeout_count,
                "success_rate": success_count / max(done_count, 1),
                "fail_rate": fail_count / max(done_count, 1),
                "timeout_rate": timeout_count / max(done_count, 1),
                "a_mag_mean": a_mag_sum / steps_per_update,
                "stop_frac": stop_count / steps_per_update,
                "zone_frac": zone_count / steps_per_update,
                **loss,
                **std,
                "teacher_coef": teacher_coef,
            }
            last_info = info
            log_writer.writerow(info)
            log_file.flush()
            traj_file.flush()

            if int(cfg.SAVE_UPDATE_EVERY) > 0 and update % int(cfg.SAVE_UPDATE_EVERY) == 0:
                if stage == 1:
                    save_spr(update_dir / f"spr_{update:06d}.pt", model, update, step_done, info)
                elif stage in (2, 3):
                    save_policy(update_dir / f"ppo_{update:06d}.pt", model, update, step_done, teacher_coef, info)
                else:
                    save_model(update_dir / f"full_{update:06d}.pt", model, opt, update, step_done, best, teacher_coef, info)
            val_info = None
            if stage != 1 and int(cfg.VAL_EVERY) > 0 and update % int(cfg.VAL_EVERY) == 0:
                val_info = val(env, model, int(cfg.VAL_STEPS), step_done, update)
                val_row = {"update": update, "step": step_done, **val_info, **std, "teacher_coef": teacher_coef}
                val_writer.writerow(val_row)
                val_file.flush()
                obs, _ = env.reset()
                priv_t = obs["policy"]
                image_t = camera_rgb(env)
                goal_t = env.end_obs()
                if bool(cfg.SAVE_BEST) and update >= int(cfg.BEST_WARMUP):
                    rate = val_info["success_rate"]
                    if rate >= best + float(cfg.BEST_MARGIN):
                        best = rate
                        save_model(log_dir / "best.pt", model, opt, update, step_done, best, teacher_coef, val_row)
                        (log_dir / "best_info.txt").write_text(
                            "\n".join(f"{key} = {value}" for key, value in val_row.items()) + "\n",
                            encoding="utf-8",
                        )

            tb.add_scalar("time/fps", info["fps"], step_done)
            tb.add_scalar("time/sample_fps", info["sample_fps"], step_done)
            tb.add_scalar("time/collect_s", info["collect_s"], step_done)
            tb.add_scalar("time/learn_s", info["learn_s"], step_done)
            tb.add_scalar("rollout/success", info["success_rate"], step_done)
            tb.add_scalar("rollout/fail", info["fail_rate"], step_done)
            tb.add_scalar("rollout/timeout", info["timeout_rate"], step_done)
            tb.add_scalar("train/pi_loss", info["pi_loss"], step_done)
            tb.add_scalar("train/v_loss", info["v_loss"], step_done)
            if update_spr:
                tb.add_scalar("train/spr_loss", spr_loss_val, step_done)
            tb.add_scalar("train/teacher_loss", info["teacher_loss"], step_done)
            tb.add_scalar("train/kl", info["kl"], step_done)
            tb.add_scalar("policy/raw_std_mean", info["raw_std_mean"], step_done)
            tb.add_scalar("policy/raw_std_min", info["raw_std_min"], step_done)
            tb.add_scalar("policy/raw_std_max", info["raw_std_max"], step_done)
            tb.add_scalar("policy/std_mean", info["std_mean"], step_done)
            tb.add_scalar("policy/teacher_coef", teacher_coef, step_done)

            if update % int(cfg.LOG_EVERY) == 0:
                groups = [
                    (
                        "time",
                        [
                            ("update", update),
                            ("stage", stage),
                            ("step", step_done),
                            ("fps", info["fps"]),
                            ("sample_fps", info["sample_fps"]),
                        ],
                    ),
                    (
                        "rollout",
                        [
                            ("done", done_count),
                            ("success", info["success_rate"]),
                            ("fail", info["fail_rate"]),
                            ("timeout", info["timeout_rate"]),
                        ],
                    ),
                    (
                        "policy",
                        [
                            ("raw_std", info["raw_std_mean"]),
                            ("std", info["std_mean"]),
                            ("teacher_coef", teacher_coef),
                        ],
                    ),
                    (
                        "train",
                        [
                            ("v_loss", info["v_loss"]),
                            ("spr_loss", spr_loss_val),
                            ("teacher_loss", info["teacher_loss"]),
                            ("kl", info["kl"]),
                        ],
                    ),
                ]
                if val_info is not None:
                    groups.append(("val", [("success", val_info["success_rate"]), ("fail", val_info["fail_rate"]), ("timeout", val_info["timeout_rate"])]))
                print_log(groups)

        save_model(log_dir / "last.pt", model, opt, updates, total_steps, best, teacher_coef, last_info)
        print(f"[INFO] saved {log_dir / 'last.pt'}")
    finally:
        tb.close()
        log_file.close()
        spr_file.close()
        val_file.close()
        traj_file.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
