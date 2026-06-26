"""Train student with the local PPO implementation.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p train/student.py
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

import student_cfg


parser = argparse.ArgumentParser(description="Train student with custom PPO.")
parser.add_argument("--show", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if student_cfg.DEVICE is not None:
    args_cli.device = student_cfg.DEVICE
if not args_cli.show:
    args_cli.headless = True
    args_cli.livestream = 0
args_cli.enable_cameras = bool(student_cfg.USE_CAMERA)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ppo import ActorCritic, Rollout, ppo_update
from src.sb3_env import BallPPOEnv, BallPPOEnvCfg


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
    "a_x",
    "a_y",
    "a_w",
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
    "std_mean",
    "std_min",
    "std_max",
    "teacher_coef",
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
    path = Path(student_cfg.OUT_DIR)
    if bool(student_cfg.CLEAR_OUT_DIR) and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_config(path: Path):
    lines = []
    for name in dir(student_cfg):
        if name.isupper():
            lines.append(f"{name} = {getattr(student_cfg, name)!r}")
    lines.append(f"DEVICE_USED = {args_cli.device!r}")
    (path / "config.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def open_csv(path: Path, fields: list[str]):
    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    return file, writer


def make_env() -> BallPPOEnv:
    cfg = BallPPOEnvCfg()
    cfg.seed = int(student_cfg.SEED)
    cfg.episode_length_s = float(student_cfg.EPISODE_S)
    cfg.stop_n = int(student_cfg.STOP_N)
    cfg.scene.num_envs = int(student_cfg.NUM_ENVS)
    cfg.sim.device = args_cli.device
    cfg.use_camera = bool(student_cfg.USE_CAMERA)
    cfg.read_camera = bool(student_cfg.READ_CAMERA)
    cfg.end_d_min = float(student_cfg.END_D_MIN)
    cfg.end_d_max = float(student_cfg.END_D_MAX)
    cfg.end_x_min = float(student_cfg.END_X_MIN)
    cfg.end_x_max = float(student_cfg.END_X_MAX)
    return BallPPOEnv(cfg)


def save_model(
    path: Path,
    model: ActorCritic,
    opt: torch.optim.Optimizer,
    update: int,
    step: int,
    best: float,
    teacher_coef: float,
    info: dict,
):
    std = torch.exp(model.log_std.detach()).cpu()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "update": int(update),
            "step": int(step),
            "best": float(best),
            "teacher_coef": float(teacher_coef),
            "log_std": model.log_std.detach().cpu(),
            "std": std,
            "info": info,
        },
        path,
    )


def load_teacher(device: torch.device):
    if (
        float(student_cfg.TEACHER_LOSS) <= 0.0
        and float(student_cfg.TEACHER_UP_OUT) <= 0.0
        and float(student_cfg.TEACHER_UP_TIMEOUT) <= 0.0
    ):
        return None
    path = Path(student_cfg.TEACHER_PATH)
    if not path.exists():
        raise FileNotFoundError(f"teacher model not found: {path}")
    return PPO.load(str(path), device=device)


def teacher_actions(teacher, obs: torch.Tensor) -> torch.Tensor | None:
    if teacher is None:
        return None
    actions, _ = teacher.predict(obs.detach().cpu().numpy(), deterministic=True)
    return torch.as_tensor(actions, device=obs.device, dtype=obs.dtype)


def traj_row(
    env: BallPPOEnv,
    phase: str,
    step: int,
    update: int,
    t: int,
    env_id: int,
    reward: torch.Tensor,
    done: bool,
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
        "a_x": f"{float(action[0]):.4f}",
        "a_y": f"{float(action[1]):.4f}",
        "a_w": f"{float(action[2]):.4f}",
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


def std_info(model: ActorCritic) -> dict[str, float]:
    std = torch.exp(model.log_std.detach())
    return {
        "std_mean": float(std.mean().cpu()),
        "std_min": float(std.min().cpu()),
        "std_max": float(std.max().cpu()),
    }


def fmt_value(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.3f}"
    return str(value)


def print_log(groups: list[tuple[str, list[tuple[str, object]]]]):
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


def val(
    env: BallPPOEnv,
    model: ActorCritic,
    max_steps: int,
    step: int,
    update: int,
    traj_writer: csv.DictWriter | None,
) -> dict:
    obs, _ = env.reset()
    obs_t = obs["policy"]
    steps = int(max_steps) if int(max_steps) > 0 else int(env.max_episode_length)
    done_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros(env.num_envs, device=env.device)
    success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    fail = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    timeout = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        for t in range(steps):
            action = model.predict(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action)
            active = ~done_once
            returns[active] += reward[active]
            lengths[active] += 1
            done = (terminated | truncated) & active
            if traj_writer is not None:
                for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    traj_writer.writerow(
                        traj_row(env, "val", step, update, t, int(env_id), reward[int(env_id)], bool(done[int(env_id)]))
                    )
            if torch.any(done):
                success[done] = env.last_success[done]
                fail[done] = env.last_fail[done]
                timeout[done] = env.last_timeout[done] | truncated[done]
                done_once[done] = True
                if bool(done_once.all()):
                    break
            obs_t = obs["policy"]
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


def main():
    seed = int(student_cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = out_dir()
    write_config(log_dir)
    log_file, log_writer = open_csv(log_dir / "log.csv", LOG_FIELDS)
    val_file, val_writer = open_csv(log_dir / "val.csv", VAL_FIELDS)
    train_traj_file, train_traj_writer = open_csv(log_dir / "traj_train_env0.csv", TRAJ_FIELDS)
    val_traj_file, val_traj_writer = open_csv(log_dir / "traj_val.csv", TRAJ_FIELDS)
    update_dir = log_dir / "updates"
    if int(student_cfg.SAVE_UPDATE_EVERY) > 0:
        update_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir / "tb"))
    env = make_env()
    device = torch.device(env.device)
    activation = ACTIVATIONS[str(student_cfg.ACTIVATION)]
    model = ActorCritic(
        obs_dim=int(env.cfg.observation_space),
        act_dim=int(env.cfg.action_space),
        pi_hidden=list(student_cfg.POLICY_NET),
        vf_hidden=list(student_cfg.VALUE_NET),
        activation=activation,
        init_std=float(student_cfg.STD_INIT),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(student_cfg.LR), eps=1e-5)
    teacher = load_teacher(device)
    teacher_coef = float(student_cfg.TEACHER_LOSS)
    teacher_min = float(student_cfg.TEACHER_LOSS_MIN)
    teacher_down_success = float(student_cfg.TEACHER_DOWN_SUCCESS)
    teacher_up_out = float(student_cfg.TEACHER_UP_OUT)
    teacher_up_timeout = float(student_cfg.TEACHER_UP_TIMEOUT)

    obs, _ = env.reset()
    obs_t = obs["policy"]
    best = -1.0
    last_info = {}
    start = time.perf_counter()
    steps_per_update = env.num_envs * int(student_cfg.N_STEPS)
    total_steps = int(student_cfg.TOTAL_STEPS)
    updates = int(math.ceil(total_steps / steps_per_update))
    print(
        f"[INFO] student PPO out={log_dir} envs={env.num_envs} total_steps={total_steps} "
        f"updates={updates} n_steps={student_cfg.N_STEPS} device={env.device} camera={int(student_cfg.USE_CAMERA)} "
        f"teacher_loss={teacher_coef:.3f}"
    )

    try:
        for update in range(1, updates + 1):
            rollout = Rollout(
                steps=int(student_cfg.N_STEPS),
                envs=env.num_envs,
                obs_dim=int(env.cfg.observation_space),
                act_dim=int(env.cfg.action_space),
                device=device,
            )
            collect_start = time.perf_counter()
            done_count = 0
            success_count = 0
            fail_count = 0
            timeout_count = 0

            model.train()
            for t in range(int(student_cfg.N_STEPS)):
                with torch.no_grad():
                    action, logp, value = model.act(obs_t)
                obs_next, reward, terminated, truncated, _ = env.step(action)
                timeout = truncated & (~terminated)
                if torch.any(timeout):
                    with torch.no_grad():
                        terminal_value = model.critic(env.last_policy_obs[timeout]).squeeze(-1)
                    reward = reward.clone()
                    reward[timeout] += float(student_cfg.GAMMA) * terminal_value
                done = terminated | truncated
                rollout.add(t, obs_t, action, logp, reward, done, value)
                step_now = min((update - 1) * steps_per_update + (t + 1) * env.num_envs, total_steps)
                train_traj_writer.writerow(
                    traj_row(env, "train", step_now, update, t, 0, reward[0], bool(done[0]))
                )
                if torch.any(done):
                    done_count += int(done.sum().item())
                    success_count += int((env.last_success & done).sum().item())
                    fail_count += int((env.last_fail & done).sum().item())
                    timeout_count += int(((env.last_timeout | truncated) & done).sum().item())
                obs_t = obs_next["policy"]

            with torch.no_grad():
                last_value = model.critic(obs_t).squeeze(-1)
            batch = rollout.make_batch(
                last_value=last_value,
                gamma=float(student_cfg.GAMMA),
                lam=float(student_cfg.GAE_LAMBDA),
                norm_adv=bool(student_cfg.NORMALIZE_ADVANTAGE),
            )
            batch.teacher_actions = teacher_actions(teacher, batch.obs) if teacher_coef > 0.0 else None
            collect_time = time.perf_counter() - collect_start
            learn_start = time.perf_counter()
            loss = ppo_update(
                model=model,
                opt=opt,
                batch=batch,
                batch_size=int(student_cfg.BATCH_SIZE),
                epochs=int(student_cfg.N_EPOCHS),
                clip=float(student_cfg.CLIP_RANGE),
                vf_coef=float(student_cfg.VF_COEF),
                ent_coef=float(student_cfg.ENT_COEF),
                max_grad_norm=float(student_cfg.MAX_GRAD_NORM),
                clip_vf=student_cfg.CLIP_RANGE_VF,
                teacher_coef=teacher_coef,
            )
            learn_time = time.perf_counter() - learn_start
            step_done = min(update * steps_per_update, total_steps)
            fps = int(student_cfg.N_STEPS) / max(collect_time, 1e-6)
            sample_fps = steps_per_update / max(collect_time, 1e-6)
            rates = {
                "success_rate": success_count / max(done_count, 1),
                "fail_rate": fail_count / max(done_count, 1),
                "timeout_rate": timeout_count / max(done_count, 1),
            }
            std = std_info(model)
            train_info = {
                "update": update,
                "step": step_done,
                "fps": fps,
                "sample_fps": sample_fps,
                "collect_s": collect_time,
                "learn_s": learn_time,
                "done": done_count,
                "success": success_count,
                "fail": fail_count,
                "timeout": timeout_count,
                **rates,
                **loss,
                **std,
                "teacher_coef": teacher_coef,
            }
            last_info = train_info

            val_info = None
            if int(student_cfg.VAL_EVERY) > 0 and update % int(student_cfg.VAL_EVERY) == 0:
                write_val_traj = int(student_cfg.VAL_TRAJ_EVERY) > 0 and update % int(student_cfg.VAL_TRAJ_EVERY) == 0
                val_info = val(
                    env,
                    model,
                    int(student_cfg.VAL_STEPS),
                    step_done,
                    update,
                    val_traj_writer if write_val_traj else None,
                )
                teacher_coef += (
                    -teacher_down_success * val_info["success_rate"]
                    + teacher_up_out * val_info["fail_rate"]
                    + teacher_up_timeout * val_info["timeout_rate"]
                )
                teacher_coef = max(teacher_coef, teacher_min)
                val_row = {
                    "update": update,
                    "step": step_done,
                    **val_info,
                    **std,
                    "teacher_coef": teacher_coef,
                }
                val_writer.writerow(val_row)
                val_file.flush()
                if write_val_traj:
                    val_traj_file.flush()
                for key in ("success_rate", "fail_rate", "timeout_rate", "mean_return", "mean_len"):
                    tb.add_scalar(f"val/{key}", val_info[key], step_done)
                obs, _ = env.reset()
                obs_t = obs["policy"]
                if bool(student_cfg.SAVE_BEST) and update >= int(student_cfg.BEST_WARMUP):
                    rate = val_info["success_rate"]
                    if rate >= best + float(student_cfg.BEST_MARGIN):
                        best = rate
                        save_model(log_dir / "best.pt", model, opt, update, step_done, best, teacher_coef, val_info)
                        (log_dir / "best_info.txt").write_text(
                            "\n".join(f"{key} = {value}" for key, value in val_row.items()) + "\n",
                            encoding="utf-8",
                        )

            if int(student_cfg.SAVE_EVERY) > 0 and update % int(student_cfg.SAVE_EVERY) == 0:
                save_model(log_dir / f"model_{update}.pt", model, opt, update, step_done, best, teacher_coef, val_info or train_info)

            if int(student_cfg.SAVE_UPDATE_EVERY) > 0 and update % int(student_cfg.SAVE_UPDATE_EVERY) == 0:
                save_model(
                    update_dir / f"update_{update:06d}.pt",
                    model,
                    opt,
                    update,
                    step_done,
                    best,
                    teacher_coef,
                    val_info or train_info,
                )

            log_writer.writerow(train_info)
            log_file.flush()
            train_traj_file.flush()
            tb.add_scalar("time/fps", fps, step_done)
            tb.add_scalar("time/sample_fps", sample_fps, step_done)
            tb.add_scalar("time/collect_s", collect_time, step_done)
            tb.add_scalar("time/learn_s", learn_time, step_done)
            tb.add_scalar("rollout/success", rates["success_rate"], step_done)
            tb.add_scalar("rollout/fail", rates["fail_rate"], step_done)
            tb.add_scalar("rollout/timeout", rates["timeout_rate"], step_done)
            tb.add_scalar("train/pi_loss", loss["pi_loss"], step_done)
            tb.add_scalar("train/v_loss", loss["v_loss"], step_done)
            tb.add_scalar("train/teacher_loss", loss["teacher_loss"], step_done)
            tb.add_scalar("train/entropy", loss["entropy"], step_done)
            tb.add_scalar("train/kl", loss["kl"], step_done)
            tb.add_scalar("policy/std_mean", std["std_mean"], step_done)
            tb.add_scalar("policy/std_min", std["std_min"], step_done)
            tb.add_scalar("policy/std_max", std["std_max"], step_done)
            tb.add_scalar("policy/teacher_coef", teacher_coef, step_done)

            if update % int(student_cfg.LOG_EVERY) == 0:
                groups = [
                    (
                        "time",
                        [
                            ("update", update),
                            ("step", step_done),
                            ("fps", fps),
                        ],
                    ),
                    (
                        "rollout",
                        [
                            ("done", done_count),
                            ("success", rates["success_rate"]),
                            ("fail", rates["fail_rate"]),
                            ("timeout", rates["timeout_rate"]),
                        ],
                    ),
                    (
                        "policy",
                        [
                            ("std", std["std_mean"]),
                            ("teacher_coef", teacher_coef),
                        ],
                    ),
                    (
                        "train",
                        [
                            ("teacher_loss", loss["teacher_loss"]),
                            ("kl", loss["kl"]),
                        ],
                    ),
                ]
                if val_info is not None:
                    groups.append(
                        (
                            "val",
                            [
                                ("success", val_info["success_rate"]),
                                ("fail", val_info["fail_rate"]),
                                ("timeout", val_info["timeout_rate"]),
                            ],
                        )
                    )
                print_log(groups)

        save_model(log_dir / "last.pt", model, opt, updates, total_steps, best, teacher_coef, last_info)
        print(f"[INFO] saved {log_dir / 'last.pt'} time={time.perf_counter() - start:.1f}s")
    finally:
        tb.close()
        log_file.close()
        val_file.close()
        train_traj_file.close()
        val_traj_file.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
