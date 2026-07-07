"""Train PPO student with a small CNN visual feature.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p train/cnn_student.py
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

import cnn_student_cfg as cfg


parser = argparse.ArgumentParser(description="Train PPO student with a small CNN visual feature.")
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
from src.cnn_ppo import CNNActorCritic, CNNRollout, cnn_ppo_update
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.noise_env import NoiseStudentEnvCfg, make_noise_student_env


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
    if float(cfg.TEACHER_LOSS) <= 0.0:
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


def stage_of(step: int) -> int:
    s1 = int(cfg.STAGE1_STEPS)
    s2 = s1 + int(cfg.STAGE2_STEPS)
    if step < s1:
        return 1
    if step < s2:
        return 2
    return 3


def stage_teacher_coef(step: int, stage: int) -> float:
    if stage == 1:
        return float(cfg.TEACHER_LOSS)
    if stage == 2:
        frac = (step - int(cfg.STAGE1_STEPS)) / max(int(cfg.STAGE2_STEPS), 1)
        return float(cfg.TEACHER_LOSS) * max(1.0 - frac, 0.0)
    return 0.0


def std_info(model: CNNActorCritic) -> dict[str, float]:
    std = torch.exp(model.log_std.detach())
    return {
        "std_mean": float(std.mean().cpu()),
        "std_min": float(std.min().cpu()),
        "std_max": float(std.max().cpu()),
    }


def model_cfg_dict() -> dict:
    return {
        "z_dim": int(cfg.Z_DIM),
        "goal_dim": 2,
        "act_dim": 3,
        "pi_hidden": list(cfg.POLICY_NET),
        "vf_hidden": list(cfg.VALUE_NET),
        "activation": str(cfg.ACTIVATION),
        "init_std": float(cfg.STD_INIT),
        "std_max": float(cfg.STD_MAX),
        "freeze_cnn": bool(cfg.FREEZE_CNN),
    }


def save_model(path: Path, model: CNNActorCritic, opt: torch.optim.Optimizer, update: int, step: int, best: float, teacher_coef: float, info: dict):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "update": int(update),
            "step": int(step),
            "best": float(best),
            "teacher_coef": float(teacher_coef),
            "model_cfg": model_cfg_dict(),
            "info": info,
        },
        path,
    )


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


def val(env: MDPStudentEnv, model: CNNActorCritic, max_steps: int) -> dict[str, float]:
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
    width_r = max(len(str(right)) for _, right in rows)
    line = "-" * (width_l + width_r + 7)
    print(line)
    for left, right in rows:
        print(f"| {left:<{width_l}} | {fmt_value(right):>{width_r}} |")
    print(line)


def main() -> None:
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = out_dir()
    write_config(log_dir)
    log_file, log_writer = open_csv(log_dir / "log.csv", LOG_FIELDS)
    val_file, val_writer = open_csv(log_dir / "val.csv", VAL_FIELDS)
    traj_file, traj_writer = open_csv(log_dir / "traj_train_env0.csv", TRAJ_FIELDS)
    update_dir = log_dir / "updates"
    if int(cfg.SAVE_UPDATE_EVERY) > 0:
        update_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir / "tb"))

    env = make_env()
    device = torch.device(env.device)
    model = CNNActorCritic(
        z_dim=int(cfg.Z_DIM),
        goal_dim=2,
        act_dim=int(env.cfg.action_space),
        pi_hidden=list(cfg.POLICY_NET),
        vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)],
        init_std=float(cfg.STD_INIT),
    ).to(device)
    if bool(cfg.FREEZE_CNN):
        model.encoder.requires_grad_(False)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=float(cfg.LR), eps=1e-5)
    teacher = load_teacher(device)
    if (int(cfg.STAGE1_STEPS) > 0 or int(cfg.STAGE2_STEPS) > 0) and teacher is None:
        raise ValueError("teacher stages require a teacher policy")

    obs, _ = env.reset()
    priv_t = obs["policy"]
    image_t = camera_rgb(env)
    goal_t = env.end_obs()

    steps_per_update = env.num_envs * int(cfg.N_STEPS)
    total_steps = int(cfg.TOTAL_STEPS)
    updates = int(math.ceil(total_steps / steps_per_update))
    stage3_steps = max(total_steps - int(cfg.STAGE1_STEPS) - int(cfg.STAGE2_STEPS), 0)
    print(
        f"[INFO] cnn student PPO out={log_dir} envs={env.num_envs} total_steps={total_steps} "
        f"updates={updates} stage_steps={cfg.STAGE1_STEPS}/{cfg.STAGE2_STEPS}/{stage3_steps} "
        f"n_steps={cfg.N_STEPS} z_dim={cfg.Z_DIM} std_init={cfg.STD_INIT} std_max={cfg.STD_MAX} "
        f"freeze_cnn={cfg.FREEZE_CNN} device={env.device}"
    )

    best = -1.0
    last_info = {}
    teacher_coef = 0.0
    start_time = time.perf_counter()
    try:
        for update in range(1, updates + 1):
            step_start = (update - 1) * steps_per_update
            stage = stage_of(step_start)
            teacher_coef = stage_teacher_coef(step_start, stage)
            rollout = CNNRollout(
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
                    action, logp, value = model.act(image_t, goal_t)
                    if teacher is not None:
                        teacher_a0 = teacher_actions(teacher, priv_t[0:1])[0]
                    else:
                        teacher_a0 = torch.zeros(int(env.cfg.action_space), device=device)
                obs_next, reward, terminated, truncated, _ = env.step(action)
                timeout = truncated & (~terminated)
                done = terminated | truncated
                if torch.any(timeout):
                    with torch.no_grad():
                        terminal_value = model.value(env.terminal_rgb[timeout], env.terminal_end_obs[timeout])
                    reward = reward.clone()
                    reward[timeout] += float(cfg.GAMMA) * terminal_value
                rollout.add(t, image_t, goal_t, action, logp, reward, done, value)
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
                lam=float(cfg.GAE_LAMBDA),
                norm_adv=bool(cfg.NORMALIZE_ADVANTAGE),
            )
            if teacher is not None and teacher_coef > 0.0:
                batch.teacher_actions = teacher_actions(teacher, priv_buf.reshape(-1, priv_buf.shape[-1]))
            collect_s = time.perf_counter() - collect_start

            learn_start = time.perf_counter()
            loss = cnn_ppo_update(
                model=model,
                opt=opt,
                batch=batch,
                batch_size=int(cfg.BATCH_SIZE),
                epochs=int(cfg.N_EPOCHS),
                clip=float(cfg.CLIP_RANGE),
                vf_coef=float(cfg.VF_COEF),
                ent_coef=float(cfg.ENT_COEF),
                teacher_coef=teacher_coef,
                max_grad_norm=float(cfg.MAX_GRAD_NORM),
                clip_vf=cfg.CLIP_RANGE_VF,
                log_std_max=math.log(float(cfg.STD_MAX)),
            )
            learn_s = time.perf_counter() - learn_start

            step_done = min(update * steps_per_update, total_steps)
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
                save_model(update_dir / f"ppo_{update:06d}.pt", model, opt, update, step_done, best, teacher_coef, info)

            val_info = None
            if int(cfg.VAL_EVERY) > 0 and update % int(cfg.VAL_EVERY) == 0:
                val_info = val(env, model, int(cfg.VAL_STEPS))
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
            tb.add_scalar("train/teacher_loss", info["teacher_loss"], step_done)
            tb.add_scalar("train/kl", info["kl"], step_done)
            tb.add_scalar("policy/std", info["std_mean"], step_done)
            tb.add_scalar("policy/teacher_coef", teacher_coef, step_done)

            if update % int(cfg.LOG_EVERY) == 0:
                groups = [
                    ("time", [("update", update), ("stage", stage), ("step", step_done), ("fps", info["fps"]), ("sample_fps", info["sample_fps"])]),
                    ("rollout", [("done", done_count), ("success", info["success_rate"]), ("fail", info["fail_rate"]), ("timeout", info["timeout_rate"])]),
                    ("policy", [("std", info["std_mean"]), ("teacher_coef", teacher_coef)]),
                    ("train", [("v_loss", info["v_loss"]), ("teacher_loss", info["teacher_loss"]), ("kl", info["kl"])]),
                ]
                if val_info is not None:
                    groups.append(("val", [("success", val_info["success_rate"]), ("fail", val_info["fail_rate"]), ("timeout", val_info["timeout_rate"])]))
                print_log(groups)

        save_model(log_dir / "last.pt", model, opt, updates, total_steps, best, teacher_coef, last_info)
        print(f"[INFO] saved {log_dir / 'last.pt'}")
    finally:
        tb.close()
        log_file.close()
        val_file.close()
        traj_file.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
