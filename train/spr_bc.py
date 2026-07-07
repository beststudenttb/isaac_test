"""Three-stage SPR + teacher BC training.

Stage 1: teacher (with noise) drives the env, only SPR/encoder trains.
Stage 2: SPR frozen, student drives, actor trains with teacher BC.
Stage 3: BC continues, SPR/encoder updates at low frequency (joint fine-tune).
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

import spr_bc_cfg as cfg


parser = argparse.ArgumentParser(description="Train SPR visual policy with teacher BC.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--show", action="store_true")
parser.add_argument("--noise", action="store_true")
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import task_cfg
from src.cv_extractor.spr_state import SPRStateNet
from src.mdp_student_env import MDPStudentEnvCfg, make_mdp_student_env
from src.noise_env import NoiseStudentEnvCfg, make_noise_student_env
from src.spr_ppo import SPRActorCritic


ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
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
    "teacher_mse",
    "spr_loss",
    "std",
]


def make_env():
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
    env_cfg.end_d_min = float(cfg.END_D_MIN)
    env_cfg.end_d_max = float(cfg.END_D_MAX)
    env_cfg.end_x_min = float(cfg.END_X_MIN)
    env_cfg.end_x_max = float(cfg.END_X_MAX)
    make_env_fn = make_noise_student_env if args_cli.noise else make_mdp_student_env
    return make_env_fn(env_cfg)


def camera_rgb(env):
    image = env.camera.data.output["rgb"]
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def image_u8(image):
    image = image.detach()
    if image.dtype == torch.uint8:
        return image.cpu()
    if float(image.max().detach().cpu()) <= 1.0:
        image = image * 255.0
    return torch.clamp(image, 0.0, 255.0).to(torch.uint8).cpu()


def teacher_action(teacher, obs):
    action, _ = teacher.predict(obs.detach().cpu().numpy(), deterministic=True)
    action = torch.as_tensor(action, device=obs.device, dtype=obs.dtype)
    x_ok = torch.abs(obs[:, 0] - obs[:, 3]) <= float(task_cfg.STOP_X_TOL) / (float(task_cfg.IMAGE_WIDTH) * 0.5)
    d_ok = torch.abs(obs[:, 1] - obs[:, 2]) <= float(task_cfg.STOP_D_TOL) / float(task_cfg.FAIL_FAR)
    action[x_ok & d_ok] = 0.0
    return action


def explore_action(target):
    action = target.clone()
    if float(cfg.TEACHER_NOISE_STD) > 0.0:
        action = action + torch.randn_like(action) * float(cfg.TEACHER_NOISE_STD)
    if float(cfg.RAND_ACTION_P) > 0.0:
        rand = torch.rand(action.shape[0], device=action.device) < float(cfg.RAND_ACTION_P)
        action[rand] = torch.empty_like(action[rand]).uniform_(-1.0, 1.0)
    return torch.clamp(action, -1.0, 1.0)


def stage_of(update):
    if update <= int(cfg.STAGE1_UPDATES):
        return 1
    if update <= int(cfg.STAGE1_UPDATES) + int(cfg.STAGE2_UPDATES):
        return 2
    return 3


def save(path, model, opt, update, step, info):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "update": int(update),
            "step": int(step),
            "spr_cfg": {
                "input_shape": (int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3),
                "z_dim": int(cfg.Z_DIM),
                "action_dim": 3,
                "fpn_dim": int(cfg.FPN_DIM),
                "hidden_dim": int(cfg.SPR_HIDDEN),
                "pool_size": int(cfg.SPR_POOL),
                "pretrained": False,
                "target_tau": float(cfg.SPR_TAU),
                "freeze_stem_layers": bool(cfg.FREEZE_STEM_LAYERS),
            },
            "info": info,
        },
        path,
    )


def train_bc(model, opt, images, next_images, goals, targets, actions, train_actor, train_encoder, use_spr):
    n = targets.shape[0]
    total_mse = 0.0
    total_spr = 0.0
    count = 0
    for _ in range(int(cfg.N_EPOCHS)):
        ids = torch.randperm(n, device=targets.device)
        for start in range(0, n, int(cfg.BATCH_SIZE)):
            idx = ids[start : start + int(cfg.BATCH_SIZE)]
            z = model.encode(images[idx.cpu()], grad=train_encoder)
            loss = torch.zeros((), device=targets.device)
            if train_actor:
                z_scale = float(cfg.ACTOR_ENCODER_COEF) if train_encoder else 0.0
                mean = model.actor(model.actor_obs(z, goals[idx], z_grad_scale=z_scale))
                mse = (mean - targets[idx]).pow(2).mean()
                loss = loss + mse
            else:
                with torch.no_grad():
                    mean = model.actor(model.actor_obs(z, goals[idx]))
                    mse = (mean - targets[idx]).pow(2).mean()
            spr_loss = torch.zeros((), device=targets.device)
            if use_spr:
                spr_loss = model.spr_loss(z, next_images[idx.cpu()], actions[idx])
                loss = loss + float(cfg.SPR_COEF) * spr_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.MAX_GRAD_NORM))
            opt.step()
            if use_spr:
                model.spr.update_target()
            total_mse += float(mse.detach().cpu())
            total_spr += float(spr_loss.detach().cpu())
            count += 1
    return total_mse / max(count, 1), total_spr / max(count, 1)


def main():
    random.seed(int(cfg.SEED))
    np.random.seed(int(cfg.SEED))
    torch.manual_seed(int(cfg.SEED))

    out = Path(cfg.OUT_DIR)
    if bool(cfg.CLEAR_OUT_DIR) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.py").write_text("\n".join(f"{k} = {getattr(cfg, k)!r}" for k in dir(cfg) if k.isupper()) + "\n", encoding="utf-8")
    log_file = (out / "log.csv").open("w", newline="", encoding="utf-8")
    log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    log.writeheader()
    update_dir = out / "updates"
    if int(cfg.SAVE_UPDATE_EVERY) > 0:
        update_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()
    device = torch.device(env.device)
    teacher = PPO.load(str(cfg.TEACHER_PATH), device=device)
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
        activation=ACT[str(cfg.ACTIVATION)],
        init_std=float(cfg.STD_INIT),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.LR), eps=1e-5)

    obs, _ = env.reset()
    priv = obs["policy"]
    image = camera_rgb(env)
    goal = env.end_obs()

    steps_per_update = env.num_envs * int(cfg.N_STEPS)
    total_steps = int(cfg.TOTAL_STEPS)
    updates = int(math.ceil(total_steps / steps_per_update))
    stage3_updates = max(updates - int(cfg.STAGE1_UPDATES) - int(cfg.STAGE2_UPDATES), 0)
    print(
        f"[INFO] spr bc out={out} envs={env.num_envs} updates={updates} device={env.device} "
        f"stages={cfg.STAGE1_UPDATES}/{cfg.STAGE2_UPDATES}/{stage3_updates}"
    )

    start_time = time.perf_counter()
    last_info = {}
    try:
        for update in range(1, updates + 1):
            stage = stage_of(update)
            images = torch.empty(
                (int(cfg.N_STEPS), env.num_envs, int(task_cfg.IMAGE_HEIGHT), int(task_cfg.IMAGE_WIDTH), 3),
                dtype=torch.uint8,
                device="cpu",
            )
            next_images = torch.empty_like(images)
            goals = torch.zeros((int(cfg.N_STEPS), env.num_envs, 2), device=device)
            targets = torch.zeros((int(cfg.N_STEPS), env.num_envs, int(env.cfg.action_space)), device=device)
            actions = torch.zeros_like(targets)
            done_n = 0
            succ_n = 0
            fail_n = 0
            timeout_n = 0

            collect_start = time.perf_counter()
            for t in range(int(cfg.N_STEPS)):
                with torch.no_grad():
                    target = teacher_action(teacher, priv)
                    if stage == 1:
                        action = explore_action(target)
                    else:
                        action, _, _ = model.act(image, goal)
                obs_next, _reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                next_image = camera_rgb(env)
                images[t].copy_(image_u8(image))
                next_images[t].copy_(image_u8(next_image))
                goals[t].copy_(goal.detach())
                targets[t].copy_(target.detach())
                actions[t].copy_(action.detach())
                if torch.any(done):
                    done_n += int(done.sum().item())
                    succ_n += int((env.last_success & done).sum().item())
                    fail_n += int((env.last_fail & done).sum().item())
                    timeout_n += int(((env.last_timeout | truncated) & done).sum().item())
                priv = obs_next["policy"]
                image = next_image
                goal = env.end_obs()
            collect_s = time.perf_counter() - collect_start

            learn_start = time.perf_counter()
            if stage == 1:
                train_actor, train_encoder, use_spr = False, True, True
            elif stage == 2:
                train_actor, train_encoder, use_spr = True, False, False
            else:
                spr_turn = int(cfg.SPR_UPDATE_EVERY) > 0 and update % int(cfg.SPR_UPDATE_EVERY) == 0
                train_actor, train_encoder, use_spr = True, spr_turn, spr_turn
            mse, spr_loss = train_bc(
                model,
                opt,
                images.reshape(-1, *images.shape[2:]),
                next_images.reshape(-1, *next_images.shape[2:]),
                goals.reshape(-1, 2),
                targets.reshape(-1, int(env.cfg.action_space)),
                actions.reshape(-1, int(env.cfg.action_space)),
                train_actor,
                train_encoder,
                use_spr,
            )
            learn_s = time.perf_counter() - learn_start
            step = min(update * steps_per_update, total_steps)
            std = float(torch.exp(model.log_std.detach()).mean().cpu())
            info = {
                "update": update,
                "stage": stage,
                "step": step,
                "fps": step / max(time.perf_counter() - start_time, 1e-6),
                "sample_fps": steps_per_update / max(collect_s, 1e-6),
                "collect_s": collect_s,
                "learn_s": learn_s,
                "done": done_n,
                "success": succ_n,
                "fail": fail_n,
                "timeout": timeout_n,
                "teacher_mse": mse,
                "spr_loss": spr_loss,
                "std": std,
            }
            last_info = info
            log.writerow(info)
            log_file.flush()

            if int(cfg.SAVE_UPDATE_EVERY) > 0 and update % int(cfg.SAVE_UPDATE_EVERY) == 0:
                save(update_dir / f"update_{update:06d}.pt", model, opt, update, step, info)
            if update == int(cfg.STAGE1_UPDATES):
                save(out / "stage1_end.pt", model, opt, update, step, info)
            if update == int(cfg.STAGE1_UPDATES) + int(cfg.STAGE2_UPDATES):
                save(out / "stage2_end.pt", model, opt, update, step, info)

            if update % int(cfg.LOG_EVERY) == 0:
                print(
                    f"upd={update} s{stage} step={step} fps={info['fps']:.1f} sample={info['sample_fps']:.1f} "
                    f"std={std:.3f} mse={mse:.3f} spr={spr_loss:.3f} "
                    f"succ={succ_n / max(done_n, 1):.3f} fail={fail_n / max(done_n, 1):.3f} timeout={timeout_n / max(done_n, 1):.3f}"
                )

        save(out / "last.pt", model, opt, updates, total_steps, last_info)
    finally:
        log_file.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
