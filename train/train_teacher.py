"""Train SB3 PPO teacher with privileged px_x/dist observation.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p train/train_teacher.py
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import teacher_cfg


parser = argparse.ArgumentParser(description="Train privileged teacher PPO.")
parser.add_argument("--show", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if teacher_cfg.DEVICE is not None:
    args_cli.device = teacher_cfg.DEVICE
if not args_cli.show:
    args_cli.headless = True
    args_cli.livestream = 0
args_cli.enable_cameras = bool(teacher_cfg.USE_CAMERA)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sb3_env import BallPPOEnvCfg, make_sb3_env
from src.val_callback import TrainTrajCallback, ValCallback


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def make_out_dir() -> Path:
    out_dir = teacher_cfg.OUT_DIR
    if bool(teacher_cfg.CLEAR_OUT_DIR) and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_config(out_dir: Path):
    lines = [
        f"out_dir = {str(teacher_cfg.OUT_DIR)!r}",
        f"clear_out_dir = {bool(teacher_cfg.CLEAR_OUT_DIR)}",
        f"num_envs = {int(teacher_cfg.NUM_ENVS)}",
        f"total_steps = {int(teacher_cfg.TOTAL_STEPS)}",
        f"episode_s = {float(teacher_cfg.EPISODE_S)}",
        f"seed = {int(teacher_cfg.SEED)}",
        f"use_camera = {bool(teacher_cfg.USE_CAMERA)}",
        f"read_camera = {bool(teacher_cfg.READ_CAMERA)}",
        f"n_steps = {int(teacher_cfg.N_STEPS)}",
        f"batch_size = {int(teacher_cfg.BATCH_SIZE)}",
        f"n_epochs = {int(teacher_cfg.N_EPOCHS)}",
        f"lr = {float(teacher_cfg.LR)}",
        f"gamma = {float(teacher_cfg.GAMMA)}",
        f"gae_lambda = {float(teacher_cfg.GAE_LAMBDA)}",
        f"clip_range = {float(teacher_cfg.CLIP_RANGE)}",
        f"clip_range_vf = {teacher_cfg.CLIP_RANGE_VF!r}",
        f"normalize_advantage = {bool(teacher_cfg.NORMALIZE_ADVANTAGE)}",
        f"ent_coef = {float(teacher_cfg.ENT_COEF)}",
        f"vf_coef = {float(teacher_cfg.VF_COEF)}",
        f"max_grad_norm = {float(teacher_cfg.MAX_GRAD_NORM)}",
        f"use_sde = {bool(teacher_cfg.USE_SDE)}",
        f"sde_sample_freq = {int(teacher_cfg.SDE_SAMPLE_FREQ)}",
        f"target_kl = {teacher_cfg.TARGET_KL!r}",
        f"stats_window_size = {int(teacher_cfg.STATS_WINDOW_SIZE)}",
        f"policy = {teacher_cfg.POLICY!r}",
        f"policy_net = {list(teacher_cfg.POLICY_NET)!r}",
        f"value_net = {list(teacher_cfg.VALUE_NET)!r}",
        f"activation = {teacher_cfg.ACTIVATION!r}",
        f"ortho_init = {bool(teacher_cfg.ORTHO_INIT)}",
        f"std_init = {float(teacher_cfg.STD_INIT)}",
        f"log_std_init = {math.log(float(teacher_cfg.STD_INIT))}",
        f"full_std = {bool(teacher_cfg.FULL_STD)}",
        f"use_expln = {bool(teacher_cfg.USE_EXPLN)}",
        f"squash_output = {bool(teacher_cfg.SQUASH_OUTPUT)}",
        f"share_features_extractor = {bool(teacher_cfg.SHARE_FEATURES_EXTRACTOR)}",
        f"normalize_images = {bool(teacher_cfg.NORMALIZE_IMAGES)}",
        f"optimizer_eps = {float(teacher_cfg.OPTIMIZER_EPS)}",
        f"rollout_buffer_class = {teacher_cfg.ROLLOUT_BUFFER_CLASS!r}",
        f"rollout_buffer_kwargs = {teacher_cfg.ROLLOUT_BUFFER_KWARGS!r}",
        f"save_every = {int(teacher_cfg.SAVE_EVERY)}",
        f"save_train_traj = {bool(teacher_cfg.SAVE_TRAIN_TRAJ)}",
        f"save_val_traj = {bool(teacher_cfg.SAVE_VAL_TRAJ)}",
        f"val_every_rollouts = {int(teacher_cfg.VAL_EVERY_ROLLOUTS)}",
        f"val_max_steps = {int(teacher_cfg.VAL_MAX_STEPS)}",
        f"save_best = {bool(teacher_cfg.SAVE_BEST)}",
        f"best_warmup_steps = {int(teacher_cfg.BEST_WARMUP_STEPS)}",
        f"best_margin = {float(teacher_cfg.BEST_MARGIN)}",
        f"log_interval = {int(teacher_cfg.LOG_INTERVAL)}",
        f"verbose = {int(teacher_cfg.VERBOSE)}",
        f"progress = {bool(teacher_cfg.PROGRESS)}",
        f"device = {args_cli.device}",
    ]
    (out_dir / "config.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_cfg() -> BallPPOEnvCfg:
    cfg = BallPPOEnvCfg()
    cfg.seed = int(teacher_cfg.SEED)
    cfg.episode_length_s = float(teacher_cfg.EPISODE_S)
    cfg.scene.num_envs = int(teacher_cfg.NUM_ENVS)
    cfg.sim.device = args_cli.device
    cfg.use_camera = bool(teacher_cfg.USE_CAMERA)
    cfg.read_camera = bool(teacher_cfg.READ_CAMERA)
    return cfg


def policy_kwargs() -> dict:
    name = str(teacher_cfg.ACTIVATION)
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown ACTIVATION: {name}")
    std_init = float(teacher_cfg.STD_INIT)
    if std_init <= 0.0:
        raise ValueError(f"STD_INIT must be > 0, got {std_init}")
    activation = ACTIVATIONS[name]
    return {
        "net_arch": {
            "pi": list(teacher_cfg.POLICY_NET),
            "vf": list(teacher_cfg.VALUE_NET),
        },
        "activation_fn": activation,
        "ortho_init": bool(teacher_cfg.ORTHO_INIT),
        "log_std_init": math.log(std_init),
        "full_std": bool(teacher_cfg.FULL_STD),
        "use_expln": bool(teacher_cfg.USE_EXPLN),
        "squash_output": bool(teacher_cfg.SQUASH_OUTPUT),
        "share_features_extractor": bool(teacher_cfg.SHARE_FEATURES_EXTRACTOR),
        "normalize_images": bool(teacher_cfg.NORMALIZE_IMAGES),
        "optimizer_kwargs": {"eps": float(teacher_cfg.OPTIMIZER_EPS)},
    }


def main():
    seed = int(teacher_cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = make_out_dir()
    write_config(out_dir)

    env = make_sb3_env(make_cfg(), fast_variant=True)
    callbacks = []
    if bool(teacher_cfg.SAVE_TRAIN_TRAJ):
        callbacks.append(TrainTrajCallback(out_dir))
    if int(teacher_cfg.SAVE_EVERY) > 0:
        save_freq = max(int(teacher_cfg.SAVE_EVERY) // int(teacher_cfg.NUM_ENVS), 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(out_dir),
                name_prefix="teacher",
                verbose=1,
            )
        )
    if int(teacher_cfg.VAL_EVERY_ROLLOUTS) > 0:
        callbacks.append(
            ValCallback(
                out_dir=out_dir,
                every_rollouts=int(teacher_cfg.VAL_EVERY_ROLLOUTS),
                max_steps=int(teacher_cfg.VAL_MAX_STEPS),
                warmup_steps=int(teacher_cfg.BEST_WARMUP_STEPS),
                margin=float(teacher_cfg.BEST_MARGIN),
                save_best=bool(teacher_cfg.SAVE_BEST),
                save_traj=bool(teacher_cfg.SAVE_VAL_TRAJ),
            )
        )

    model = PPO(
        teacher_cfg.POLICY,
        env,
        seed=seed,
        verbose=int(teacher_cfg.VERBOSE),
        tensorboard_log=str(out_dir / "tb"),
        device=args_cli.device,
        n_steps=int(teacher_cfg.N_STEPS),
        batch_size=int(teacher_cfg.BATCH_SIZE),
        n_epochs=int(teacher_cfg.N_EPOCHS),
        learning_rate=float(teacher_cfg.LR),
        gamma=float(teacher_cfg.GAMMA),
        gae_lambda=float(teacher_cfg.GAE_LAMBDA),
        clip_range=float(teacher_cfg.CLIP_RANGE),
        clip_range_vf=teacher_cfg.CLIP_RANGE_VF,
        normalize_advantage=bool(teacher_cfg.NORMALIZE_ADVANTAGE),
        ent_coef=float(teacher_cfg.ENT_COEF),
        vf_coef=float(teacher_cfg.VF_COEF),
        max_grad_norm=float(teacher_cfg.MAX_GRAD_NORM),
        use_sde=bool(teacher_cfg.USE_SDE),
        sde_sample_freq=int(teacher_cfg.SDE_SAMPLE_FREQ),
        rollout_buffer_class=teacher_cfg.ROLLOUT_BUFFER_CLASS,
        rollout_buffer_kwargs=teacher_cfg.ROLLOUT_BUFFER_KWARGS,
        target_kl=teacher_cfg.TARGET_KL,
        stats_window_size=int(teacher_cfg.STATS_WINDOW_SIZE),
        policy_kwargs=policy_kwargs(),
    )

    print(
        f"[INFO] train teacher out={out_dir} envs={teacher_cfg.NUM_ENVS} "
        f"total_steps={teacher_cfg.TOTAL_STEPS} episode_s={teacher_cfg.EPISODE_S} device={args_cli.device}"
    )
    start = time.perf_counter()
    try:
        model.learn(
            total_timesteps=int(teacher_cfg.TOTAL_STEPS),
            callback=callbacks,
            log_interval=int(teacher_cfg.LOG_INTERVAL),
            progress_bar=bool(teacher_cfg.PROGRESS),
        )
        model.save(out_dir / "last")
        print(f"[INFO] saved {out_dir / 'last.zip'}")
    finally:
        env.close()
        print(f"[INFO] train time {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
