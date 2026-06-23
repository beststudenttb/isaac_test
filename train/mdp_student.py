"""Train PPO student on frozen MDP latent state.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p train/mdp_student.py
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

import mdp_student_cfg as cfg


parser = argparse.ArgumentParser(description="Train PPO student on MDP latent state.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--mdp", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if cfg.DEVICE is not None:
    args_cli.device = cfg.DEVICE
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

from src.mdp_state import MDPStateNet
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.ppo import ActorCritic, Rollout, ppo_update
from src import task_cfg


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
    "mdp_update",
    "mdp_loss",
    "mdp_dyn",
    "mdp_contrast",
    "mdp_idm",
    "mdp_reward",
    "mdp_done",
    "mdp_value",
    "mdp_retrieval1",
    "mdp_eff_rank",
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
    path = Path(cfg.OUT_DIR)
    if bool(cfg.CLEAR_OUT_DIR) and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_config(path: Path, mdp_path: Path, obs_dim: int):
    lines = []
    for name in dir(cfg):
        if name.isupper():
            lines.append(f"{name} = {getattr(cfg, name)!r}")
    lines.append(f"MDP_PATH_USED = {mdp_path!r}")
    lines.append(f"OBS_DIM = {int(obs_dim)!r}")
    lines.append(f"DEVICE_USED = {args_cli.device!r}")
    (path / "config.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def open_csv(path: Path, fields: list[str]):
    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    return file, writer


def make_env():
    env_cfg = MDPStudentEnvCfg()
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
    return make_mdp_student_env(env_cfg)


def load_mdp(device: torch.device) -> MDPStateNet:
    path = args_cli.mdp or Path(cfg.MDP_PATH)
    if not path.exists():
        raise FileNotFoundError(f"MDP checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model = MDPStateNet(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if bool(cfg.FREEZE_MDP_BACKBONE):
        for module in (model.stem, model.layer1, model.layer2, model.layer3):
            for param in module.parameters():
                param.requires_grad_(False)
    return model


def camera_rgb(env) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def mdp_observe(
    mdp: MDPStateNet,
    image: torch.Tensor,
    prev_action: torch.Tensor,
    prev_state: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        out = mdp.observe(image, prev_action, prev_state)
    return {
        "belief": out["belief"].detach(),
        "z": out["z"].detach(),
        "state_feature": out["state_feature"].float().detach(),
    }


def load_teacher(device: torch.device):
    if (
        float(cfg.TEACHER_LOSS) <= 0.0
        and float(cfg.TEACHER_UP_OUT) <= 0.0
        and float(cfg.TEACHER_UP_TIMEOUT) <= 0.0
    ):
        return None
    path = Path(cfg.TEACHER_PATH)
    if not path.exists():
        raise FileNotFoundError(f"teacher model not found: {path}")
    return PPO.load(str(path), device=device)


def teacher_actions(teacher, priv_obs: torch.Tensor) -> torch.Tensor | None:
    if teacher is None:
        return None
    actions, _ = teacher.predict(priv_obs.detach().cpu().numpy(), deterministic=True)
    actions = torch.as_tensor(actions, device=priv_obs.device, dtype=priv_obs.dtype)
    x_ok = torch.abs(priv_obs[:, 0]) <= float(task_cfg.STOP_X_TOL) / (float(task_cfg.IMAGE_WIDTH) * 0.5)
    d_ok = torch.abs(priv_obs[:, 1] - float(task_cfg.STOP_D) / float(task_cfg.FAIL_FAR)) <= (
        float(task_cfg.STOP_D_TOL) / float(task_cfg.FAIL_FAR)
    )
    actions[x_ok & d_ok] = 0.0
    return actions


def next_mdp_state(
    mdp: MDPStateNet,
    env,
    prev_state: dict[str, torch.Tensor],
    action: torch.Tensor,
    done: torch.Tensor,
) -> dict[str, torch.Tensor]:
    image = camera_rgb(env)
    state = {
        "belief": prev_state["belief"].clone(),
        "z": prev_state["z"].clone(),
    }
    prev_action = action.clone()
    if torch.any(done):
        state["belief"][done] = 0.0
        state["z"][done] = 0.0
        prev_action[done] = 0.0
    return mdp_observe(mdp, image, prev_action, state)


def terminal_mdp_obs(
    mdp: MDPStateNet,
    env,
    prev_state: dict[str, torch.Tensor],
    action: torch.Tensor,
    timeout: torch.Tensor,
) -> torch.Tensor:
    state = {
        "belief": prev_state["belief"][timeout],
        "z": prev_state["z"][timeout],
    }
    out = mdp_observe(mdp, env.last_rgb[timeout], action[timeout], state)
    return out["state_feature"]


def mdp_contrast_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = pred @ target.t() / float(cfg.MDP_CONTRAST_TAU)
    labels = torch.arange(pred.shape[0], device=pred.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    retrieval1 = (logits.argmax(dim=1) == labels).float().mean()
    return loss, retrieval1


def mdp_effective_rank(z: torch.Tensor) -> torch.Tensor:
    z = z.float() - z.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(z)
    prob = singular / singular.sum().clamp_min(1e-8)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum()
    return entropy.exp()


def mdp_returns(reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(reward)
    g = torch.zeros_like(reward[:, 0])
    for t in reversed(range(reward.shape[1])):
        g = reward[:, t] + float(cfg.GAMMA) * g * (1.0 - done[:, t])
        out[:, t] = g
    return out


def train_mdp_online(
    mdp: MDPStateNet,
    opt: torch.optim.Optimizer,
    images: torch.Tensor,
    next_images: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> dict[str, float]:
    device = next(mdp.parameters()).device
    env_count = images.shape[1]
    batch_envs = min(int(cfg.MDP_BATCH_ENVS), int(env_count))
    info = {}
    mdp.eval()
    for _ in range(int(cfg.MDP_UPDATES)):
        ids = torch.randperm(env_count)[:batch_envs]
        seq_len = min(int(cfg.MDP_SEQ_LEN), int(images.shape[0]))
        seq_start = int(torch.randint(0, int(images.shape[0]) - seq_len + 1, (1,)).item())
        seq_end = seq_start + seq_len
        image = images[seq_start:seq_end, ids].to(device)
        next_image = next_images[seq_start:seq_end, ids].to(device)
        action = actions[seq_start:seq_end, ids].to(device)
        reward = rewards[seq_start:seq_end, ids].to(device).unsqueeze(-1)
        done = dones[seq_start:seq_end, ids].to(device).unsqueeze(-1)
        ret = mdp_returns(reward.permute(1, 0, 2), done.permute(1, 0, 2))

        batch_size = len(ids)
        state = mdp.initial(batch_size, device)
        zero_action = torch.zeros(batch_size, int(mdp.action_dim), device=device)
        obs = mdp.observe(image[0], zero_action, state)
        dyn_losses = []
        idm_losses = []
        reward_losses = []
        done_losses = []
        value_losses = []
        z_list = []
        pred_list = []
        target_list = []

        for t in range(action.shape[0]):
            pred = mdp.imagine(obs, action[t])
            next_post = mdp.observe(next_image[t], action[t], obs)
            dyn_losses.append(torch.nn.functional.smooth_l1_loss(pred["next_z_pred_unit"], next_post["z_unit"].detach()))
            idm_losses.append(torch.nn.functional.smooth_l1_loss(mdp.inverse(obs["z_img"], next_post["z_img"]), action[t]))
            reward_losses.append(torch.nn.functional.smooth_l1_loss(pred["reward_pred"], reward[t]))
            done_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(pred["done_logit"], done[t]))
            value_losses.append(torch.nn.functional.smooth_l1_loss(obs["value"], ret[:, t]))
            z_list.append(obs["z"])
            pred_list.append(pred["next_z_pred_unit"])
            target_list.append(next_post["z_unit"].detach())

            keep = 1.0 - done[t]
            obs = {
                "belief": next_post["belief"] * keep,
                "z": next_post["z"] * keep,
                "z_img": next_post["z_img"] * keep,
                "state_feature": next_post["state_feature"] * keep,
                "value": next_post["value"] * keep,
                "probe": next_post["probe"] * keep,
            }
            if t + 1 < action.shape[0] and bool(done[t].any()):
                reset_obs = mdp.observe(image[t + 1], zero_action, mdp.initial(batch_size, device))
                mask = done[t]
                for key in ("belief", "z", "z_img", "state_feature", "value", "probe"):
                    obs[key] = torch.where(mask, reset_obs[key], obs[key])

        z_all = torch.cat(z_list, dim=0)
        pred_all = torch.cat(pred_list, dim=0)
        target_all = torch.cat(target_list, dim=0)
        dyn = torch.stack(dyn_losses).mean()
        contrast, retrieval1 = mdp_contrast_loss(pred_all, target_all)
        idm = torch.stack(idm_losses).mean()
        rew = torch.stack(reward_losses).mean()
        done_loss = torch.stack(done_losses).mean()
        val = torch.stack(value_losses).mean()
        eff_rank = mdp_effective_rank(z_all.detach())
        loss = (
            float(cfg.MDP_DYN_W) * dyn
            + float(cfg.MDP_CONTRAST_W) * contrast
            + float(cfg.MDP_IDM_W) * idm
            + float(cfg.MDP_REWARD_W) * rew
            + float(cfg.MDP_DONE_W) * done_loss
            + float(cfg.MDP_VALUE_W) * val
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdp.parameters(), float(cfg.MAX_GRAD_NORM))
        opt.step()
        info = {
            "mdp_loss": float(loss.detach().cpu()),
            "mdp_dyn": float(dyn.detach().cpu()),
            "mdp_contrast": float(contrast.detach().cpu()),
            "mdp_idm": float(idm.detach().cpu()),
            "mdp_reward": float(rew.detach().cpu()),
            "mdp_done": float(done_loss.detach().cpu()),
            "mdp_value": float(val.detach().cpu()),
            "mdp_retrieval1": float(retrieval1.detach().cpu()),
            "mdp_eff_rank": float(eff_rank.detach().cpu()),
        }
    return info


def std_info(model: ActorCritic) -> dict[str, float]:
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
            "mdp_path": str(args_cli.mdp or cfg.MDP_PATH),
            "info": info,
        },
        path,
    )


def save_mdp(path: Path, mdp: MDPStateNet, update: int, info: dict):
    torch.save(
        {
            "model": mdp.state_dict(),
            "model_cfg": {
                "input_shape": mdp.input_shape,
                "z_dim": mdp.z_dim,
                "action_dim": mdp.action_dim,
                "belief_dim": mdp.belief_dim,
                "feature_dim": mdp.feature_dim,
                "hidden_dim": mdp.hidden_dim,
                "pretrained": False,
                "freeze_backbone": bool(cfg.FREEZE_MDP_BACKBONE),
                "train_layer3": bool(getattr(mdp, "train_layer3", False)),
            },
            "update": int(update),
            "info": info,
        },
        path,
    )


def val(
    env: MDPStudentEnv,
    mdp: MDPStateNet,
    model: ActorCritic,
    max_steps: int,
    step: int,
    update: int,
    traj_writer: csv.DictWriter | None,
) -> dict[str, float]:
    env.reset()
    zero_action = torch.zeros(env.num_envs, int(env.cfg.action_space), device=env.device)
    mdp_state = mdp_observe(mdp, camera_rgb(env), zero_action, None)
    obs_t = mdp_state["state_feature"]
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
            _obs, reward, terminated, truncated, _ = env.step(action)
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
            mdp_state = next_mdp_state(mdp, env, mdp_state, action, terminated | truncated)
            obs_t = mdp_state["state_feature"]
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


def main():
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = out_dir()
    env = make_env()
    device = torch.device(env.device)
    mdp = load_mdp(device)
    teacher = load_teacher(device)
    obs_dim = int(mdp.belief_dim + mdp.z_dim)
    write_config(log_dir, args_cli.mdp or Path(cfg.MDP_PATH), obs_dim)
    log_file, log_writer = open_csv(log_dir / "log.csv", LOG_FIELDS)
    val_file, val_writer = open_csv(log_dir / "val.csv", VAL_FIELDS)
    train_traj_file, train_traj_writer = open_csv(log_dir / "traj_train_env0.csv", TRAJ_FIELDS)
    val_traj_file, val_traj_writer = open_csv(log_dir / "traj_val.csv", TRAJ_FIELDS)
    update_dir = log_dir / "updates"
    if int(cfg.SAVE_UPDATE_EVERY) > 0:
        update_dir.mkdir(parents=True, exist_ok=True)
    mdp_update_dir = log_dir / "mdp_updates"
    if bool(cfg.TRAIN_MDP):
        mdp_update_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(log_dir / "tb"))

    activation = ACTIVATIONS[str(cfg.ACTIVATION)]
    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=int(env.cfg.action_space),
        pi_hidden=list(cfg.POLICY_NET),
        vf_hidden=list(cfg.VALUE_NET),
        activation=activation,
        init_std=float(cfg.STD_INIT),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.LR), eps=1e-5)
    mdp_opt = torch.optim.Adam((param for param in mdp.parameters() if param.requires_grad), lr=float(cfg.MDP_LR), eps=1e-5)
    teacher_coef = float(cfg.TEACHER_LOSS)
    teacher_min = float(cfg.TEACHER_LOSS_MIN)
    teacher_down_success = float(cfg.TEACHER_DOWN_SUCCESS)
    teacher_up_out = float(cfg.TEACHER_UP_OUT)
    teacher_up_timeout = float(cfg.TEACHER_UP_TIMEOUT)

    obs, _ = env.reset()
    priv_t = obs["policy"]
    zero_action = torch.zeros(env.num_envs, int(env.cfg.action_space), device=device)
    mdp_state = mdp_observe(mdp, camera_rgb(env), zero_action, None)
    obs_t = mdp_state["state_feature"]

    steps_per_update = env.num_envs * int(cfg.N_STEPS)
    total_steps = int(cfg.TOTAL_STEPS)
    updates = int(math.ceil(total_steps / steps_per_update))
    print(
        f"[INFO] mdp student PPO out={log_dir} envs={env.num_envs} total_steps={total_steps} "
        f"updates={updates} n_steps={cfg.N_STEPS} obs_dim={obs_dim} device={env.device} "
        f"teacher_loss={teacher_coef:.3f}"
    )
    start_time = time.perf_counter()
    last_info = {}
    best = -1.0
    try:
        for update in range(1, updates + 1):
            train_mdp_now = bool(cfg.TRAIN_MDP) and update > int(cfg.MDP_WARMUP) and update % int(cfg.MDP_EVERY) == 0
            rollout = Rollout(
                steps=int(cfg.N_STEPS),
                envs=env.num_envs,
                obs_dim=obs_dim,
                act_dim=int(env.cfg.action_space),
                device=device,
            )
            done_count = 0
            success_count = 0
            fail_count = 0
            timeout_count = 0
            collect_start = time.perf_counter()
            if train_mdp_now:
                image_buf = torch.empty(
                    (int(cfg.N_STEPS), env.num_envs, env.cfg.image_height, env.cfg.image_width, 3),
                    dtype=torch.uint8,
                    device="cpu",
                )
                next_image_buf = torch.empty_like(image_buf)
                action_buf = torch.empty((int(cfg.N_STEPS), env.num_envs, int(env.cfg.action_space)), device="cpu")
                reward_buf = torch.empty((int(cfg.N_STEPS), env.num_envs), device="cpu")
                done_buf = torch.empty((int(cfg.N_STEPS), env.num_envs), device="cpu")
            priv_buf = torch.zeros(
                (int(cfg.N_STEPS), env.num_envs, int(env.cfg.observation_space)),
                device=device,
            )

            model.train()
            for t in range(int(cfg.N_STEPS)):
                if train_mdp_now:
                    image_buf[t].copy_(camera_rgb(env).detach().cpu())
                with torch.no_grad():
                    action, logp, value = model.act(obs_t)
                obs_next, reward, terminated, truncated, _ = env.step(action)
                raw_reward = reward.clone()
                timeout = truncated & (~terminated)
                if torch.any(timeout):
                    if env.last_rgb is None:
                        raise RuntimeError("timeout bootstrap requires env.last_rgb")
                    with torch.no_grad():
                        terminal_obs = terminal_mdp_obs(mdp, env, mdp_state, action, timeout)
                        terminal_value = model.critic(terminal_obs).squeeze(-1)
                    reward = reward.clone()
                    reward[timeout] += float(cfg.GAMMA) * terminal_value
                done = terminated | truncated
                if train_mdp_now:
                    next_image = camera_rgb(env).detach().cpu()
                    if torch.any(done):
                        next_image[done.cpu()] = env.terminal_rgb[done].detach().cpu()
                    next_image_buf[t].copy_(next_image)
                    action_buf[t].copy_(action.detach().cpu())
                    reward_buf[t].copy_(raw_reward.detach().cpu())
                    done_buf[t].copy_(done.detach().float().cpu())
                rollout.add(t, obs_t, action, logp, reward, done, value)
                priv_buf[t].copy_(priv_t.detach())
                step_now = min((update - 1) * steps_per_update + (t + 1) * env.num_envs, total_steps)
                train_traj_writer.writerow(
                    traj_row(env, "train", step_now, update, t, 0, reward[0], bool(done[0]))
                )
                if torch.any(done):
                    done_count += int(done.sum().item())
                    success_count += int((env.last_success & done).sum().item())
                    fail_count += int((env.last_fail & done).sum().item())
                    timeout_count += int(((env.last_timeout | truncated) & done).sum().item())
                mdp_state = next_mdp_state(mdp, env, mdp_state, action, done)
                obs_t = mdp_state["state_feature"]
                priv_t = obs_next["policy"]

            with torch.no_grad():
                last_value = model.critic(obs_t).squeeze(-1)
            batch = rollout.make_batch(
                last_value=last_value,
                gamma=float(cfg.GAMMA),
                lam=float(cfg.GAE_LAMBDA),
                norm_adv=bool(cfg.NORMALIZE_ADVANTAGE),
            )
            priv_batch = priv_buf.reshape(-1, priv_buf.shape[-1])
            batch.teacher_actions = (
                teacher_actions(teacher, priv_batch) if teacher is not None and teacher_coef > 0.0 else None
            )
            collect_s = time.perf_counter() - collect_start
            learn_start = time.perf_counter()
            mdp_info = {
                "mdp_loss": 0.0,
                "mdp_dyn": 0.0,
                "mdp_contrast": 0.0,
                "mdp_idm": 0.0,
                "mdp_reward": 0.0,
                "mdp_done": 0.0,
                "mdp_value": 0.0,
                "mdp_retrieval1": 0.0,
                "mdp_eff_rank": 0.0,
            }
            if train_mdp_now:
                loss = {"pi_loss": 0.0, "v_loss": 0.0, "teacher_loss": 0.0, "entropy": 0.0, "kl": 0.0}
                mdp_info = train_mdp_online(
                    mdp,
                    mdp_opt,
                    image_buf,
                    next_image_buf,
                    action_buf,
                    reward_buf,
                    done_buf,
                )
                save_mdp(mdp_update_dir / f"update_{update:06d}.pt", mdp, update, mdp_info)
            else:
                loss = ppo_update(
                    model=model,
                    opt=opt,
                    batch=batch,
                    batch_size=int(cfg.BATCH_SIZE),
                    epochs=int(cfg.N_EPOCHS),
                    clip=float(cfg.CLIP_RANGE),
                    vf_coef=float(cfg.VF_COEF),
                    ent_coef=float(cfg.ENT_COEF),
                    max_grad_norm=float(cfg.MAX_GRAD_NORM),
                    clip_vf=cfg.CLIP_RANGE_VF,
                    teacher_coef=teacher_coef,
                )
            learn_s = time.perf_counter() - learn_start
            step_done = min(update * steps_per_update, total_steps)
            std = std_info(model)
            info = {
                "update": update,
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
                **loss,
                **std,
                "teacher_coef": teacher_coef,
                "mdp_update": int(train_mdp_now),
                **mdp_info,
            }
            last_info = info
            log_writer.writerow(info)
            log_file.flush()
            train_traj_file.flush()

            if int(cfg.SAVE_EVERY) > 0 and update % int(cfg.SAVE_EVERY) == 0:
                save_model(log_dir / f"model_{update}.pt", model, opt, update, step_done, best, teacher_coef, info)
            if int(cfg.SAVE_UPDATE_EVERY) > 0 and update % int(cfg.SAVE_UPDATE_EVERY) == 0:
                save_model(update_dir / f"update_{update:06d}.pt", model, opt, update, step_done, best, teacher_coef, info)

            val_info = None
            if int(cfg.VAL_EVERY) > 0 and update % int(cfg.VAL_EVERY) == 0:
                write_val_traj = int(cfg.VAL_TRAJ_EVERY) > 0 and update % int(cfg.VAL_TRAJ_EVERY) == 0
                val_info = val(
                    env,
                    mdp,
                    model,
                    int(cfg.VAL_STEPS),
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
                val_row = {"update": update, "step": step_done, **val_info, **std, "teacher_coef": teacher_coef}
                val_writer.writerow(val_row)
                val_file.flush()
                if write_val_traj:
                    val_traj_file.flush()
                for key in ("success_rate", "fail_rate", "timeout_rate", "mean_return", "mean_len"):
                    tb.add_scalar(f"val/{key}", val_info[key], step_done)
                obs, _ = env.reset()
                priv_t = obs["policy"]
                mdp_state = mdp_observe(mdp, camera_rgb(env), zero_action, None)
                obs_t = mdp_state["state_feature"]
                if bool(cfg.SAVE_BEST) and update >= int(cfg.BEST_WARMUP):
                    rate = val_info["success_rate"]
                    if rate >= best + float(cfg.BEST_MARGIN):
                        best = rate
                        save_model(log_dir / "best.pt", model, opt, update, step_done, best, teacher_coef, val_row)
                        save_mdp(log_dir / "best_mdp.pt", mdp, update, val_row)
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
            tb.add_scalar("train/entropy", info["entropy"], step_done)
            tb.add_scalar("train/kl", info["kl"], step_done)
            tb.add_scalar("train/teacher_loss", info["teacher_loss"], step_done)
            tb.add_scalar("policy/std_mean", info["std_mean"], step_done)
            tb.add_scalar("policy/std_min", info["std_min"], step_done)
            tb.add_scalar("policy/std_max", info["std_max"], step_done)
            tb.add_scalar("policy/teacher_coef", teacher_coef, step_done)
            tb.add_scalar("mdp/update", info["mdp_update"], step_done)
            tb.add_scalar("mdp/loss", info["mdp_loss"], step_done)
            tb.add_scalar("mdp/dyn", info["mdp_dyn"], step_done)
            tb.add_scalar("mdp/contrast", info["mdp_contrast"], step_done)
            tb.add_scalar("mdp/idm", info["mdp_idm"], step_done)
            tb.add_scalar("mdp/reward", info["mdp_reward"], step_done)
            tb.add_scalar("mdp/done", info["mdp_done"], step_done)
            tb.add_scalar("mdp/value", info["mdp_value"], step_done)
            tb.add_scalar("mdp/retrieval1", info["mdp_retrieval1"], step_done)
            tb.add_scalar("mdp/eff_rank", info["mdp_eff_rank"], step_done)

            if update % int(cfg.LOG_EVERY) == 0:
                groups = [
                    (
                        "time",
                        [
                            ("update", update),
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
                            ("std", info["std_mean"]),
                            ("teacher_coef", teacher_coef),
                        ],
                    ),
                    (
                        "train",
                        [
                            ("teacher_loss", info["teacher_loss"]),
                            ("kl", info["kl"]),
                            ("entropy", info["entropy"]),
                        ],
                    ),
                    (
                        "mdp",
                        [
                            ("update", info["mdp_update"]),
                            ("loss", info["mdp_loss"]),
                            ("dyn", info["mdp_dyn"]),
                            ("con", info["mdp_contrast"]),
                            ("idm", info["mdp_idm"]),
                            ("retr", info["mdp_retrieval1"]),
                            ("rank", info["mdp_eff_rank"]),
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
        save_mdp(log_dir / "last_mdp.pt", mdp, updates, last_info)
        print(f"[INFO] saved {log_dir / 'last.pt'}")
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
