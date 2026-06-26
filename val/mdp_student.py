"""Offline validate MDP student update checkpoints.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p val/mdp_student.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import mdp_student_cfg as cfg


parser = argparse.ArgumentParser(description="Offline validate MDP student update checkpoints.")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--num-episodes", type=int, default=int(cfg.NUM_EPISODES))
parser.add_argument("--start", type=int, default=int(cfg.START))
parser.add_argument("--stride", type=int, default=int(cfg.STRIDE))
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cv_extractor.mdp_state import MDPStateNet
from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.ppo import ActorCritic


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

VAL_FIELDS = [
    "checkpoint",
    "mdp_checkpoint",
    "update",
    "mdp_update",
    "step",
    "num_envs",
    "num_episodes",
    "episodes",
    "success_rate",
    "fail_rate",
    "timeout_rate",
    "mean_return",
    "mean_len",
]

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


def root_dir() -> Path:
    return Path(cfg.OUT_DIR)


def update_num(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def make_env() -> MDPStudentEnv:
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
    env_cfg.end_d_min = float(cfg.END_D_MIN)
    env_cfg.end_d_max = float(cfg.END_D_MAX)
    env_cfg.end_x_min = float(cfg.END_X_MIN)
    env_cfg.end_x_max = float(cfg.END_X_MAX)
    return make_mdp_student_env(env_cfg)


def camera_rgb(env: MDPStudentEnv) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def load_mdp(path: Path, device: torch.device) -> MDPStateNet:
    ckpt = torch.load(path, map_location=device)
    model = MDPStateNet(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if bool(cfg.FREEZE_MDP_BACKBONE):
        for module in (model.stem, model.layer1, model.layer2, model.layer3):
            for param in module.parameters():
                param.requires_grad_(False)
    return model


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


def next_mdp_state(
    mdp: MDPStateNet,
    env: MDPStudentEnv,
    prev_state: dict[str, torch.Tensor],
    action: torch.Tensor,
    done: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = {
        "belief": prev_state["belief"].clone(),
        "z": prev_state["z"].clone(),
    }
    prev_action = action.clone()
    if torch.any(done):
        state["belief"][done] = 0.0
        state["z"][done] = 0.0
        prev_action[done] = 0.0
    return mdp_observe(mdp, camera_rgb(env), prev_action, state)


def load_policy(path: Path, obs_dim: int, device: torch.device) -> tuple[ActorCritic, dict]:
    ckpt = torch.load(path, map_location=device)
    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=3,
        pi_hidden=list(cfg.POLICY_NET),
        vf_hidden=list(cfg.VALUE_NET),
        activation=ACTIVATIONS[str(cfg.ACTIVATION)],
        init_std=float(cfg.STD_INIT),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


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
    width_r = max(len(str(right)) for _, right in rows)
    line = "-" * (width_l + width_r + 7)
    print(line)
    for left, right in rows:
        print(f"| {left:<{width_l}} | {fmt_value(right):>{width_r}} |")
    print(line)


def val(
    env: MDPStudentEnv,
    mdp: MDPStateNet,
    model: ActorCritic,
    num_episodes: int,
    update: int,
    step: int,
    traj_writer: csv.DictWriter | None = None,
    env0_traj_writer: csv.DictWriter | None = None,
) -> dict[str, float]:
    steps = int(cfg.VAL_STEPS) if int(cfg.VAL_STEPS) > 0 else int(env.max_episode_length)
    episodes = env.num_envs * int(num_episodes)
    success_count = 0
    fail_count = 0
    timeout_count = 0
    return_sum = 0.0
    len_sum = 0.0
    zero_action = torch.zeros(env.num_envs, int(env.cfg.action_space), device=env.device)

    model.eval()
    mdp.eval()
    with torch.no_grad():
        for episode_id in range(int(num_episodes)):
            env.reset()
            mdp_state = mdp_observe(mdp, camera_rgb(env), zero_action, None)
            obs_t = env.mdp_policy_obs(mdp_state)
            done_once = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            returns = torch.zeros(env.num_envs, device=env.device)
            lengths = torch.zeros(env.num_envs, device=env.device)
            success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            fail = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            timeout = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

            for t in range(steps):
                action = model.predict(obs_t)
                _obs, reward, terminated, truncated, _info = env.step(action)
                active = ~done_once
                returns[active] += reward[active]
                lengths[active] += 1
                done = (terminated | truncated) & active
                row_t = episode_id * steps + t
                if traj_writer is not None:
                    for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
                        traj_writer.writerow(
                            traj_row(env, "offline_val", step, update, row_t, int(env_id), reward[int(env_id)], bool(done[int(env_id)]))
                        )
                if env0_traj_writer is not None and bool(active[0].item()):
                    env0_traj_writer.writerow(
                        traj_row(env, "offline_val_env0", step, update, row_t, 0, reward[0], bool(done[0]))
                    )
                if torch.any(done):
                    success[done] = env.last_success[done]
                    fail[done] = env.last_fail[done]
                    timeout[done] = env.last_timeout[done] | truncated[done]
                    done_once[done] = True
                    if bool(done_once.all()):
                        break
                mdp_state = next_mdp_state(mdp, env, mdp_state, action, terminated | truncated)
                obs_t = env.mdp_policy_obs(mdp_state)

            timeout[~done_once] = True
            lengths[~done_once] = steps
            success_count += int(success.sum().item())
            fail_count += int(fail.sum().item())
            timeout_count += int(timeout.sum().item())
            return_sum += float(returns.sum().cpu())
            len_sum += float(lengths.sum().cpu())

    return {
        "success_rate": success_count / episodes,
        "fail_rate": fail_count / episodes,
        "timeout_rate": timeout_count / episodes,
        "mean_return": return_sum / episodes,
        "mean_len": len_sum / episodes,
    }


def best_key(row: dict) -> tuple:
    return (
        float(row["success_rate"]),
        -float(row["fail_rate"]),
        -float(row["timeout_rate"]),
        float(row["mean_return"]),
    )


def pick_mdp(policy_update: int, mdp_paths: list[Path], policy_ckpt: dict, base_mdp_path: Path) -> Path:
    valid = [path for path in mdp_paths if update_num(path) <= int(policy_update)]
    if valid:
        return valid[-1]
    if base_mdp_path.exists():
        return base_mdp_path
    path = Path(policy_ckpt["mdp_path"])
    if not path.exists():
        raise FileNotFoundError(f"MDP checkpoint not found: {base_mdp_path}")
    return path


def mdp_path_from_row(root: Path, row: dict, base_mdp_path: Path) -> Path:
    if int(row["mdp_update"]) > 0:
        return root / "mdp_updates" / row["mdp_checkpoint"]
    return base_mdp_path


def load_existing_best(root: Path, out_csv: Path, base_mdp_path: Path) -> tuple[dict | None, Path | None, Path | None]:
    if not out_csv.exists():
        return None, None, None
    best_row = None
    with out_csv.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if best_row is None or best_key(row) > best_key(best_row):
                best_row = row
    if best_row is None:
        return None, None, None
    return (
        best_row,
        root / "updates" / best_row["checkpoint"],
        mdp_path_from_row(root, best_row, base_mdp_path),
    )


def write_config(path: Path, env: MDPStudentEnv):
    lines = [
        f"run_name = 'mdp_student'",
        f"num_envs = {env.num_envs!r}",
        f"num_episodes = {int(args_cli.num_episodes)!r}",
        f"start = {int(args_cli.start)!r}",
        f"stride = {int(args_cli.stride)!r}",
        f"episodes_per_checkpoint = {env.num_envs * int(args_cli.num_episodes)!r}",
        f"val_steps = {int(cfg.VAL_STEPS)!r}",
        f"episode_s = {float(cfg.EPISODE_S)!r}",
        f"stop_n = {int(cfg.STOP_N)!r}",
        f"seed = {int(cfg.SEED)!r}",
        f"device = {env.device!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    seed = int(cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = root_dir()
    update_dir = root / "updates"
    mdp_update_dir = root / "mdp_updates"
    if not update_dir.exists():
        raise FileNotFoundError(f"update dir not found: {update_dir}")

    policy_paths = sorted(update_dir.glob("update_*.pt"))
    mdp_paths = sorted(mdp_update_dir.glob("update_*.pt")) if mdp_update_dir.exists() else []
    base_mdp_path = args_cli.mdp or Path(cfg.MDP_PATH)
    if not mdp_paths and not base_mdp_path.exists():
        raise FileNotFoundError(f"MDP checkpoint not found: {base_mdp_path}")
    if not policy_paths:
        raise FileNotFoundError(f"no policy checkpoints found: {update_dir}")
    start = int(args_cli.start)
    stride = int(args_cli.stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive: {stride}")
    policy_paths = [
        path for path in policy_paths
        if update_num(path) >= start and (update_num(path) - start) % stride == 0
    ]
    if not policy_paths:
        raise FileNotFoundError(f"no policy checkpoints found after start={start}, stride={stride}: {update_dir}")

    env = make_env()
    device = torch.device(env.device)
    write_config(root / "offline_val_config.py", env)
    out_csv = root / "offline_val.csv"
    env0_traj_csv = root / "traj_offline_env0.csv"
    best_row = None
    best_policy_path = None
    best_mdp_path = None
    if start > 0:
        best_row, best_policy_path, best_mdp_path = load_existing_best(root, out_csv, base_mdp_path)

    print(
        f"[INFO] offline val run=mdp_student policies={len(policy_paths)} mdp_updates={len(mdp_paths)} "
        f"start={start} stride={stride} envs={env.num_envs} episodes={args_cli.num_episodes} device={env.device}"
    )
    try:
        mode = "a" if int(args_cli.start) > 0 else "w"
        with out_csv.open(mode, newline="", encoding="utf-8") as file, env0_traj_csv.open(mode, newline="", encoding="utf-8") as env0_file:
            writer = csv.DictWriter(file, fieldnames=VAL_FIELDS)
            env0_writer = csv.DictWriter(env0_file, fieldnames=TRAJ_FIELDS)
            if mode == "w":
                writer.writeheader()
                env0_writer.writeheader()

            for policy_path in policy_paths:
                policy_update = update_num(policy_path)
                policy_ckpt = torch.load(policy_path, map_location="cpu")
                mdp_path = pick_mdp(policy_update, mdp_paths, policy_ckpt, base_mdp_path)
                mdp = load_mdp(mdp_path, device)
                obs_dim = int(mdp.belief_dim + mdp.z_dim + 2)
                model, _ckpt = load_policy(policy_path, obs_dim, device)
                update = int(policy_ckpt.get("update", policy_update))
                step = int(policy_ckpt.get("step", -1))
                mdp_update = update_num(mdp_path) if mdp_path.parent.name == "mdp_updates" else 0
                info = val(
                    env,
                    mdp,
                    model,
                    int(args_cli.num_episodes),
                    update,
                    step,
                    env0_traj_writer=env0_writer,
                )
                row = {
                    "checkpoint": policy_path.name,
                    "mdp_checkpoint": mdp_path.name,
                    "update": update,
                    "mdp_update": mdp_update,
                    "step": step,
                    "num_envs": env.num_envs,
                    "num_episodes": int(args_cli.num_episodes),
                    "episodes": env.num_envs * int(args_cli.num_episodes),
                    **info,
                }
                writer.writerow(row)
                file.flush()
                env0_file.flush()
                if best_row is None or best_key(row) > best_key(best_row):
                    best_row = row
                    best_policy_path = policy_path
                    best_mdp_path = mdp_path
                print_log(
                    [
                        (
                            "model",
                            [
                                ("update", row["update"]),
                                ("mdp_update", row["mdp_update"]),
                                ("step", row["step"]),
                            ],
                        ),
                        (
                            "val",
                            [
                                ("episodes", row["episodes"]),
                                ("success", row["success_rate"]),
                                ("fail", row["fail_rate"]),
                                ("timeout", row["timeout_rate"]),
                                ("return", row["mean_return"]),
                                ("len", row["mean_len"]),
                            ],
                        ),
                    ]
                )

        if best_row is None or best_policy_path is None or best_mdp_path is None:
            raise RuntimeError("offline val did not produce best checkpoint")
        shutil.copy2(best_policy_path, root / "best_offline.pt")
        if best_mdp_path.parent.name == "mdp_updates":
            shutil.copy2(best_mdp_path, root / "best_offline_mdp.pt")
        (root / "best_offline_info.txt").write_text(
            "\n".join(f"{key} = {value}" for key, value in best_row.items()) + "\n",
            encoding="utf-8",
        )

        best_mdp = load_mdp(best_mdp_path, device)
        best_model, _ = load_policy(best_policy_path, int(best_mdp.belief_dim + best_mdp.z_dim), device)
        with (root / "traj_best_offline.csv").open("w", newline="", encoding="utf-8") as traj_file:
            traj_writer = csv.DictWriter(traj_file, fieldnames=TRAJ_FIELDS)
            traj_writer.writeheader()
            val(
                env,
                best_mdp,
                best_model,
                int(args_cli.num_episodes),
                int(best_row["update"]),
                int(best_row["step"]),
                traj_writer,
            )
        print(
            f"[INFO] best checkpoint={best_policy_path.name} mdp={best_mdp_path.name} "
            f"success={best_row['success_rate']:.3f} fail={best_row['fail_rate']:.3f} timeout={best_row['timeout_rate']:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
