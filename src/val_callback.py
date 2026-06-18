"""Deterministic validation callback for the teacher PPO."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


TRAJ_FIELDS = [
    "phase",
    "step",
    "rollout",
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


def make_traj_row(base_env, phase: str, step: int, rollout: int, t: int, env_id: int, reward, done: bool) -> dict:
    action = base_env.last_move_actions[env_id].detach().cpu().numpy()
    robot_xy = base_env.last_robot_xy[env_id].detach().cpu().numpy()
    target_xy = base_env.last_target_xy[env_id].detach().cpu().numpy()
    return {
        "phase": phase,
        "step": int(step),
        "rollout": int(rollout),
        "t": int(t),
        "env_id": env_id,
        "px_x": f"{float(base_env.last_px_x[env_id].item()):.4f}",
        "dist": f"{float(base_env.last_dist[env_id].item()):.4f}",
        "a_x": f"{float(action[0]):.4f}",
        "a_y": f"{float(action[1]):.4f}",
        "a_w": f"{float(action[2]):.4f}",
        "reward": f"{float(reward):.4f}",
        "done": int(done),
        "success": int(bool(base_env.last_success[env_id].item())),
        "fail": int(bool(base_env.last_fail[env_id].item())),
        "timeout": int(bool(base_env.last_timeout[env_id].item())),
        "robot_x": f"{float(robot_xy[0]):.4f}",
        "robot_y": f"{float(robot_xy[1]):.4f}",
        "head_yaw": f"{float(base_env.last_head_yaw[env_id].item()):.4f}",
        "target_x": f"{float(target_xy[0]):.4f}",
        "target_y": f"{float(target_xy[1]):.4f}",
    }


class TrainTrajCallback(BaseCallback):
    def __init__(self, out_dir: Path):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.rollouts = 0
        self.t = 0
        self.file = None
        self.writer = None

    def _on_training_start(self) -> None:
        self.file = (self.out_dir / "traj_train_env0.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=TRAJ_FIELDS)
        self.writer.writeheader()

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        row = make_traj_row(
            self.training_env.unwrapped,
            phase="train",
            step=self.num_timesteps,
            rollout=self.rollouts,
            t=self.t,
            env_id=0,
            reward=rewards[0],
            done=bool(dones[0]),
        )
        self.writer.writerow(row)
        self.t += 1
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts += 1
        if self.file is not None:
            self.file.flush()

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None


class ValCallback(BaseCallback):
    def __init__(
        self,
        out_dir: Path,
        every_rollouts: int,
        max_steps: int,
        warmup_steps: int,
        margin: float,
        save_best: bool,
        save_traj: bool,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.every_rollouts = int(every_rollouts)
        self.max_steps = int(max_steps)
        self.warmup_steps = int(warmup_steps)
        self.margin = float(margin)
        self.save_best = bool(save_best)
        self.save_traj = bool(save_traj)
        self.rollouts = 0
        self.best = -1.0

    def _on_rollout_end(self) -> None:
        self.rollouts += 1
        if self.every_rollouts <= 0 or self.rollouts % self.every_rollouts != 0:
            return

        result = self.run_val()
        self.log_result(result)
        print(
            f"[INFO] val step={result['step']} success={result['success_rate']:.3f} "
            f"fail={result['fail_rate']:.3f} timeout={result['timeout_rate']:.3f} "
            f"return={result['mean_return']:.2f} len={result['mean_len']:.1f}"
        )
        if self.save_best and self.num_timesteps >= self.warmup_steps:
            if result["success_rate"] >= self.best + self.margin:
                self.best = result["success_rate"]
                self.model.save(self.out_dir / "best_val")
                self.write_best(result)
                print(
                    f"[INFO] new best_val success_rate={result['success_rate']:.3f} "
                    f"step={self.num_timesteps}"
                )

    def _on_step(self) -> bool:
        return True

    def run_val(self) -> dict:
        env = self.training_env
        base_env = env.unwrapped
        obs = env.reset()
        num_envs = int(env.num_envs)
        max_steps = self.max_steps if self.max_steps > 0 else int(base_env.max_episode_length)

        done_once = np.zeros(num_envs, dtype=bool)
        returns = np.zeros(num_envs, dtype=np.float64)
        lengths = np.zeros(num_envs, dtype=np.int64)
        success = np.zeros(num_envs, dtype=bool)
        fail = np.zeros(num_envs, dtype=bool)
        timeout = np.zeros(num_envs, dtype=bool)

        val_file = None
        val_writer = None
        if self.save_traj:
            val_path = self.out_dir / "traj_val.csv"
            val_exists = val_path.exists()
            val_file = val_path.open("a", newline="", encoding="utf-8")
            val_writer = csv.DictWriter(val_file, fieldnames=TRAJ_FIELDS)
            if not val_exists:
                val_writer.writeheader()

        for val_t in range(max_steps):
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, _infos = env.step(actions)
            active = ~done_once
            if self.save_traj:
                for env_id in np.flatnonzero(active):
                    val_writer.writerow(
                        make_traj_row(
                            base_env,
                            phase="val",
                            step=self.num_timesteps,
                            rollout=self.rollouts,
                            t=val_t,
                            env_id=int(env_id),
                            reward=rewards[env_id],
                            done=bool(dones[env_id]),
                        )
                    )
            returns[active] += rewards[active]
            lengths[active] += 1

            done_ids = np.flatnonzero(dones & active)
            if len(done_ids) > 0:
                success_buf = base_env.last_success.detach().cpu().numpy()
                fail_buf = base_env.last_fail.detach().cpu().numpy()
                timeout_buf = base_env.last_timeout.detach().cpu().numpy()
                success[done_ids] = success_buf[done_ids]
                fail[done_ids] = fail_buf[done_ids]
                timeout[done_ids] = timeout_buf[done_ids]
                done_once[done_ids] = True
                if done_once.all():
                    break

        if val_file is not None:
            val_file.close()

        timeout[~done_once] = True
        lengths[~done_once] = max_steps

        obs = env.reset()
        self.model._last_obs = obs
        self.model._last_episode_starts = np.ones(num_envs, dtype=bool)

        return {
            "step": int(self.num_timesteps),
            "rollout": int(self.rollouts),
            "episodes": int(num_envs),
            "success_rate": float(success.mean()),
            "fail_rate": float(fail.mean()),
            "timeout_rate": float(timeout.mean()),
            "mean_return": float(returns.mean()),
            "mean_len": float(lengths.mean()),
        }

    def write_best(self, result: dict):
        lines = [f"{key} = {value}" for key, value in result.items()]
        (self.out_dir / "best_val_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def log_result(self, result: dict):
        for key in ("success_rate", "fail_rate", "timeout_rate", "mean_return", "mean_len"):
            self.logger.record(f"val/{key}", result[key])
