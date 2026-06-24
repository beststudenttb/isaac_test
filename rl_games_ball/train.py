"""Train rl-games PPO on the ball camera environment.

Run from project root:

    ./IsaacLab/isaaclab.sh -p rl_games_ball/train.py --num-envs 64
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train rl-games PPO on ball camera env.")
parser.add_argument("--cfg", type=Path, default=Path("rl_games_ball/rl_games_ppo_cnn_lstm_asym.yaml"))
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--max-epochs", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
args_cli.livestream = 0
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_games_ball.env import RlGamesBallEnvCfg, make_env


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    agent_cfg = load_cfg(args_cli.cfg)

    env_cfg = RlGamesBallEnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = make_env(env_cfg)

    params = agent_cfg["params"]
    if args_cli.max_epochs is not None:
        params["config"]["max_epochs"] = int(args_cli.max_epochs)
    if args_cli.checkpoint is not None:
        params["load_checkpoint"] = True
        params["load_path"] = str(args_cli.checkpoint)

    rl_device = params["config"]["device"]
    clip_obs = params["env"].get("clip_observations", math.inf)
    clip_actions = params["env"].get("clip_actions", math.inf)
    obs_groups = params["env"].get("obs_groups")
    concate_obs_groups = params["env"].get("concate_obs_groups", True)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    params["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False})
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
