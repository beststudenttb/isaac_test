"""off-policy 线的 Isaac env 工厂(Isaac 绑定集中在这一层,agent/buffer/logging 保持无 Isaac)。

make_env 把散在 train/val 各脚本里逐字重复的 env_cfg 构造收敛成一处。接受 cfg 与
noise/num_envs/random_stop/device 作参数,不依赖任何全局 args_cli。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch

from src.mdp_student_env import MDPStudentEnv, MDPStudentEnvCfg, make_mdp_student_env
from src.noise_env import NoiseStudentEnvCfg, make_noise_student_env


def end_range(cfg, random_stop: bool) -> tuple[float, float, float, float]:
    if random_stop:
        return (float(cfg.RANDOM_END_D_MIN), float(cfg.RANDOM_END_D_MAX),
                float(cfg.RANDOM_END_X_MIN), float(cfg.RANDOM_END_X_MAX))
    return (float(cfg.END_D_MIN), float(cfg.END_D_MAX),
            float(cfg.END_X_MIN), float(cfg.END_X_MAX))


def make_env(cfg, *, noise: bool, num_envs: int | None, random_stop: bool, device: str) -> MDPStudentEnv:
    end_d_min, end_d_max, end_x_min, end_x_max = end_range(cfg, random_stop)
    env_cfg = NoiseStudentEnvCfg() if noise else MDPStudentEnvCfg()
    env_cfg.seed = int(cfg.SEED)
    env_cfg.episode_length_s = float(cfg.EPISODE_S)
    env_cfg.stop_n = int(cfg.STOP_N)
    env_cfg.scene.num_envs = int(num_envs) if num_envs is not None else int(cfg.NUM_ENVS)
    env_cfg.sim.device = device
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
    make_env_fn = make_noise_student_env if noise else make_mdp_student_env
    return make_env_fn(env_cfg)


def camera_rgb(env: MDPStudentEnv) -> torch.Tensor:
    image = env.camera.data.output["rgb"]
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image
