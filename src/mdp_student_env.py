"""MDP latent student training environment config."""

from __future__ import annotations

from isaaclab.utils import configclass

import torch

from src.sb3_env import BallPPOEnv, BallPPOEnvCfg


@configclass
class MDPStudentEnvCfg(BallPPOEnvCfg):
    use_camera = True
    read_camera = True


class MDPStudentEnv(BallPPOEnv):
    def __init__(self, cfg: MDPStudentEnvCfg):
        super().__init__(cfg)
        self.terminal_rgb = None

    def compute_reward(self) -> torch.Tensor:
        reward = super().compute_reward()
        done = self.reset_terminated | self.reset_time_outs
        if torch.any(done) and self.camera is not None:
            image = self.camera.data.output["rgb"]
            if image.shape[-1] > 3:
                image = image[..., :3]
            if self.terminal_rgb is None or self.terminal_rgb.shape != image.shape:
                self.terminal_rgb = torch.empty_like(image)
            self.terminal_rgb[done] = image[done]
        return reward


def make_mdp_student_env(cfg: MDPStudentEnvCfg | None = None) -> MDPStudentEnv:
    return MDPStudentEnv(cfg or MDPStudentEnvCfg())
