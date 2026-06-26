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
        self.terminal_end_obs = None

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
            end_obs = self.end_obs()
            if self.terminal_end_obs is None or self.terminal_end_obs.shape != end_obs.shape:
                self.terminal_end_obs = torch.empty_like(end_obs)
            self.terminal_end_obs[done] = end_obs[done]
        return reward

    def mdp_policy_obs(self, mdp_state: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((mdp_state["state_feature"], self.end_obs()), dim=-1)


def make_mdp_student_env(cfg: MDPStudentEnvCfg | None = None) -> MDPStudentEnv:
    return MDPStudentEnv(cfg or MDPStudentEnvCfg())
