"""任务三的 actor-critic:encoder 冻结,特征在 rollout 后一次性算好。

为什么不能沿用 `SPRFixedActorCritic`:它的 `obs()` 每次都调 encoder,而 PPO 的
每个 minibatch 都会调一次 —— encoder 是冻结的,这纯属白算。后果有两个:
  1. 慢:实测 learn_s(5.77s) > collect_s(4.08s),瓶颈在学习侧不在渲染。
  2. 直接挡住正确的超参:BATCH_SIZE=1024 时要把 1024 张 224x224 的图过 ResNet18,
     光输入 616MB、FPN 中间层 1.6GB,加 Isaac 常驻的 3-4GB,12GB 卡必爆(实测炸)。

这里把 encoder 只用在采集阶段(64 个 env,便宜),`evaluate()` 直接吃特征。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.ppo import mlp

ENCODE_CHUNK = 128


class FreeFeatureActorCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        goal_dim: int,
        act_dim: int,
        pi_hidden: list[int],
        vf_hidden: list[int],
        activation: type[nn.Module],
        init_std: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.goal_dim = int(goal_dim)
        self.act_dim = int(act_dim)
        obs_dim = int(encoder.z_dim) + self.goal_dim
        self.actor = mlp(obs_dim, pi_hidden, self.act_dim, activation)
        self.critic = mlp(obs_dim, vf_hidden, 1, activation)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), torch.log(torch.tensor(float(init_std)))))

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        return self.encoder.encode(image).float()

    @torch.no_grad()
    def encode_rollout(self, images: torch.Tensor) -> torch.Tensor:
        """整批 rollout 的图像 -> 特征,分块跑,结果留在 CPU(4096x256 只有 4MB)。"""
        device = next(self.parameters()).device
        out = torch.empty((len(images), int(self.encoder.z_dim)), dtype=torch.float32)
        for i in range(0, len(images), ENCODE_CHUNK):
            out[i : i + ENCODE_CHUNK] = self.encode(images[i : i + ENCODE_CHUNK].to(device)).cpu()
        return out

    def obs_from_feature(self, feature: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        feature = feature.to(device=goal.device, dtype=goal.dtype)
        return torch.cat((feature, goal), dim=1)

    def dist_from_obs(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    # --- 采集阶段:输入是图像 ---

    def act(self, image: torch.Tensor, goal: torch.Tensor):
        obs = self.obs_from_feature(self.encode(image), goal)
        dist = self.dist_from_obs(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), self.critic(obs).squeeze(-1)

    def value(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.critic(self.obs_from_feature(self.encode(image), goal)).squeeze(-1)

    def predict(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.actor(self.obs_from_feature(self.encode(image), goal))

    # --- 学习阶段:输入已经是特征,不再过 encoder ---

    def evaluate(self, feature: torch.Tensor, goal: torch.Tensor, action: torch.Tensor):
        obs = self.obs_from_feature(feature, goal)
        dist = self.dist_from_obs(obs)
        return (
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.critic(obs).squeeze(-1),
            self.actor(obs),
        )


class AdapterActorCritic(FreeFeatureActorCritic):
    """2026-09-10 "通识基底 + 策略训适配器":骨干(stem/layer1-3/FPN)冻结,proj(18816->256)+LayerNorm 从编码器
    的权重出发、交给 PPO 的 loss 一起训。缓存的是 proj 之前的 18816 维池化特征(fp16,33792 张约 1.3 GB),
    minibatch 里只过适配器和 actor/critic。t=0 时和冻结变体完全一样。"""

    def __init__(self, encoder: nn.Module, adapter_hidden: int = 0, **kw):
        super().__init__(encoder=encoder, **kw)
        net = encoder.net
        if adapter_hidden > 0:   # 两层适配器(2026-09-11):池化特征 -> hidden -> 256 + LayerNorm,能非线性地长出线性读不出的量(x)
            self.adapter = nn.Sequential(nn.Linear(net.proj.in_features, adapter_hidden), nn.Tanh(), nn.Linear(adapter_hidden, net.proj.out_features), nn.LayerNorm(net.proj.out_features))
        else:
            self.adapter = nn.Sequential(nn.Linear(net.proj.in_features, net.proj.out_features), nn.LayerNorm(net.proj.out_features))
            self.adapter[0].load_state_dict(net.proj.state_dict())
            self.adapter[1].load_state_dict(net.norm.state_dict())
        self.adapter.requires_grad_(True)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            pooled = self.encoder.pooled(image).float()
        return self.adapter(pooled)

    @torch.no_grad()
    def encode_rollout(self, images: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        self.encoder.eval()
        out = torch.empty((len(images), int(self.encoder.pooled_dim)), dtype=torch.float16)
        for i in range(0, len(images), ENCODE_CHUNK):
            out[i : i + ENCODE_CHUNK] = self.encoder.pooled(images[i : i + ENCODE_CHUNK].to(device)).half().cpu()
        return out

    def obs_from_feature(self, feature: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        feature = feature.to(device=goal.device, dtype=torch.float32)
        if feature.shape[-1] != self.encoder.z_dim:   # 学习阶段:缓存的是池化特征,先过适配器
            feature = self.adapter(feature)
        return torch.cat((feature, goal.to(torch.float32)), dim=1)
