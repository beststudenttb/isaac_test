"""No-prior visual encoder: image -> 256-d shared_feature, nothing else.

Ported from the Webots `FreeSpatialFeatureExtractor` (handoff doc 2026-09-03 §4.1).
The point of this class is what it does *not* have: no heat-map head, no
soft-argmax, no x/y head, no distance head, no bins.  Those all amount to
writing "there is one target in the image, report its x" into the architecture
before training starts.  What stays is ResNet18 + FPN (so a far, small target
survives downsampling) and a *fixed* spatial grid pool, which never asserts the
scene contains a single object.

Pooling uses an explicit avg_pool2d rather than adaptive pooling so the numbers
stay comparable with the Webots runs (grid=7 divides 56/28/14 exactly).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class FreeSpatialFeatureExtractor(nn.Module):
    """Image -> {"shared_feature": (B, feature_dim)}."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        fpn_channels: int = 128,
        grid: int = 7,
        feature_dim: int = 256,
        pretrained: bool = False,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.fpn_channels = int(fpn_channels)
        self.grid = int(grid)
        self.feature_dim = int(feature_dim)

        weights = ResNet18_Weights.DEFAULT if bool(pretrained) else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        c = self.fpn_channels
        self.lat2 = nn.Conv2d(64, c, kernel_size=1)
        self.lat3 = nn.Conv2d(128, c, kernel_size=1)
        self.lat4 = nn.Conv2d(256, c, kernel_size=1)
        self.smooth2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        pooled_dim = c * 3 * self.grid * self.grid
        self.proj = nn.Linear(pooled_dim, self.feature_dim)
        self.norm = nn.LayerNorm(self.feature_dim)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward_pooled(self, image) -> torch.Tensor:
        """image -> proj 之前的多尺度 7x7 池化特征 (B, 3*c*grid*grid)。适配器变体用:骨干冻结,proj+norm 交给策略训。"""
        x = self._to_nchw(image)
        x = (x - self.imagenet_mean) / self.imagenet_std
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        return torch.cat([self._grid_pool(p2), self._grid_pool(p3), self._grid_pool(p4)], dim=1)

    def forward(self, image) -> dict[str, torch.Tensor]:
        x = self._to_nchw(image)
        x = (x - self.imagenet_mean) / self.imagenet_std

        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)

        pooled = torch.cat([self._grid_pool(p2), self._grid_pool(p3), self._grid_pool(p4)], dim=1)
        feature = self.norm(self.proj(pooled))
        return {"shared_feature": feature}

    def _grid_pool(self, feature: torch.Tensor) -> torch.Tensor:
        k = feature.shape[-1] // self.grid
        return F.avg_pool2d(feature, kernel_size=k, stride=k).flatten(1)

    def _to_nchw(self, image) -> torch.Tensor:
        device = next(self.parameters()).device
        tensor = image.to(device)
        is_uint8 = tensor.dtype == torch.uint8
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float()
        if is_uint8:
            tensor = tensor / 255.0
        return tensor


class FrozenFreeEncoder(nn.Module):
    """任务三用:加载 stage2 的表征,冻结,套上 `SPRFixedActorCritic` 要的接口。

    只需要 `.encode(image)` 和 `.z_dim`,所以 actor-critic 那边一行都不用改。
    """

    def __init__(self, checkpoint: str, **encoder_cfg):
        super().__init__()
        self.net = FreeSpatialFeatureExtractor(**encoder_cfg)
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.net.eval()
        self.net.requires_grad_(False)
        self.z_dim = int(encoder_cfg["feature_dim"])
        self.pooled_dim = int(self.net.proj.in_features)

    def encode(self, image) -> torch.Tensor:
        return self.net(image)["shared_feature"]

    def pooled(self, image) -> torch.Tensor:
        return self.net.forward_pooled(image)
