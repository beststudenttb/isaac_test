"""Compact visual encoder for Isaac ball tracking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18


class BallVisionNet(nn.Module):
    """Image -> reserve_feature, x, distance.

    The model keeps the part we actually need for RL: a compact 128-d state.
    It uses a ResNet18 feature map with a small attention head instead of
    carrying over the old Webots FPN class structure.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        reserve_dim: int = 128,
        attention_channels: int = 64,
        pretrained: bool = True,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.reserve_dim = int(reserve_dim)

        weights = ResNet18_Weights.DEFAULT if bool(pretrained) else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        channels = 256
        attn_channels = int(attention_channels)
        self.attention = nn.Sequential(
            nn.Conv2d(channels + 2, attn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(attn_channels, 1, kernel_size=1),
        )
        raw_dim = channels * 2 + 3
        self.reserve = nn.Sequential(
            nn.Linear(raw_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.reserve_dim),
            nn.ReLU(),
        )
        self.x_head = nn.Sequential(nn.Linear(self.reserve_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.distance_head = nn.Sequential(nn.Linear(self.reserve_dim, 64), nn.ReLU(), nn.Linear(64, 1))

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

    def forward(self, image) -> dict[str, torch.Tensor]:
        x = self._to_nchw(image)
        x = (x - self.imagenet_mean) / self.imagenet_std

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feature = self.layer3(x)

        coord = self._coord_grid(feature)
        score = self.attention(torch.cat([feature, coord], dim=1))
        heatmap = torch.softmax(score.flatten(1), dim=1).view_as(score)

        target_feature = (feature * heatmap).sum(dim=(2, 3))
        global_feature = feature.mean(dim=(2, 3))
        heatmap_xy = (heatmap * coord).sum(dim=(2, 3))
        heatmap_peak = heatmap.flatten(1).max(dim=1, keepdim=True).values

        raw_feature = torch.cat([target_feature, global_feature, heatmap_xy, heatmap_peak], dim=1)
        reserve_feature = self.reserve(raw_feature)

        return {
            "reserve_feature": reserve_feature,
            "x_pred": self.x_head(reserve_feature),
            "distance_pred": self.distance_head(reserve_feature),
            "heatmap": heatmap,
            "heatmap_xy": heatmap_xy,
            "heatmap_peak": heatmap_peak,
        }

    def _coord_grid(self, feature: torch.Tensor) -> torch.Tensor:
        height, width = feature.shape[-2:]
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=feature.device)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=feature.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).expand(feature.shape[0], -1, -1, -1)

    def _to_nchw(self, image) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(image, torch.Tensor):
            tensor = image.to(device)
        else:
            tensor = torch.as_tensor(image, device=device)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float()
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        return tensor


class MobileBallNet(nn.Module):
    """MobileNet front layers + the same attention/state heads."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        reserve_dim: int = 128,
        attention_channels: int = 32,
        pretrained: bool = True,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.reserve_dim = int(reserve_dim)

        weights = MobileNet_V3_Small_Weights.DEFAULT if bool(pretrained) else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = nn.Sequential(*list(backbone.features[:4]))

        channels = 24
        attn_channels = int(attention_channels)
        self.attention = nn.Sequential(
            nn.Conv2d(channels + 2, attn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(attn_channels, 1, kernel_size=1),
        )
        raw_dim = channels * 2 + 3
        self.reserve = nn.Sequential(
            nn.Linear(raw_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.reserve_dim),
            nn.ReLU(),
        )
        self.x_head = nn.Sequential(nn.Linear(self.reserve_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.distance_head = nn.Sequential(nn.Linear(self.reserve_dim, 64), nn.ReLU(), nn.Linear(64, 1))

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

    def forward(self, image) -> dict[str, torch.Tensor]:
        x = self._to_nchw(image)
        x = (x - self.imagenet_mean) / self.imagenet_std
        feature = self.features(x)

        coord = self._coord_grid(feature)
        score = self.attention(torch.cat([feature, coord], dim=1))
        heatmap = torch.softmax(score.flatten(1), dim=1).view_as(score)

        target_feature = (feature * heatmap).sum(dim=(2, 3))
        global_feature = feature.mean(dim=(2, 3))
        heatmap_xy = (heatmap * coord).sum(dim=(2, 3))
        heatmap_peak = heatmap.flatten(1).max(dim=1, keepdim=True).values

        raw_feature = torch.cat([target_feature, global_feature, heatmap_xy, heatmap_peak], dim=1)
        reserve_feature = self.reserve(raw_feature)

        return {
            "reserve_feature": reserve_feature,
            "x_pred": self.x_head(reserve_feature),
            "distance_pred": self.distance_head(reserve_feature),
            "heatmap": heatmap,
            "heatmap_xy": heatmap_xy,
            "heatmap_peak": heatmap_peak,
        }

    def _coord_grid(self, feature: torch.Tensor) -> torch.Tensor:
        height, width = feature.shape[-2:]
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=feature.device)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=feature.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).expand(feature.shape[0], -1, -1, -1)

    def _to_nchw(self, image) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(image, torch.Tensor):
            tensor = image.to(device)
        else:
            tensor = torch.as_tensor(image, device=device)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float()
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        return tensor


class OldFPNBallNet(nn.Module):
    """Old FPN reserve structure with a selectable backbone."""

    def __init__(
        self,
        backbone: str = "resnet",
        input_shape: tuple[int, int, int] = (224, 224, 3),
        fpn_channels: int = 128,
        reserve_dim: int = 128,
        bin_count: int = 8,
        pretrained: bool = True,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.reserve_dim = int(reserve_dim)
        self.bin_count = int(bin_count)
        self.backbone = str(backbone)

        if self.backbone == "resnet":
            weights = ResNet18_Weights.DEFAULT if bool(pretrained) else None
            base = resnet18(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            stage_channels = (64, 128, 256)
        elif self.backbone == "mobile":
            weights = MobileNet_V3_Small_Weights.DEFAULT if bool(pretrained) else None
            features = list(mobilenet_v3_small(weights=weights).features)
            self.stem = nn.Identity()
            self.layer1 = nn.Sequential(features[0], features[1])
            self.layer2 = nn.Sequential(features[2], features[3])
            self.layer3 = nn.Sequential(features[4], features[5], features[6], features[7], features[8])
            stage_channels = (16, 24, 48)
        else:
            raise ValueError(f"unknown backbone: {backbone}")

        c = int(fpn_channels)
        self.lat2 = nn.Conv2d(stage_channels[0], c, kernel_size=1)
        self.lat3 = nn.Conv2d(stage_channels[1], c, kernel_size=1)
        self.lat4 = nn.Conv2d(stage_channels[2], c, kernel_size=1)
        self.smooth2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c + 2, c // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c // 2, 1, kernel_size=1),
        )

        shared_dim = c * 4 + 4
        self.reserve_layer = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.reserve_dim),
            nn.ReLU(),
        )
        self.x_head = nn.Sequential(nn.Linear(self.reserve_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.distance_head = nn.Sequential(nn.Linear(self.reserve_dim, 128), nn.ReLU(), nn.Linear(128, 1))

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

        coords = self._coord_grid(p2)
        heatmap_logits = self.heatmap_head(torch.cat([p2, coords], dim=1))
        heatmap = torch.softmax(heatmap_logits.flatten(1), dim=1).view_as(heatmap_logits)
        heatmap_x = (heatmap * coords[:, 0:1]).sum(dim=(2, 3))
        y_pred = (heatmap * coords[:, 1:2]).sum(dim=(2, 3))
        heatmap_flat = heatmap.flatten(1)
        heatmap_max = heatmap_flat.max(dim=1, keepdim=True).values
        heatmap_entropy = -(heatmap_flat * torch.log(heatmap_flat + 1e-8)).sum(dim=1, keepdim=True)
        target_p2 = self._weighted_pool(p2, heatmap)
        target_p3 = self._weighted_pool(p3, heatmap)
        target_p4 = self._weighted_pool(p4, heatmap)
        global_feat = p4.mean(dim=(2, 3))
        shared_feature = torch.cat(
            [target_p2, target_p3, target_p4, global_feat, heatmap_x, y_pred, heatmap_max, heatmap_entropy],
            dim=1,
        )
        reserve_feature = self.reserve_layer(shared_feature)
        x_pred = self.x_head(reserve_feature)
        distance_pred = self.distance_head(reserve_feature)

        return {
            "shared_feature": shared_feature,
            "reserve_feature": reserve_feature,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "distance_pred": distance_pred,
            "heatmap": heatmap,
            "heatmap_max": heatmap_max,
            "heatmap_entropy": heatmap_entropy,
        }

    def _weighted_pool(self, feature: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        weight = F.interpolate(heatmap, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        weight = weight / weight.sum(dim=(2, 3), keepdim=True)
        return (feature * weight).sum(dim=(2, 3))

    def _coord_grid(self, feature: torch.Tensor) -> torch.Tensor:
        height, width = feature.shape[-2:]
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=feature.device)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=feature.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).expand(feature.shape[0], -1, -1, -1)

    def _to_nchw(self, image) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(image, torch.Tensor):
            tensor = image.to(device)
        else:
            tensor = torch.as_tensor(image, device=device)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float()
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        return tensor


class OldBallVisionNet(OldFPNBallNet):
    def __init__(self, **kwargs):
        super().__init__(backbone="resnet", **kwargs)


class OldMobileBallNet(OldFPNBallNet):
    def __init__(self, **kwargs):
        super().__init__(backbone="mobile", **kwargs)
