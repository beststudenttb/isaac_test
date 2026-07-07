"""SPR-style visual state encoder."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class SPRStateNet(nn.Module):
    """ResNet18 + FPN + coord + conv encoder for SPR pretraining."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        z_dim: int = 128,
        action_dim: int = 3,
        fpn_dim: int = 128,
        hidden_dim: int = 256,
        pool_size: int = 7,
        pretrained: bool = True,
        target_tau: float = 0.99,
        freeze_stem_layers: bool = False,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.fpn_dim = int(fpn_dim)
        self.hidden_dim = int(hidden_dim)
        self.pool_size = int(pool_size)
        self.target_tau = float(target_tau)
        self.freeze_stem_layers = bool(freeze_stem_layers)

        self.encoder = SPRFPNEncoder(
            input_shape=self.input_shape,
            z_dim=self.z_dim,
            fpn_dim=self.fpn_dim,
            hidden_dim=self.hidden_dim,
            pool_size=self.pool_size,
            pretrained=pretrained,
            freeze_stem_layers=self.freeze_stem_layers,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)

        self.transition = nn.Sequential(
            nn.Linear(self.z_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.z_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.z_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.z_dim),
        )

    def encode(self, image) -> torch.Tensor:
        if self.freeze_stem_layers:
            self.encoder.freeze_modules()
        return self.encoder(image)

    @torch.no_grad()
    def encode_target(self, image) -> torch.Tensor:
        return self.target_encoder(image)

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = action.to(device=z.device, dtype=z.dtype)
        return self.transition(torch.cat((z, action), dim=1))

    def spr_pred(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.projector(z))

    @torch.no_grad()
    def spr_target(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z).detach()

    @torch.no_grad()
    def update_target(self) -> None:
        tau = self.target_tau
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)
        for online, target in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            target.copy_(online)


class SPRFPNEncoder(nn.Module):
    """ResNet18 feature pyramid encoder without explicit heatmap."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        z_dim: int,
        fpn_dim: int,
        hidden_dim: int,
        pool_size: int,
        pretrained: bool,
        freeze_stem_layers: bool,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.z_dim = int(z_dim)
        self.fpn_dim = int(fpn_dim)
        self.hidden_dim = int(hidden_dim)
        self.pool_size = int(pool_size)
        self.freeze_stem_layers = bool(freeze_stem_layers)

        weights = ResNet18_Weights.DEFAULT if bool(pretrained) else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        if self.freeze_stem_layers:
            self.freeze_modules()

        c = self.fpn_dim
        self.lat1 = nn.Conv2d(64, c, kernel_size=1)
        self.lat2 = nn.Conv2d(128, c, kernel_size=1)
        self.lat3 = nn.Conv2d(256, c, kernel_size=1)
        self.smooth1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(c * 3 + 2, c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(c * self.pool_size * self.pool_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.z_dim),
        )

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

    def forward(self, image) -> torch.Tensor:
        x = self._to_nchw(image)
        x = (x - self.imagenet_mean) / self.imagenet_std

        if self.freeze_stem_layers:
            with torch.no_grad():
                x = self.stem(x)
                c1 = self.layer1(x)
                c2 = self.layer2(c1)
            x = x.detach()
            c1 = c1.detach()
            c2 = c2.detach()
        else:
            x = self.stem(x)
            c1 = self.layer1(x)
            c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        p1 = self.smooth1(p1)
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)

        p2 = F.interpolate(p2, size=p1.shape[-2:], mode="nearest")
        p3 = F.interpolate(p3, size=p1.shape[-2:], mode="nearest")
        coord = self._coord_grid(p1)
        feature = self.conv(torch.cat((p1, p2, p3, coord), dim=1))
        feature = F.adaptive_avg_pool2d(feature, (self.pool_size, self.pool_size))
        z = self.head(feature.flatten(1))
        return z

    def freeze_modules(self) -> None:
        for module in (self.stem, self.layer1, self.layer2):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def _coord_grid(self, feature: torch.Tensor) -> torch.Tensor:
        height, width = feature.shape[-2:]
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=feature.device)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=feature.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack((x_grid, y_grid), dim=0).unsqueeze(0).expand(feature.shape[0], -1, -1, -1)

    def _to_nchw(self, image) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(image, torch.Tensor):
            tensor = image.to(device)
        else:
            tensor = torch.as_tensor(image, device=device)

        is_uint8 = tensor.dtype == torch.uint8
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float()
        if is_uint8:
            tensor = tensor / 255.0
        return tensor
