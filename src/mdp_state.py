"""Image to MDP latent state model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MDPStateNet(nn.Module):
    """Learn z=f(img) and one-step latent dynamics."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        z_dim: int = 32,
        action_dim: int = 3,
        belief_dim: int = 128,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        train_layer3: bool = False,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.belief_dim = int(belief_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.backbone_frozen = bool(freeze_backbone)
        self.train_layer3 = bool(train_layer3)

        weights = ResNet18_Weights.DEFAULT if bool(pretrained) else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        if self.backbone_frozen:
            frozen_modules = (self.stem, self.layer1, self.layer2)
            if not self.train_layer3:
                frozen_modules = frozen_modules + (self.layer3,)
            for module in frozen_modules:
                for param in module.parameters():
                    param.requires_grad_(False)

        c = int(feature_dim)
        h = int(hidden_dim)
        self.attention = nn.Sequential(
            nn.Conv2d(c + 2, h, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(h, 1, kernel_size=1),
        )
        self.img_head = nn.Sequential(
            nn.Linear(c * 2 + 3, h),
            nn.ReLU(),
            nn.Linear(h, self.z_dim),
        )

        self.core = nn.GRUCell(self.z_dim + self.action_dim, self.belief_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(self.belief_dim, h),
            nn.ReLU(),
            nn.Linear(h, self.z_dim),
        )
        self.post_head = nn.Sequential(
            nn.Linear(self.belief_dim + self.z_dim, h),
            nn.ReLU(),
            nn.Linear(h, self.z_dim),
        )

        state_dim = self.belief_dim + self.z_dim
        self.reward_head = nn.Sequential(nn.Linear(state_dim + self.action_dim, h), nn.ReLU(), nn.Linear(h, 1))
        self.value_head = nn.Sequential(nn.Linear(state_dim, h), nn.ReLU(), nn.Linear(h, 1))
        self.done_head = nn.Sequential(nn.Linear(state_dim + self.action_dim, h), nn.ReLU(), nn.Linear(h, 1))
        self.probe_head = nn.Sequential(nn.Linear(state_dim, h), nn.ReLU(), nn.Linear(h, 2))
        self.inverse_head = nn.Sequential(
            nn.Linear(self.z_dim * 2, h),
            nn.ReLU(),
            nn.Linear(h, self.action_dim),
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

    def initial(self, batch_size: int, device=None) -> dict[str, torch.Tensor]:
        device = device or next(self.parameters()).device
        return {
            "belief": torch.zeros(batch_size, self.belief_dim, device=device),
            "z": torch.zeros(batch_size, self.z_dim, device=device),
        }

    def encode(self, image) -> dict[str, torch.Tensor]:
        x = self._to_nchw(image)
        x = (x - self.imagenet_mean) / self.imagenet_std

        if self.backbone_frozen:
            with torch.no_grad():
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
            if self.train_layer3:
                feature = self.layer3(x)
            else:
                with torch.no_grad():
                    feature = self.layer3(x)
        else:
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

        img_feature = torch.cat([target_feature, global_feature, heatmap_xy, heatmap_peak], dim=1)
        z_img = self.img_head(img_feature)
        return {
            "z_img": z_img,
            "z_img_unit": F.normalize(z_img, dim=1),
            "img_feature": img_feature,
            "heatmap": heatmap,
            "heatmap_xy": heatmap_xy,
            "heatmap_peak": heatmap_peak,
        }

    def observe(self, image, prev_action: torch.Tensor, prev_state: dict[str, torch.Tensor] | None = None):
        enc = self.encode(image)
        z_img = enc["z_img"]
        if prev_state is None:
            prev_state = self.initial(z_img.shape[0], z_img.device)

        prev_action = prev_action.to(device=z_img.device, dtype=z_img.dtype)
        belief = self.core(torch.cat([prev_state["z"], prev_action], dim=1), prev_state["belief"])
        prior_z = self.prior_head(belief)
        post_z = self.post_head(torch.cat([belief, z_img], dim=1))
        state_feature = torch.cat([belief, post_z], dim=1)

        enc.update(
            {
                "belief": belief,
                "z": post_z,
                "z_unit": F.normalize(post_z, dim=1),
                "prior_z": prior_z,
                "prior_z_unit": F.normalize(prior_z, dim=1),
                "state_feature": state_feature,
                "value": self.value_head(state_feature),
            }
        )
        probe_raw = self.probe_head(state_feature.detach())
        enc["probe"] = torch.cat(
            [
                torch.tanh(probe_raw[:, 0:1]),
                torch.sigmoid(probe_raw[:, 1:2]),
            ],
            dim=1,
        )
        return enc

    def imagine(self, state: dict[str, torch.Tensor], action: torch.Tensor):
        z = state["z"]
        belief = state["belief"]
        action = action.to(device=z.device, dtype=z.dtype)

        next_belief = self.core(torch.cat([z, action], dim=1), belief)
        next_z = self.prior_head(next_belief)
        next_state_feature = torch.cat([next_belief, next_z], dim=1)
        za = torch.cat([state["state_feature"], action], dim=1)
        return {
            "next_belief": next_belief,
            "next_z_pred": next_z,
            "next_z_pred_unit": F.normalize(next_z, dim=1),
            "next_state_feature": next_state_feature,
            "reward_pred": self.reward_head(za),
            "done_logit": self.done_head(za),
            "next_value": self.value_head(next_state_feature),
        }

    def inverse(self, z_img: torch.Tensor, next_z_img: torch.Tensor) -> torch.Tensor:
        return self.inverse_head(torch.cat([z_img, next_z_img], dim=1))

    def forward(
        self,
        image,
        action: torch.Tensor | None = None,
        next_image=None,
        prev_action: torch.Tensor | None = None,
        prev_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if prev_action is None:
            batch = image.shape[0] if isinstance(image, torch.Tensor) and image.ndim == 4 else 1
            prev_action = torch.zeros(batch, self.action_dim, device=next(self.parameters()).device)

        out = self.observe(image, prev_action, prev_state)

        if action is not None:
            out.update(self.imagine(out, action))

        if next_image is not None:
            if action is None:
                next_enc = self.encode(next_image)
                out["next_z"] = next_enc["z_img"]
                out["next_z_unit"] = next_enc["z_img_unit"]
            else:
                next_post = self.observe(next_image, action, out)
                out["next_z"] = next_post["z"]
                out["next_z_unit"] = next_post["z_unit"]

        return out

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

        is_uint8 = tensor.dtype == torch.uint8
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float()
        if is_uint8:
            tensor = tensor / 255.0
        return tensor
