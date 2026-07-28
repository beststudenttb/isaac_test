"""spr_z 模式的冻结 SPR encoder 加载(纯 torch,无 Isaac)。

spr_z 线用一个预训练好的 SPRStateNet 把图像编码成 z 喂给 DrQ;encoder 全程冻结
(eval + requires_grad_(False),07-10 实证 train() 会让 BN running stats 漂移)。
spr_coadapt 线不用这个——它在 agent 内部从头联合训练自己的 SPR。
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.cv_extractor.spr_state import SPRStateNet


def load_spr(ckpt_path: Path, device: torch.device) -> SPRStateNet:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"SPR checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    spr = SPRStateNet(**ckpt["spr_cfg"]).to(device)
    spr.load_state_dict({k[len("spr."):]: v for k, v in ckpt["model"].items() if k.startswith("spr.")})
    spr.eval()
    spr.requires_grad_(False)
    return spr
