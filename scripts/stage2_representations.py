"""任务二第二步:头冻结,只训 encoder。

交接文档 2026-09-03 §3。六个臂各得到一个表征,唯一变量是冻结头承载的信号。

    python scripts/stage2_representations.py --arm A

encoder 从 stage1 存下的同一个随机初值出发;头从 stage1 加载并冻结。
训练只用 train 侧 episode,探针的 held 侧 episode 对 encoder 也是没见过的。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "./src")

from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import ARM_DIMS, make_head

LR = 1e-4
BATCH = 64
PROBE_EVERY = 100
PROBE_N = 6000
FEATURE_BATCH = 256


@torch.no_grad()
def encode(encoder, frames: torch.Tensor, device: str) -> np.ndarray:
    encoder.eval()
    out = np.empty((len(frames), FREE_SPATIAL_CONFIG["feature_dim"]), dtype=np.float64)
    for i in range(0, len(frames), FEATURE_BATCH):
        batch = frames[i : i + FEATURE_BATCH].to(device, non_blocking=True)
        out[i : i + FEATURE_BATCH] = encoder(batch)["shared_feature"].float().cpu().numpy()
    encoder.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the encoder against a frozen readout head.")
    parser.add_argument("--arm", required=True, choices=tuple(ARM_DIMS))
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", type=Path, default=fr.OUT_ROOT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arm_dir = args.out_root / args.arm

    tr = fr.load_transitions()
    train_mask = np.load(args.out_root / "train_mask.npy")
    target = torch.as_tensor(np.load(arm_dir / "target.npy"))

    frames = fr.load_frames(tr["img"])
    train_idx = np.flatnonzero(train_mask)
    held_idx = np.flatnonzero(~train_mask)

    # 探针集固定,跨臂跨步都用同一批帧。ridge 在 train 侧拟合、held 侧读数。
    rng = np.random.default_rng(args.seed)
    probe_tr = rng.choice(train_idx, PROBE_N, replace=False)
    probe_he = rng.choice(held_idx, PROBE_N, replace=False)
    probe_idx = np.concatenate([probe_tr, probe_he])
    probe_frames = frames[probe_idx]
    probe_train_mask = np.concatenate([np.ones(PROBE_N, bool), np.zeros(PROBE_N, bool)])
    probe_seen = tr["dist"][probe_idx] > fr.LOST_D
    probe_x = fr.norm_x(tr["px_x"], tr["dist"])[probe_idx]
    probe_d = fr.norm_d(tr["dist"])[probe_idx]
    probe_a = np.clip(tr["action"], -1.0, 1.0)[probe_idx]

    encoder = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(args.device)
    encoder.load_state_dict(torch.load(args.out_root / "encoder_init.pt"))
    head = make_head(FREE_SPATIAL_CONFIG["feature_dim"], ARM_DIMS[args.arm]).to(args.device)
    head.load_state_dict(torch.load(arm_dir / "head.pt"))
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(encoder.parameters(), lr=LR)
    target = target.to(args.device)

    csv_path = arm_dir / "stage2.csv"
    fields = ["update", "loss", "r2_x", "r2_d", "r2_a", "eff_rank", "minutes"]
    file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    start = time.perf_counter()
    loss_run = 0.0
    for update in range(args.updates + 1):
        if update % PROBE_EVERY == 0:
            feats = encode(encoder, probe_frames, args.device)
            row = {
                "update": update,
                "loss": f"{loss_run / max(1, min(update, PROBE_EVERY)):.6f}" if update else "nan",
                "r2_x": f"{fr.r2_split(feats[probe_seen], probe_x[probe_seen], probe_train_mask[probe_seen]):.6f}",
                "r2_d": f"{fr.r2_split(feats[probe_seen], probe_d[probe_seen], probe_train_mask[probe_seen]):.6f}",
                "r2_a": f"{fr.r2_split(feats, probe_a, probe_train_mask):.6f}",
                "eff_rank": f"{fr.eff_rank(feats[~probe_train_mask]):.4f}",
                "minutes": f"{(time.perf_counter() - start) / 60:.2f}",
            }
            writer.writerow(row)
            file.flush()
            print(f"[{args.arm}] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            loss_run = 0.0
            torch.save(encoder.state_dict(), arm_dir / "encoder_final.pt")

        if update == args.updates:
            break
        idx = train_idx[np.random.randint(0, len(train_idx), BATCH)]
        batch = frames[idx].to(args.device, non_blocking=True)
        loss = ((head(encoder(batch)["shared_feature"]) - target[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_run += float(loss.item())

    file.close()
    print(f"[{args.arm}] done -> {arm_dir}")


if __name__ == "__main__":
    main()
