"""受控 SPR 对照臂:同架构、同数据、同初值、同预算,只把目标函数换成 SPR。

用户 2026-09-04 要求把 SPR 表征也放进任务三对比。现成的 `stage1_sigma_ablation.pt`
是 z=128、不同架构、训练数据来源未核实,拿去比会把"目标函数"和"架构"混在一起。
这里唯一的变量是目标函数:

  行为塑造(A/ax/...)  冻结读出头 -> 回归 teacher 信号
  SPR(本脚本)         预测未来 latent,自监督,与任务/回报无关

预期分界线(离线已有半边证据):SPR 会保留一切可预测的东西,包括那个与任务无关的
蓝色干扰球;行为塑造的会把它扔掉(ax 臂干扰球 R² 0.0055 vs 随机 encoder 0.855)。

    python scripts/stage2_spr.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "./src")

from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr

LR = 1e-4          # 与 stage2 一致
BATCH = 64         # 与 stage2 一致
PROBE_EVERY = 100  # 与 stage2 一致
PROBE_N = 6000     # 与 stage2 一致
FEATURE_BATCH = 256
TARGET_TAU = 0.99
HIDDEN = 256


class SPRHeads(nn.Module):
    """transition + projector + predictor。SPR 的全部可训练附件。"""

    def __init__(self, z_dim: int, act_dim: int = 3):
        super().__init__()
        self.transition = nn.Sequential(nn.Linear(z_dim + act_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, z_dim))
        self.projector = nn.Sequential(nn.Linear(z_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, z_dim))
        self.predictor = nn.Sequential(nn.Linear(z_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, z_dim))


@torch.no_grad()
def encode(encoder, frames, device):
    encoder.eval()
    out = np.empty((len(frames), FREE_SPATIAL_CONFIG["feature_dim"]), dtype=np.float64)
    for i in range(0, len(frames), FEATURE_BATCH):
        out[i : i + FEATURE_BATCH] = encoder(frames[i : i + FEATURE_BATCH].to(device))["shared_feature"].float().cpu().numpy()
    encoder.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled SPR-objective arm.")
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", type=Path, default=fr.OUT_ROOT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arm_dir = args.out_root / "spr"
    arm_dir.mkdir(parents=True, exist_ok=True)

    tr = fr.load_transitions()
    train_mask = np.load(args.out_root / "train_mask.npy")
    train_idx = np.flatnonzero(train_mask)
    held_idx = np.flatnonzero(~train_mask)

    frames = fr.load_frames(tr["img"])
    next_frames = fr.load_frames(tr["next_img"])
    action = torch.as_tensor(np.clip(tr["action"], -1.0, 1.0)).to(args.device)

    rng = np.random.default_rng(args.seed)
    probe_idx = np.concatenate([rng.choice(train_idx, PROBE_N, replace=False), rng.choice(held_idx, PROBE_N, replace=False)])
    probe_frames = frames[probe_idx]
    probe_train_mask = np.concatenate([np.ones(PROBE_N, bool), np.zeros(PROBE_N, bool)])
    probe_seen = tr["dist"][probe_idx] > fr.LOST_D
    probe_x = fr.norm_x(tr["px_x"], tr["dist"])[probe_idx]
    probe_d = fr.norm_d(tr["dist"])[probe_idx]
    probe_a = np.clip(tr["action"], -1.0, 1.0)[probe_idx]

    # 与六个行为臂共用同一个随机初值。
    online = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(args.device)
    online.load_state_dict(torch.load(args.out_root / "encoder_init.pt"))
    target = copy.deepcopy(online).requires_grad_(False)
    heads = SPRHeads(FREE_SPATIAL_CONFIG["feature_dim"]).to(args.device)
    opt = torch.optim.Adam(list(online.parameters()) + list(heads.parameters()), lr=LR)

    csv_path = arm_dir / "stage2.csv"
    fields = ["update", "loss", "r2_x", "r2_d", "r2_a", "eff_rank", "minutes"]
    file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    start = time.perf_counter()
    loss_run = 0.0
    for update in range(args.updates + 1):
        if update % PROBE_EVERY == 0:
            feats = encode(online, probe_frames, args.device)
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
            print("[spr] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            loss_run = 0.0
            torch.save(online.state_dict(), arm_dir / "encoder_final.pt")

        if update == args.updates:
            break
        idx = train_idx[np.random.randint(0, len(train_idx), BATCH)]
        z = online(frames[idx].to(args.device))["shared_feature"]
        z_hat = heads.transition(torch.cat((z, action[idx]), dim=1))
        pred = heads.predictor(heads.projector(z_hat))
        with torch.no_grad():
            z_next = target(next_frames[idx].to(args.device))["shared_feature"]
            proj_next = heads.projector(z_next)
        loss = -F.cosine_similarity(pred, proj_next.detach(), dim=-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():  # target encoder 走 EMA
            for p_t, p_o in zip(target.parameters(), online.parameters()):
                p_t.mul_(TARGET_TAU).add_(p_o, alpha=1.0 - TARGET_TAU)
        loss_run += float(loss.item())

    file.close()
    print(f"[spr] done -> {arm_dir}")


if __name__ == "__main__":
    main()
