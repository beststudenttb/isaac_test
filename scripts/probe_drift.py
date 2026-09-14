"""表征漂移对策略的冲击,以及线性重对齐能不能吸收掉。

用户 2026-09-04:"很小的表征漂移,对策略的影响,我觉得应该是很大的"。

策略 = 冻结头 ∘ encoder(stage1 的 head.pt 就是 特征 -> 动作 的读出)。
两种扰动:权重乘性噪声、继续训练若干步。都不需要 Isaac。

    python scripts/probe_drift.py --arm A

关键是第三列 a_realign:先拟合线性映射 W: z_new -> z_old 再过头。
吸收得掉 => 漂移只是换基,策略可救;吸收不掉 => 信息真的变了。
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "./src")

from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import ARM_DIMS, make_head

PROBE_N = 4000
FEATURE_BATCH = 256
NOISE_LEVELS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)
STEP_COUNTS = (1, 3, 10, 30, 100)
STEP_LR = 1e-4
BATCH = 64


@torch.no_grad()
def features(encoder, frames, device) -> np.ndarray:
    encoder.eval()
    out = np.empty((len(frames), FREE_SPATIAL_CONFIG["feature_dim"]), dtype=np.float64)
    for i in range(0, len(frames), FEATURE_BATCH):
        out[i : i + FEATURE_BATCH] = encoder(frames[i : i + FEATURE_BATCH].to(device))["shared_feature"].float().cpu().numpy()
    return out


def act(head, z: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        return head(torch.as_tensor(z, dtype=torch.float32, device=device)).cpu().numpy()


def linear_map(src: np.ndarray, dst: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    a = np.hstack([src, np.ones((len(src), 1))])
    return np.linalg.solve(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ dst)


def apply_map(w: np.ndarray, src: np.ndarray) -> np.ndarray:
    return np.hstack([src, np.ones((len(src), 1))]) @ w


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation drift vs policy impact.")
    parser.add_argument("--arm", default="A", choices=tuple(ARM_DIMS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = fr.OUT_ROOT
    arm_dir = root / args.arm
    tr = fr.load_transitions()
    train_mask = np.load(root / "train_mask.npy")
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(np.flatnonzero(train_mask), PROBE_N, replace=False)
    eval_idx = rng.choice(np.flatnonzero(~train_mask), PROBE_N, replace=False)
    idx = np.concatenate([fit_idx, eval_idx])
    fit_m = np.concatenate([np.ones(PROBE_N, bool), np.zeros(PROBE_N, bool)])

    frames = fr.load_frames(tr["img"][idx])
    target = torch.as_tensor(np.load(arm_dir / "target.npy")).to(args.device)
    train_idx = np.flatnonzero(train_mask)
    train_frames = fr.load_frames(tr["img"][train_idx])

    base = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(args.device)
    base.load_state_dict(torch.load(arm_dir / "encoder_final.pt", map_location=args.device))
    head = make_head(FREE_SPATIAL_CONFIG["feature_dim"], ARM_DIMS[args.arm]).to(args.device)
    head.load_state_dict(torch.load(arm_dir / "head.pt", map_location=args.device))
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    z0 = features(base, frames, args.device)
    a0 = act(head, z0, args.device)
    a_scale = a0[~fit_m].std(0).mean()
    z_scale = np.linalg.norm(z0[~fit_m], axis=1).mean()
    print(f"[INFO] arm={args.arm}  ‖z‖={z_scale:.3f}  std(a)={a_scale:.4f}")

    rows = []

    def record(kind: str, amount, encoder) -> None:
        z1 = features(encoder, frames, args.device)
        a1 = act(head, z1, args.device)
        e = ~fit_m
        drift = float(np.linalg.norm(z1[e] - z0[e], axis=1).mean() / z_scale)
        impact = float(np.linalg.norm(a1[e] - a0[e], axis=1).mean() / a_scale)
        w = linear_map(z1[fit_m], z0[fit_m])
        a_re = act(head, apply_map(w, z1), args.device)
        impact_re = float(np.linalg.norm(a_re[e] - a0[e], axis=1).mean() / a_scale)
        rows.append(
            {
                "kind": kind,
                "amount": amount,
                "drift_z": round(drift, 5),
                "impact_a": round(impact, 4),
                "impact_a_realign": round(impact_re, 4),
                "amplification": round(impact / max(drift, 1e-9), 1),
            }
        )
        print("  " + " ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)

    print("[INFO] 权重乘性噪声")
    for eps in NOISE_LEVELS:
        pert = copy.deepcopy(base)
        gen = torch.Generator(device=args.device).manual_seed(0)
        with torch.no_grad():
            for p in pert.parameters():
                p.mul_(1.0 + eps * torch.randn(p.shape, generator=gen, device=args.device))
        record("noise", eps, pert)

    print("[INFO] 继续训练(同一目标,lr=1e-4)")
    cont = copy.deepcopy(base)
    opt = torch.optim.Adam(cont.parameters(), lr=STEP_LR)
    cont.train()
    done = 0
    for n in STEP_COUNTS:
        while done < n:
            b = train_idx[np.random.randint(0, len(train_idx), BATCH)]
            loss = ((head(cont(train_frames[np.searchsorted(train_idx, b)].to(args.device))["shared_feature"]) - target[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
        record("steps", n, cont)
        cont.train()

    out = arm_dir / "drift.csv"
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
