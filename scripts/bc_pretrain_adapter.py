"""VA 预训练(2026-09-11):冻结骨干 + 两层适配器 + 策略头,在 teacher 的行为数据上做动作回归。

    python scripts/bc_pretrain_adapter.py --data ./data_score_k03noy_d_300k --backbone models/vision/score_k03noy_imnet/encoder_init.pt \
        --out models/vision/bc_d_imnet --steps 6000

产物:adapter.pt(18816 -> 512 -> 256 + LayerNorm)、head.pt(258 -> 64 -> 64 -> act_dim,输入 = 适配器输出 + 2 维 goal 常量,
与 PPO 的 actor 同形,可以直接当 actor 初值)、probe.csv(适配器输出对 x/d 的线性/单维可读性、秩)。
骨干只算一次:先把全部帧过冻结骨干得到 18816 维池化特征,缓存在 GPU/CPU,之后只训小网络,几分钟。
目标 = teacher 实际执行的动作(只看行为),原始尺度(头直接当 actor)。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "./src")

from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr

END_OBS = (1.5 / 8.0, 0.0)


def make_adapter(pooled_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(pooled_dim, hidden), nn.Tanh(), nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))


def make_head(in_dim: int, act_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, act_dim))


def ridge_r2(f, y, m, lam=1e-2):
    mu, sd = f[m].mean(0), f[m].std(0) + 1e-9
    a = np.hstack([(f[m] - mu) / sd, np.ones((m.sum(), 1))]); b = np.hstack([(f[~m] - mu) / sd, np.ones(((~m).sum(), 1))])
    w = np.linalg.lstsq(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ y[m], rcond=None)[0]; p = b @ w
    return float(1 - ((y[~m] - p) ** 2).sum() / ((y[~m] - y[~m].mean()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--backbone", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--units", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    tr = fr.load_transitions(a.data, units=a.units)
    mask = fr.episode_split(tr["episode"], seed=0)
    act = np.clip(tr["action"], -1.0, 1.0).astype(np.float32)            # 实际执行的采样动作
    mu_a, sd_a = act[mask].mean(0), act[mask].std(0) + 1e-8   # 只记录,不归一:头要直接当 PPO 的 actor 用,输出必须是原始动作尺度
    target = torch.as_tensor(act, device=dev)
    print(f"[INFO] frames {len(tr['img'])} train {mask.sum()} held {(~mask).sum()} act_dim {act.shape[1]}", flush=True)

    enc = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(dev)
    enc.load_state_dict(torch.load(a.backbone, map_location=dev)); enc.eval()
    frames = fr.load_frames(tr["img"], a.data)
    with torch.no_grad():
        pooled = torch.cat([enc.forward_pooled(frames[i:i + 256].to(dev)).half().cpu() for i in range(0, len(frames), 256)])   # (N, 18816) fp16,留在内存(24 万帧 9 GB)
    del frames
    print(f"[INFO] pooled {tuple(pooled.shape)} cached", flush=True)

    adapter = make_adapter(pooled.shape[1], a.hidden, FREE_SPATIAL_CONFIG["feature_dim"]).to(dev)
    head = make_head(FREE_SPATIAL_CONFIG["feature_dim"] + 2, act.shape[1]).to(dev)
    goal = torch.tensor(END_OBS, device=dev).expand(a.batch, 2)
    opt = torch.optim.Adam(list(adapter.parameters()) + list(head.parameters()), lr=a.lr)
    tr_idx = np.flatnonzero(mask); he_idx = np.flatnonzero(~mask)

    def eval_mse(idx):
        adapter.eval(); head.eval(); tot = 0.0
        with torch.no_grad():
            for i in range(0, len(idx), 1024):
                j = idx[i:i + 1024]; z = adapter(pooled[j].to(dev).float())
                p = head(torch.cat([z, torch.tensor(END_OBS, device=dev).expand(len(j), 2)], 1))
                tot += float(((p - target[j]) ** 2).mean()) * len(j)
        adapter.train(); head.train(); return tot / len(idx)

    for step in range(1, a.steps + 1):
        j = tr_idx[np.random.randint(0, len(tr_idx), a.batch)]
        z = adapter(pooled[j].to(dev).float()); p = head(torch.cat([z, goal], 1))
        loss = ((p - target[j]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"[BC] step {step} train_loss {float(loss):.4f} held_mse {eval_mse(he_idx):.4f}", flush=True)

    torch.save(adapter.state_dict(), a.out / "adapter.pt"); torch.save(head.state_dict(), a.out / "head.pt")
    (a.out / "act_stats.txt").write_text(f"mu {mu_a.tolist()} sd {sd_a.tolist()}\n")

    # 探针:适配器输出里 x/d 可读吗(它没见过 x/d 标签)
    adapter.eval(); rng = np.random.default_rng(0)
    pi = np.concatenate([rng.choice(tr_idx, 3000, replace=False), rng.choice(he_idx, 3000, replace=False)]); pm = np.r_[np.ones(3000, bool), np.zeros(3000, bool)]
    seen = tr["dist"][pi] > 0
    with torch.no_grad():
        z = torch.cat([adapter(pooled[pi[i:i + 1024]].to(dev).float()) for i in range(0, len(pi), 1024)]).cpu().numpy().astype(np.float64)
    x = fr.norm_x(tr["px_x"], tr["dist"])[pi]; d = fr.norm_d(tr["dist"])[pi]
    zs, xs, ds, ms = z[seen], x[seen], d[seen], pm[seen]
    row = {"r2_x": ridge_r2(zs, xs, ms), "r2_d": ridge_r2(zs, ds, ms),
           "best1_x": max(ridge_r2(zs[:, [k]], xs, ms) for k in range(zs.shape[1])), "best1_d": max(ridge_r2(zs[:, [k]], ds, ms) for k in range(zs.shape[1])),
           "eff_rank": fr.eff_rank(zs[~ms]), "held_mse": eval_mse(he_idx)}
    with (a.out / "probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow({k: f"{v:.4f}" for k, v in row.items()})
    print("[PROBE] " + " ".join(f"{k}={v:.4f}" for k, v in row.items()), flush=True)


main()
