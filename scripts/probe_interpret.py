"""解释性探针:读出 x/d 需要多复杂的读出器。

用户 2026-09-08:"越简单的拟合越能说明表征的解释性更好"。

四档容量,从最苛刻到最宽松:
  best1   256 维里**单个最好的维度**   -> 若 R² 高,说明表征里有一根轴"就是" x(或 d)
  best2   最好的两维组合
  lin     全部 256 维线性
  mlp     [64,64] MLP(非线性上界)

    python scripts/probe_interpret.py
"""

from __future__ import annotations

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
from stage1_helpers import make_head

PROBE_N = 5000
DEVICE = "cuda"
ARMS = ("init", "A", "V", "R", "sup")


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ridge_r2(f, y, m, lam=1e-2):
    ftr, fte, ytr, yte = f[m], f[~m], y[m], y[~m]
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-9
    a = np.hstack([(ftr - mu) / sd, np.ones((len(ftr), 1))])
    b = np.hstack([(fte - mu) / sd, np.ones((len(fte), 1))])
    w = np.linalg.lstsq(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ ytr, rcond=None)[0]
    return r2(b @ w, yte)


def mlp_r2(f, y, m, steps=8000):
    mu, sd = f[m].mean(0), f[m].std(0) + 1e-9
    f = (f - mu) / sd
    ym, ys = y[m].mean(), y[m].std()
    X = torch.as_tensor(f, dtype=torch.float32, device=DEVICE)
    Y = torch.as_tensor((y - ym) / ys, dtype=torch.float32, device=DEVICE)[:, None]
    torch.manual_seed(0)
    net = make_head(f.shape[1], 1).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xtr, Ytr = X[m], Y[m]
    for _ in range(steps):
        i = torch.randint(0, len(Xtr), (512,), device=DEVICE)
        loss = ((net(Xtr[i]) - Ytr[i]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        p = net(X[~m]).squeeze(-1).cpu().numpy() * ys + ym
    return r2(p, y[~m])


@torch.no_grad()
def encode(path, frames):
    enc = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEVICE)
    enc.load_state_dict(torch.load(path, map_location=DEVICE)); enc.eval()
    return np.concatenate([enc(frames[i:i+256].to(DEVICE))["shared_feature"].float().cpu().numpy()
                           for i in range(0, len(frames), 256)]).astype(np.float64)


def main():
    root = fr.OUT_ROOT
    tr = fr.load_transitions()
    tm = np.load(root / "train_mask.npy")
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.flatnonzero(tm), PROBE_N, replace=False),
                          rng.choice(np.flatnonzero(~tm), PROBE_N, replace=False)])
    m = np.concatenate([np.ones(PROBE_N, bool), np.zeros(PROBE_N, bool)])
    seen = tr["dist"][idx] > fr.LOST_D
    targets = {"x": fr.norm_x(tr["px_x"], tr["dist"])[idx], "d": fr.norm_d(tr["dist"])[idx]}
    frames = fr.load_frames(tr["img"][idx])

    rows = []
    for arm in ARMS:
        p = root / "encoder_init.pt" if arm == "init" else root / arm / "encoder_final.pt"
        if not p.exists():
            continue
        f = encode(p, frames)
        for name, y in targets.items():
            fs, ys, ms = f[seen], y[seen], m[seen]
            # best1:逐维单变量线性回归,取最好的
            per = [ridge_r2(fs[:, [j]], ys, ms) for j in range(fs.shape[1])]
            order = np.argsort(per)[::-1]
            b1 = per[order[0]]
            b2 = ridge_r2(fs[:, order[:2]], ys, ms)
            lin = ridge_r2(fs, ys, ms)
            ml = mlp_r2(fs, ys, ms)
            rows.append({"arm": arm, "target": name, "best1": round(b1, 4), "best2": round(b2, 4),
                         "lin256": round(lin, 4), "mlp": round(ml, 4), "best_dim": int(order[0])})
            print(f"  {arm:5s} {name}: best1={b1:.4f} best2={b2:.4f} lin={lin:.4f} mlp={ml:.4f}", flush=True)

    with (root / "interpret.csv").open("w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {root/'interpret.csv'}")


main()
