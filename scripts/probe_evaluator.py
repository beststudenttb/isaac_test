"""策略无关的表征筛子。

用户 2026-09-04:"我需要一个通用的评估器"。通用的不存在,这里做的是**必要条件**——
每个量都能证伪(烂 => 这表征肯定不行),不能证真。全部同一份 50k 数据、同一架构,跨表征可比。

    python scripts/probe_evaluator.py

四组量:
  饿探针-限维    只给前 k 个主成分。随机特征把信息摊薄在几十个方向上,砍到 k=2 就掉;
                 组织好的表征信息集中,砍完还在。这是"有没有因子"的直接检验。
  饿探针-限样本  只给 n 个训练样本。随机特征要很多样本才拟合得动。
  行为距离对齐   corr(||z_i - z_j||, ||a*_i - a*_j||)。bisimulation 的可算版本:
                 该分的分了吗、该并的并了吗。
  critic 友好度  折扣回报的可预测性(MC),以及 TD(0) 不动点的质量。
                 仓库 2026-07-16 定位到真病根是信用分配,这两个量量的就是它。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "./src")

from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import ARMS

PROBE_N = 6000
FEATURE_BATCH = 256
DEVICE = "cuda"
GAMMA = 0.99
DIM_BUDGETS = (2, 4, 8, 16)
SAMPLE_BUDGETS = (50, 200, 1000)
PAIR_N = 200000


def returns_to_go(tr: dict[str, np.ndarray]) -> np.ndarray:
    """每条 episode 内的折扣回报。CSV 按 index 递增,同一 episode 内 t 也递增。"""
    reward = tr["reward"]
    out = np.zeros(len(reward), dtype=np.float32)
    order = np.lexsort((tr["t"], tr["episode"]))
    running, prev_ep = 0.0, -1
    for i in order[::-1]:
        ep = tr["episode"][i]
        if ep != prev_ep:
            running, prev_ep = 0.0, ep
        running = reward[i] + GAMMA * running
        out[i] = running
    return out


def ridge(a: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    a = np.hstack([a, np.ones((len(a), 1))])
    return np.linalg.lstsq(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ y, rcond=None)[0]


def r2_budget(f_tr, y_tr, f_te, y_te, lam: float = 1e-2) -> float:
    mu, sd = f_tr.mean(0), f_tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    w = ridge((f_tr - mu) / sd, y_tr, lam)
    pred = np.hstack([(f_te - mu) / sd, np.ones((len(f_te), 1))]) @ w
    return float(1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean(0)) ** 2).sum())


@torch.no_grad()
def encode(path: Path, frames: torch.Tensor) -> np.ndarray:
    enc = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEVICE)
    enc.load_state_dict(torch.load(path, map_location=DEVICE))
    enc.eval()
    out = np.empty((len(frames), FREE_SPATIAL_CONFIG["feature_dim"]), dtype=np.float64)
    for i in range(0, len(frames), FEATURE_BATCH):
        out[i : i + FEATURE_BATCH] = enc(frames[i : i + FEATURE_BATCH].to(DEVICE))["shared_feature"].float().cpu().numpy()
    return out


def main() -> None:
    root = fr.OUT_ROOT
    tr = fr.load_transitions()
    ret = returns_to_go(tr)

    # teacher 的确定性策略 mean,作为"最优动作"用于行为距离。
    from stable_baselines3 import PPO

    model = PPO.load(str(fr.TEACHER_PATH), device="cpu")
    with torch.no_grad():
        a_star = np.clip(
            model.policy.get_distribution(torch.as_tensor(fr.teacher_obs(tr["px_x"], tr["dist"]))).distribution.mean.numpy(),
            -1.0,
            1.0,
        )

    train_mask = np.load(root / "train_mask.npy")
    rng = np.random.default_rng(0)
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(train_mask), PROBE_N, replace=False),
            rng.choice(np.flatnonzero(~train_mask), PROBE_N, replace=False),
        ]
    )
    split = np.concatenate([np.ones(PROBE_N, bool), np.zeros(PROBE_N, bool)])

    frames = fr.load_frames(tr["img"][idx])
    next_frames = fr.load_frames(tr["next_img"][idx])
    y_a = a_star[idx]
    y_ret = ret[idx][:, None]
    rew = tr["reward"][idx][:, None]
    not_done = (1.0 - tr["done"][idx])[:, None]

    pairs_i = rng.integers(0, PROBE_N, PAIR_N)
    pairs_j = rng.integers(0, PROBE_N, PAIR_N)
    da = np.linalg.norm(y_a[PROBE_N:][pairs_i] - y_a[PROBE_N:][pairs_j], axis=1)

    rows = []
    targets = [("init", root / "encoder_init.pt")] + [(a, root / a / "encoder_final.pt") for a in ARMS]
    for name, path in targets:
        if not path.exists():
            continue
        f = encode(path, frames)
        f_next = encode(path, next_frames)
        tr_m, te_m = split, ~split
        row = {"arm": name}

        # 饿探针-限维:在训练侧做 PCA,只留前 k 个主成分。
        z = (f - f[tr_m].mean(0)) / (f[tr_m].std(0) + 1e-9)
        _, _, vt = np.linalg.svd(z[tr_m], full_matrices=False)
        for k in DIM_BUDGETS:
            p = z @ vt[:k].T
            row[f"a@d{k}"] = round(r2_budget(p[tr_m], y_a[tr_m], p[te_m], y_a[te_m]), 4)

        # 饿探针-限样本:全维度,但只给 n 个训练样本。
        tr_idx = np.flatnonzero(tr_m)
        for n in SAMPLE_BUDGETS:
            sub = rng.choice(tr_idx, n, replace=False)
            row[f"a@n{n}"] = round(r2_budget(f[sub], y_a[sub], f[te_m], y_a[te_m], lam=1.0), 4)

        # 行为距离对齐(只在 held 侧)。
        fh = f[te_m]
        dz = np.linalg.norm(fh[pairs_i] - fh[pairs_j], axis=1)
        row["align"] = round(float(np.corrcoef(dz, da)[0, 1]), 4)

        # critic 友好度:MC 折扣回报可预测性,以及 TD(0) 不动点残差。
        row["mc_ret"] = round(r2_budget(f[tr_m], y_ret[tr_m], f[te_m], y_ret[te_m]), 4)
        # TD(0):在固定特征上迭代 V <- r + gamma * V(s') ,线性 V。
        w = np.zeros((f.shape[1] + 1, 1))
        mu, sd = f[tr_m].mean(0), f[tr_m].std(0) + 1e-9
        ftr = np.hstack([(f[tr_m] - mu) / sd, np.ones((tr_m.sum(), 1))])
        fnx = np.hstack([(f_next[tr_m] - mu) / sd, np.ones((tr_m.sum(), 1))])
        fte = np.hstack([(f[te_m] - mu) / sd, np.ones((te_m.sum(), 1))])
        gram = ftr.T @ ftr + 1e-2 * len(ftr) * np.eye(ftr.shape[1])
        for _ in range(200):
            target = rew[tr_m] + GAMMA * not_done[tr_m] * (fnx @ w)
            w = np.linalg.solve(gram, ftr.T @ target)
        pred = fte @ w
        row["td0_ret"] = round(float(1 - ((y_ret[te_m] - pred) ** 2).sum() / ((y_ret[te_m] - y_ret[te_m].mean(0)) ** 2).sum()), 4)
        rows.append(row)
        print("  " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    fields = list(rows[0])
    head = f"{'arm':6s}" + "".join(f"{k:>9s}" for k in fields[1:])
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['arm']:6s}" + "".join(f"{r[k]:9.4f}" for k in fields[1:]))
    with (root / "evaluator.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {root / 'evaluator.csv'}")


if __name__ == "__main__":
    main()
