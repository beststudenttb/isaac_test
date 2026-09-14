"""汇总任务二:六臂探针 + 交接文档 §2.2 的人工特权对照。

    python scripts/summarise_free.py

探针集与 stage2 完全同一批帧(同 seed、同选法),所以数字可直接横比。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "./src")

import free_repr as fr
from stage1_helpers import ARMS

PROBE_N = 6000  # 与 stage2_representations.PROBE_N 一致。


def probe_indices(train_idx, held_idx, seed: int = 0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(train_idx, PROBE_N, replace=False), rng.choice(held_idx, PROBE_N, replace=False)])


def main() -> None:
    root = fr.OUT_ROOT
    tr = fr.load_transitions()
    train_mask = np.load(root / "train_mask.npy")
    idx = probe_indices(np.flatnonzero(train_mask), np.flatnonzero(~train_mask))
    probe_train_mask = np.concatenate([np.ones(PROBE_N, bool), np.zeros(PROBE_N, bool)])

    x = fr.norm_x(tr["px_x"], tr["dist"])[idx]
    d = fr.norm_d(tr["dist"])[idx]
    a = np.clip(tr["action"], -1.0, 1.0)[idx]

    # §2.2 对照:人工特权状态直接线性预测 teacher 动作。
    priv2 = np.stack([x, d], axis=1)
    priv5 = np.stack([x, d, x * x, d * d, x * d], axis=1)
    base2 = fr.r2_split(priv2, a, probe_train_mask)
    base5 = fr.r2_split(priv5, a, probe_train_mask)

    print("交接文档 §2.2 对照(同一探针、同一 episode 切分):")
    print(f"  人工特权 (x, d)        2 维 -> a   R2_a = {base2:.4f}")
    print(f"  人工特权 + 二次项      5 维 -> a   R2_a = {base5:.4f}")
    print()

    rows = []
    for arm in ARMS:
        path = root / arm / "stage2.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as file:
            log = list(csv.DictReader(file))
        first, last = log[0], log[-1]
        rows.append(
            {
                "arm": arm,
                "updates": last["update"],
                "loss": last["loss"],
                "r2_x": last["r2_x"],
                "r2_d": last["r2_d"],
                "r2_a": last["r2_a"],
                "eff_rank": last["eff_rank"],
                "r2_a_start": first["r2_a"],
                "eff_rank_start": first["eff_rank"],
            }
        )

    head = f"{'arm':5s} {'loss':>8s} {'R2_x':>8s} {'R2_d':>8s} {'R2_a':>8s} {'eff_rank':>9s}   (起点 R2_a / rank)"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['arm']:5s} {float(r['loss']):8.4f} {float(r['r2_x']):8.4f} {float(r['r2_d']):8.4f} "
            f"{float(r['r2_a']):8.4f} {float(r['eff_rank']):9.3f}   ({float(r['r2_a_start']):.4f} / {float(r['eff_rank_start']):.2f})"
        )
    print()
    winners = [r for r in rows if float(r["r2_a"]) > base2]
    print(f"R2_a 超过人工特权 2 维基线({base2:.4f})的臂: {[r['arm'] for r in winners] or '无'}")

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
