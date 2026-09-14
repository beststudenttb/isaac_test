"""坐标探针:表征把哪一套坐标做成了线性可读的,又扔掉了什么。

交接文档 §8 开放项 3(蓝球)+ 用户 2026-09-04 的补充(视觉直径能替代 d)。
全部从图像颜色阈值直接量,不需要 Isaac。

    python scripts/probe_nuisance.py

三组量:
  特权坐标   norm_x / norm_d          —— teacher 实际吃的那两维
  图像原生   直径 / 球心 y / 下缘 y   —— 全部 ∝ 1/d,与 d 的线性相关只有 0.94
  任务无关   蓝球 px_x / 视觉大小     —— 与红球标签和 teacher 行为完全无关

判据:图像原生这一族的 R2 明显高于 norm_d,就是"表征自主选了图像坐标而不是
特权度量坐标"的直接证据;蓝球 R2 掉下去,是"扔掉了任务无关变量"的直接证据。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_image

sys.path.insert(0, "./src")

from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import ARMS

PROBE_N = 6000  # 与 stage2_representations.PROBE_N 一致。
MIN_PIXELS = 12
FEATURE_BATCH = 256
DEVICE = "cuda"
COLS = torch.arange(224.0)
ROWS = torch.arange(224.0)


def blob(mask: torch.Tensor) -> tuple[float, float, float, float]:
    """等效直径、质心 x、质心 y、下缘 y。"""
    m = mask.float()
    n = float(m.sum())
    return (
        2.0 * float(np.sqrt(n / np.pi)),
        float((m.sum(0) * COLS).sum() / n),
        float((m.sum(1) * ROWS).sum() / n),
        float(torch.nonzero(mask)[:, 0].max()),
    )


def image_labels(names: np.ndarray) -> dict[str, np.ndarray]:
    n = len(names)
    out = {k: np.zeros(n, dtype=np.float32) for k in ("diam", "cy", "bottom", "blue_x", "blue_size")}
    out["red_seen"] = np.zeros(n, dtype=bool)
    out["blue_seen"] = np.zeros(n, dtype=bool)
    for i, name in enumerate(names):
        im = read_image(str(fr.DATA_DIR / name)).float()
        r, g, b = im[0], im[1], im[2]
        red = (r > 90) & (r > b * 1.8) & (r > g * 1.8)
        blue = (b > 90) & (b > r * 1.8) & (b > g * 1.8)
        if float(red.sum()) > MIN_PIXELS:
            out["red_seen"][i] = True
            out["diam"][i], _, out["cy"][i], out["bottom"][i] = blob(red)
        if float(blue.sum()) > MIN_PIXELS:
            out["blue_seen"][i] = True
            d, bx, _, _ = blob(blue)
            out["blue_x"][i], out["blue_size"][i] = bx, d
    return out


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
    train_mask = np.load(root / "train_mask.npy")
    rng = np.random.default_rng(0)
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(train_mask), PROBE_N, replace=False),
            rng.choice(np.flatnonzero(~train_mask), PROBE_N, replace=False),
        ]
    )
    split = np.concatenate([np.ones(PROBE_N, bool), np.zeros(PROBE_N, bool)])

    print("[INFO] measuring image-native quantities from pixels")
    lab = image_labels(tr["img"][idx])
    red, blue = lab["red_seen"], lab["blue_seen"]
    print(f"[INFO] red visible {red.mean():.1%}  blue visible {blue.mean():.1%}")

    dist = tr["dist"][idx]
    priv_x = fr.norm_x(tr["px_x"], tr["dist"])[idx]
    priv_d = fr.norm_d(tr["dist"])[idx]
    inv_d = np.where(dist > fr.LOST_D, 1.0 / np.maximum(dist, 1e-3), 0.0).astype(np.float32)

    frames = fr.load_frames(tr["img"][idx])
    probes = [
        ("x", priv_x, red),
        ("d", priv_d, red),
        ("1/d", inv_d, red),
        ("diam", lab["diam"], red),
        ("cy", lab["cy"], red),
        ("bottom", lab["bottom"], red),
        ("blue_x", lab["blue_x"], blue),
        ("blue_sz", lab["blue_size"], blue),
    ]

    rows = []
    targets = [("init", root / "encoder_init.pt")] + [(a, root / a / "encoder_final.pt") for a in list(ARMS) + ["spr"]]
    for name, path in targets:
        if not path.exists():
            continue
        f = encode(path, frames)
        row = {"arm": name}
        for key, y, mask in probes:
            row[key] = round(fr.r2_split(f[mask], y[mask], split[mask]), 4)
        row["rank"] = round(fr.eff_rank(f[~split]), 2)
        rows.append(row)
        print("  " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    fields = list(rows[0])
    head = f"{'arm':6s}" + "".join(f"{k:>9s}" for k in fields[1:])
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['arm']:6s}" + "".join(f"{r[k]:9.4f}" for k in fields[1:]))
    print("\n特权 d vs 图像原生族(diam/cy/bottom):后者明显更高 = 表征选了图像坐标。")
    print("blue_x / blue_sz 相对 init 掉得越多 = 任务无关信息被扔得越干净。")

    with (root / "nuisance.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {root / 'nuisance.csv'}")


if __name__ == "__main__":
    main()
