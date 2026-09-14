"""Shared pieces for the emergence line: data, teacher signals, probes.

Handoff doc 2026-09-03 §3/§4.  Everything here is offline — no Isaac.

The CSV stores *raw* labels (`px_x` in pixels with a -1 sentinel, `dist` in
metres with a 0 sentinel) and *unclamped* teacher actions.  The env clamps to
[-1, 1] inside `sb3_env.apply_actions`, so behaviour is the clamped value and
that is what the arms are supervised on.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_image

# 管线标签(2026-09-09):k03 = k=0.3 三维动作;k03noy = k=0.3 去掉侧移 a_y,动作 [a_x, a_w]。
TAG = "k03noy"
PIPELINES = {
    "k03": (Path("./data_score_k03_1m"), Path("./models/rl/teacher_score_k0.3/last.zip"), Path("./models/vision/score_k03"), (0, 1, 2)),
    "k03noy": (Path("./data_score_k03noy_1m"), Path("./models/rl/teacher_score_k0.3_noy/last.zip"), Path("./models/vision/score_k03noy"), (0, 2)),
}
DATA_DIR, TEACHER_PATH, OUT_ROOT, ACT_IDX = PIPELINES[TAG]
ACT_DIM = len(ACT_IDX)  # CSV 永远三列(a_x, a_y, a_w);无侧移管线只取 (0, 2),A 臂目标随之变 2 维。

IMAGE_WIDTH = 224.0
FAIL_FAR = 8.0
LOST_D = 0.0
STOP_D = 1.5
END_OBS = (STOP_D / FAIL_FAR, 0.0)  # 固定 stop 目标，goal 退化成常量。


def load_transitions(data_dir: Path = DATA_DIR, units: int = 2) -> dict[str, np.ndarray]:
    """读前 `units` 个单元。1M 张图解码进内存要 150GB,机器只有 125GB,
    所以按单元取子集 —— 单元边界不切断轨迹,取几个都是完整数据集。
    2026-09-08 默认 3->2:每单元约 15.7GB,两个单元 31GB,可以两个 stage2 进程并行(62GB)而不 OOM。"""
    rows = []
    for u in range(units):
        unit_dir = data_dir / f"unit_{u:02d}"
        with (unit_dir / "transitions.csv").open(newline="", encoding="utf-8") as file:
            for r in csv.DictReader(file):
                r["img"] = f"unit_{u:02d}/img/{r['img']}"
                r["next_img"] = f"unit_{u:02d}/img/{r['next_img']}"
                r["env_id"] = f"{u}{r['env_id']}"
                rows.append(r)
    out = {
        "img": np.array([r["img"] for r in rows]),
        "next_img": np.array([r["next_img"] for r in rows]),
        "px_x": np.array([float(r["px_x"]) for r in rows], dtype=np.float32),
        "dist": np.array([float(r["dist"]) for r in rows], dtype=np.float32),
        "reward": np.array([float(r["reward"]) for r in rows], dtype=np.float32),
        "t": np.array([int(r["t"]) for r in rows], dtype=np.int64),
        "done": np.array([int(r["done"]) for r in rows], dtype=np.float32),
        "episode": np.array([int(r["env_id"]) * 100000 + int(r["episode"]) for r in rows], dtype=np.int64),
    }
    out["action"] = np.stack(
        [np.array([float(r[k]) for r in rows], dtype=np.float32) for k in ("a_x", "a_y", "a_w")], axis=1
    )
    # 新数据集额外记了 teacher 的确定性 mean 与 V(s):A 臂用 mu(无采样噪声),V 臂直接取用。
    out["mu"] = np.stack(
        [np.array([float(r[k]) for r in rows], dtype=np.float32) for k in ("mu_x", "mu_y", "mu_w")], axis=1
    )
    out["action"] = out["action"][:, list(ACT_IDX)]
    out["mu"] = out["mu"][:, list(ACT_IDX)]
    out["value"] = np.array([float(r["value"]) for r in rows], dtype=np.float32)
    return out


def norm_x(px_x: np.ndarray, dist: np.ndarray) -> np.ndarray:
    seen = dist > LOST_D
    x_n = px_x / (IMAGE_WIDTH * 0.5) - 1.0
    return np.where(seen, np.clip(x_n, -1.0, 1.0), -1.0).astype(np.float32)


def norm_d(dist: np.ndarray) -> np.ndarray:
    seen = dist > LOST_D
    d_n = dist / FAIL_FAR
    return np.where(seen, np.clip(d_n, 0.0, 1.0), 0.0).astype(np.float32)


def teacher_obs(px_x: np.ndarray, dist: np.ndarray) -> np.ndarray:
    n = len(dist)
    return np.stack(
        [
            norm_x(px_x, dist),
            norm_d(dist),
            np.full(n, END_OBS[0], dtype=np.float32),
            np.full(n, END_OBS[1], dtype=np.float32),
        ],
        axis=1,
    )


def load_frames(names: np.ndarray, data_dir: Path = DATA_DIR) -> torch.Tensor:
    """Decode every frame into one uint8 NCHW tensor held in RAM (~7.5 GB)."""
    frames = torch.empty((len(names), 3, 224, 224), dtype=torch.uint8)
    for i, name in enumerate(names):
        frames[i] = read_image(str(data_dir / name))
    return frames


def episode_split(episode: np.ndarray, seed: int = 0) -> np.ndarray:
    """Train mask that keeps whole episodes on one side (no temporal leakage)."""
    keys = np.unique(episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    train_keys = set(keys[: len(keys) // 2].tolist())
    return np.array([e in train_keys for e in episode], dtype=bool)


def r2_split(feature: np.ndarray, target: np.ndarray, train_mask: np.ndarray, lam: float = 1e-2) -> float:
    """Ridge linear probe, R2 on the held-out side.  Same math as the Webots probe."""
    if target.ndim == 1:
        target = target[:, None]
    f_tr, f_te = feature[train_mask], feature[~train_mask]
    y_tr, y_te = target[train_mask], target[~train_mask]
    mu, sd = f_tr.mean(0), f_tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    a = np.hstack([(f_tr - mu) / sd, np.ones((len(f_tr), 1), dtype=np.float64)])
    b = np.hstack([(f_te - mu) / sd, np.ones((len(f_te), 1), dtype=np.float64)])
    w = np.linalg.lstsq(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ y_tr, rcond=None)[0]
    return float(1 - ((y_te - b @ w) ** 2).sum() / ((y_te - y_te.mean(0)) ** 2).sum())


def eff_rank(feature: np.ndarray) -> float:
    """Entropy exponent of the correlation spectrum."""
    z = (feature - feature.mean(0)) / (feature.std(0) + 1e-9)
    z = z[:, np.isfinite(z).all(0)]
    s = np.linalg.svd(z, compute_uv=False) ** 2
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def teacher_values(tr: dict[str, np.ndarray], teacher_path: Path = TEACHER_PATH) -> np.ndarray:
    """Teacher V(s) recomputed offline from the rebuilt privileged observation.

    Verified exact: feeding this observation back through the teacher reproduces
    the recorded actions to within the policy's own log_std on all three dims.
    """
    from stable_baselines3 import PPO

    model = PPO.load(str(teacher_path), device="cpu")
    obs = torch.as_tensor(teacher_obs(tr["px_x"], tr["dist"]))
    with torch.no_grad():
        return model.policy.predict_values(obs).numpy().ravel().astype(np.float32)
