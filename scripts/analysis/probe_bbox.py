"""bbox teacher 的 student 表征可解释性:哪些量是"轴对齐"可读的 —— (x_c, 直径) 还是 (xmin, xmax, ymin, ymax) 还是 (x, d)。
用法: python probe_bbox.py <data_dir> <kind:full|adapter> <path> [backbone_pt]
   full:    FreeSpatialFeatureExtractor 的 state_dict(donor.pt / encoder_final.pt)
   adapter: bc_pretrain_adapter 的 adapter.pt,骨干 = backbone_pt
"""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "./src")
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from pathlib import Path
DEV = "cuda"; N = 2500
data = Path(sys.argv[1]); kind = sys.argv[2]; path = Path(sys.argv[3])
tr = fr.load_transitions(data, units=1); m_all = fr.episode_split(tr["episode"], seed=0); seen = tr["dist"] > 0.3
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.flatnonzero(m_all & seen), N, replace=False), rng.choice(np.flatnonzero((~m_all) & seen), N, replace=False)])
m = np.r_[np.ones(N, bool), np.zeros(N, bool)]; frames = fr.load_frames(tr["img"][idx], data)
px = tr["px_x"][idx].astype(np.float64); d = tr["dist"][idx].astype(np.float64)
r = 31.06 / d; cy = 76.0 + 83.1 / d
T = {"x_c": px, "diam": 2 * r, "d": d, "1/d": 1 / d, "cy": cy, "xmin": px - r, "xmax": px + r, "ymin": cy - r, "ymax": cy + r}
def ridge(f, y, lam=1e-2):
    mu, sd = f[m].mean(0), f[m].std(0) + 1e-9
    a = np.hstack([(f[m] - mu) / sd, np.ones((m.sum(), 1))]); b = np.hstack([(f[~m] - mu) / sd, np.ones(((~m).sum(), 1))])
    w = np.linalg.lstsq(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ y[m], rcond=None)[0]; p = b @ w
    return float(1 - ((y[~m] - p) ** 2).sum() / ((y[~m] - y[~m].mean()) ** 2).sum())
@torch.no_grad()
def encode():
    if kind == "full":
        e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(path, map_location=DEV)); e.eval()
        return np.concatenate([e(frames[i:i+128].to(DEV))["shared_feature"].float().cpu().numpy() for i in range(0, len(frames), 128)]).astype(np.float64)
    e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(sys.argv[4], map_location=DEV)); e.eval()
    sd = torch.load(path, map_location=DEV); h = sd["0.weight"].shape[0]
    ad = nn.Sequential(nn.Linear(sd["0.weight"].shape[1], h), nn.Tanh(), nn.Linear(h, 256), nn.LayerNorm(256)).to(DEV); ad.load_state_dict(sd); ad.eval()
    return np.concatenate([ad(e.forward_pooled(frames[i:i+128].to(DEV))).float().cpu().numpy() for i in range(0, len(frames), 128)]).astype(np.float64)
z = encode()
print(f"{kind} {path}  eff_rank {fr.eff_rank(z[~m]):.2f}   (held {N} 帧, d>0.3)")
print(f"{'量':6s}{'lin256':>8s}{'best1':>8s}{'轴#':>5s}{'best2':>8s}")
axes = {}
for k, y in T.items():
    per = [ridge(z[:, [j]], y) for j in range(z.shape[1])]; order = np.argsort(per)[::-1]
    axes[k] = int(order[0])
    print(f"{k:6s}{ridge(z, y):8.3f}{per[order[0]]:8.3f}{order[0]:5d}{ridge(z[:, order[:2]], y):8.3f}")
print("同一根轴最能读出的量:", {k: axes[k] for k in T})
# 关键对比:各量的 best1 —— (x_c, diam) 对 (xmin, xmax) 对 (d)
