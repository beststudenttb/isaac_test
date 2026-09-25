"""序列行为距离 D 的对齐诊断(离线,不开 Isaac)。
D_seq(i,j) = sum_k g^k ||mu_{i+k} - mu_{j+k}|| / sum_k g^k   (mu = teacher 确定性动作,沿各自轨迹展开)
D_1(i,j)   = ||mu_i - mu_j||                                  (逐点对照 = 现在 A 臂损失看得见的东西)
对每个表征量:raw 斯皮尔曼(||z_i-z_j||, D) 以及 线性度量探针 W:256->8 拟合 ||Wz_i-Wz_j||≈D 的 held R²/ρ。
参考上界:teacher 自己的隐层2、teacher 的特权观测。
用法: CUDA_VISIBLE_DEVICES=g python metric_probe.py <mask>
"""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "./src"); sys.path.insert(0, ".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr

M = sys.argv[1]; MASK = "" if M == "xd" else M
DATA = Path(f"./data_wk_{M}"); ROOT = Path(f"./models/vision/wk_{M}")
import glob
TZIP = sorted(glob.glob(f"models/rl/teacher_score_wk_{M}_s*/last.zip"))[-1]
DEV = "cuda"; H = 32; G = 0.95; N = 3000; NPAIR = 200_000; K = 8
torch.manual_seed(0)

tr = fr.load_transitions(DATA, units=1)
mu = tr["mu"].astype(np.float32)
order = np.lexsort((tr["t"], tr["episode"])); pos = np.empty(len(order), np.int64); pos[order] = np.arange(len(order))
ep_s, t_s = tr["episode"][order], tr["t"][order]
ok = np.ones(len(order), bool)
for k in range(1, H):
    nxt = np.full(len(order), False)
    nxt[: len(order) - k] = (ep_s[k:] == ep_s[: len(order) - k]) & (t_s[k:] == t_s[: len(order) - k] + k)
    ok &= nxt
cand_sorted = np.flatnonzero(ok)
seen = tr["dist"] > 0.3
m_all = fr.episode_split(tr["episode"], seed=0)
cand_rows = order[cand_sorted]
cand_rows = cand_rows[seen[cand_rows]]
rng = np.random.default_rng(0)
tr_rows = rng.choice(cand_rows[m_all[cand_rows]], N, replace=False)
he_rows = rng.choice(cand_rows[~m_all[cand_rows]], N, replace=False)
rows = np.concatenate([tr_rows, he_rows])
# 未来动作张量 (2N, H, adim)
fut = np.stack([mu[order[pos[rows] + k]] for k in range(H)], 1)
w = G ** np.arange(H); w = w / w.sum()

def pairs(lo, hi, n):
    a = rng.integers(lo, hi, n); b = rng.integers(lo, hi, n); keep = a != b
    return a[keep], b[keep]
ia, ib = pairs(0, N, NPAIR); ja, jb = pairs(N, 2 * N, NPAIR)
def dist_seq(a, b):
    dd = np.linalg.norm(fut[a] - fut[b], axis=2)  # (P,H)
    return (dd * w).sum(1)
D_tr, D_he = dist_seq(ia, ib), dist_seq(ja, jb)
D1_tr = np.linalg.norm(fut[ia, 0] - fut[ib, 0], axis=1); D1_he = np.linalg.norm(fut[ja, 0] - fut[jb, 0], axis=1)

frames = fr.load_frames(tr["img"][rows], DATA)
@torch.no_grad()
def enc(p):
    e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(p, map_location=DEV)); e.eval()
    return np.concatenate([e(frames[i:i + 128].to(DEV))["shared_feature"].float().cpu().numpy() for i in range(0, len(frames), 128)])

def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean(); return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

def report(name, Z):
    Z = Z.astype(np.float32); mu_, sd_ = Z[:N].mean(0), Z[:N].std(0) + 1e-6; Zn = (Z - mu_) / sd_
    out = [name]
    for D_t, D_h, ia_, ib_, ja_, jb_ in [(D_tr, D_he, ia, ib, ja, jb), (D1_tr, D1_he, ia, ib, ja, jb)]:
        raw = spear(np.linalg.norm(Z[ja_] - Z[jb_], axis=1), D_h)
        zt = torch.tensor(Zn, device=DEV)
        W = nn.Linear(Z.shape[1], K, bias=False).to(DEV); opt = torch.optim.Adam(W.parameters(), lr=3e-3)
        ta, tb, td = torch.tensor(ia_, device=DEV), torch.tensor(ib_, device=DEV), torch.tensor(D_t, device=DEV, dtype=torch.float32)
        for step in range(3000):
            s = torch.randint(0, len(ta), (8192,), device=DEV)
            p = (W(zt[ta[s]]) - W(zt[tb[s]])).norm(dim=1)
            loss = ((p - td[s]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ph = (W(zt[torch.tensor(ja_, device=DEV)]) - W(zt[torch.tensor(jb_, device=DEV)])).norm(dim=1).cpu().numpy()
        r2 = 1 - ((ph - D_h) ** 2).mean() / D_h.var()
        out += [raw, float(r2), spear(ph, D_h)]
    print(f"{out[0]:12s}{out[1]:9.3f}{out[2]:9.3f}{out[3]:9.3f} |{out[4]:9.3f}{out[5]:9.3f}{out[6]:9.3f}", flush=True)

print(f"== teacher {M} ==  H={H} γ={G}  held 帧 {N}  对数 {len(ja)}  D_seq 均值 {D_he.mean():.3f} 方差 {D_he.var():.4f}  D_1 均值 {D1_he.mean():.3f}")
print(f"{'表征':12s}{'raw ρ':>9s}{'lin8 R²':>9s}{'lin8 ρ':>9s} |{'raw ρ1':>9s}{'lin8 R²1':>9s}{'lin8 ρ1':>9s}   (左=序列 D,右=逐点 D_1)")
for name, p in [("init", ROOT / "encoder_init.pt")] + [(a, ROOT / a / "donor.pt") for a in ("A", "V", "R", "sup")]:
    if p.exists(): report(name, enc(p))
# 参考:teacher 特权观测 与 teacher 隐层2
from stable_baselines3 import PPO
pol = PPO.load(TZIP, device="cpu").policy
px = tr["px_x"][rows].astype(np.float64); d = tr["dist"][rows].astype(np.float64); r = 31.06 / d; cy = 76.0 + 83.1 / d
fx = 112 / np.tan(np.radians(40)); bearing = np.degrees(np.arctan((112 - px) / fx)); rng_m = d / np.cos(np.radians(bearing)); half = 112.0
obs = {"": np.stack([px / half - 1, d / 8.0, np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1),
       "diam": np.stack([px / half - 1, np.clip(2 * r / 224, 0, 1), np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1),
       "xyd": np.stack([px / half - 1, np.clip(2 * r / 224, 0, 1), np.full_like(d, 1.5 / 8), cy / half - 1], 1),
       "bbox": np.stack([(px - r) / half - 1, (px + r) / half - 1, (cy - r) / half - 1, (cy + r) / half - 1], 1).clip(-1, 1),
       "ang": np.stack([np.clip(bearing / 40, -1, 1), np.clip(rng_m / 8, 0, 1), np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1)}[MASK]
report("特权obs", obs)
with torch.no_grad(): h2 = pol.mlp_extractor.policy_net(torch.as_tensor(obs, dtype=torch.float32)).numpy()
report("teacher隐2", h2)
