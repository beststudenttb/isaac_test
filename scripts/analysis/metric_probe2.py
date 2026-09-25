"""v2:把"当下动作差"控制掉,只测表征有没有编码"未来分岔"。
D1    = ||mu_i - mu_j||                         当下动作差(A 臂损失能看见的全部)
Dfar  = mean_{k in [8,32)} ||mu_{i+k}-mu_{j+k}|| 远未来行为差(路径)
测:1) 全体对上 raw ρ(z 距离, Dfar) 与 偏相关 ρ(z, Dfar | D1)
    2) 硬对(D1 最小的 10%:此刻动作几乎一样)上 raw ρ 与 lin8 held R²
用法: CUDA_VISIBLE_DEVICES=g python metric_probe2.py <mask>
"""
import sys, glob, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "./src"); sys.path.insert(0, ".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr

M = sys.argv[1]; MASK = "" if M == "xd" else M
DATA = Path(f"./data_wk_{M}"); ROOT = Path(f"./models/vision/wk_{M}")
TZIP = sorted(glob.glob(f"models/rl/teacher_score_wk_{M}_s*/last.zip"))[-1]
DEV = "cuda"; H = 32; N = 4000; NPAIR = 400_000; K = 8
torch.manual_seed(0)
tr = fr.load_transitions(DATA, units=1); mu = tr["mu"].astype(np.float32)
order = np.lexsort((tr["t"], tr["episode"])); pos = np.empty(len(order), np.int64); pos[order] = np.arange(len(order))
ep_s, t_s = tr["episode"][order], tr["t"][order]; L = len(order)
ok = np.ones(L, bool)
for k in range(1, H):
    nxt = np.zeros(L, bool); nxt[: L - k] = (ep_s[k:] == ep_s[: L - k]) & (t_s[k:] == t_s[: L - k] + k); ok &= nxt
cand = order[np.flatnonzero(ok)]; cand = cand[tr["dist"][cand] > 0.3]
m_all = fr.episode_split(tr["episode"], seed=0); rng = np.random.default_rng(0)
rows = np.concatenate([rng.choice(cand[m_all[cand]], N, replace=False), rng.choice(cand[~m_all[cand]], N, replace=False)])
fut = np.stack([mu[order[pos[rows] + k]] for k in range(H)], 1)
def pr(lo, hi, n):
    a = rng.integers(lo, hi, n); b = rng.integers(lo, hi, n); k = a != b; return a[k], b[k]
ia, ib = pr(0, N, NPAIR); ja, jb = pr(N, 2 * N, NPAIR)
d1 = lambda a, b: np.linalg.norm(fut[a, 0] - fut[b, 0], axis=1)
dfar = lambda a, b: np.linalg.norm(fut[a, 8:] - fut[b, 8:], axis=2).mean(1)
D1t, D1h = d1(ia, ib), d1(ja, jb); DFt, DFh = dfar(ia, ib), dfar(ja, jb)
rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum() + 1e-30))
spear = lambda a, b: corr(rank(a), rank(b))
def resid(y, x):
    xr = rank(x); xr = xr - xr.mean(); yr = rank(y); yr = yr - yr.mean(); return yr - xr * (yr * xr).sum() / (xr ** 2).sum()
hard_t = D1t <= np.quantile(D1t, 0.10); hard_h = D1h <= np.quantile(D1h, 0.10)
frames = fr.load_frames(tr["img"][rows], DATA)
@torch.no_grad()
def enc(p):
    e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(p, map_location=DEV)); e.eval()
    return np.concatenate([e(frames[i:i + 128].to(DEV))["shared_feature"].float().cpu().numpy() for i in range(0, len(frames), 128)])
def fit_lin(Z, a, b, y, ah, bh, yh):
    mu_, sd_ = Z[:N].mean(0), Z[:N].std(0) + 1e-6; zt = torch.tensor((Z - mu_) / sd_, device=DEV)
    W = nn.Linear(Z.shape[1], K, bias=False).to(DEV); opt = torch.optim.Adam(W.parameters(), lr=3e-3)
    ta, tb, td = (torch.tensor(v, device=DEV) for v in (a, b, y.astype(np.float32)))
    for _ in range(3000):
        s = torch.randint(0, len(ta), (8192,), device=DEV)
        loss = (((W(zt[ta[s]]) - W(zt[tb[s]])).norm(dim=1) - td[s]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): ph = (W(zt[torch.tensor(ah, device=DEV)]) - W(zt[torch.tensor(bh, device=DEV)])).norm(dim=1).cpu().numpy()
    return 1 - ((ph - yh) ** 2).mean() / yh.var(), spear(ph, yh)
print(f"== {M} ==  N={N} 对={len(ja)}  Dfar 均值 {DFh.mean():.3f}  硬对(D1 最小 10%,阈 {np.quantile(D1h,0.1):.3f})上 Dfar 均值 {DFh[hard_h].mean():.3f} 标准差 {DFh[hard_h].std():.3f}")
print(f"   基线:全体 ρ(D1,Dfar)={spear(D1h,DFh):.3f}  硬对上 ρ(D1,Dfar)={spear(D1h[hard_h],DFh[hard_h]):.3f}")
print(f"{'表征':12s}{'ρ(z,Dfar)':>10s}{'偏ρ|D1':>9s} |{'硬 ρ(z,Dfar)':>13s}{'硬 lin8 R²':>11s}{'硬 lin8 ρ':>10s}")
def report(name, Z):
    Z = Z.astype(np.float32); zdh = np.linalg.norm(Z[ja] - Z[jb], axis=1)
    a = spear(zdh, DFh); b = corr(resid(zdh, D1h), resid(DFh, D1h))
    r2, rho = fit_lin(Z, ia[hard_t], ib[hard_t], DFt[hard_t], ja[hard_h], jb[hard_h], DFh[hard_h])
    print(f"{name:12s}{a:10.3f}{b:9.3f} |{spear(zdh[hard_h],DFh[hard_h]):13.3f}{r2:11.3f}{rho:10.3f}", flush=True)
for name, p in [("init", ROOT / "encoder_init.pt")] + [(a, ROOT / a / "donor.pt") for a in ("A", "V", "R", "sup")]:
    if p.exists(): report(name, enc(p))
from stable_baselines3 import PPO
pol = PPO.load(TZIP, device="cpu").policy
px = tr["px_x"][rows].astype(np.float64); d = tr["dist"][rows].astype(np.float64); r = 31.06 / d; cy = 76.0 + 83.1 / d
fx = 112 / np.tan(np.radians(40)); bearing = np.degrees(np.arctan((112 - px) / fx)); rg = d / np.cos(np.radians(bearing)); half = 112.0
obs = {"": np.stack([px / half - 1, d / 8, np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1),
       "diam": np.stack([px / half - 1, np.clip(2 * r / 224, 0, 1), np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1),
       "xyd": np.stack([px / half - 1, np.clip(2 * r / 224, 0, 1), np.full_like(d, 1.5 / 8), cy / half - 1], 1),
       "bbox": np.stack([(px - r) / half - 1, (px + r) / half - 1, (cy - r) / half - 1, (cy + r) / half - 1], 1).clip(-1, 1),
       "ang": np.stack([np.clip(bearing / 40, -1, 1), np.clip(rg / 8, 0, 1), np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1)}[MASK].astype(np.float32)
report("特权obs", obs)
with torch.no_grad(): report("teacher隐2", pol.mlp_extractor.policy_net(torch.as_tensor(obs)).numpy())
