"""v3:用**截断后**的动作(= 真实行为)做序列行为距离。
D1   = ||clip(mu_i) - clip(mu_j)||            当下行为差(逐点 BC 损失能看见的全部)
Dfar = mean_{k in [8,32)} ||clip(mu)_{i+k} - clip(mu)_{j+k}||  远未来行为差
基线 = 只用 D1 分箱回归 Dfar 的 held R²;表征 = lin8 度量探针的 held R²。
Δ = 表征 - 基线 > 0 才说明"表征知道未来分岔,而不只是当下动作"。
另测硬对(D1 最小 10%,含饱和造成的同动作不同状态)。
用法: CUDA_VISIBLE_DEVICES=g python metric_probe3.py <mask>
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
DEV = "cuda"; H = 32; N = 4000; NPAIR = 400_000; K = 8; torch.manual_seed(0)
tr = fr.load_transitions(DATA, units=1); mu = np.clip(tr["mu"], -1, 1).astype(np.float32)
order = np.lexsort((tr["t"], tr["episode"])); pos = np.empty(len(order), np.int64); pos[order] = np.arange(len(order))
ep_s, t_s = tr["episode"][order], tr["t"][order]; L = len(order); ok = np.ones(L, bool)
for k in range(1, H):
    nx = np.zeros(L, bool); nx[: L - k] = (ep_s[k:] == ep_s[: L - k]) & (t_s[k:] == t_s[: L - k] + k); ok &= nx
cand = order[np.flatnonzero(ok)]; cand = cand[tr["dist"][cand] > 0.3]
m_all = fr.episode_split(tr["episode"], seed=0); rng = np.random.default_rng(0)
rows = np.concatenate([rng.choice(cand[m_all[cand]], N, replace=False), rng.choice(cand[~m_all[cand]], N, replace=False)])
fut = np.stack([mu[order[pos[rows] + k]] for k in range(H)], 1)
def pr(lo, hi, n):
    a = rng.integers(lo, hi, n); b = rng.integers(lo, hi, n); k = a != b; return a[k], b[k]
ia, ib = pr(0, N, NPAIR); ja, jb = pr(N, 2 * N, NPAIR)
D1t = np.linalg.norm(fut[ia, 0] - fut[ib, 0], axis=1); D1h = np.linalg.norm(fut[ja, 0] - fut[jb, 0], axis=1)
DFt = np.linalg.norm(fut[ia, 8:] - fut[ib, 8:], axis=2).mean(1); DFh = np.linalg.norm(fut[ja, 8:] - fut[jb, 8:], axis=2).mean(1)
rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum() + 1e-30))
spear = lambda a, b: corr(rank(a), rank(b))
edges = np.quantile(D1t, np.linspace(0, 1, 201)); edges[0] -= 1e-6; edges[-1] += 1e-6
bt = np.clip(np.digitize(D1t, edges) - 1, 0, 199); bh = np.clip(np.digitize(D1h, edges) - 1, 0, 199)
bm = np.array([DFt[bt == i].mean() if (bt == i).sum() else DFt.mean() for i in range(200)])
r2 = lambda p, y: float(1 - ((p - y) ** 2).mean() / y.var())
base_all = r2(bm[bh], DFh)
hard_t = D1t <= np.quantile(D1t, 0.10); hard_h = D1h <= np.quantile(D1h, 0.10)
base_hard = r2(bm[bh[hard_h]], DFh[hard_h])
frames = fr.load_frames(tr["img"][rows], DATA)
@torch.no_grad()
def enc(p):
    e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(p, map_location=DEV)); e.eval()
    return np.concatenate([e(frames[i:i + 128].to(DEV))["shared_feature"].float().cpu().numpy() for i in range(0, len(frames), 128)])
def fit(Z, a, b, y, ah, bh_, yh):
    m_, s_ = Z[:N].mean(0), Z[:N].std(0) + 1e-6; zt = torch.tensor((Z - m_) / s_, device=DEV)
    W = nn.Linear(Z.shape[1], K, bias=False).to(DEV); opt = torch.optim.Adam(W.parameters(), lr=3e-3)
    ta, tb = torch.tensor(a, device=DEV), torch.tensor(b, device=DEV); td = torch.tensor(y.astype(np.float32), device=DEV)
    for _ in range(3000):
        s = torch.randint(0, len(ta), (8192,), device=DEV)
        loss = (((W(zt[ta[s]]) - W(zt[tb[s]])).norm(dim=1) - td[s]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): p = (W(zt[torch.tensor(ah, device=DEV)]) - W(zt[torch.tensor(bh_, device=DEV)])).norm(dim=1).cpu().numpy()
    return r2(p, yh), spear(p, yh)
print(f"== {M} ==  截断动作  held 对 {len(ja)}  Dfar 均值 {DFh.mean():.3f}  ρ(D1,Dfar)={spear(D1h,DFh):.3f}")
print(f"   硬对(D1≤{np.quantile(D1h,0.1):.3f},{hard_h.sum()} 对):Dfar 均值 {DFh[hard_h].mean():.3f} 标准差 {DFh[hard_h].std():.3f}")
print(f"   只用 D1 的基线 held R²(Dfar):全体 {base_all:.3f}  硬对 {base_hard:.3f}")
print(f"{'表征':12s}{'全体R²':>8s}{'Δ基线':>8s}{'ρ':>7s} |{'硬对R²':>8s}{'硬Δ':>8s}{'硬ρ':>7s}")
def rep(name, Z):
    Z = Z.astype(np.float32)
    a1, s1 = fit(Z, ia, ib, DFt, ja, jb, DFh); a2, s2 = fit(Z, ia[hard_t], ib[hard_t], DFt[hard_t], ja[hard_h], jb[hard_h], DFh[hard_h])
    print(f"{name:12s}{a1:8.3f}{a1-base_all:+8.3f}{s1:7.3f} |{a2:8.3f}{a2-base_hard:+8.3f}{s2:7.3f}", flush=True)
for n_, p in [("init", ROOT / "encoder_init.pt")] + [(a, ROOT / a / "donor.pt") for a in ("A", "V", "R", "sup")]:
    if p.exists(): rep(n_, enc(p))
from stable_baselines3 import PPO
pol = PPO.load(TZIP, device="cpu").policy
px = tr["px_x"][rows].astype(np.float64); d = tr["dist"][rows].astype(np.float64); r = 31.06 / d; cy = 76.0 + 83.1 / d
fx = 112 / np.tan(np.radians(40)); bg = np.degrees(np.arctan((112 - px) / fx)); rg = d / np.cos(np.radians(bg)); hf = 112.0
obs = {"": np.stack([px / hf - 1, d / 8, np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1),
       "diam": np.stack([px / hf - 1, np.clip(2 * r / 224, 0, 1), np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1),
       "xyd": np.stack([px / hf - 1, np.clip(2 * r / 224, 0, 1), np.full_like(d, 1.5 / 8), cy / hf - 1], 1),
       "bbox": np.stack([(px - r) / hf - 1, (px + r) / hf - 1, (cy - r) / hf - 1, (cy + r) / hf - 1], 1).clip(-1, 1),
       "ang": np.stack([np.clip(bg / 40, -1, 1), np.clip(rg / 8, 0, 1), np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1)}[MASK].astype(np.float32)
rep("特权obs", obs)
with torch.no_grad(): h2 = pol.mlp_extractor.policy_net(torch.as_tensor(obs)).numpy()
rep("teacher隐2", h2)
