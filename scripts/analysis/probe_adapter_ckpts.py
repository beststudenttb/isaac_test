"""适配器变体:沿训练 ckpt 看适配器输出的表征离 teacher 的特权状态 (x,d) 有多近。
用法: CUDA_VISIBLE_DEVICES=0 python probe_adapter_ckpts.py <run_dir> <encoder_root> [ckpt 步长]
输出:每个 ckpt 的 r2_x / r2_d / best1_x / best1_d / eff_rank / 对 sup 表征的 CKA / 对 R 表征的 CKA
"""
import sys, glob, re, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "./src")
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from pathlib import Path
DEV = "cuda"; N = 2000
run = Path(sys.argv[1]); eroot = Path(sys.argv[2]); every = int(sys.argv[3]) if len(sys.argv) > 3 else 30
tr = fr.load_transitions(); tm = np.load(fr.OUT_ROOT / "train_mask.npy"); rng = np.random.default_rng(0)
seen = tr["dist"] > 0
idx = np.concatenate([rng.choice(np.flatnonzero(tm & seen), N, replace=False), rng.choice(np.flatnonzero((~tm) & seen), N, replace=False)])
m = np.concatenate([np.ones(N, bool), np.zeros(N, bool)]); frames = fr.load_frames(tr["img"][idx])
x = fr.norm_x(tr["px_x"], tr["dist"])[idx]; d = fr.norm_d(tr["dist"])[idx]
def ridge(f, y, lam=1e-2):
    mu, sd = f[m].mean(0), f[m].std(0) + 1e-9
    a = np.hstack([(f[m] - mu) / sd, np.ones((m.sum(), 1))]); b = np.hstack([(f[~m] - mu) / sd, np.ones(((~m).sum(), 1))])
    w = np.linalg.lstsq(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ y[m], rcond=None)[0]; p = b @ w
    return float(1 - ((y[~m] - p) ** 2).sum() / ((y[~m] - y[~m].mean()) ** 2).sum())
def cka(a, b):
    a = a[~m] - a[~m].mean(0); b = b[~m] - b[~m].mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))
@torch.no_grad()
def encode_full(p):
    e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(p, map_location=DEV)); e.eval()
    return np.concatenate([e(frames[i:i+128].to(DEV))["shared_feature"].float().cpu().numpy() for i in range(0, len(frames), 128)]).astype(np.float64)
@torch.no_grad()
def pooled(p):
    e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(p, map_location=DEV)); e.eval()
    return torch.cat([e.forward_pooled(frames[i:i+128].to(DEV)).float() for i in range(0, len(frames), 128)])
P = pooled(eroot / "encoder_init.pt")
ref = {"sup": encode_full(fr.OUT_ROOT / "sup/encoder_final.pt"), "R": encode_full(fr.OUT_ROOT / "R/encoder_final.pt")}
ckpts = sorted(glob.glob(str(run / "updates/ppo_*.pt")), key=lambda s: int(re.findall(r"(\d+)\.pt", s)[0]))
ckpts = [c for c in ckpts if int(re.findall(r"(\d+)\.pt", c)[0]) % every == 0] + [str(run / "last.pt")]
print(f"{'ckpt':8s}{'r2_x':>7s}{'r2_d':>7s}{'best1_x':>9s}{'best1_d':>9s}{'rank':>7s}{'CKA sup':>9s}{'CKA R':>7s}")
for c in ckpts:
    sd = torch.load(c, map_location=DEV)["model"]
    if "adapter.2.weight" in sd:   # 两层适配器
        h = sd["adapter.0.weight"].shape[0]
        ad = nn.Sequential(nn.Linear(P.shape[1], h), nn.Tanh(), nn.Linear(h, 256), nn.LayerNorm(256)).to(DEV)
    else:
        ad = nn.Sequential(nn.Linear(P.shape[1], 256), nn.LayerNorm(256)).to(DEV)
    ad.load_state_dict({k[len("adapter."):]: v for k, v in sd.items() if k.startswith("adapter.")})
    with torch.no_grad(): z = ad(P).cpu().numpy().astype(np.float64)
    b1x = max(ridge(z[:, [j]], x) for j in range(z.shape[1])); b1d = max(ridge(z[:, [j]], d) for j in range(z.shape[1]))
    name = re.findall(r"(\d+)\.pt", c)[0].lstrip("0") if "ppo_" in c else "last"
    print(f"{name:8s}{ridge(z, x):7.3f}{ridge(z, d):7.3f}{b1x:9.3f}{b1d:9.3f}{fr.eff_rank(z[~m]):7.2f}{cka(z, ref['sup']):9.3f}{cka(z, ref['R']):7.3f}", flush=True)
