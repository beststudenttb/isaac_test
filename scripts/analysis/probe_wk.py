"""周末批次探针:对一份数据集下的各表征(A/V/R/sup donor + init)量 任务量/无关量 的可读性(线性 train/held、单轴、主方向、秩)+ 互换测试。
用法: CUDA_VISIBLE_DEVICES=g python probe_wk.py <data_dir> <vision_root> <teacher_zip> <mask>
"""
import sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "./src"); sys.path.insert(0, ".")
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import make_head
from pathlib import Path
DEV = "cuda"; N = 3000
data, root, tzip, mask = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
tr = fr.load_transitions(data, units=1); m_all = fr.episode_split(tr["episode"], seed=0); seen = tr["dist"] > 0.3
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.flatnonzero(m_all & seen), N, replace=False), rng.choice(np.flatnonzero((~m_all) & seen), N, replace=False)])
m = np.r_[np.ones(N, bool), np.zeros(N, bool)]; frames = fr.load_frames(tr["img"][idx], data)
px = tr["px_x"][idx].astype(np.float64); d = tr["dist"][idx].astype(np.float64); r = 31.06 / d; cy = 76.0 + 83.1 / d
fx = 112 / np.tan(np.radians(40)); bearing = np.degrees(np.arctan((112 - px) / fx)); rng_m = d / np.cos(np.radians(bearing))
f = frames.float(); R_, G_, B_ = f[:, 0], f[:, 1], f[:, 2]; blue = (B_ > 90) & (B_ > R_ * 1.8) & (B_ > G_ * 1.8)
bsz = blue.flatten(1).sum(1).float().numpy().astype(np.float64); has_blue = bsz > 30
xs = torch.arange(224).view(1, 1, 224).expand(len(f), 224, 224).float(); bx = ((blue.float() * xs).flatten(1).sum(1) / blue.flatten(1).sum(1).clamp(min=1)).numpy().astype(np.float64)
T = {"x_c": px, "diam": 2 * r, "d": d, "bearing": bearing, "range": rng_m, "cy": cy, "xmin": px - r, "xmax": px + r, "blue_x": bx, "blue_sz": bsz, "shuffled_d": rng.permutation(d)}
MASKS = {k: (has_blue if k.startswith("blue") else np.ones(len(px), bool)) for k in T}
def ridge(F, y, mk, lam=1e-2):
    trm, tem = m & mk, (~m) & mk
    mu, sd = F[trm].mean(0), F[trm].std(0) + 1e-9; a = np.hstack([(F[trm] - mu) / sd, np.ones((trm.sum(), 1))]); b = np.hstack([(F[tem] - mu) / sd, np.ones((tem.sum(), 1))])
    w = np.linalg.lstsq(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ y[trm], rcond=None)[0]
    return float(1 - ((y[trm] - a @ w) ** 2).sum() / ((y[trm] - y[trm].mean()) ** 2).sum()), float(1 - ((y[tem] - b @ w) ** 2).sum() / ((y[tem] - y[tem].mean()) ** 2).sum())
def pc_r2(z, y, mk):
    zc = z - z[m].mean(0); U, S, Vt = np.linalg.svd(zc[m], full_matrices=False); pcs = zc @ Vt[:2].T; var = S ** 2 / (S ** 2).sum()
    out = []
    for j in range(2):
        trm, tem = m & mk, (~m) & mk
        a = np.c_[pcs[trm, j], np.ones(trm.sum())]; w = np.linalg.lstsq(a, y[trm], rcond=None)[0]; p = np.c_[pcs[tem, j], np.ones(tem.sum())] @ w
        out.append(float(1 - ((y[tem] - p) ** 2).sum() / ((y[tem] - y[tem].mean()) ** 2).sum()))
    return out, var[:2]
@torch.no_grad()
def enc(p):
    e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(p, map_location=DEV)); e.eval()
    return np.concatenate([e(frames[i:i + 128].to(DEV))["shared_feature"].float().cpu().numpy() for i in range(0, len(frames), 128)]).astype(np.float64)
print(f"数据 {data}  mask={mask}  held {N} 帧  有蓝球 {has_blue.mean():.0%}")
reps = [("init", root / "encoder_init.pt")] + [(a, root / a / "donor.pt") for a in ("A", "V", "R", "sup") if (root / a / "donor.pt").exists()]
Z = {}
for name, p in reps:
    z = enc(p); Z[name] = z
    print(f"\n== {name}  秩 {fr.eff_rank(z[~m]):.2f} ==")
    print(f"{'量':11s}{'lin_tr':>8s}{'lin_he':>8s}{'best1':>7s}{'PC1':>7s}{'PC2':>7s}")
    for k, y in T.items():
        mk = MASKS[k]; rt, rh = ridge(z, y, mk); b1 = max(ridge(z[:, [j]], y, mk)[1] for j in range(z.shape[1])); pc, var = pc_r2(z, y, mk)
        print(f"{k:11s}{rt:8.3f}{rh:8.3f}{b1:7.3f}{pc[0]:7.3f}{pc[1]:7.3f}")
    print(f"   主成分占比 {np.round(var, 3)}")
# 互换测试:A 臂 z -> 线性 -> teacher 观测 -> teacher 策略,动作与 teacher 确定性动作的一致性
try:
    from stable_baselines3 import PPO
    tch = PPO.load(tzip, device="cpu"); pol = tch.policy
    half = 112.0
    if mask == "": obs = np.stack([px / half - 1, d / 8.0, np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1)
    elif mask == "diam": obs = np.stack([px / half - 1, np.clip(2 * r / 224, 0, 1), np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1)
    elif mask == "xyd": obs = np.stack([px / half - 1, np.clip(2 * r / 224, 0, 1), np.full_like(d, 1.5 / 8), cy / half - 1], 1)
    elif mask == "bbox": obs = np.stack([(px - r) / half - 1, (px + r) / half - 1, (cy - r) / half - 1, (cy + r) / half - 1], 1).clip(-1, 1)
    elif mask == "ang": obs = np.stack([np.clip(bearing / 40, -1, 1), np.clip(rng_m / 8, 0, 1), np.full_like(d, 1.5 / 8), np.zeros_like(d)], 1)
    else: obs = None
    if obs is not None and "A" in Z:
        def tact(o):
            with torch.no_grad(): return pol.action_net(pol.mlp_extractor.policy_net(torch.as_tensor(o, dtype=torch.float32))).numpy().clip(-1, 1)
        base = tact(obs.astype(np.float32))
        z = Z["A"]; mu, sd = z[m].mean(0), z[m].std(0) + 1e-9; a = np.hstack([(z[m] - mu) / sd, np.ones((m.sum(), 1))]); b = np.hstack([(z[~m] - mu) / sd, np.ones(((~m).sum(), 1))])
        w = np.linalg.lstsq(a.T @ a + 1e-2 * len(a) * np.eye(a.shape[1]), a.T @ obs[m], rcond=None)[0]; obs_hat = b @ w
        act_hat = tact(obs_hat.astype(np.float32)); bh = base[~m]
        act_true = np.clip(tr["action"][idx], -1, 1)[~m]
        r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
        print(f"\n互换测试(held):z_A -线性-> teacher 观测 -> teacher 策略,与 teacher 确定性动作 R² {r2(act_hat, bh):.3f}(teacher 自身采样噪声底 {r2(act_true, bh):.3f});z -> 观测 线性 R² {r2(obs_hat, obs[~m]):.3f}")
except Exception as ex:
    print("互换测试跳过:", ex)
