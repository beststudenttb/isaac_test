"""Le Lan et al. AISTATS 2022 (arXiv 2203.00543) 的 effective dimension,搬到我们的冻结表征上。
定义 2: d_eff(Φ) = S · max_i ||P_Φ e_i||^2,P_Φ 是列空间的正交投影;rank(Φ) ≤ d_eff ≤ S。
他们的 bound = 近似误差 ||P⊥V||^2 + 估计误差(∝ σ² d_eff / n)。
我们的探针只量了第一项(而且全部饱和);第二项才是没量过的。顺手把 Z 存下来给后续复用。CPU。
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(8)
ROOT=Path("models/vision/g4_diam_angle"); DATA=Path("./data_g4_diam_angle")
CACHE=Path("/home/tb/.claude/jobs/2174f11a/tmp/zcache"); CACHE.mkdir(exist_ok=True)
N=2000; rg=np.random.default_rng(0)
tr=fr.load_transitions(DATA,units=1)
ok=(tr["dist"]>0.3)&(tr["done"]<0.5)
idx=rg.choice(np.flatnonzero(ok),N,replace=False)   # 与 wang_props.py 同一 seed 同一子集
np.save(CACHE/"idx.npy",idx)
for k in ("value","reward","px_x","dist"): np.save(CACHE/f"{k}.npy",tr[k][idx])
np.save(CACHE/"mu.npy",tr["mu"][idx])
F0=fr.load_frames(tr["img"][idx],DATA)

@torch.no_grad()
def enc(p,frames):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(frames[i:i+64])["shared_feature"].float() for i in range(0,len(frames),64)]).numpy().astype(np.float64)

def effdim(Z,tol=1e-8):
    Zc=Z-Z.mean(0)
    U,s,_=np.linalg.svd(Zc,full_matrices=False)
    r=int((s>tol*s[0]).sum())
    P_ii=(U[:,:r]**2).sum(1)                  # 杠杆值 = ||P e_i||^2
    return len(Z)*float(P_ii.max()), r, float(P_ii.mean()*len(Z)), float(s[0]/s[r-1])

V=np.load(CACHE/"value.npy").astype(np.float64)
def approx_err(Z,y):
    Zc=np.hstack([Z-Z.mean(0),np.ones((len(Z),1))])
    w=np.linalg.lstsq(Zc,y,rcond=None)[0]; res=y-Zc@w
    return float((res**2).sum()/((y-y.mean())**2).sum())

print("effective dimension(N=2000,k=256,故 256 ≤ d_eff ≤ 2000;越小越好)")
print(f"{'表征':6s}{'d_eff':>9s}{'rank':>7s}{'相干度µ':>10s}{'条件数':>12s}{'V近似误差':>12s}")
for n_,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p,F0); np.save(CACHE/f"z_{n_}.npy",Z.astype(np.float32))
    d,r,dm,cond=effdim(Z)
    print(f"{n_:6s}{d:9.0f}{r:7d}{d/r:10.2f}{cond:12.1f}{approx_err(Z,V):12.4f}",flush=True)
