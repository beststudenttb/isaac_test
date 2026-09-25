"""Wang et al. 2024 (AIJ, arXiv 2203.15955) 的表征性质度量,原样搬到我们的冻结表征上。
他们的设定和我们几乎一样:训练任务上学表征 -> 冻结 -> 迁移到相关任务;他们的结论是
"迁移最好的表征:complexity reduction 高、dynamics awareness / diversity 中高、
orthogonality / sparsity 中等"。我们正好有 红球PPO / 蓝球PPO 两个成绩可以做因变量。
价值函数取 teacher 的 V(s)(这就是本任务的价值函数)。CPU。
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(10)
ROOT=Path("models/vision/g4_diam_angle"); DATA=Path("./data_g4_diam_angle")
N=2500; rg=np.random.default_rng(7)
tr=fr.load_transitions(DATA,units=1)
ok=(tr["dist"]>0.3)&(tr["done"]<0.5)
idx=rg.choice(np.flatnonzero(ok),N,replace=False)
V=tr["value"][idx].astype(np.float64)
F0=fr.load_frames(tr["img"][idx],DATA); F1=fr.load_frames(tr["next_img"][idx],DATA)

@torch.no_grad()
def enc(p,frames):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(frames[i:i+64])["shared_feature"].float() for i in range(0,len(frames),64)]).numpy().astype(np.float64)

iu=np.triu_indices(N,1)
dv=np.abs(V[:,None]-V[None,:])[iu]
def props(Z,Zn):
    d=np.sqrt(np.maximum(((Z[:,None,:]-Z[None,:,:])**2).sum(-1),0))[iu] if N<=1500 else None
    if d is None:  # 省内存:用 |a|^2+|b|^2-2ab
        G=Z@Z.T; s=np.diag(G); d=np.sqrt(np.maximum(s[:,None]+s[None,:]-2*G,0))[iu]
    Lrep=float((dv/(d+1e-12)).mean())
    # dynamics awareness:后继 vs 随机
    perm=rg.permutation(N)
    d_rand=np.linalg.norm(Z-Z[perm],axis=1); d_succ=np.linalg.norm(Z-Zn,axis=1)
    da=float((d_rand.sum()-d_succ.sum())/d_rand.sum())
    # diversity
    div=1.0-float(np.minimum((dv/dv.max())/((d/d.max())+1e-2),1.0).mean())
    # orthogonality
    nz=Z/ (np.linalg.norm(Z,axis=1,keepdims=True)+1e-12)
    C=np.abs(nz@nz.T)[iu]; orth=1.0-float(C.mean())
    sp=float((np.abs(Z)<1e-10).mean())
    return Lrep,da,div,orth,sp,float(d.mean())

rows={}
for n_,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): print("missing",p); continue
    rows[n_]=props(enc(p,F0),enc(p,F1)); print("done",n_,flush=True)
Lmax=max(r[0] for r in rows.values())
print("\nWang et al. 2024 表征性质(g4_diam_angle,N=2500,价值=teacher V)")
print(f"{'表征':6s}{'复杂度降低':>12s}{'动力学感知':>12s}{'多样性':>10s}{'正交性':>10s}{'稀疏度':>10s}{'平均距离':>10s}")
for n_,(L,da,div,orth,sp,dm) in rows.items():
    print(f"{n_:6s}{1-L/Lmax:12.3f}{da:12.3f}{div:10.3f}{orth:10.3f}{sp:10.3f}{dm:10.2f}")
