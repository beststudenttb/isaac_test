"""Wang et al. 的四个可算性质,推广到四种 teacher 配置 × 五个表征。价值取各自 teacher 的 V。
complexity reduction 的归一化在**每个配置内部**做(和 Wang 一样,按同一批被比较的表征取 L_max)。CPU。
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(8)
N=2000
print(f"{'配置':14s}{'表征':6s}{'复杂度降低':>12s}{'动力学感知':>12s}{'多样性':>10s}{'正交性':>10s}",flush=True)
@torch.no_grad()
def enc(p,frames):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(frames[i:i+64])["shared_feature"].float() for i in range(0,len(frames),64)]).numpy().astype(np.float64)
for cfg in ("diam_angle","diam_cam","ang_angle","ang_cam"):
    ROOT=Path(f"models/vision/g4_{cfg}"); DATA=Path(f"./data_g4_{cfg}")
    rg=np.random.default_rng(0)
    tr=fr.load_transitions(DATA,units=1)
    ok=(tr["dist"]>0.3)&(tr["done"]<0.5)
    idx=rg.choice(np.flatnonzero(ok),N,replace=False)
    V=tr["value"][idx].astype(np.float64)
    F0=fr.load_frames(tr["img"][idx],DATA); F1=fr.load_frames(tr["next_img"][idx],DATA)
    iu=np.triu_indices(N,1); dv=np.abs(V[:,None]-V[None,:])[iu]
    res={}
    for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
        p=ROOT/rel
        if not p.exists(): continue
        Z=enc(p,F0); Zn=enc(p,F1)
        G=Z@Z.T; s=np.diag(G); dsm=np.sqrt(np.maximum(s[:,None]+s[None,:]-2*G,0))[iu]
        L=float((dv/(dsm+1e-12)).mean())
        perm=rg.permutation(N)
        dr=np.linalg.norm(Z-Z[perm],axis=1); dsucc=np.linalg.norm(Z-Zn,axis=1)
        da=float((dr.sum()-dsucc.sum())/dr.sum())
        div=1.0-float(np.minimum((dv/dv.max())/((dsm/dsm.max())+1e-2),1.0).mean())
        nz=Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-12)
        orth=1.0-float(np.abs(nz@nz.T)[iu].mean())
        res[arm]=(L,da,div,orth)
    Lmax=max(v[0] for v in res.values())
    for arm,(L,da,div,orth) in res.items():
        print(f"{cfg:14s}{arm:6s}{1-L/Lmax:12.3f}{da:12.3f}{div:10.3f}{orth:10.3f}",flush=True)
    del F0,F1,tr
