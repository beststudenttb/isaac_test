"""探针会不会过拟合 / 有没有偏好(CPU)。
对每个表征:线性探针的 train vs held R²、打乱标签的对照、样本量扫描、以及单轴探针的"选最大"偏差。
用法: CUDA_VISIBLE_DEVICES="" python g4_overfit.py <OBS> <RM>
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(10)
O,RM=sys.argv[1],sys.argv[2]; N=3000
DATA=Path(f"./data_g4_{O}_{RM}"); ROOT=Path(f"./models/vision/g4_{O}_{RM}")
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
m_all=fr.episode_split(tr["episode"],seed=0); rg=np.random.default_rng(0)
idx=np.concatenate([rg.choice(np.flatnonzero(m_all&seen),N,replace=False), rg.choice(np.flatnonzero((~m_all)&seen),N,replace=False)])
m=np.r_[np.ones(N,bool),np.zeros(N,bool)]
frames=fr.load_frames(tr["img"][idx],DATA)
px=tr["px_x"][idx].astype(np.float64); d=tr["dist"][idx].astype(np.float64)
fx=112/np.tan(np.radians(40)); bear=np.degrees(np.arctan((112-px)/fx))
Y={"x_c":px,"方位角":bear,"打乱的x_c":rg.permutation(px)}
def ridge(F,y,mask,lam):
    mu,sd=F[mask].mean(0),F[mask].std(0)+1e-9
    a=np.hstack([(F[mask]-mu)/sd,np.ones((mask.sum(),1))]); b=np.hstack([(F[~mask]-mu)/sd,np.ones(((~mask).sum(),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@y[mask],rcond=None)[0]
    r2=lambda p,t:float(1-((t-p)**2).sum()/((t-t.mean())**2).sum())
    return r2(a@w,y[mask]), r2(b@w,y[~mask])
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return np.concatenate([e(frames[i:i+64])["shared_feature"].float().numpy() for i in range(0,len(frames),64)]).astype(np.float64)
print(f"== {O}_{RM} ==  每格 train/held", flush=True)
print(f"{'表征':6s}{'x_c λ=1e-2':>16s}{'x_c λ=1e-4':>16s}{'打乱标签':>14s}{'N=500':>12s}{'N=3000':>12s}{'单轴最大':>10s}{'单轴中位':>10s}", flush=True)
for n_,p in [("init",ROOT/"encoder_init.pt")]+[(a,ROOT/a/"donor.pt") for a in ("A","V","R","sup")]:
    if not p.exists(): continue
    Z=enc(p)
    t2,h2=ridge(Z,Y["x_c"],m,1e-2); t4,h4=ridge(Z,Y["x_c"],m,1e-4)
    ts,hs=ridge(Z,Y["打乱的x_c"],m,1e-2)
    sm=m.copy(); sm[:]=False; sm[rg.choice(np.flatnonzero(m),500,replace=False)]=True
    _,h500=ridge(Z,Y["x_c"],sm,1e-2)
    ax=sorted(ridge(Z[:,[j]],Y["x_c"],m,1e-2)[1] for j in range(Z.shape[1]))
    print(f"{n_:6s}{t2:8.3f}/{h2:<7.3f}{t4:8.3f}/{h4:<7.3f}{ts:7.3f}/{hs:<6.3f}{h500:12.3f}{h2:12.3f}{ax[-1]:10.3f}{ax[len(ax)//2]:10.3f}", flush=True)
