"""z 的度量到底对齐谁:状态,还是它自己的监督目标?(CPU)
对同一批点对,算 ||Δz|| 与以下各量的秩相关:状态距离、|Δa|、|ΔV|、|ΔR|、|Δ(x,d)|。
如果每个臂都最对齐它自己的目标,那就说明 z 学的是目标的几何,不是状态的几何。
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(6)
M=sys.argv[1]; N=3000
DATA=Path(f"./data_wk_{M}"); ROOT=Path(f"./models/vision/wk_{M}")
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
rg_=np.random.default_rng(0); idx=rg_.choice(np.flatnonzero(seen),N,replace=False)
frames=fr.load_frames(tr["img"][idx],DATA)
px=tr["px_x"][idx].astype(np.float64); d=tr["dist"][idx].astype(np.float64)
fx=112/np.tan(np.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng_m=d/np.cos(np.radians(bear))
a=np.clip(tr["action"][idx],-1,1).astype(np.float64); V=tr["value"][idx].astype(np.float64); R=tr["reward"][idx].astype(np.float64)
S=np.c_[bear/8.5,(rng_m-1.5)/0.5]; XD=np.c_[px/112-1,d/8]
ii,jj=np.triu_indices(N,1); s=rg_.choice(len(ii),1_500_000,replace=False); ii,jj=ii[s],jj[s]
Q={"状态(奖励单位)":np.linalg.norm(S[ii]-S[jj],axis=1),"|Δ动作|":np.linalg.norm(a[ii]-a[jj],axis=1),
   "|Δa_x|":np.abs(a[ii,0]-a[jj,0]),"|Δa_w|":np.abs(a[ii,1]-a[jj,1]),
   "|ΔV|":np.abs(V[ii]-V[jj]),"|ΔR|":np.abs(R[ii]-R[jj]),"|Δ(x,d)|":np.linalg.norm(XD[ii]-XD[jj],axis=1)}
def rk(v): return np.argsort(np.argsort(v)).astype(np.float32)
Qr={k:rk(v) for k,v in Q.items()}
def sp(x,y):
    x=x-x.mean(); y=y-y.mean(); return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()))
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return np.concatenate([e(frames[i:i+64])["shared_feature"].float().numpy() for i in range(0,len(frames),64)]).astype(np.float64)
print(f"== {M} == {N} 帧 {len(ii)} 对。行=表征,列=ρ(||Δz||, 该量)。粗体应出现在各臂自己的目标上。",flush=True)
print(f"{'表征':10s}"+"".join(f"{k:>14s}" for k in Q),flush=True)
for n_,p in [("init",ROOT/"encoder_init.pt")]+[(x,ROOT/x/"donor.pt") for x in ("A","V","R","sup")]:
    if not p.exists(): continue
    Z=enc(p); dz=rk(np.linalg.norm(Z[ii]-Z[jj],axis=1))
    print(f"{n_:10s}"+"".join(f"{sp(dz,Qr[k]):14.3f}" for k in Q),flush=True)
