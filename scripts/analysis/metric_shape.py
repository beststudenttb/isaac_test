"""z 的距离和状态距离一致吗(CPU)。
状态用奖励自己的单位:b = 方位角/8.5°,r = (距离-1.5)/0.5m。局部拟合 ||Δz||² = a·Δb² + c·Δr²,
sqrt(a)、sqrt(c) = z 空间每一个"奖励单位"被拉伸多少;按距离分段看放大倍率的分布。
用法: python metric_shape.py <mask> [N]
"""
import sys, glob, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(8)
M=sys.argv[1]; MASK="" if M=="xd" else M; N=int(sys.argv[2]) if len(sys.argv)>2 else 3000
DATA=Path(f"./data_wk_{M}"); ROOT=Path(f"./models/vision/wk_{M}")
TZIP=sorted(glob.glob(f"models/rl/teacher_score_wk_{M}_s*/last.zip"))[-1]
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
rng_=np.random.default_rng(0); idx=rng_.choice(np.flatnonzero(seen),N,replace=False)
frames=fr.load_frames(tr["img"][idx],DATA)
px=tr["px_x"][idx].astype(np.float64); d=tr["dist"][idx].astype(np.float64)
fx=112/np.tan(np.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rg=d/np.cos(np.radians(bear))
b=bear/8.5; r=(rg-1.5)/0.5                       # 奖励自己的单位
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return np.concatenate([e(frames[i:i+64])["shared_feature"].float().numpy() for i in range(0,len(frames),64)]).astype(np.float64)
ii,jj=np.triu_indices(N,1)
sub=rng_.choice(len(ii),min(len(ii),3_000_000),replace=False); ii,jj=ii[sub],jj[sub]
db=b[ii]-b[jj]; dr=r[ii]-r[jj]; ds=np.sqrt(db**2+dr**2)
near=ds<0.5
rmid=0.5*(rg[ii]+rg[jj])
BINS=[(1.3,1.7),(1.7,2.5),(2.5,4.0),(4.0,9.0)]
def rk(v): return np.argsort(np.argsort(v)).astype(float)
print(f"== {M} ==  {N} 帧,{len(ii)} 对,其中局部对(状态距<0.5 单位) {near.sum()}", flush=True)
print(f"{'表征':11s}{'全局ρ(|Δz|,|ΔS|)':>16s} | 每奖励单位的拉伸 sqrt(a)=方位角 sqrt(c)=距离,按距离分段", flush=True)
print(f"{'':11s}{'':16s} | {'1.3-1.7m':>16s}{'1.7-2.5m':>16s}{'2.5-4m':>16s}{'>4m':>16s}", flush=True)
def rep(name,Z):
    dz=np.linalg.norm(Z[ii]-Z[jj],axis=1); dz/= np.sqrt((dz**2).mean())   # 归一化掉各臂的总体尺度
    g=float(np.corrcoef(rk(dz),rk(ds))[0,1]); out=[]
    for lo,hi in BINS:
        s=near&(rmid>=lo)&(rmid<hi)
        if s.sum()<300: out.append((float('nan'),float('nan'),s.sum())); continue
        A=np.c_[db[s]**2, dr[s]**2]; y=dz[s]**2
        w=np.linalg.lstsq(A,y,rcond=None)[0]
        out.append((np.sqrt(max(w[0],0)),np.sqrt(max(w[1],0)),s.sum()))
    cells="".join(f"{o[0]:7.2f}/{o[1]:<8.2f}" for o in out)
    print(f"{name:11s}{g:16.3f} | {cells}", flush=True)
    return [o[2] for o in out]
cnt=None
for n_,p in [("init",ROOT/"encoder_init.pt")]+[(a,ROOT/a/"donor.pt") for a in ("A","V","R","sup")]:
    if p.exists(): cnt=rep(n_,enc(p))
from stable_baselines3 import PPO
pol=PPO.load(TZIP,device="cpu").policy
r_=31.06/d; cy=76.0+83.1/d; hf=112.0
full={"":np.stack([px/hf-1,d/8,np.full_like(d,1.5/8),np.zeros_like(d)],1),
      "diam":np.stack([px/hf-1,np.clip(2*r_/224,0,1),np.full_like(d,1.5/8),np.zeros_like(d)],1),
      "xyd":np.stack([px/hf-1,np.clip(2*r_/224,0,1),np.full_like(d,1.5/8),cy/hf-1],1),
      "bbox":np.stack([(px-r_)/hf-1,(px+r_)/hf-1,(cy-r_)/hf-1,(cy+r_)/hf-1],1).clip(-1,1),
      "ang":np.stack([np.clip(bear/40,-1,1),np.clip(rg/8,0,1),np.full_like(d,1.5/8),np.zeros_like(d)],1)}[MASK].astype(np.float32)
with torch.no_grad(): rep("teacher隐2",pol.mlp_extractor.policy_net(torch.as_tensor(full)).numpy().astype(np.float64))
rep("特权obs",full.astype(np.float64))
print(f"{'':11s}各段局部对数: {cnt}", flush=True)
