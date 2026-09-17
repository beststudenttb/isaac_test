"""监督信号"要求区分什么",就在表征里留下什么(CPU,只用 teacher 信号,不用 x/d)。
在同一批观测上找两类样本对:
  S1: 奖励与价值都几乎相同,但动作差别大   -> 拟合 A 必须分开它们;拟合 V/R 不必
  S2: 动作几乎相同,但奖励或价值差别大     -> 拟合 V/R 必须分开;拟合 A 不必
看 A/V/R/init 的 z 分别把哪一类分得更开。
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(12)
COMB="diam_angle"; ROOT=Path(f"models/vision/g4_{COMB}"); DATA=Path(f"./data_g4_{COMB}")
N=4000; rg=np.random.default_rng(0)
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
idx=rg.choice(np.flatnonzero(seen),N,replace=False)
frames=fr.load_frames(tr["img"][idx],DATA)
a=np.clip(tr["mu"][idx],-1,1).astype(np.float64); V=tr["value"][idx].astype(np.float64); R=tr["reward"][idx].astype(np.float64)
ii,jj=np.triu_indices(N,1); s=rg.choice(len(ii),1_500_000,replace=False); ii,jj=ii[s],jj[s]
dax=np.abs(a[ii,0]-a[jj,0])/a[:,0].std()   # 前进维:主要跟距离走
daw=np.abs(a[ii,1]-a[jj,1])/a[:,1].std()   # 转向维:唯一带方位角的通道
da =np.maximum(dax,daw)
dv=np.abs(V[ii]-V[jj])/V.std(); dr=np.abs(R[ii]-R[jj])/R.std()
q=lambda x,p: np.quantile(x,p)
# S1:价值与奖励都几乎相同(即距离相同),但转向动作差别大 -> 只有拟合 A 必须分开
S1=(dv<q(dv,.20))&(dr<q(dr,.20))&(daw>q(daw,.80))
# S2:两维动作都几乎相同,但价值或奖励差别大 -> 只有拟合 V/R 必须分开
S2=(dax<q(dax,.20))&(daw<q(daw,.20))&((dv>q(dv,.80))|(dr>q(dr,.80)))
print(f"S1(奖励价值同、动作不同) {S1.sum()} 对;S2(动作同、奖励价值不同) {S2.sum()} 对",flush=True)
print(f"  S1 上:|Δa_w| 中位 {np.median(daw[S1]):.2f},|Δa_x| {np.median(dax[S1]):.3f},|ΔV| {np.median(dv[S1]):.3f},|ΔR| {np.median(dr[S1]):.3f}",flush=True)
print(f"  S2 上:|Δa_w| 中位 {np.median(daw[S2]):.3f},|Δa_x| {np.median(dax[S2]):.3f},|ΔV| {np.median(dv[S2]):.2f},|ΔR| {np.median(dr[S2]):.2f}\n",flush=True)
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(frames[i:i+64])["shared_feature"].float() for i in range(0,len(frames),64)]).numpy().astype(np.float64)
print(f"{'表征':7s}{'S1 上的 |Δz|':>13s}{'S2 上的 |Δz|':>13s}{'S1/S2':>8s}{'(全体均值=1 归一)':>18s}",flush=True)
for n_,rel in [("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("init","encoder_init.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p); Zs=(Z-Z.mean(0))/(Z.std(0)+1e-9)
    dz=np.linalg.norm(Zs[ii]-Zs[jj],axis=1); dz=dz/dz.mean()
    print(f"{n_:7s}{dz[S1].mean():13.3f}{dz[S2].mean():13.3f}{dz[S1].mean()/dz[S2].mean():8.2f}",flush=True)
