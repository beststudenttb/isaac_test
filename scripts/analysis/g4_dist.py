"""数据里各量的分布 vs 探针的 R²,以及换成绝对误差后排名变不变(CPU)。
用法: CUDA_VISIBLE_DEVICES="" python g4_dist.py <OBS> <RM>
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
fx=112/np.tan(np.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng_m=d/np.cos(np.radians(bear)); diam=62.12/d
Q={"x_c(px)":px,"直径(px)":diam,"方位角(deg)":bear,"距离(m)":rng_m}
near=(rng_m>=1.3)&(rng_m<=1.7)
print(f"== {O}_{RM} ==  全体 {len(px)} 帧,其中停车区(1.3–1.7m) {near.sum()} 帧({near.mean():.0%})", flush=True)
print(f"{'量':12s}{'标准差':>9s}{'p1':>9s}{'p50':>9s}{'p99':>9s}{'停车区内标准差':>14s}{'区内/全体':>10s}", flush=True)
for k,y in Q.items():
    p=np.percentile(y,[1,50,99])
    print(f"{k:12s}{y.std():9.3f}{p[0]:9.3f}{p[1]:9.3f}{p[2]:9.3f}{y[near].std():14.3f}{y[near].std()/y.std():10.2f}", flush=True)
def ridge_pred(F,y,lam=1e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    a=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); b=np.hstack([(F-mu)/sd,np.ones((len(F),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@y[m],rcond=None)[0]
    return b@w
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return np.concatenate([e(frames[i:i+64])["shared_feature"].float().numpy() for i in range(0,len(frames),64)]).astype(np.float64)
print(f"\n线性探针的**绝对误差**(held,原单位)。括号里是 R²。", flush=True)
print(f"{'表征':6s}{'x_c 全体 RMSE(px)':>20s}{'x_c 停车区 RMSE':>18s}{'距离 全体 RMSE(m)':>19s}{'距离 停车区 RMSE':>18s}", flush=True)
for n_,p in [("init",ROOT/"encoder_init.pt")]+[(a,ROOT/a/"donor.pt") for a in ("A","V","R","sup")]:
    if not p.exists(): continue
    Z=enc(p); out=[]
    for k in ("x_c(px)","距离(m)"):
        y=Q[k]; pr=ridge_pred(Z,y); h=~m; hn=h&near
        rmse=np.sqrt(((pr[h]-y[h])**2).mean()); r2=1-((pr[h]-y[h])**2).sum()/((y[h]-y[h].mean())**2).sum()
        rmse_n=np.sqrt(((pr[hn]-y[hn])**2).mean()); r2_n=1-((pr[hn]-y[hn])**2).sum()/((y[hn]-y[hn].mean())**2).sum()
        out += [f"{rmse:7.2f}({r2:5.2f})", f"{rmse_n:7.2f}({r2_n:6.2f})"]
    print(f"{n_:6s}"+"".join(f"{c:>20s}" for c in out), flush=True)
