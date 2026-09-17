"""交叉探针矩阵:把 Light-weight probing(RLC 2024)的协议反过来用。
他们问"表征里能不能读出奖励与专家动作";我们问"用某个信号训出的表征,让哪些 RL 量变得容易读"。
行=编码器(init/A/V/R/sup),列=读出目标(teacher 动作两维、V、R)。held 按回合切分。CPU。
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(10)
ROOT=Path("models/vision/g4_diam_angle"); DATA=Path("./data_g4_diam_angle")
N=3000; rg=np.random.default_rng(0)
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
m_all=fr.episode_split(tr["episode"],seed=0)
idx=np.concatenate([rg.choice(np.flatnonzero(m_all&seen),N,replace=False), rg.choice(np.flatnonzero((~m_all)&seen),N,replace=False)])
m=np.r_[np.ones(N,bool),np.zeros(N,bool)]
frames=fr.load_frames(tr["img"][idx],DATA)
a=np.clip(tr["mu"][idx],-1,1).astype(np.float64)
T={"动作 a_x":a[:,0],"动作 a_w":a[:,1],"价值 V":tr["value"][idx].astype(np.float64),"奖励 R":tr["reward"][idx].astype(np.float64)}
def ridge(F,y,lam=1e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    A=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[~m]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
def sample_eff(F,y,n):
    sm=np.zeros(len(y),bool); sm[rg.choice(np.flatnonzero(m),n,replace=False)]=True
    mu,sd=F[sm].mean(0),F[sm].std(0)+1e-9
    A=np.hstack([(F[sm]-mu)/sd,np.ones((sm.sum(),1))]); B=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(A.T@A+1e-2*len(A)*np.eye(A.shape[1]),A.T@y[sm],rcond=None)[0]
    p=B@w; yy=y[~m]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(frames[i:i+64])["shared_feature"].float() for i in range(0,len(frames),64)]).numpy().astype(np.float64)
print("交叉探针(held R²);对角线由构造决定,看的是非对角\n",flush=True)
print(f"{'编码器':8s}"+"".join(f"{k:>12s}" for k in T),flush=True)
Z={}
for n_,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Z[n_]=enc(p); print(f"{n_:8s}"+"".join(f"{ridge(Z[n_],y):12.3f}" for y in T.values()),flush=True)
print("\n样本效率(只用 200 个训练样本,考察可访问性而非是否存在)",flush=True)
print(f"{'编码器':8s}"+"".join(f"{k:>12s}" for k in T),flush=True)
for n_ in Z:
    print(f"{n_:8s}"+"".join(f"{sample_eff(Z[n_],y,200):12.3f}" for y in T.values()),flush=True)
