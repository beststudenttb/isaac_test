"""表征里有没有 teacher 之外的东西(CPU,不碰 GPU)。
两个方向:
  正向 R²(teacher观测 | z)  —— 表征装下了 teacher 的信息吗
  反向 z 的方差里,能被 teacher 观测解释掉多少;残差再看能不能被蓝球/完整状态解释
参照:teacher 自己的隐层2 是 teacher 观测的确定性函数,反向残差应当≈0。
用法: python residual.py <mask>
"""
import sys, glob, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(16); DEV="cpu"
M=sys.argv[1]; MASK="" if M=="xd" else M
DATA=Path(f"./data_wk_{M}"); ROOT=Path(f"./models/vision/wk_{M}")
TZIP=sorted(glob.glob(f"models/rl/teacher_score_wk_{M}_s*/last.zip"))[-1]
N=2500; torch.manual_seed(0)
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
m_all=fr.episode_split(tr["episode"],seed=0); rng=np.random.default_rng(0)
idx=np.concatenate([rng.choice(np.flatnonzero(m_all&seen),N,replace=False),
                    rng.choice(np.flatnonzero((~m_all)&seen),N,replace=False)])
m=np.r_[np.ones(N,bool),np.zeros(N,bool)]
frames=fr.load_frames(tr["img"][idx],DATA)
px=tr["px_x"][idx].astype(np.float64); d=tr["dist"][idx].astype(np.float64)
r=31.06/d; cy=76.0+83.1/d; fx=112/np.tan(np.radians(40))
bear=np.degrees(np.arctan((112-px)/fx)); rg=d/np.cos(np.radians(bear)); hf=112.0
T={"":np.stack([px/hf-1,d/8,np.full_like(d,1.5/8),np.zeros_like(d)],1),
   "diam":np.stack([px/hf-1,np.clip(2*r/224,0,1),np.full_like(d,1.5/8),np.zeros_like(d)],1),
   "xyd":np.stack([px/hf-1,np.clip(2*r/224,0,1),np.full_like(d,1.5/8),cy/hf-1],1),
   "bbox":np.stack([(px-r)/hf-1,(px+r)/hf-1,(cy-r)/hf-1,(cy+r)/hf-1],1).clip(-1,1),
   "ang":np.stack([np.clip(bear/40,-1,1),np.clip(rg/8,0,1),np.full_like(d,1.5/8),np.zeros_like(d)],1)}[MASK]
T=T[:,:2] if MASK in ("","diam","ang") else T     # 后两维是常量 goal / 与 mask 有关,只留真正变化的
f=frames.float(); R_,G_,B_=f[:,0],f[:,1],f[:,2]
blue=(B_>90)&(B_>R_*1.8)&(B_>G_*1.8); bsz=blue.flatten(1).sum(1).float().numpy().astype(np.float64)
xs=torch.arange(224).view(1,1,224).expand(len(f),224,224).float()
ys=torch.arange(224).view(1,224,1).expand(len(f),224,224).float()
den=blue.flatten(1).sum(1).clamp(min=1).float()
bx=((blue.float()*xs).flatten(1).sum(1)/den).numpy().astype(np.float64)
by=((blue.float()*ys).flatten(1).sum(1)/den).numpy().astype(np.float64)
has=(bsz>30).astype(np.float64)
BLUE=np.stack([bx/112-1,by/112-1,np.sqrt(bsz)/30,has],1)
S=np.stack([px/hf-1,d/8],1)          # 完整任务状态(图像里除蓝球外一切都是它的函数)
def fit_mlp(X,Y,hid=128,steps=4000,lr=3e-3):
    xs_,ys_=torch.tensor(X,dtype=torch.float32),torch.tensor(Y,dtype=torch.float32)
    mu,sd=xs_[m].mean(0),xs_[m].std(0)+1e-6; xs_=(xs_-mu)/sd
    net=nn.Sequential(nn.Linear(X.shape[1],hid),nn.Tanh(),nn.Linear(hid,hid),nn.Tanh(),nn.Linear(hid,Y.shape[1]))
    opt=torch.optim.Adam(net.parameters(),lr=lr)
    ti=torch.tensor(np.flatnonzero(m)); 
    for _ in range(steps):
        s=ti[torch.randint(0,len(ti),(512,))]
        loss=((net(xs_[s])-ys_[s])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): p=net(xs_).numpy()
    h=~m; sse=((p[h]-Y[h])**2).sum(0); sst=((Y[h]-Y[h].mean(0))**2).sum(0)
    return float(1-sse.sum()/sst.sum()), (1-sse/np.maximum(sst,1e-12))
def ridge(F,y,lam=1e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9; a=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); b=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@y[m],rcond=None)[0]
    p=b@w; return float(1-((y[~m]-p)**2).sum()/((y[~m]-y[~m].mean(0))**2).sum())
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return np.concatenate([e(frames[i:i+64])["shared_feature"].float().numpy() for i in range(0,len(frames),64)]).astype(np.float64)
print(f"== {M} == teacher 观测 {T.shape[1]} 维  held {N} 帧", flush=True)
print(f"{'表征':12s}{'R²(T|z)线性':>12s}{'R²(T|z)MLP':>11s} | z 的方差被解释掉多少: {'T':>6s}{'T+蓝球':>8s}{'状态+蓝球':>10s}{'残差':>7s}", flush=True)
def rep(name,Z):
    Z=Z.astype(np.float64); f1=ridge(Z,T); f2,_=fit_mlp(Z,T)
    e_T,_=fit_mlp(T,Z); e_TB,_=fit_mlp(np.c_[T,BLUE],Z); e_SB,_=fit_mlp(np.c_[S,BLUE],Z)
    print(f"{name:12s}{f1:12.3f}{f2:11.3f} | {'':21s}{e_T:6.3f}{e_TB:8.3f}{e_SB:10.3f}{1-e_SB:7.3f}", flush=True)
for n_,p in [("init",ROOT/"encoder_init.pt")]+[(a,ROOT/a/"donor.pt") for a in ("A","V","R","sup")]:
    if p.exists(): rep(n_,enc(p))
from stable_baselines3 import PPO
pol=PPO.load(TZIP,device="cpu").policy
full={"":np.stack([px/hf-1,d/8,np.full_like(d,1.5/8),np.zeros_like(d)],1),
      "diam":np.stack([px/hf-1,np.clip(2*r/224,0,1),np.full_like(d,1.5/8),np.zeros_like(d)],1),
      "xyd":np.stack([px/hf-1,np.clip(2*r/224,0,1),np.full_like(d,1.5/8),cy/hf-1],1),
      "bbox":np.stack([(px-r)/hf-1,(px+r)/hf-1,(cy-r)/hf-1,(cy+r)/hf-1],1).clip(-1,1),
      "ang":np.stack([np.clip(bear/40,-1,1),np.clip(rg/8,0,1),np.full_like(d,1.5/8),np.zeros_like(d)],1)}[MASK]
with torch.no_grad(): h2=pol.mlp_extractor.policy_net(torch.as_tensor(full,dtype=torch.float32)).numpy().astype(np.float64)
rep("teacher隐2",h2)
