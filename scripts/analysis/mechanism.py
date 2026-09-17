"""机制链的两个自变量,四种配置 × 四个臂:
(a) 目标在**随机初值编码器**上的线性可读性(= 这个目标有多难,梯度要出多大力)
(b) donor 训完后各层权重被拖走的相对距离 ‖ΔW‖/‖W_init‖
因变量在别处:蓝球在 z 层还剩多少(ccgp_all)、换任务的 PPO 成绩。CPU。
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(6)
N=3000
GROUPS=[("stem",["stem.0."]),("c2",["layer1."]),("c3",["layer2."]),("c4",["layer3."]),
        ("FPN",["lat2.","lat3.","lat4.","smooth2.","smooth3.","smooth4."]),("proj",["proj."])]
torch.manual_seed(1)
INIT=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).state_dict()
def dW(sd,pref):
    num=den=0.0
    for k,v in sd.items():
        if not any(k.startswith(p) for p in pref): continue
        if "running_" in k or "num_batches" in k: continue
        d=(v.float()-INIT[k].float()); num+=float((d**2).sum()); den+=float((INIT[k].float()**2).sum())
    return (num/den)**0.5
def ridge(F,y,m,lam=1e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    A=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[~m]
    return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
@torch.no_grad()
def enc(p,frames):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(frames[i:i+64])["shared_feature"].float() for i in range(0,len(frames),64)]).numpy().astype(np.float64)
print(f"{'配置':12s}{'臂':5s}{'目标在随机编码器上的线性R²':>28s}{'  proj ΔW':>10s}{'stem ΔW':>9s}{'c2':>7s}{'c3':>7s}{'c4':>7s}",flush=True)
for cfg in ("diam_angle","diam_cam","ang_angle","ang_cam"):
    ROOT=Path(f"models/vision/g4_{cfg}"); DATA=Path(f"./data_g4_{cfg}")
    rg=np.random.default_rng(0)
    tr=fr.load_transitions(DATA,units=1)
    ok=(tr["dist"]>0.3)&(tr["done"]<0.5)
    m_all=fr.episode_split(tr["episode"],seed=0)
    idx=np.concatenate([rg.choice(np.flatnonzero(m_all&ok),N,replace=False),
                        rg.choice(np.flatnonzero((~m_all)&ok),N,replace=False)])
    m=np.r_[np.ones(N,bool),np.zeros(N,bool)]
    px,d=tr["px_x"][idx].astype(np.float64),tr["dist"][idx].astype(np.float64)
    diam=np.clip(62.12/np.clip(d,0.3,None)/224.0,0,1)
    TGT={"A":np.clip(tr["mu"][idx],-1,1).astype(np.float64),
         "V":tr["value"][idx][:,None].astype(np.float64),
         "R":tr["reward"][idx][:,None].astype(np.float64),
         "sup":np.stack([px/112.0-1.0,diam],1)}
    Z0=enc(ROOT/"encoder_init.pt",fr.load_frames(tr["img"][idx],DATA))
    for arm in ("A","V","R","sup"):
        y=TGT[arm]
        r2=float(np.mean([ridge(Z0,y[:,j],m) for j in range(y.shape[1])]))   # 多维目标取各维平均
        sd=torch.load(ROOT/arm/"donor.pt",map_location="cpu")
        v=[dW(sd,p) for _,p in GROUPS]
        print(f"{cfg:12s}{arm:5s}{r2:28.3f}{v[5]:10.3f}{v[0]:9.3f}{v[1]:7.3f}{v[2]:7.3f}{v[3]:7.3f}",flush=True)
    del tr
