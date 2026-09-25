"""机制拆分臂的三项测量,和已有的 A/V/R/sup/init 并排:
  1) 目标在**随机初值编码器**上的线性可读性(= 有多难提取)
  2) proj 层权重被拖走的相对距离 ‖ΔW‖/‖W_init‖
  3) 干扰物(蓝球)在 z 里还剩多少 + 干扰物造成的 z 位移
CPU。
"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
import free_repr as fr
torch.set_num_threads(10)
ROOT=Path("models/vision/g4_diam_angle"); DATA=Path("./data_g4_diam_angle"); DD=Path("data_distract")
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("Asym","Asym/donor.pt"),("Aw","Aw/donor.pt"),
      ("Ax","Ax/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
torch.manual_seed(1)
INIT=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).state_dict()
def dW(sd,pref=("proj.",)):
    num=den=0.0
    for k,v in sd.items():
        if not any(k.startswith(p) for p in pref) or "running_" in k or "num_batches" in k: continue
        d=(v.float()-INIT[k].float()); num+=float((d**2).sum()); den+=float((INIT[k].float()**2).sum())
    return (num/den)**0.5
@torch.no_grad()
def enc(p,frames,bs=64):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(frames[i:i+bs])["shared_feature"].float() for i in range(0,len(frames),bs)]).numpy().astype(np.float64)
def ridge(F,y,m,lam=3e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    A=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[~m]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
# --- 目标在随机先验上的可读性 ---
N=3000; rg=np.random.default_rng(0)
tr=fr.load_transitions(DATA,units=1); ok=(tr["dist"]>0.3)&(tr["done"]<0.5)
m_all=fr.episode_split(tr["episode"],seed=0)
idx=np.concatenate([rg.choice(np.flatnonzero(m_all&ok),N,replace=False),
                    rg.choice(np.flatnonzero((~m_all)&ok),N,replace=False)])
m=np.r_[np.ones(N,bool),np.zeros(N,bool)]
mu_a=np.clip(tr["mu"][idx],-1,1).astype(np.float64)
TGT={"A":mu_a,"Asym":np.stack([mu_a[:,0],np.abs(mu_a[:,1])],1),"Aw":mu_a[:,1:2],"Ax":mu_a[:,0:1],
     "V":tr["value"][idx][:,None].astype(np.float64),"R":tr["reward"][idx][:,None].astype(np.float64)}
Z0=enc(ROOT/"encoder_init.pt",fr.load_frames(tr["img"][idx],DATA))
prior={k:float(np.mean([ridge(Z0,y[:,j],m) for j in range(y.shape[1])])) for k,y in TGT.items()}
del tr
# --- 干扰物残留 / 位移(受控渲染)---
rows=list(csv.DictReader(open(DD/"meta.csv")))
cond=np.array([r["cond"] for r in rows]); rd=np.array([int(r["round"]) for r in rows])
key=np.array([f"{r['round']}_{r['env']}" for r in rows]); b2=np.array([float(r["bear2"]) for r in rows])
keep=np.flatnonzero((cond=="base")|(cond=="d_none"))
FR=torch.stack([read_image(str(DD/rows[j]["img"])) for j in keep])
sub=[rows[j] for j in keep]; c2=cond[keep]; rd2=rd[keep]; k2=key[keep]; bb=b2[keep]
mm=(c2=="base")&(rd2%2==0); hh=(c2=="base")&(rd2%2==1)
print(f"{'臂':7s}{'目标随机先验可读性':>20s}{'proj ΔW':>10s}{'干扰物残留 R²':>15s}{'干扰物位移':>12s}",flush=True)
for arm,rel in ARMS:
    p=ROOT/rel
    if not p.exists(): print(f"{arm:7s} 缺 {rel}"); continue
    Z=enc(p,FR)
    msk=np.zeros(len(Z),bool); msk[np.flatnonzero(mm)]=True
    r2=ridge(Z[(c2=="base")],bb[(c2=="base")],(rd2[(c2=="base")]%2==0))
    bi={k2[i]:i for i in np.flatnonzero(c2=="d_none")}
    ii=[i for i in np.flatnonzero(c2=="base") if k2[i] in bi]
    shift=float(np.mean(np.linalg.norm(Z[ii]-Z[[bi[k2[i]] for i in ii]],axis=1)/
                        (np.linalg.norm(Z[[bi[k2[i]] for i in ii]],axis=1)+1e-9)))
    pr=prior.get(arm,float("nan"))
    w=dW(torch.load(p,map_location="cpu")) if arm!="init" else 0.0
    print(f"{arm:7s}{pr:20.3f}{w:10.3f}{r2:15.3f}{shift:12.3f}",flush=True)
print("\n读法:若 Asym/Ax(对称、好提取)像 R 一样留住干扰物,而 Aw(反对称、难提取)像 A 一样擦掉,"
      "则主因是**对称性/可提取性**,不是目标维数。",flush=True)
