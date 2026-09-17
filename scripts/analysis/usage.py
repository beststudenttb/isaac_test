"""Usage 轴(五轴里唯一还空着的那格):表征里编码了某个量,策略到底用没用它。
理论对应 Mante et al. Nature 2013:task-relevant 与 task-irrelevant 都存在于混合活动里,
真正的任务选择发生在读出端。做法参考 TCAV / active subspace:
  1. 在受控析因渲染上,用岭回归拿到 z 空间里的"红球方位方向" u_red 与"蓝球方位方向" u_blue(单位向量)
  2. 取训好的冻结 PPO actor(纯 MLP on [z, goal]),算雅可比 J = ∂a/∂z
  3. 报 ||J·u_red|| 与 ||J·u_blue||:沿这个方向挪动 z 一个单位,动作变多少
  4. 再算活跃子空间 C = E[JᵀJ] 的主方向,看它和 u_red / u_blue 的夹角
goal 在本实验里是常量 (1.5/8.0, 0.0)。纯 CPU。
"""
import sys, csv, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(6)
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_factorial")
GOAL=torch.tensor([[1.5/8.0,0.0]],dtype=torch.float32)
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx))
Y={"red":bear,"blue":g("blue_bear")}
print("解码析因帧...",flush=True)
FR=torch.stack([read_image(str(D/r["img"])) for r in rows])
@torch.no_grad()
def enc(p,bs=48):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(FR[i:i+bs])["shared_feature"].float() for i in range(0,len(FR),bs)]).numpy().astype(np.float64)
def ridge_dir(Z,y,m,lam=3e-2):
    mu,sd=Z[m].mean(0),Z[m].std(0)+1e-9
    A=np.hstack([(Z[m]-mu)/sd,np.ones((m.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0][:-1]
    return (w/sd)/np.linalg.norm(w/sd)          # 折回原始 z 坐标系,再单位化
def load_actor(run):
    p=Path(f"models/rl/{run}/last.pt")
    if not p.exists(): return None
    sd=torch.load(p,map_location="cpu")["model"]
    w=[(k,v) for k,v in sd.items() if k.startswith("actor.")]
    layers=[];  dims=sorted({int(k.split(".")[1]) for k,_ in w})
    for i in dims:
        W=sd[f"actor.{i}.weight"]; b=sd[f"actor.{i}.bias"]
        lin=nn.Linear(W.shape[1],W.shape[0]); lin.weight.data=W; lin.bias.data=b
        layers.append(lin); layers.append(nn.Tanh())
    return nn.Sequential(*layers[:-1]).eval()
print(f"\n{'表征':6s}{'策略':6s}{'|J·u_red|':>11s}{'|J·u_blue|':>12s}{'蓝/红':>8s}{'主方向·u_red':>14s}{'主方向·u_blue':>14s}",flush=True)
for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p); m=seen
    u={k:ridge_dir(Z,y,m) for k,y in Y.items()}
    Zt=torch.tensor(Z[m],dtype=torch.float32)
    for task in ("red","blue"):
        run=f"score_k03noy_student_{arm}{'_donor' if arm!='init' else ''}_g4_diam_angle_{task}"
        act=load_actor(run)
        if act is None: continue
        idx=np.random.default_rng(0).choice(len(Zt),400,replace=False)
        Js=[]
        for i in idx:
            z=Zt[i:i+1].clone().requires_grad_(True)
            out=act(torch.cat([z,GOAL],dim=1))
            J=torch.stack([torch.autograd.grad(out[0,j],z,retain_graph=(j==0))[0][0] for j in range(out.shape[1])])
            Js.append(J.detach().numpy())
        Js=np.stack(Js)                                   # (n, act_dim, 256)
        nr=float(np.linalg.norm(Js@u["red"],axis=1).mean())
        nb=float(np.linalg.norm(Js@u["blue"],axis=1).mean())
        C=np.einsum("nij,nik->njk",Js,Js).mean(0)
        ev,V_=np.linalg.eigh(C); v1=V_[:,-1]
        print(f"{arm:6s}{task:6s}{nr:11.4f}{nb:12.4f}{nb/(nr+1e-12):8.3f}"
              f"{abs(float(v1@u['red'])):14.3f}{abs(float(v1@u['blue'])):14.3f}",flush=True)
