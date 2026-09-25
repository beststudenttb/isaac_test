"""2x2 的 20 个表征:两组坐标的读出阶梯 + 对各自 teacher 观测的残差(纯 CPU)。
用法: CUDA_VISIBLE_DEVICES="" python g4_probe.py <OBS> <RM>
"""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(12)
O, RM = sys.argv[1], sys.argv[2]; N = 2000; torch.manual_seed(0)
DATA = Path(f"./data_g4_{O}_{RM}"); ROOT = Path(f"./models/vision/g4_{O}_{RM}")
tr = fr.load_transitions(DATA, units=1); seen = tr["dist"] > 0.3
m_all = fr.episode_split(tr["episode"], seed=0); rg_ = np.random.default_rng(0)
idx = np.concatenate([rg_.choice(np.flatnonzero(m_all & seen), N, replace=False),
                      rg_.choice(np.flatnonzero((~m_all) & seen), N, replace=False)])
m = np.r_[np.ones(N, bool), np.zeros(N, bool)]
frames = fr.load_frames(tr["img"][idx], DATA)
px = tr["px_x"][idx].astype(np.float64); d = tr["dist"][idx].astype(np.float64)
fx = 112/np.tan(np.radians(40)); bear = np.degrees(np.arctan((112-px)/fx)); rng_m = d/np.cos(np.radians(bear))
diam = 62.12/d
f = frames.float(); R_,G_,B_ = f[:,0],f[:,1],f[:,2]
blue = (B_>90)&(B_>R_*1.8)&(B_>G_*1.8); bsz = blue.flatten(1).sum(1).float().numpy().astype(np.float64)
xs = torch.arange(224).view(1,1,224).expand(len(f),224,224).float()
bx = ((blue.float()*xs).flatten(1).sum(1)/blue.flatten(1).sum(1).clamp(min=1)).numpy().astype(np.float64)
BLUE = np.stack([bx/112-1, np.sqrt(bsz)/30, (bsz>30).astype(float)],1)
QTY = {"x_c":px, "直径":diam, "方位角":bear, "距离":rng_m, "蓝球":bsz}
hf=112.0
T = {"diam": np.stack([px/hf-1, np.clip(diam/224,0,1)],1),
     "ang":  np.stack([np.clip(bear/40,-1,1), np.clip(rng_m/8,0,1)],1)}[O]
def r2(p,y): 
    h=~m; return float(1-((p[h]-y[h])**2).sum()/((y[h]-y[h].mean(0))**2).sum())
def ridge(F,y,lam=1e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9; a=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); b=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@y[m],rcond=None)[0]
    p=b@w; yy=y[~m]; return float(1-((yy-p)**2).sum()/((yy-yy.mean(0))**2).sum())
def mlp(X,Y,hid=64,steps=2500,lr=3e-3):
    X=np.atleast_2d(X.T).T if X.ndim==1 else X; Y=np.atleast_2d(Y.T).T if Y.ndim==1 else Y
    xs_=torch.tensor(X,dtype=torch.float32); ys_=torch.tensor(Y,dtype=torch.float32)
    mu,sd=xs_[m].mean(0),xs_[m].std(0)+1e-6; xs_=(xs_-mu)/sd
    net=nn.Sequential(nn.Linear(X.shape[1],hid),nn.Tanh(),nn.Linear(hid,hid),nn.Tanh(),nn.Linear(hid,Y.shape[1]))
    opt=torch.optim.Adam(net.parameters(),lr=lr); ti=torch.tensor(np.flatnonzero(m))
    for _ in range(steps):
        s=ti[torch.randint(0,len(ti),(256,))]
        loss=((net(xs_[s])-ys_[s])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): p=net(xs_).numpy()
    return r2(p,Y)
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return np.concatenate([e(frames[i:i+64])["shared_feature"].float().numpy() for i in range(0,len(frames),64)]).astype(np.float64)
print(f"== teacher 观测={O} 奖励={RM} ==  held {N} 帧", flush=True)
print(f"{'表征':7s}" + "".join(f"{k:>20s}" for k in QTY) + f"{'z|T':>8s}{'残差':>7s}", flush=True)
print(f"{'':7s}" + "".join(f"{'单轴/线性/MLP':>20s}" for _ in QTY), flush=True)
for name,p in [("init",ROOT/"encoder_init.pt")]+[(a,ROOT/a/"donor.pt") for a in ("A","V","R","sup")]:
    if not p.exists(): continue
    Z=enc(p); cells=[]
    for k,y in QTY.items():
        b1=max(ridge(Z[:,[j]],y) for j in range(Z.shape[1])); li=ridge(Z,y); ml=mlp(Z,y)
        cells.append(f"{b1:5.2f}/{li:5.2f}/{ml:5.2f}")
    eT=mlp(T,Z,hid=128,steps=3000)
    print(f"{name:7s}" + "".join(f"{c:>20s}" for c in cells) + f"{eT:8.3f}{1-eT:7.3f}", flush=True)
