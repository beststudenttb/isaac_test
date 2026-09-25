"""z 里除了「任务直接相关的量(红球位置)」之外还剩什么。
做法一:把 z 拆成 任务流形 g(红球方位,红球距离) + 残差 r,只从 r 解码干扰物的位置/存在。
做法二:几何逐像素固定、只换目标颜色,看 z 还能不能解码出「目标是什么颜色」——
        H1 颜色被留在 z 里(额外信息) vs H2 颜色只在映射里用来选中目标、用完就丢。
全部用已渲好的受控帧,纯 CPU。
"""
import sys, csv, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(10)
ROOT=Path("models/vision/g4_diam_angle")
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("Aw","Aw/donor.pt"),("Ax","Ax/donor.pt"),
      ("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
def load(ds):
    rows=list(csv.DictReader(open(Path(ds)/"meta.csv")))
    return rows
@torch.no_grad()
def enc(p,ds,rows,bs=64):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    out=[]
    for i in range(0,len(rows),bs):
        out.append(e(torch.stack([read_image(str(Path(ds)/r["img"])) for r in rows[i:i+bs]]))["shared_feature"].float())
    return torch.cat(out).numpy().astype(np.float64)
def mlp_fit(X,Y,m,hid=(128,128),steps=3000,seed=0):
    torch.manual_seed(seed)
    mu,sd=X[m].mean(0),X[m].std(0)+1e-9; ym,ys=Y[m].mean(0),Y[m].std(0)+1e-9
    Xa=torch.tensor((X[m]-mu)/sd,dtype=torch.float32); Ya=torch.tensor((Y[m]-ym)/ys,dtype=torch.float32)
    L=[];prev=X.shape[1]
    for h in hid: L+=[nn.Linear(prev,h),nn.Tanh()]; prev=h
    L+=[nn.Linear(prev,Y.shape[1])]; net=nn.Sequential(*L)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    for _ in range(steps):
        i=torch.randint(0,len(Xa),(256,))
        loss=((net(Xa[i])-Ya[i])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P=net(torch.tensor((X-mu)/sd,dtype=torch.float32)).numpy()*ys+ym
    return P
def ridge_r2(F,y,m,lam=3e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    A=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[~m]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
def acc(F,y,m,lam=3e-2):   # 多类:每类一个 one-vs-rest 岭回归,取 argmax
    K=int(y.max())+1
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    A=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    G=A.T@A+lam*len(A)*np.eye(A.shape[1]); S=[]
    for k in range(K):
        t=(y[m]==k).astype(np.float64)*2-1
        S.append(B@np.linalg.lstsq(G,A.T@t,rcond=None)[0])
    return float((np.argmax(np.stack(S,1),1)==y[~m]).mean()), 1.0/K

# ============ 做法一:残差里有什么(data_distract)============
print("="*78,flush=True)
print("做法一:z = g(红球方位,红球距离) + 残差;只从残差解码干扰物",flush=True)
D1="data_distract"; r1=load(D1)
g=lambda k: np.array([float(x[k]) for x in r1])
b1,rr1=g("bear1"),g("rng1"); b2,rr2=g("bear2"),g("rng2")
cond=np.array([x["cond"] for x in r1]); rd=np.array([int(x["round"]) for x in r1])
has2=(cond!="d_none")
m=(rd%2==0)
XY=np.stack([b1,rr1],1)
print(f"{'臂':6s}{'残差占方差':>11s}{'干扰方位·全z':>14s}{'干扰方位·残差':>14s}{'干扰在不在·全z':>16s}{'干扰在不在·残差':>16s}",flush=True)
Zc={}
for arm,rel in ARMS:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p,D1,r1); Zc[arm]=Z
    P=mlp_fit(XY,Z,m)                       # 用红球位置预测 z
    res=Z-P
    frac=float((res.var(0).sum())/(Z.var(0).sum()))
    a_full=ridge_r2(Z[has2],b2[has2],m[has2])
    a_res =ridge_r2(res[has2],b2[has2],m[has2])
    y=has2.astype(np.int64)
    p_full,_=acc(Z,y,m); p_res,ch=acc(res,y,m)
    print(f"{arm:6s}{frac:11.3f}{a_full:14.3f}{a_res:14.3f}{p_full*100:15.1f}%{p_res*100:15.1f}%",flush=True)
print(f"(存在性的随机基线 = {ch*100:.0f}%)",flush=True)

# ============ 做法二:几何固定,只换目标颜色 ============
print("\n"+"="*78,flush=True)
print("做法二:几何逐像素相同、只换目标球颜色 -> z 还能不能解码出颜色?",flush=True)
D2="data_identity"; r2_=load(D2)
c2=np.array([x["cond"] for x in r2_]); rd2=np.array([int(x["round"]) for x in r2_])
COL=["base","darkred","orange","yellow","magenta","green","cyan","white"]   # 目标槽被改色的条件
sel=np.isin(c2,COL)
lab=np.array([COL.index(c) if c in COL else -1 for c in c2])
mm=sel&(rd2%2==0)
print(f"{'臂':6s}{'目标颜色 8 类准确率':>20s}{'(随机 12.5%)':>14s}",flush=True)
for arm,rel in ARMS:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p,D2,r2_)
    sub=np.flatnonzero(sel); Zs=Z[sub]; ys=lab[sub]; ms=(rd2[sub]%2==0)
    a,ch2=acc(Zs,ys,ms)
    print(f"{arm:6s}{a*100:19.1f}%{'':>14s}",flush=True)
print("\n读法:做法二若准确率≈12.5%(随机)=> H2,颜色只在映射里用来选中目标、用完就丢;"
      "\n      若显著高于随机 => H1,颜色作为额外信息被留在 z 里。",flush=True)
