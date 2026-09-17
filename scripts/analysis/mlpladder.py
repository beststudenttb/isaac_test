"""线性读不出来 ≠ 没有。PPO 的 actor 是 MLP,所以要问的是"MLP 读不读得出来"。
用析因渲染的缓存 z,对红球/蓝球方位做 线性 / MLP[64] / MLP[64,64] 三档,按 round 奇偶留出。CPU。
"""
import sys, csv, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"."); torch.set_num_threads(8)
from pathlib import Path
D=Path("data_factorial"); C=Path("/home/tb/.claude/jobs/2174f11a/tmp/zfact")
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx))
Y={"红球方位":bear,"蓝球方位":g("blue_bear"),"蓝球距离":g("blue_rng")}
rd=g("round"); m=(rd%2==0)&seen; h=(rd%2==1)&seen
def lin(Z,y,lam=3e-2):
    mu,sd=Z[m].mean(0),Z[m].std(0)+1e-9
    A=np.hstack([(Z[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(Z[h]-mu)/sd,np.ones((h.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[h]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
def mlp(Z,y,hid,steps=4000,seed=0):
    torch.manual_seed(seed)
    mu,sd=Z[m].mean(0),Z[m].std(0)+1e-9
    ym,ys=y[m].mean(),y[m].std()+1e-9
    Xa=torch.tensor((Z[m]-mu)/sd,dtype=torch.float32); ya=torch.tensor((y[m]-ym)/ys,dtype=torch.float32)[:,None]
    Xb=torch.tensor((Z[h]-mu)/sd,dtype=torch.float32)
    L=[]; prev=Z.shape[1]
    for k in hid: L+=[nn.Linear(prev,k),nn.Tanh()]; prev=k
    L+=[nn.Linear(prev,1)]; net=nn.Sequential(*L)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    for s in range(steps):
        i=torch.randint(0,len(Xa),(256,))
        loss=((net(Xa[i])-ya[i])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): p=net(Xb).numpy()[:,0]*ys+ym
    yy=y[h]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
print("读出器阶梯(held R²,析因渲染):线性 | MLP[64] | MLP[64,64]\n",flush=True)
hdr="".join(f"{k:>30s}" for k in Y)
print(f"{'表征':6s}{hdr}",flush=True)
for arm in ("init","A","V","R","sup"):
    f=C/f"z_{arm}.npy"
    if not f.exists(): print(f"{arm}: 缺 z 缓存"); continue
    Z=np.load(f).astype(np.float64)
    cells=[]
    for k,y in Y.items():
        cells.append(f"{lin(Z,y):9.3f}{mlp(Z,y,[64]):10.3f}{mlp(Z,y,[64,64]):11.3f}")
    print(f"{arm:6s}"+"".join(cells),flush=True)
