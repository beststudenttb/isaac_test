"""把 z 按"读出头第一层用到的方向"和"零空间"劈开,看红球/蓝球信息各自落在哪一半。
head 第一层是 Linear(256,64):落在其零空间(192 维)的方向对头的输出完全无影响,这是精确结论。
只用 A/V/R 三个行为信号臂。数据用受控析因渲染(7 个变量独立采样,跨轮次留出)。
"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from stage1_helpers import make_head
from torchvision.io import read_image
torch.set_num_threads(12)
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_factorial")
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng=d/np.cos(np.radians(bear))
Y={"红球方位":bear,"红球距离":rng,"蓝球方位":g("blue_bear"),"蓝球距离":g("blue_rng"),
   "光强":np.log(g("light_I")),"墙亮度":(g("wall_r")+g("wall_g")+g("wall_b"))/3}
rd=g("round"); m=(rd%2==0)&seen; h=(rd%2==1)&seen
def ridge(F,y,lam=1e-2):
    if F.shape[1]==0: return float('nan')
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    A=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(F[h]-mu)/sd,np.ones((h.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[h]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
@torch.no_grad()
def enc(p,bs=64):
    cz=Path(f"/home/tb/.claude/jobs/2174f11a/tmp/zf_{p.parent.name}.npy")
    if cz.exists(): return np.load(cz)
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    out=[]
    for i in range(0,len(rows),bs):
        f=torch.stack([read_image(str(D/r["img"])) for r in rows[i:i+bs]])
        out.append(e(f)["shared_feature"].float())
    Z=torch.cat(out).numpy().astype(np.float64); np.save(cz,Z); return Z
print("z 劈成两半:头用到的子空间 vs 头的零空间(后者对监督输出零影响)\n",flush=True)
for arm in ["A","V","R"]:
    Z=enc(ROOT/arm/"donor.pt")
    hd=make_head(256, 2 if arm=="A" else 1)
    hd.load_state_dict(torch.load(ROOT/arm/"head.pt",map_location="cpu"))
    W=hd[0].weight.detach().numpy()                      # (64,256)
    U,S,Vt=np.linalg.svd(W, full_matrices=True)
    r=int((S>1e-6*S[0]).sum())
    P_use=Vt[:r].T                                       # 头真正用到的 r 个方向
    P_null=Vt[r:].T                                      # 零空间
    Zc=Z-Z.mean(0)
    Zu=Zc@P_use; Zn=Zc@P_null
    fu=(Zu**2).sum()/ (Zc**2).sum()
    print(f"--- {arm} 臂  头用到 {r} 维 / 零空间 {256-r} 维;z 的能量有 {fu:.1%} 落在用到的那半 ---",flush=True)
    print(f"{'量':10s}{'全部 256':>10s}{'头用到的':>10s}{'零空间':>10s}",flush=True)
    for k,y in Y.items():
        print(f"{k:10s}{ridge(Z,y):10.3f}{ridge(Zu,y):10.3f}{ridge(Zn,y):10.3f}",flush=True)
    print(flush=True)
