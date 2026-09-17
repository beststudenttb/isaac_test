"""验 PiSCO 的层间结论(底层任务无关、顶层任务专属)在我们的设定上成不成立。
对 stem/c2/c3/c4/z 分别测红球与蓝球的可读性。受控析因渲染,按轮次奇偶留出。CPU。
"""
import sys, csv, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(10)
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_factorial")
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng=d/np.cos(np.radians(bear))
Y={"红球方位":bear,"红球距离":rng,"蓝球方位":g("blue_bear"),"蓝球距离":g("blue_rng")}
rd=g("round"); m=(rd%2==0)&seen; h=(rd%2==1)&seen
def ridge(Fm,y,lam=3e-2):
    mu,sd=Fm[m].mean(0),Fm[m].std(0)+1e-9
    A=np.hstack([(Fm[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(Fm[h]-mu)/sd,np.ones((h.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[h]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
@torch.no_grad()
def layers(src, bs=32):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(src,map_location="cpu")); e.eval()
    acc={k:[] for k in ("stem","c2","c3","c4","z")}
    for i in range(0,len(rows),bs):
        f=torch.stack([read_image(str(D/r["img"])) for r in rows[i:i+bs]])
        x=e._to_nchw(f); x=(x-e.imagenet_mean)/e.imagenet_std
        x=e.stem(x); c2=e.layer1(x); c3=e.layer2(c2); c4=e.layer3(c3)
        pool=lambda t: F.adaptive_avg_pool2d(t,2).flatten(1)
        acc["stem"].append(pool(x)); acc["c2"].append(pool(c2)); acc["c3"].append(pool(c3)); acc["c4"].append(pool(c4))
        acc["z"].append(e(f)["shared_feature"].float())
    return {k: torch.cat(v).numpy().astype(np.float64) for k,v in acc.items()}
print("各层的红球/蓝球可读性(held R²,2×2 池化展平)\n",flush=True)
for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("R","R/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    L_=layers(p)
    print(f"--- {arm} ---",flush=True)
    print(f"{'层':7s}{'维数':>7s}"+"".join(f"{k:>11s}" for k in Y),flush=True)
    for k in ("stem","c2","c3","c4","z"):
        Fm=L_[k]; print(f"{k:7s}{Fm.shape[1]:7d}"+"".join(f"{ridge(Fm,y):11.3f}" for y in Y.values()),flush=True)
    print(flush=True)
