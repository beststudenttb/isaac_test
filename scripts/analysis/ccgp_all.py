"""把 §15 的两个关键 CCGP 数推广到四种 teacher 配置 × 五个表征(n=20),看结论是不是只在 diam_angle 成立。
只留三列:红球方位同分布 / 红球方位跨蓝球位置(不受干扰) / 蓝球方位同分布(抹没抹掉)。
帧只解码一次,复用给 20 个编码器。CPU。
"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(8)
D=Path("data_factorial")
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx))
bb=g("blue_bear"); rd=g("round")
SPL=[("红球方位/同分布",bear,(rd%2==0)&seen,(rd%2==1)&seen),
     ("红球方位/跨蓝球位置",bear,(bb<0)&seen,(bb>=0)&seen),
     ("蓝球方位/同分布",bb,(rd%2==0)&seen,(rd%2==1)&seen)]
print("解码帧...",flush=True)
FR=torch.stack([read_image(str(D/r["img"])) for r in rows])
def ridge(F,y,tr,te,lam=3e-2):
    mu,sd=F[tr].mean(0),F[tr].std(0)+1e-9
    A=np.hstack([(F[tr]-mu)/sd,np.ones((tr.sum(),1))]); B=np.hstack([(F[te]-mu)/sd,np.ones((te.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[tr],rcond=None)[0]
    p=B@w; yy=y[te]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
@torch.no_grad()
def enc(src,bs=48):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(src,map_location="cpu")); e.eval()
    return torch.cat([e(FR[i:i+bs])["shared_feature"].float() for i in range(0,len(FR),bs)]).numpy().astype(np.float64)
print(f"\n{'配置':14s}{'表征':6s}"+"".join(f"{n:>22s}" for n,_,_,_ in SPL),flush=True)
for cfg in ("diam_angle","diam_cam","ang_angle","ang_cam"):
    ROOT=Path(f"models/vision/g4_{cfg}")
    for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
        p=ROOT/rel
        if not p.exists(): continue
        Z=enc(p)
        cells=[]
        for nm,y,a,b in SPL:
            cells.append(f"{0.5*(ridge(Z,y,a,b)+ridge(Z,y,b,a)):22.3f}")
        print(f"{cfg:14s}{arm:6s}"+"".join(cells),flush=True)
