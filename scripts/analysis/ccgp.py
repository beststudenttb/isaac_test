"""CCGP(cross-condition generalization performance,Bernardi et al., Cell 2020)搬到我们的表征上。
他们把"抽象"定义成:解码器在**训练时没见过的条件**上还能不能解码。我们的读出器阶梯全部饱和,
是因为训练/留出是同分布随机切的;换成按条件切,才能分出"能读"和"以可泛化的格式编码"。
受控析因渲染 data_factorial:60 轮 × 64 env,位置/光照/墙色/地板色独立采样。CPU。
"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(8)
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_factorial")
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng_=d/np.cos(np.radians(bear))
Y={"红球方位":bear,"红球距离":rng_,"蓝球方位":g("blue_bear"),"蓝球距离":g("blue_rng")}
rd=g("round"); md=lambda v: v<np.median(v)
SPL={
 "同分布(轮次奇偶)": (rd%2==0, rd%2==1),
 "跨距离(近→远)":    (md(rng_), ~md(rng_)),
 "跨光照(暗→亮)":    (md(g("light_I")), ~md(g("light_I"))),
 "跨墙色":            (md(g("wall_r")), ~md(g("wall_r"))),
 "跨地板色":          (md(g("floor_g")), ~md(g("floor_g"))),
 "跨蓝球位置":        (g("blue_bear")<0, g("blue_bear")>=0),
 "跨红球位置":        (g("set_bear")<0, g("set_bear")>=0),
}
# 每个目标量在哪些划分下是有意义的(不能拿目标自己当划分轴,那是外推不是泛化)
SKIP={"红球方位":{"跨红球位置"},"红球距离":{"跨距离"},"蓝球方位":{"跨蓝球位置"},"蓝球距离":{}}
def ridge(F,y,tr,te,lam=3e-2):
    mu,sd=F[tr].mean(0),F[tr].std(0)+1e-9
    A=np.hstack([(F[tr]-mu)/sd,np.ones((tr.sum(),1))]); B=np.hstack([(F[te]-mu)/sd,np.ones((te.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[tr],rcond=None)[0]
    p=B@w; yy=y[te]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
@torch.no_grad()
def enc(src,bs=32):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(src,map_location="cpu")); e.eval()
    out=[]
    for i in range(0,len(rows),bs):
        out.append(e(torch.stack([read_image(str(D/r["img"])) for r in rows[i:i+bs]]))["shared_feature"].float())
    return torch.cat(out).numpy().astype(np.float64)
print("CCGP:同分布可读性 vs 跨条件可读性(held R²);seen 帧\n",flush=True)
for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p)
    print(f"--- {arm} ---",flush=True)
    print(f"{'划分':20s}"+"".join(f"{k:>11s}" for k in Y),flush=True)
    for nm,(a,b) in SPL.items():
        cells=[]
        for tgt,y in Y.items():
            if nm in SKIP[tgt]: cells.append(f"{'--':>11s}"); continue
            r1=ridge(Z,y,a&seen,b&seen); r2=ridge(Z,y,b&seen,a&seen)
            cells.append(f"{0.5*(r1+r2):11.3f}")
        print(f"{nm:20s}"+"".join(cells),flush=True)
    print(flush=True)
