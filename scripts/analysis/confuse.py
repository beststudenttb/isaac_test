"""两球靠近时,红球方位的读出会不会被带偏?
身份检测 -> 误差与两球间距无关;显著性/位置检测 -> 两球靠近(甚至重叠)时误差暴涨。
数据:data_factorial(两球位置独立采样,天然有大量靠近与重叠帧)。探针在训练半边拟合后冻住。CPU。
"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(6)
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_factorial")
CACHE=Path("/home/tb/.claude/jobs/2174f11a/tmp/zfact"); CACHE.mkdir(exist_ok=True)
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40))
bear=np.degrees(np.arctan((112-px)/fx)); bb=g("blue_bear"); rd=g("round")
px_blue=112-fx*np.tan(np.radians(bb))                    # 蓝球的像素横坐标
dia_r=62.12/np.clip(g("set_rng"),0.3,None); dia_b=62.12/np.clip(g("blue_rng"),0.3,None)
gap_px=np.abs(px-px_blue)                                # 画面上两球中心的像素间距
overlap=gap_px < 0.5*(dia_r+dia_b)                       # 判为重叠
m=(rd%2==0)&seen; h=(rd%2==1)&seen
print(f"总帧 {seen.sum()};重叠帧 {int((overlap&seen).sum())}({(overlap&seen).mean()*100:.1f}%)",flush=True)
print(f"两球像素间距分位:10% {np.percentile(gap_px[seen],10):.0f}  50% {np.percentile(gap_px[seen],50):.0f}  90% {np.percentile(gap_px[seen],90):.0f}\n",flush=True)
@torch.no_grad()
def enc(p,bs=48):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(torch.stack([read_image(str(D/r["img"])) for r in rows[i:i+bs]]))["shared_feature"].float()
                      for i in range(0,len(rows),bs)]).numpy().astype(np.float64)
BINS=[(0,20),(20,40),(40,70),(70,110),(110,224)]
print(f"{'表征':6s}{'重叠帧|误差|':>12s}{'非重叠|误差|':>12s}"+"".join(f"{f'{a}-{b}px':>11s}" for a,b in BINS),flush=True)
for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    f=CACHE/f"z_{arm}.npy"
    Z=np.load(f) if f.exists() else enc(p)
    if not f.exists(): np.save(f,Z.astype(np.float32))
    Z=Z.astype(np.float64)
    mu,sd=Z[m].mean(0),Z[m].std(0)+1e-9
    A_=np.hstack([(Z[m]-mu)/sd,np.ones((m.sum(),1))])
    w=np.linalg.lstsq(A_.T@A_+3e-2*len(A_)*np.eye(A_.shape[1]),A_.T@bear[m],rcond=None)[0]
    pred=np.hstack([(Z-mu)/sd,np.ones((len(Z),1))])@w
    err=np.abs(pred-bear)
    o=h&overlap; n=h&(~overlap)
    cells=[f"{err[o].mean():12.2f}",f"{err[n].mean():12.2f}"]
    for a,b in BINS:
        s=h&(gap_px>=a)&(gap_px<b)
        cells.append(f"{err[s].mean():11.2f}" if s.sum()>20 else f"{'--':>11s}")
    print(f"{arm:6s}"+"".join(cells),flush=True)
print("\n(单位:度。探针只在 round 偶数帧拟合,表中全部是留出帧)",flush=True)
