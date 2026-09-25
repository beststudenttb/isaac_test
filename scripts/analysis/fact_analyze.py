"""析因渲染数据的分析:每个变量能从 z 解出多少 + 每个变量单独变化引起的 Δz(折算成任务等价量)。"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
DEV = sys.argv[1] if len(sys.argv)>1 else "cpu"
torch.set_num_threads(12)
D=Path("data_factorial"); rows=list(csv.DictReader(open(D/"meta.csv")))
fac=[r for r in rows if r["kind"]=="factorial"]; pair=[r for r in rows if r["kind"]=="pair"]
fx=112/math.tan(math.radians(40))
def load(rs, bs=400):
    return torch.stack([read_image(str(D/r["img"])) for r in rs])
@torch.no_grad()
def enc(p, rs, bs=64):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(p,map_location=DEV)); e.eval()
    out=[]
    for i in range(0,len(rs),bs):
        f=torch.stack([read_image(str(D/r["img"])) for r in rs[i:i+bs]]).to(DEV)
        out.append(e(f)["shared_feature"].float().cpu())
    return torch.cat(out).numpy().astype(np.float64)
g=lambda rs,k: np.array([float(r[k]) for r in rs])
seen=g(fac,"dist")>0.3
V={"方位角(deg)":g(fac,"bearing"),"x_c(px)":g(fac,"px_x"),"距离(m)":g(fac,"rng"),
   "直径(px)":62.12/np.maximum(g(fac,"dist"),1e-6),"蓝球方位":g(fac,"blue_bear"),"蓝球距离":g(fac,"blue_rng"),
   "光强(log)":np.log(g(fac,"light_I")),"光色暖冷":g(fac,"light_r")-g(fac,"light_b"),
   "墙亮度":(g(fac,"wall_r")+g(fac,"wall_g")+g(fac,"wall_b"))/3,"地板亮度":(g(fac,"floor_r")+g(fac,"floor_g")+g(fac,"floor_b"))/3}
rd=g(fac,"round"); m=(rd%2==0)&seen; h=(rd%2==1)&seen     # 按 round 切分,光照/背景跨组泛化
def ridge(F,y,lam=1e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    A=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(F[h]-mu)/sd,np.ones((h.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[h]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
REPS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
ROOT=Path("models/vision/g4_diam_angle")
print(f"析因数据 {len(fac)} 帧(看得见球 {seen.sum()}),按 round 奇偶切分\n",flush=True)
print(f"{'表征':6s}"+"".join(f"{k:>12s}" for k in V),flush=True)
Zs={}
for n,rel in REPS:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p,fac); Zs[n]=(p,Z)
    print(f"{n:6s}"+"".join(f"{ridge(Z,y):12.3f}" for y in V.values()),flush=True)
# --- 配对扰动 ---
conds=[]
for r in pair:
    if r["cond"] not in conds: conds.append(r["cond"])
byc={c:[r for r in pair if r["cond"]==c] for c in conds}
base=byc["base"]
dia=lambda rs: 62.12/np.maximum(g(rs,"dist"),1e-6)
print(f"\n配对扰动:每个变量单独变化引起的 Δz,折算成等价的球位移(停车容差 3° / 0.2m)",flush=True)
print(f"{'表征':6s}{'扰动':14s}{'等价角度°':>10s}{'等价距离m':>11s}",flush=True)
for n,(p,_) in Zs.items():
    Zp={c: enc(p, byc[c]) for c in conds}
    s=Zp["base"].std(0)+1e-9
    nrm=lambda c: np.linalg.norm((Zp[c]-Zp["base"])/s,axis=1)
    dpb=np.abs(g(byc["bear+5px"],"px_x")-g(base,"px_x")); dpd=np.abs(dia(byc["rng+2pxdiam"])-dia(base))
    kb=(nrm("bear+5px")/np.maximum(dpb,1e-6)).mean(); kr=(nrm("rng+2pxdiam")/np.maximum(dpd,1e-6)).mean()
    for c in conds[1:]:
        v=nrm(c).mean(); eq_px=v/kb; eq_deg=math.degrees(math.atan(eq_px/fx)); eq_m=(v/kr)/(62.12/1.5**2)
        print(f"{n:6s}{c:14s}{eq_deg:10.2f}{eq_m:11.3f}",flush=True)
    print(flush=True)
