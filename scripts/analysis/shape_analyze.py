"""形状实验分析:探针只在 base(红球@槽1)上拟合后冻住,喂给每个条件看它跟着谁。
颜色已知是判据(§22),这里问形状是否也是。CPU。
"""
import sys, csv, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(8)
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_shape")
C=Path("/home/tb/.claude/jobs/2174f11a/tmp/zshape"); C.mkdir(exist_ok=True)
rows=list(csv.DictReader(open(D/"meta.csv")))
cond=np.array([r["cond"] for r in rows]); rd=np.array([int(r["round"]) for r in rows])
b1=np.array([float(r["bear1"]) for r in rows]); b2=np.array([float(r["bear2"]) for r in rows])
o1=np.array([r["obj1"] for r in rows]); o2=np.array([r["obj2"] for r in rows])
r1=np.array([float(r["rng1"]) for r in rows])
sc=np.array([float(r.get("scale",1.0)) for r in rows]); lf=np.array([float(r.get("lift",0.0)) for r in rows])
c1=np.array([r["col1"] for r in rows]); c2=np.array([r["col2"] for r in rows])
CONDS=list(dict.fromkeys(cond.tolist()))
base=cond=="base"; fit=base&(rd%2==0); te=base&(rd%2==1)
print(f"{len(rows)} 帧,{len(CONDS)} 条件;base 拟合 {fit.sum()} / 留出 {te.sum()}\n",flush=True)
@torch.no_grad()
def enc(p,bs=48):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(torch.stack([read_image(str(D/r["img"])) for r in rows[i:i+bs]]))["shared_feature"].float()
                      for i in range(0,len(rows),bs)]).numpy().astype(np.float64)
def probe(Z,y,m,lam=3e-2):
    mu,sd=Z[m].mean(0),Z[m].std(0)+1e-9
    A=np.hstack([(Z[m]-mu)/sd,np.ones((m.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    return lambda Zx: np.hstack([(Zx-mu)/sd,np.ones((len(Zx),1))])@w
r2=lambda p,y: float(1-((y-p)**2).sum()/((y-y.mean())**2).sum())
for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    f=C/f"z_{arm}.npy"
    Z=np.load(f).astype(np.float64) if f.exists() else enc(p)
    if not f.exists(): np.save(f,Z.astype(np.float32))
    pb=probe(Z,b1,fit)(Z); pr=probe(Z,r1,fit)(Z)
    print(f"=== {arm} ===  base 留出:方位 R²={r2(pb[te],b1[te]):.3f}  距离 R²={r2(pr[te],r1[te]):.3f}",flush=True)
    print(f"{'条件':15s}{'槽1':18s}{'槽2':18s}{'跟槽1':>9s}{'跟槽2':>9s}{'|误差|槽1':>11s}{'|误差|槽2':>11s}",flush=True)
    for c in CONDS:
        m=cond==c
        if m.sum()<30: continue
        h1=o1[m][0]!="none"; h2=o2[m][0]!="none"
        s1=f"{c1[m][0]}·{o1[m][0]}" if h1 else "—"; s2=f"{c2[m][0]}·{o2[m][0]}" if h2 else "—"
        g=lambda ok,y: (f"{r2(pb[m],y[m]):9.3f}" if ok else f"{'--':>9s}")
        e=lambda ok,y: (f"{np.abs(pb[m]-y[m]).mean():11.2f}" if ok else f"{'--':>11s}")
        print(f"{c:15s}{s1:18s}{s2:18s}{g(h1,b1)}{g(h2,b2)}{e(h1,b1)}{e(h2,b2)}",flush=True)
    # 距离线索:同一批几何下,读出的距离相对基线偏了多少
    bm=cond=="base"
    print(f"  {'距离线索':12s}{'缩放':>6s}{'抬高':>6s}{'读出距离均值':>12s}{'相对 base':>11s}{'大小线索预期':>13s}",flush=True)
    for c,tag in [("base","基线"),("scale_half","半径0.5r"),("scale_two","半径2r"),("float_30","抬高0.3m"),("float_60","抬高0.6m")]:
        m=cond==c
        if m.sum()<30: continue
        exp = r1[m].mean()/sc[m][0] if sc[m][0]!=1.0 else r1[m].mean()   # 纯大小线索:看起来的距离 = 真距离/缩放
        print(f"  {tag:12s}{sc[m][0]:6.1f}{lf[m][0]:6.2f}{pr[m].mean():12.2f}{pr[m].mean()-pr[bm].mean():11.2f}{exp-r1[bm].mean():13.2f}",flush=True)
    print(flush=True)
print("读法:red_cube/cone/cyl 若接近 base -> 形状无关,只看颜色;conflict 指向槽2 -> 颜色赢,指向槽1 -> 形状赢。",flush=True)
