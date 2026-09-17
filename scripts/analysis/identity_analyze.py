"""身份实验分析:探针在 base 条件(红球在槽1、蓝球在槽2)上拟合后**冻住**,
然后喂给每个条件,看它的输出跟着哪个球走。
读出跟颜色走 -> 身份是颜色条件的;跟槽位走 -> 是位置/显著性。CPU。
"""
import sys, csv, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(8)
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_identity")
rows=list(csv.DictReader(open(D/"meta.csv")))
cond=np.array([r["cond"] for r in rows]); rd=np.array([int(r["round"]) for r in rows])
b1=np.array([float(r["bear1"]) for r in rows]); b2=np.array([float(r["bear2"]) for r in rows])
r1=np.array([float(r["rng1"]) for r in rows]);  r2=np.array([float(r["rng2"]) for r in rows])
c1=np.array([r["col1"] for r in rows]); c2=np.array([r["col2"] for r in rows])
CONDS=[c for c in dict.fromkeys(cond.tolist())]
base=cond=="base"; fit=base&(rd%2==0); test_base=base&(rd%2==1)
print(f"共 {len(rows)} 帧,{len(CONDS)} 个条件;base 拟合 {fit.sum()} 帧、留出 {test_base.sum()} 帧\n",flush=True)
@torch.no_grad()
def enc(p,bs=48):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(torch.stack([read_image(str(D/r["img"])) for r in rows[i:i+bs]]))["shared_feature"].float()
                      for i in range(0,len(rows),bs)]).numpy().astype(np.float64)
def fit_probe(Z,y,m,lam=3e-2):
    mu,sd=Z[m].mean(0),Z[m].std(0)+1e-9
    A=np.hstack([(Z[m]-mu)/sd,np.ones((m.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    return lambda Zx: np.hstack([(Zx-mu)/sd,np.ones((len(Zx),1))])@w
def r2(pred,y):
    return float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum())
for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p)
    fb=fit_probe(Z,b1,fit); fr=fit_probe(Z,r1,fit)     # 在 base 上学"目标(槽1)方位角"和"目标距离"
    fb2=fit_probe(Z,b2,fit)                            # 同时学"干扰球(槽2)方位角",给出**固定外观下**的蓝球可读性
    pb=fb(Z); pr=fr(Z); pb2=fb2(Z)
    print(f"=== {arm} ===  base 留出:槽1方位 R²={r2(pb[test_base],b1[test_base]):.3f}  槽1距离 R²={r2(pr[test_base],r1[test_base]):.3f}"
          f"  |  **槽2(蓝球)方位 R²={r2(pb2[test_base],b2[test_base]):.3f}**  <- 固定外观下的蓝球可读性,与析因渲染(随机光照/墙/地板)对比",flush=True)
    print(f"{'条件':12s}{'槽1色':9s}{'槽2色':9s}{'方位 R²(跟槽1)':>16s}{'方位 R²(跟槽2)':>16s}{'|误差|→槽1':>12s}{'|误差|→槽2':>12s}",flush=True)
    for c in CONDS:
        m=cond==c
        if m.sum()<30: continue
        has1=c1[m][0]!="none"; has2=c2[m][0]!="none"
        cell=lambda ok,y: (f"{r2(pb[m],y[m]):16.3f}" if ok else f"{'--':>16s}")
        err =lambda ok,y: (f"{np.abs(pb[m]-y[m]).mean():12.2f}" if ok else f"{'--':>12s}")
        print(f"{c:12s}{c1[m][0]:9s}{c2[m][0]:9s}{cell(has1,b1)}{cell(has2,b2)}{err(has1,b1)}{err(has2,b2)}",flush=True)
    print(flush=True)
print("读法:探针只在 base 上学过。若某条件下'跟槽2'的 R² 高而'跟槽1'低,说明读出跟着颜色/身份跑,而不是跟着槽位。",flush=True)
