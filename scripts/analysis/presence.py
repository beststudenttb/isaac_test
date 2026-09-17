"""存在 vs 位置:A 把干扰球的**位置**擦掉了(R²=-0.12),那它还知不知道场上**有**这个球?
身份渲染里几何逐帧可配对,所以能做干净的二分类:
  测试1 存在性: base(红@1,蓝@2) vs only_red1(红@1,无@2) —— 槽1几何完全相同
  测试2 颜色  : base(蓝@2)      vs two_red(红@2)        —— 两槽几何都相同,只有槽2颜色不同
  空对照      : base 随机对半分,应当≈50%
按 round 奇偶留出。CPU。
"""
import sys, csv, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(8)
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_identity")
C=Path("/home/tb/.claude/jobs/2174f11a/tmp/zident"); C.mkdir(exist_ok=True)
rows=list(csv.DictReader(open(D/"meta.csv")))
cond=np.array([r["cond"] for r in rows]); rd=np.array([int(r["round"]) for r in rows])
@torch.no_grad()
def enc(p,bs=48):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(torch.stack([read_image(str(D/r["img"])) for r in rows[i:i+bs]]))["shared_feature"].float()
                      for i in range(0,len(rows),bs)]).numpy().astype(np.float64)
def clf(Z,pos,neg,lam=3e-2,rng=None):
    """岭回归当分类器:+1/-1,报留出集准确率。"""
    y=np.zeros(len(Z)); y[pos]=1.0; y[neg]=-1.0
    use=pos|neg
    if rng is not None:                       # 空对照:同一批帧随机贴标签
        y=np.where(rng.random(len(Z))<0.5,1.0,-1.0)
    tr=use&(rd%2==0); te=use&(rd%2==1)
    mu,sd=Z[tr].mean(0),Z[tr].std(0)+1e-9
    A=np.hstack([(Z[tr]-mu)/sd,np.ones((tr.sum(),1))]); B=np.hstack([(Z[te]-mu)/sd,np.ones((te.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[tr],rcond=None)[0]
    return float((np.sign(B@w)==y[te]).mean())
m=lambda c: cond==c
print(f"{'表征':6s}{'干扰球在不在(base vs only_red1)':>32s}{'槽2是蓝还是红(base vs two_red)':>32s}{'空对照':>10s}",flush=True)
for arm,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    f=C/f"z_{arm}.npy"
    if f.exists(): Z=np.load(f).astype(np.float64)
    else: Z=enc(p); np.save(f,Z.astype(np.float32))
    a=clf(Z,m("base"),m("only_red1"))
    b=clf(Z,m("base"),m("two_red"))
    n=clf(Z,m("base"),m("only_red1"),rng=np.random.default_rng(0))
    print(f"{arm:6s}{a*100:31.1f}%{b*100:31.1f}%{n*100:9.1f}%",flush=True)
print("\n读法:'在不在'高而位置读不出 -> 只擦了位置,没擦存在;两个都≈50% -> 干扰球对这个表征完全不可见。",flush=True)
