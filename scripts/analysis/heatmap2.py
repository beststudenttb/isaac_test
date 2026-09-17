"""高分辨率空间归因热图(Grad-CAM 风格)。
上一版的错误:把梯度归到了**池化之后**的特征(那里按构造只有 7×7)。正确做法是归到**池化之前**的
FPN 特征图 —— p2 是 56×56,足够画出平滑的热图。
    CAM(x,y) = Σ_c  α_c · p2[c,x,y]，  α_c = Σ_{x,y} ∂(探针输出)/∂p2[c,x,y]
探针 = 在该数据集 base 条件上拟合的"红球方位角"读出(冻结)。三尺度分别算再相加。CPU。
"""
import sys, csv, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(8)
ap=argparse.ArgumentParser()
ap.add_argument("--data", default="data_distract")
ap.add_argument("--conds", default="")
ap.add_argument("--env", type=int, default=7)
ap.add_argument("--round", type=int, default=1)
ap.add_argument("--out", default="figs/heatmap_distract.png")
a=ap.parse_args()
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"]=font_manager.FontProperties(fname=f).get_name(); break
ROOT=Path("models/vision/g4_diam_angle"); D=Path(a.data)
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
rows=list(csv.DictReader(open(D/"meta.csv")))
cond=np.array([r["cond"] for r in rows]); rd=np.array([int(r["round"]) for r in rows])
b1=np.array([float(r["bear1"]) for r in rows])
CONDS=[c for c in dict.fromkeys(cond.tolist())] if not a.conds else a.conds.split(",")
def encoder(rel):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(ROOT/rel,map_location="cpu")); e.eval(); return e
def fpn(e,x):
    x=e._to_nchw(x); x=(x-e.imagenet_mean)/e.imagenet_std
    s=e.stem(x); c2=e.layer1(s); c3=e.layer2(c2); c4=e.layer3(c3)
    p4=e.lat4(c4); p3=e.lat3(c3)+F.interpolate(p4,size=c3.shape[-2:],mode="nearest")
    p2=e.lat2(c2)+F.interpolate(p3,size=c2.shape[-2:],mode="nearest")
    return e.smooth2(p2), e.smooth3(p3), e.smooth4(p4)
def head(e,ps):
    pooled=torch.cat([e._grid_pool(p) for p in ps],dim=1)
    return e.norm(e.proj(pooled))
def fit_probe(e):
    m=(cond=="base")&(rd%2==0); idx=np.flatnonzero(m); Z=[]
    with torch.no_grad():
        for i in range(0,len(idx),48):
            fr=torch.stack([read_image(str(D/rows[j]["img"])) for j in idx[i:i+48]])
            Z.append(e(fr)["shared_feature"].float())
    Z=torch.cat(Z).numpy().astype(np.float64)
    mu,sd=Z.mean(0),Z.std(0)+1e-9
    A_=np.hstack([(Z-mu)/sd,np.ones((len(Z),1))])
    w=np.linalg.lstsq(A_.T@A_+3e-2*len(A_)*np.eye(A_.shape[1]),A_.T@b1[idx],rcond=None)[0]
    return torch.tensor(w[:-1]/sd,dtype=torch.float32), torch.tensor(mu,dtype=torch.float32)
def cam(e,w,mu,img):
    x=img[None].float()
    ps=[p.detach().requires_grad_(True) for p in fpn(e,x)]
    out=((head(e,ps)[0]-mu)*w).sum()
    gs=torch.autograd.grad(out,ps)
    m=0.0
    for p,g in zip(ps,gs):                      # Grad-CAM:通道权重 = 该通道梯度的空间和
        al=g.sum(dim=(2,3),keepdim=True)
        c=(al*p).sum(1,keepdim=True)            # (1,1,H,W)
        m=m+F.interpolate(c,size=(224,224),mode="bilinear",align_corners=False)[0,0]
    m=m.detach().numpy()
    return np.abs(m)                            # 取幅值:关注"用到了哪里",不分正负
sel_rows=[]
for c in CONDS:
    hit=[r for r in rows if r["cond"]==c and int(r["env"])==a.env and int(r["round"])==a.round]
    if hit: sel_rows.append((c,hit[0]))
n=len(sel_rows)
fig,axes=plt.subplots(n,len(ARMS)+1,figsize=(2.5*(len(ARMS)+1),2.5*n),squeeze=False)
ENC={arm:encoder(rel) for arm,rel in ARMS}
PRB={arm:fit_probe(ENC[arm]) for arm,_ in ARMS}
for i,(c,r) in enumerate(sel_rows):
    img=read_image(str(D/r["img"])); rgb=img.permute(1,2,0).numpy()
    lbl=f"{c}\n槽2={r.get('col2','-')}·{r.get('obj2','-')}"
    axes[i][0].imshow(rgb); axes[i][0].set_ylabel(lbl,fontsize=8); axes[i][0].set_xticks([]); axes[i][0].set_yticks([])
    if i==0: axes[i][0].set_title("输入",fontsize=10)
    for j,(arm,_) in enumerate(ARMS):
        h=cam(ENC[arm],*PRB[arm],img); h=h/(h.max()+1e-9)
        ax=axes[i][j+1]; ax.imshow(rgb); ax.imshow(h,cmap="jet",alpha=0.5,vmin=0,vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i==0: ax.set_title(arm,fontsize=11)
plt.suptitle("对『红球方位角』读出的空间归因(Grad-CAM,FPN p2/p3/p4 三尺度合成)",fontsize=12)
plt.tight_layout(); Path(a.out).parent.mkdir(exist_ok=True); plt.savefig(a.out,dpi=140)
print(f"[DONE] {n} 条件 × {len(ARMS)} 表征 -> {a.out}",flush=True)
