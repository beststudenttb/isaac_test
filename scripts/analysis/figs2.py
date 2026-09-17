"""两张不依赖扰动的图(前四次归因图都败在"一扰动就跑出分布"):
图一 颜色调谐曲线:目标球颜色从红排到蓝,看"还能不能追踪到它"(身份实验数据,全在分布内)。
图二 z 空间最近邻检索:拿一帧做查询,在全部帧里找 z 最近的几帧排出来,看它认为什么是"同一个场景"。
CPU。
"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(10)
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"]=font_manager.FontProperties(fname=f).get_name(); break
ROOT=Path("models/vision/g4_diam_angle"); Path("figs").mkdir(exist_ok=True)
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
ENC={}
for arm,rel in ARMS:
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(ROOT/rel,map_location="cpu")); e.eval(); ENC[arm]=e
@torch.no_grad()
def encode(e,D,rs,bs=64):
    out=[]
    for i in range(0,len(rs),bs):
        out.append(e(torch.stack([read_image(str(D/r["img"])) for r in rs[i:i+bs]]))["shared_feature"].float())
    return torch.cat(out).numpy().astype(np.float64)
def ridge(Z,y,m,h,lam=3e-2):
    mu,sd=Z[m].mean(0),Z[m].std(0)+1e-9
    A=np.hstack([(Z[m]-mu)/sd,np.ones((m.sum(),1))]); B=np.hstack([(Z[h]-mu)/sd,np.ones((h.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[m],rcond=None)[0]
    p=B@w; yy=y[h]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())

# ---------- 图一:颜色调谐曲线 ----------
D1=Path("data_identity"); r1=list(csv.DictReader(open(D1/"meta.csv")))
c1=np.array([r["cond"] for r in r1]); rd1=np.array([int(r["round"]) for r in r1])
b1=np.array([float(r["bear1"]) for r in r1])
ORDER=[("base","红\n(1,0,0)"),("darkred","暗红\n(.42,0,0)"),("orange","橙\n(1,.45,0)"),("yellow","黄\n(1,.95,0)"),
       ("magenta","品红\n(1,0,.55)"),("green","绿\n(0,.72,.1)"),("cyan","青\n(0,.85,1)"),
       ("white","白\n(.92,.92,.92)"),("swap","蓝\n(0,.1,1)")]
print("图一:编码 data_identity ...",flush=True)
Zs={arm:encode(ENC[arm],D1,r1) for arm,_ in ARMS}
plt.figure(figsize=(9,4.6))
tab={}
for arm,_ in ARMS:
    Z=Zs[arm]; fitm=(c1=="base")&(rd1%2==0); ys=[]
    for cd,_lab in ORDER:
        m=(c1==cd)&(rd1%2==1)
        if m.sum()<30: ys.append(np.nan); continue
        mu,sd=Z[fitm].mean(0),Z[fitm].std(0)+1e-9
        A=np.hstack([(Z[fitm]-mu)/sd,np.ones((fitm.sum(),1))])
        w=np.linalg.lstsq(A.T@A+3e-2*len(A)*np.eye(A.shape[1]),A.T@b1[fitm],rcond=None)[0]
        B=np.hstack([(Z[m]-mu)/sd,np.ones((m.sum(),1))]); p=B@w; yy=b1[m]
        ys.append(float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum()))
    tab[arm]=ys
    plt.plot(range(len(ORDER)),ys,marker="o",lw=2,label=arm)
plt.xticks(range(len(ORDER)),[l for _,l in ORDER],fontsize=8)
plt.axhline(0,color="k",lw=0.8,ls="--"); plt.ylim(-1.2,1.05)
plt.ylabel("还能不能追踪到这个球(held R²)"); plt.xlabel("把目标球换成这个颜色(几何逐像素不变)")
plt.title("颜色调谐:探针在红球上学过之后,冻住不动,换目标球颜色",fontsize=12)
plt.legend(ncol=5,fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figs/color_tuning.png",dpi=150); print("-> figs/color_tuning.png",flush=True)
for arm,_ in ARMS: print(f"  {arm:5s}"+"".join(f"{v:8.3f}" for v in tab[arm]),flush=True)

# ---------- 图二:z 空间最近邻检索 ----------
D2=Path("data_distract"); r2=list(csv.DictReader(open(D2/"meta.csv")))
print(f"图二:编码 data_distract({len(r2)} 帧)...",flush=True)
Z2={arm:encode(ENC[arm],D2,r2) for arm,_ in ARMS}
b2=np.array([float(r["bear1"]) for r in r2]); g2=np.array([float(r["rng1"]) for r in r2])
QC="base"; qi=[i for i,r in enumerate(r2) if r["cond"]==QC and int(r["env"])==26 and int(r["round"])==0][0]
K=5
fig,ax=plt.subplots(len(ARMS),K+1,figsize=(2.1*(K+1),2.1*len(ARMS)),squeeze=False)
for i,(arm,_) in enumerate(ARMS):
    Z=Z2[arm]; zn=Z/ (np.linalg.norm(Z,axis=1,keepdims=True)+1e-9)
    d=1-zn@zn[qi]
    order=np.argsort(d)
    ax[i][0].imshow(read_image(str(D2/r2[qi]["img"])).permute(1,2,0).numpy())
    ax[i][0].set_xticks([]); ax[i][0].set_yticks([]); ax[i][0].set_ylabel(arm,fontsize=12)
    if i==0: ax[i][0].set_title("查询帧",fontsize=10)
    shown=0
    for j in order:
        if j==qi: continue
        ax[i][shown+1].imshow(read_image(str(D2/r2[j]["img"])).permute(1,2,0).numpy())
        ax[i][shown+1].set_xticks([]); ax[i][shown+1].set_yticks([])
        ax[i][shown+1].set_xlabel(f"{r2[j]['col2']}·{r2[j]['obj2']}\nΔ方位 {b2[j]-b2[qi]:+.1f}°",fontsize=7)
        if i==0: ax[i][shown+1].set_title(f"第 {shown+1} 近",fontsize=10)
        shown+=1
        if shown>=K: break
plt.suptitle("z 空间最近邻:它认为哪些帧是「同一个场景」(5760 帧,干扰物属性遍历)",fontsize=12)
plt.tight_layout(); plt.savefig("figs/retrieval.png",dpi=150); print("-> figs/retrieval.png",flush=True)
print("[DONE]",flush=True)
