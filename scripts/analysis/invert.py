"""特征反演(Mahendran & Vedaldi, CVPR 2015):固定 z_target = E(原图),
优化一张图 x 使 E(x) 逼近 z_target,加 TV 正则。反解出来的图里有什么 = z 里留了什么。
不需要探针、不需要扰动、不需要归因。GPU0(纯 torch,不开 Isaac)。
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
ap=argparse.ArgumentParser()
ap.add_argument("--data", default="data_distract")
ap.add_argument("--conds", default="base,d_red_ball,d_none")
ap.add_argument("--env", type=int, default=26); ap.add_argument("--round", type=int, default=0)
ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--tv", type=float, default=2e-3)
ap.add_argument("--out", default="figs/inversion.png")
a=ap.parse_args()
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"]=font_manager.FontProperties(fname=f).get_name(); break
dev="cuda" if torch.cuda.is_available() else "cpu"
ROOT=Path("models/vision/g4_diam_angle"); D=Path(a.data)
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
rows=list(csv.DictReader(open(D/"meta.csv")))
def tv(x):
    return ((x[:,:,1:,:]-x[:,:,:-1,:])**2).mean()+((x[:,:,:,1:]-x[:,:,:,:-1])**2).mean()
def invert(e,img):
    with torch.no_grad():
        zt=e(img[None].to(dev))["shared_feature"].float()
    x=torch.full((1,3,224,224),0.75,device=dev)+0.02*torch.randn(1,3,224,224,device=dev)
    x=x.clamp(0,1).requires_grad_(True)
    opt=torch.optim.Adam([x],lr=0.05)
    for t in range(a.steps):
        z=e((x.clamp(0,1)*255.0))["shared_feature"].float()
        loss=((z-zt)**2).sum()/ (zt**2).sum() + a.tv*tv(x)
        opt.zero_grad(); loss.backward(); opt.step()
        if t%300==0: print(f"    step {t:4d} loss {loss.item():.4f}",flush=True)
    return x.detach().clamp(0,1)[0].cpu().permute(1,2,0).numpy(), float(loss.item())
sel=[]
for c in a.conds.split(","):
    h=[r for r in rows if r["cond"]==c and int(r["env"])==a.env and int(r["round"])==a.round]
    if h: sel.append((c,h[0]))
fig,ax=plt.subplots(len(sel),len(ARMS)+1,figsize=(2.5*(len(ARMS)+1),2.5*len(sel)),squeeze=False)
for j,(arm,rel) in enumerate(ARMS):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(ROOT/rel,map_location="cpu"))
    e.eval().to(dev)
    for p in e.parameters(): p.requires_grad_(False)
    for i,(c,r) in enumerate(sel):
        img=read_image(str(D/r["img"])).float()
        if j==0:
            ax[i][0].imshow(img.permute(1,2,0).numpy()/255.0); ax[i][0].set_xticks([]); ax[i][0].set_yticks([])
            ax[i][0].set_ylabel(f"{c}\n干扰={r['col2']}·{r['obj2']}",fontsize=8)
            if i==0: ax[i][0].set_title("原图",fontsize=11)
        print(f"  {arm} / {c}",flush=True)
        rec,l=invert(e,img)
        ax[i][j+1].imshow(rec); ax[i][j+1].set_xticks([]); ax[i][j+1].set_yticks([])
        ax[i][j+1].set_xlabel(f"loss {l:.3f}",fontsize=7)
        if i==0: ax[i][j+1].set_title(arm,fontsize=11)
plt.suptitle("特征反演:从 z 反解图像 —— 反出来有什么,就是 z 里留了什么",fontsize=12)
plt.tight_layout(); Path("figs").mkdir(exist_ok=True); plt.savefig(a.out,dpi=145)
print(f"[DONE] -> {a.out}",flush=True)
