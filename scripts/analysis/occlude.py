"""遮挡法的表征重要度图:不用探针,直接量表征本身。
在图上滑一个方块把它涂成灰色,看 ‖z_遮 − z_原‖ / ‖z_原‖ 有多大 —— 球被遮住时表征大幅改变,
背景被遮住几乎不动,所以出来是块状清晰的"表征在乎哪里"的图,不是梯度那种糊的。
分数累加到方块覆盖的像素上再求平均,所以是逐像素的。CPU。
"""
import sys, csv, argparse, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(10)
ap=argparse.ArgumentParser()
ap.add_argument("--data", default="data_distract")
ap.add_argument("--conds", default="base,d_red_ball,d_red_cube,d_red_cone,d_orange,d_green,d_white,d_none")
ap.add_argument("--env", type=int, default=7)
ap.add_argument("--round", type=int, default=1)
ap.add_argument("--patch", type=int, default=24)
ap.add_argument("--stride", type=int, default=12)
ap.add_argument("--out", default="figs/occlusion_distract.png")
a=ap.parse_args()
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"]=font_manager.FontProperties(fname=f).get_name(); break
ROOT=Path("models/vision/g4_diam_angle"); D=Path(a.data)
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
rows=list(csv.DictReader(open(D/"meta.csv")))
POS=[(y,x) for y in range(0,224-a.patch+1,a.stride) for x in range(0,224-a.patch+1,a.stride)]
print(f"每张图 {len(POS)} 个遮挡位置(方块 {a.patch}px,步长 {a.stride}px)",flush=True)
@torch.no_grad()
def imap(e,img):
    base=e(img[None])["shared_feature"].float()
    acc=np.zeros((224,224)); cnt=np.zeros((224,224))
    for i in range(0,len(POS),64):
        chunk=POS[i:i+64]
        batch=img[None].repeat(len(chunk),1,1,1).clone()
        for k,(y,x) in enumerate(chunk): batch[k,:,y:y+a.patch,x:x+a.patch]=128
        z=e(batch)["shared_feature"].float()
        d=(torch.norm(z-base,dim=1)/(torch.norm(base)+1e-9)).numpy()
        for k,(y,x) in enumerate(chunk):
            acc[y:y+a.patch,x:x+a.patch]+=d[k]; cnt[y:y+a.patch,x:x+a.patch]+=1
    return acc/np.maximum(cnt,1)
sel=[]
for c in a.conds.split(","):
    hit=[r for r in rows if r["cond"]==c and int(r["env"])==a.env and int(r["round"])==a.round]
    if hit: sel.append((c,hit[0]))
ENC={}
for arm,rel in ARMS:
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(ROOT/rel,map_location="cpu")); e.eval(); ENC[arm]=e
n=len(sel)
fig,axes=plt.subplots(n,len(ARMS)+1,figsize=(2.4*(len(ARMS)+1),2.4*n),squeeze=False)
for i,(c,r) in enumerate(sel):
    img=read_image(str(D/r["img"])); rgb=img.permute(1,2,0).numpy()
    axes[i][0].imshow(rgb); axes[i][0].set_xticks([]); axes[i][0].set_yticks([])
    axes[i][0].set_ylabel(f"{c}\n干扰={r.get('col2','-')}·{r.get('obj2','-')}",fontsize=8)
    if i==0: axes[i][0].set_title("输入",fontsize=11)
    for j,(arm,_) in enumerate(ARMS):
        h=imap(ENC[arm],img); h=h/(h.max()+1e-9)
        ax=axes[i][j+1]; ax.imshow(rgb); ax.imshow(h,cmap="jet",alpha=0.55,vmin=0,vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i==0: ax.set_title(arm,fontsize=11)
        print(f"{c:14s}{arm:5s} 峰值 {h.max():.3f}",flush=True)
plt.suptitle("遮挡法:遮住这块,表征 ‖Δz‖/‖z‖ 变化多少(越红=表征越在乎)",fontsize=12)
plt.tight_layout(); Path(a.out).parent.mkdir(exist_ok=True); plt.savefig(a.out,dpi=140)
print(f"[DONE] -> {a.out}",flush=True)
