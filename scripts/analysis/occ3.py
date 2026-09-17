"""遮挡式表征重要度图(修正版):
  目标 = 表征本身 ‖z(遮) − z(原)‖ / ‖z(原)‖,恒为正、不需要探针、无符号问题。
  填充 = 该图**中位颜色**(就是地板/墙的灰)。上一版填固定灰导致"背景贴灰块=新增物体",图糊了。
  分数累加到方块覆盖的像素再求均值 -> 逐像素图。CPU。
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
ap.add_argument("--conds", default="base,d_red_ball,d_red_cube,d_red_cone,d_green,d_none")
ap.add_argument("--env", type=int, default=26); ap.add_argument("--round", type=int, default=0)
ap.add_argument("--patch", type=int, default=20); ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--out", default="figs/occlusion_v3.png")
a=ap.parse_args()
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"]=font_manager.FontProperties(fname=f).get_name(); break
ROOT=Path("models/vision/g4_diam_angle"); D=Path(a.data)
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
rows=list(csv.DictReader(open(D/"meta.csv")))
POS=[(y,x) for y in range(0,224-a.patch+1,a.stride) for x in range(0,224-a.patch+1,a.stride)]
print(f"{len(POS)} 个遮挡位置(方块 {a.patch}px,步长 {a.stride}px)",flush=True)
from torchvision.transforms.functional import gaussian_blur
@torch.no_grad()
def imap(e,img):
    blur=gaussian_blur(img.float()[None],kernel_size=51,sigma=18.0)[0]  # 模糊填充:空地板本来就平滑->等于没动
    base=e(img[None])["shared_feature"].float(); nb=torch.norm(base)+1e-9
    acc=np.zeros((224,224)); cnt=np.zeros((224,224))
    for i in range(0,len(POS),64):
        ch=POS[i:i+64]
        b=img[None].float().repeat(len(ch),1,1,1).clone()
        for k,(y,x) in enumerate(ch): b[k,:,y:y+a.patch,x:x+a.patch]=blur[:,y:y+a.patch,x:x+a.patch]
        d=(torch.norm(e(b)["shared_feature"].float()-base,dim=1)/nb).numpy()
        for k,(y,x) in enumerate(ch):
            acc[y:y+a.patch,x:x+a.patch]+=d[k]; cnt[y:y+a.patch,x:x+a.patch]+=1
    return acc/np.maximum(cnt,1)
sel=[]
for c in a.conds.split(","):
    h=[r for r in rows if r["cond"]==c and int(r["env"])==a.env and int(r["round"])==a.round]
    if h: sel.append((c,h[0]))
ENC={}
for arm,rel in ARMS:
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(ROOT/rel,map_location="cpu")); e.eval(); ENC[arm]=e
fig,ax=plt.subplots(len(sel),len(ARMS)+1,figsize=(2.5*(len(ARMS)+1),2.5*len(sel)),squeeze=False)
stat={}
for i,(c,r) in enumerate(sel):
    img=read_image(str(D/r["img"])); rgb=img.permute(1,2,0).numpy()
    ax[i][0].imshow(rgb); ax[i][0].set_xticks([]); ax[i][0].set_yticks([])
    ax[i][0].set_ylabel(f"{c}\n干扰={r['col2']}·{r['obj2']}",fontsize=8)
    if i==0: ax[i][0].set_title("输入",fontsize=11)
    for j,(arm,_) in enumerate(ARMS):
        h=imap(ENC[arm],img)
        bg=np.median(h)                      # 背景基线:填中位色后遮空地应接近它
        hh=np.maximum(h-bg,0); hh=hh/(hh.max()+1e-9)
        ax[i][j+1].imshow(rgb); ax[i][j+1].imshow(hh,cmap="jet",alpha=0.5,vmin=0,vmax=1)
        ax[i][j+1].set_xticks([]); ax[i][j+1].set_yticks([])
        if i==0: ax[i][j+1].set_title(arm,fontsize=11)
        stat[(c,arm)]=(h.max(),bg,h.max()/max(bg,1e-9))
        print(f"{c:12s}{arm:5s} 峰值 {h.max():.4f}  背景中位 {bg:.4f}  峰/背 {h.max()/max(bg,1e-9):.1f}",flush=True)
plt.suptitle("遮挡该处后表征的变化 ‖Δz‖/‖z‖(已减去背景基线;模糊填充)",fontsize=12)
plt.tight_layout(); Path("figs").mkdir(exist_ok=True); plt.savefig(a.out,dpi=140)
print(f"[DONE] -> {a.out}",flush=True)
