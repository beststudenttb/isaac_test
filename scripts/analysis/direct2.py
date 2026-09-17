"""不拟合、不扰动、不求梯度:直接用表征网络自己的权重算每个位置对 z 的贡献。
z = LayerNorm(proj(pooled)),pooled = 三个尺度各自的 7×7 网格平均。
所以 FPN 特征图上像素 (y,x) 对 z(LayerNorm 之前)的贡献向量就是
    W[:, 该像素所属网格格子的那 128 个通道列] @ p[:,y,x] / (格子像素数)
取它的范数就是"这个位置把 z 推了多远"。一次前向 + 一次矩阵乘,精确值。CPU。
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
torch.set_num_threads(6)
ap=argparse.ArgumentParser()
ap.add_argument("--data", default="data_distract")
ap.add_argument("--conds", default="base,d_red_ball,d_red_cube,d_red_cone,d_green,d_white,d_none")
ap.add_argument("--env", type=int, default=26); ap.add_argument("--round", type=int, default=0)
ap.add_argument("--out", default="figs/direct_contrib_v2.png")
a=ap.parse_args()
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"]=font_manager.FontProperties(fname=f).get_name(); break
ROOT=Path("models/vision/g4_diam_angle"); D=Path(a.data); G=7; C=FREE_SPATIAL_CONFIG["fpn_channels"]
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
rows=list(csv.DictReader(open(D/"meta.csv")))
@torch.no_grad()
def fpn(e,x):
    x=e._to_nchw(x); x=(x-e.imagenet_mean)/e.imagenet_std
    s=e.stem(x); c2=e.layer1(s); c3=e.layer2(c2); c4=e.layer3(c3)
    p4=e.lat4(c4); p3=e.lat3(c3)+F.interpolate(p4,size=c3.shape[-2:],mode="nearest")
    p2=e.lat2(c2)+F.interpolate(p3,size=c2.shape[-2:],mode="nearest")
    return [e.smooth2(p2), e.smooth3(p3), e.smooth4(p4)]
@torch.no_grad()
def contrib(e,img):
    W=e.proj.weight                                   # (256, 3*C*49)
    ps=fpn(e,img[None].float())
    out=0.0
    for s,p in enumerate(ps[:1]):                     # 只用 p2(56×56),混入 p3/p4 会被粗尺度拖糊
        H=p.shape[-1]; k=H//G                         # 每格 k×k 个像素
        Ws=W[:, s*C*G*G:(s+1)*C*G*G].reshape(-1,C,G,G)   # (256,C,7,7)
        # 每个像素 -> 它所在的格子 (i,j) -> 取 Ws[:,:,i,j] (256,C),与 p[:,:,y,x] 相乘
        idx_y=torch.arange(H)//k; idx_x=torch.arange(H)//k
        Wpix=Ws[:,:,idx_y][:,:,:,idx_x]               # (256,C,H,W)
        v=torch.einsum("dchw,chw->dhw",Wpix,p[0])/ (k*k)
        m=v.norm(dim=0,keepdim=True)[None]            # (1,1,H,W)
        out=out+F.interpolate(m,size=(224,224),mode="bilinear",align_corners=False)[0,0]
    return out.numpy()
sel=[]
for c in a.conds.split(","):
    h=[r for r in rows if r["cond"]==c and int(r["env"])==a.env and int(r["round"])==a.round]
    if h: sel.append((c,h[0]))
fig,ax=plt.subplots(len(sel),len(ARMS)+1,figsize=(2.5*(len(ARMS)+1),2.5*len(sel)),squeeze=False)
for j,(arm,rel) in enumerate(ARMS):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(ROOT/rel,map_location="cpu")); e.eval()
    for i,(c,r) in enumerate(sel):
        img=read_image(str(D/r["img"])); rgb=img.permute(1,2,0).numpy()
        if j==0:
            ax[i][0].imshow(rgb); ax[i][0].set_xticks([]); ax[i][0].set_yticks([])
            ax[i][0].set_ylabel(f"{c}\n干扰={r['col2']}·{r['obj2']}",fontsize=8)
            if i==0: ax[i][0].set_title("输入",fontsize=11)
        h=contrib(e,img)
        lo,hi=np.percentile(h,70),np.percentile(h,99.5)   # 扣背景基线 + 百分位截断
        h=np.clip((h-lo)/max(hi-lo,1e-9),0,1)
        ax[i][j+1].imshow(rgb); ax[i][j+1].imshow(h,cmap="jet",alpha=0.6,vmin=0,vmax=1)
        ax[i][j+1].set_xticks([]); ax[i][j+1].set_yticks([])
        if i==0: ax[i][j+1].set_title(arm,fontsize=11)
        print(f"{c:12s}{arm:5s} ok",flush=True)
plt.suptitle("每个位置把 z 推了多远(只用 FPN p2 56×56,已扣背景基线,百分位 70–99.5 截断)",fontsize=12)
plt.tight_layout(); Path("figs").mkdir(exist_ok=True); plt.savefig(a.out,dpi=145)
print(f"[DONE] -> {a.out}",flush=True)
