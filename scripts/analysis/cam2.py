"""两种归因图并排:
  LayerCAM(Jiang et al. 2021)在 FPN p2(56×56)上 —— 逐像素梯度权重,专为浅层设计,分辨率比
    Grad-CAM 的 7×7 细 8 倍。map = ReLU( Σ_c ReLU(∂y/∂A_c) ⊙ A_c )
  RISE(Petsiuk et al. BMVC 2018)—— 随机多尺度掩码扰动,无梯度。这里用删除式:
    saliency(p) = E[ |读出(遮住p) − 读出(原图)| ],掩码填该图中位颜色(=地板灰),避免"贴灰块=新增物体"。
两者原理完全不同(梯度 vs 扰动),一致才可信。y = 冻结的"红球方位角"读出。CPU。
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
torch.set_num_threads(10)
ap=argparse.ArgumentParser()
ap.add_argument("--data", default="data_distract")
ap.add_argument("--cam-conds", default="base,d_red_ball,d_red_cube,d_red_cone,d_orange,d_green,d_white,d_none")
ap.add_argument("--rise-conds", default="base,d_red_ball,d_red_cube,d_none")
ap.add_argument("--env", type=int, default=26)
ap.add_argument("--round", type=int, default=0)
ap.add_argument("--nmask", type=int, default=800)
a=ap.parse_args()
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"]=font_manager.FontProperties(fname=f).get_name(); break
ROOT=Path("models/vision/g4_diam_angle"); D=Path(a.data)
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
rows=list(csv.DictReader(open(D/"meta.csv")))
cond=np.array([r["cond"] for r in rows]); rd=np.array([int(r["round"]) for r in rows])
b1=np.array([float(r["bear1"]) for r in rows])
def fpn(e,x):
    x=e._to_nchw(x); x=(x-e.imagenet_mean)/e.imagenet_std
    s=e.stem(x); c2=e.layer1(s); c3=e.layer2(c2); c4=e.layer3(c3)
    p4=e.lat4(c4); p3=e.lat3(c3)+F.interpolate(p4,size=c3.shape[-2:],mode="nearest")
    p2=e.lat2(c2)+F.interpolate(p3,size=c2.shape[-2:],mode="nearest")
    return e.smooth2(p2), e.smooth3(p3), e.smooth4(p4)
def z_from(e,ps): return e.norm(e.proj(torch.cat([e._grid_pool(p) for p in ps],dim=1)))
def fit(e):
    m=(cond=="base")&(rd%2==0); idx=np.flatnonzero(m); Z=[]
    with torch.no_grad():
        for i in range(0,len(idx),48):
            Z.append(e(torch.stack([read_image(str(D/rows[j]["img"])) for j in idx[i:i+48]]))["shared_feature"].float())
    Z=torch.cat(Z).numpy().astype(np.float64); mu,sd=Z.mean(0),Z.std(0)+1e-9
    A_=np.hstack([(Z-mu)/sd,np.ones((len(Z),1))])
    w=np.linalg.lstsq(A_.T@A_+3e-2*len(A_)*np.eye(A_.shape[1]),A_.T@b1[idx],rcond=None)[0]
    return torch.tensor(w[:-1]/sd,dtype=torch.float32), torch.tensor(mu,dtype=torch.float32), float(w[-1])
def layercam(e,w,mu,img):
    ps=[p.detach().requires_grad_(True) for p in fpn(e,img[None].float())]
    y=((z_from(e,ps)[0]-mu)*w).sum()
    gs=torch.autograd.grad(y,ps)
    m=F.relu((F.relu(gs[0])*ps[0]).sum(1,keepdim=True))       # 只用 p2(56×56)
    return F.interpolate(m,size=(224,224),mode="bilinear",align_corners=False)[0,0].detach().numpy()
@torch.no_grad()
def readout(e,w,mu,batch): return ((e(batch)["shared_feature"].float()-mu)*w).sum(1)
@torch.no_grad()
def rise(e,w,mu,img,N,cell=8,p=0.5,seed=0):
    g=torch.Generator().manual_seed(seed)
    med=img.float().flatten(1).median(1).values.view(3,1,1)
    y0=readout(e,w,mu,img[None].float())[0]
    sal=torch.zeros(224,224); cnt=torch.zeros(224,224)
    up=224//cell+1
    for i in range(0,N,64):
        n=min(64,N-i)
        grid=(torch.rand(n,1,cell,cell,generator=g)<p).float()
        big=F.interpolate(grid,size=(up*cell,up*cell),mode="bilinear",align_corners=False)
        ox=torch.randint(0,cell,(n,),generator=g); oy=torch.randint(0,cell,(n,))
        M=torch.stack([big[k,0,oy[k]:oy[k]+224,ox[k]:ox[k]+224] for k in range(n)])
        batch=img.float()[None]*M[:,None]+med[None]*(1-M[:,None])
        d=(readout(e,w,mu,batch)-y0).abs()
        hid=1-M
        sal+=(hid*d[:,None,None]).sum(0); cnt+=hid.sum(0)
    return (sal/torch.clamp(cnt,min=1)).numpy()
ENC={}; PRB={}
for arm,rel in ARMS:
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(ROOT/rel,map_location="cpu")); e.eval()
    ENC[arm]=e; PRB[arm]=fit(e)[:2]
def pick(c):
    h=[r for r in rows if r["cond"]==c and int(r["env"])==a.env and int(r["round"])==a.round]
    return h[0] if h else None
def draw(conds,fn,title,func):
    sel=[(c,pick(c)) for c in conds.split(",")]; sel=[(c,r) for c,r in sel if r]
    fig,ax=plt.subplots(len(sel),len(ARMS)+1,figsize=(2.4*(len(ARMS)+1),2.4*len(sel)),squeeze=False)
    for i,(c,r) in enumerate(sel):
        img=read_image(str(D/r["img"])); rgb=img.permute(1,2,0).numpy()
        ax[i][0].imshow(rgb); ax[i][0].set_xticks([]); ax[i][0].set_yticks([])
        ax[i][0].set_ylabel(f"{c}\n干扰={r['col2']}·{r['obj2']}",fontsize=8)
        if i==0: ax[i][0].set_title("输入",fontsize=11)
        for j,(arm,_) in enumerate(ARMS):
            h=func(ENC[arm],*PRB[arm],img); h=np.maximum(h,0); h=h/(h.max()+1e-9)
            ax[i][j+1].imshow(rgb); ax[i][j+1].imshow(h,cmap="jet",alpha=0.5,vmin=0,vmax=1)
            ax[i][j+1].set_xticks([]); ax[i][j+1].set_yticks([])
            if i==0: ax[i][j+1].set_title(arm,fontsize=11)
        print(f"{fn} {c} 行完成",flush=True)
    plt.suptitle(title,fontsize=12); plt.tight_layout()
    Path("figs").mkdir(exist_ok=True); plt.savefig(f"figs/{fn}",dpi=140); print(f"-> figs/{fn}",flush=True)
draw(a.cam_conds,"layercam_distract.png","LayerCAM @ FPN p2(56×56):对『红球方位角』读出的梯度归因",layercam)
draw(a.rise_conds,"rise_distract.png",f"RISE({a.nmask} 随机掩码,填地板色):遮住该处时读出改变多少",
     lambda e,w,mu,img: rise(e,w,mu,img,a.nmask))
print("[DONE]",flush=True)
