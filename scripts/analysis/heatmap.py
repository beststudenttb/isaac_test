"""空间贡献热图(spatial contribution map,不是 attention —— 网络里没有 attention 层)。
架构上 z 的每一维都能精确追溯到"哪个 FPN 尺度的哪一格 7×7",所以用 梯度×输入 对 pooled 特征做归因:
    贡献(尺度 s, 格 i,j) = Σ_c  ∂(探针输出)/∂pooled[s,c,i,j] × pooled[s,c,i,j]
探针 = 在该数据集 base 条件上拟合的"红球方位角"。CPU。
"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(8)
ROOT=Path("models/vision/g4_diam_angle")
OUT=Path("/home/tb/Downloads/isaac_test/figs"); OUT.mkdir(exist_ok=True)
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
# (数据集, 条件, 图上标题)
PICKS=[("data_identity","base","红球 + 蓝球"),
       ("data_identity","two_red","红球 + 红球"),
       ("data_shape","two_red_sh","红球 + 红立方")]
ENV_K=7   # 固定看同一个 env,便于横向比较
def load(ds):
    rows=list(csv.DictReader(open(Path(ds)/"meta.csv")))
    return rows
def fit_probe(ds,rows,arm,enc):
    g=lambda k: np.array([float(r[k]) for r in rows])
    cond=np.array([r["cond"] for r in rows]); rd=g("round")
    b1=g("bear1"); m=(cond=="base")&(rd%2==0)
    idx=np.flatnonzero(m)
    Z=[]
    with torch.no_grad():
        for i in range(0,len(idx),48):
            fr=torch.stack([read_image(str(Path(ds)/rows[j]["img"])) for j in idx[i:i+48]])
            Z.append(enc(fr)["shared_feature"].float())
    Z=torch.cat(Z).numpy().astype(np.float64)
    mu,sd=Z.mean(0),Z.std(0)+1e-9
    A=np.hstack([(Z-mu)/sd,np.ones((len(Z),1))])
    w=np.linalg.lstsq(A.T@A+3e-2*len(A)*np.eye(A.shape[1]),A.T@b1[idx],rcond=None)[0]
    return torch.tensor(w[:-1]/sd,dtype=torch.float32), torch.tensor(mu,dtype=torch.float32)
def contrib(enc,w,mu,img):
    x=img[None].float()
    pooled=enc.forward_pooled(x).detach().requires_grad_(True)
    z=enc.norm(enc.proj(pooled))
    out=((z[0]-mu)*w).sum()
    gsum,=torch.autograd.grad(out,pooled)
    c=(gsum*pooled).detach()[0].view(3,FREE_SPATIAL_CONFIG["fpn_channels"],7,7).sum(1)
    return c.sum(0).numpy()          # 三个尺度相加 -> 7×7
fig,axes=plt.subplots(len(PICKS),len(ARMS)+1,figsize=(3.0*(len(ARMS)+1),3.0*len(PICKS)))
for r_i,(ds,cd,title) in enumerate(PICKS):
    rows=load(ds)
    sel=[r for r in rows if r["cond"]==cd and int(r["env"])==ENV_K and int(r["round"])==1]
    if not sel: print("缺帧",ds,cd); continue
    img=read_image(str(Path(ds)/sel[0]["img"]))
    axes[r_i,0].imshow(img.permute(1,2,0).numpy()); axes[r_i,0].set_title(title,fontsize=11)
    axes[r_i,0].axis("off")
    for a_i,(arm,rel) in enumerate(ARMS):
        e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG)
        e.load_state_dict(torch.load(ROOT/rel,map_location="cpu")); e.eval()
        w,mu=fit_probe(ds,rows,arm,e)
        h=contrib(e,w,mu,img)
        ax=axes[r_i,a_i+1]
        ax.imshow(img.permute(1,2,0).numpy(),alpha=0.55)
        v=np.abs(h).max()+1e-9
        ax.imshow(np.kron(h,np.ones((32,32))),cmap="bwr",vmin=-v,vmax=v,alpha=0.55)
        ax.set_title(arm,fontsize=11); ax.axis("off")
        print(f"{ds}/{cd} {arm} 贡献范围 [{h.min():.3f},{h.max():.3f}]",flush=True)
plt.suptitle("对『红球方位角』读出的空间贡献(红=正向贡献,蓝=负向;7×7 网格)",fontsize=13)
plt.tight_layout(); plt.savefig(OUT/"heatmap_distractor.png",dpi=130)
print(f"\n[DONE] -> {OUT/'heatmap_distractor.png'}",flush=True)
