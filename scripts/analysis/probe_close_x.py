import sys, numpy as np, torch
sys.path.insert(0, "./src")
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
DEVICE="cuda"; N=6000
root=fr.OUT_ROOT; tr=fr.load_transitions(); tm=np.load(root/"train_mask.npy")
rng=np.random.default_rng(0)
seen=tr["dist"]>fr.LOST_D
# 训练帧:均匀采;测试帧:近处多采一些,保证每个距离段都有 ≥800 帧
tr_idx=rng.choice(np.flatnonzero(tm&seen),N,replace=False)
te_pool=np.flatnonzero((~tm)&seen); d_te=tr["dist"][te_pool]
te_idx=np.concatenate([rng.choice(te_pool[(d_te>=1.3)&(d_te<1.7)],1500,replace=False),
                       rng.choice(te_pool[(d_te>=1.7)&(d_te<2.5)],1500,replace=False),
                       rng.choice(te_pool[(d_te>=2.5)],1500,replace=False)])
idx=np.concatenate([tr_idx,te_idx]); m=np.concatenate([np.ones(len(tr_idx),bool),np.zeros(len(te_idx),bool)])
y=(tr["px_x"][idx]-112.0); d=tr["dist"][idx]
frames=fr.load_frames(tr["img"][idx])
@torch.no_grad()
def encode(path):
    enc=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEVICE); enc.load_state_dict(torch.load(path,map_location=DEVICE)); enc.eval()
    return np.concatenate([enc(frames[i:i+256].to(DEVICE))["shared_feature"].float().cpu().numpy() for i in range(0,len(frames),256)]).astype(np.float64)
def ridge_fit(f,y,lam=1e-2):
    mu,sd=f.mean(0),f.std(0)+1e-9; a=np.hstack([(f-mu)/sd,np.ones((len(f),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@y,rcond=None)[0]
    return lambda g: np.hstack([(g-mu)/sd,np.ones((len(g),1))])@w
bins=[(1.3,1.7,"1.3-1.7m"),(1.7,2.5,"1.7-2.5m"),(2.5,99,">2.5m")]
print("像素偏差 x 的 256 维线性读出:在全距离训练帧上拟合,按距离段看 held-out RMSE(px)与 R²")
print(f"{'arm':6s}"+"".join(f"{'RMSE '+b[2]:>14s}{'R²':>7s}" for b in bins)+f"{'|y| std@近':>10s}")
for arm in ["init","A","V","R","sup"]:
    p=root/"encoder_init.pt" if arm=="init" else root/arm/"encoder_final.pt"
    f=encode(p); pred=ridge_fit(f[m],y[m])(f[~m]); yt=y[~m]; dt=d[~m]
    row=f"{arm:6s}"
    for lo,hi,name in bins:
        s=(dt>=lo)&(dt<hi); e=pred[s]-yt[s]
        row+=f"{np.sqrt((e**2).mean()):14.2f}{1-(e**2).sum()/((yt[s]-yt[s].mean())**2).sum():7.3f}"
    s=(dt>=1.3)&(dt<1.7); row+=f"{yt[s].std():10.1f}"
    print(row,flush=True)
