"""donor 编码器 vs stage2 表征:是"长得一样"(特征空间线性等价)还是只是"表现一致"(头能读出同样的东西)。
用法: CUDA_VISIBLE_DEVICES=0 python donor_vs_final.py <donor_root> <final_root>
"""
import sys, csv, numpy as np, torch
sys.path.insert(0, "./src")
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import make_head, ARM_DIMS, ARMS
from pathlib import Path
DEV="cuda"; N=5000
droot=Path(sys.argv[1]); froot=Path(sys.argv[2])
tr=fr.load_transitions(); tm=np.load(froot/"train_mask.npy"); rng=np.random.default_rng(0)
idx=np.concatenate([rng.choice(np.flatnonzero(tm),N,replace=False), rng.choice(np.flatnonzero(~tm),N,replace=False)])
m=np.concatenate([np.ones(N,bool),np.zeros(N,bool)]); frames=fr.load_frames(tr["img"][idx])
@torch.no_grad()
def encode(path):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(path,map_location=DEV)); e.eval()
    return np.concatenate([e(frames[i:i+256].to(DEV))["shared_feature"].float().cpu().numpy() for i in range(0,len(frames),256)]).astype(np.float64)
def ridge_r2(f,y,lam=1e-2):  # 用 train 侧拟合、held 侧评估,y 可多维,返回各维 R² 的均值
    ftr,fte,ytr,yte=f[m],f[~m],y[m],y[~m]; mu,sd=ftr.mean(0),ftr.std(0)+1e-9
    a=np.hstack([(ftr-mu)/sd,np.ones((len(ftr),1))]); b=np.hstack([(fte-mu)/sd,np.ones((len(fte),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@ytr,rcond=None)[0]; p=b@w
    return float(np.mean(1-((yte-p)**2).sum(0)/((yte-yte.mean(0))**2).sum(0)+1e-12))
def cka(x,y):  # 线性 CKA,held 侧
    x=x[~m]-x[~m].mean(0); y=y[~m]-y[~m].mean(0)
    return float(np.linalg.norm(x.T@y)**2/(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)))
def head_mse(arm,f):
    h=make_head(256,ARM_DIMS[arm]).to(DEV); h.load_state_dict(torch.load(froot/arm/"head.pt",map_location=DEV)); h.eval()
    t=np.load(froot/arm/"target.npy")[idx]
    with torch.no_grad(): p=h(torch.as_tensor(f,dtype=torch.float32,device=DEV)).cpu().numpy()
    return float(((p-t)**2)[~m].mean())
print("== 复现性:重跑的 stage1 头 vs 原头(权重最大绝对差;为 0 说明 donor 重跑和原来一致) ==")
for a in ARMS:
    h0=torch.load(froot/a/"head.pt",map_location="cpu"); h1=torch.load(droot/a/"head.pt",map_location="cpu")
    print(f"  {a}: {max(float((h0[k]-h1[k]).abs().max()) for k in h0):.2e}")
i0=torch.load(froot/"encoder_init.pt",map_location="cpu"); i1=torch.load(droot/"encoder_init.pt",map_location="cpu")
print(f"  encoder_init 最大差: {max(float((i0[k].float()-i1[k].float()).abs().max()) for k in i0 if i0[k].dtype.is_floating_point):.2e}\n")
feats={"init":encode(froot/"encoder_init.pt")}
for a in ARMS: feats[f"donor_{a}"]=encode(droot/a/"donor.pt"); feats[f"final_{a}"]=encode(froot/a/"encoder_final.pt")
print("== 同臂:donor vs stage2 表征 ==")
print(f"{'arm':4s}{'CKA':>7s}{'donor→final R²':>15s}{'final→donor R²':>15s}{'rank donor':>11s}{'rank final':>11s}{'头MSE donor':>12s}{'头MSE final':>12s}")
for a in ARMS:
    d,f=feats[f"donor_{a}"],feats[f"final_{a}"]
    print(f"{a:4s}{cka(d,f):7.3f}{ridge_r2(d,f):15.3f}{ridge_r2(f,d):15.3f}{fr.eff_rank(d[~m]):11.2f}{fr.eff_rank(f[~m]):11.2f}{head_mse(a,d):12.4f}{head_mse(a,f):12.4f}")
print("\n== 对照:跨臂 / 对 init 的 CKA(同一行的量尺) ==")
names=["init"]+[f"final_{a}" for a in ARMS]+[f"donor_{a}" for a in ARMS]
print(f"{'':9s}"+"".join(f"{n:>9s}" for n in names))
for n1 in names: print(f"{n1:9s}"+"".join(f"{cka(feats[n1],feats[n2]):9.3f}" for n2 in names))
print("\n== 头交叉读:A 的头读别的臂的表征(MSE,越小越通用) ==")
for a in ARMS: print(f"head_{a}: "+"  ".join(f"{n}={head_mse(a,feats[n]):.3f}" for n in ["init"]+[f"final_{b}" for b in ARMS]))
