"""方向敏感性:小球沿距离/角度方向动一点,z 的哪些维在响应?不敏感的维里还有没有信息?(CPU)
用状态接近的帧对估计雅可比 dz/d(方向),两套坐标各做一次。
用法: CUDA_VISIBLE_DEVICES="" python g4_jacobian.py <OBS> <RM>
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(10)
O,RM=sys.argv[1],sys.argv[2]; N=3000
DATA=Path(f"./data_g4_{O}_{RM}"); ROOT=Path(f"./models/vision/g4_{O}_{RM}")
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
rg=np.random.default_rng(0); idx=rg.choice(np.flatnonzero(seen),N,replace=False)
frames=fr.load_frames(tr["img"][idx],DATA)
px=tr["px_x"][idx].astype(np.float64); d=tr["dist"][idx].astype(np.float64)
fx=112/np.tan(np.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng_m=d/np.cos(np.radians(bear)); diam=62.12/d
f=frames.float(); R_,G_,B_=f[:,0],f[:,1],f[:,2]
blue=(B_>90)&(B_>R_*1.8)&(B_>G_*1.8); bsz=blue.flatten(1).sum(1).float().numpy().astype(np.float64)
xs=torch.arange(224).view(1,1,224).expand(len(f),224,224).float()
bx=((blue.float()*xs).flatten(1).sum(1)/blue.flatten(1).sum(1).clamp(min=1)).numpy().astype(np.float64)
gray=f.mean(1).flatten(1).mean(1).numpy().astype(np.float64)   # 整帧亮度,当作"场景无关量"的代理
COORD={"物理(方位角°,距离m)":(bear,rng_m),"相机(x_c px,直径 px)":(px,diam)}
ii,jj=np.triu_indices(N,1); s=rg.choice(len(ii),2_000_000,replace=False); ii,jj=ii[s],jj[s]
@torch.no_grad()
def enc(p):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return np.concatenate([e(frames[i:i+64])["shared_feature"].float().numpy() for i in range(0,len(frames),64)]).astype(np.float64)
def ridge_r2(F,y,tr_m,lam=1e-2):
    if F.shape[1]==0: return float('nan')
    mu,sd=F[tr_m].mean(0),F[tr_m].std(0)+1e-9
    a=np.hstack([(F[tr_m]-mu)/sd,np.ones((tr_m.sum(),1))]); b=np.hstack([(F[~tr_m]-mu)/sd,np.ones(((~tr_m).sum(),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@y[tr_m],rcond=None)[0]
    p=b@w; yy=y[~tr_m]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
half=np.zeros(N,bool); half[:N//2]=True
print(f"== {O}_{RM} ==",flush=True)
for name,p in [("init",ROOT/"encoder_init.pt")]+[(a,ROOT/a/"donor.pt") for a in ("A","V","R","sup")]:
    if not p.exists(): continue
    Z=enc(p); Zs=(Z-Z.mean(0))/(Z.std(0)+1e-9)
    print(f"\n--- {name} ---",flush=True)
    for cname,(u,v) in COORD.items():
        du=u[ii]-u[jj]; dv=v[ii]-v[jj]
        su,sv=u.std(),v.std(); near=(np.abs(du)/su<0.15)&(np.abs(dv)/sv<0.15)
        a_,b_=du[near],dv[near]; dZ=Zs[ii[near]]-Zs[jj[near]]
        G=np.array([[ (a_*a_).sum(),(a_*b_).sum()],[(a_*b_).sum(),(b_*b_).sum()]])
        J=np.linalg.solve(G, np.stack([a_@dZ, b_@dZ]))          # (2,256) 每个方向的响应向量
        ja,jb=np.abs(J[0]),np.abs(J[1])
        cos=float(J[0]@J[1]/(np.linalg.norm(J[0])*np.linalg.norm(J[1])+1e-12))
        top=lambda x,k=8: np.sort(x)[::-1][:k].sum()/x.sum()
        ka=np.argsort(ja)[::-1][:8]; kb=np.argsort(jb)[::-1][:8]
        ins=np.flatnonzero((ja<0.05*ja.max())&(jb<0.05*jb.max()))
        print(f"  {cname}  对={near.sum()}  两方向响应向量夹角 cos={cos:+.2f}  前8维占能量 {top(ja):.0%}/{top(jb):.0%}  "
              f"top8 重合 {len(set(ka)&set(kb))}/8  两方向都不敏感的维 {len(ins)}/256",flush=True)
        if len(ins)>=5:
            print(f"      从这 {len(ins)} 个不敏感维读:x_c {ridge_r2(Z[:,ins],px,half):+.3f}  距离 {ridge_r2(Z[:,ins],rng_m,half):+.3f}  "
                  f"蓝球位置 {ridge_r2(Z[:,ins],bx,half):+.3f}  蓝球大小 {ridge_r2(Z[:,ins],bsz,half):+.3f}  帧亮度 {ridge_r2(Z[:,ins],gray,half):+.3f}",flush=True)
