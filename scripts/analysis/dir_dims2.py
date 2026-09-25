"""重做:按图像像素变化归一的方向敏感度 + 等大小子集的解码对比。
敏感度 s_dim = |Δz_dim| / |图像上的像素变化|,距离方向用直径变化 px,角度方向用球心横移 px。
解码对比一律用同样维数(top-k vs bottom-k vs 随机 k)。
"""
import sys, csv, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
import free_repr as fr
torch.set_num_threads(12)
COMB=sys.argv[1] if len(sys.argv)>1 else "diam_angle"
ROOT=Path(f"models/vision/g4_{COMB}"); DATA=Path(f"./data_g4_{COMB}")
D=Path("data_probe_grid"); rows=list(csv.DictReader(open(D/"grid.csv")))
byc={}
for r in rows: byc.setdefault(r["cond"],{})[int(r["env"])]=r
envs=sorted(byc["base"])
CONDS=("base","r+0.05","r+0.30","b+1","b+5")
gf={c: torch.stack([read_image(str(D/byc[c][e]["img"])) for e in envs]) for c in CONDS}
def arr(c,k): return np.array([float(byc[c][e][k]) for e in envs])
diam=lambda c: 62.12/np.maximum(arr(c,"dist"),1e-6)
# 每个网格点上,两种扰动在图像上分别造成多少像素变化
dpx_r=np.abs(diam("r+0.30")-diam("base"))          # 直径变化 px
dpx_b=np.abs(arr("b+5","px_x")-arr("base","px_x")) # 球心横移 px
rng0=arr("base","rng")
BANDS=[("近<2.2m",rng0<2.2),("中2.2-4.5",(rng0>=2.2)&(rng0<4.5)),("远>4.5",rng0>=4.5)]
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
m_all=fr.episode_split(tr["episode"],seed=0); rg=np.random.default_rng(0); N=3000
idx=np.concatenate([rg.choice(np.flatnonzero(m_all&seen),N,replace=False), rg.choice(np.flatnonzero((~m_all)&seen),N,replace=False)])
m=np.r_[np.ones(N,bool),np.zeros(N,bool)]; nf=fr.load_frames(tr["img"][idx],DATA)
px=tr["px_x"][idx].astype(np.float64); d=tr["dist"][idx].astype(np.float64)
fx=112/np.tan(np.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng_m=d/np.cos(np.radians(bear))
@torch.no_grad()
def enc(p,f,bs=32):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(f[i:i+bs])["shared_feature"].float() for i in range(0,len(f),bs)]).numpy().astype(np.float64)
def ridge(F,y,lam=1e-2):
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    a=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); b=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@y[m],rcond=None)[0]
    p=b@w; yy=y[~m]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
print(f"== {COMB} == 敏感度按图像像素变化归一;解码子集一律 16 维", flush=True)
print(f"平均像素变化:距离扰动 0.30m -> 直径变 {dpx_r.mean():.2f}px(近 {dpx_r[BANDS[0][1]].mean():.2f} / 远 {dpx_r[BANDS[2][1]].mean():.2f});角度扰动 5° -> 横移 {dpx_b.mean():.2f}px(近 {dpx_b[BANDS[0][1]].mean():.2f} / 远 {dpx_b[BANDS[2][1]].mean():.2f})\n", flush=True)
for name,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Zg={c: enc(p,gf[c]) for c in CONDS}; s=Zg["base"].std(0)+1e-9
    Sr=np.abs((Zg["r+0.30"]-Zg["base"])/s)/dpx_r[:,None]   # (64,256) 每像素的响应
    Sb=np.abs((Zg["b+5"]  -Zg["base"])/s)/dpx_b[:,None]
    sr,sb=Sr.mean(0),Sb.mean(0)
    Zn=enc(p,nf,64)
    ar=np.argsort(sr)[::-1]; ab=np.argsort(sb)[::-1]; rnd=rg.permutation(256)
    print(f"--- {name} ---   每像素总响应 距离 {np.linalg.norm(Sr,axis=1).mean():.3f} / 角度 {np.linalg.norm(Sb,axis=1).mean():.3f}", flush=True)
    print(f"      两 top16 重合 {len(set(ar[:16])&set(ab[:16]))}/16  敏感度相关 ρ={np.corrcoef(sr,sb)[0,1]:+.2f}", flush=True)
    print(f"{'子集(16维)':18s}{'解距离':>9s}{'解方位角':>10s}", flush=True)
    for lbl,cols in [("距离 top16",ar[:16]),("距离 bottom16",ar[-16:]),("角度 top16",ab[:16]),("角度 bottom16",ab[-16:]),("随机 16",rnd[:16])]:
        print(f"{lbl:18s}{ridge(Zn[:,cols],rng_m):9.3f}{ridge(Zn[:,cols],bear):10.3f}", flush=True)
    print(f"{'全部 256':18s}{ridge(Zn,rng_m):9.3f}{ridge(Zn,bear):10.3f}\n", flush=True)
