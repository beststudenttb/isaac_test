"""信息是不是冗余地摊在所有维上?随机子集大小扫描 + 无关扰动折算成"等价于多大的任务变化"。"""
import sys, csv, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
import free_repr as fr
torch.set_num_threads(12)
COMB="diam_angle"; ROOT=Path(f"models/vision/g4_{COMB}"); DATA=Path(f"./data_g4_{COMB}")
D=Path("data_probe_grid"); rows=list(csv.DictReader(open(D/"grid.csv")))
byc={}
for r in rows: byc.setdefault(r["cond"],{})[int(r["env"])]=r
envs=sorted(byc["base"]); CONDS=["base","r+0.30","b+5","blue_move","light_half","light_x2","light_warm"]
gf={c: torch.stack([read_image(str(D/byc[c][e]["img"])) for e in envs]) for c in CONDS}
arr=lambda c,k: np.array([float(byc[c][e][k]) for e in envs])
diam=lambda c: 62.12/np.maximum(arr(c,"dist"),1e-6)
dpx_r=np.abs(diam("r+0.30")-diam("base")); dpx_b=np.abs(arr("b+5","px_x")-arr("base","px_x"))
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
KS=[1,2,4,8,16,32,64,128,256]
print(f"随机子集解码(每个 k 取 5 组随机维求均值),{COMB}\n{'表征':6s}{'量':7s}"+"".join(f"{'k='+str(k):>8s}" for k in KS)+f"{'有效秩':>8s}",flush=True)
Zs={}
for name,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Zn=enc(p,nf,64); Zs[name]=(p,Zn)
    ev=np.linalg.svd((Zn-Zn.mean(0)),compute_uv=False)**2; ev=ev/ev.sum(); er=float(np.exp(-(ev*np.log(ev+1e-12)).sum()))
    for lbl,y in [("距离",rng_m),("方位角",bear)]:
        cells=[]
        for k in KS:
            v=[ridge(Zn[:,rg.choice(256,k,replace=False)],y) for _ in range(5 if k<256 else 1)]
            cells.append(np.mean(v))
        print(f"{name:6s}{lbl:7s}"+"".join(f"{c:8.3f}" for c in cells)+(f"{er:8.2f}" if lbl=="距离" else ""),flush=True)
print(f"\n无关扰动折算成等价的任务变化(用同一表征上 Δz 对像素的斜率换算):",flush=True)
print(f"{'表征':6s}{'扰动':12s}{'等价横移 px':>12s}{'等价角度 °':>11s}{'等价直径 px':>12s}{'等价距离 m@1.5':>15s}",flush=True)
for name,(p,Zn) in Zs.items():
    Zg={c: enc(p,gf[c]) for c in CONDS}; s=Zg["base"].std(0)+1e-9
    nrm=lambda c: np.linalg.norm((Zg[c]-Zg["base"])/s,axis=1)
    kr=(nrm("r+0.30")/np.maximum(dpx_r,1e-6)).mean()   # z单位 / 直径px
    kb=(nrm("b+5")/np.maximum(dpx_b,1e-6)).mean()      # z单位 / 横移px
    for c in ["blue_move","light_half","light_x2","light_warm"]:
        v=nrm(c).mean()
        eq_px=v/kb; eq_deg=np.degrees(np.arctan(eq_px/fx)); eq_diam=v/kr; eq_m=eq_diam/(62.12/1.5**2)
        print(f"{name:6s}{c:12s}{eq_px:12.1f}{eq_deg:11.2f}{eq_diam:12.2f}{eq_m:15.3f}",flush=True)
