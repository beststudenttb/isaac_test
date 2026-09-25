"""按方向分开的敏感维分析(CPU)。
维度排序来自**受控网格**(只动一个量,无混淆);解码在**自然数据集**上做(样本多、可留出)。
问题:对距离敏感的维能解出什么?对距离**不**敏感的维还能不能解出距离?角度同理。
"""
import sys, csv, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
import free_repr as fr
torch.set_num_threads(12)
COMB = sys.argv[1] if len(sys.argv) > 1 else "diam_angle"
ROOT = Path(f"models/vision/g4_{COMB}"); DATA = Path(f"./data_g4_{COMB}")
# --- 网格 ---
D=Path("data_probe_grid"); rows=list(csv.DictReader(open(D/"grid.csv")))
byc={}
for r in rows: byc.setdefault(r["cond"],{})[int(r["env"])]=r
envs=sorted(byc["base"])
gf={c: torch.stack([read_image(str(D/byc[c][e]["img"])) for e in envs]) for c in ("base","r+0.30","b+5")}
# --- 自然数据 ---
tr=fr.load_transitions(DATA,units=1); seen=tr["dist"]>0.3
m_all=fr.episode_split(tr["episode"],seed=0); rg=np.random.default_rng(0); N=3000
idx=np.concatenate([rg.choice(np.flatnonzero(m_all&seen),N,replace=False), rg.choice(np.flatnonzero((~m_all)&seen),N,replace=False)])
m=np.r_[np.ones(N,bool),np.zeros(N,bool)]
nf=fr.load_frames(tr["img"][idx],DATA)
px=tr["px_x"][idx].astype(np.float64); d=tr["dist"][idx].astype(np.float64)
fx=112/np.tan(np.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng_m=d/np.cos(np.radians(bear))
@torch.no_grad()
def enc(p,f,bs=32):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(f[i:i+bs])["shared_feature"].float() for i in range(0,len(f),bs)]).numpy().astype(np.float64)
def ridge(F,y,lam=1e-2):
    if F.shape[1]==0: return float('nan')
    mu,sd=F[m].mean(0),F[m].std(0)+1e-9
    a=np.hstack([(F[m]-mu)/sd,np.ones((m.sum(),1))]); b=np.hstack([(F[~m]-mu)/sd,np.ones(((~m).sum(),1))])
    w=np.linalg.lstsq(a.T@a+lam*len(a)*np.eye(a.shape[1]),a.T@y[m],rcond=None)[0]
    p=b@w; yy=y[~m]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
print(f"== {COMB} ==  维度排序来自受控网格,解码在 {N}+{N} 自然帧上", flush=True)
print(f"{'表征':6s}{'维度子集':22s}{'解码 距离':>10s}{'解码 方位角':>12s}", flush=True)
for name,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=ROOT/rel
    if not p.exists(): continue
    Zg={c: enc(p,gf[c]) for c in gf}; s=Zg["base"].std(0)+1e-9
    sr=np.abs((Zg["r+0.30"]-Zg["base"])/s).mean(0)   # 每维对距离的敏感度
    sb=np.abs((Zg["b+5"]  -Zg["base"])/s).mean(0)    # 每维对角度的敏感度
    Zn=enc(p,nf,64)
    for lbl,cols in [("距离 top16", np.argsort(sr)[::-1][:16]),
                     ("距离 bottom128", np.argsort(sr)[:128]),
                     ("角度 top16", np.argsort(sb)[::-1][:16]),
                     ("角度 bottom128", np.argsort(sb)[:128]),
                     ("全部 256", np.arange(256))]:
        print(f"{name:6s}{lbl:22s}{ridge(Zn[:,cols],rng_m):10.3f}{ridge(Zn[:,cols],bear):12.3f}", flush=True)
    ov=len(set(np.argsort(sr)[::-1][:16]) & set(np.argsort(sb)[::-1][:16]))
    print(f"{'':6s}两个 top16 的重合 {ov}/16   敏感度相关 ρ={np.corrcoef(sr,sb)[0,1]:+.2f}\n", flush=True)
