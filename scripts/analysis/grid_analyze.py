"""受控网格的 Δz 分析(CPU):任务扰动 vs 无关扰动,以及按方向的敏感维。"""
import sys, csv, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(12)
D=Path("data_probe_grid"); rows=list(csv.DictReader(open(D/"grid.csv")))
byc={}
for r in rows: byc.setdefault(r["cond"],{})[int(r["env"])]=r
order=["base","r+0.05","r+0.30","b+1","b+5","blue_move","light_half","light_x2","light_warm"]
envs=sorted(byc["base"]); n=len(envs)
frames={c: torch.stack([read_image(str(D/byc[c][e]["img"])) for e in envs]) for c in order}
rng0=np.array([float(byc["base"][e]["rng"]) for e in envs]); bea0=np.array([float(byc["base"][e]["bearing"]) for e in envs])
@torch.no_grad()
def enc(p, f):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(f[i:i+32])["shared_feature"].float() for i in range(0,len(f),32)]).numpy().astype(np.float64)
BANDS=[("近 1.5-2.0m",(rng0<2.2)),("中 2.5-4m",(rng0>=2.2)&(rng0<4.5)),("远 5-6m",(rng0>=4.5))]
print(f"{'表征':6s}{'条件':11s}"+"".join(f"{b[0]:>12s}" for b in BANDS)+f"{'全体':>10s}", flush=True)
for name,rel in [("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]:
    p=Path("models/vision/g4_diam_angle")/rel
    if not p.exists(): continue
    Z={c: enc(p, frames[c]) for c in order}
    s=Z["base"].std(0).mean()      # 用基准网格上的整体尺度归一
    for c in order[1:]:
        dz=np.linalg.norm((Z[c]-Z["base"])/ (Z["base"].std(0)+1e-9), axis=1)
        cells="".join(f"{dz[m].mean():12.2f}" for _,m in BANDS)
        print(f"{name:6s}{c:11s}{cells}{dz.mean():10.2f}", flush=True)
    print(flush=True)
