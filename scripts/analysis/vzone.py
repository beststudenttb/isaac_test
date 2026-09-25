"""判 V 的两个候选:(甲)价值函数在目标附近饱和 -> z 在决胜区分辨率低;(乙)纯优化不稳。
量各臂的**局部分辨率**:把状态按"离停车配置有多远"分层,在每层里取相邻状态对,
看 ‖Δz‖ 除以真实状态差(方位角 deg / 距离 m)。停车容差是 3°/0.2m,所以区内那一层最要紧。
用 data_g4_diam_angle(训练分布,含 teacher 的 V),纯 CPU。
"""
import sys, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
torch.set_num_threads(10)
ROOT=Path("models/vision/g4_diam_angle"); DATA=Path("./data_g4_diam_angle")
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),
      ("Ax","Ax/donor.pt"),("Aw","Aw/donor.pt"),("sup","sup/donor.pt")]
N=4000; rg=np.random.default_rng(0)
tr=fr.load_transitions(DATA,units=1)
ok=(tr["dist"]>0.3)&(tr["done"]<0.5)
idx=rg.choice(np.flatnonzero(ok),N,replace=False)
px,d=tr["px_x"][idx].astype(np.float64),tr["dist"][idx].astype(np.float64)
import math
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng_=d/np.cos(np.radians(bear))
V=tr["value"][idx].astype(np.float64)
frames=fr.load_frames(tr["img"][idx],DATA)
# 离停车配置的距离(以容差为单位):方位 3°,距离 0.2m,停在 1.5m
tol=np.sqrt((bear/3.0)**2+((rng_-1.5)/0.2)**2)
BANDS=[(0,1,"区内(≤1 容差)"),(1,3,"1–3 容差"),(3,8,"3–8 容差"),(8,1e9,">8 容差")]
ii,jj=np.triu_indices(N,1); s=rg.choice(len(ii),2_000_000,replace=False); ii,jj=ii[s],jj[s]
db=np.abs(bear[ii]-bear[jj]); dr=np.abs(rng_[ii]-rng_[jj])
ds=np.sqrt((db/3.0)**2+(dr/0.2)**2)          # 状态差,以容差为单位
near=(ds>0.05)&(ds<0.6)                       # 只取"邻近对",量局部斜率
print(f"{'臂':6s}"+"".join(f"{n:>16s}" for _,_,n in BANDS)+f"{'区内/区外比':>12s}",flush=True)
@torch.no_grad()
def enc(p,bs=64):
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(p,map_location="cpu")); e.eval()
    return torch.cat([e(frames[i:i+bs])["shared_feature"].float() for i in range(0,len(frames),bs)]).numpy().astype(np.float64)
for arm,rel in ARMS:
    p=ROOT/rel
    if not p.exists(): continue
    Z=enc(p); Zs=(Z-Z.mean(0))/(Z.std(0)+1e-9)
    dz=np.linalg.norm(Zs[ii]-Zs[jj],axis=1)
    cells=[]; vals=[]
    for lo,hi,_ in BANDS:
        m=near&(tol[ii]>=lo)&(tol[ii]<hi)&(tol[jj]>=lo)&(tol[jj]<hi)
        v=float(np.median(dz[m]/ds[m])) if m.sum()>200 else float("nan")
        vals.append(v); cells.append(f"{v:16.2f}")
    ratio=vals[0]/vals[2] if vals[2]==vals[2] and vals[2]>0 else float("nan")
    print(f"{arm:6s}"+"".join(cells)+f"{ratio:12.2f}",flush=True)
print("\n读法:数值 = 每 1 个容差单位的状态变化对应多少 ‖Δz‖(已按维度标准化)。"
      "\n若 V 的『区内』列显著低于 R,说明它在决胜区分辨率被压扁(候选甲);"
      "\n若 V 与 R 在各层都相当,则 V 的不稳是纯优化问题(候选乙)。",flush=True)
