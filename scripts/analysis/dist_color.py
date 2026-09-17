"""同一几何下换干扰物,z 移动多远。
每一轮每个 env 的 15 个条件几何逐像素相同,只有干扰槽的属性不同,所以
    d(X) = mean_over(round,env) ‖ z(干扰=X) − z(无干扰) ‖ / ‖z(无干扰)‖
就是"这个干扰物把表征推了多远"。无拟合、无扰动、全部在分布内。CPU。
"""
import sys, csv, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(10)
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"]=font_manager.FontProperties(fname=f).get_name(); break
ROOT=Path("models/vision/g4_diam_angle"); D=Path("data_distract")
ARMS=[("init","encoder_init.pt"),("A","A/donor.pt"),("V","V/donor.pt"),("R","R/donor.pt"),("sup","sup/donor.pt")]
rows=list(csv.DictReader(open(D/"meta.csv")))
key=np.array([f"{r['round']}_{r['env']}" for r in rows]); cond=np.array([r["cond"] for r in rows])
ORDER=[("d_red_ball","红"),("d_orange","橙"),("d_magenta","品红"),("d_yellow","黄"),
       ("d_green","绿"),("d_grey","灰"),("d_white","白"),("base","蓝\n(训练见过)")]
@torch.no_grad()
def enc(e,idx,bs=64):
    out=[]
    for i in range(0,len(idx),bs):
        out.append(e(torch.stack([read_image(str(D/rows[j]["img"])) for j in idx[i:i+bs]]))["shared_feature"].float())
    return torch.cat(out).numpy().astype(np.float64)
base_idx={key[i]:i for i in np.flatnonzero(cond=="d_none")}
res={}
for arm,rel in ARMS:
    e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); e.load_state_dict(torch.load(ROOT/rel,map_location="cpu")); e.eval()
    ks=sorted(base_idx); Zb=enc(e,[base_idx[k] for k in ks]); pos={k:i for i,k in enumerate(ks)}
    row=[]
    for cd,_ in ORDER:
        idx=[i for i in np.flatnonzero(cond==cd) if key[i] in pos]
        Z=enc(e,idx); ref=Zb[[pos[key[i]] for i in idx]]
        row.append(float(np.mean(np.linalg.norm(Z-ref,axis=1)/(np.linalg.norm(ref,axis=1)+1e-9))))
    res[arm]=row; print(f"{arm:5s}"+"".join(f"{v:7.3f}" for v in row),flush=True)
x=np.arange(len(ORDER)); w=0.16
plt.figure(figsize=(9.5,4.6))
for i,(arm,_) in enumerate(ARMS): plt.bar(x+(i-2)*w,res[arm],w,label=arm)
plt.xticks(x,[l for _,l in ORDER],fontsize=8); plt.ylabel("‖z(有干扰) − z(无干扰)‖ / ‖z‖")
plt.xlabel("干扰球的颜色(目标红球位置不变,几何逐像素相同)")
plt.title("干扰物颜色扫描:干扰物是这个颜色时,表征被推多远(全为球形,只有颜色不同)",fontsize=13)
plt.legend(); plt.grid(axis="y",alpha=0.3); plt.tight_layout()
plt.savefig("figs/distractor_color_sweep.png",dpi=150); print("-> figs/distractor_color_sweep.png",flush=True)
