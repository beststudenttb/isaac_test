"""在**冻结的表征**上监督训一个动作头,存成 <arm>_act/head.pt。
用来把"这个 z 上存不存在好策略(可表达性)"和"PPO 找不找得到(可优化性)"分开。
所有臂用同一套流程(同帧数、同步数),包括 A 臂自己,保证可比。
用法: python fit_action_head.py <vision_root> <arm> [n_frames]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "./src"); sys.path.insert(0, ".")
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import make_head
torch.set_num_threads(10)
ROOT = Path(sys.argv[1]); ARM = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
DATA = Path(str(ROOT).replace("models/vision/g4_", "./data_g4_"))
src = ROOT / "encoder_init.pt" if ARM == "init" else ROOT / ARM / "donor.pt"
out = ROOT / f"{ARM}_act"; out.mkdir(parents=True, exist_ok=True)
if (out / "head.pt").exists():
    print(f"[SKIP] {out} 已有"); sys.exit(0)
tr = fr.load_transitions(DATA, units=1)
rg = np.random.default_rng(0); idx = rg.choice(len(tr["img"]), min(N, len(tr["img"])), replace=False)
m = fr.episode_split(tr["episode"], seed=0)[idx]
frames = fr.load_frames(tr["img"][idx], DATA)
y = np.clip(tr["action"][idx], -1.0, 1.0).astype(np.float32)
stats = json.load(open(ROOT / "target_stats.json"))["A"]
mu, sd = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
yz = torch.tensor((y - mu) / sd)
enc = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG); enc.load_state_dict(torch.load(src, map_location="cpu")); enc.eval()
with torch.no_grad():
    Z = torch.cat([enc(frames[i:i+64])["shared_feature"].float() for i in range(0, len(frames), 64)])
head = make_head(int(FREE_SPATIAL_CONFIG["feature_dim"]), yz.shape[1])
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
ti = torch.tensor(np.flatnonzero(m)); hi = torch.tensor(np.flatnonzero(~m))
for step in range(6000):
    s = ti[torch.randint(0, len(ti), (256,))]
    loss = ((head(Z[s]) - yz[s]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    htr = float(((head(Z[ti]) - yz[ti]) ** 2).mean()); hhe = float(((head(Z[hi]) - yz[hi]) ** 2).mean())
torch.save(head.state_dict(), out / "head.pt")
torch.save(enc.state_dict(), out / "donor.pt")
print(f"[FIT] {ROOT.name}/{ARM}  帧 {len(idx)}  train_mse {htr:.4f}  held_mse {hhe:.4f} -> {out}")
