"""4 物体受控渲染的表征侧分析(CPU)。回答三件事:

A. **各槽可读性**:在 base 条件上拟合 z -> 各槽方位角,留出集 R²。
   直接填周末计划 §3 那张预登记预测表(只做颜色过滤 / 只做形状过滤 / 真的区分目标 / 不过滤)。
B. **颜色敏感度**:base ↔ swap 的 ‖Δz‖/‖z‖。同一轮里几何逐像素一致,颜色是唯一变量。
C. **机制判决**:把 base 上拟合好的"槽0(红球)方位"探针**冻住**,喂给 swap
   (槽0 变蓝、槽2/3 变红)。若读出跟着**槽位**走 -> z 报告的是"那个位置的物体";
   若跟着**红色**走 -> z 报告的是"目标(红的那个)在哪",这就是 A 换目标必塌的直接原因。
D. **形状线索**:allred 条件(四个都红)下各槽还能不能读出。颜色失去区分力,
   剩下的只有形状与位置。

    CUDA_VISIBLE_DEVICES="" python scripts/analysis/w4_probe.py
"""
import sys, csv, numpy as np, torch
sys.path.insert(0, "./src"); sys.path.insert(0, ".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from torchvision.io import read_image
torch.set_num_threads(8)

ROOT = Path("models/vision/w4"); D = Path("data_w4_probe")
SLOT = {0: "红球", 1: "红方", 2: "蓝球", 3: "蓝方"}
rows = list(csv.DictReader(open(D / "meta.csv")))
cond = np.array([r["cond"] for r in rows]); rd = np.array([int(r["round"]) for r in rows])
env = np.array([int(r["env"]) for r in rows])
bear = np.stack([np.array([float(r[f"bear{k}"]) for r in rows]) for k in range(4)])   # (4, N)
vis = np.stack([np.array([int(r[f"vis{k}"]) for r in rows]) for k in range(4)])
CONDS = list(dict.fromkeys(cond.tolist()))
base = cond == "base"; fit = base & (rd % 2 == 0); held = base & (rd % 2 == 1)
print(f"共 {len(rows)} 帧,条件 {CONDS};base 拟合 {fit.sum()} / 留出 {held.sum()}\n", flush=True)

@torch.no_grad()
def enc(p, bs=48):
    e = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG)
    e.load_state_dict(torch.load(p, map_location="cpu")); e.eval()
    out = []
    for i in range(0, len(rows), bs):
        imgs = torch.stack([read_image(str(D / r["img"])) for r in rows[i:i + bs]])
        out.append(e(imgs)["shared_feature"].float())
    return torch.cat(out).numpy().astype(np.float64)

def probe(Z, y, m, lam=3e-2):
    mu, sd = Z[m].mean(0), Z[m].std(0) + 1e-9
    A = np.hstack([(Z[m] - mu) / sd, np.ones((m.sum(), 1))])
    w = np.linalg.lstsq(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ y[m], rcond=None)[0]
    return lambda Zx: np.hstack([(Zx - mu) / sd, np.ones((len(Zx), 1))]) @ w

def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

ARMS = [("init", "encoder_init.pt"), ("A", "A/donor.pt"), ("V", "V/donor.pt"),
        ("R", "R/donor.pt"), ("sup", "sup/donor.pt")]
for arm, rel in ARMS:
    p = ROOT / rel
    if not p.exists(): print(f"[跳过] {p} 不存在"); continue
    Z = enc(p)
    print(f"\n================ {arm} ================", flush=True)

    # A. 各槽可读性(base 留出)
    fns = {}
    line = ""
    for k in range(4):
        m = fit & (vis[k] == 1); h = held & (vis[k] == 1)
        if m.sum() < 40 or h.sum() < 20: line += f"{SLOT[k]}: n/a  "; continue
        f = probe(Z, bear[k], m); fns[k] = f
        line += f"{SLOT[k]}:{r2(f(Z[h]), bear[k][h]):6.3f}  "
    print(f"A 各槽方位可读性(base 留出 R²)  {line}", flush=True)

    # D. allred:颜色失去区分力,只剩形状与位置
    line = ""
    for k in range(4):
        m = (cond == "allred") & (vis[k] == 1)
        if m.sum() < 60: line += f"{SLOT[k]}: n/a  "; continue
        n = m.sum(); idx = np.where(m)[0]; tr = idx[: n // 2]; te = idx[n // 2:]
        mm = np.zeros(len(rows), bool); mm[tr] = True
        f = probe(Z, bear[k], mm)
        line += f"{SLOT[k]}:{r2(f(Z[te]), bear[k][te]):6.3f}  "
    print(f"D allred 内自拟合(四个都红,只剩形状/位置)  {line}", flush=True)

    # B. 颜色敏感度:同 (round, env) 下 base vs 其他条件的 ‖Δz‖/‖z‖
    key = {(r_, e_): i for i, (c_, r_, e_) in enumerate(zip(cond, rd, env)) if c_ == "base"}
    zn = np.linalg.norm(Z[base], axis=1).mean()
    line = ""
    for c in CONDS:
        if c == "base": continue
        idx = [(i, key[(rd[i], env[i])]) for i in range(len(rows))
               if cond[i] == c and (rd[i], env[i]) in key]
        d = np.linalg.norm(Z[[a for a, _ in idx]] - Z[[b for _, b in idx]], axis=1).mean()
        line += f"{c}:{d/zn:6.3f}  "
    print(f"B 颜色敏感度 ‖Δz‖/‖z‖(几何逐像素一致)  {line}", flush=True)

    # C. 机制判决:base 上学的"槽0"探针,冻住后喂 swap(槽0 变蓝、槽2/3 变红)
    if 0 in fns:
        m = (cond == "swap") & (vis[0] == 1)
        pred = fns[0](Z[m])
        cand = {f"槽0({SLOT[0]},swap 后是蓝)": bear[0][m]}
        for k in (1, 2, 3):
            cand[f"槽{k}({SLOT[k]},swap 后 2/3 是红)"] = bear[k][m]
        best = max(cand, key=lambda n: r2(pred, cand[n]))
        print("C swap 下冻住的槽0探针跟着谁走:  " +
              "  ".join(f"{n}:{r2(pred, v):6.3f}" for n, v in cand.items()), flush=True)
        print(f"   -> 最像 **{best}**", flush=True)
