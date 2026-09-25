"""w4 四物体批次汇总:末 6 个 ckpt 取最好(排除 last)。槽 0=红球 1=红方 2=蓝球 3=蓝方。"""
import csv, glob, os, statistics, sys
SLOT = {0: "红球", 1: "红方", 2: "蓝球", 3: "蓝方"}
ARMS = ["A_donor", "R_donor", "init", "V_donor", "V_donor_s2", "sup_donor"]
def cell(arm, slot):
    a = arm.replace("_s2", "_s2") if arm.endswith("_s2") else arm
    d = f"models/rl/score_k03noy_student_{a}_w4_s{slot}"
    best = None; n = 0
    for f in sorted(glob.glob(f"{d}/eval_score*.csv")):
        ck = os.path.basename(f).replace("eval_score", "").replace(".csv", "").lstrip("_") or "last"
        if ck == "last": continue
        R = list(csv.DictReader(open(f)))
        if not R: continue
        n += 1
        s = 100.0 * sum(int(r["v1_success"]) for r in R) / len(R)
        de = statistics.mean(float(r["tail_de"]) for r in R)
        if best is None or s > best[0]: best = (s, de, ck)
    return best, n
print(f"{'表征':<12}" + "".join(f"{SLOT[k]:>11}" for k in range(4)) + f"{'红均':>9}{'蓝均':>9}{'蓝/红':>8}")
for arm in ARMS:
    vals = []; ns = []
    for k in range(4):
        b, n = cell(arm, k); vals.append(b); ns.append(n)
    if not any(vals): continue
    txt = ""
    for b, n in zip(vals, ns):
        txt += f"{b[0]:>8.1f}({n})" if b else f"{'--':>11}"
    red = [b[0] for b in vals[:2] if b]; blue = [b[0] for b in vals[2:] if b]
    rm = statistics.mean(red) if red else None; bm = statistics.mean(blue) if blue else None
    print(f"{arm:<12}{txt}" +
          (f"{rm:>9.1f}" if rm is not None else f"{'--':>9}") +
          (f"{bm:>9.1f}" if bm is not None else f"{'--':>9}") +
          (f"{bm/rm:>8.2f}" if (rm and bm is not None) else f"{'--':>8}"))
print("\n括号=可用 ckpt 数(满 5;last 一律排除)")
# 各槽的**内在难度本来就不同**(init 红方 76.6 > 红球 45.3),原始分不可横向比,
# 必须对 init 归一化:比值 = 该臂该槽 / init 该槽。1.0 = 与随机初始化的编码器同水平。
base = {k: (cell("init", k)[0][0] if cell("init", k)[0] else None) for k in range(4)}
if any(v is not None for v in base.values()):
    print("\n对 init 归一化(比值 = 该臂 / init 同槽;1.0 = 与随机编码器同水平):")
    print(f"{'表征':<12}" + "".join(f"{SLOT[k]:>11}" for k in range(4)))
    for arm in ARMS:
        if arm == "init": continue
        row = ""
        for k in range(4):
            b, _ = cell(arm, k)
            row += f"{b[0]/base[k]:>11.2f}" if (b and base[k]) else f"{'--':>11}"
        if row.strip("- "): print(f"{arm:<12}{row}")
    print(f"{'init(基线)':<10}" + "".join(f"{base[k]:>11.1f}" if base[k] is not None else f"{'--':>11}" for k in range(4)))
print("末段 de(m):")
for arm in ARMS:
    row = ""
    for k in range(4):
        b, _ = cell(arm, k)
        row += f"{b[1]:>11.3f}" if b else f"{'--':>11}"
    if row.strip("- "): print(f"{arm:<12}{row}")
