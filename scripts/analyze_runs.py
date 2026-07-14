#!/usr/bin/env python3
"""跨线对比训练日志:SPR/MDP student(PPO)与 DrQ-v2(off-policy)的 schema 不同,这里统一读。

用法:
    python3 scripts/analyze_runs.py models/rl/drqv2_noise models/rl/spr_student
    python3 scripts/analyze_runs.py --sigma approved_models/2026_07_14_drqv2_pixels
    python3 scripts/analyze_runs.py --offline models/rl/drqv2

两个必须小心的坑(今天都踩过):
1. success_rate 是**每个窗口**的比率,分母是该窗口内结束的 episode 数(DrQ-v2 平均只有 2.19 个),
   所以只会取 0/⅓/½/⅔/1 这些离散值。**跨窗口聚合必须按 episode 数加权**,直接对这一列取平均是错的。
2. 窗口内没有 episode 结束时,代码写的是 `success/max(episodes,1)` = `0.0000`,
   和"真的 0% 成功"在日志里长得一模一样。加权聚合会自动把这些窗口的权重变成 0,直接平均则不会。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


# 两种 log schema 的列名映射:(episode 数, success 率, σ, 进度轴, 进度轴名)
SCHEMAS = {
    "drqv2": dict(eps="episodes", succ="success_rate", fail="fail_rate", tmo="timeout_rate",
                  sigma="stddev", x="frame", xname="frame",
                  extra=["critic_loss", "actor_loss", "q1"]),
    "ppo": dict(eps="done", succ=None, fail=None, tmo=None,  # PPO 线记的是 success 计数,不是率
                sigma="std_mean", x="update", xname="update",
                extra=["zone_frac", "stop_frac", "a_mag_mean", "kl", "v_loss"]),
}


def detect(fields: list[str]) -> str:
    if "stddev" in fields and "frame" in fields:
        return "drqv2"
    if "std_mean" in fields and "update" in fields:
        return "ppo"
    raise ValueError(f"无法识别的 log schema,列名: {fields[:8]}...")


def load(run: Path) -> tuple[str, list[dict]]:
    log = run / "log.csv"
    if not log.is_file():
        raise FileNotFoundError(f"没有 log.csv: {log}")
    with log.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"log.csv 是空的: {log}")
    return detect(list(rows[0])), rows


def outcomes(kind: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回每行的 (episode 数, success 数, fail 数, timeout 数) —— 一律换算成**计数**再聚合。"""
    s = SCHEMAS[kind]
    eps = np.array([float(r[s["eps"]]) for r in rows])
    if kind == "ppo":  # PPO 线直接记的就是计数
        succ = np.array([float(r["success"]) for r in rows])
        fail = np.array([float(r["fail"]) for r in rows])
        tmo = np.array([float(r["timeout"]) for r in rows])
    else:  # DrQ-v2 记的是率 -> 乘回 episode 数还原成计数。空窗口的率是空串(新)或假的 0(旧),乘 0 后都无害
        rate = lambda col: np.array([float(r[col]) if r[col] else 0.0 for r in rows])
        succ = rate(s["succ"]) * eps
        fail = rate(s["fail"]) * eps
        tmo = rate(s["tmo"]) * eps
    return eps, succ, fail, tmo


def chunk_table(run: Path, nchunks: int = 10) -> None:
    kind, rows = load(run)
    s = SCHEMAS[kind]
    eps, succ, fail, tmo = outcomes(kind, rows)
    x = np.array([float(r[s["x"]]) for r in rows])
    sigma = np.array([float(r[s["sigma"]]) for r in rows])
    extra = [c for c in s["extra"] if c in rows[0]]

    print(f"\n=== {run}  [{kind}]  {len(rows)} 行,{int(eps.sum())} 个 episode ===")
    hdr = f"{s['xname']:>9} {'eps':>5} {'succ':>6} {'fail':>6} {'tmo':>6} {'sigma':>6}"
    hdr += "".join(f" {c[:9]:>9}" for c in extra)
    print(hdr)
    bounds = np.linspace(0, len(rows), nchunks + 1).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        e = eps[a:b].sum()
        d = max(e, 1.0)
        line = (f"{x[b-1]:>9.0f} {int(e):>5} {succ[a:b].sum()/d:>6.3f} "
                f"{fail[a:b].sum()/d:>6.3f} {tmo[a:b].sum()/d:>6.3f} {sigma[a:b].mean():>6.3f}")
        for c in extra:
            v = np.array([float(r[c]) for r in rows[a:b]]).mean()
            line += f" {v:>9.3f}"
        print(line)
    tot = eps.sum()
    print(f"{'累计':>9} {int(tot):>5} {succ.sum()/max(tot,1):>6.3f} "
          f"{fail.sum()/max(tot,1):>6.3f} {tmo.sum()/max(tot,1):>6.3f}")
    empty = int((eps == 0).sum())
    if empty:
        print(f"  注:{empty}/{len(rows)} 个窗口没有 episode 结束(日志里写成 0.0000,加权后权重为 0)")


def sigma_table(run: Path, bins: int = 14) -> None:
    """本项目的核心图:success 是 σ 的函数。按 σ 分箱,而不是按时间。"""
    kind, rows = load(run)
    s = SCHEMAS[kind]
    eps, succ, _, _ = outcomes(kind, rows)
    sigma = np.array([float(r[s["sigma"]]) for r in rows])

    # BC 在场时的 success 是 teacher 刷出来的,不是 RL 学的 —— 不标出来这张表会撒谎。
    tcoef = (np.array([float(r["teacher_coef"]) for r in rows])
             if "teacher_coef" in rows[0] else None)

    lo, hi = sigma.min(), sigma.max()
    print(f"\n=== {run}  success vs σ  (σ 从 {hi:.3f} 降到 {lo:.3f}) ===")
    hdr = f"{'σ 区间':>16} {'eps':>6} {'success':>8} {'P(stop)/步':>11}"
    if tcoef is not None:
        hdr += f" {'teacher':>8}"
    print(hdr)
    edges = np.linspace(lo, hi, bins + 1)
    from math import erf, sqrt
    for a, b in zip(edges[:-1], edges[1:]):
        m = (sigma >= a) & (sigma < b) if b < hi else (sigma >= a)
        e = eps[m].sum()
        if e == 0:
            continue
        mid = 0.5 * (a + b)
        # 成功要求三维动作同时 |a_i| < STOP_EPS=0.05
        p = (2 * (0.5 * (1 + erf(0.05 / mid / sqrt(2)))) - 1) ** 3
        line = f"  [{a:.3f}, {b:.3f}) {int(e):>6} {succ[m].sum()/e:>8.3f} {p:>11.4f}"
        if tcoef is not None:
            # 按 episode 数加权的平均 teacher 权重;>0 说明这一箱的 success 有 BC 的功劳
            w = eps[m]
            line += f" {(tcoef[m] * w).sum() / max(w.sum(), 1):>8.2f}"
        print(line)
    if tcoef is not None:
        print("  teacher 列 = 该箱内按 episode 加权的 BC 权重;>0 时 success 不能算作 RL 的成果")


def offline(run: Path) -> None:
    f = run / "offline_val.csv"
    if not f.is_file():
        print(f"  (没有 offline_val.csv: {f})")
        return
    with f.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    xcol = "frame" if "frame" in rows[0] else "update"
    print(f"\n=== {run}  离线 val(确定性 μ),{len(rows)} 个 checkpoint ===")
    print(f"{xcol:>9} {'succ':>6} {'fail':>6} {'tmo':>6} {'final_de':>9} {'final_xe':>9}")
    step = max(len(rows) // 14, 1)
    for r in rows[::step]:
        print(f"{float(r[xcol]):>9.0f} {float(r['success_rate']):>6.3f} {float(r['fail_rate']):>6.3f} "
              f"{float(r['timeout_rate']):>6.3f} {float(r['final_de_mean']):>9.3f} {float(r['final_xe_mean']):>9.2f}")
    best = max(rows, key=lambda r: float(r["success_rate"]))
    print(f"  best: {best['checkpoint']} succ={float(best['success_rate']):.4f} "
          f"de={float(best['final_de_mean']):.3f}m xe={float(best['final_xe_mean']):.2f}px")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="训练输出目录(含 log.csv)")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--sigma", action="store_true", help="按 σ 分箱,而不是按时间")
    ap.add_argument("--offline", action="store_true", help="同时打印 offline_val.csv")
    args = ap.parse_args()

    for run in args.runs:
        try:
            if args.sigma:
                sigma_table(run)
            else:
                chunk_table(run, args.chunks)
            if args.offline:
                offline(run)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n=== {run} ===\n  跳过: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
