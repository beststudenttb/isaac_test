"""第二层:在冻结表征上跑真正的优化器,不是线性探针。

用户 2026-09-04:"线性头只能说明线性头能够把表征给评估,但是并不能说明这个表征对这个策略友好"。
仓库自己的反例(2026_07_16):真值 oracle 的折扣回报 R²=0.78 完全可解码,但 TD(0) 连 oracle 也
学不出来(-0.08)。所以可解码性不是充分条件。

这一层 = "训策略" 减去 "跑环境":真正的 [64,64] MLP、真正的自举 TD 目标、真正的梯度下降,
只是数据是固定的 50k。测不到闭环(策略改变数据分布),那部分只能靠 Isaac。

    python scripts/offline_policy_probe.py

  critic   n-step TD (n=1/5/17/MC) 的学习曲线:收敛速度、最终精度、发不发散
  样本效率 10% / 30% / 100% 数据
  actor    同一个 MLP 降到同样 loss 要多少步(优化条件数,不是可解码性)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "./src")

from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import ARMS, make_head

DEVICE = "cuda"
GAMMA = 0.99
FEATURE_BATCH = 512
N_STEPS = (1, 5, 17)
FRACTIONS = (0.1, 0.3, 1.0)
CRITIC_STEPS = 4000
ACTOR_STEPS = 4000
BATCH = 256
LR = 3e-4
EVAL_EVERY = 200


def episode_order(tr):
    """按 (episode, t) 排序,并给出每行在其 episode 内的位置与该 episode 的长度。"""
    order = np.lexsort((tr["t"], tr["episode"]))
    ep = tr["episode"][order]
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    lengths = np.diff(np.r_[starts, len(ep)])
    pos = np.arange(len(ep)) - np.repeat(starts, lengths)
    ep_len = np.repeat(lengths, lengths)
    return order, pos, ep_len


def nstep_targets(tr, order, pos, ep_len, n: int):
    """返回 (n 步折扣奖励和, 自举系数 gamma^k, 自举目标的行号)。"""
    reward = tr["reward"][order]
    m = len(order)
    acc = np.zeros(m, dtype=np.float64)
    coef = np.zeros(m, dtype=np.float64)
    boot = np.arange(m)
    remain = ep_len - pos  # 到 episode 结束还有几步
    for k in range(n):
        live = k < remain
        acc[live] += (GAMMA ** k) * reward[np.arange(m)[live] + k]
    k_eff = np.minimum(n, remain)
    hit_end = k_eff < n  # episode 在 n 步内结束 -> 不自举
    coef = np.where(hit_end, 0.0, GAMMA ** n)
    boot = np.arange(m) + k_eff
    boot = np.minimum(boot, m - 1)
    return acc.astype(np.float32), coef.astype(np.float32), boot


def mc_returns(tr, order, pos, ep_len) -> np.ndarray:
    reward = tr["reward"][order]
    out = np.zeros(len(order), dtype=np.float32)
    running = 0.0
    for i in range(len(order) - 1, -1, -1):
        if pos[i] == ep_len[i] - 1:
            running = 0.0
        running = reward[i] + GAMMA * running
        out[i] = running
    return out


@torch.no_grad()
def encode_all(path: Path, names: np.ndarray) -> torch.Tensor:
    enc = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEVICE)
    enc.load_state_dict(torch.load(path, map_location=DEVICE))
    enc.eval()
    out = torch.empty((len(names), FREE_SPATIAL_CONFIG["feature_dim"]), dtype=torch.float32, device=DEVICE)
    for i in range(0, len(names), FEATURE_BATCH):
        chunk = fr.load_frames(names[i : i + FEATURE_BATCH])
        out[i : i + FEATURE_BATCH] = enc(chunk.to(DEVICE))["shared_feature"].float()
    return out


def r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    root = fr.OUT_ROOT
    tr = fr.load_transitions()
    order, pos, ep_len = episode_order(tr)
    mc = mc_returns(tr, order, pos, ep_len)
    train_mask = np.load(root / "train_mask.npy")[order]
    action = np.clip(tr["action"], -1.0, 1.0)[order]

    targets = {n: nstep_targets(tr, order, pos, ep_len, n) for n in N_STEPS}
    img = tr["img"][order]

    mc_t = torch.as_tensor(mc, device=DEVICE)
    act_t = torch.as_tensor(action, device=DEVICE)
    tr_idx = torch.as_tensor(np.flatnonzero(train_mask), device=DEVICE)
    te_idx = torch.as_tensor(np.flatnonzero(~train_mask), device=DEVICE)
    print(f"[INFO] train {len(tr_idx)} held {len(te_idx)}")

    rows = []
    checkpoints = [("init", root / "encoder_init.pt")] + [(a, root / a / "encoder_final.pt") for a in ARMS]
    for name, path in checkpoints:
        if not path.exists():
            continue
        print(f"[INFO] encoding for arm={name}", flush=True)
        z = encode_all(path, img)
        mu, sd = z[tr_idx].mean(0), z[tr_idx].std(0) + 1e-6
        z = (z - mu) / sd

        for frac in FRACTIONS:
            g = torch.Generator(device=DEVICE).manual_seed(0)
            sub = tr_idx[torch.randperm(len(tr_idx), generator=g, device=DEVICE)[: int(len(tr_idx) * frac)]]

            for n in list(N_STEPS) + ["MC"]:
                torch.manual_seed(0)
                critic = make_head(FREE_SPATIAL_CONFIG["feature_dim"], 1).to(DEVICE)
                opt = torch.optim.Adam(critic.parameters(), lr=LR)
                if n == "MC":
                    acc = coef = boot = None
                else:
                    a_np, c_np, b_np = targets[n]
                    acc = torch.as_tensor(a_np, device=DEVICE)
                    coef = torch.as_tensor(c_np, device=DEVICE)
                    boot = torch.as_tensor(b_np, device=DEVICE)

                curve, diverged = [], False
                for step in range(CRITIC_STEPS + 1):
                    if step % EVAL_EVERY == 0:
                        with torch.no_grad():
                            curve.append(r2(critic(z[te_idx]).squeeze(-1), mc_t[te_idx]))
                        if not np.isfinite(curve[-1]) or abs(curve[-1]) > 1e3:
                            diverged = True
                            break
                    if step == CRITIC_STEPS:
                        break
                    i = sub[torch.randint(0, len(sub), (BATCH,), device=DEVICE)]
                    if n == "MC":
                        target = mc_t[i]
                    else:
                        with torch.no_grad():
                            target = acc[i] + coef[i] * critic(z[boot[i]]).squeeze(-1)
                    loss = ((critic(z[i]).squeeze(-1) - target) ** 2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                best = max(curve)
                reach = next((k * EVAL_EVERY for k, v in enumerate(curve) if v >= 0.9 * best), -1) if best > 0 else -1
                rows.append(
                    {
                        "arm": name,
                        "frac": frac,
                        "n": n,
                        "critic_r2": round(best, 4),
                        "steps_to_90": reach,
                        "diverged": int(diverged),
                    }
                )
                print("  " + " ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)

        # actor 优化条件数:同一个 MLP 降到同样 loss 要多少步。
        torch.manual_seed(0)
        actor = make_head(FREE_SPATIAL_CONFIG["feature_dim"], 3).to(DEVICE)
        opt = torch.optim.Adam(actor.parameters(), lr=LR)
        hist = []
        for step in range(ACTOR_STEPS + 1):
            if step % EVAL_EVERY == 0:
                with torch.no_grad():
                    hist.append(float(((actor(z[te_idx]) - act_t[te_idx]) ** 2).mean()))
            if step == ACTOR_STEPS:
                break
            i = tr_idx[torch.randint(0, len(tr_idx), (BATCH,), device=DEVICE)]
            loss = ((actor(z[i]) - act_t[i]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        floor = min(hist)
        thresh = 0.02
        reach = next((k * EVAL_EVERY for k, v in enumerate(hist) if v <= thresh), -1)
        rows.append({"arm": name, "frac": 1.0, "n": "actor", "critic_r2": round(floor, 5), "steps_to_90": reach, "diverged": 0})
        print(f"  arm={name} actor_final_mse={floor:.5f} steps_to_mse{thresh}={reach}", flush=True)
        del z
        torch.cuda.empty_cache()

    with (root / "offline_policy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {root / 'offline_policy.csv'}")


if __name__ == "__main__":
    main()
