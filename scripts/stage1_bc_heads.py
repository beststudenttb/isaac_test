"""任务二第一步:训出读出头,然后冻结。

交接文档 2026-09-03 §3。六个臂,一个变量:冻结头承载什么信号。

    python scripts/stage1_bc_heads.py

**头是移植来的。** 每个臂先用一个一次性的 donor encoder 和头联合训练(普通 BC),
训完把 donor 扔掉、只留头。stage2 再把这个头装到另一个随机初值的 encoder 上冻结。

为什么不直接把头拟合到 stage2 那个 encoder 自己的特征上:那样 encoder 会坐在一个
由该头定义的刀刃极小值上,任何一步都掉下去,stage2 的轨迹就成了"掉下去再爬回来"的
优化器伪影,不是冻结头的语义压力(实测 sup 臂 loss 0.0023 -> 一步 0.71,Adam/SGD 都一样)。
更要命的是各臂起点 loss 会差三个数量级(sup 0.002 vs rnd 1.0),§8 开放项 1 那个随机头
对照就没法解释了 —— 它跟主臂的差异会混在"信号内容"和"起点拟合质量"两件事里。
移植头让所有臂起点在同一量级,rnd 才是干净对照。

这也正是 Webots §2.1 的做法:那里的"冻结 actor 头"就是在别的 encoder 上训出来的。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "./src")

from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
import free_repr as fr
from stage1_helpers import ARM_DIMS, ARMS, make_head

DONOR_SEED = 1  # donor encoder 的种子,与 stage2 共享初值(seed 0)不同。
DONOR_STEPS = 3000
DONOR_BATCH = 64
DONOR_LR = 1e-4
FEATURE_BATCH = 256
CALIBRATE_BATCHES = 200


@torch.no_grad()
def calibrate_bn(encoder: nn.Module, frames: torch.Tensor, device: str) -> None:
    """把 BN 的 running stats 跑到真实激活上。

    pretrained=False 时 running stats 初值是 (0,1),与真实激活差很远,会让 eval 模式
    (抽特征、探针、任务三)和 train 模式(训练)看到两组不同的特征。
    """
    encoder.train()
    for _ in range(CALIBRATE_BATCHES):
        idx = np.random.randint(0, len(frames), FEATURE_BATCH)
        encoder(frames[idx].to(device))


SUP_TARGET = "xd"


def sup_target(tr: dict[str, np.ndarray]) -> np.ndarray:
    """sup 臂 = teacher 自己看到的那两维(与 sb3_env.policy_obs 一致),由 --sup-target 选。"""
    px, d = tr["px_x"].astype(np.float64), tr["dist"].astype(np.float64)
    seen = d > 0.0
    kind = SUP_TARGET
    if kind == "cam":      # (x_c, 直径) —— obs_mask="diam"
        diam = np.clip(62.12 / np.clip(d, 0.3, None) / 224.0, 0.0, 1.0)
        return np.stack([fr.norm_x(tr["px_x"], tr["dist"]), np.where(seen, diam, 0.0)], axis=1).astype(np.float32)
    if kind == "ang":      # (方位角/40, 距离/8) —— obs_mask="ang"
        fx = 112.0 / np.tan(np.radians(40.0))
        bear = np.degrees(np.arctan((112.0 - px) / fx))
        rng_m = d / np.cos(np.radians(bear))
        return np.stack([np.where(seen, np.clip(bear / 40.0, -1, 1), -1.0),
                         np.where(seen, np.clip(rng_m / 8.0, 0, 1), 0.0)], axis=1).astype(np.float32)
    return np.stack([fr.norm_x(tr["px_x"], tr["dist"]), fr.norm_d(tr["dist"])], axis=1)


def build_targets(tr: dict[str, np.ndarray], train_mask: np.ndarray, a_target: str = "mu") -> tuple[dict[str, np.ndarray], dict]:
    """每个目标维度按训练集统计做 z-score。

    不做的话 A 臂 85.6% 的 loss 来自 a_x(退化成 ax 臂)、AV 臂 99.95% 来自 value
    (退化成 V 臂),§3 的五臂对照就没有对照可言了。
    """
    mu = np.clip(tr["mu"] if a_target == "mu" else tr["action"], -1.0, 1.0)   # teacher 的确定性策略(或实际执行的采样动作),行为是 clamp 后的值
    raw = {
        "A": mu,
        "V": tr["value"][:, None],
        "R": tr["reward"][:, None],
        "rnd": mu,
        "sup": sup_target(tr),
        # 机制拆分:把"维数 / 对称性 / 可提取性"三者解耦。ACT_IDX=(a_x, a_w),所以 mu[:,1] 是转向。
        "Asym": np.stack([mu[:, 0], np.abs(mu[:, 1])], axis=1),
        "Aw": mu[:, 1:2],
        "Ax": mu[:, 0:1],
    }
    stats, out = {}, {}
    for arm, y in raw.items():
        mu, sd = y[train_mask].mean(0), y[train_mask].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out[arm] = ((y - mu) / sd).astype(np.float32)
        stats[arm] = {"mean": mu.tolist(), "std": sd.tolist()}
    return out, stats


@torch.no_grad()
def eval_mse(encoder, head, frames, target, idx, device) -> float:
    encoder.eval()
    total, n = 0.0, 0
    for i in range(0, len(idx), FEATURE_BATCH):
        chunk = idx[i : i + FEATURE_BATCH]
        pred = head(encoder(frames[chunk].to(device))["shared_feature"])
        total += float(((pred - target[chunk]) ** 2).mean().item()) * len(chunk)
        n += len(chunk)
    encoder.train()
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and freeze the stage-2 readout heads.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--donor-steps", type=int, default=DONOR_STEPS)
    parser.add_argument("--donor-seed", type=int, default=DONOR_SEED)
    parser.add_argument("--head-seed", type=int, default=None)
    parser.add_argument("--a-target", choices=("mu", "sampled"), default="mu")
    parser.add_argument("--sup-target", choices=("xd", "cam", "ang"), default="xd", help="sup 臂回归哪套坐标:xd=(x,d) 旧版,cam=(x_c,直径),ang=(方位角,距离)")
    parser.add_argument("--data", type=Path, default=fr.DATA_DIR)  # 数据目录(默认跟管线标签)
    parser.add_argument("--units", type=int, default=2)  # A 臂目标:teacher 网络的动作均值,或它实际执行的采样动作(只看行为)  # 只换头的初值;None = 跟 --seed。共享编码器初值和数据切分仍由 --seed 定  # 换 donor 种子,检验表征是否依赖 donor 的随机坐标
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", type=Path, default=fr.OUT_ROOT)
    parser.add_argument("--arms", nargs="+", default=None)
    args = parser.parse_args()

    global SUP_TARGET
    SUP_TARGET = str(args.sup_target)

    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("[INFO] loading transitions")
    tr = fr.load_transitions(args.data, units=args.units)
    train_mask = fr.episode_split(tr["episode"], seed=args.seed)
    train_idx = np.flatnonzero(train_mask)
    held_idx = np.flatnonzero(~train_mask)
    print(f"[INFO] episodes {len(np.unique(tr['episode']))} train {len(train_idx)} held {len(held_idx)}")

    targets, stats = build_targets(tr, train_mask, args.a_target)
    print("[INFO] loading frames into RAM")
    frames = fr.load_frames(tr["img"], args.data)

    # stage2 六个臂共用的 encoder 随机初值,保证唯一变量是冻结头承载的信号。
    torch.manual_seed(args.seed)
    student = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(args.device)
    calibrate_bn(student, frames[train_idx], args.device)
    if args.arms is None:
        torch.save(student.state_dict(), args.out_root / "encoder_init.pt")
    else:
        student.load_state_dict(torch.load(args.out_root / "encoder_init.pt"))  # 补臂时沿用同一初值
    student.eval()
    print(f"[INFO] shared encoder init (BN calibrated) -> {args.out_root / 'encoder_init.pt'}")

    rows = []
    for arm in (args.arms or ARMS):
        arm_dir = args.out_root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        np.save(arm_dir / "target.npy", targets[arm])
        target = torch.as_tensor(targets[arm]).to(args.device)

        torch.manual_seed(args.seed if args.head_seed is None else args.head_seed)  # 每个臂的头从同一个初值出发。
        head = make_head(FREE_SPATIAL_CONFIG["feature_dim"], ARM_DIMS[arm]).to(args.device)

        if arm == "rnd":
            # §8 开放项 1:随机初始化并冻结的头,不训练。
            donor_held = float("nan")
        else:
            torch.manual_seed(args.donor_seed)
            donor = FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(args.device)
            calibrate_bn(donor, frames[train_idx], args.device)
            opt = torch.optim.Adam(list(donor.parameters()) + list(head.parameters()), lr=DONOR_LR)
            donor.train()
            for step in range(args.donor_steps):
                idx = train_idx[np.random.randint(0, len(train_idx), DONOR_BATCH)]
                loss = ((head(donor(frames[idx].to(args.device))["shared_feature"]) - target[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            donor_held = eval_mse(donor, head, frames, target, held_idx, args.device)
            torch.save(donor.state_dict(), arm_dir / "donor.pt")  # 2026-09-09 起留下 donor,用于和 stage2 表征比"长得一样还是只是表现一致"
            del donor, opt

        torch.save(head.state_dict(), arm_dir / "head.pt")
        # 移植缺口:这个头装到 stage2 的共享 encoder 上,起点有多差。
        gap_train = eval_mse(student, head, frames, target, train_idx, args.device)
        gap_held = eval_mse(student, head, frames, target, held_idx, args.device)
        rows.append(
            {
                "arm": arm,
                "dim": ARM_DIMS[arm],
                "donor_held_mse": f"{donor_held:.4f}",
                "transplant_train_mse": f"{gap_train:.4f}",
                "transplant_held_mse": f"{gap_held:.4f}",
            }
        )
        print(f"[INFO] {arm:4s} dim={ARM_DIMS[arm]} donor_held={donor_held:.4f} transplant_held={gap_held:.4f}", flush=True)

    with (args.out_root / ("stage1.csv" if args.arms is None else "stage1_extra.csv")).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_root / "target_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if args.arms is None:
        np.save(args.out_root / "train_mask.npy", train_mask)
    print(f"[INFO] stage1 done -> {args.out_root}")


if __name__ == "__main__":
    main()
