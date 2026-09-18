"""Head definition shared by stage1 (fits it) and stage2 (freezes it)."""

from __future__ import annotations

import torch.nn as nn

ARMS = ("A", "V", "R", "sup")
from free_repr import ACT_DIM

ARM_DIMS = {"A": ACT_DIM, "V": 1, "R": 1, "rnd": ACT_DIM, "sup": 2,
            # 2026-09-18 机制拆分臂:只改 student 回归什么,不碰环境/teacher/数据,因而与 A、R 完全可比。
            "Asym": ACT_DIM,  # (a_x, |a_w|)  2 维、对称、应当好提取
            "Aw": 1,          # a_w           1 维、反对称、随机先验上只有 0.473
            "Ax": 1}          # a_x           1 维、对称、随机先验上 0.965
# A   teacher 的确定性动作 mu(不是采样动作 —— 采样动作 a_y 维 26% 方差是噪声)
# V   teacher 的 V(s)
# R   每步的位置分(任务 v3 的奖励)
# rnd 随机初始化并冻结的头。**必须保留**:2026-09-04 实测它与 A 在五个策略无关度量上
#     打平(赢 2 平 2 输 1),是唯一能判定"teacher 信号到底有没有用"的对照。
# sup 显式 (norm_x, norm_d) 监督 —— 人工设计的表征,整篇要被超越的靶子。
HEAD_HIDDEN = (64, 64)  # 对齐 Isaac teacher 的 actor net_arch。


def make_head(feature_dim: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = feature_dim
    for width in HEAD_HIDDEN:
        layers += [nn.Linear(last, width), nn.Tanh()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)
