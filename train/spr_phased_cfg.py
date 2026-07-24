"""分阶段 SPR-PPO 配置(forward + JSRL 串联)。

并行跑两张卡时用环境变量选组合与 encoder 模式,输出目录自动带 tag 不互相覆盖:
    PHASED_COMBO=A PHASED_ENCODER=frozen GPU=0 ./IsaacLab/isaaclab.sh -p train/spr_phased.py --noise
    PHASED_COMBO=B PHASED_ENCODER=frozen GPU=1 ./IsaacLab/isaaclab.sh -p train/spr_phased.py --noise

Combo A = [forward, jsrl]:先无teacher自己开(收 σ_w),再 JSRL 学精确停。主赌。
Combo B = [jsrl, forward]:先 JSRL,再 forward。较弱的对照(σ_w 仍只能靠后段 forward 收)。
ENCODER:frozen(全程冻结,和 arm A 一致,推荐先跑)。
        cotrain(仅 Combo A 的 forward 段训 encoder,切 JSRL 冻结)——注意:裸解冻 encoder 已实证
        2 个 update 内塌缩(见 07-06),且会混淆 σ_w 实验;先把 frozen 跑清楚再碰 cotrain。
"""

import os
from pathlib import Path


COMBO = os.environ.get("PHASED_COMBO", "A").upper()          # A | B
ENCODER_MODE = os.environ.get("PHASED_ENCODER", "frozen")    # frozen | cotrain
_TAG = f"{COMBO.lower()}_{ENCODER_MODE}"

OUT_DIR = Path(f"./models/rl/spr_phased_{_TAG}")
RANDOM_STOP_OUT_DIR = Path(f"./models/rl/spr_phased_{_TAG}_random_stop")
CLEAR_OUT_DIR = True

# 冻结/初始 encoder 来源(和 arm A / JSRL 逐字节相同的 z)。
SPR_CKPT = Path("./models/rl/_encoders/stage1_sigma_ablation.pt")

TEACHER_PATH = "./models/rl/teacher_ppo/best_val.zip"
RANDOM_TEACHER_PATH = "./models/rl/teacher_ppo_random_stop/best_val.zip"

NUM_ENVS = 128
N_STEPS = 64
UPDATES = 1400          # forward(≤600)+ jsrl(其余);forward 提前收敛则 jsrl 拿到更多。
EPISODE_S = 15.0
STOP_N = 1
SEED = 1
DEVICE = None

RENDERING_MODE = "balanced"
ANTIALIASING_MODE = "Off"
DLSS_MODE = None
RERENDER_ON_RESET = 1

END_D_MIN = 1.5
END_D_MAX = 1.5
END_X_MIN = 0.0
END_X_MAX = 0.0
RANDOM_END_D_MIN = 1.3
RANDOM_END_D_MAX = 1.8
RANDOM_END_X_MIN = -20.0
RANDOM_END_X_MAX = 20.0

# ---- 阶段表 ----
# forward: mode=forward(h_eff=0,student 从第0步自己开、自己转头对中 -> a_w 拿到接近信号)。
#          switch_sigma_w:σ_w 降到该值即切下一阶段(触底切,不是等它涨回来);min_updates 前不切;
#          max_updates 到了强制切(σ_w 没降下来则日志告警,假设可能证伪)。
# jsrl:    mode=jsrl,当前反向课程(回火已在 src/spr_jsrl.py 禁用,只进不退)。
#          ratchet_w:a_w 的 σ 只降不升(棘轮),防切 JSRL 后区内 reward 沙漠把 σ_w 吹回。
#          floor_x:a_x 的 σ 抬回该 floor —— forward 段 a_x 已塌到 ~0.02,JSRL 段还要靠它探索精刹。
_FORWARD = {
    "name": "forward", "mode": "forward", "std_max": 0.3, "std_min": 0.02,
    "min_updates": 50, "max_updates": 600, "switch_sigma_w": 0.10,
}
_JSRL = {
    "name": "jsrl", "mode": "jsrl", "std_max": 0.3, "std_min": 0.02,
    "ratchet_w": True, "floor_x": 0.10, "max_updates": 900,
}
# Combo B 的 forward 在后,负责收 σ_w,故 ratchet_w/floor_x 挂在它上;jsrl 在前只学停。
_JSRL_FIRST = {
    "name": "jsrl", "mode": "jsrl", "std_max": 0.3, "std_min": 0.02, "max_updates": 500,
}
_FORWARD_LAST = {
    "name": "forward", "mode": "forward", "std_max": 0.3, "std_min": 0.02,
    "ratchet_w": True, "floor_x": 0.10, "min_updates": 50, "max_updates": 900, "switch_sigma_w": 0.10,
}
PHASES = [_FORWARD, _JSRL] if COMBO == "A" else [_JSRL_FIRST, _FORWARD_LAST]

# ---- 反向课程 / JSRL(沿用 spr_jsrl_cfg)----
JSRL_H_BASE = 150
JSRL_N_RUNGS = 10
JSRL_UP_THRESH = 0.80
JSRL_DOWN_THRESH = 0.30   # 回火已在 src/spr_jsrl.py 禁用,此值不再使用。
JSRL_EMA = 0.90
JSRL_COOLDOWN = 10
TEACHER_NOISE_STD = 0.0
MAX_COLLECT_ITERS = 4000

# ---- co-train encoder(仅 cotrain 模式的 forward 段用)----
SPR_K = 8         # SPR k 步转移目标(与 spr_student 一致)。
SPR_COEF = 1.0    # SPR loss 权重。

# ---- PPO ----
BATCH_SIZE = 256
N_EPOCHS = 4
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
NORMALIZE_ADVANTAGE = True
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

POLICY_NET = [64, 64]
VALUE_NET = [64, 64]
ACTIVATION = "tanh"
STD_INIT = 0.3    # forward 段要从 0.3 起探索(±45° 转头);per-phase std_max 再管上限。
STD_MIN = 0.02

SAVE_UPDATE_EVERY = 10
LOG_EVERY = 1
