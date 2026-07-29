"""SAC + HER off-policy config(train/sac.py 与 val/sac.py 共用 --cfg sac_cfg)。

动机:破 spr_coadapt/JSRL 一路都撞的稀疏精确停止冷启动死锁 —— SAC 随机策略在 zone 内能采到
a_mag<STOP_EPS 的低动作,HER 把 goal relabel 到实际到达位置,把这些低动作组合成 in_zone&a<eps=
success 样本喂给 critic。判据:success_rate 能不能起来(尤其 stop_frac 从 0 起来)。

先跑固定任务(END_* 固定),两张卡分别 spr_z(冻结 z)与 pixels(可训 CNN)。
"""

from pathlib import Path


OUT_DIR = Path("./models/rl/sac_spr")          # spr_z 模式输出。
PIXELS_OUT_DIR = Path("./models/rl/sac_pixels")  # pixels 模式输出。
RANDOM_STOP_SUFFIX = "_random_stop"
CLEAR_OUT_DIR = True

OBS_MODE = "spr_z"  # "spr_z" 或 "pixels";两张卡分别用 --obs-mode 覆盖。
SPR_CKPT = Path("./models/rl/_encoders/stage1_sigma_ablation.pt")  # 与 drq_student arm A 同一个冻结 z。

NUM_ENVS = 64
TOTAL_ENV_STEPS = 2_000_000
SEED_STEPS = 4000
UPDATES_PER_TICK = 16  # replay ratio = 16*256/64 = 64。
EPISODE_S = 15.0
STOP_N = 1  # HER 单步 relabel 依赖 STOP_N==1(success ⟺ 单步 in_zone & a<eps),必须为 1。
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

CAPACITY_PER_ENV = 4096  # pixels: 64*4096*224*224*3 uint8 ≈ 37GB;spr_z 不存图。
BATCH_SIZE = 256
GAMMA = 0.99
LR = 1e-4
TAU = 0.01
FEATURE_DIM = 50
HIDDEN_DIM = 1024

# SAC 熵温度(自适应)。target_entropy = -act_dim 是连续控制惯例。
ALPHA_INIT = 0.1
ALPHA_LR = 1e-4
TARGET_ENTROPY = -3.0

# HER future relabel。
HER_RATIO = 0.5      # 一半样本做 goal relabel。
HER_FUTURE_H = 20    # future 目标在同 episode 后 H 步内采。

PIXELS_RES = 84   # DrQ 标准训练分辨率(存储恒 224,采样 GPU resize)。
AUG_PAD = 4

SAVE_EVERY_STEPS = 25_000
LOG_EVERY_TICKS = 25

# val 参数(val/sac.py 用)。
NUM_EPISODES = 1
START = 0
STRIDE = 1
VAL_STEPS = 0
