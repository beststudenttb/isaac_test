"""SAC + HER + SPR 共适应 config(train/sac.py 与 val/sac.py 共用 --cfg sac_coadapt_cfg)。

这条线 = spr_coadapt(DDPG 版,2M success≡0)换成 SAC+HER 再打一次。encoder 侧完全照抄
train/spr_coadapt_cfg.py(SPR_* 与 ENC_LR_RATIO / BATCH_SIZE 一字不改),算法侧完全照抄
train/sac_cfg.py(ALPHA_* / TARGET_ENTROPY / HER_*),所以两边的差都是单变量:

- vs spr_coadapt(DDPG):只换算法 → 答"死锁是不是被 SAC+HER 破了"(看 stop_frac / success_rate);
- vs sac_cfg 的 spr_z(冻结 z,88%):只换表征 → 答"表征同步训练要付多少代价"(看 z_std / q1)。

判据同 spr_coadapt:先看塌不塌(z_std→0、q1 发散即塌),再看 success_rate 能不能起来。
"""

from pathlib import Path


OUT_DIR = Path("./models/rl/sac_coadapt")
RANDOM_STOP_SUFFIX = "_random_stop"
CLEAR_OUT_DIR = True

OBS_MODE = "spr_coadapt"
SPR_CKPT = None  # 从头联合,不加载预训练 encoder。

# --- SPR co-adapt(与 train/spr_coadapt_cfg.py 完全一致)---
SPR_Z_DIM = 128     # z 维度(= repr_dim,进 actor/critic trunk)。
SPR_FPN = 128       # FPN 通道。
SPR_HIDDEN = 256    # transition/projector/predictor 隐层。
SPR_POOL = 7        # FPN 空间池化。
SPR_TAU = 0.99      # target encoder EMA 系数。
SPR_K = 8           # K 步 latent rollout。
SPR_COEF = 1.0      # spr_loss 权重 λ。
ENC_LR_RATIO = 0.1  # encoder 学习率 = critic 的 1/10(双时间尺度:表征慢、策略快)。

NUM_ENVS = 64            # 存图:64 x 4096 x 224x224x3 uint8 ≈ 39GB,留出并行跑第二条线的 RAM。
TOTAL_ENV_STEPS = 2_000_000
SEED_STEPS = 4000
UPDATES_PER_TICK = 16    # batch 64 → replay ratio = 16*64/64 = 16(arm A 的 64 是 batch 256 才够)。
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

CAPACITY_PER_ENV = 4096
BATCH_SIZE = 64  # co-adapt 的 encoder 可训练:256 实测 OOM(SPR-FPN 在 224 下 cat 成 [B,386,56,56])。
GAMMA = 0.99
LR = 1e-4
TAU = 0.01
FEATURE_DIM = 50
HIDDEN_DIM = 1024

# SAC 熵温度(自适应),与 sac_cfg 一致。pixels 线实测 alpha 会塌到 0.012、entropy 顶到 target
# 后策略重新饱和;这里保持同参数以便单变量对比,若 co-adapt 也这么塌,再调 TARGET_ENTROPY。
ALPHA_INIT = 0.1
ALPHA_LR = 1e-4
TARGET_ENTROPY = -3.0

# HER future relabel。
HER_RATIO = 0.5      # 一半样本做 goal relabel。
HER_FUTURE_H = 20    # future 目标在同 episode 后 H 步内采。

PIXELS_RES = 112  # spr_coadapt 未用(走 SPR encoder),保留给 agent_cfg 兼容。
AUG_PAD = 6       # 未用。

SAVE_EVERY_STEPS = 25_000
LOG_EVERY_TICKS = 25

# val 参数(val/sac.py 用)。
NUM_EPISODES = 1
START = 0
STRIDE = 1
VAL_STEPS = 0
