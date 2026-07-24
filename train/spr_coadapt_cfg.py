"""SPR + DrQ-v2 端到端共适应 config。

原文 SPR 范式搬到 off-policy:critic 的 TD 梯度 + SPR 自监督 loss 同时更新同一个
encoder(从头联合,不预训练冻结)。双时间尺度靠 ENC_LR_RATIO(encoder 学习率 = critic 的
1/10)压住表征漂移。判据是稳定性(z_std 塌不塌、spr_loss/q 发不发散),不是成功率。
"""

from pathlib import Path


OUT_DIR = Path("./models/rl/spr_coadapt")
RANDOM_STOP_SUFFIX = "_random_stop"
CLEAR_OUT_DIR = True

OBS_MODE = "spr_coadapt"
SPR_CKPT = None  # 从头联合,不加载预训练 encoder(save_checkpoint 会记 None)。

# --- SPR co-adapt ---
SPR_Z_DIM = 128     # z 维度(= repr_dim,进 actor/critic trunk)。
SPR_FPN = 128       # FPN 通道。
SPR_HIDDEN = 256    # transition/projector/predictor 隐层。
SPR_POOL = 7        # FPN 空间池化。
SPR_TAU = 0.99      # target encoder EMA 系数。
SPR_K = 8           # K 步 latent rollout(与其余 SPR 线一致)。
SPR_COEF = 1.0      # spr_loss 权重 λ。
ENC_LR_RATIO = 0.1  # encoder 学习率 = critic 的 1/10(双时间尺度:表征慢、策略快)。

NUM_ENVS = 128           # CLI 覆盖。
TOTAL_ENV_STEPS = 2_000_000  # 机制探针:先看早期塌不塌,不追满训。
SEED_STEPS = 4000
UPDATES_PER_TICK = 16    # co-adapt 每 update 跑 ResNet18 fwd+bwd,贵;先 16(replay ratio=16*256/128=32)。
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

CAPACITY_PER_ENV = 4096  # 存图:128 env x 4096 x 224x224x3 uint8 ≈ 77GB(总容量 = 64x8192,不爆 125GB RAM)。
N_STEP = 3
BATCH_SIZE = 64  # co-adapt 的 encoder 可训练:256 实测 OOM(SPR-FPN 在 224 下 p1/p2/p3 都是 56x56,
# cat 成 [B,386,56,56],batch256 单张量就 1.24GB,整个 update 峰值 >11.5GB)。64 实测峰值 3.09GB,
# 给 128 env 的 Isaac 留余量。与 spr_bc(64)/spr_student(32,"可训练视觉 encoder 显存开销大")一致;
# arm A 的 256 是 encoder 冻结才能用的值。
LR = 1e-4
GAMMA = 0.99
TAU = 0.01
FEATURE_DIM = 50
HIDDEN_DIM = 1024

STD_START = 0.3
STD_END = 0.05
STD_DECAY_STEPS = 1_500_000
NOISE_CLIP = 0.3

PIXELS_RES = 112  # spr_coadapt 未用(走 SPR encoder),保留给 agent_cfg 兼容。
AUG_PAD = 6       # 未用。

SAVE_EVERY_STEPS = 25_000
LOG_EVERY_TICKS = 25
