"""Configuration for the standalone DrQ-v2 off-policy baseline."""

from pathlib import Path


OUT_DIR = Path("./models/rl/drqv2")
RANDOM_STOP_SUFFIX = "_random_stop"
NOISE_SUFFIX = "_noise"  # --noise 单独一个输出目录,避免 CLEAR_OUT_DIR 清掉干净 env 的结果。
CLEAR_OUT_DIR = True

NUM_ENVS = 64
TOTAL_FRAMES = 2_000_000
SEED_FRAMES = 4_000
UPDATES_PER_TICK = 16  # replay ratio = UPDATES_PER_TICK * BATCH_SIZE / NUM_ENVS;官方 DrQ-v2 是 256/2 = 128。
EPISODE_S = 15.0
STOP_N = 1  # 训练时 stop 一次即 success,与 SPR/MDP 线一致;val 用 3。stop_steps 是连续计数,3 会让 σ=0.3 下的成功概率从 2e-3/步 掉到 1e-8/步。
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

IMAGE_SIZE = 224
FRAME_STACK = 1
CAPACITY_PER_ENV = 4096  # 图像常驻 CPU 内存:4096 * 64 env * 224*224*3 uint8 ≈ 39.5GB(机器 125GB);8192 会到 79GB。
MIN_TIMEOUT_INTERVAL = 300
NSTEP = 3
DISCOUNT = 0.99
BATCH_SIZE = 128

LR = 1e-4
FEATURE_DIM = 50
HIDDEN_DIM = 1024
CRITIC_TARGET_TAU = 0.01
NUM_EXPL_STEPS = SEED_FRAMES
STDDEV_START = 0.3
STDDEV_END = 0.02
STDDEV_DURATION = 1_500_000
STDDEV_CLIP = 0.3
AUG_PAD = 4

# 每多少个 tick(= update)存一次 checkpoint,文件名 drqv2_{tick:06d}.pt(对齐 SPR 的 full_{update:06d})。
# tick 总数 = TOTAL_FRAMES / num_envs;32 env 下 2M frame = 62500 tick,除以 SAVE_UPDATE_EVERY 得 checkpoint 数。
SAVE_UPDATE_EVERY = 780  # 32 env: ~80 个 checkpoint。
LOG_EVERY_TICKS = 25
