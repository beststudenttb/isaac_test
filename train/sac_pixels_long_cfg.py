"""SAC + HER pixels 线的长训版:2M 换 8M,其余一字不改(继承 sac_cfg)。

2M 那版 val best 0.484 @875k、末尾崩回 bang-bang。这条只问一件事:给 4 倍的步数,
可训 CNN 能不能自己走出来。输出另起目录,不覆盖 models/rl/sac_pixels 的既有结果。
"""

from pathlib import Path

from sac_cfg import *  # noqa: F401,F403  (同目录,运行时 train/ 在 sys.path)

OUT_DIR = Path("./models/rl/sac_pixels_8m")
PIXELS_OUT_DIR = Path("./models/rl/sac_pixels_8m")

TOTAL_ENV_STEPS = 8_000_000

# env 64→128。另外两个数跟着改,是为了让"配方"与 2M 那版完全一致,只有 env 数变:
# - CAPACITY_PER_ENV 4096→2048:128x2048 = 262144 帧,与 64x4096 同容量、同 39.5GB RAM
#   (4096 会变 79GB,和 GPU1 上 co-adapt 的 39.5GB 加起来爆 125GB);
# - UPDATES_PER_TICK 16→32:replay ratio 保持 32*256/128 = 64,总梯度步数仍是 2M 次。
NUM_ENVS = 128
CAPACITY_PER_ENV = 2048
UPDATES_PER_TICK = 32
