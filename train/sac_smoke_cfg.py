"""SAC+HER 冒烟用 cfg:继承 sac_cfg,把 SEED/SAVE/TOTAL/容量调小,几十 tick 走完
seed→update→save→log→last 全链路后自然退出(不必杀进程)。仅冒烟用,产物落 *_smoke 目录。
"""

from pathlib import Path

from sac_cfg import *  # noqa: F401,F403  (同目录,运行时 train/ 在 sys.path)

OUT_DIR = Path("./models/rl/sac_smoke_spr")
PIXELS_OUT_DIR = Path("./models/rl/sac_smoke_pixels")

NUM_ENVS = 16
TOTAL_ENV_STEPS = 4000
SEED_STEPS = 128
SAVE_EVERY_STEPS = 1000
CAPACITY_PER_ENV = 512
UPDATES_PER_TICK = 4
LOG_EVERY_TICKS = 10
