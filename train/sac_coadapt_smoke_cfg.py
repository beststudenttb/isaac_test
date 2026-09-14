"""SAC+HER+co-adapt 冒烟用 cfg:继承 sac_coadapt_cfg,把 SEED/SAVE/TOTAL/容量调小,几十 tick
走完 seed→update(含 spr_loss)→save→log→last 全链路后自然退出。仅冒烟用,产物落 *_smoke 目录。
"""

from pathlib import Path

from sac_coadapt_cfg import *  # noqa: F401,F403  (同目录,运行时 train/ 在 sys.path)

OUT_DIR = Path("./models/rl/sac_coadapt_smoke")

NUM_ENVS = 8
TOTAL_ENV_STEPS = 2000
SEED_STEPS = 128
SAVE_EVERY_STEPS = 1000
CAPACITY_PER_ENV = 256
UPDATES_PER_TICK = 2
LOG_EVERY_TICKS = 10
