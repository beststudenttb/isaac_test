"""SPR co-adapt 离线评估 config(val/offpolicy.py --cfg spr_coadapt_cfg)。

OUT_DIR 与 train/spr_coadapt_cfg.py 一致。spr_coadapt 的 encoder 内置在 agent,不需外部
spr_encoder.pt(SPR_CKPT 仅占位,val/offpolicy 只在 obs_mode==spr_z 时才 load_spr)。
"""

from pathlib import Path


OUT_DIR = Path("./models/rl/spr_coadapt")
RANDOM_STOP_SUFFIX = "_random_stop"
SPR_CKPT = "spr_encoder.pt"  # 占位;spr_coadapt 不用。

NUM_ENVS = 128
NUM_EPISODES = 1
START = 0
STRIDE = 1
VAL_STEPS = 0

EPISODE_S = 15.0
STOP_N = 3
SEED = 0
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
