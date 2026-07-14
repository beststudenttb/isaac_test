"""Configuration for standalone DrQ-v2 offline validation."""

from pathlib import Path


OUT_DIR = Path("./models/rl/drqv2")  # 必须与 train/drqv2_cfg.py 完全一致(07-11 `_q` 目录对不上把 pipeline 搞崩过)。
RANDOM_STOP_SUFFIX = "_random_stop"
NOISE_SUFFIX = "_noise"

NUM_ENVS = 32
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

IMAGE_SIZE = 224
FRAME_STACK = 1
