"""JSRL SPR-PPO offline val config。checkpoint 全是 full_*.pt(自包含),不需要 SPR 基文件。"""

from pathlib import Path


OUT_DIR = Path("./models/rl/spr_jsrl")  # 与 train/spr_jsrl_cfg.py 对齐。
RANDOM_STOP_OUT_DIR = Path("./models/rl/spr_jsrl_random_stop")

NUM_ENVS = 32  # 离线 val 默认并行环境数量;64 会在 val 启动时内存崩溃。
NUM_EPISODES = 1
START = 0
STRIDE = 1
VAL_STEPS = 0  # 0 表示使用 env max_episode_length。

EPISODE_S = 15.0
STOP_N = 3  # 连续多少 step stop 后 success done(val 用 3,train 用 1)。
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

POLICY_NET = [64, 64]
VALUE_NET = [64, 64]
ACTIVATION = "tanh"
STD_INIT = 0.5
