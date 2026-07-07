"""SPR three-stage teacher-BC config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/spr_bc")
CLEAR_OUT_DIR = True

NUM_ENVS = 64
TOTAL_STEPS = 8_192_000
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

Z_DIM = 128
FPN_DIM = 128
SPR_HIDDEN = 256
SPR_POOL = 7
SPR_PRETRAINED = True
FREEZE_STEM_LAYERS = True
SPR_TAU = 0.99
SPR_COEF = 1.0

TEACHER_PATH = "./models/rl/teacher_ppo/best_val.zip"

N_STEPS = 64
BATCH_SIZE = 64
N_EPOCHS = 4
LR = 3e-4
MAX_GRAD_NORM = 0.5

POLICY_NET = [64, 64]
VALUE_NET = [64, 64]
ACTIVATION = "tanh"
STD_INIT = 0.5

STAGE1_UPDATES = 500  # 阶段1:teacher 加噪带跑,只训 SPR/encoder。
STAGE2_UPDATES = 500  # 阶段2:冻结 SPR,student 自跑,teacher BC 训 actor;其余 update 为阶段3联合微调。
TEACHER_NOISE_STD = 0.3  # 阶段1 teacher 动作上的高斯噪声 std。
RAND_ACTION_P = 0.1  # 阶段1 每步整 env 替换为均匀随机动作的概率。
SPR_UPDATE_EVERY = 5  # 阶段3 每多少个 update 做一次 SPR/encoder 更新;0 表示阶段3不再更新 SPR。
ACTOR_ENCODER_COEF = 0.0  # BC 梯度进 encoder 的比例;0 表示 actor 路径对 z detach。

SAVE_UPDATE_EVERY = 1
LOG_EVERY = 1
