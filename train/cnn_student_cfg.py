"""Small CNN visual student PPO config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/cnn_fixed_student")  # 固定随机 CNN student 输出目录。
RANDOM_STOP_OUT_DIR = Path("./models/rl/cnn_fixed_student_random_stop")  # --random-stop 输出目录。
CLEAR_OUT_DIR = True  # 从头训练时清空输出目录。

NUM_ENVS = 64  # 并行环境数量。
N_STEPS = 64  # 每轮 rollout step 数。
UPDATES = 2000  # 总 update 数。
TOTAL_STEPS = NUM_ENVS * N_STEPS * UPDATES  # 总采样步数。
EPISODE_S = 15.0  # episode 时长，单位 s。
STOP_N = 1  # 训练时 stop 一次即 success。
SEED = 1  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。

RENDERING_MODE = "balanced"  # 相机渲染模式。
ANTIALIASING_MODE = "Off"  # 抗锯齿/上采样模式。
DLSS_MODE = None  # DLSS 模式；非 DLSS 时为 None。
RERENDER_ON_RESET = 1  # reset 后重渲染次数。

END_D_MIN = 1.5  # 固定任务 stop 距离最小值。
END_D_MAX = 1.5  # 固定任务 stop 距离最大值。
END_X_MIN = 0.0  # 固定任务 stop 图像位置最小值。
END_X_MAX = 0.0  # 固定任务 stop 图像位置最大值。
RANDOM_END_D_MIN = 1.3  # 随机任务 stop 距离最小值。
RANDOM_END_D_MAX = 1.8  # 随机任务 stop 距离最大值。
RANDOM_END_X_MIN = -20.0  # 随机任务 stop 图像位置最小值。
RANDOM_END_X_MAX = 20.0  # 随机任务 stop 图像位置最大值。

Z_DIM = 128  # 小 CNN 输出特征维度。
FREEZE_CNN = True  # 固定随机 CNN 特征，只训练 actor/critic。
STAGE1_UPDATES = 500  # 满 teacher BC 的 update 数。
STAGE2_UPDATES = 300  # teacher 线性退火的 update 数。
STAGE1_STEPS = NUM_ENVS * N_STEPS * STAGE1_UPDATES
STAGE2_STEPS = NUM_ENVS * N_STEPS * STAGE2_UPDATES

TEACHER_PATH = "./models/rl/teacher_ppo/best_val.zip"  # 固定任务 teacher。
RANDOM_TEACHER_PATH = "./models/rl/teacher_ppo_random_stop/best_val.zip"  # 随机任务 teacher。
TEACHER_LOSS = 5.0  # 初始 teacher MSE 权重。

BATCH_SIZE = 32  # PPO minibatch 大小。
N_EPOCHS = 4  # 每批 rollout 训练次数。
LR = 3e-4  # 学习率。
GAMMA = 0.99  # 折扣因子。
GAE_LAMBDA = 0.95  # GAE lambda。
CLIP_RANGE = 0.2  # PPO clip。
CLIP_RANGE_VF = None  # value clip；None 表示不用。
NORMALIZE_ADVANTAGE = True  # 是否标准化 advantage。
ENT_COEF = 0.0  # 熵系数。
VF_COEF = 0.5  # value loss 系数。
MAX_GRAD_NORM = 0.5  # 梯度裁剪。

POLICY_NET = [64, 64]  # actor hidden sizes。
VALUE_NET = [64, 64]  # critic hidden sizes。
ACTIVATION = "tanh"  # tanh/relu/elu。
STD_INIT = 0.3  # 初始动作标准差。
STD_MAX = 0.3  # 动作标准差上限，训练中只允许降到更小。

SAVE_UPDATE_EVERY = 10  # 每多少 update 保存一次。
VAL_EVERY = 0  # 训练中 val；0 表示关闭。
VAL_STEPS = 0  # val 最大 step；0 表示 env max_episode_length。
SAVE_BEST = True  # 训练中 val 时是否保存 best。
BEST_WARMUP = 20  # best warmup update。
BEST_MARGIN = 0.02  # best 最小提升。
LOG_EVERY = 1  # 每多少 update 打印一次。
