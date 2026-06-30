"""Teacher PPO training config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/teacher_ppo")  # 训练现场输出目录，不上传 Git。
RANDOM_STOP_OUT_DIR = Path("./models/rl/teacher_ppo_random_stop")  # --random-stop 时的输出目录。
CLEAR_OUT_DIR = True  # 每次开始训练前清空 OUT_DIR。

NUM_ENVS = 512  # 并行环境数量。
TOTAL_STEPS = 8_192_000  # SB3 total_timesteps，所有 env 合计 transition 数。
EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 1  # 训练时只要在 stop 区输出一次 stop 就 success done。
SEED = 0  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 默认或命令行 --device。
USE_CAMERA = False  # 是否在 teacher 训练环境里启用相机。
READ_CAMERA = False  # 是否每 step 读取相机 RGB，用来测试视觉开销。

END_D_MIN = 1.5  # teacher state/reward 里的停止距离最小值，单位 m。
END_D_MAX = 1.5  # teacher state/reward 里的停止距离最大值，单位 m。
END_X_MIN = 0.0  # teacher state/reward 里的图像停止位置最小值，单位 px，0 表示图像中心。
END_X_MAX = 0.0  # teacher state/reward 里的图像停止位置最大值，单位 px。

RANDOM_END_D_MIN = 1.3  # --random-stop 时的停止距离最小值，单位 m。
RANDOM_END_D_MAX = 1.8  # --random-stop 时的停止距离最大值，单位 m。
RANDOM_END_X_MIN = -20.0  # --random-stop 时的图像停止位置最小值，单位 px。
RANDOM_END_X_MAX = 20.0  # --random-stop 时的图像停止位置最大值，单位 px。

N_STEPS = 64  # PPO 每个 env 每次 rollout 的 step 数。
BATCH_SIZE = 1024  # PPO minibatch 大小。
N_EPOCHS = 10  # 每批 rollout 重复优化次数。
LR = 3e-4  # 学习率。
GAMMA = 0.99  # 折扣因子。
GAE_LAMBDA = 0.95  # GAE lambda。
CLIP_RANGE = 0.2  # PPO clip 范围。
CLIP_RANGE_VF = None  # value clip 范围；None 表示不启用。
NORMALIZE_ADVANTAGE = True  # 是否标准化 advantage。
ENT_COEF = 0.0  # 熵奖励系数。
VF_COEF = 0.5  # value loss 系数。
MAX_GRAD_NORM = 0.5  # 梯度裁剪上限。
USE_SDE = False  # 是否使用 gSDE 探索。
SDE_SAMPLE_FREQ = -1  # gSDE 采样频率，-1 表示只在 rollout 开始采样。
TARGET_KL = None  # KL 早停阈值；None 表示关闭，可试 0.03 或 0.05。
STATS_WINDOW_SIZE = 100  # SB3 日志滑动窗口大小。

POLICY = "MlpPolicy"  # SB3 policy 类型。
POLICY_NET = [64, 64]  # actor MLP hidden sizes。
VALUE_NET = [64, 64]  # critic MLP hidden sizes。
ACTIVATION = "tanh"  # tanh/relu/elu/leaky_relu。
ORTHO_INIT = True  # 是否使用正交初始化。
STD_INIT = 0.5  # 连续动作 Gaussian 初始标准差，传入 PPO 前会取 log。
FULL_STD = True  # gSDE 用；每个特征-动作维度独立 std。
USE_EXPLN = False  # gSDE 用；用 expln 保持 std 正值。
SQUASH_OUTPUT = False  # gSDE 用；是否 tanh squash 动作输出。
SHARE_FEATURES_EXTRACTOR = True  # actor/critic 是否共享特征提取器。
NORMALIZE_IMAGES = True  # SB3 policy 图像归一化开关；当前非图像观测影响不大。
OPTIMIZER_EPS = 1e-5  # Adam eps。
ROLLOUT_BUFFER_CLASS = None  # 默认 RolloutBuffer。
ROLLOUT_BUFFER_KWARGS = None  # 默认 rollout buffer 参数。

SAVE_UPDATE_EVERY = 1  # 每多少个 PPO update 保存一次 checkpoint；1 表示每个 update 都保存。
SAVE_TRAIN_TRAJ = True  # 是否保存 env0 全训练轨迹。
LOG_INTERVAL = 10  # SB3 log_interval。
VERBOSE = 1  # SB3 输出详细程度。
PROGRESS = False  # 是否显示 SB3 progress bar。
