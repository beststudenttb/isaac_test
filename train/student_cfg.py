"""Student PPO config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/student_ppo")  # student PPO 输出目录。
RANDOM_STOP_OUT_DIR = Path("./models/rl/student_ppo_random_stop")  # --random-stop 时的输出目录。
CLEAR_OUT_DIR = True  # 每次训练前清空 OUT_DIR。

NUM_ENVS = 512  # 并行环境数量。
TOTAL_STEPS = 8_192_000  # 总采样步数，按 SB3 total_timesteps 语义。
EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 1  # 训练时只要在 stop 区输出一次 stop 就 success done。
SEED = 0  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。
USE_CAMERA = False  # 复现阶段默认不用相机。
READ_CAMERA = False  # 是否读取相机 RGB。
END_D_MIN = 1.5  # state/reward 里的停止距离最小值，单位 m。
END_D_MAX = 1.5  # state/reward 里的停止距离最大值，单位 m。
END_X_MIN = 0.0  # state/reward 里的图像停止位置最小值，单位 px，0 表示图像中心。
END_X_MAX = 0.0  # state/reward 里的图像停止位置最大值，单位 px。
RANDOM_END_D_MIN = 1.3  # --random-stop 时的停止距离最小值，单位 m。
RANDOM_END_D_MAX = 1.8  # --random-stop 时的停止距离最大值，单位 m。
RANDOM_END_X_MIN = -20.0  # --random-stop 时的图像停止位置最小值，单位 px。
RANDOM_END_X_MAX = 20.0  # --random-stop 时的图像停止位置最大值，单位 px。
TEACHER_PATH = "./models/rl/teacher_ppo/best_val.zip"  # teacher SB3 PPO 模型路径。
RANDOM_TEACHER_PATH = "./models/rl/teacher_ppo_random_stop/best_val.zip"  # --random-stop 时的 teacher 路径。
TEACHER_LOSS = 0.0  # teacher imitation loss 权重；0 表示关闭。
TEACHER_LOSS_MIN = 0.01  # teacher imitation loss 权重下限。
TEACHER_DOWN_SUCCESS = 0.00  # 每次 val 后，按 success_rate 乘该值降低 teacher 权重。
TEACHER_UP_OUT = 0.00  # 每次 val 后，按 out/fail_rate 乘该值提高 teacher 权重。
TEACHER_UP_TIMEOUT = 0.0  # 每次 val 后，按 timeout_rate 乘该值调整 teacher 权重；当前占位为 0。

N_STEPS = 64  # 每个 env 每轮 rollout 的 step 数。
BATCH_SIZE = 1024  # PPO minibatch 大小。
N_EPOCHS = 10  # 每批 rollout 重复训练次数。
LR = 3e-4  # 学习率。
GAMMA = 0.99  # 折扣因子。
GAE_LAMBDA = 0.95  # GAE lambda。
CLIP_RANGE = 0.2  # policy clip 范围。
CLIP_RANGE_VF = None  # value clip 范围；None 表示不用。
NORMALIZE_ADVANTAGE = True  # 是否标准化 advantage。
ENT_COEF = 0.0  # 熵系数。
VF_COEF = 0.5  # value loss 系数。
MAX_GRAD_NORM = 0.5  # 梯度裁剪。

POLICY_NET = [64, 64]  # actor hidden sizes。
VALUE_NET = [64, 64]  # critic hidden sizes。
ACTIVATION = "tanh"  # tanh/relu/elu。
STD_INIT = 0.5  # 初始动作标准差。

SAVE_EVERY = 50  # 每多少 update 保存一次 checkpoint；0 表示不保存中间模型。
SAVE_UPDATE_EVERY = 1  # 每多少 update 保存一次逐 update 模型；0 表示关闭。
VAL_EVERY = 0  # 每多少 update 跑一次 deterministic val；0 表示关闭。
VAL_TRAJ_EVERY = 0  # 每多少 update 写一次完整 val 轨迹；0 表示不写。
VAL_STEPS = 0  # val 最多 step；0 表示使用 env max_episode_length。
SAVE_BEST = True  # 是否保存 best.pt。
BEST_WARMUP = 20  # 多少 update 之前不保存 best。
BEST_MARGIN = 0.02  # success_rate 至少超过历史最好该值才保存。

LOG_EVERY = 1  # 每多少 update 打印一次日志。
