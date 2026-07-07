"""SPR visual student PPO config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/spr_student")  # SPR student 输出目录。
RANDOM_STOP_OUT_DIR = Path("./models/rl/spr_student_random_stop")  # --random-stop 时的输出目录。
CLEAR_OUT_DIR = True  # 从头训练时清空输出目录。

NUM_ENVS = 64  # 并行环境数量。
N_STEPS = 64  # 每个 env 每轮 rollout step 数。
UPDATES = 2000  # 总 update 数。
TOTAL_STEPS = NUM_ENVS * N_STEPS * UPDATES  # 总采样步数,由上面三个基数相乘。
EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 1  # 训练时 stop 一次即 success。
SEED = 1  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。

RENDERING_MODE = "balanced"  # 相机渲染模式。
ANTIALIASING_MODE = "Off"  # SPR 默认不用时间上采样。
DLSS_MODE = None  # ANTIALIASING_MODE 不是 DLSS 时保持 None。
RERENDER_ON_RESET = 1  # reset 后重渲染次数。

END_D_MIN = 1.5  # 固定任务停止距离最小值。
END_D_MAX = 1.5  # 固定任务停止距离最大值。
END_X_MIN = 0.0  # 固定任务停止图像位置最小值。
END_X_MAX = 0.0  # 固定任务停止图像位置最大值。
RANDOM_END_D_MIN = 1.3  # 随机任务停止距离最小值。
RANDOM_END_D_MAX = 1.8  # 随机任务停止距离最大值。
RANDOM_END_X_MIN = -20.0  # 随机任务停止图像位置最小值。
RANDOM_END_X_MAX = 20.0  # 随机任务停止图像位置最大值。

Z_DIM = 128  # SPR latent 维度。
FPN_DIM = 128  # FPN 通道数。
SPR_HIDDEN = 256  # SPR transition/projector hidden。
SPR_POOL = 7  # encoder 池化尺寸。
SPR_PRETRAINED = True  # ResNet18 是否使用 ImageNet 预训练。
FREEZE_STEM_LAYERS = True  # 冻结 stem/layer1/layer2，只训练 layer3 及后续表征层。
SPR_TAU = 0.99  # target encoder EMA 系数。
SPR_COEF = 1.0  # SPR 一步转移 loss 权重。
SPR_K = 8  # SPR 使用多少步转移目标。
STAGE1_UPDATES = 200  # 阶段1 update 数:teacher 加噪带跑,只训 SPR/encoder。
STAGE2_UPDATES = 500  # 阶段2 update 数:冻结 SPR,student PPO + teacher BC(全权重)。
STAGE3_UPDATES = 300  # 阶段3 update 数:冻结 SPR,teacher 权重线性退火到 0;0 表示跳过(硬切);其余归阶段4。
STAGE1_STEPS = NUM_ENVS * N_STEPS * STAGE1_UPDATES
STAGE2_STEPS = NUM_ENVS * N_STEPS * STAGE2_UPDATES
STAGE3_STEPS = NUM_ENVS * N_STEPS * STAGE3_UPDATES
TEACHER_NOISE_STD = 0.3  # 阶段1 teacher 动作上的高斯噪声 std。
RAND_ACTION_P = 0.1  # 阶段1 每步整 env 替换为均匀随机动作的概率。
SPR_UPDATE_EVERY = 0  # 阶段4 每多少个 update 更新一次 SPR/encoder;0=冻结(裸解冻已实证 2 次更新内塌缩,见 07-06 分析)。
ACTOR_ENCODER_COEF = 0.0  # SPR 更新时 actor loss 回传 encoder 的比例。

TEACHER_PATH = "./models/rl/teacher_ppo/best_val.zip"  # 固定任务 teacher。
RANDOM_TEACHER_PATH = "./models/rl/teacher_ppo_random_stop/best_val.zip"  # 随机任务 teacher。
TEACHER_LOSS = 5.0  # teacher imitation loss 权重；0 表示关闭。
TEACHER_LOSS_TYPE = "mse"  # "mse" 或 "nll"(CE/最大似然,会同时更新 std;梯度含 1/σ² 放大,权重需大幅调小)。
TEACHER_LOSS_MIN = 0.01  # teacher loss 下限。
TEACHER_DOWN_SUCCESS = 0.0  # val 后按 success 降低 teacher 权重。
TEACHER_UP_OUT = 0.0  # val 后按 fail 提高 teacher 权重。
TEACHER_UP_TIMEOUT = 0.0  # val 后按 timeout 调整 teacher 权重。

BATCH_SIZE = 32  # PPO minibatch 大小；可训练视觉 encoder 显存开销大，先用小 batch。
N_EPOCHS = 4  # 每批 rollout 重复训练次数。
LR = 3e-4  # 学习率。
GAMMA = 0.99  # 折扣因子。
GAE_LAMBDA = 0.95  # GAE lambda。
CLIP_RANGE = 0.2  # PPO clip。
CLIP_RANGE_VF = None  # value clip；None 表示不用。
NORMALIZE_ADVANTAGE = True  # 是否标准化 advantage。
ENT_COEF = 0.0  # 熵系数。
VF_COEF = 0.5  # value loss 系数。
Q_COEF = 0.2  # Q(z,a) 辅助 loss 系数;阶段1塑形表征,阶段2/3只训头,不改变 PPO 主流程。
Q_WARMUP_UPDATES = 50  # 阶段1 Q 系数线性升满所需 update 数(Q 最不准时话语权最小)。
RETURN_SCALE = 10.0  # Q 回归目标的固定归一常数(return/10),保持 q_loss O(1)。
MAX_GRAD_NORM = 0.5  # 梯度裁剪。

POLICY_NET = [64, 64]  # actor hidden sizes。
VALUE_NET = [64, 64]  # critic hidden sizes。
ACTIVATION = "tanh"  # tanh/relu/elu。
STD_INIT = 0.5  # 初始动作标准差。
STD_MIN = 0.02  # 动作标准差下限。
STD_MAX = 0.3  # 动作标准差上限。

SAVE_UPDATE_EVERY = 10  # 每多少 update 保存一次;阶段1存 spr_*.pt(仅SPR),阶段2/3存 ppo_*.pt(仅策略),阶段4存 full_*.pt(完整)。
VAL_EVERY = 0  # 每多少 update 跑 deterministic val；0 表示关闭。
VAL_STEPS = 0  # val 最大 step；0 表示使用 env max_episode_length。
SAVE_BEST = True  # 是否按 val success 保存 best。
BEST_WARMUP = 20  # 多少 update 前不保存 best。
BEST_MARGIN = 0.02  # best success 至少提升多少才保存。
LOG_EVERY = 1  # 每多少 update 打印一次日志。
