"""MDP latent student PPO config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/mdp_student")  # MDP latent student PPO 输出目录。
CLEAR_OUT_DIR = True  # 每次从头训练前清空 OUT_DIR。

NUM_ENVS = 64  # 并行环境数量。
TOTAL_STEPS = 8_192_000  # 总采样步数，按 SB3 total_timesteps 语义。
EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 3  # 连续多少 step stop 后 success done。
SEED = 1  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。

RENDERING_MODE = "balanced"  # 相机渲染模式。
ANTIALIASING_MODE = "Off"  # 抗锯齿/上采样模式；世界模型默认不用时间上采样。
DLSS_MODE = None  # DLSS 模式；ANTIALIASING_MODE 不是 DLSS 时保持 None。
RERENDER_ON_RESET = 1  # reset 后重渲染次数，避免相机图像停留在 reset 前一帧。
END_D_MIN = 1.5  # state/reward 里的停止距离最小值，单位 m。
END_D_MAX = 1.5  # state/reward 里的停止距离最大值，单位 m。
END_X_MIN = 0.0  # state/reward 里的图像停止位置最小值，单位 px，0 表示图像中心。
END_X_MAX = 0.0  # state/reward 里的图像停止位置最大值，单位 px。

MDP_PATH = Path("./models/vision/mdp_state_priv_probe_grad/last.pt")  # 预训练 MDP state checkpoint。
FREEZE_MDP_BACKBONE = True  # 在线微调 MDP 时冻结 ResNet backbone。
TRAIN_MDP = False  # 是否在 PPO 中周期性微调 MDP。
MDP_WARMUP = 500  # 前多少 update 不训练 MDP，只训练策略。
MDP_EVERY = 20  # warmup 后每多少 update 训练一次 MDP。
MDP_UPDATES = 16  # 每次触发 MDP 训练时做多少次更新。
MDP_BATCH_ENVS = 16    # 每次 MDP 更新抽多少个 env 的轨迹。
MDP_SEQ_LEN = 16  # 每次 MDP 更新使用的连续轨迹长度。
MDP_LR = 1e-5  # MDP 在线微调学习率。
MDP_DYN_W = 1.0  # MDP z 动力学 loss 权重。
MDP_CONTRAST_W = 0.0  # MDP transition contrastive loss 权重。
MDP_CONTRAST_TAU = 0.1  # MDP contrastive softmax 温度。
MDP_IDM_W = 1.0  # MDP 逆动力学 loss 权重。
MDP_VAR_W = 1.0  # MDP VICReg 方差项权重。
MDP_COV_W = 1.0  # MDP VICReg 协方差项权重。
MDP_VAR_GAMMA = 1.0  # MDP VICReg 方差项目标 std 阈值。
MDP_REWARD_W = 1.0  # MDP reward 预测 loss 权重。
MDP_DONE_W = 0.2  # MDP done 预测 loss 权重。
MDP_VALUE_W = 0.1  # MDP value 预测 loss 权重。

TEACHER_PATH = "./models/rl/teacher_ppo/best_val.zip"  # teacher SB3 PPO 模型路径。
TEACHER_LOSS = 1  # teacher imitation loss 权重；0 表示关闭。
TEACHER_LOSS_MIN = 0.01  # teacher imitation loss 权重下限。
TEACHER_DOWN_MDP = 0.3  # 每次 MDP 更新后固定降低的 teacher imitation loss 权重。
TEACHER_DOWN_SUCCESS = 0.00  # 每次 val 后，按 success_rate 乘该值降低 teacher 权重。
TEACHER_UP_OUT = 0.00  # 每次 val 后，按 out/fail_rate 乘该值提高 teacher 权重。
TEACHER_UP_TIMEOUT = 0.0  # 每次 val 后，按 timeout_rate 乘该值调整 teacher 权重；当前占位为 0。

N_STEPS = 64  # 每个 env 每轮 rollout 的 step 数。
BATCH_SIZE = 1024  # PPO minibatch 大小。
N_EPOCHS = 10  # 每批 rollout 重复训练次数。
LR = 3e-4  # PPO 学习率。
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

SAVE_EVERY = 0  # 每多少 update 保存一次 checkpoint；0 表示不保存中间模型。
SAVE_UPDATE_EVERY = 1  # 每多少 update 保存一次逐 update 模型；0 表示关闭。
VAL_EVERY = 0  # 每多少 update 跑一次 deterministic val；0 表示关闭。
VAL_TRAJ_EVERY = 0  # 每多少 update 写一次完整 val 轨迹；0 表示不写。
VAL_STEPS = 0  # val 最多 step；0 表示使用 env max_episode_length。
SAVE_BEST = True  # 是否保存 best.pt 和 best_mdp.pt。
BEST_WARMUP = 20  # 多少 update 之前不保存 best。
BEST_MARGIN = 0.02  # success_rate 至少超过历史最好该值才保存。
LOG_EVERY = 1  # 每多少 update 打印一次日志。
