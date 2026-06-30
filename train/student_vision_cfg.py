"""Vision student PPO config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/student_vision")  # 视觉 student PPO 输出目录。
RANDOM_STOP_OUT_DIR = Path("./models/rl/student_vision_random_stop")  # --random-stop 时的输出目录。
CLEAR_OUT_DIR = True  # 每次训练前清空 OUT_DIR。

NUM_ENVS = 128  # 并行环境数量；视觉训练先用较小数量测试显存和速度。
TOTAL_STEPS = 8_192_000  # 总采样步数，按 SB3 total_timesteps 语义。
EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 1  # 训练时只要在 stop 区输出一次 stop 就 success done。
SEED = 1  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。
USE_CAMERA = True  # 视觉 student 必须开启相机。
READ_CAMERA = True  # 每步读取相机 RGB。
RERENDER_ON_RESET = 1  # reset 后重渲染次数，避免相机图像停留在 reset 前一帧。
RENDERING_MODE = "balanced"  # 视觉训练渲染模式，用于临时测速。
ANTIALIASING_MODE = "DLSS"  # 视觉训练抗锯齿/上采样模式，用于临时测速。
DLSS_MODE = 0  # DLSS 模式：0 performance，1 balanced，2 quality，3 auto。
END_D_MIN = 1.5  # state/reward 里的停止距离最小值，单位 m。
END_D_MAX = 1.5  # state/reward 里的停止距离最大值，单位 m。
END_X_MIN = 0.0  # state/reward 里的图像停止位置最小值，单位 px，0 表示图像中心。
END_X_MAX = 0.0  # state/reward 里的图像停止位置最大值，单位 px。
RANDOM_END_D_MIN = 1.3  # --random-stop 时的停止距离最小值，单位 m。
RANDOM_END_D_MAX = 1.8  # --random-stop 时的停止距离最大值，单位 m。
RANDOM_END_X_MIN = -20.0  # --random-stop 时的图像停止位置最小值，单位 px。
RANDOM_END_X_MAX = 20.0  # --random-stop 时的图像停止位置最大值，单位 px。

VISION_CHOICE = "old_reserve"  # 从文件末尾 VISION_CHOICES 里选择一个视觉 state。
FREEZE_VISION = True  # 第一版先冻结视觉 encoder，只训练 PPO。
ENCODER_FP16 = True  # 冻结 encoder 前向使用 fp16 autocast。

TEACHER_PATH = "./models/rl/teacher_ppo/best_val.zip"  # teacher SB3 PPO 模型路径。
RANDOM_TEACHER_PATH = "./models/rl/teacher_ppo_random_stop/best_val.zip"  # --random-stop 时的 teacher 路径。
TRAIN_MODE = "student"  # student 使用 teacher loss；teacher 关闭 teacher loss。
TEACHER_LOSS = 1.0  # teacher imitation loss 权重；0 表示关闭。
TEACHER_LOSS_MIN = 0.01  # teacher imitation loss 权重下限。
TEACHER_DOWN_SUCCESS = 0.3  # 每次 val 后，按 success_rate 乘该值降低 teacher 权重。
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


VISION_CHOICES = {
    "resnet_reserve": {
        "model": "resnet",
        "ckpt": Path("./models/vision/cv_resnet/best.pt"),
        "state": "reserve_feature",
        "dim": 128,
    },
    "resnet_pred": {
        "model": "resnet",
        "ckpt": Path("./models/vision/cv_resnet/best.pt"),
        "state": "pred",
        "dim": 2,
    },
    "mobile_reserve": {
        "model": "mobile",
        "ckpt": Path("./models/vision/cv_mobile/best.pt"),
        "state": "reserve_feature",
        "dim": 128,
    },
    "mobile_pred": {
        "model": "mobile",
        "ckpt": Path("./models/vision/cv_mobile/best.pt"),
        "state": "pred",
        "dim": 2,
    },
    "old_shared": {
        "model": "old",
        "ckpt": Path("./models/vision/cv_old/best.pt"),
        "state": "shared_feature",
        "dim": 516,
    },
    "old_reserve": {
        "model": "old",
        "ckpt": Path("./models/vision/cv_old/best.pt"),
        "state": "reserve_feature",
        "dim": 128,
    },
    "old_pred": {
        "model": "old",
        "ckpt": Path("./models/vision/cv_old/best.pt"),
        "state": "pred",
        "dim": 2,
    },
    "old_mobile_shared": {
        "model": "old-mobile",
        "ckpt": Path("./models/vision/cv_old_mobile/best.pt"),
        "state": "shared_feature",
        "dim": 516,
    },
    "old_mobile_reserve": {
        "model": "old-mobile",
        "ckpt": Path("./models/vision/cv_old_mobile/best.pt"),
        "state": "reserve_feature",
        "dim": 128,
    },
    "old_mobile_pred": {
        "model": "old-mobile",
        "ckpt": Path("./models/vision/cv_old_mobile/best.pt"),
        "state": "pred",
        "dim": 2,
    },
}  # 可对比的视觉 state 配置。
