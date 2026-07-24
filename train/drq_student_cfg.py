"""DrQ-v2 off-policy student config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/drq_student_spr")  # spr_z 模式输出目录。
PIXELS_OUT_DIR = Path("./models/rl/drq_student_pixels")  # pixels 模式输出目录。
RANDOM_STOP_SUFFIX = "_random_stop"  # --random-stop 时输出目录追加后缀。
CLEAR_OUT_DIR = True  # 从头训练时清空输出目录。

OBS_MODE = "spr_z"  # "spr_z"(冻结 SPR z + goal) 或 "pixels"(DrQ-v2 原版:可训练 CNN + 增广)。
# spr_z 模式的冻结 encoder 来源。指向保全的副本而不是 models/rl/spr_student/stage1_end.pt:
# (1) 那个目录会被下次 SPR 跑的 CLEAR_OUT_DIR 清掉;(2) 这份(md5 cfd52c0f)正是 σ 调度消融
# 那次 PPO 用的同一个 encoder —— arm A 的意义就是"同一个 z,只换算法",encoder 必须逐字节一致。
SPR_CKPT = Path("./models/rl/_encoders/stage1_sigma_ablation.pt")

NUM_ENVS = 32  # 并行环境数量。
TOTAL_ENV_STEPS = 4_000_000  # 总采样 env 步数(所有 env 合计);2M 收官时离线曲线仍是正斜率(76.6% 且还在涨),加倍看能到哪。
SEED_STEPS = 4000  # 前多少 env 步用均匀随机动作填 buffer(也是学习 warmup)。
UPDATES_PER_TICK = 64  # 每个采集 tick 做多少次梯度更新;128 env 下 replay ratio=64*256/128=128,与 arm B(pixels,32 env)一致;总更新 15625tick*64=1M,也与 pixels 对齐。
EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 1  # 训练时 stop 一次即 success。
SEED = 1  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。

RENDERING_MODE = "balanced"  # 相机渲染模式。
ANTIALIASING_MODE = "Off"  # 抗锯齿模式。
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

CAPACITY_PER_ENV = 8192  # replay 每 env 行数;32 env x 8192 x 224x224x3 uint8 约 38GB 内存。
N_STEP = 3  # n-step TD。
BATCH_SIZE = 256  # 每次梯度更新的样本数。
LR = 1e-4  # actor/critic/encoder 学习率。
GAMMA = 0.99  # 折扣因子。
TAU = 0.01  # critic target EMA 系数。
FEATURE_DIM = 50  # trunk 输出维度(LayerNorm+Tanh)。
HIDDEN_DIM = 1024  # actor/critic MLP hidden。

STD_START = 0.3  # 探索噪声 σ 起点(调度值,不是可学参数)。
STD_END = 0.05  # 探索噪声 σ 终点;=STOP_EPS,μ≈0 时训练期 P(stop)/步约 0.32。
STD_DECAY_STEPS = 1_500_000  # σ 线性衰减用的 env 步数。
NOISE_CLIP = 0.3  # target smoothing / actor 采样的噪声截断。

PIXELS_RES = 112  # pixels 模式训练分辨率(存储恒为 224,采样时 GPU resize)。
AUG_PAD = 6  # RandomShiftsAug 平移幅度(112 分辨率约对应原版 84/pad4 的比例)。

SAVE_EVERY_STEPS = 25_000  # 每多少 env 步存一个 drq_*.pt(2M 步 → 80 个)。
LOG_EVERY_TICKS = 25  # 每多少 tick 聚合写一行 log.csv。
