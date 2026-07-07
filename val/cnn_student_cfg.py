"""Small CNN visual student offline val config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/cnn_fixed_student")  # 与 train 对齐的输出目录。
RANDOM_STOP_OUT_DIR = Path("./models/rl/cnn_fixed_student_random_stop")  # --random-stop 输出目录。

NUM_ENVS = 32  # 离线 val 并行环境数量。
NUM_EPISODES = 1  # 每个 checkpoint 跑多少轮 env。
START = 0  # 从哪个 update 开始验证。
STRIDE = 1  # 每隔多少个 checkpoint 验证一次。
VAL_STEPS = 0  # val 最多 step；0 表示 env max_episode_length。

EPISODE_S = 15.0  # episode 时长，单位 s。
STOP_N = 3  # val 连续 stop 判定步数。
SEED = 0  # 随机种子。
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
POLICY_NET = [64, 64]  # actor hidden sizes。
VALUE_NET = [64, 64]  # critic hidden sizes。
ACTIVATION = "tanh"  # tanh/relu/elu。
STD_INIT = 0.3  # 初始动作标准差。
STD_MAX = 0.3  # 动作标准差上限。
