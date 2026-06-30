"""Privileged student offline val config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/student_ppo")  # student PPO 输出目录。
RANDOM_STOP_OUT_DIR = Path("./models/rl/student_ppo_random_stop")  # --random-stop 时的 student PPO 输出目录。

NUM_ENVS = 32  # 离线 val 默认并行环境数量。
NUM_EPISODES = 1  # 每个 checkpoint 跑多少轮 env。
START = 0  # 从哪个 update 开始验证。
STRIDE = 1  # 每隔多少个 update 验证一次。
VAL_STEPS = 0  # val 最多 step；0 表示使用 env max_episode_length。

EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 3  # 连续多少 step stop 后 success done。
SEED = 0  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。
USE_CAMERA = False  # 特权 student 离线 val 默认不用相机。
READ_CAMERA = False  # 是否读取相机 RGB。
END_D_MIN = 1.5  # state/reward 里的停止距离最小值，单位 m。
END_D_MAX = 1.5  # state/reward 里的停止距离最大值，单位 m。
END_X_MIN = 0.0  # state/reward 里的图像停止位置最小值，单位 px，0 表示图像中心。
END_X_MAX = 0.0  # state/reward 里的图像停止位置最大值，单位 px。
RANDOM_END_D_MIN = 1.3  # --random-stop 时的停止距离最小值，单位 m。
RANDOM_END_D_MAX = 1.8  # --random-stop 时的停止距离最大值，单位 m。
RANDOM_END_X_MIN = -20.0  # --random-stop 时的图像停止位置最小值，单位 px。
RANDOM_END_X_MAX = 20.0  # --random-stop 时的图像停止位置最大值，单位 px。

POLICY_NET = [64, 64]  # actor hidden sizes。
VALUE_NET = [64, 64]  # critic hidden sizes。
ACTIVATION = "tanh"  # tanh/relu/elu。
STD_INIT = 0.5  # 初始动作标准差。
