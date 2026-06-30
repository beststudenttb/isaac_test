"""Teacher offline val config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/teacher_ppo")  # teacher PPO 输出目录。
RANDOM_STOP_OUT_DIR = Path("./models/rl/teacher_ppo_random_stop")  # --random-stop 时的 teacher PPO 输出目录。

NUM_ENVS = 512  # 离线 val 默认并行环境数量。
NUM_EPISODES = 1  # 每个 checkpoint 跑多少轮 env。
START = 0  # 从哪个 update 开始验证。
STRIDE = 1  # 每隔多少个 update 验证一次。
VAL_STEPS = 0  # val 最多 step；0 表示使用 env max_episode_length。

EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 5  # 连续多少 step stop 后 success done。
SEED = 0  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。
USE_CAMERA = False  # teacher 离线 val 默认不用相机。
READ_CAMERA = False  # 是否读取相机 RGB。

END_D_MIN = 1.5  # teacher state/reward 里的停止距离最小值，单位 m。
END_D_MAX = 1.5  # teacher state/reward 里的停止距离最大值，单位 m。
END_X_MIN = 0.0  # teacher state/reward 里的图像停止位置最小值，单位 px，0 表示图像中心。
END_X_MAX = 0.0  # teacher state/reward 里的图像停止位置最大值，单位 px。

RANDOM_END_D_MIN = 1.3  # --random-stop 时的停止距离最小值，单位 m。
RANDOM_END_D_MAX = 1.8  # --random-stop 时的停止距离最大值，单位 m。
RANDOM_END_X_MIN = -20.0  # --random-stop 时的图像停止位置最小值，单位 px。
RANDOM_END_X_MAX = 20.0  # --random-stop 时的图像停止位置最大值，单位 px。
