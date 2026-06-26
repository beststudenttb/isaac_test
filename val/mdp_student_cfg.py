"""MDP student offline val config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/mdp_student")  # MDP student 输出目录。
MDP_PATH = Path("./models/vision/mdp_state_priv_probe_grad/last.pt")  # 预训练 MDP state checkpoint。

NUM_ENVS = 32  # 离线 val 默认并行环境数量。
NUM_EPISODES = 1  # 每个 checkpoint 跑多少轮 env。
START = 0  # 从哪个 policy update 开始验证；0 表示从第一个有 MDP checkpoint 的 update 开始。
STRIDE = 1  # 每隔多少个 policy update 验证一次。
VAL_STEPS = 0  # val 最多 step；0 表示使用 env max_episode_length。

EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 3  # 连续多少 step stop 后 success done。
SEED = 0  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。

RENDERING_MODE = "balanced"  # 相机渲染模式。
ANTIALIASING_MODE = "Off"  # 抗锯齿/上采样模式。
DLSS_MODE = None  # DLSS 模式；ANTIALIASING_MODE 不是 DLSS 时保持 None。
RERENDER_ON_RESET = 1  # reset 后重渲染次数。
END_D_MIN = 1.5  # state/reward 里的停止距离最小值，单位 m。
END_D_MAX = 1.5  # state/reward 里的停止距离最大值，单位 m。
END_X_MIN = 0.0  # state/reward 里的图像停止位置最小值，单位 px，0 表示图像中心。
END_X_MAX = 0.0  # state/reward 里的图像停止位置最大值，单位 px。

FREEZE_MDP_BACKBONE = True  # val 时保持 MDP backbone 冻结。

POLICY_NET = [64, 64]  # actor hidden sizes。
VALUE_NET = [64, 64]  # critic hidden sizes。
ACTIVATION = "tanh"  # tanh/relu/elu。
STD_INIT = 0.5  # 初始动作标准差。
