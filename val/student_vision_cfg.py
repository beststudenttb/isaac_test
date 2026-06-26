"""Vision student offline val config."""

from pathlib import Path


OUT_DIR = Path("./models/rl/student_vision")  # 视觉 student PPO 输出目录。

NUM_ENVS = 128  # 离线 val 默认并行环境数量。
NUM_EPISODES = 1  # 每个 checkpoint 跑多少轮 env。
START = 0  # 从哪个 update 开始验证。
STRIDE = 1  # 每隔多少个 update 验证一次。
VAL_STEPS = 0  # val 最多 step；0 表示使用 env max_episode_length。

EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 3  # 视觉 student 成功需要连续 stop 的 step 数。
SEED = 1  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。
RERENDER_ON_RESET = 1  # reset 后重渲染次数。
ENCODER_FP16 = True  # 冻结 encoder 前向使用 fp16 autocast。
END_D_MIN = 1.5  # state/reward 里的停止距离最小值，单位 m。
END_D_MAX = 1.5  # state/reward 里的停止距离最大值，单位 m。
END_X_MIN = 0.0  # state/reward 里的图像停止位置最小值，单位 px，0 表示图像中心。
END_X_MAX = 0.0  # state/reward 里的图像停止位置最大值，单位 px。

POLICY_NET = [64, 64]  # actor hidden sizes。
VALUE_NET = [64, 64]  # critic hidden sizes。
ACTIVATION = "tanh"  # tanh/relu/elu。
STD_INIT = 0.5  # 初始动作标准差。


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
