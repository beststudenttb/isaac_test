"""DrQ student offline val config."""

from pathlib import Path


# OUT_DIR 必须与 train/drq_student_cfg.py 一致(07-11 spr val 曾因目录不一致 FileNotFoundError 崩掉 pipeline)。
OUT_DIR = Path("./models/rl/drq_student_spr")  # spr_z 模式输出目录。
PIXELS_OUT_DIR = Path("./models/rl/drq_student_pixels")  # pixels 模式输出目录。
RANDOM_STOP_SUFFIX = "_random_stop"  # --random-stop 时输出目录追加后缀。
SPR_CKPT = "spr_encoder.pt"  # spr_z 模式配对的冻结 encoder(train 启动时已复制进输出目录)。

NUM_ENVS = 32  # 离线 val 默认并行环境数量。
NUM_EPISODES = 1  # 每个 checkpoint 跑多少轮 env。
START = 0  # 从哪个 env step 开始验证。
STRIDE = 1  # 每隔多少个 checkpoint 验证一次。
VAL_STEPS = 0  # val 最多 step；0 表示使用 env max_episode_length。

EPISODE_S = 15.0  # 单个 episode 最长仿真时间，单位秒。
STOP_N = 3  # 连续多少 step stop 后 success done(与其余线的离线 val 一致)。
SEED = 0  # 随机种子。
DEVICE = None  # None 表示使用 IsaacLab 命令行 device。

RENDERING_MODE = "balanced"  # 相机渲染模式。
ANTIALIASING_MODE = "Off"  # 抗锯齿模式。
DLSS_MODE = None  # DLSS 模式；ANTIALIASING_MODE 不是 DLSS 时保持 None。
RERENDER_ON_RESET = 1  # reset 后重渲染次数。
END_D_MIN = 1.5  # 固定任务停止距离最小值，单位 m。
END_D_MAX = 1.5  # 固定任务停止距离最大值，单位 m。
END_X_MIN = 0.0  # 固定任务停止图像位置最小值，单位 px。
END_X_MAX = 0.0  # 固定任务停止图像位置最大值，单位 px。
RANDOM_END_D_MIN = 1.3  # --random-stop 时的停止距离最小值，单位 m。
RANDOM_END_D_MAX = 1.8  # --random-stop 时的停止距离最大值，单位 m。
RANDOM_END_X_MIN = -20.0  # --random-stop 时的图像停止位置最小值，单位 px。
RANDOM_END_X_MAX = 20.0  # --random-stop 时的图像停止位置最大值，单位 px。
