"""DrQ-v2 Isaac environment parameters."""

import numpy as np

# 仿真基础参数。
DECIMATION = 1
DT = 0.04
EPISODE_S = 15.0
NUM_ENVS = 64
ENV_SPACING = 16.0

# 云端视觉训练默认不用时间上采样，避免 DLSS 历史帧影响 MDP/视觉一致性。
RENDERING_MODE = "balanced"
ANTIALIASING_MODE = "Off"
DLSS_MODE = None
RERENDER_ON_RESET = 1

# 场景和机器人。
ROBOT_USD = "./drq_env/assets/robots/ball_robot.usd"
ROOM_SIZE = 16.0
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.1
ROBOT_Z = 0.5
TARGET_RADIUS = 0.23

# 小球 reset 采样范围。
DIST_MIN = 2.0
DIST_MAX = 7.0
ANGLE_DEG = 45.0

# 相机。
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
FOCAL_LENGTH = 24.0
FOV_X_DEG = 80.0

# 几何标签约定。
LOST_X = -1.0
LOST_D = 0.0
STOP_D = 1.5
STOP_D_TOL = 0.2
STOP_X_TOL = 10.0

# 控制和终止。
STOP_EPS = 0.05
STOP_N = 3
FAIL_NEAR = 1.0
FAIL_FAR = 8.0
XY_SPEED = 1.0
HEAD_SPEED = np.pi / 2

# reward。
K_SEARCH = 0.05
K_TIME = 0.01
K_X = 0.1
K_D = 1.0
SIG_X = 5.0
SIG_D = 0.1

R_FIND = 1.0
R_LOST = -2.0
R_STOP_IN = 1.0
R_STOP_OUT = -2.0
R_STOP = 3.0
R_SUCCESS = 10.0
R_FAIL = -10.0
