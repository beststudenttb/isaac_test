"""任务参数，主要供手动修改。"""
import numpy as np

# 环境基础设置。
DECIMATION = 1  # 每次控制对应的仿真 decimation。
DT = 0.04  # 仿真步长，单位 s。
EPISODE_S = 10.0  # 默认 episode 时长，训练脚本可覆盖。
NUM_ENVS = 64  # 默认环境数量，训练脚本可覆盖。
ENV_SPACING = 16.0  # 默认环境间距，单位 m。
RENDERING_MODE = "quality"  # 默认渲染模式。

# 场景和机器人。
ROBOT_USD = "./assets/robots/ball_robot.usd"  # 机器人 USD 路径。
ROOM_SIZE = 16.0  # 单个房间边长，单位 m。
WALL_HEIGHT = 2.0  # 墙高度，单位 m。
WALL_THICKNESS = 0.1  # 墙厚度，单位 m。
ROBOT_Z = 0.5  # 机器人中心 z 高度，单位 m。
TARGET_RADIUS = 0.23  # 目标球半径，单位 m。

# 小球 reset 采样范围。
DIST_MIN = 2.0  # 小球最小前向距离，单位 m。
DIST_MAX = 7.0  # 小球最大前向距离，单位 m。
ANGLE_DEG = 45.0  # 小球相对机器人前方的左右角度范围，单位 deg。

# 相机默认设置。
USE_CAMERA = False  # 默认不开相机，视觉训练脚本会覆盖。
IMAGE_WIDTH = 224  # 相机图像宽度。
IMAGE_HEIGHT = 224  # 相机图像高度。
FOCAL_LENGTH = 24.0  # 相机 focal length。
FOV_X_DEG = 80.0  # 相机水平视场角，单位 deg。

LOST_X = -1.0  # 看不到球时返回的 px_x。
LOST_D = 0.0  # 看不到球时返回的 dist。

STOP_D = 1.5  # stop 区域中心距离，单位 m。
STOP_D_TOL = 0.2  # stop 区域距离容差，单位 m。
STOP_X_TOL = 10.0  # stop 区域图像中心容差，单位 px。

STOP_EPS = 0.05  # 三个动作绝对值都小于该值时视为 stop。

FAIL_NEAR = 1.0  # 机器人到球平面距离小于该值时 fail。
FAIL_FAR = 8.0  # 机器人到球平面距离大于该值时 fail。

XY_SPEED = 1.0  # 机器人 x/y 平移最大速度，单位 m/s。
HEAD_SPEED = np.pi / 2  # 头部最大角速度，单位 rad/s，默认 pi/2。

K_SEARCH = 0.05  # search 阶段 x/y 平移动作惩罚系数。
K_TIME = 0.01  # 看到球之后的每步时间惩罚系数，search 阶段不使用。
K_X = 1.0  # approach 阶段 px_x 误差改善 reward 系数。
K_D = 1.0  # approach 阶段 dist 误差改善 reward 系数。
K_STOP_A = 0.5  # 保留参数；当前 stop 区域不再使用动作平方惩罚。
K_STOP_DA = 1.0  # stop 区域内动作幅度变小的改善 reward 系数。

SIG_X = 5.0  # stop 区域二维高斯的图像 x 标准差，单位 px。
SIG_D = 0.1  # stop 区域二维高斯的距离标准差，单位 m。

R_FIND = 1.0  # 从看不到球到看到球时的一次性 reward。
R_LOST = -2.0  # 从看到球到看不到球时的一次性 penalty。
R_STOP_IN = 1.0  # 第一次进入 stop 区域时的一次性 reward。
R_STOP_OUT = -2.0  # 从 stop 区域离开时的一次性 penalty。
R_STOP_Q = 0.3  # 保留参数。
R_STOP = 3.0  # stop 区域内输出 stop 信号时按二维高斯强度给的奖励。
R_SUCCESS = 10.0  # success done 时的固定终止 reward。
R_FAIL = -10.0  # fail done 或 timeout 时的终止 penalty。
