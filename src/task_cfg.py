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
ANGLE_DEG = 60.0  # 小球相对机器人前方的左右角度范围，单位 deg。±60=张角120°,相机 FOV 只有 80°(±40°),33% 的 episode 开局看不见球(原 ±45 只有 3%),搜索成为常态而非边缘情况。

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
SCORE_MODE = "angle"  # 2026-09-11 用户:奖励与 stop 判定改用物理量 —— 目标相对机器人正前方的方位角(deg)与真实距离(m);"pixel" 为旧的像素版
STOP_ANG_TOL = 3.0  # angle 模式的 stop 区方位角容差,单位 deg(fx=133.5px 时 3° ≈ 7px)

STOP_EPS = 0.05  # 三个动作绝对值都小于该值时视为 stop。

FAIL_NEAR = 0.5  # 机器人到球平面距离小于该值时 fail。
FAIL_FAR = 8.0  # 机器人到球平面距离大于该值时 fail。

XY_SPEED = 1.0  # 机器人 x/y 平移最大速度，单位 m/s。
HEAD_SPEED = np.pi / 2  # 头部最大角速度，单位 rad/s，默认 pi/2。

K_SEARCH = 0.05  # search 阶段 x/y 平移动作惩罚系数。
K_SEARCH_W = 0.01  # search 阶段转头动作奖励系数。
K_TIME = 0.003  # 看到球之后的每步时间惩罚系数，search 阶段不使用。
K_X = 1.0  # approach 阶段 px_x 误差改善 reward 系数。
K_D = 1.0  # approach 阶段 dist 误差改善 reward 系数。
K_STOP_A = 0.5  # 保留参数；当前 stop 区域不再使用动作平方惩罚。
K_STOP_DA = 0.0  # stop 区域内动作幅度变小的改善 reward 系数。

SIG_X = 5.0  # stop 区域二维高斯的图像 x 标准差，单位 px。
SIG_D = 0.1  # stop 区域二维高斯的距离标准差，单位 m。

R_FIND = 1.0  # 从看不到球到看到球时的一次性 reward。
R_LOST = -2.0  # 从看到球到看不到球时的一次性 penalty。
R_STOP_IN = 1.0  # 第一次进入 stop 区域时的一次性 reward。
R_STOP_OUT = -2.0  # 从 stop 区域离开时的一次性 penalty。
R_STOP_Q = 0.3  # 保留参数。
R_STOP = 0.0  # stop 区域内输出 stop 信号时按二维高斯强度给的奖励。
R_SUCCESS = 10.0  # success done 时的固定终止 reward。
R_FAIL = -10.0  # fail done 或 timeout 时的终止 penalty。

# ---- 任务 v3「稠密位置分」(2026-09-04) ----
# 每步得分 score = 1 / (1 + xe/SCORE_X_SCALE + de/SCORE_D_SCALE),上限 1,看不见球给 0。
# episode 固定 375 步、无提前终止、无 stop 判定,回报 = 375 步得分之和。
# 两个尺度就是"误差多大时得分掉一半"。在 teacher 真实轨迹上选的:全程单调有梯度
# (4m 处 0.162 -> 1.5m 处 0.891),且 1.7m 与 1.5m 还差 0.14(纯线性只差 0.016)。
SCORE_ANG_SCALE = 8.5  # angle 模式位置分里方位角的尺度,deg(≈ 像素版 20px 在画面中心的角度)
SCORE_X_SCALE = 20.0  # 图像横向误差的半分尺度,单位 px。
# cam 模式(2026-09-14):误差量全在相机坐标 (x_px, 直径px) 上,尺度取成在停车点与 angle 模式等价。
# 直径(px) = DIAM_K / d(从真实帧拟合,DIAM_K = 2*31.06);d=1.5m 时直径 41.4px。
# 8.5° ≈ 20px(fx=133.5);1.5m 处 0.5m ≈ 13.8px 直径变化;3° ≈ 7px;0.2m ≈ 5.5px。
DIAM_K = 62.12
SCORE_DIAM_SCALE = 13.8  # cam 模式距离分的半分尺度,单位 px(直径)
STOP_X_TOL_CAM = 7.0     # cam 模式 stop 区横向容差,px(≈3°)
STOP_DIAM_TOL = 5.5      # cam 模式 stop 区直径容差,px(≈0.2m @1.5m)
SCORE_D_SCALE = 0.5   # 距离误差的半分尺度,单位 m。
# 动作惩罚(2026-09-08):score *= 1 - K * max(0, max|a| - DEAD)。乘法形式:远处 score 小、
# 惩罚绝对量也小,不伤接近段;近处 score≈1 才咬住。DEAD 以内不罚,给 |a|<0.05 一个平台。
# K=0 即无惩罚(v3 原版)。teacher 不罚也不震荡(区内 max|a| 均值 0.083),罚是为了让 student 停下来。
SCORE_ACTION_K = 0.3  # 2026-09-08 扫描 k∈{0,.3,.6,1,1.5} 后选定:v1 成功率 100%、停车误差 1.4cm、接近段满速、回报只掉 11
SCORE_ACTION_DEAD = 0.05

R_HOVER = 1.0  # 任务 v2(hover):seen 时每步按 stop 高斯给的状态分,无事件项、无提前终止。
