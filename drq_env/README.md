# drq_env

这是给 DrQ-v2 / 其他像素强化学习代码使用的 IsaacLab 仿真环境包，只包含环境、配置和机器人 USD，不包含 PPO、CV、MDP 训练代码。

## 文件

- `env.py`：IsaacLab `DirectRLEnv` 环境，返回相机 RGB 图像。
- `task_cfg.py`：仿真、相机、reset、reward 和 done 参数。
- `assets/robots/ball_robot.usd`：机器人 USD。

## 观测和动作

观测：

```python
obs["policy"].shape == (num_envs, 3, 224, 224)
```

返回的是 TiledCamera 的 RGB，通道顺序为 CHW。是否转 float、归一化或做 augmentation 交给 DrQ-v2 代码处理。

动作：

```python
action.shape == (num_envs, 3)
action in [-1, 1]
```

三个动作分别是：

```text
x 方向速度
y 方向速度
头部 yaw 角速度
```

平移动作按当前相机/头部朝向旋转到世界平面后更新机器人位置。

## 默认设置

默认配置在 `task_cfg.py`：

```python
NUM_ENVS = 64
DT = 0.04
EPISODE_S = 15.0
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
RENDERING_MODE = "balanced"
ANTIALIASING_MODE = "Off"
```

`ROBOT_USD` 写死为：

```python
ROBOT_USD = "./drq_env/assets/robots/ball_robot.usd"
```

因此建议从包含 `drq_env/` 的项目根目录启动训练。

## 使用示例

在 IsaacLab 启动后的训练代码中：

```python
from drq_env import DrqBallEnvCfg, make_env

cfg = DrqBallEnvCfg()
cfg.scene.num_envs = 64
cfg.sim.device = "cuda:0"

env = make_env(cfg)
obs, info = env.reset()
```

如果你的 DrQ-v2 代码需要 Gymnasium 风格 wrapper，需要在外层自行适配 IsaacLab `DirectRLEnv` 的 batched tensor 接口。

## 注意

- 这个环境默认开启相机。
- 默认不用 DLSS，避免时间上采样历史帧影响视觉状态一致性。
- reward / done 逻辑来自当前 Isaac 任务：找球、接近 stop 区域、连续 stop 成功、过近/过远失败。
