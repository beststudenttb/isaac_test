# rl_games_ball

这是用于云服务器测试的 IsaacLab + rl_games PPO 代码包。

目标结构：

- actor：RGB 图像输入，CNN + LSTM。
- critic：非对称特权状态输入，通过 rl_games `central_value_config` 单独训练。
- 环境：IsaacLab `DirectRLEnv`，每个 env 一个独立房间、一个球形机器人、一个红色目标球。

## 文件

- `env.py`：IsaacLab 环境。
- `task_cfg.py`：仿真、相机、reward、done 参数。
- `rl_games_ppo_cnn_lstm_asym.yaml`：rl_games PPO 配置。
- `train.py`：启动 IsaacLab 并运行 rl_games。
- `run_train.sh`：一键训练脚本。
- `assets/robots/ball_robot.usd`：机器人 USD。

## 观测

Actor 观测：

```python
obs["policy"].shape == (num_envs, 3, 224, 224)
```

Critic 特权状态：

```python
obs["critic"].shape == (num_envs, 8)
```

8 个 critic 特权量为：

```text
x_norm
d_norm
seen
forward_norm
left_norm
robot_x_norm
robot_y_norm
head_yaw_norm
```

Actor 看不到这些 critic state。

## 运行

从项目根目录运行：

```bash
./IsaacLab/isaaclab.sh -p rl_games_ball/train.py --num-envs 64
```

或者：

```bash
bash rl_games_ball/run_train.sh
```

可修改：

```bash
NUM_ENVS=128 MAX_EPOCHS=2000 bash rl_games_ball/run_train.sh
```

## 注意

- 默认 `balanced + Off`，不使用 DLSS。
- 默认图像直接来自 TiledCamera，未除以 255；yaml 中 `clip_observations=255.0`，`normalize_input=false`。
- 如果 rl_games 对 CNN 输入要求不同，可优先调整 `env.camera_rgb_chw()` 或 yaml 的 `normalize_input`。
- 当前 central value critic 是 MLP。如果 rl_games 在 recurrent actor + central value 组合下要求 critic 也 recurrent，可以在 `central_value_config.network` 下补与 actor 类似的 `rnn` 配置。
