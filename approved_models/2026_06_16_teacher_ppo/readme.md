# 2026-06-16 Teacher PPO

## 内容

- `policy.zip`: rollout 60 对应的 checkpoint，来源 `models/teacher_ppo/teacher_7864320_steps.zip`。
- `config.py`: 当次训练现场配置快照。
- `best_val_info.txt`: SB3 训练时保存的 best 记录。

## 训练设置

- 环境数量：2048
- 每轮 rollout step：64
- 总步数目标：10,000,000
- 策略：SB3 PPO `MlpPolicy`
- 观测：归一化后的 `px_x, dist`
- 动作：`x, y, omega`
- 相机：训练时未启用，相机状态由几何投影计算

## 指标

`policy.zip` 对应的 rollout 60 轨迹统计：

- success_rate: 0.988
- fail_rate: 0.000
- timeout_rate: 0.012
- mean_len: 116.9

当前 `best_val.zip` 对应 rollout 40，success_rate 为 0.981。由于 best 保存使用 `BEST_MARGIN=0.02`，rollout 60 没有覆盖 `best_val.zip`，因此本目录手动保存 rollout 60 checkpoint。

## 备注

失败样本主要集中在右侧大角度远目标，末尾通常已经进入 stop 区，但连续 stop step 不足 5 次，被 timeout 截断。
