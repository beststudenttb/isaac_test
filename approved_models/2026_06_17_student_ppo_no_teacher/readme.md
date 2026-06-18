# 2026-06-17 Student PPO No Teacher

自写 PPO student，使用特权观测 `px_x/dist`，不加入 teacher loss，用作 teacher BC 实验的对照组。

## 文件

- `model.pt`: 训练结束的 `last.pt`，建议优先用于策略预览。
- `best.pt`: 按 deterministic val success 保存的 best 模型。
- `config.py`: 本次训练参数。
- `log.csv`: 每个 update 的训练过程指标。
- `val.csv`: 每次 deterministic val 的评估指标。

训练现场的大轨迹文件未复制进本目录，仍在：

- `models/student_ppo/traj_train_env0.csv`
- `models/student_ppo/traj_val.csv`

## 主要配置

- `NUM_ENVS = 512`
- `TOTAL_STEPS = 8_192_000`
- `N_STEPS = 64`
- `TEACHER_LOSS = 0.0`
- `USE_CAMERA = False`

## 结果

`val.csv` 显示：

- update 20: success 0.221, fail 0.000, timeout 0.781
- update 40: success 1.000, fail 0.000, timeout 0.000
- update 240: success 1.000, fail 0.000, timeout 0.000

由于 val 间隔为 20 update，早期收敛差异只能粗略判断：no-teacher 在 update 20 仍有大量 timeout，update 40 后达到稳定成功。
