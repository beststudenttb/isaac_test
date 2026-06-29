# 2026-06-17 Student PPO Teacher BC

自写 PPO student，使用特权观测 `px_x/dist`，并加入 SB3 teacher 的 deterministic action MSE 作为 teacher loss。

## 文件

- `model.pt`: 训练结束的 `last.pt`，建议优先用于策略预览。
- `best.pt`: 第一次达到 best 条件时保存的模型。
- `config.py`: 本次训练参数。
- `log.csv`: 每个 update 的训练过程指标。
- `val.csv`: 每次 deterministic val 的评估指标。
- `deps/teacher_ppo/policy.zip`: 对应时期的 2 维特权 teacher，来自 `approved_models/2026_06_16_teacher_ppo/policy.zip`。

注意：原配置写的是 `models/teacher_ppo/best_val.zip`，该精确文件当前现场未找到；本目录保存的是已归档的同代 teacher checkpoint。

训练现场的大轨迹文件未复制进本目录，仍在：

- `models/student_ppo/traj_train_env0.csv`
- `models/student_ppo/traj_val.csv`

## 主要配置

- `NUM_ENVS = 512`
- `TOTAL_STEPS = 8_192_000`
- `N_STEPS = 64`
- `TEACHER_LOSS = 1.0`
- `USE_CAMERA = False`

## 结果

`val.csv` 后期指标：

- update 200: success 0.984, fail 0.004, timeout 0.012
- update 220: success 0.971, fail 0.000, timeout 0.029
- update 240: success 0.998, fail 0.000, timeout 0.002

`best.pt` 在 update 20 保存，当时 success 已经达到 1.0；后续模型平均长度更短，因此看策略时优先试 `model.pt`。
