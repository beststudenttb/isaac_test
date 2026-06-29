# 2026-06-17 Student PPO Teacher 64env Val5 U62

自写 PPO student，64 env，使用特权观测 `px_x/dist`，加入 teacher loss。

## 文件

- `model.pt`: 从 `models/student_ppo/model_50.pt` 复制，当前可用的最新 checkpoint。
- `model_50.pt`: 原始 update 50 checkpoint。
- `best.pt`: 按 val success 保存的 best checkpoint。
- `best_info.txt`: best checkpoint 对应指标。
- `config.py`: 本次训练配置快照。
- `log.csv`: 训练日志，记录到 update 62。
- `val.csv`: deterministic val 日志，`VAL_EVERY = 5`。
- `deps/teacher_ppo/policy.zip`: 对应时期的 2 维特权 teacher，来自 `approved_models/2026_06_16_teacher_ppo/policy.zip`。

注意：原配置写的是 `models/teacher_ppo/best_val.zip`，该精确文件当前现场未找到；本目录保存的是已归档的同代 teacher checkpoint。

未复制大轨迹文件：

- `models/student_ppo/traj_train_env0.csv`
- `models/student_ppo/traj_val.csv`

## 主要配置

- `NUM_ENVS = 64`
- `TOTAL_STEPS = 8_192_000`
- `N_STEPS = 64`
- `VAL_EVERY = 5`
- `VAL_TRAJ_EVERY = 20`
- `TEACHER_LOSS = 1.0`
- `TEACHER_LOSS_MIN = 0.01`

## 结果

`val.csv` 早期指标：

- update 5: success 0.000, fail 0.750, timeout 0.250
- update 10: success 0.000, fail 0.000, timeout 1.000
- update 15: success 1.000, fail 0.000, timeout 0.000
- update 20: success 1.000, fail 0.000, timeout 0.000

`best.pt` 在 update 20 保存，当时 success 1.000，mean_len 188.375。

## 注意

当前训练日志记录到 update 62，但由于 `SAVE_EVERY = 50`，本目录最新可用模型权重是 update 50。
