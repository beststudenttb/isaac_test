# 2026-06-18 student vision old_xd_student

保存当前 `models/student_vision/old_xd_student` 的核心结果，避免重新训练时被清空。

- `last.pt`: 当前训练末尾模型。
- `update_000500.pt`: 离线 val 中表现较好的过程模型，`success_rate=0.8125`。
- `config.py`: 训练配置快照。
- `log.csv`: 训练过程日志。
- `val.csv`: 训练时 val 日志。
- `offline_val.csv`: 离线 val 结果。
- `offline_val_config.py`: 离线 val 配置。
- `deps/cv_old/best.pt`: 视觉预训练模型。
- `deps/teacher_ppo/policy.zip`: 对应时期的 2 维特权 teacher，来自 `approved_models/2026_06_16_teacher_ppo/policy.zip`。

没有保存完整 `updates/` 目录；原目录约 526MB，其中 `updates/` 约 485MB。

注意：原配置写的是 `models/teacher_ppo/best_val.zip` 和 `models/cv_old/best.pt`。其中 teacher 的精确 `best_val.zip` 当前现场未找到；本目录保存的是已归档的同代 teacher checkpoint。CV checkpoint 已按当前现场 `models/vision/cv_old/best.pt` 补齐。
