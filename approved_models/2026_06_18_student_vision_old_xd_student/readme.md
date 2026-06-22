# 2026-06-18 student vision old_xd_student

保存当前 `models/student_vision/old_xd_student` 的核心结果，避免重新训练时被清空。

- `last.pt`: 当前训练末尾模型。
- `update_000500.pt`: 离线 val 中表现较好的过程模型，`success_rate=0.8125`。
- `config.py`: 训练配置快照。
- `log.csv`: 训练过程日志。
- `val.csv`: 训练时 val 日志。
- `offline_val.csv`: 离线 val 结果。
- `offline_val_config.py`: 离线 val 配置。

没有保存完整 `updates/` 目录；原目录约 526MB，其中 `updates/` 约 485MB。
