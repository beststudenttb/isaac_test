# 2026-07-05 SPR student

周末 SPR student 实验保存。该实验使用固定 stop 任务，SPR stage1 预训练后，在固定 `stage1_end.pt` 上训练 PPO student，并用离线 deterministic val 选择 best。

## 主要文件

- `stage1_end.pt`: stage1 结束的 SPR encoder / target / transition checkpoint。
- `best_offline.pt`: 离线评估选出的 best policy checkpoint，对应 `best_offline_info.txt`。
- `teacher_best_val.zip`: 训练时使用的固定任务 teacher，来自 `models/rl/teacher_ppo/best_val.zip`。
- `config.py`: 训练现场配置。
- `offline_val_config.py`: 离线评估配置。
- `log.csv`: 训练过程日志。
- `spr_log.csv`: SPR loss / eff_rank 等表征日志。
- `offline_val.csv`: 各 checkpoint 离线评估结果。
- `val.csv`: 训练期 val 文件。
- `traj_best_offline.csv`: best checkpoint 的全 env 离线轨迹。
- `traj_offline_env0.csv`: 离线评估 env0 轨迹。
- `traj_train_env0.csv`: 训练 env0 轨迹。
- `preview_best.csv`: 单 env preview 轨迹。

## 结果摘要

- best 离线评估对应 `ppo_000890.pt + stage1_end.pt`。
- `success_rate = 0.9375`, `fail_rate = 0.0`, `timeout_rate = 0.0625`。
- 中期固定 SPR 的 PPO 可以达到较高成功率；stage4 继续更新 SPR 后表征出现坍缩风险。

## 恢复到默认路径

如需在默认脚本中直接使用，可手动复制：

```bash
cp approved_models/2026_07_05_spr_student/stage1_end.pt models/rl/spr_student/stage1_end.pt
cp approved_models/2026_07_05_spr_student/best_offline.pt models/rl/spr_student/best_offline.pt
cp approved_models/2026_07_05_spr_student/teacher_best_val.zip models/rl/teacher_ppo/best_val.zip
```
