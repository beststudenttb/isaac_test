# 2026-06-22 Student Vision Old State Compare

本目录保存三组使用 `old` 视觉模型不同 state 的视觉 student PPO 结果，用于后续汇报和复现实验对比。

## 共同设置

- 视觉模型：`models/vision/cv_old/best.pt`
- 渲染：`balanced + DLSS mode 0`
- 训练环境数：128
- PPO rollout：`N_STEPS = 64`
- 总训练步数：8192000
- stop 连续步数：`STOP_N = 3`
- 离线评估：32 env，1 episode/env，stride 1
- teacher loss：0.0

## 结果

| 目录 | state | best update | best success | last success | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `old_xd/` | `pred` | 628 | 0.84375 | 0.28125 | 中期可用，后期退化；失败多为未对准中心就停住。 |
| `old_feature/` | `reserve_feature` | 710 | 0.0 | 0.0 | 基本失败；能到 stop 附近，但不输出 stop 动作。 |
| `old_shared/` | `shared_feature` | 918 | 1.0 | 0.96875 | 当前最好；成功率和轨迹质量都明显优于另外两个。 |

## 文件说明

每个子目录包含：

- `best_offline.pt`：离线评估选出的 best 模型。
- `last.pt`：训练结束模型。
- `model_1000.pt`：第 1000 update 保存模型。
- `config.py`：训练配置快照。
- `offline_val_config.py`：离线评估配置快照。
- `log.csv`：训练过程日志。
- `offline_val.csv`：逐 checkpoint 离线评估结果。
- `traj_best_offline.csv`：best checkpoint 的离线评估轨迹。
- `best_offline_info.txt`：best checkpoint 摘要。

`updates/` 未保存，因为三个目录的逐 update checkpoint 合计接近 1.5G，不适合作为 approved 结果上传。

## 依赖模型

- `deps/cv_old/best.pt`: 三组视觉 student 共同使用的 `old` 视觉预训练模型。
- `deps/cv_old/log.csv`: 对应视觉预训练日志。
