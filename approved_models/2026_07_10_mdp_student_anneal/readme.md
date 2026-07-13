# 2026-07-10 MDP student (noise env, teacher 退火)

噪声环境（红色目标球 + 蓝色干扰球，`--noise`）下的 MDP latent student，固定 stop 任务（`END_D=1.5, END_X=0`，`RANDOM_STOP=False`）。视觉 encoder 全程冻结（`TRAIN_MDP=False`, `FREEZE_MDP_BACKBONE=True`），只训练 PPO。

与 `2026_07_08_mdp_student_noise` 的**唯一区别是 teacher imitation loss 加了退火**：前 250 个 update 保持 `teacher_coef=1.0`，第 251~400 个 update 线性退到 0，此后恒为 0。encoder、teacher、seed、env 全部相同（encoder md5 `1a177e1d...`，与 07-08 那次逐字节一致）。

跑满 1000 个 update（`NUM_ENVS=128`, `N_STEPS=64`，共 8192000 步）。

## 主要文件

- `best_offline.pt`: 离线 deterministic 评估选出的 best policy（`update_000996.pt`，逐字节相同），对应 `best_offline_info.txt`。
- `last.pt`: update 1000 的 policy。
- `config.py`: 训练现场配置快照。
- `offline_val_config.py`: 离线评估配置。
- `log.csv`: 训练日志（1000 行）。
- `offline_val.csv`: 全部 1000 个 checkpoint 的离线评估结果。
- `val.csv`: 训练期 val 文件（本次 `VAL_EVERY=0`，只有表头）。
- `traj_best_offline.csv`: best checkpoint 全 env 离线轨迹。
- `traj_offline_env0.csv`: 离线评估 env0 轨迹。
- `traj_train_env0.csv`: 训练 env0 轨迹。
- `deps/mdp_state_priv_probe_grad/last.pt`: 本次使用的 MDP state encoder（md5 `1a177e1d822a3ab84ad5de069fd1f40d`）。
- `deps/mdp_state_priv_probe_grad/log.csv`: 该 encoder 的预训练日志。
- `deps/teacher_ppo/best_val.zip`: 4 维 goal teacher（md5 `63c8548599d8935a752c7f3acede31a6`）。

## 结果摘要

离线 deterministic 评估 best（`best_offline_info.txt`）：

- `checkpoint = update_000996.pt`
- `success_rate = 0.9609`, `fail_rate = 0.0`, `timeout_rate = 0.0391`
- `mean_return = 9.4640`, `mean_len = 118.41`

末 50 个 checkpoint 的离线 success：均值 91.9%，最低 76.6%，最高 96.1%。`fail_rate` 全程为 0，失败样本全部是 timeout。

训练 rollout 与离线评估都锁在退火时间点上：

| update | teacher_coef | rollout success | 离线 success（均值 / 最好） |
| ---: | ---: | ---: | ---: |
| 1-250 | 1.00 | 0.3% | 0.1% / 4.7% |
| 251-400 | 0.84→0.17 | 0.4% | 0.1% / 1.6% |
| 401-450 | 0.00 | 7.7% | 0.1% / 0.8% |
| 451-600 | 0.00 | 75.4% | 32.9% / 69.5% |
| 601-800 | 0.00 | 91.2% | 48.7% / 75.0% |
| 801-1000 | 0.00 | 96.7% | 82.0% / 96.1% |

## 与 07-08 那次的对比（同 seed、同 encoder、同 env）

前 250 个 update 两次几乎逐行重合（`teacher_loss` 0.0523 vs 0.0524，`std_mean` 0.453 vs 0.448），差异只有 GPU 非确定性带来的漂移。因此**退火是唯一变量**。

但**不能据此说「退火把 34.4% 提到了 96.1%」**：

- 07-08 那次的 34.4% 只是它**前 407 个 checkpoint**里的最好值，其 751-1000 段从未做过离线评估，且那批中间 checkpoint 已随 `CLEAR_OUT_DIR=True` 删除，无法补测。
- 07-08 那次在 teacher 全开的情况下，rollout 也在 401-500 段自己涨到了 63.4%。**成功率在 update 400 附近出现，两次都有，不是退火独有的现象。**
- 量级正确的对比是末段 rollout：07-08 为 86.4%，本次为 96.0%。

唯一还能做的干净对比：两次的 `last.pt`（update 1000）都还在，对它们各跑一次 `val/mdp_student.py` 即可得到同 update 数下「退火 vs 不退火」的确定性成绩。尚未做。

## 为什么这条线能成功（与 SPR 线的关键差异）

success 要求三个动作分量同时 `|a_i| < STOP_EPS = 0.05`（`src/task_cfg.py:39`）。因此 success 概率由策略的动作噪声 σ 决定。

本 cfg **没有 `STD_MIN`/`STD_MAX` 上下限**，σ 可以从 `STD_INIT=0.5` 自由下降：

| update | σ (std_mean) | P(单维 <0.05) | P(stop)/步 | 出区次数/ep | success |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1-250 | 0.309 | 0.129 | 0.0021 | 2.2 | 0% |
| 401-500 | 0.118 | 0.328 | 0.0354 | 1.5 | 40% |
| 901-1000 | 0.034 | 0.861 | 0.638 | 0.1 | 98% |

σ 塌到 0.034 之后，stop 从「抽奖」变成大概率事件，正反馈（成功 → σ 更小 → 更容易成功）转起来。对照 `2026_07_10_spr_student_center`：那条线的 σ 被 `STD_MAX=0.3` 钉死，每 episode 被噪声踢出 stop 区 7.5 次，最终崩溃。

## 重要限制（读结果前必看）

1. **encoder 带特权 probe 梯度。** `mdp_state_priv_probe_grad` 的 `PROBE_W=1.0` 且 `probe_head` 输入未 detach（`src/cv_extractor/mdp_state.py:167`），px/dist 的特权监督会反传进 encoder。这条线不是纯自监督表征，属对照变体。
2. **encoder 与 `2026_06_27_weekend_students/deps/` 里的同名文件不是同一个。** 本次 md5 `1a177e1d...`，weekend 那份是 `1e63006c...`。不要混用。
3. `models/rl/teacher_ppo/` 现场没有 `config.py`，所以 `deps/teacher_ppo/` 只有 `best_val.zip`。
4. 训练期 `VAL_EVERY=0`，`val.csv` / `traj_val.csv` 只有表头；所有确定性成绩来自训练后的离线评估。

## 未保存内容

- `updates/` 下 1000 个过程 checkpoint（约 356MB）。
- `tb/` tensorboard events。

## 恢复到默认路径

```bash
mkdir -p models/vision/mdp_state_priv_probe_grad models/rl/teacher_ppo models/rl/mdp_student
cp approved_models/2026_07_10_mdp_student_anneal/deps/mdp_state_priv_probe_grad/last.pt models/vision/mdp_state_priv_probe_grad/last.pt
cp approved_models/2026_07_10_mdp_student_anneal/deps/teacher_ppo/best_val.zip models/rl/teacher_ppo/best_val.zip
cp approved_models/2026_07_10_mdp_student_anneal/best_offline.pt models/rl/mdp_student/best_offline.pt
```

训练/评估入口：

```bash
./scripts/run_mdp_student_pipeline.sh
```
