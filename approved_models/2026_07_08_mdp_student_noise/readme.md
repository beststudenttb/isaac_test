# 2026-07-08 MDP student (noise env)

噪声环境（红色目标球 + 蓝色干扰球，`--noise`）下的 MDP latent student，固定 stop 任务（`END_D=1.5, END_X=0`，`RANDOM_STOP=False`）。视觉 encoder 全程冻结（`TRAIN_MDP=False`, `FREEZE_MDP_BACKBONE=True`），只训练 PPO。

跑满 1000 个 update（`NUM_ENVS=128`, `N_STEPS=64`，共 8192000 步）。

## 主要文件

- `best_offline.pt`: 离线 deterministic 评估选出的 best policy，对应 `best_offline_info.txt`（`update=372`）。
- `last.pt`: update 1000 的 policy。
- `config.py`: 训练现场配置快照。
- `offline_val_config.py`: 离线评估配置。
- `log.csv`: 训练日志（1000 行）。
- `offline_val.csv`: 各 checkpoint 离线评估结果。
- `val.csv`: 训练期 val 文件（本次 `VAL_EVERY=0`，只有表头）。
- `traj_best_offline.csv`: best checkpoint 全 env 离线轨迹。
- `traj_offline_env0.csv`: 离线评估 env0 轨迹。
- `traj_train_env0.csv`: 训练 env0 轨迹。
- `deps/mdp_state_priv_probe_grad/last.pt`: 本次使用的 MDP state encoder，来自 `models/vision/mdp_state_priv_probe_grad/last.pt`。
- `deps/mdp_state_priv_probe_grad/log.csv`: 该 encoder 的预训练日志。
- `deps/teacher_ppo/best_val.zip`: 训练时使用的 4 维 goal teacher，来自 `models/rl/teacher_ppo/best_val.zip`（md5 `63c8548599d8935a752c7f3acede31a6`）。

## 结果摘要

训练 rollout success（按 episode 加权）：

| update | success | fail | timeout | std_mean | teacher_coef |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1-100 | 0.4% | 1.5% | 98.2% | 0.399 | 1.0 |
| 251-500 | 45.2% | 0.0% | 54.8% | 0.128 | 1.0 |
| 501-750 | 73.4% | 0.0% | 26.6% | 0.057 | 1.0 |
| 751-1000 | 86.4% | 0.0% | 13.7% | 0.040 | 1.0 |

离线 deterministic 评估 best（`best_offline_info.txt`）：

- `checkpoint = update_000372.pt`
- `success_rate = 0.34375`, `fail_rate = 0.0`, `timeout_rate = 0.65625`
- `mean_return = 2.1878`, `mean_len = 301.48`

失败样本全部是 timeout，`fail_rate` 全程为 0，策略不撞墙不越界。

## 重要限制（读结果前必看）

1. **teacher 全程没有退火。** `TEACHER_LOSS=1.0`，而 `teacher_coef` 只在 `TRAIN_MDP` 分支和 `VAL_EVERY` 块里被修改，本次两者分别是 `False` 和 `0`，所以 `teacher_coef` 从 update 1 到 1000 恒为 1.0。**86.4% 是 teacher MSE 持续作用下的结果，不能说明策略能脱离 teacher 独立工作。**
2. **离线评估不完整。** `offline_val.csv` 只覆盖 update 1..407（407 行），训练末段（751-1000，rollout 86.4%）**从未做过 deterministic 评估**。表格里的 86.4% 只是训练 rollout 数字。
3. **encoder 带特权 probe 梯度。** `mdp_state_priv_probe_grad` 的 `PROBE_W=1.0` 且 `probe_head` 输入未 detach（`src/cv_extractor/mdp_state.py:167`），px/dist 的特权监督会反传进 encoder。该 encoder 预训练末期 `probe_x_err=3.29px`、`probe_d_err=0.165m`。因此这条线不是纯自监督表征，属于对照变体。
4. **encoder 与 `2026_06_27_weekend_students/deps/` 里的同名文件不是同一个。** 本次 encoder md5 `1a177e1d822a3ab84ad5de069fd1f40d`，weekend 那份是 `1e63006c872b8fac253c60ad9cbdff73`（7/7 训练）。不要混用。
5. `models/rl/teacher_ppo/` 现场没有 `config.py`，所以 `deps/teacher_ppo/` 只有 `best_val.zip`，与 `2026_06_25_mdp_student_update421/deps/teacher_ppo/` 的结构不同。

## 未保存内容

- `updates/` 下 1000 个过程 checkpoint（约 356MB）。
- `tb/` tensorboard events。

## 恢复到默认路径

```bash
mkdir -p models/vision/mdp_state_priv_probe_grad models/rl/teacher_ppo models/rl/mdp_student
cp approved_models/2026_07_08_mdp_student_noise/deps/mdp_state_priv_probe_grad/last.pt models/vision/mdp_state_priv_probe_grad/last.pt
cp approved_models/2026_07_08_mdp_student_noise/deps/teacher_ppo/best_val.zip models/rl/teacher_ppo/best_val.zip
cp approved_models/2026_07_08_mdp_student_noise/best_offline.pt models/rl/mdp_student/best_offline.pt
```

训练/评估入口：

```bash
./scripts/run_mdp_student_pipeline.sh
```
