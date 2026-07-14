# 2026_07_14 DrQ-v2 像素 student（off-policy）

**项目里第一个不用特权信息、不用 teacher、不用 BC，从原始像素从零学出来的策略。**
离线 deterministic success **97.7%**。

## 成绩

`best_offline.pt` = `drqv2_01000000.pt`（frame 1,000,000），128 env × 1 episode 离线评估：

| 指标 | 值 |
| --- | ---: |
| success_rate | **0.9766** |
| fail_rate | 0.0000 |
| timeout_rate | 0.0234 |
| final_de_mean | 0.044 m |
| final_xe_mean | 3.13 px |
| mean_len | 110 步 |
| mean_return | 8.98 |

离线曲线（`offline_val.csv`，44 个 checkpoint）：frame 850k 之前**全部为 0**，925k 突然到 0.961，1.0M 到 0.977，之后维持。

## 配置

- `MDPStudentEnv`（**干净 env，不是 noise env**），`STOP_N=1`，`EPISODE_S=15`，固定任务（end_d=1.5, end_x=0）。
- 32 env（命令行覆盖了 cfg 的 64）、batch 128、`UPDATES_PER_TICK=16`、n-step 3、γ=0.99、τ=0.01。
- 输入 224×224×3 单帧 + goal(2)；4 层 Conv32 encoder（repr_dim 352,800）+ trunk(50) + hidden 1024。
- σ 线性调度 0.30 → 0.02 / 1.5M frame，`stddev_clip=0.3`，`aug_pad=4`。
- 跑到 frame 1,119,200 被 Ctrl+C 中断（Kit 吞掉 SIGINT，不走 Python 的 finally，所以没有 `last.pt`）；此时已收敛并稳定在 ~99% 训练 rollout。

## 结论：成功是 σ 调度换来的，不是表征换来的

训练日志（`log.csv`）里的分界点极其干净：

| frame | σ | rollout success |
| ---: | ---: | ---: |
| 0 – 880k | 0.30 → 0.136 | **0.000** |
| 894k | 0.134 | 0.06 ← 第一次撞到 |
| 933k | 0.126 | 0.97 |
| 952k – 1.12M | 0.123 → 0.10 | 0.98 – 1.00 |

success 要求三维动作同时 `|a_i| < STOP_EPS = 0.05`，发现概率 `P ≈ [2Φ(0.05/σ) − 1]³`：

- σ = 0.30 → **2.3e-3 / 步**
- σ = 0.134 → **2.5e-2 / 步**（约 11 倍）

策略早就学会了"走到位"（env0 轨迹里 `|dist − end_d|` 从 1.78 m 降到 1.14 m），它缺的只是**发出一个足够小的动作的能力**——而这个能力被 σ 卡死。σ 一旦被调度到 0.13 附近，stop 事件变得可采，replay 把这几次稀有成功反复回放，Q1 在 4 万帧内从 −0.23 爆到 4.33，actor 立刻锁上。

对照 `2026_07_10_mdp_student_anneal`：那条 PPO 线的 σ **自由衰减** 0.42 → 0.033，success 在 **σ ≈ 0.12** 起飞 → 96%。**两条成功的线起飞的 σ 几乎是同一个数**，而两条失败的 SPR PPO 线的 σ 分别被钉死在 0.299 和 0.100（都顶着 clamp）。

所以这条线的价值不在于表征，而在于**把 σ 从"被学的参数"变成"外生的调度"**。DDPG 的 actor 走 `∇_a Q`，不需要 `log π` 的梯度，σ 才可以外生化。

## 文件

| 文件 | 说明 |
| --- | --- |
| `best_offline.pt` | **eval-only**：`critic` / `critic_target` 已剥离（原 223 MB → 71.6 MB，GitHub 单文件上限 100 MB）。只含 `encoder` + `actor`，可推理、**不能续训**。确定性动作与完整版逐位一致。 |
| `config.py` | 训练配置快照 |
| `offline_val_config.py` | 离线评估配置快照 |
| `log.csv` | 训练日志（1399 个窗口，每 25 tick 一行） |
| `offline_val.csv` | 44 个 checkpoint 的离线成绩 |
| `best_offline_info.txt` | 最佳 checkpoint 的指标 |
| `traj_best_offline.csv` | 最佳 checkpoint 的离线轨迹 |
| `traj_train_env0.csv` | 训练时 env0 的逐步轨迹 |

## 复现

checkpoint 是自包含的（encoder 在里面，不像 SPR 的 `ppo_*.pt` 需要配 `stage1_end.pt`）。把 `best_offline.pt` 放进 `models/rl/drqv2/updates/` 并改名成 `drqv2_01000000.pt`，然后：

```bash
env -u DISPLAY ./IsaacLab/isaaclab.sh -p val/drqv2.py --num-envs 128 --num-episodes 1
```

注意**不要加 `--noise`**——这次跑的是干净 env，加了 `--noise` 会去找 `models/rl/drqv2_noise/`。

## 注意事项

- **干净 env，不是 noise env**，所以和 `2026_07_10_mdp_student_anneal` / `2026_07_13_spr_student_bc1` 这些 noise 线**不能直接横比**。noise 版是单独一条线（`models/rl/drqv2_noise/`）。
- reward 常数用的是 06-30（commit `ac1604b`）之后的版本：`R_STOP = 0.0`（stop 区内的稠密奖励是关闭的），`R_STOP_IN=+1` / `R_STOP_OUT=−2`，timeout 不罚。和 MDP anneal 那次一致。
- `aug_pad=4` 配 224 分辨率只有 1.8% 的位移（DrQ-v2 原版是 4/84 = 4.8%），这个正则化基本没在起作用。下次要么把 pad 提到 ~11，要么降分辨率。
