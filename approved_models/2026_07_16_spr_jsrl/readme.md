# 2026_07_16 SPR-PPO + JSRL 反向课程（反面结果）

**结论:反向课程让 on-policy PPO 从 0% 爬到 ~60%,但撞天花板、救不了 σ 病。同一个冻结 z,off-policy(arm A)满分 100%,PPO+课程只到 62.5%。**

这是"课程解决接近问题、但解决不了精确对准问题"的干净反面证据,用来对照 `2026_07_14_drqv2_pixels` 的 arm A（冻结 SPR z + DrQ-v2）。

## 是什么

teacher 带路(反向课程/JSRL):teacher 先把机器人开到"快停车"的状态,student 从那里接手、只学最后一段;学会了再把接棒点(步数上限 h)往前挪。**冻结的 stage1 SPR encoder(和 arm A 逐字节相同,encoder 权重 hash `84fd8456...`)+ PPO,只收集 student 步(approach B,不 mask),base spr_ppo_update 原样训练。**

- 交棒条件不用特权信息,只看 teacher 自己的动作幅度:`handover = (episode_step >= h) or (max|a_teacher| < STOP_EPS)`。
- h 由 student 成功率驱动(EMA 平滑 + 双阈值 + 回火 + cooldown):success EMA > 0.6 降 h(接棒点前移),< 0.3 回火。
- `NoiseStudentEnv`,128 env,1000 update,`STD_INIT=0.2`、`STD_MAX=0.3`(log_std 可训练)。

## 成绩

`best_offline.pt` = `full_000600.pt`（update 600,h=20），128 env 离线 deterministic μ：

| 指标 | 值 |
| --- | ---: |
| success_rate | **0.625** |
| fail_rate | 0.031 |
| timeout_rate | 0.344 |
| final_de_mean | 1.092 m |
| final_xe_mean | 20.6 px |
| mean_return | 6.66 |

离线曲线（确定性 μ，`offline_val.csv`，100 个 checkpoint）：

| update | h | success | de | xe |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 150 | 0.000 | 2.34m | 53px |
| 490 | 63 | 0.125 | 2.61m | 35px |
| **600** | **20** | **0.625** ← 峰值 | 1.09m | 20.6px |
| 730 | 11 | 0.469 | 1.01m | 29px |
| 970 | **0** | 0.406 | 1.83m | 26px |

**峰值在 h=20（teacher 还帮一点）时达到；h→0 逼 student 独立做全程后反而跌回 0.41 —— 课程降太快，policy 跟不上、退化。**

## 和 arm A 的判决对照（同一个冻结 z）

| 算法 | 离线 success | de | xe |
| --- | ---: | ---: | ---: |
| `2026_07_14_drqv2_pixels` 的 arm A（DrQ-v2, off-policy） | **1.000** | 0.022m | 0.9px |
| 本次（PPO + 反向课程） | **0.625** | 1.09m | 20.6px |

encoder 权重 hash 两边都是 `84fd8456...`，**逐字节同一个 z，差别纯在算法**。

## 根因：σ 病没解决，只被课程绕过一半

训练日志（`log.csv`）：
- **`raw_std_max` 整整 880 个 update 焊死在 0.300 clamp 上**（`a_w` 转头维度），`raw_std_min` 掉到 0.020（`a_x`/`a_y` 塌到下限）。三维永久分裂。
- 反向课程解决了"走到停车区"（success 0 → 60%，远好于纯 SPR-PPO 的 0%），**但解决不了"精确保持对准"**：`a_w` 的 σ 顶死 0.3，连带 μ_w 也没被磨精确 —— 确定性 μ 都停不准（de=1m、xe=20px）。
- 机制：交棒时球已对准（~10px），但 student 的 `a_w` 带 σ=0.3 噪声一转就把球晃出视野 → observation 退化成 `dist=0/px=-1` 哨兵（球不可见）→ 找不回 → 超时。noise env 的蓝球干扰放大了这一点。
- arm A 的 replay 能给出干净的 advantage，把三维都收敛（de 0.022m）；PPO 在这个 reward 下转头维度的 σ 就是收不下来。

**强化 arm A 的核心 claim：同一个 z，off-policy 的 replay 是解开精确对准的唯一钥匙；课程堆在 PPO 上顶多到 62%。**

## 文件

| 文件 | 说明 |
| --- | --- |
| `best_offline.pt` | `full_000600.pt`，**自包含**（含冻结 encoder + 策略），43MB。`SPRStateNet(**spr_cfg)` + `SPRActorCritic` + `load_state_dict(ck["model"])` 加载 |
| `config.py` / `offline_val_config.py` | 训练 / 评估配置快照 |
| `log.csv` | 1000 个 update（含 `jsrl_h` / `student_frac` / `jsrl_success_*` / `raw_std_*` / `zone_frac`） |
| `offline_val.csv` | 100 个 checkpoint 的离线成绩（列含 `jsrl_h`） |
| `best_offline_info.txt` | 最佳 checkpoint 指标 |
| `traj_best_offline.csv` / `traj_train_env0.csv` | 最佳离线轨迹 / 训练 env0 轨迹（`student` 列标记该步是否 student 执行） |

## 复现 / 代码

- `src/spr_jsrl.py`（`JSRLCurriculum` + `JSRLRollout(SPRRollout)`，继承、只收集 student 步）
- `train/spr_jsrl.py` + `train/spr_jsrl_cfg.py`、`val/spr_jsrl.py` + `val/spr_jsrl_cfg.py`、`scripts/run_spr_jsrl_pipeline.sh`
- 冻结 encoder 来源 `models/rl/_encoders/stage1_sigma_ablation.pt`（现场副本，非入库）
- `src/spr_ppo.py` 保持 pristine —— approach B 只收 student 步,base update 不需要改。

## 待办（下一版调参再跑对比）

- 降档太快 → h 峰值能力在 h=20，之后退化。可放缓 `JSRL_H_DOWN`（0.75→0.9）或换成基于交棒时"是否在区内"的状态旋钮。
- `a_w` σ 顶死 0.3 是天花板根因 → 可给 `a_w` 单独更小的 `STD_MAX`（per-dim clamp），或缩 `STOP_X_TOL` 让 teacher 交棒时对得更死。
