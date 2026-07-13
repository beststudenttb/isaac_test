# 2026-07-10 SPR student (noise env, spr_loss 去中心化)

噪声环境（红色目标球 + 蓝色干扰球，`--noise`）下的 SPR latent student，固定 stop 任务（`END_D=1.5, END_X=0`）。四阶段：

| stage | update | 内容 | 执行的动作 |
| ---: | ---: | --- | --- |
| 1 | 1-100 | 只训 SPR/encoder，不动策略 | 加噪 teacher（`TEACHER_NOISE_STD=0.3`）|
| 2 | 101-350 | 冻结 SPR，PPO + teacher BC（`TEACHER_LOSS=5.0`）| 学生采样动作 |
| 3 | 351-500 | 冻结 SPR，teacher 权重线性退火到 0 | 学生采样动作 |
| 4 | 501-1000 | 冻结 SPR，纯 PPO（`SPR_UPDATE_EVERY=0`）| 学生采样动作 |

本次相对上一版的**唯一改动**：`src/spr_ppo.py` 的 `spr_loss_k` 在 `F.normalize` 之前对 pred/target 沿 batch 维做了中心化，堵死常数偏置捷径；`spr_diagnostics` 同步改为测量中心化后的量。

## 主要文件

- `best_offline.pt`: 离线 deterministic 评估选出的 best policy（`updates/ppo_000410.pt`，逐字节相同）。**这是 `ppo_*` 类型，只含策略，必须配 `stage1_end.pt` 的 SPR 使用。**
- `best_offline_info.txt`: 对应指标。
- `stage1_end.pt`: 阶段1 结束时的完整模型，`ppo_*` checkpoint 的 SPR 来源。
- `teacher_best_val.zip`: 训练时使用的 4 维 goal teacher（md5 `63c8548599d8935a752c7f3acede31a6`）。
- `config.py` / `offline_val_config.py`: 训练与离线评估的配置快照。
- `log.csv`: 训练日志（1000 行）。
- `spr_log.csv`: 阶段1 的表征指标（100 行）。
- `offline_val.csv`: 离线评估结果（本次抽样 `--stride 10`，9 个 checkpoint）。
- `val.csv`: 训练期 val（`VAL_EVERY=0`，只有表头）。
- `traj_best_offline.csv` / `traj_offline_env0.csv` / `traj_train_env0.csv`: 轨迹。

## 结果摘要

### 表征：去中心化成功了

阶段1 结束时的 encoder，在 `data_mdp_teacher_50k_noise` 上做离线线性探针：

| 指标 | 上一版（有常数偏置） | 本次（去中心化） |
| --- | ---: | ---: |
| signal/offset | 0.030 | **1.153** |
| z_std | 0.0011 | **0.331** |
| 线性探针 z→px_x R² | 0.683 | **0.9977** |
| 线性探针 z→dist R² | 0.919 | **0.9946** |
| identity_gap | 0.0075 | **1.33** |
| target_sim | ≈1.0 | 0.88 |

`px_x` 的 R² 0.9977 **高于** `mdp_state_priv_probe_grad` 那个带特权监督的 encoder（0.851）。表征不是瓶颈。

遗留：`eff_rank` 约 18/128（维度坍缩仍在），`spr_loss` 收敛到 9.5e-5，远低于随机基线 1.56e-2。

### 策略：stage2/3 满分，stage4 崩到 0

离线 deterministic 评估（`--num-envs 128`，`STOP_N=3`）：

| checkpoint | stage | teacher_coef | success | timeout | final_de | final_xe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_000110 | 2 | 5.0 | 93.8% | 7.0% | 0.168 | 3.14 |
| ppo_000210 | 2 | 5.0 | **100%** | 0 | 0.140 | 2.99 |
| ppo_000310 | 2 | 5.0 | 95.3% | 4.7% | 0.179 | 4.62 |
| **ppo_000410** | 3 | ~0.5 | **100%** | 0 | 0.126 | 3.57 |
| full_000510 | 4 | 0 | 72.7% | 27.3% | 0.185 | 15.8 |
| full_000610 | 4 | 0 | **0%** | 93.0% | 0.568 | 64.8 |
| full_000710 | 4 | 0 | 0% | 100% | 1.296 | 11.4 |
| full_000810 | 4 | 0 | 0% | 100% | 0.762 | 5.70 |
| full_000910 | 4 | 0 | 0% | 99.2% | 0.518 | 3.41 |

**`ppo_000410` 的确定性成功率是 100%（128 episode，0 fail 0 timeout），`mean_len` 117 步。** 推理时不需要 teacher。

而训练 rollout 的 success 只有 stage2 40.3% / stage3 41.4%——两者的差距是本次实验最重要的发现。

## 崩溃机制（已用日志和轨迹证实）

1. **success 判据是动作空间里的一个硬阈值盒子。** `STOP_EPS=0.05`，三个动作分量必须同时 `|a_i|<0.05`（`src/task_cfg.py:39`）。
2. **均值策略早就学会停了。** `teacher_actions()`（`train/spr_student.py:253-255`）在区内把 teacher 目标**硬置零**，teacher 区内动作 std 仅 `0.029/0.012/0.031`。学生区内动作均值 `(+0.018, +0.021, -0.011)` ≈ 0，执行动作被 σ·ε 主导。

   离线在数据集图像上跑 `ppo_000410` 的确定性均值动作（无噪声），学到的区内控制律**符号完全正确**：`mu_w` 对 px 误差斜率 −0.0094（corr −0.479，偏右就往回转），`mu_x` 对 dist 误差斜率 +0.628（太远就前进）。用滞后一步的输入算训练轨迹也一致：`corr(a_w, px_err[t-1])` 在 SPR stage2/3 是 −0.354 / −0.324，与 MDP 同量级。

   **注意**：`traj_train_env0.csv` 的行是在 `env.step` 之后写的（`train/spr_student.py:601`），`px_x` 是动作执行后的状态。直接算 `corr(a_w, px_err[t])` 会得到 +0.19，那是本步动作对 px 的因果位移（σ=0.30 时 3.03 px，而区内 px 误差 std 仅 5-6 px）把符号压翻的假象，**不是正反馈**。
3. **σ 被 clamp 钉死。** `STD_INIT=0.5 > STD_MAX=0.3`，PPO 的第一个 update（101）σ 立刻被削到 0.3002，此后 σ(a_x)、σ(a_w) 常年读数 0.2999。
4. **σ=0.3 让它每 11 步就被踢出 stop 区。** `dt=0.040s`，`HEAD_SPEED=π/2`，FOV 80°/224px = 160 px/rad → σ(a_w)=0.3 每步造成 3.03 px 抖动，而 `STOP_X_TOL` 只有 ±10 px。实测出区原因中 **78-84% 是像素条件破了**，距离条件只占 8-13%。
5. **每次出区罚 −2.0（`R_STOP_OUT`）。** 于是进入 stop 区在 on-policy 回报上是净亏的：

| stage | 出区次数/ep | 出区罚金 | success | 成功奖 | 实测平均 episode 回报 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 19.6 | −39.3 | 40% | +1.72 | **−16.85** |
| 2 | 10.1 | −20.2 | 31% | +1.34 | **−9.64** |
| 3 | 7.5 | −15.0 | 39% | +1.66 | **−5.80** |

   区内也没有任何正的稠密奖励：`app = seen & prev_seen & (~stop)`（`src/sb3_env.py:209`）把 approach shaping 在区内关掉了，`K_STOP_DA=0`、`R_STOP=0`，每步只有 `-k_time=-0.003`。

6. **PPO 收不到「降低 σ」的梯度，这是本次实验的核心结论。** teacher loss 是 MSE（只作用于均值，对 `log_std` 梯度为 0），`ENT_COEF=0`，所以 σ 只能靠 PPO 的 `Δlogσ_i ∝ Cov(A, ε_i²)` 移动（`ε=(a-μ)/σ`，advantage 已归一化）。该项非零要求回报对当前动作**光滑且凹**。

   而 stop 区内唯一可归因到动作的奖励项是 stop 指示函数 `1[所有 |a_i|<0.05]`（`app` shaping 被 `~stop` 关掉，`R_STOP=0`，`K_STOP_DA=0`）。它给 σ 的收缩力 `g(σ) ∝ P(stop)·(1-E[ε²|stop])`：σ=0.30 时 g=0.0023，σ=0.034 时 g=0.293，**相差 128 倍**。σ 要变小得先停成功过，要停成功 σ 得先小——先有鸡还是先有蛋。

   实测 `Cov(A, ε²)` 代理量：SPR stage2 区内 `ε_w²` 为 **+0.013**、stage3 为 **+0.038**（弱正），区外仅 −0.026 ~ −0.036。合计约等于 0，σ 遂停在 clamp 上不动。

   **注意** `STD_MAX=0.3` 不是 σ 降不下去的原因（下界是 `STD_MIN=0.02`，没挡住），它只决定了起点。原因是没有梯度。

7. **stage2/3 里 BC 是唯一的对手。** `TEACHER_LOSS=5.0` 把均值按在正确位置，压过了「别进 stop 区」的负梯度。`teacher_coef` 退到 0 后该负梯度失去对手，PPO 在约 100 个 update 里把策略推出 stop 区（离线成绩 510 → 72.7%，610 → 0%）。同一窗口（update 501-600）内控制时间混淆：进过 stop 区的 episode 平均回报 −3.83，从不进区的 −1.08。**PPO 搬到了更优的一边，它是对的**——在 σ(a_w)=0.3 的前提下，躲开球确实是该 reward 下的最优行为。

## 为什么 MDP 那条线的 σ 能降下来

同 env、同 reward、同 `R_STOP_OUT=-2.0`，teacher loss 也同样是 MSE（`src/ppo.py:170`）。差别**不在 clamp**，在于**它在进入 stop 区之前有 400 个 update 待在区外**：

| | 区内步占比 | σ | `Cov(A, ε_x²)` 代理 |
| --- | ---: | ---: | ---: |
| MDP 1-100 | 1.7% | 0.49→0.31 | **−0.191** |
| MDP 101-250 | 0.0% | 0.245 | +0.020 |
| MDP 401-500 | 4.4% | 0.118 | −0.048 |
| SPR stage2 | **56.0%** | 0.298 | −0.001 |
| SPR stage3 | **64.6%** | 0.295 | 区内 `ε_w²` +0.038 |

区外的接近阶段，回报对动作是凹的（冲过头或落后都变差），`Cov(A, ε²)` 是干净的负值。MDP 靠这 400 个 update 把 σ 从 0.49 压到 0.16；等它真正进区时 `P(stop)=0.035`，收缩力已达满值的 11%，正反馈自行启动。

SPR 从 PPO 的第一个 update 起就有 56% 的步在区内，从未有过这个阶段。直接原因是 `TEACHER_LOSS=5.0`（MDP 为 1.0），5 倍强的 BC 立刻把它摁进 stop 区。

雪上加霜的是 SPR 的 advantage 噪声大 6 倍：每步 reward 方差 0.156 vs MDP 的 0.025，每 episode 在区界上来回穿 22.7 次（MDP 5.1 次），每次往回报里砸一个 ±1/−2，这些量归因不到当前动作。`NORMALIZE_ADVANTAGE=True` 后 `corr` 就是信噪比，本就微弱的区外负信号被稀释掉。

这是个闭环：σ 大 → 停不下来 → 长期滞留区内并反复穿越区界 → σ 的梯度既消失（区内无稠密奖励）又被淹没（穿越噪声）→ σ 保持大。

**方法学说明**：`μ` 和 `V` 未记录，上表的 `Cov(A, ε²)` 是用 `(px误差, dist)` 的二次回归近似 `μ̂`/`V̂`、用经验折扣回报代替 GAE 得到的**代理量**。量级可信，精确值不可信。

## 已确认排除的原因

- **不是表征坍缩。** 全局线性探针 R² 0.9977 / 0.9946。stop 区**邻域内**单独拟合也一样好：`z→px_x` MAE 0.16 px、`z→dist` MAE 0.0040 m，而区宽是 ±10 px / ±0.2 m，分辨率比区宽细约 60 倍。
- **不是区内控制器学坏了。** 确定性均值动作是正确的回中律（见崩溃机制第 2 条）。
- **不是「比 MDP 更容易被踢出区」。** 排除 episode reset 后，SPR stage3 的出区危险率 0.0387/步、区内平均停留 25.9 步，比 MDP 401-500（0.0526/步、19.0 步）还稳。差别只在停不下来：σ=0.295 时 P(stop)/步=0.0024，23 步内中奖 5.5%（实测 12/269 段 = 4.5%）；MDP σ=0.034 时 P(stop)/步=0.638，7.3 步内近乎必中（实测 50/54 段 = 93%）。
- **不是 encoder 被 PPO 破坏。** `full_*.pt` 与 `stage1_end.pt` 的 `spr.*` 权重逐比特相同（最大差 0.0），只有 BatchNorm 的 `running_mean/var` 漂了（最大 0.08），因为 `train/spr_student.py:563` 的 `model.train()` 让 BN 统计量在冻结 encoder 上继续更新。
- **不是 KL 爆炸。** KL 从 0.31 涨到 13.8 发生在 update 701 之后，那时 `zone_frac` 早已是 0.001、`v_loss` 已掉到 0.004。**KL 爆炸是结果而非原因。**

## 重要限制

1. **本次 `offline_val.csv` 只有 9 行**（`--stride 10` 抽样），不是全部 90 个 checkpoint。
2. **离线 val 的 `STOP_N=3`，训练是 `STOP_N=1`。** 两者判据不同；离线用确定性均值动作，σ 不参与，所以 `STOP_N` 的差别不影响本次结论。
3. **`val/spr_student_cfg.py` 的 `OUT_DIR` 原本是 `./models/rl/spr_student_q`，与 train 的 `./models/rl/spr_student` 不一致**，导致本次 pipeline 的 val 段直接 `FileNotFoundError` 退出。已改为一致；`config.py` 快照是改之前写的。
4. `scripts/run_spr_student_pipeline.sh` 的 `VAL_ENVS` 默认 128，而 `val/spr_student_cfg.py:10` 注明「64 会在 val 启动时内存崩溃」。本次 val 是手动用 `--num-envs 128` 跑通的，但 sh 默认路径未验证。
5. 未保存 `last.pt`（update 1000，已崩溃的策略）、`stage2_end.pt`、`stage3_end.pt`。

## 未保存内容

- `updates/` 下 100 个过程 checkpoint（约 4.5GB）。
- `tb/` tensorboard events。
- `last.pt` / `stage2_end.pt` / `stage3_end.pt`（各 85MB）。

## 恢复到默认路径

```bash
mkdir -p models/rl/teacher_ppo models/rl/spr_student
cp approved_models/2026_07_10_spr_student_center/teacher_best_val.zip models/rl/teacher_ppo/best_val.zip
cp approved_models/2026_07_10_spr_student_center/stage1_end.pt models/rl/spr_student/stage1_end.pt
cp approved_models/2026_07_10_spr_student_center/best_offline.pt models/rl/spr_student/best_offline.pt
```

`best_offline.pt` 只含策略，加载时需要 `stage1_end.pt` 提供 SPR/encoder（见 `val/spr_student.py` 的 `SPR_CKPT` 规则）。

训练/评估入口：

```bash
./scripts/run_spr_student_pipeline.sh
```
