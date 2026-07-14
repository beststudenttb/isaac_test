# 2026_07_14 SPR student，TEACHER_LOSS=5 / STD_MAX=0.1（反面结果）

**这是一个失败实验，保留是因为它的日志是"σ 才是分水岭"这个结论的核心证据。**

问题：把 BC 权重加到 5、把 σ 上限压到 0.1，能不能先用 BC 蒸馏出策略，再交接给 PPO？
答案：**BC 段能到 100%，但 teacher 一撤干净就断崖式塌到 0%。**

## 成绩

`best_offline.pt` = `ppo_000460.pt`，128 env 离线 deterministic：

| 指标 | 值 |
| --- | ---: |
| success_rate | **1.0000** |
| fail_rate | 0.0000 |
| timeout_rate | 0.0000 |
| final_de_mean | 0.041 m |
| final_xe_mean | 1.14 px |
| mean_len | 119 步 |

**但 checkpoint 自带的元数据显示 update 460 时 `teacher_coef = 4.775`（满权重是 5.0）** —— 这个 100% 是 BC 给的，不是 PPO 学的。同一个 checkpoint 里 `log_std` 三维都是 `-2.3026`，即 σ 恰好 **0.1000** = `STD_MAX` 的 clamp 值。

teacher 退火过程中的离线成绩：

| update | teacher_coef | 离线 success | timeout |
| ---: | ---: | ---: | ---: |
| 460 | 4.78 | **1.000** | 0.000 |
| 530 | 2.9 | 0.992 | 0.008 |
| 600 | 0.65 | 0.117 | 0.883 |
| 670 | 0 | 0.008 | 0.992 |
| 740 – 950 | 0 | **0.000** | 1.000 |

`final_de` 在 stage4 漂到 2.19 m，`final_xe` 一度到 67 px。

## 配置

- `NoiseStudentEnv`，128 env × 64 step，1000 update。
- 阶段：100（SPR-only，teacher 加噪带跑）/ 350（全权重 BC，`TEACHER_LOSS=5`）/ 200（线性退火到 0）/ 350（纯 PPO）。
- `STD_INIT=0.5`、`STD_MIN=0.02`、`STD_MAX=0.1`、`TEACHER_LOSS_TYPE="mse"`。
- reward 常数是 06-30（`ac1604b`）之后的版本：`R_STOP=0.0`、`R_STOP_IN=+1` / `R_STOP_OUT=-2`、timeout 不罚。

## 结论

### 1. σ 从头到尾顶死 clamp，连 teacher 撤光之后也是

`raw_std_max`（clamp 之前的原始 σ）从 update 101 起就一直是 **0.100**，即 PPO 的梯度**始终**想把 σ 推到 0.1 以上——**包括 `teacher_coef=0` 的整个 stage4**。

对照 `2026_07_13_spr_student_bc1`（同样的线，`STD_MAX=0.3`）：那次 stage4 的 σ 自由落到 **0.168**。两次互相印证：**PPO 在这个 reward 下的 σ 均衡点大约是 0.14–0.17**，而它需要降到 ~0.13 以下才可能撞到成功。

这也说明当初那个早期止损信号是对的：**stage2 的 `raw_std_max` 一旦顶到 `STD_MAX`，这次跑就已经废了**，可以立刻杀掉（本可省 10 小时）。

### 2. 这次的死法是"停得下来，但走不到了"——和之前几次不同

σ=0.1 时 P(stop)/步 = `[2Φ(0.05/0.1)-1]³` = **5.6%**，完全够用。日志里 `stop_frac` 也确实**不降反升**（0.007 → 0.026）：策略在发 stop 动作。

崩的是 **`zone_frac`：0.29 → 0.000**，`a_mag_mean` 0.65 → 0.19，`v_loss` 塌到 0.006。**策略学会了停，但不再走向目标。**

### 3. 为什么"什么都不做"是当前 reward 下的理性最优

- `R_STOP = 0.0` —— 停在 stop 区里**每一步的收益是 0**，唯一的钱在终止时的 `R_SUCCESS * stop_q`。
- `R_STOP_IN = +1` / `R_STOP_OUT = -2` —— **进出一个来回净 −1**。σ=0.1 的噪声下策略会被反复踢进踢出，每个来回交 1 分的税。
- timeout **不罚**（`r_fail` 只加在 `fail` 上）。

于是"别进 stop 区、站着等超时、拿 0 分"是一个真实的局部最优。**PPO 没学崩，它找到了这个吸引子。**

### 4. 但 reward 不是唯一解释——MDP 用同一套 reward 拿了 96%

`2026_07_10_mdp_student_anneal` 是 07-10 跑的，`R_STOP` 已经是 0.0，和这次完全同一套 reward。它的 σ **自由衰减** 0.42 → 0.033，success 在 **σ ≈ 0.12** 起飞 → 96%。

`2026_07_14_drqv2_pixels`（σ 外生调度）在 **σ = 0.134** 起飞 → 97.7%。

**两条成功的线起飞的 σ 几乎是同一个数；两条失败的 SPR PPO 线的 σ 分别被钉死在 0.299 和 0.100。**

### 5. 真正的病灶：PPO 在 SPR z 上"拆"而不是"建"

同样的 reward、σ 都在 0.1 量级：

- **MDP 的 PPO 从零构建出了一个解**（BC 段离线只有 ~0%，是 PPO 自己爬起来的）。
- **SPR 的 PPO 拿到 BC 喂的 100% 的解，然后把它拆了。**

这跟"z 里有没有信息"无关（变量分解已证明 SPR z 的任务方差占比 75.5%，和能跑到 96% 的 MDP `img_feature` 的 75.0% 持平）。

→ 下一步的判决实验：**SPR + PPO，不用 BC，`log_std` 冻结、σ 按 0.3 → 0.02 外生调度**（`STD_SCHEDULE=True`）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `best_offline.pt` | `ppo_000460.pt`，**只含策略**，必须配同目录 `stage1_end.pt` 使用 |
| `stage1_end.pt` | 本次 stage1 训出的 SPR/encoder，md5 `696811c7...` |
| `teacher_best_val.zip` | 4 维 goal teacher，md5 `63c8548...`（与 `2026_07_13_spr_student_bc1` 同一份） |
| `config.py` / `offline_val_config.py` | 训练 / 评估配置快照 |
| `log.csv` | 1000 个 update 的训练日志（含 `raw_std_*`、`zone_frac`、`stop_frac`、`teacher_coef`） |
| `spr_log.csv` | stage1 的 SPR 诊断 |
| `offline_val.csv` | 90 个 checkpoint 的离线成绩 |
| `traj_best_offline.csv` / `traj_offline_env0.csv` / `traj_train_env0.csv` | 轨迹 |

## 注意事项

- **`stage1_end.pt` 又是一份新的 encoder**：md5 `696811c7...`，与 `2026_07_13_spr_student_bc1`（`ef6d8647...`）和 `2026_07_10_spr_student_center`（`e9464932...`）**都不是同一份**。三次 stage1 各自重训过，跨实验对比时 encoder 不是单变量。
- 和 `2026_07_13_spr_student_bc1` 的唯一区别是 `TEACHER_LOSS`（1.0 → 5.0）和 `STD_MAX`（0.3 → 0.1）；stage 划分也不同（那次是 100/350/200/350，这次一样）。
