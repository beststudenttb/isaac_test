# 2026-07-13 SPR student (TEACHER_LOSS=1，反证实验)

**这是一次失败的实验，保留它是因为它证伪了一个假设，不是因为模型有用。**

噪声环境（`--noise`）下的 SPR 视觉 student，固定 stop 任务（`END_D=1.5, END_X=0`）。四阶段：
stage1 1-100（只训 SPR/encoder，加噪 teacher 带跑）、stage2 101-450（冻结 SPR，PPO + BC 满权重）、
stage3 451-650（BC 线性退火到 0）、stage4 651-1000（纯 PPO）。

相对 `2026_07_10_spr_student_center` 的改动：

| | 07_10 | 本次 |
| --- | ---: | ---: |
| `TEACHER_LOSS` | 5.0 | **1.0**（与 MDP student 对齐） |
| `STAGE2_UPDATES` | 250 | 350 |
| `STAGE3_UPDATES` | 150 | 200 |

改动动机（**事后证明是错的**）：当时认为 `TEACHER_LOSS=5.0` 把学生过早压进 stop 区，
使它长期待在「reward 沙漠」里（区内 `k_stop_da=0`、`r_stop=0`、approach shaping 关闭），
导致 `Cov(A, ε²)≈0`、σ 拿不到收缩梯度。预测：BC 减弱 → 学生更多待在区外 → σ 能从 0.30 降下来。

## 主要文件

- `best_offline.pt`: 离线 deterministic 最好的 checkpoint（`updates/ppo_000160.pt`，逐字节相同）。
  **是 `ppo_*` 类型，只含策略**，必须配同目录 `stage1_end.pt` 的 SPR/encoder 使用。
- `stage1_end.pt`: 本次 stage1 结束时的 SPR/encoder（md5 `ef6d8647b4e904dacccebceabaed74b4`）。
- `teacher_best_val.zip`: 4 维 goal teacher（md5 `63c8548599d8935a752c7f3acede31a6`，与 07_10 同一份）。
- `config.py` / `offline_val_config.py`: 训练与离线评估配置快照。
- `log.csv`（1000 行）、`spr_log.csv`（100 行）、`offline_val.csv`（90 个 checkpoint）、`val.csv`（`VAL_EVERY=0`，只有表头）。
- `traj_best_offline.csv` / `traj_offline_env0.csv` / `traj_train_env0.csv`。

## 结果：全面退步

| | 07_10 (BC=5) | 本次 (BC=1) |
| --- | ---: | ---: |
| 离线 deterministic 最好 | **100%**（`ppo_000410`） | **11.7%**（`ppo_000160`） |
| update 1000 | 0% | 0.8% |
| stage2 rollout success | 0.40~0.47 | 0.02~0.08 |
| stage2 `zone_frac` | 0.52~0.58 | **0.02~0.03** |

`best_offline_info.txt`：`ppo_000160`，success 11.7%，fail 0%，timeout 88.3%，`mean_len` 353.6。

## 结论 1：BC 权重是唯一让学生愿意进 stop 区的力量

`zone_frac` 从 0.55 掉到 0.02（27 倍）。σ=0.3 时进出 stop 区的期望收益是负的
（`R_STOP_IN=+1`，`R_STOP_OUT=-2`，区内每步只有 `-k_time`），PPO 自己的账本说「别进去」。
BC=5 是在强行压制这个梯度；BC 一减弱，PPO 立刻选择永不进入，均值策略也就从没学会停。

## 结论 2：证伪「区内 reward 沙漠导致 σ 不动」

**两次跑 `zone_frac` 差 27 倍，`raw_std_mean` 却都钉在 0.299。**

| BC 满权重段 | `zone_frac` | `raw_std_mean` |
| --- | ---: | ---: |
| 07_10 (BC=5) | 0.55 | 0.2991 |
| 本次 (BC=1) | 0.02 | 0.2993 |

一个 98% 时间待在区外的策略，σ 照样一步不动 350 个 update。所以「σ 不收缩」不可能由
「区内没有 action-dependent reward」解释——该假设作废。

## 结论 3：真正的机制——BC 在把 σ 往上顶，`STD_MAX` 在挡

`src/spr_ppo.py:422` 记录的 `raw_std_*` 是 **clamp 之前**的值。BC 满权重期间：

- 本次：268/370 个 update 的 `raw_std_max` > 0.3（被 `STD_MAX` 拽回）
- 07_10：321/374 个 update 同样越界

即 **BC 在场时 PPO 的 log_std 梯度是正的**——它想要 σ > 0.3，是 `STD_MAX=0.3` 在挡着。
BC 一退，梯度立刻翻负，两次跑的 σ 曲线几乎重合：

| `teacher_coef` | 1.0 | 0.88 | 0.63 | 0.38 | 0.13 | → 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_std_mean`（本次） | 0.2993 | 0.2991 | 0.2971 | 0.2889 | 0.2809 | 0.25 → 0.19 → 0.14 |

解释：MSE BC 把均值 μ 钉在 teacher 上，而那不是 PPO 认为的最优点。PPO 想移动 μ 又移不动，
唯一能提高「采到高优势动作」概率的手段就是放大 σ。这跟 stop 区内外无关，跟表征质量无关。

## 结论 4：stage4 的 σ 下降不是自举，是发散

σ 确实降到 0.139，但同期 `entropy` → −2.26、`kl` 0.18 → **5.29**、`clip_frac` 0.74，
离线 `final_de` 从 0.2 涨到 0.5~1.6 m（容差 0.2 m），离线 `mean_return` 从 0.50 掉到 0.20。
策略塌在「不接近、不停止、耗到 timeout」的退化行为上。

## 与 MDP 线的等权重对照（重要）

`2026_07_10_mdp_student_anneal` 的 `TEACHER_LOSS` 也是 1.0。它的 BC-only 段（update 1-250）
离线 deterministic 只有 0.1%（最好 4.7%）、`zone_frac` 1.7%——**和本次几乎一致**。
但 MDP 在 PPO 接手后涨到 96.1%，本次 SPR 塌到 0%。

所以在 BC 权重、teacher、env、退火形状全部对齐的条件下，PPO 在 MDP latent 上成功、在 SPR latent 上失败。
这是目前最干净的一组对照，也是待解释的核心问题。

## 已知的对比污染

- 两次 SPR 跑的 `stage1_end.pt` **不是同一份**（本次 `ef6d8647...`，07_10 `e9464932...`）。
  stage1 质量相当但非逐字节相同：`spr_loss` 都是 0.0001，`pred_cos` 0.995/0.995，
  `eff_rank` 18.05/17.03，`identity_gap` 0.92/1.35。所以 BC=5 vs BC=1 严格说不是单变量对照，
  encoder 和 stage 长度也变了。但 σ 钉在 0.299 这个现象在两份 encoder 上完全一致，
  结论 2/3 不受影响。

## 未保存内容

- `updates/` 下 100 个过程 checkpoint。
- `tb/` tensorboard events。

## 恢复到默认路径

```bash
mkdir -p models/rl/teacher_ppo models/rl/spr_student
cp approved_models/2026_07_13_spr_student_bc1/teacher_best_val.zip models/rl/teacher_ppo/best_val.zip
cp approved_models/2026_07_13_spr_student_bc1/stage1_end.pt        models/rl/spr_student/stage1_end.pt
cp approved_models/2026_07_13_spr_student_bc1/best_offline.pt      models/rl/spr_student/best_offline.pt
```

训练/评估入口：

```bash
./scripts/run_spr_student_pipeline.sh
```
