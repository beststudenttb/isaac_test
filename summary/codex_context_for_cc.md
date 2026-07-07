# Codex Context For CC

本文档用于把 Codex 当前上下文交给 CC。重点是记录用户在这段对话里反复表达过的设计意图、判断和疑问。后半部分附上当前技术状态，便于对照代码审核。

## 用户观点对话摘录

### 1. 项目目标不是单纯刷 docking 成功率

用户：

> 我不是只想把现在这个任务做到最好。我想要的是视觉能学到 state，后面可以扩展到更复杂的任务。

Codex理解：

- 当前 docking 只是测试场景，不是最终目标。
- 用户更关心视觉 latent 能否作为通用 state，而不是 px/dist 回归精度。
- 如果任务太简单，人工 px/dist 表征天然占优，自监督表征很难体现价值。

用户：

> 自主表征很难打过人工表征，所以我需要一个复杂一点的任务。至少是人工不好给表征的任务。

Codex理解：

- 后续可以考虑机械臂抓取、取回、多阶段任务、遮挡或目标状态变化。
- 当前简单 stop 区任务可能不足以证明 MDP/SPR latent 的价值。

### 2. stop 区域的设计动机

用户：

> 我给 stop 区域不是为了很高精度地停在一个点。后续还有抓取、带回之类任务，所以停止区域只要够用。

Codex理解：

- stop 区域是为了给后续任务一个可接受的预操作状态。
- 不需要把 docking 本身做成极高精度控制。
- reward 不应过度鼓励在 stop 区内微调到极限精度，从而导致公转或停停动动。

用户：

> 如果进入 stop 区就应该停，停得越好 reward 可以高一点，但不能变成一直动着刷分。

Codex理解：

- success 应是主要停止信号。
- `r_stop * stop_q` 这种 dense stop reward 需要谨慎。
- stop 内持续 reward 可能让策略认为“动起来找更高 q”比“停住”更好。

### 3. 关于连续 stop、一次 stop 和延长 rollout

用户：

> 连续 stop 是为了给准确停止信号。如果一次 stop 就能认为成功，可以试试，这样 stop 信号不那么稀疏。

Codex理解：

- 训练阶段可以用 stop 一次即 success 降低稀疏性。
- 评估阶段仍可用连续 stop 判断稳定性。

用户：

> 如果在 stop 区随机出过一次 stop 动作，能不能后面强制 000 补完成功？000 不改变 state。

Codex理解：

- 用户想利用真实到达 stop 区且输出过 stop 的状态，补充更密集的停下学习信号。
- 关键争议是强制动作是否能进入 PPO actor 训练；如果进入 PPO，需要 logprob 与动作一致。
- 更干净的替代是把 stop 区内状态做 BC 监督，而不是伪造 PPO rollout。

### 4. 关于 teacher 的角色

用户：

> 只用 teacher 不就变成示教了吗？

Codex理解：

- 用户不希望主线退化成 behavior cloning。
- teacher 是辅助、引导、约束，不应替代 PPO 的 reward 主信号。
- 主线应保留 PPO：`PPO loss + teacher MSE + 表征 loss`。

用户：

> teacher 给的是动作，不是概率。我的 actor 是概率分布，teacher 怎么给 prob 信号？

Codex理解：

- teacher deterministic action 只能天然给动作点，不给完整分布。
- MSE 是动作均值监督。
- NLL 是人为假设 teacher action 应在 student 分布下高概率，会影响 std。
- 如果只关注模仿动作值，MSE 更符合用户当前理解；如果要压 std，才考虑 NLL 或单独 std 约束。

用户：

> teacher loss 怎么传进 actor？它怎么知道往哪边移动？

Codex理解：

- MSE 梯度直接把 actor mean 拉向 teacher action。
- 它不是策略梯度，而是 supervised gradient。
- PPO 策略梯度仍来自 reward。

### 5. 关于 std 发散

用户：

> std 是怎么变大的？去看看昨晚那次 spr student 的原因。

Codex检查后的结论：

- 昨晚 `models/rl/spr_student` 里 `spr_loss` 全程为 0，SPR 实际没更新。
- 旧逻辑要求 `std_mean < 0.3` 才开启 SPR，但 std 从 0.5 附近开始后一路增大。
- teacher MSE 下降，说明 mean 在学 teacher。
- std 从约 0.5 增大到约 4.8，env0 轨迹动作饱和率持续上升。
- 因此这次不是 SPR 表征漂移造成 std 发散，而是 PPO 在稀疏成功信号、action clamp、采样动作噪声下推大了 `log_std`。

用户：

> 那就 teacher 改回 MSE，训练初开启 SPR。PPO 信号要保留。

Codex当前改动：

- `train/spr_student.py` 保留 PPO 主信号。
- teacher loss 仍是 MSE。
- 新增 `SPR_START_ACTIVE=True`，`SPR_UPDATE_EVERY=1`，让 SPR 从训练开始更新。

### 6. 关于 MDP / SPR 表征

用户：

> 我想要的是 img 到 z 的变化不要突变，policy 适应不了。

Codex理解：

- 用户担心在线表征训练导致 state 漂移，actor/critic 追不上。
- 表征更新频率、冻结/解冻策略、target network 慢更新都需要考虑。

用户：

> z,a 到 z' 这个还是比较好的，但是这个 z 可能不是好的表征。

Codex理解：

- 动力学可预测不等于 policy 可用。
- SPR/MDP 可能学到“可预测”的 latent，但不一定包含 stop 控制需要的信息。
- 需要看 PPO 成功率、失败轨迹和是否能表达目标相关状态。

用户：

> 我现在有 z,a->z'、z,z'->a、z,a->r。我感觉是不是不要 x 和 dist 头好点？

Codex理解：

- 用户倾向减少人工特权监督，避免变成 px/dist 回归。
- px/dist 可以作为 probe，但不一定作为主 loss。
- 如果 px/dist 反传 encoder，就偏离“自学习表征”的目标。

用户：

> k-step 我觉得有必要，但连续控制后期微调的时候，k步变化可能很小，所以先留结构，先不一定用。

Codex理解：

- k-step 可以增强长期一致性，但当前任务步长小、视觉变化小，过强 k-step 不一定稳定。
- 先保留结构，逐步实验更合适。

### 7. 关于 SPR 结构

用户：

> 我想简化 cv，不要显式 heatmap，只用 resnet18 主干 + FPN 增强。

Codex当前实现：

- `src/cv_extractor/spr_state.py` 使用 ResNet18 + FPN + coord + conv。
- 没有显式 heatmap。
- 有 online encoder / target encoder / transition / projector / predictor。

用户：

> SPR 原本 RL 信号是主信号。我现在如果把表征和 PPO 拆开，就没有主信号。

Codex理解：

- SPR 应接入 PPO，而不是完全独立于 PPO。
- 当前主线应是 `PPO + SPR auxiliary + teacher MSE`。
- 纯 `teacher MSE + SPR` 只适合诊断，不是主线。

### 8. 关于随机 stop 目标

用户：

> 重点不是随机，而是每个 env 的 reward 终止 state 和机器人 state 里的任务目标对齐。

Codex理解：

- env0 可以是 `(end_d=1.5,end_x=0)`，env1 可以是 `(end_d=2.0,end_x=10)`。
- 每个 env 的 observation 和 reward 都必须使用自己的 end state。
- end state 应由 env 持有，不应由训练脚本临时拼出两套不一致逻辑。

用户：

> MDP 的高维 state 里拼 2 维任务目标会不会太弱？

Codex理解：

- 简单 concat 可能被高维 latent 淹没。
- 后续可考虑 goal encoder 或更强的 goal fusion。
- 但当前先保持简单实现，避免一次改太多变量。

### 9. 关于 rl_games

用户：

> rl_games 云服务器跑出了模型，我只想拿 best 回来看效果，不需要太多分析。

Codex理解：

- rl_games 是对照实验，不是当前主要开发路线。
- 云端 runs/checkpoint 输出过大，占满磁盘，需要限制。
- 本地需要单 env preview 和轨迹 CSV，用来判断是否公转、是否停在区外、是否穿墙。

### 10. 关于归档和 approved_models

用户：

> approved 里面要拿着就能用。视觉模型、teacher、student 都要保存，不要只保存 RL。

Codex理解：

- 保存实验时必须自包含。
- 需要包含 policy、teacher、视觉预训练、MDP/SPR checkpoint、config、trajectory/readme。
- 如果旧实验依赖模型找不到，要在 readme 明确说明不可完整复现。

## 当前技术状态摘要

### 主要文件

- 环境：`src/sb3_env.py`
- 视觉 env：`src/mdp_student_env.py`
- MDP 表征：`src/cv_extractor/mdp_state.py`
- MDP student：`train/mdp_student.py`
- SPR encoder：`src/cv_extractor/spr_state.py`
- SPR PPO：`src/spr_ppo.py`
- SPR student 主入口：`train/spr_student.py`
- SPR 配置：`train/spr_student_cfg.py`
- SPR BC 诊断入口：`train/spr_bc.py`

### 当前 SPR student 结构

```text
loss = PPO pi_loss
     + vf_coef * value_loss
     + teacher_coef * teacher_mse
     + SPR_COEF * spr_loss
     - entropy_coef * entropy
```

当前配置重点：

```python
SPR_START_ACTIVE = True
SPR_UPDATE_EVERY = 1
TEACHER_LOSS = 2.0
SPR_K = 8
ACTOR_ENCODER_COEF = 0.1
```

### 当前待 CC 审核的问题

1. `SPR_START_ACTIVE=True` + 每 update 更新 SPR 是否太激进。
2. `ACTOR_ENCODER_COEF=0.1` 是否合理，actor 梯度是否应该进 encoder。
3. std 发散优先用 clamp、阈值暂停 SPR，还是低频 SPR。
4. stop reward 是否应保持 `R_STOP=0`，只靠 success；或者加弱动作变小奖励。
5. MDP/SPR 是否应继续避免 px/dist 主监督。
6. goal 信息是否需要单独编码，而不是只 concat。
7. 如果想证明自监督表征价值，下一步任务复杂化应如何设计。
