# Project Summary

本项目是视觉强化学习机器人研究项目，从原 Webots + PPO + teacher guidance + imitation learning 路线迁移到 Isaac / IsaacLab，用于测试并行仿真、相机采集和后续视觉策略训练。整体走 train-by-cheat：特权信息 teacher PPO 训练视觉 student。

当前阶段：teacher PPO、自写 student PPO、视觉 student、MDP student、SPR student 和离线评估入口都已就绪；当前重点是比较人工视觉表征、MDP latent 和 SPR latent 在 PPO student 中的稳定性，重点观察 success 曲线、std、teacher loss 和失败轨迹。

## 研究方向与总体框架（核心）

研究目标是**纯视觉自监督表征学习**：学一个可作策略 state 的视觉潜在表征 `h_t = Encoder(o_t)`，目的不是把单个任务做到最优，而是学到**通用、物体中心、像可控 MDP state** 的表征，以支撑当前 docking 与未来目标不那么明确的任务（goal-conditioned、抓取取回等）。

总体三段框架（递进，后段不一定做得到）：

1. 特权信息 teacher 策略（已有）。
2. CV 视觉表征预训练（当前阶段）。
3. 带 LBC、且 CV 部分也参与训练的 end2end 联合模型（目标，存在做不到的风险）。

已确定的边界与约定：

- **RL 阶段除 reward 外不使用任何特权信息**：student 观测纯视觉，teacher BC 属训练期蒸馏、部署纯视觉（learn-by-cheat 形态）。
- **下游策略走 model-free**，以保留泛化能力。
- 研究关注**状态转移一致性**与“state 下能决定动作”，而非 px/dist 数值精度；**px/dist 仅作诊断探针评估表征可访问性，不作表征的主监督目标**。
- 表征应保留连续相对位姿（距离 / bearing / 横向偏移 / 可见性 / 动作效果），以便后续把任务目标拼进 state 做 goal-conditioned 行为（如“球在左、停 3m”）；纯 px/dist 监督会得到**任务塌缩**表征，仅作对照基线（等价 cv_old）。

未定（随摸索调整）：

- CV 预训练是否使用特权信息：当前倾向只把 px/dist 用作 probe 诊断，不作为 encoder 主监督。
- 表征损失当前已搭好 forward、IDM 和 transition contrastive 架构；具体权重仍按实验调整。

当前主要方向：

- 使用 Isaac 生成单机器人房间环境，每个机器人独立房间，避免看到其他环境的小球。
- 机器人主体由 `assets/robots/ball_robot.usd` 提供，结构包含 body、head、eye；eye 用于挂载相机。
- 采集脚本 `scripts/collect_cv_dataset.py` 使用 TiledCamera 并行采集图像，标签重点是小球图像横向位置 `px_x` 和距离 `dist`。
- 数据集默认输出到 `./data_isaac/`，运行前会清空旧数据；按 train/test/val 概率分目录保存，文件名使用全局共享编号。label 一行格式 `x_norm dist dist_bin 0 0 0 visible`。
- 找球任务阶段定义固定记录在 `summary/task_flow.md`。
- 当前 Isaac 带视觉相机的并行性能瓶颈主要来自相机渲染本身，env 数量、分辨率、rendering mode 都会明显影响速度和图像质量。
- 视觉训练当前临时使用 `balanced + DLSS performance(dlss_mode=0)`；采集脚本支持用参数指定渲染模式和抗锯齿模式。
- `train/cv.py` 使用 `train/test/val = 7/2/1` 的语义：`train` 参与训练，`test` 用于训练过程评估和保存 best，`val` 只在 `--val` 时做最终查看。
- `scripts/run_vision_pipeline.sh` 可顺序执行：采集 50000 张图像、训练四个 CV 模型、用 `old + xd` 接入视觉 student。

## MDP / SPR 视觉表征方向（当前重点）

目标是学习一个视觉 latent，使它满足：

- `T(h_t, a_t) -> h_{t+1}`：动作条件下的 latent 转移一致。
- `I(z_img_t, z_img_{t+1}) -> a_t`：逆动力学能从纯视觉 latent 变化中读出动作效果。
- transition contrastive：预测的下一 latent 应更接近真实下一 latent，而不是 batch 内其他候选。
- px/dist probe：只作为诊断头，输入处 detach，不把特权几何监督反传给 encoder。

MDP 当前实现状态：

1. `src/cv_extractor/mdp_state.py` 中 `MDPStateNet` 包含 image encoder、GRU belief、posterior/prior z、forward dynamics、IDM、reward/value/done/probe heads。
2. `train/mdp_state.py` 是离线 MDP 表征预训练入口；使用归一化 latent 做 forward dyn loss，记录 `contrast`、`idm`、`retrieval1`、`eff_rank` 和 probe 诊断。
3. `train/mdp_student.py` 的在线 MDP 微调逻辑已同步到相同 loss 语义，后续接 PPO 时不需要再重新对齐。
4. `train/mdp_student.py` 当前默认使用冻结的预训练 MDP checkpoint，不在 PPO 内继续训练 MDP；后续如果恢复在线 MDP 微调，相关代码和 cfg 仍保留。
5. 当前 contrastive 仍使用 batch 内展平负样本，可能包含同轨迹邻近帧的假负样本；先跑对照实验，后续必要时改 masked InfoNCE。

SPR 当前实现状态：

1. `src/cv_extractor/spr_state.py` 提供 ResNet18 + FPN + coord + conv 的 SPR encoder，含 online encoder、target encoder、transition、projector 和 predictor。
2. `src/spr_ppo.py` 提供 SPR 视觉 state 的 PPO 组件：actor/critic、rollout、PPO update 和 k-step SPR loss。
3. `train/spr_student.py` 是当前 SPR + PPO 主入口：保留 PPO 主信号，teacher 使用 MSE 辅助，SPR/encoder 当前配置为从训练开始更新。
4. 近期一次 `spr_student` 训练中 `spr_loss` 全程为 0，原因是旧配置只有 `std_mean < 0.3` 才开启 SPR，而该条件从未满足；因此那次结果不能说明 SPR 本身有效或无效。
5. `train/spr_bc.py` 是短名诊断入口，只用于 teacher MSE + SPR 的简单对照；主线仍使用 `train/spr_student.py`。

## RL 当前实验状态

- `src/sb3_env.py` 已把 teacher / student 的特权观测改为 `[px, dist, end_d, end_x]`，并在 reset 时由 env 统一采样 stop 目标；reward、stop 区域和 success 判断使用同一份 end state。
- approach 阶段的状态改善 reward 已改为固定尺度：`px` 改善除以 `IMAGE_WIDTH=224`，`dist` 改善除以 `DIST_MAX - FAIL_NEAR = 6`，`K_X/K_D` 后续手动调。
- 视觉 student 和 MDP student 的策略输入都显式拼接 `end_obs`，避免随机 stop 目标时策略不知道任务目标。
- teacher `update_88` 离线评估约 `506/512` 成功，失败样本全部是 timeout，主要原因是初始目标在相机右侧视野外且距离远，search 阶段转头较慢。
- 当前正在比较 `old+shared`、MDP state、SPR state 等视觉 student；离线 val 仍主要看 success 曲线、best update 和失败轨迹。
- teacher MSE 只监督 actor mean，不会直接约束 `log_std`；若 std 发散，需要单独处理 std 或使用 NLL 作为诊断实验。
- rl_games PPO 环境包已能在云端训练和本地 preview；云端 runs/checkpoint 容易占用大量磁盘，后续只保留必要 best/last 和关键 CSV。
- 近期 SPR student 的一次训练中 SPR 实际未激活，std 发散主要来自 PPO 采样、action clamp 和稀疏成功信号；当前已改为从训练开始启用 SPR。

## 近期实验结论

- teacher 可以加速视觉 student 早期形成策略，但 teacher MSE 不能约束 std，也不能保证后期 PPO 不偏离。
- `old+shared`、MDP state 和 SPR state 的差异需要看 deterministic offline val 与失败轨迹，不能只看训练 rollout 的 success。
- stop 区域 reward 仍是关键不稳定来源：密集 stop 奖励可能诱导刷分或边缘行为，纯 success 信号又过于稀疏。
- 随机 stop 任务已经具备基本代码路径；后续重点是检查任务目标进入 state 后是否真的被策略利用。
- 自监督表征要想超过人工 px/dist，需要更复杂或更难人工表征的任务，否则简单 docking 上人工几何特征天然占优。

当前目录约定：

- `assets/` 放固定资源。
- `scripts/` 放可直接运行的脚本。
- `src/` 放后续可复用逻辑代码。
- `train/` 放训练入口。
- `val/` 放离线评估入口。
- `models/` 放训练现场输出，不上传 Git。
- `approved_models/` 放确认保留的模型和说明，需要上传 Git。
- `approved_models/` 中每个确认实验应尽量自包含：policy、teacher、视觉预训练、MDP/SPR checkpoint、配置和 trajectory 都要能复现。
- `summary/` 放规则、项目状态、TODO 和每日总结。
- `data_isaac/` 放本地生成数据，不上传 Git。

## 视觉预训练方案（阶段一，进行中）

目的：在 `data_isaac` 上训练和筛选 encoder checkpoint，作为后续 RL 实验的视觉状态起点；Webots 只用于对齐任务语义、标签格式、旧结构和关键超参，不直接照搬旧项目复杂训练框架。

原视觉管线参考（只读，只参考语义和参数，原项目 `../visual_rlrobot`）：

- 模型 `src/visual_rlrobot/visual_rlrobot/cv_extractor/simple_cnn.py`
- 配置 `cv_extractor/config.py`
- 训练 `train_cv_xbin.py`，超参 `hyperparams.py` 的 `CV_XBIN_TRAIN_CONFIG`

当前实现：

1. 视觉模型在 `src/cv_extractor/simple_cnn.py`，统一输出 `reserve_feature`、`x_pred` 和 `distance_pred`。
2. `train/cv.py` 支持四个模型：`resnet`、`mobile`、`old`、`old-mobile`。
3. `resnet` / `mobile` 是当前轻量 attention 结构，分别使用 ResNet18 和 MobileNetV3-small backbone。
4. `old` 是旧 Webots 的 `ResNet18FPNReserveXBinFeatureExtractor` 结构迁移版，仅改类名和输入适配；`old-mobile` 使用同一套 old FPN/reserve/head 结构，但 backbone 换为 MobileNetV3-small。
5. **bin head 先删**（原脚本 `bin_loss` 被置零，dist_bin 实际未参与）。后续 RL 若需离散距离信号再加回并真正用 CE 训。
6. 增强 / loss / 超参沿用旧实验中已验证的核心设定：增强（仅 train）亮度 ×U(0.9,1.1)、对比度绕 127.5 ×U(0.9,1.1)、30% 概率加高斯噪声 std=3；loss = `SmoothL1(x) + SmoothL1(dist)`（权重均 1.0）；updates=1000, batch=64, lr=1e-4, Adam，按 test_loss 存 best。
7. `scripts/train_cv_all.py` 可按顺序训练或 val 四个模型，产出分别位于 `models/vision/cv_resnet`、`models/vision/cv_mobile`、`models/vision/cv_old`、`models/vision/cv_old_mobile`。
8. 运行环境：纯 PyTorch，不依赖 Isaac，可在 conda `isaac_test` 直接跑；`--device auto` 会优先使用 CUDA。

## 联合训练方案（阶段二，后续方向）

最终目标：视觉 latent 当 RL 的 state，且持续参与训练让表征随任务演化；难点是防止"状态漂移"导致已训策略失效。关键判断：

1. **M 轮视觉更新以特权监督 loss 为主，RL 梯度进 encoder 最多作小附加项 / 后续消融。**
   - (A) 特权监督：继续训 x/dist，标签由 Isaac 每步近免费提供（正是 teacher 的特权信息），特征锚在几何意义上，漂移有界且良性。
   - (B) RL 策略梯度穿过 encoder：才是端到端自学表征，但也是不稳定与漂移主因。先验证稳的 (A)，(B) 作消融。
2. **防漂移约束对象是"encoder 改变引起的策略行为变化"，不是 PPO 自身的 clip/ratio。** 正确做法是在最近观测上加行为保持信赖域 `KL( π_frozen(·|f_old(o)) ‖ π_frozen(·|f_new(o)) ) ≤ ε`，即更新 encoder 后同一冻结策略喂新特征产生的动作分布要接近旧的。M 轮目标 `L = 特权监督loss + λ·KL锚`。
3. **critic 同样会随 encoder 改变失效。** 仅冻策略不够：视觉更新时 critic 一起冻、更新后再 warm-up 几轮，或把 critic 纳入锚约束。
4. **M 要小、走信赖域。**

分阶段交替训练草案：

```
循环：
  Phase A（训策略）：encoder 冻结，PPO 更新 π/V 共 N 轮
  Phase B（训视觉，M ≪ N 轮）：π 冻结(critic 一起冻)，更新 encoder
      L = 特权监督loss(x,dist) + λ · KL( π(·|f_old(o)) ‖ π(·|f_new(o)) )
  Phase B 后：warm-up 几轮 critic 适配新 encoder
```

必须有的对照基线：**冻结 encoder**（Webots 原稿那套）作 control，再比较 (A)+KL锚 vs (A)+(B) 是否真的赢过冻结、赢多少；复杂度要用数据 justify。
