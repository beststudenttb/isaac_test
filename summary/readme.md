# Project Summary

本项目是视觉强化学习机器人研究项目，从原 Webots + PPO + teacher guidance + imitation learning 路线迁移到 Isaac / IsaacLab，用于测试并行仿真、相机采集和后续视觉策略训练。整体走 train-by-cheat：特权信息 teacher PPO 训练视觉 student。

当前阶段：teacher PPO、自写 student PPO、视觉 student 训练入口、视觉数据采集和 CV 预训练入口已就绪，正在把冻结 encoder 接入视觉 student 做第一版复现。

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

当前目录约定：

- `assets/` 放固定资源。
- `scripts/` 放可直接运行的脚本。
- `src/` 放后续可复用逻辑代码。
- `train/` 放训练入口。
- `val/` 放离线评估入口。
- `models/` 放训练现场输出，不上传 Git。
- `approved_models/` 放确认保留的模型和说明，需要上传 Git。
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

最终目标：中间特征（`reserve_feature`）当 RL 的 state，且持续参与训练让表征随任务演化；难点是防止"状态漂移"导致已训策略失效。关键判断：

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
