# Project Summary

本项目是视觉强化学习机器人研究项目，从原 Webots + PPO + teacher guidance + imitation learning 路线迁移到 Isaac / IsaacLab，用于测试并行仿真、相机采集和后续视觉策略训练。整体走 train-by-cheat：特权信息 teacher PPO 训练视觉 student。

当前阶段：环境生成与视觉数据采集已就绪，正进入**视觉预训练**。

当前主要方向：

- 使用 Isaac 生成单机器人房间环境，每个机器人独立房间，避免看到其他环境的小球。
- 机器人主体由 `assets/robots/ball_robot.usd` 提供，结构包含 body、head、eye；eye 用于挂载相机。
- 采集脚本 `scripts/collect_cv_dataset.py` 使用 TiledCamera 并行采集图像，标签重点是小球图像横向位置 `px_x` 和距离 `dist`。
- 数据集默认输出到 `./data_isaac/`，运行前会清空旧数据；按 train/test/val 概率分目录保存，文件名使用全局共享编号。label 一行格式 `x_norm dist dist_bin 0 0 0 visible`。
- 当前 Isaac 带视觉相机的并行性能瓶颈主要来自相机渲染本身，env 数量、分辨率、rendering mode 都会明显影响速度和图像质量。

当前目录约定：

- `assets/` 放固定资源。
- `scripts/` 放可直接运行的脚本。
- `src/` 放后续可复用逻辑代码。
- `train/` 放训练和评估入口。
- `summary/` 放规则、项目状态、TODO 和每日总结。
- `data_isaac/` 放本地生成数据，不上传 Git。

## 视觉预训练方案（阶段一，进行中）

目的：在 `data_isaac` 上原样重训一个 encoder checkpoint，作为后续 RL 实验的共同起点；与 Webots 结果可对照，越忠实越好，与联合训练无关。

原视觉管线参考（只读，原项目 `../visual_rlrobot`）：

- 模型 `src/visual_rlrobot/visual_rlrobot/cv_extractor/simple_cnn.py`
- 配置 `cv_extractor/config.py`
- 训练 `train_cv_xbin.py`，超参 `hyperparams.py` 的 `CV_XBIN_TRAIN_CONFIG`

已确定项：

1. **state 张量 = `reserve_feature`（128 维）**，不用 `shared_feature`。reserve 是两 head 共享瓶颈，紧凑且已被 x/dist 监督压成任务相关表征；shared 维度高、噪声大、漂移难控。
2. **主结构 = `ResNet18FPNReserveXBin`（x 和 dist 都从 reserve 后解码）**；`ResNet18FPNXBin`（dist 从 shared 直接 detach 解码）留作对照。两者共享 backbone：ResNet18(ImageNet 预训练) + 小 FPN + heatmap → shared。
3. **bin head 先删**（原脚本 `bin_loss` 被置零，dist_bin 实际未参与）。后续 RL 若需离散距离信号再加回并真正用 CE 训。
4. 增强 / loss / 超参照旧：增强（仅 train）亮度 ×U(0.9,1.1)、对比度绕 127.5 ×U(0.9,1.1)、30% 概率加高斯噪声 std=3；loss = `SmoothL1(x) + SmoothL1(dist)`（权重均 1.0）；updates=1000, batch=64, lr=1e-4, Adam，按 test_loss 存 best。
5. 运行环境：纯 PyTorch，不依赖 Isaac，可在 conda `isaac_test` 直接跑。
6. 代码组织按目录约定：模型/逻辑放 `src/`，训练入口放 `train/`，产出到 `models/`。

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
