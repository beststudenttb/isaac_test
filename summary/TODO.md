# TODO

## 采集与数据验证

- 继续确认 Isaac 图像质量问题：比较 `quality`、`balanced`、`performance` 以及抗锯齿/settle frames 对清晰度和速度的影响。
- 用小规模采集先验证 `scripts/collect_cv_dataset.py` 能否在采满 `--samples` 后自动退出。
- 确认手动修改后的 `assets/robots/ball_robot.usd` 中 eye 俯仰角、相机挂载结构和碰撞设置是否符合后续训练需求。
- 后续开始训练前，确认 `px_x`、`dist` 标签和真实图像读取结果之间的误差范围。

## 阶段一：视觉预训练（代码移植）

- 移植模型到 `src/`：`ResNet18FPNReserveXBin` + `ResNet18FPNXBin`（去掉未用的 bin 分支），共享 ResNet18+FPN+heatmap backbone。
- 移植结构配置（`RESNET18_FPN_RESERVE_XBIN_CONFIG` 等）。
- 写训练入口到 `train/`：读 `data_isaac` 的 png+txt 的数据集类、训练/评估循环、增强、checkpoint。
- 产出 `models/cv_reserve/cv_xbin_best.pt` 等，并与 Webots 结果对照。

## 阶段二：联合训练（开放问题）

- teacher PPO 的观测/动作/奖励定义（需读原 Webots env 对齐，尚未读取）。
- N、M、λ、ε 的具体取值（实验调）。
- (B) RL 梯度进 encoder 是否纳入，取决于 (A)+KL锚 的效果。
