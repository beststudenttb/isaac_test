# TODO

## 采集与数据验证

- 确认手动修改后的 `assets/robots/ball_robot.usd` 中 eye 俯仰角、相机挂载结构和碰撞设置是否符合后续训练需求。
- 继续用肉眼抽查数据集图片质量，并确认 `px_x`、`dist` 标签和真实图像读取结果之间的误差范围。
- 用 `balanced + DLSS performance` 重新采集数据后，确认 CV 训练和视觉 student 推理时的图像分布一致。

## 阶段一：视觉预训练

- 用新采集数据顺序训练并比较四个 CV 模型：`resnet`、`mobile`、`old`、`old-mobile`。
- 用 `--val --size` 查看最终 `val` 集上的 gt/pred 和 mean err，重点比较原始像素 `x` 误差与 `dist` 误差。
- 根据四模型结果决定后续 RL 使用的 encoder checkpoint。
- 若 `old` 系明显更好，再考虑是否恢复 bin head 或补充 shared feature 对照。
- 试验纯表征预训练：用特权状态定义物理距离，把 RGB 映射到归一化 latent space，使 latent 距离结构接近特权状态距离；第一版不接回归 head，只训练表征空间。
  - 输入：RGB 图像与对应特权状态 `px_x/dist`。
  - 表征：`image_encoder(RGB) -> z`，`z` 做归一化，避免尺度漂移。
  - 监督：batch 内两两计算特权状态距离，并约束 latent 两两距离与之匹配。
  - 距离定义：`px_x` 与 `dist` 分开归一化，stop 附近可提高权重，避免远距离差异主导。
  - PPO 对照：使用纯 `z` 作为 state，与当前 `xd/shared/feature` 结果比较。

## 阶段二：视觉 student

- 先跑冻结 encoder + `old + xd` 的视觉 student，确认能否复现特权 student 的基本行为。
- 对比 `xd`、`shared`、`feature` 作为 PPO state 的效果。
- 引入视觉 student 前再次确认 `student_obs` / `priv_obs` 分离方案。
- MDP latent 接入 RL 前提醒用户决定 ResNet backbone 是否继续冻结、只解冻 `layer3`，还是全量解冻。

## 多 agent 协作（暂停）

- plan / state 通信机制暂不启用；后续如果确实需要双 agent 并行，再重新设计并定稿。

## 阶段二：联合训练（开放问题）

- teacher PPO 的观测/动作/奖励定义（需读原 Webots env 对齐，尚未读取）。
- N、M、λ、ε 的具体取值（实验调）。
- (B) RL 梯度进 encoder 是否纳入，取决于 (A)+KL锚 的效果。
