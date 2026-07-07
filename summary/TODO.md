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
- 对比 `old+shared`、MDP state、SPR state 的 success 曲线、best update 和失败轨迹。
- 引入视觉 student 前再次确认 `student_obs` / `priv_obs` 分离方案。
- MDP latent 接入 RL 前提醒用户决定 ResNet backbone 是否继续冻结、只解冻 `layer3`，还是全量解冻。
- 跑当前 `train/spr_student.py`，确认 `spr_loss` 从 update 1 开始非零，并记录 std 是否继续发散。
- 如果 std 继续变大，分别测试：std clamp、std 阈值控制 SPR 更新、固定低频 SPR 更新。
- SPR 四阶段正式跑：阶段1 数据量从 50~100 update 起步；停止标准看 eff_rank/spr_loss 平台与 spr_bc BC 探针，不追求固定条数；多样性(加噪、随机 goal、spawn 范围)优先于条数。
- 阶段4 表征 replay 方案（触发条件：解冻后表征退化、或远端/search 状态成功率下降，再实施）：CPU 蓄水池保留 M=10~20 个历史 rollout 的图像对（约 4~8 万帧、6~12GB 内存），阶段4 SPR minibatch 从当前 rollout 与 buffer 混采。理由：SSL 对过时数据合法；防止策略变好后数据覆盖塌缩、encoder 遗忘远端状态。阶段1 分布平稳，不需要 replay。
- 图像级遮挡增强（cutout）实验：patch 以背景为主，球至多部分遮挡（≤1/3、低概率）；同一时间对 (t, t+k) 必须用相同 patch 位置（避免注入假运动淹没真实小位移）；保留约一半无增强样本；与 3D 场景遮挡物（任务复杂化/评估用）分开，不混为一个实验。
- 对 `old+shared` 后期成功率下降的轨迹做统一统计：进入 stop 前是否对齐、stop 内是否动、出 stop 是由 x/y/omega 哪个动作导致。
- 比较固定 stop 与随机 stop 的 teacher/student 差异，确认随机目标是否被策略真正利用。

## rl_games / 云服务器

- 限制 rl_games 训练输出，只保存必要 best/last 和关键 CSV，避免 runs 目录占满云服务器数据盘。
- 用 `scripts/preview_rl_games.py` 对云端 checkpoint 做单 env 轨迹检查，重点看是否停在 stop 区外、是否穿墙、是否公转。

## 模型归档

- 每次把模型保存到 `approved_models/` 时，检查该实验是否自包含：policy、teacher、视觉预训练、MDP/SPR checkpoint、config、trajectory 是否都在说明中写清楚。
- 清点旧 `approved_models/` 中缺失的依赖模型；能从现场文件恢复的补齐，找不到的在 readme 中明确标注不可完整复现。

## 多 agent 协作（暂停）

- plan / state 通信机制暂不启用；后续如果确实需要双 agent 并行，再重新设计并定稿。

## 阶段二：联合训练（开放问题）

- teacher PPO 的观测/动作/奖励定义（需读原 Webots env 对齐，尚未读取）。
- N、M、λ、ε 的具体取值（实验调）。
- (B) RL 梯度进 encoder 是否纳入，取决于 (A)+KL锚 的效果。
- 判断 SPR 是否需要 IDM、k-step 或 value/reward 辅助；每次只改一个主要变量，避免无法归因。
