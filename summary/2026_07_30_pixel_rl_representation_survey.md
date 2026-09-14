# 像素输入 RL 与表征学习:文献核实与分析(2026-07-30)

> 用途:论文 Related Work / Discussion 章节的引用基础。全部条目已逐条打开 arXiv 原文页 / OpenReview / PMLR 核实;个别 venue 信息无法从原始页面确认的,已明确标注。

## 0. 背景动机

本项目在 Isaac 视觉停靠任务上观察到:冻结 SPR 表征的 off-policy 线稳定成功,而从头可训 CNN 及 SPR 联合训练的线均出现 Q 值发散、策略饱和、run 间方差巨大。这一现象与像素 RL 文献中反复报告的模式一致——TD 学习直接驱动视觉 encoder 时存在系统性的不稳定机制,而将表征学习与 RL 解耦(冻结预训练表征)可以绕开这些机制。本文档梳理相关文献的失败机制分析、各类表征学习方案的对比结论,以及核心争点"TD 梯度是否适合训练视觉表征"的正反证据。

## 1. 问题定义:像素 RL 中表征从哪来

像素输入 RL 必须回答一个问题:视觉 encoder 的参数由什么信号驱动。文献中的方案可归为四类路线:

| 路线 | 表征训练信号 | 代表工作 |
|---|---|---|
| A. TD 端到端 | 仅 critic 的 TD 梯度(通常阻断 actor 梯度)回传进 encoder | DrQ, DrQ-v2, RAD, A-LIX |
| B. TD + SSL 联合 | TD 梯度 + 自监督辅助 loss 同时训 encoder | SAC+AE, CURL, SPR |
| C. 增广正则 | 仍是 TD 端到端,但以输入增广作为隐式正则(与 A 常合并) | DrQ 系, RAD |
| D. 解耦 / 冻结预训练 | encoder 由无监督/预训练目标单独训练,RL 梯度不进 encoder | ATC, PVR, R3M, MVP, VIP |

其中 A/C 的立场是"TD 梯度可以训表征,只要加对正则";B 的立场是"TD 梯度信号太弱/太不稳,需要辅助信号";D 的立场是"TD 梯度对表征不必要甚至有害,解耦更好"。

## 2. 失败机制的文献分析

### 2.1 Deadly triad:函数近似 + bootstrapping + off-policy 的发散

van Hasselt et al. (2018), *Deep Reinforcement Learning and the Deadly Triad*, arXiv:1812.02648。经典理论预测三要素结合时价值估计可发散;该文实证考察深度 Q 网络中 deadly triad 的实际表现,指出理论与实践存在缝隙——DQN + 经验回放这类组合经常成功,说明发散不是必然,但各组件(网络规模、bootstrap 长度、优先级采样等)确实系统性地影响不稳定的出现。这是"价值学习内在不稳"论述的总源头。
出处:https://arxiv.org/abs/1812.02648 (arXiv preprint;未在原始页面查到正式会议 venue)

### 2.2 特征共适应与隐式退化:TD 梯度对表征的内在压力

- Kumar et al., *DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization*, ICLR 2022 (Spotlight), arXiv:2112.04716。核心主张:SGD 的隐式正则化在监督学习中有益,但在 TD/bootstrapping 下变为有害——隐式正则会偏向"excessive aliasing"的退化解,使 Bellman backup 两端的 (s,a) 表征发生共适应(feature co-adaptation)、点积增大、表征塌缩;因此价值学习需要**显式**正则(DR3)来对抗。这是"TD 梯度天然不适合训表征"最直接的机制性证据。
  出处:https://arxiv.org/abs/2112.04716 与 https://openreview.net/forum?id=POvMvLi91f
- Kumar et al., *Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning*, ICLR 2021, arXiv:2010.14498。补充证据:bootstrapping 与梯度优化的"pathological interaction"导致价值网络特征矩阵的有效秩持续下降(rank collapse),梯度步数越多表达能力越差,且与性能退化相关;控制 rank collapse 可改善性能。
  出处:https://arxiv.org/abs/2010.14498

### 2.3 像素 critic 的灾难性自过拟合(visual deadly triad)

Cetin et al., *Stabilizing Off-Policy Deep Reinforcement Learning from Pixels*, ICML 2022, arXiv:2207.00986。核心主张:TD 学习 + 卷积 encoder + 中等幅度奖励构成"visual deadly triad",导致 **catastrophic self-overfitting**——非平稳的 TD 目标使 CNN encoder 产生不平滑、判别性过强的特征,critic 对自己生成的目标过拟合,训练崩溃。解法 A-LIX 对 encoder 梯度做自适应平滑正则;作者同时指出图像增广(DrQ 类)之所以有效,本质上起的就是类似的梯度平滑/正则作用。这一篇同时是"端到端可行"与"TD 梯度裸训 encoder 必炸"两个论断的证据(见第 3 节)。
出处:https://arxiv.org/abs/2207.00986

### 2.4 Capacity loss / 可塑性丧失

- Lyle, Rowland & Dabney, *Understanding and Preventing Capacity Loss in Reinforcement Learning*, ICLR 2022, arXiv:2204.09560。核心主张:在一系列非平稳目标上训练的网络会逐渐丧失快速拟合新目标的能力(capacity loss),在稀疏奖励下最严重(早期长时间只能拟合近零目标,表征先塌了,等奖励信号出现时已学不动);InFeR(把一部分特征回归到初始化值)可缓解,在 Montezuma's Revenge 等稀疏奖励任务上收益显著。**这与本项目"稀疏精确停止任务上从头训 CNN 冷启动死锁"的观察直接对应。**
  出处:https://arxiv.org/abs/2204.09560
- Lyle et al., *Understanding plasticity in neural networks*, ICML 2023 (oral), arXiv:2303.01486。后续机制分析:可塑性丧失与 loss landscape 曲率变化深度相关,且常在没有饱和单元的情况下发生;一系列参数化与优化设计选择可保持可塑性。
  出处:https://arxiv.org/abs/2303.01486

### 2.5 Primacy bias:过拟合早期经验

Nikishin et al., *The Primacy Bias in Deep Reinforcement Learning*, ICML 2022, arXiv:2205.07802。核心主张:深度 RL agent 倾向于过拟合早期交互数据、忽视后期证据(primacy bias);周期性重置网络的一部分(保留 replay buffer)在 Atari 100k 与 DMC 上一致性提升。含义:encoder/critic 在低质量早期数据上形成的表征会锁死后续学习——又一条"表征被 RL 训练过程自身损坏"的机制。
出处:https://arxiv.org/abs/2205.07802

### 2.6 评估层面:seed 方差与统计规范

- Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018, arXiv:1709.06560。核心主张:深度 RL 结果对随机种子、超参数、代码实现高度敏感,复现困难;没有显著性检验与规范报告时,无法判断"改进"是否真实。
  出处:https://arxiv.org/abs/1709.06560
- Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021 (Outstanding Paper), arXiv:2108.13264。核心主张:少量 run 下仅报告 mean/median 点估计不可靠,点估计结论与严格统计分析结论可有实质分歧;建议报告区间估计、performance profiles、IQM(interquartile mean)聚合指标与 stratified bootstrap 置信区间(rliable 库)。**本项目 run 间方差巨大的现象说明:比较可训 CNN 线与冻结表征线时必须按此规范做多 seed + IQM 报告。**
  出处:https://arxiv.org/abs/2108.13264

## 3. 核心争点:TD 梯度适不适合训练视觉表征——正反证据

### 正方:TD 端到端可行(但都带附加条件)

| 证据 | 主张 | 成功条件 |
|---|---|---|
| DrQ — Kostrikov, Yarats & Fergus, *Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels*, ICLR 2021 (Spotlight), arXiv:2004.13649 | 无需辅助 loss 或预训练,仅靠输入增广对 Q 函数做正则,SAC 即可从像素稳定学习,超过 model-based 与对比学习(CURL) | **必须有增广正则**;DMC 稠密奖励基准 |
| RAD — Laskin et al., *Reinforcement Learning with Augmented Data*, NeurIPS 2020, arXiv:2004.14990 | 简单 RL 算法 + 数据增广即可超过复杂的 SOTA 方法(含辅助 loss 方法),并改善泛化 | 同上,增广是关键成分 |
| DrQ-v2 — Yarats, Fergus, Lazaric & Pinto, *Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning*, ICLR 2022, arXiv:2107.09645 | 端到端从像素首次(model-free)解决 humanoid 运动任务 | 增广 + 换 DDPG/n-step return + 探索噪声调度;DMC 稠密奖励;逐任务调参的 benchmark 环境 |
| SAC+AE — Yarats et al., *Improving Sample Efficiency in Model-Free Reinforcement Learning from Images*, arXiv:1910.01741(preprint,原始页面未见正式 venue) | critic 梯度**可以**训 encoder:因 SAC 中策略是 Q 诱导的 Boltzmann 分布投影,"Q 函数包含全部任务相关信息",故 critic 信号足够 | 但(1)**actor 梯度必须阻断**:§4.4 "Preventing the actor's gradients from updating the convolutional encoder helps to improve performance even further";(2)还需 AE 重建辅助 loss + 谨慎的 VAE→确定性 AE 替换才稳定 |
| A-LIX — Cetin et al., ICML 2022(见 2.3) | 端到端可行且不需增广/辅助 loss | 但需对 encoder 梯度做显式自适应平滑正则——等于承认裸 TD 梯度会自过拟合 |
| RLPD — Ball et al., *Efficient Online Reinforcement Learning with Offline Data*, ICML 2023, arXiv:2302.02948 | 价值学习可以稳定高效 | §4.2 "Layer Normalization Mitigates Catastrophic Overestimation":LayerNorm 使 Q 值被权重范数界住("the Q-values are bounded by the norm of the weight layer, even for actions outside the dataset"),抑制 critic 对数据外动作的灾难性外推/发散 |
| BRO — Nauman et al., *Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control*, NeurIPS 2024 (Spotlight), arXiv:2405.16158 | critic 网络可以放大并稳定训练,model-free 近乎解决 Dog/Humanoid | 前提是"strong regularization allows for effective scaling of the critic networks"——强正则是放大的先决条件 |

正方小结:没有一篇"裸 TD 端到端"的成功案例——每个成功都伴随增广、梯度平滑、LayerNorm、reset 或阻断 actor 梯度中的至少一种;且成功域集中在稠密奖励的 benchmark 套件。

### 反方:TD 梯度对表征不必要 / 有害

| 证据 | 主张 |
|---|---|
| ATC — Stooke et al., *Decoupling Representation Learning from Reinforcement Learning*, arXiv:2009.08319(2021;PMLR/ICML 2021 收录,arXiv 页面未标注,见参考文献注) | 摘要原文:"training the encoder exclusively using ATC matches or outperforms end-to-end RL in most environments"——完全用无监督时序对比目标训 encoder(RL 不回传梯度)即可匹配或超过端到端;还验证了预训练后**冻结权重**用于 RL 的设置 |
| PVR — Parisi et al., *The Unsurprising Effectiveness of Pre-Trained Vision Models for Control*, ICML 2022, arXiv:2203.03580 | 摘要原文:"pre-trained visual representations can be competitive or even better than ground-truth state representations to train control policies"——仅用域外视觉数据预训练的冻结表征,可媲美甚至超过真值状态特征 |
| R3M — Nair et al., *R3M: A Universal Visual Representation for Robot Manipulation*, CoRL 2022, arXiv:2203.12601 | R3M "can be used as a frozen perception module for downstream policy learning";相比从头训练提升超过 20%,相比 CLIP/MoCo 超过 10%(模仿学习设置) |
| MVP — Xiao et al., *Masked Visual Pre-training for Motor Control*, arXiv:2203.06173(preprint,原始页面未见正式 venue) | 摘要原文:"We then freeze the visual encoder and train neural network controllers on top with reinforcement learning. We do not perform any task-specific fine-tuning of the encoder"——冻结 MAE encoder + RL 训上层,比有监督 encoder 绝对成功率高至 80% |
| VIP — Ma et al., *VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training*, ICLR 2023 (Spotlight), arXiv:2210.00030 | 把表征学习形式化为离线 goal-conditioned 价值问题,从人类视频自监督预训练;**冻结**表征可为未见任务零样本生成稠密奖励 |
| DR3 / implicit under-parameterization(见 2.2) | 机制层面:TD 梯度自带表征退化压力(共适应、rank collapse),不加显式正则必然侵蚀表征 |
| Li et al., *Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels?*, NeurIPS 2022, arXiv:2206.05266 | 对 TD+SSL **联合训练**路线的否定性结论,原文:"existing SSL framework for RL fails to bring meaningful improvement over the baselines only taking advantage of image augmentation when the same amount of data and augmentation is used";即使进化搜索 loss 组合也"fails to meaningfully outperform...carefully designed image augmentations";"no single self-supervised loss or image augmentation method can dominate all environments";结论是"the current framework for joint optimization of SSL and RL is limited" |
| Cetin et al.(见 2.3) | 裸 TD 梯度训 CNN 导致 catastrophic self-overfitting——正方阵营内部对反方机制的确认 |

### 争点裁决(供论文 Discussion 使用)

两方并不真正矛盾,可综合为三点:
1. **TD 梯度携带任务相关信号但同时携带退化压力**(共适应、rank collapse、自过拟合、可塑性侵蚀);正则化(增广/梯度平滑/LayerNorm/reset)决定净效果的符号。
2. **奖励密度是分水岭**:正方成功案例几乎全部在稠密奖励 benchmark;稀疏奖励下 TD 信号长期为零,capacity loss(Lyle et al. 2022)预言表征先塌、后学不动——与本项目冷启动死锁观察一致。
3. **联合(TD+SSL)是最脆弱的中间态**:Li et al. 2022 表明 SSL 辅助 loss 在联合优化框架下增益不稳定;而完全解耦(ATC/PVR/R3M/MVP/VIP)反而稳定成立。即"要么正则化好 TD 端到端,要么彻底解耦;两种梯度混在同一 encoder 里最不可靠"。

## 4. 各路线对比结论与适用条件

| 路线 | 证据强度 | 适用条件 | 风险 |
|---|---|---|---|
| A/C. TD 端到端 + 增广正则(DrQ-v2 型) | 在 DMC/Atari 稠密奖励基准上最强 | 稠密奖励、增广对任务语义无破坏、可接受较大 seed 方差、有调度/调参预算 | 稀疏奖励冷启动死锁;自过拟合(需 A-LIX 型梯度正则兜底);primacy bias(需 reset) |
| B. TD + SSL 联合 | 单篇工作(CURL/SPR)各自报正增益,但系统评估(Li et al. 2022)显示相对增广基线增益不稳定 | SSL 目标与任务高度对齐时可能有益 | 两种梯度在同一 encoder 上冲突;结论不稳、环境依赖 |
| D. 解耦训练 / 冻结预训练表征 | ATC(域内无监督)与 PVR/R3M/MVP/VIP(域外大规模预训练)一致支持 | 预训练数据分布覆盖任务视觉域;下游任务的决定性信息可被通用表征保留 | 表征-任务错配时上限受损;无法在线适配任务特异特征 |
| 通用稳定器(与路线正交) | LayerNorm(RLPD §4.2、BRO)、周期 reset(Nikishin et al.)、显式特征正则(DR3、InFeR)、encoder 梯度平滑(A-LIX) | 任何 off-policy 价值学习 | — |

对本项目的直接含义:稀疏精确停止 + off-policy 的设定同时踩中 capacity loss(2.4)、自过拟合(2.3)、共适应(2.2)三个机制的高危区,冻结 SPR 表征线的稳定成功与 ATC/PVR 一系的解耦结论方向一致;而可训 CNN 线的 Q 发散与 run 间大方差分别对应 2.1–2.3 与 2.6 的文献预期。

## 5. 完整参考文献列表(全部经原始页面核实)

1. **Deep Reinforcement Learning and the Deadly Triad** — Hado van Hasselt, Yotam Doron, Florian Strub, Matteo Hessel, Nicolas Sonnerat, Joseph Modayil. 2018, arXiv preprint(原始页面未标注会议 venue)。https://arxiv.org/abs/1812.02648 — 实证考察函数近似+bootstrapping+off-policy 三要素在深度 Q 学习中何时导致不稳定/发散。
2. **DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization** — Aviral Kumar, Rishabh Agarwal, Tengyu Ma, Aaron Courville, George Tucker, Sergey Levine. ICLR 2022 (Spotlight)。https://arxiv.org/abs/2112.04716 / https://openreview.net/forum?id=POvMvLi91f — TD 下 SGD 隐式正则有害,导致特征共适应/aliasing,需显式正则对抗。
3. **Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning** — Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, Sergey Levine. ICLR 2021。https://arxiv.org/abs/2010.14498 — bootstrapping+梯度优化导致价值网络特征有效秩塌缩,与性能退化相关。
4. **Stabilizing Off-Policy Deep Reinforcement Learning from Pixels** — Edoardo Cetin, Philip J. Ball, Steve Roberts, Oya Celiktutan. ICML 2022。https://arxiv.org/abs/2207.00986 — "visual deadly triad"导致 catastrophic self-overfitting;A-LIX 自适应平滑 encoder 梯度;增广的本质是梯度正则。
5. **Understanding and Preventing Capacity Loss in Reinforcement Learning** — Clare Lyle, Mark Rowland, Will Dabney. ICLR 2022。https://arxiv.org/abs/2204.09560 — 非平稳目标序列使网络丧失拟合新目标能力,稀疏奖励下最严重;InFeR。
6. **Understanding plasticity in neural networks** — Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney. ICML 2023 (oral)。https://arxiv.org/abs/2303.01486 — 可塑性丧失与 loss landscape 曲率变化相关,可通过参数化/优化设计缓解。
7. **The Primacy Bias in Deep Reinforcement Learning** — Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron Courville. ICML 2022。https://arxiv.org/abs/2205.07802 — 过拟合早期经验;周期性重置网络部分参数一致性提升。
8. **Improving Sample Efficiency in Model-Free Reinforcement Learning from Images** — Denis Yarats, Amy Zhang, Ilya Kostrikov, Brandon Amos, Joelle Pineau, Rob Fergus. 2019, arXiv preprint(原始页面未标注 venue;方法名 SAC+AE,见其 §4.5)。https://arxiv.org/abs/1910.01741 — 确定性 AE 辅助 loss 稳定联合训练;§4.4 明确阻断 actor 梯度进 encoder、仅 critic 梯度训 encoder。
9. **Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels** (DrQ) — Ilya Kostrikov, Denis Yarats, Rob Fergus. ICLR 2021 (Spotlight)。https://arxiv.org/abs/2004.13649 / https://openreview.net/forum?id=GY6-6sTvGaf — 输入增广对 Q 函数正则,无需辅助 loss/预训练即可从像素学习。
10. **Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning** (DrQ-v2) — Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto. ICLR 2022。https://arxiv.org/abs/2107.09645 / https://openreview.net/forum?id=_SJ-_yyes8 — 增广+DDPG/n-step+噪声调度,model-free 首次纯像素解 humanoid。
11. **Reinforcement Learning with Augmented Data** (RAD) — Michael Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, Aravind Srinivas. NeurIPS 2020。https://arxiv.org/abs/2004.14990 — 简单 RL+增广超过复杂 SOTA,并改善泛化。
12. **CURL: Contrastive Unsupervised Representations for Reinforcement Learning** — Michael Laskin, Aravind Srinivas, Pieter Abbeel(PMLR 作者序;arXiv 页面为 Srinivas, Laskin, Abbeel)。ICML 2020, PMLR 119:5639-5650。https://arxiv.org/abs/2004.04136 / https://proceedings.mlr.press/v119/laskin20a.html — 对比学习辅助 loss 与 off-policy RL 联合;首个接近 state-based 样本效率的像素方法(主张)。
13. **Data-Efficient Reinforcement Learning with Self-Predictive Representations** (SPR) — Max Schwarzer, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, Philip Bachman. ICLR 2021。https://arxiv.org/abs/2007.05929 — 多步潜空间自预测 + EMA 目标 encoder + 增广一致性,Atari 100k 大幅提升。
14. **Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels?** — Xiang Li, Jinghuan Shang, Srijan Das, Michael S. Ryoo. NeurIPS 2022。https://arxiv.org/abs/2206.05266 — 同等数据与增广下,现有 SSL+RL 联合框架相对纯增广基线无有意义增益;联合优化框架本身受限。
15. **Decoupling Representation Learning from Reinforcement Learning** (ATC) — Adam Stooke, Kimin Lee, Pieter Abbeel, Michael Laskin. 2021(arXiv 页面未标注 venue;通常引为 ICML 2021, PMLR 139——此 venue 信息未在本次打开的页面直接核实)。https://arxiv.org/abs/2009.08319 — 仅用无监督时序对比目标训 encoder 即匹配或超过端到端 RL;含冻结权重设置。
16. **The Unsurprising Effectiveness of Pre-Trained Vision Models for Control** (PVR) — Simone Parisi, Aravind Rajeswaran, Senthil Purushwalkam, Abhinav Gupta. ICML 2022, PMLR 162:17359-17371。https://arxiv.org/abs/2203.03580 — 仅域外数据预训练的冻结视觉表征可媲美甚至超过真值状态表征。注意确切标题为 "Unsurprising"(非 "(Un)Surprising")。
17. **R3M: A Universal Visual Representation for Robot Manipulation** — Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, Abhinav Gupta. CoRL 2022。https://arxiv.org/abs/2203.12601 — Ego4D 预训练的冻结感知模块,较从头训练 +20% 以上。
18. **Masked Visual Pre-training for Motor Control** (MVP) — Tete Xiao, Ilija Radosavovic, Trevor Darrell, Jitendra Malik. 2022, arXiv preprint(原始页面未标注 venue)。https://arxiv.org/abs/2203.06173 — MAE 预训练后冻结 encoder、RL 只训上层控制器,不做任务特定微调。
19. **VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training** — Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, Amy Zhang. ICLR 2023 (Spotlight / notable-top-25%)。https://arxiv.org/abs/2210.00030 — 表征学习形式化为离线 goal-conditioned 价值问题;冻结表征零样本生成稠密奖励。
20. **Efficient Online Reinforcement Learning with Offline Data** (RLPD) — Philip J. Ball, Laura Smith, Ilya Kostrikov, Sergey Levine. ICML 2023。https://arxiv.org/abs/2302.02948 — §4.2 "Layer Normalization Mitigates Catastrophic Overestimation":LayerNorm 以权重范数界住数据外动作的 Q 值,抑制 critic 发散。
21. **Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control** (BRO) — Michal Nauman, Mateusz Ostaszewski, Krzysztof Jankowski, Piotr Miłoś, Marek Cygan. NeurIPS 2024 (Spotlight)。https://arxiv.org/abs/2405.16158 — 强正则是 critic 网络放大的先决条件;model-free 近乎解决 Dog/Humanoid。
22. **Deep Reinforcement Learning that Matters** — Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger. AAAI 2018。https://arxiv.org/abs/1709.06560 — 种子/超参/实现导致的高方差与不可复现;呼吁规范报告与显著性检验。
23. **Deep Reinforcement Learning at the Edge of the Statistical Precipice** — Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. NeurIPS 2021 (Outstanding Paper)。https://arxiv.org/abs/2108.13264 — 少 run 点估计不可靠;建议 IQM、区间估计、performance profiles、stratified bootstrap(rliable)。

### 核实状态备注

- 全部 23 条的标题/作者/摘要主张均来自本次实际打开的 arXiv abs 页、ar5iv 正文、OpenReview 或 PMLR 页面。
- venue 标注为 preprint 的三条(#1 deadly triad、#8 SAC+AE、#18 MVP):arXiv 原始页面无正式 venue 记录,引用时按 arXiv preprint 处理。
- #15 ATC 的 ICML 2021 venue 未在本次打开的页面直接核实,引用时如需 venue 请再查 PMLR v139。
- 原文英文引句(SAC+AE §4.4、RLPD §4.2、ATC/PVR/MVP/Li et al. 摘要句)均逐字取自 ar5iv 正文或 arXiv 摘要页。
