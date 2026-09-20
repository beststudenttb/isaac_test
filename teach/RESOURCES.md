# 像素输入 RL 与表征学习:资源

全部条目在 2026-07-30 逐条打开过原始页面核实(arXiv abs / ar5iv 正文 / OpenReview / PMLR)。
完整 23 篇的核实版清单在 `../summary/2026_07_30_pixel_rl_representation_survey.md`;
这里只留**最锋利的几篇**,按"什么时候该翻它"标注。

## Knowledge

### 先读这几篇(每条路线的代表)

- [Stabilizing Off-Policy Deep RL from Pixels — Cetin et al., ICML 2022](https://arxiv.org/abs/2207.00986)
  提出 "visual deadly triad" 与 catastrophic self-overfitting;并论证**图像增广之所以有效,本质是
  对 encoder 梯度做平滑正则**。什么时候翻:想搞懂"为什么裸着用 TD 梯度训 CNN 会崩",以及
  "增广到底在干什么"。**这是本主题第一优先的一篇。**

- [Decoupling Representation Learning from Reinforcement Learning — Stooke et al., 2021](https://arxiv.org/abs/2009.08319)
  纯无监督目标训 encoder(RL 梯度不回传),效果匹配或超过端到端;含"预训练后冻结"设置。
  什么时候翻:要为"冻结表征"路线找最直接的背书。(venue 通常引 ICML 2021 / PMLR v139,
  但本次未从原始页面核实到,引用前请再查。)

- [Does Self-supervised Learning Really Improve RL from Pixels? — Li et al., NeurIPS 2022](https://arxiv.org/abs/2206.05266)
  对 "TD + 自监督联合训练" 路线的**否定性系统评估**:同等数据与增广下,相对纯增广基线没有
  有意义的增益。什么时候翻:解释 spr_coadapt 这类联合训练线为什么最脆弱。

- [The Unsurprising Effectiveness of Pre-Trained Vision Models for Control — Parisi et al., ICML 2022](https://arxiv.org/abs/2203.03580)
  域外数据预训练的**冻结**表征可媲美甚至超过真值状态特征。什么时候翻:要论证"控制任务
  根本不需要端到端训视觉"。注意确切标题是 "Unsurprising"(不是 "(Un)Surprising")。

### 机制层(想知道"为什么会坏"时翻)

- [DR3: Value-Based Deep RL Requires Explicit Regularization — Kumar et al., ICLR 2022](https://arxiv.org/abs/2112.04716)
  TD 下 SGD 的隐式正则**从有益变有害**,把 Bellman 两端的 (s,a) 表征推向共适应/塌缩。
  什么时候翻:要机制性地说明"TD 梯度自带把表征学歪的压力"。

- [Understanding and Preventing Capacity Loss in RL — Lyle, Rowland & Dabney, ICLR 2022](https://arxiv.org/abs/2204.09560)
  网络在一串非平稳目标上训练会丧失拟合新目标的能力,**稀疏奖励下最严重**。
  什么时候翻:解释"稀疏任务上表征先塌了,等奖励信号出现已经学不动"。

- [Efficient Online RL with Offline Data (RLPD) — Ball et al., ICML 2023](https://arxiv.org/abs/2302.02948)
  §4.2 有一整节 "Layer Normalization Mitigates Catastrophic Overestimation":LayerNorm 让
  Q 值被权重范数界住,压住对数据外动作的外推。什么时候翻:想压 Q 发散时的第一个旋钮。

- [The Primacy Bias in Deep RL — Nikishin et al., ICML 2022](https://arxiv.org/abs/2205.07802)
  早期烂数据形成的表征会锁死后续学习;周期性重置部分网络有稳定增益。
  什么时候翻:怀疑"前期学歪了后期拉不回来"时。

### 评估规范(报数字之前翻)

- [Deep RL at the Edge of the Statistical Precipice — Agarwal et al., NeurIPS 2021 (Outstanding Paper)](https://arxiv.org/abs/2108.13264)
  少量 run 下点估计不可靠;建议 IQM、区间估计、performance profiles、stratified bootstrap。
  什么时候翻:要报"我这条线 88%"之前——尤其在 run 间方差大的时候。

## Wisdom(社区)

- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
  英文,信噪比在 reddit 里算高,常有作者本人出没。用于:把自己的诊断贴出来求打脸
  (比如"我认为 Q 飞是因为缺终止锚",看有没有人指出别的解释)。

- 学会/研究会(日本,他在读的路径上顺便积累人脉):
  日本ロボット学会(RSJ)、人工知能学会(JSAI)、計測自動制御学会(SICE) 的年次大会与研究会。
  用于:把负结果做成口头发表拿真实反馈——负结果在这些场合是可以讲的。

## Gaps

- **社区这块需要补**:上面只列了我有把握确实存在的。更对口的(比如具体的 RL Discord、
  某个 pixel-RL / robot-learning reading group、松尾研系的公开勉強会)我没有逐个核实过链接,
  下次可以专门搜一轮再补进来。
- 还缺一份"从症状查机制"的速查表(症状 → 文献里的名字 → 首选修法)。计划做成 reference 文档。
