# TODO

## 当前线:行为监督表征(2026-09,唯一在跑的线)

状态见 [`2026_09_17_status.md`](2026_09_17_status.md)。

- [x] A/V/R/sup/init 五个表征 × 红蓝两任务 × 三种子的 PPO(A、R、sup 已 n=3;V 的 seed3 在跑;init n=2)
- [x] 受控渲染三批:析因(4736)、身份(5376)、形状+距离线索(5376)
- [x] 测量:读出器阶梯、CCGP、S1/S2 区分测试、存在性二分类、可访问性曲线、
      逐层探针、权重 ΔW、Wang 六性质、有效维数、策略雅可比、混淆测试、颜色选择性曲线
- [x] 换干扰物的配对评估(红/绿干扰 × 5 表征)
- [x] 空间贡献热图(7×7,梯度×输入)
- [x] 机制拆分臂 Asym/Aw/Ax(只改 student 回归什么,环境/teacher/数据不动)+ 对应 PPO
- [ ] `scripts/render_distract.py`:目标恒为红球、干扰槽换 15 种属性 —— 已写好,等空卡
- [ ] 把 15 种干扰扩到热图与配对评估
- [x] **周末 4 物体批次** —— 结论见 [`2026_09_20_w4_results.md`](2026_09_20_w4_results.md)(24 条 PPO **0 失败**)
      原计划与执行细节::见 [`2026_09_19_weekend_plan.md`](2026_09_19_weekend_plan.md)(已预登记预测)。
      **2026-09-18 21:58 已启动**。代码:新增 `src/multi_env.py`(200 行,不动 env/noise_env/score_env),
      `collect_teacher_1m.py` / `train/free_student.py` / `val/score_student.py` 各加 `--multi --target-slot`
      (纯增量,不传 `--multi` 时行为逐行不变)。teacher 复用 `teacher_score_g4_diam_angle_s2`,理由见计划文档。
      调度:`$CLAUDE_JOB_DIR/tmp/w4.sh` —— **优先级队列 + 两卡 worker 动态领活**,
      一条失败只标 failed 不阻塞其余;队列 `w4/queue.txt` 随时可追加;状态 `w4/state.txt`。
- [ ] 论文(**用户本地写**,本仓库只交数据与图)

**明确不做**(边界):
- 不提 Q / advantage
- teacher 配置不再动(`obs_mask=diam` + `score_mode=angle`)。2×2 消融已证空:臂 91–100%、teacher 输入 0–5%、奖励度量 0–3%
- sup / init 只当基线与参照,**不是研究对象**;A/V/R 才是
- 报数口径:**best-of-last-6 ckpt**(last 不可用,极差可达 98 分);只排除 t=0 就瞎的起点

~~不再扩展监督信号~~ —— 2026-09-18 已推翻:加了 Asym/Aw/Ax 三个机制拆分臂,
因为它们**只改 student 回归什么,不碰环境/teacher/数据**,与 A、R 完全可比,
而且正是它们把"动作 vs 奖励"和"维数"两种解释同时打死的。
~~多干扰物不进本篇~~ —— 2026-09-18 已推翻:周末 4 物体批次进本篇,见上。

---

## 论文主张的档位(2026-09-18 定:**降档**)

结论的说法分三档,**这一篇只写第二档**。

| 档 | 说法 | 撑不撑得住 |
|---|---|---|
| 三档 | 表征**自主涌现**出监督信号中不存在的新信息 | ❌ init 的 8 类颜色解码就有 60.2% —— 颜色信息先验里本来就有,"涌现"一刀被砍倒 |
| **二档** | **在多条可用线索中,表征选了一条当判据 —— 而且选的不是 teacher 用的那条** | ✅ 每句话背后有表 |
| 一档 | 表征编码了任务相关量 | ✅ 别人做过,只当背景 |

**落点句**:

> 行为信号决定表征**保留多少** —— 这部分可预测、可控;但在**"用哪条线索"**上,表征自己做了选择,而且选的不是 teacher 用的那条。

- **可控半边**:Aw 臂(只回归 `a_w`)把距离从监督目标里摘掉 → 策略无法停车(`de = 4.597 m`、`|a| = 0.490`);
  Ax 臂保留距离 → 红球 84.4%。六臂"目标可提取性 ↔ 干扰残留"严格单调,零例外
  (0.965/0.964/0.901/0.857/0.710/0.455 → 0.559/0.560/0.428/0.177/−0.079/−0.116)。
- **选择性半边**:teacher 用**视觉大小**编码距离(`obs_mask=diam`,输入里明写直径),
  student 用**图像高度**(抬高 0.6 m → 读数偏 3.2–4.8 m;缩放半径几乎不动)。
  目标身份的判据是**色相**,不是形状(红方 0.996、红锥 0.986)也不是亮度(暗红 0.993),
  **而"身份"在监督信号里从未被定义**(teacher 观测 = (x_c, 直径);奖励 = `1/(1+|bearing|/8.5+|range−1.5|/0.5)`,全是坐标)。
- **选择性是训出来的,不是先验**:init 在彩色档上的调谐是平的(0.51–0.61),训练后是窄红滤波。

降档换来的:**不用再打"这是不是唯一解"那场仗** —— 选择性只需要展示"可选的有几条、它选了哪条",
不需要论证"它本来可以不选这条"。那场仗打不赢,现在也不用打。
降档丢掉的:"涌现"这个词,和冲着这个词来的读者。留给下一篇。

**"视觉大小 / 图像高度"是教科书上的两条单目深度线索**(relative size / height in visual field),
所以这条可以对着已有术语讲"线索选择",不是自造词。
另:用户 2026-09-04 就预登记过"直径能否代替 d"这个问题 —— 答案是 student 两条都没照抄,选了 teacher 没用的那条。

---

## 未来线:避障 / 搜寻 / 找回(尚未开工,先把约束定死)

### 硬约束 —— 每加一个东西,拿这条卡一遍

> **监督信号只编"在哪 / 代价多少",绝不编"是谁 / 属于哪一类"。**

- **矮墙**:把**碰撞代价**写进 reward(teacher 因此会绕),**不**给任何"这是墙 / 这是障碍"的标签、通道或额外观测维
- **候选位置**:teacher 知道目标**坐标**,**不**告诉 student 底座长什么样 —— 让它自己解出"带底座的地方是候选位"
- **目标物**:同上,只给坐标,不给身份

理由:按第二档的判据,"一个量在不在监督信号里"是**可判定**的,不用争。
而**每加一个东西让复杂任务能跑,就往"可控"栏加一笔;不故意留槽,"选择性 / 自发"栏永远加不了新条目**。
本篇仅有的两个选择性实例(身份 = 红、距离 = 图像高度)都是因为**监督只给了坐标、没给身份**才长出来的 ——
这个结构必须在复杂任务里原样保留,否则未来那篇的第二个卖点会被自己一步步喂没。

代价是实打实的:teacher 更难训、任务更难成。**认这个代价。**

### 设计要点(用户 2026-09-18 口述)

- 目标**离散化**:固定 3–4 个候选位置随机刷新;候选位可做底座之类的可识别标记(但标记的**语义**不进监督,见上)
- 单帧不够 → 考虑 GRU,或输入端加东西(双目 / 深度)。
  **记忆放在策略端,不放进编码器** —— 编码器保持"看到什么",记忆是"去过哪"
- **不解码墙**。判据是行为:如果表征真的知道墙,那么在某个子任务里,**相同的墙应该会被避开**
- 不一定要"知道这是墙",到**"障碍物这个大类"**就够;大类 / 子类要不要预分类,到时候再定
- teacher 可以吃一切信息(当作人工筛选的 oracle);**student 端仍然是无人工标注、无特权信息**
- A/V/R 研究清楚后,可以作为**联合信号**一起喂表征;后续发现别的有用信号也可以加

### 还没做的文献检索

- **"只用行为监督的 RGB 表征,能否在单帧里解码出障碍物"** —— 这条窄检索还没做

---

## 悬而未决(用户明确说"先留个疑问",不要自行销掉)

- **MSE 是否逼出过度精确?** 用户一开始就不愿用 MSE。担心 MSE 为了把监督目标读准,会把表征做得过分尖锐,
  而用户要的是"模糊表征 + 精准策略"。暂按现方案跑,**这个疑问不销**。
- **第二个创新点**(模糊表征 + 足以匹配它的精准策略)整个押在未来那条线上,本篇不碰。
- **走到那一步要重新 grill 一次**(用户 2026-09-18 原话)。


## 代码瘦身

### off-policy（arm A 族，07-27 已完成）
5 个脚本(1547 行) → `train/offpolicy.py` + `val/offpolicy.py`(543 行) + `src/rl/{agent,buffer,env,logging,spr}.py`。两份 DrQV2Agent/两份 buffer 合一，helper 模块化，CLI `--cfg` 传 cfg 模块(cfg 内容不动，只改 import 方式)。**arm B(drqv2/drqv2_agent/drqv2_buffer/train+val drqv2)按决定保留未动** —— 代价是它的 csv 字段(`frame`/无 `sample_fps`/`z_std`)和 offpolicy 不对齐；哪天要对齐就得动 arm B。

### on-policy（⚠️ 回到这条线前先瘦身，别直接加实验）
env 是**统一的 IsaacLab 继承链** `BallEnv → BallPPOEnv(sb3_env.py) → MDPStudentEnv → NoiseStudentEnv`，off-policy 和 on-policy 主力共用它(`sb3_env` 这名字误导——它是 IsaacLab DirectRLEnv，不是 gym)。按范式分三族：
- **SB3 库(teacher)**：`stable_baselines3.PPO` + `make_sb3` gym wrapper。**唯一 env 无法对齐的**(SB3 库要 gym VecEnv 接口)。→ 单独摘到 `train/sb3/` 子目录，别和自写 PPO 混。
- **自写 PPO(student / student_vision / cnn_student / spr_student / spr_bc / spr_jsrl / spr_phased / mdp_student)**：env 已齐(IsaacLab batched tensor)，但 loop 范式**真不同**(纯 PPO / +teacher BC / +JSRL 反向课程 / phased 分阶段)——是 4 个不同 seam，别硬合成 shallow dispatcher。且大半是死线(JSRL 判停、phased 探路、mdp/cnn/student 早期)。
- **表征预训练(mdp_state / cv)**：无 env/PPO，纯监督离线数据，另一类。

**建议顺序**：① teacher(SB3)单独摘出去 ② 归档死线(spr_jsrl/spr_phased/mdp_student/cnn_student/student/student_vision 早期)移出主干，这是最大瘦身 ③ 活线只剩 spr_student/spr_bc，别为几条线做大统一。env 继承链本身是好的 deep 结构(层层加语义)，不要动。理由详见与 CC 讨论(2026-07-27，codebase-design skill)。

## 世界模型线（新重点，07-17 起）

- **在冻结/共学习 SPR latent 上起 Dreamer / TD-MPC 骨架**：model-based off-policy = 样本效率天花板，直指 Regime B（实机、少样本、逐次适应）。**用现成 repo（dreamerv3 / SheepRL）接 Isaac 向量化 env，别从零手写**（几天工作量，非过夜盲挂）。见记忆 `research-direction-and-strategy`。
- 定位：把"感知（表征）+ 决策（策略）"统一进一个学习对象（world model 正是这一体）。这是 Master 论文的天花板线；off-policy 机制论文是地板。

## SPR 共适应线（spr_coadapt，07-24 起，正在跑）

- **正在 GPU0 跑**：`train/spr_coadapt.py --num-envs 128`，2M 步约 12 小时，输出 `models/rl/spr_coadapt/`。把原文 SPR 范式搬到 off-policy：critic 的 TD 梯度 + SPR 自监督 loss 联合更新同一 encoder（从头联合，ImageNet backbone）。**判据是稳定性不是成功率**。
- **盯三个探针**：`z_std`（塌向 0 = 表征塌缩）、`spr_loss`、`q1_mean`。截至 07-24 18:27（step 0.5M）：z_std 0.014→0.204（没塌 ✅）、spr_loss 1.99→0.29（在学 ✅）、**`q1_mean` −0.03→20.5（⚠️ 同任务 arm A 稳定在 ~4.3，`R_SUCCESS` 只有 10，疑 Q 高估）**。若 q1 继续单调爬且 critic_loss 同步涨 → Q 发散，旋钮是把 `ENC_LR_RATIO` 从 0.1 再压小。
- 卖点定位：**不 novel 在"联合训练"**（on-policy 的 `src/spr_ppo.py` 本就是原文范式），novel 在**离策略下这个联合环稳不稳**。
- 遗留：`BATCH_SIZE` 因显存降到 64（256 实测 OOM），replay ratio 只有 8；`UPDATES_PER_TICK` 不影响峰值显存，要补采样效率可提到 32/64。

## off-policy 线（DrQ-v2，地板）

- **`K_SEARCH_W → 0` 的搜索效率实验（最便宜、假设最明确）**：±60° 后盲区搜索仍是原地摇头（换向 49 次/集、72 步 vs 理论最优 39 步）。假设是 `K_SEARCH_W=0.01` 让摇头成了正收益解；古老 `drq_env/task_cfg.py` 没这一项且能学会单向转。
- **补一条 pixels 4M / upt64**：让 arm A（4M/64）vs arm B（2M/16）的对照完全对齐。现在结论已稳（arm A 2M 的 76.6% vs pixels 28% 也压倒性），但要无可挑剔就得补。
- **arm A 多 seed（优先，便宜）**：冻结同一个 z（`_encoders/stage1_sigma_ablation.pt`）跑 seed 2/3，把 n=1 的 100% 变成 n=3 mean±std。注意现状 seed 写死 cfg、`CLEAR_OUT_DIR=True`，多 seed 需先加 SEED/OUT_DIR 覆盖 + 独立目录，否则清掉已有结果。
- ~~arm A 归档~~：已完成（07-22/07-23 三次 spr_z 跑已归档到 `approved_models/_release/`，见 `summary/handoff_2026_07_24.md`）。

## JSRL（on-policy，已判定停止微调 → 归档负结果）

- **结论（07-17）**：初始化修复后 σ 病搬到 a_x/a_w（精度维顶 clamp）、过冲崩点、课程控制器 bang-bang；h=60 给 293 update 仍卡 0.66。**课程+初始化治不了精确刹停/对准，停止微调。** 等本次 val 跑完归档为负结果。
- 可选收尾：固定 h=90/105 + 禁 ladder 探针（一行 cfg），把"控制器太快 vs σ 病墙"分离干净，让负结果投稿无懈可击。
- on-policy 回归**真扳机**（只有这些值得回）：不靠 replay 把稀疏终止信用送到精度维——奖励重分配/return decomposition、per-dim σ 控制、刹车段 dense 辅助信号。
- **暂放**：球全随机生成 + 找不到球快速旋转（teacher 没给旋转奖励/全局生成阶段），先把前方停车做稳。

## 采集与数据验证

- 确认手动修改后的 `assets/robots/ball_robot.usd` 中 eye 俯仰角、相机挂载结构和碰撞设置是否符合后续训练需求。
- 继续用肉眼抽查数据集图片质量，并确认 `px_x`、`dist` 标签和真实图像读取结果之间的误差范围。
- 用 `balanced + DLSS performance` 重新采集数据后，确认 CV 训练和视觉 student 推理时的图像分布一致。
- 抽查 `data_isaac_noise/` 中蓝球干扰样本，确认红球标签不被蓝球遮挡或过近干扰；重点看 `old+xd` 和 `old+shared` 是否会把蓝球误当目标。

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
- 对比干净环境与带干扰环境下的 `old+xd`、`old+shared`，判断人工几何输出和 shared feature 在干扰球存在时的鲁棒性差异。
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
- 新的带干扰视觉实验如果保存到 `approved_models/`，必须同时保存 noisy CV checkpoint、student policy、teacher 依赖、配置和关键 val/train 轨迹。

## 多 agent 协作（暂停）

- plan / state 通信机制暂不启用；后续如果确实需要双 agent 并行，再重新设计并定稿。

## 阶段二：联合训练（开放问题）

- teacher PPO 的观测/动作/奖励定义（需读原 Webots env 对齐，尚未读取）。
- N、M、λ、ε 的具体取值（实验调）。
- (B) RL 梯度进 encoder 是否纳入，取决于 (A)+KL锚 的效果。
- 判断 SPR 是否需要 IDM、k-step 或 value/reward 辅助；每次只改一个主要变量，避免无法归因。
