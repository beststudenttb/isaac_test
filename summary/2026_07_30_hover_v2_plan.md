# 任务 v2「悬停 + 部署切断」重做计划 2026-07-30

一句话:把"学会停"改成"学会悬停"。训练期只给状态分、episode 强制跑满 15s(无提前终止),
停止这个离散事件移出 RL,交给部署/评估侧的切断规则。这是**任务重定义**,不是调参——
所以要重做的是"以后论文里引用数字的那几条 RL 线",v1 的结果不作废,降级为机制分析素材。

## 0. 为什么改(对着 v1 的三个病根)

| v1 病根(已实证) | v2 怎么拆 |
|---|---|
| 冷启动死锁:`success=in_zone&a<0.05` 硬阈值,актор 采不到低动作 → HER 无原料 | 没有阈值事件。悬停每步吃 `stop_q` 高斯分,梯度处处存在;区内待住物理上就要小动作,小动作从"被考核的事件"变成"拿分的手段" |
| Q 飞(coadapt/pixels q1→45-50,真实上界≈11):燃料 = `R_STOP_IN=+1/R_STOP_OUT=-2` 事件配对被 HER 打散 + 带锚样本仅 0.35% | 事件配对奖励全删。`r(s,goal)` 光滑,HER relabel 重算数学精确;悬停满程的高 Q 是真实回报不是幻觉 |
| 区内低动作样本缺失(coadapt stop_frac 峰 0.0019) | 无提前终止 → 悬停策略每条 375 步灌大量"区内+小动作"样本 |

## 1. v2 任务定义

**训练**(reward v2,fixed 375 步):
- 保留(goal 无关,与 v1 一字不动):`~seen` 搜索项(-K_SEARCH·(ax²+ay²)+K_SEARCH_W·|aw|)、
  `seen` 的 -K_TIME、R_FIND=+1 / R_LOST=-2、telescoping 接近 shaping(K_X·x_gain+K_D·d_gain,
  势函数差,HER 安全)。
- **新增主项:`R_HOVER * stop_q`,每步、只要 seen 就给**(不 gate 在区内、不 gate 在动作上)。
  `stop_q = exp(-0.5(xe/SIG_X)² - 0.5(de/SIG_D)²)`,现有高斯原样复用,R_HOVER=1.0。
- **删除:R_STOP_IN/OUT、R_SUCCESS、R_FAIL、R_STOP/K_STOP_DA(本来就是 0)。**
- **无 terminated**:success 机制不触发、out() 不终止(出界后 stop_q≈0 自然是惩罚)。
  全部 episode 都是 375 步 timeout(truncated,带 γ bootstrap——奖励自洽后 TD 有稳定不动点,
  这正是 pendulum 不需要终止锚也不飞的原因)。
- `apply_actions` 里 a<0.05 清零移动的逻辑保留(与部署切断行为一致,训练部署同 dynamics)。

**评估/部署**(切断 wrapper,env 保持 v1 判据):
- val 的 env 用 v1 模式(terminated/success 机制原样),外面套 wrapper:
  **per-env 连续在区 N 步(建议 N=3)→ 之后动作强制 0** → env 自己的
  `in_zone & a<eps → success` 触发。
- 这样 val 的 success_rate 与 v1 的 88%/92% **同判据可比**(最终都是"区内+零动作"),
  只是达成方式(策略自发 vs 规则切断)不同,论文里明写。
- val 另加悬停指标:末 100 步在区比例(zone dwell)、final_de/final_xe 照旧。

## 2. 代码改动清单

| 文件 | 改动 |
|---|---|
| `src/task_cfg.py` | 加 `R_HOVER=1.0`(v1 常量全保留不动) |
| `src/sb3_env.py` | `compute_reward`/`compute_terminated` 按 cfg 开关走 v2 分支(cfg 加 `REWARD_MODE = "v1"/"hover"`,默认 v1,正在跑的线零影响) |
| `src/rl/task_reward.py` | 加 v2 版 recompute(比 v1 简单:无事件配对、无 terminated,`new_term` 恒 False) |
| `src/rl/her_buffer.py` | relabel 走 v2 recompute 时 discount 恒 γ(读 cfg 开关) |
| `val/sac.py`(或独立 `val/sac_hover.py`) | 切断 wrapper:in-zone streak≥N → 动作置 0;zone-dwell 指标 |
| `train/sac_hover_cfg.py` 等 | 每条线一个 cfg(继承现有,只改 REWARD_MODE + OUT_DIR) |
| 离线测试 | reward v2 与 env 双实现逐项 parity;HER relabel v2 精确性;fixed-length buffer 语义 |

不动:SAC/DrQ agent 本体、buffer 布局、pipeline sh 结构(每条线复制一个 sh 换 cfg)。

## 3. 重做范围裁定("所有实验重做"其实是这些)

**必须重做(v2 出数字的线)**:
- M1 `spr_z` SAC+HER v2(冻结 z)——锚点探针
- M2 `pixels` v2(可训 CNN)
- M3 `spr_coadapt` v2(SPR 与 RL 同步)——核心研究问题:光滑奖励下可训表征还崩不崩
- M4 赢家的 random_stop 泛化 + 3 seed(mean±std)

**可选(有价值再跑)**:
- DDPG(arm A 原配方)v2:如果 DDPG 在 v2 也能行,"死锁是 DDPG 唯一的死因"这条故事就闭环
- SAC 无 HER v2 消融:光滑奖励下 HER 可能不再必要,单变量验证

**不重做(照旧有效)**:
- SPR encoder 预训练(`stage1_sigma_ablation.pt`,任务无关)、CV extractor、teacher 线
- v1 全部结果:spr_z 88%、coadapt/pixels 负结果 + Q 发散分析 → 论文的"机制诊断"章节素材,
  v1/v2 对比本身就是叙事(为什么必须重定义任务)

## 4. 时间与排期(按实测吞吐)

| 实验 | 配置 | train | val | 参考 |
|---|---|---|---|---|
| M1 spr_z v2 2M | 128 env | ~2h | ~1.2h(80ckpt) | sac_spr 实测 |
| M2 pixels v2 2M | 64 env | ~3.5h | ~0.7h | sac_pixels 实测 |
| M3 coadapt v2 2M | 64 env(与另一线并行)| ~26h | ~1.2h | sac_coadapt 实测 0.18s/update |
| M3' coadapt v2 2M | 128 env(独占卡,RAM 77GB)| ~13h | ~1.2h | DDPG coadapt 实测 |
| M4 seeds×3 + random_stop | spr_z | ~10h 合计 | | |

卡上现状:GPU0 pixels_8m 约今晚 17:20 训完 + val 到 ~20:00;GPU1 coadapt v1 约 21:15 训完 + val 到 ~22:30。
**今晚两条 v1 收尾数据照收**(pixels 8M val + coadapt v1 val,进 v1 机制故事),v2 从今晚腾出的卡开始。

## 5. 门控顺序(省 GPU 的关键)

- **G0(今天,CPU)**:代码 + 离线测试全过(reward parity / HER v2 / 无终止语义)。
- **G1**:smoke(小 env 短跑走全链路)。启 sim 由用户执行。
- **G2(M1 spr_z v2,~3.5h 全含)**:判据 = zone dwell 末 100 步 >0.8、final_xe<10px、
  wrapper success ≥ 0.88(不应低于 v1 同表征)。**过不了先怀疑 SIG_X=5px 太尖,调宽重跑 M1,
  不动其他线**——奖励设计问题必须在最便宜的线上解决。
- **G3**:M2 + M3 两卡并行。核心读数:coadapt v2 的 q1 是否回到真实回报量级
  (悬停满程 ≈ stop_q 累计,量级可算)、z_std 走势、critic_loss 相对 v1 的 6.46 降多少。
- **G4**:M4 定稿数字。

## 6. 风险与预设答案

- **SIG_X=5px 对冷启动太尖** → 远场有 telescoping shaping 接力,理论上够;不够就加宽 SIG 或
  加第二个宽高斯,G2 上便宜地试。
- **悬停≠低动作**(bang-bang 穿区刷分)→ 穿区平均 stop_q 远低于驻留,回报上就亏;
  若实测仍震荡,加 -K_A·a_mag² 的动作正则(一项,HER 安全)。
- **无终止后 fail 样本消失,策略学会贴着球乱蹭** → out() 区域 stop_q≈0 已是惩罚;
  真出问题再把 out 终止加回(B 方案,带 r_fail 锚)。
- **v2 的 Q 量级大**(悬停满程 γ 折扣和 ≈ 60-80)→ 这是真实回报,critic 拟合没问题;
  判发散看"q1 是否超过悬停满程理论值",不再是"是否超过 11"。

## 7. 待用户拍板

1. M3 coadapt v2 用 64 env(可并行)还是 128 env(独占快一倍)——取决于想不想和 M2 同时跑。
2. 可选项(DDPG v2 / 无 HER 消融)做不做、什么优先级。
3. 切断 wrapper 的 N(建议 3)。
