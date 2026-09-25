# PROJECT — isaac_test

> 长期不变的项目信息。当前状态见 `CURRENT_STATE.md`,当前计划见 `ACTIVE_PLAN.md`,
> 决策见 `DECISIONS.md`,实验史见 `RESEARCH_LOG.md`。本文件不放临时 TODO,不放未验证结论。
> 建立于 2026-09-25(会话迁移)。

## 项目目标

用 Isaac Lab 研究视觉强化学习中的**视觉表征**。整体路线是 learning-by-cheating:
teacher 在特权状态上训练,视觉 student 只看 RGB。

长期北极星(用户口述,仓库外):纯视觉的自主导航;当前的"找球并停在球前"任务是它的简化抽象。

## 当前研究问题(2026-09 起的主线)

**无人工标注、无特权信息,视觉表征能否只从一个 oracle teacher 的行为里学出来;
学出来的表征里到底有什么;不同行为信号(动作 A / 估值 V / 奖励 R)如何影响它。**

用户定的三层结构(`summary/2026_09_17_status.md` §1):
1. 表征能从行为里学出来 —— 前人做过,只作背景;
2. 学到的对行为有用 —— 前人做过,只作背景;
3. **表征里面是什么 + 各行为信号对表征学习的影响** —— 本工作的贡献。

## 硬约束(违反即偏题)

- **student 端**:只看 RGB,不得使用特权状态,不得有人工标注。teacher 端可以使用一切信息。
- 表征的监督只用 teacher/环境天然给得出的量:A(teacher 实际执行的动作)、V(teacher 的 V(s))、R(逐步奖励)。
- **sup**(回归 teacher 的手工观测)与 **init**(随机初始化、不训练)只作基线/参照,**不是研究对象**。
- 本篇**不提 Q / advantage**。
- 论文(SI2026)**只用 2 物体环境**;4 物体环境的结果只进海报,正文也不预告(用户 2026-09-24,见 `2026_09_24_cc_questions_for_paper.md`)。
- 不得在未经用户同意时启动 Isaac / isaaclab.sh(见 `.agent/REORIENT.md` 与各 agent 私有规则)。
- 共享协作规则见 `summary/agent_rule.md`;`AGENTS.md` 是 Codex 私有文件,其他 agent 不得读写。

## 非目标

- 不追求任务成功率本身(BC 已打平 teacher,PPO 在本研究里是**测量工具**,不是方法)。
- 不提出新方法;论文定位为分析类研究。
- 不做"自主涌现出新信息"的主张(在本任务上结构性不可证,见 `DECISIONS.md` D-005)。
- MDP / SPR / 世界模型 / off-policy 等 7–8 月的线**已停**,只作历史。

## 成功判据

论文层面:每一条关于"表征里有什么 / 信号如何影响表征"的主张,都能追溯到仓库内的产物,
并标出证据等级(对照实验 / 实验前预测 / 推测),且口径统一(见 `DECISIONS.md` D-003、D-008)。

## 系统结构(当前主线)

```
teacher   SB3 PPO,特权观测 (x_c, 直径)(obs_mask=diam),奖励 score_mode=angle
          train/teacher_score.py → models/rl/teacher_score_g4_diam_angle_s2/
   │ 只给行为(演示数据:scripts/collect_teacher_1m.py,带蓝色干扰球)
   ▼
表征      ResNet18(pretrained=False)+ FPN(128) + 7×7 网格池化 → 18816 → Linear → 256 → LayerNorm
          src/cv_extractor/free_spatial.py;配置 src/cv_extractor/config.py 的 FREE_SPATIAL_CONFIG
          donor 联合回归(编码器 + 头)拟合 A/V/R/sup/Ax/Aw/Asym:scripts/stage1_bc_heads.py
          → models/vision/g4_diam_angle/<arm>/donor.pt(= <arm>_donor/encoder_final.pt)
   │ 冻结
   ▼
PPO       自写 PPO 学 z → a:train/free_student.py(cfg: train/free_student_cfg.py)
          评估:val/score_student.py(64 起点,375 步,v1 判据)
任务      追红球 / 追蓝球(--chase-blue);环境 src/score_env.py、src/noise_env.py、src/env.py
          4 物体环境(仅海报):src/multi_env.py,--multi --target-slot
```

## 权威来源

| 内容 | 位置 |
|---|---|
| 源码 | `src/`、`train/`、`val/`、`scripts/` |
| 分析脚本 | `scripts/analysis/`(含 2026-09-25 从 job 临时目录归档的 37 个) |
| 分析原始输出 | `summary/results/*.txt`、`*.csv` |
| PPO 逐 ckpt 明细 | `summary/results/ppo_table.csv`(2 物体)、`w4_ppo_table.csv`(4 物体) |
| 批次日志与流水线脚本 | `summary/results/job_logs/`(`g4_state.log` 含全部 RESULT 原始行) |
| 模型与数据 | `models/`、`data_*/` —— **不进 git,只在服务器 irlab-s 上** |
| 逐日详报 | `summary/2026_09_14_interpretability_and_trajectory_report.md`(§1–§27) |
| 结果汇总 | `summary/2026_09_18_final_results.md`(2 物体)、`summary/2026_09_20_w4_results.md`(4 物体) |
| 论文问答 | `summary/2026_09_24_cc_questions_for_paper.md`(含服务器端逐条核实的回复) |
| 运行环境 | conda 环境 `isaac`(注意:`agent_rule.md` 写的是 `isaac_test`,与实际脚本不符,见 `CURRENT_STATE.md`) |
