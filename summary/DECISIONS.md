# DECISIONS — 项目决策

> 只记录能找到依据的决策。理由无法从仓库恢复时写"理由不可考",不补编故事。建立于 2026-09-25。

## D-001 — 研究问题收窄为"行为监督表征里有什么"

Status: Active
Decision: 只用 teacher 行为信号(A/V/R)训视觉表征,冻结后 PPO;贡献放在第三层——表征里是什么、各信号如何影响表征。
Reason: 用户明确:前两层(能学出来、对行为有用)前人做过,只作背景。
Evidence: `summary/2026_09_17_status.md` §1;`summary/2026_09_18_final_results.md` 开头。
Alternatives considered: 7–8 月的 MDP/SPR 表征、off-policy 机制论文、表征与策略共适应。
Why not selected: 共适应的几条线实测崩溃(见 `summary/readme.md` 历史部分、`TODO.md` 各旧线);其余放弃理由见各自历史文档,未统一记录。
Revisit if: 用户改变论文定位。

## D-002 — 主实验只用一个 teacher:`g4_diam_angle`(seed 2)

Status: Active
Decision: teacher 固定为 obs_mask=diam(相机坐标)+ score_mode=angle(物理奖励),取两个种子中回报更高的 s2。
Reason: 2×2(teacher 观测坐标 × 奖励度量)两根轴在表征与策略层面都测不出差别;臂的差异占主导。
Evidence: 报告 §9.5、§18;`job_logs/g4_state.log`("[diam_angle s1] v1=100.0 回报=273.9"、"s2 … 276.2")。
Alternatives considered: 其余三个配置(diam_cam、ang_cam、ang_angle)。
Revisit if: 需要跨 teacher 重复来回答 U1(可复用已有的 2×2 表征)。

## D-003 — PPO 成绩口径:最后 6 个检查点取最大

Status: Active(表述需更正,见 CURRENT_STATE 不一致 1)
Decision: 每条 PPO 取 u200–u240 与 last 共 6 个检查点的最大 v1 成功率;只排除 t=0 就看不见目标的起点。
Reason: 单看 last 的极差可达 98 点,不可用;看到目标后跟丢算失败(用户 2026-09-11 定)。
Evidence: `ppo_table.csv` 重算;记忆 `lost-ball-starts-not-a-concern`。
Revisit if: 需要与 4 物体批次(不含 last)统一口径。

## D-004 — sup 与 init 只作基线;不提 Q / advantage

Status: Active
Decision: sup(回归 teacher 手工观测,用了特权信息)与 init(随机不训)只作参照;论文不讨论 Q/advantage。
Reason: 用户要求;sup 违反"无特权信息",不是合法 student。
Evidence: `2026_09_17_status.md` §1。

## D-005 — 主张降为第二档("选择性"),不主张"自主涌现"

Status: Active
Decision: 论文只写"行为信号决定表征保留多少(可预测);在可用线索中表征采用了与 teacher 不同的一条"。
Reason: 第三档(涌现出新信息)撑不住:init 已能 60% 解码颜色;本任务图像内容是 (x,d) 的确定函数。
Evidence: `summary/TODO.md`「论文主张的档位」;`2026_09_24_cc_questions_for_paper.md`。
Revisit if: H4 被证实(选择来自架构先验)——届时"采用不同线索"的措辞还要再收紧。

## D-006 — 论文只用 2 物体环境;4 物体结果只进海报

Status: Active
Decision: 正文只用 2 物体环境;4 物体的结论("颜色是门、形状是选择器""A/V/R 三档""丢的是非目标")不写也不预告。
Reason: 用户 2026-09-24 决定。
Evidence: `2026_09_24_cc_questions_for_paper.md` 开头。

## D-007 — 允许加机制拆分臂 Ax / Aw / Asym

Status: Active(推翻了此前"不再扩展监督信号")
Decision: 只改 student 回归什么(a_x、a_w、(a_x,|a_w|)),环境、teacher、数据全不动。
Reason: 这是分开"信号类型"与"目标维数"两种解释的唯一办法,且与 A、R 严格可比。
Evidence: `src/stage1_helpers.py` ARM_DIMS 注释;`TODO.md` 删除线条目;`mech_probe_out.txt`。
Supersedes: "不再扩展监督信号"(`TODO.md` 删除线)。

## D-008 — 线性探针统一用 λ=3e-2

Status: Experimental(服务器端建议,本地论文是否采纳未知,U6)
Decision: 论文中所有线性可读性用 λ=3e-2 那一组;方法节写明 λ。
Reason: 同一批 z 两种 λ 数字不同;3e-2 那组由三条独立代码路径复现,且与表 1 的 MLP 列、机制表同源。
Evidence: `2026_09_24_cc_questions_for_paper.md` 回复 A2、B4。
Revisit if: 本地草稿选择了另一组——则全文统一到那一组。

## D-009 — PPO 在本研究中是测量工具,不是方法

Status: Active
Decision: 用冻结表征上的 PPO 成绩衡量"表征能否支撑某任务",不主张 RL 提升性能。
Reason: BC 直接打平 teacher(283.0 vs 276.2)。
Evidence: `job_logs/g4_state.log` [BC] 行;报告 §10。

## D-010 — 不经用户同意不启动 Isaac

Status: Active
Decision: 任何 isaaclab.sh / 仿真器启动都要先得到用户明确同意;取证优先用静态方法。
Reason: 用户多次要求。
Evidence: 各 agent 私有规则;`summary/agent_rule.md` §5。

## D-011 — 服务器端研究会话与 mutmuas 秘书会话分离

Status: Active
Decision: 2026-09-24 起,本仓库只由"研究会话"负责;mutmuas(多 agent 通信框架,仓库在 `~/mutmuas-claude`)由单独的秘书会话负责。
Reason: 用户决定,避免研究上下文被秘书工作挤占。
Evidence: 用户在会话中的指令(仓库内无更早记录)。
