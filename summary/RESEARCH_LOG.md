# RESEARCH_LOG — 实验与调查索引

> 追加式。详细记录在各自文档里,这里只做索引与**当前**状态标注;不改写历史文档的原文。
> 状态:VERIFIED / PARTIAL / INCONCLUSIVE / FAILED / SUPERSEDED。建立于 2026-09-25,之前的条目为回溯整理。

## 2026-06 ~ 2026-08 / 早期线(MDP、SPR、off-policy、JSRL、世界模型)
Question: 自监督/联合训练的视觉表征能否支撑 PPO / off-policy student。
Artifacts: `summary/2026_06_*_ubuntu.md` … `2026_07_30_status.md`、`handoff_2026_07_17.md`、`handoff_2026_07_24.md`、`2026_08_03_framework_onepager.md`。
Status: SUPERSEDED —— 全部已停。结论只作历史,不作当前依据。

## 2026-09-04 ~ 09-10 / 任务 v3(稠密位置分)与冻结表征 PPO
Question: 在新奖励下,行为监督表征 + 纯 PPO 能否学会找球停车。
Artifacts: `2026_09_06_score_task_results.md`、`2026_09_10_noy_batch_results.md`。
Status: VERIFIED(作为装置搭建);具体数字属于旧配置,不进论文。

## 2026-09-12 / 周末批次:五种粒度的 teacher → A/V/R 表征 → PPO
Artifacts: `2026_09_12_weekend_plan.md`;记忆 `weekend-2026-09-12-results`。
Result at the time: R > A > V;探针阶梯饱和,不预测 PPO。
Status: PARTIAL —— 配置与主实验不同,仅作背景。

## 2026-09-14 ~ 09-15 / 2×2(teacher 观测坐标 × 奖励度量)
Question: teacher 的输入与奖励度量会不会影响表征。
Result: 两根轴都为空,臂的差异占主导;R 94.1% 对 sup 50.8%(n=4 配置)。
Artifacts: 报告 §9;`mechanism_out.txt`(四配置);`job_logs/g4.sh`。
Status: VERIFIED。

## 2026-09-15 ~ 09-18 / 主实验:追红球 vs 追蓝球(A/V/R/sup/init × 3 种子)
Question: 把被擦除的量(干扰球)变成新目标,各表征还能否支撑任务。
Result: 见 `CURRENT_STATE.md` V1;擦除没有换来原任务优势。
Artifacts: `2026_09_18_final_results.md`;`ppo_table.csv`;`job_logs/g4_worker.sh`、`g4_state.log`。
Status: VERIFIED(PPO n=3,表征 n=1)。

## 2026-09-16 ~ 09-17 / 表征测量(析因、CCGP、S1/S2、身份、形状、颜色、存在性、归因)
Artifacts: 报告 §14–§26;`summary/results/*.txt`;`scripts/analysis/`;`figs/`。
Status: VERIFIED(各项见 CURRENT_STATE V6–V11);归因热图 FAILED(`figs/README.md`)。
Caveat: 部分数字用 λ=1e-2、部分 3e-2(CURRENT_STATE 不一致 2)。

## 2026-09-17 ~ 09-18 / 换干扰物配对评估与形状对照
Artifacts: `distractor_swap.csv`、`shape_swap.csv`(2026-09-25 从 `g4_state.log` 整理)、`job_logs/distract_eval.sh`、`shape_eval.sh`。
Status: VERIFIED。"A 绿球崩溃"的早期解读已推翻。

## 2026-09-18 / 机制拆分臂 Ax / Aw / Asym
Question: 擦除由信号类型、目标维数还是可提取性决定。
Result: 可提取性与残留六臂单调;Ax/Aw 的 PPO 与事先写下的预测一致。
Artifacts: `mech_probe_out.txt`、`scripts/analysis/mech_probe.py`、`job_logs/mech_split.sh`、`mech_ppo.sh`;报告 §27。
Status: VERIFIED(测量);因果解释为 HYPOTHESIS(CURRENT_STATE H1、H2)。

## 2026-09-19 ~ 09-20 / 4 物体环境批次
Question: 拆开"丢弃干扰物"与"丢弃非红物体"。
Result: 见 `2026_09_20_w4_results.md`;事先写下的判决句(A 追红方 ≥80%)未触发。
Artifacts: `2026_09_19_weekend_plan.md`(预登记)、`w4_ppo_table.csv`、`w4_probe_raw.txt`、`job_logs/w4.sh`、`w4_state.log`。
Status: VERIFIED;**仅海报**(D-006)。

## 2026-09-24 / 论文问答核实(无新实验)
Question: 本地 CC 写 SI2026 原稿时的 20 条待确认问题。
What was actually run: 只读核对源码与产物;另在 CPU 上重算 λ 对比、读 LayerNorm γ/β、算 V 目标斜率(未存脚本)。
Result: 发现口径"含 last"、λ 敏感、可提取性定义不一致、init 已偏好画面高度等,见该文件回复。
Artifacts: `2026_09_24_cc_questions_for_paper.md`。
Status: VERIFIED(核对);S3、S4 两项的计算未保存为脚本。

## 2026-09-25 / 会话迁移
What was done: 按证据重建状态;把只存在于 job 临时目录的分析脚本(37)、原始输出(20)、批次日志与流水线脚本(22)归档进仓库;建立本套文档。
Status: 见 `.agent/HANDOFF.md`。
