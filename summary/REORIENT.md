# REORIENT — 新会话接手流程

> 放在 `summary/` 而不是 `.agent/`:`.gitignore` 第 15 行有意忽略 `.agent/`,而本流程必须随仓库保存。

你在接手一个长期运行的研究项目。**不要依赖任何之前的对话记录**,也不要先改代码。先重建状态。

## 1. 读(按顺序)

1. `summary/agent_rule.md`(共享规则);Claude 读 `CLAUDE.md`。**不要读 `AGENTS.md`**(Codex 私有)。
2. `summary/PROJECT.md`
3. `summary/CURRENT_STATE.md` —— 先读"口径"一节
4. `summary/ACTIVE_PLAN.md`
5. `summary/HANDOFF.md`

## 2. 查仓库

```bash
git fetch origin && git status && git log --oneline -10 && git log --oneline HEAD..origin/main
git diff --stat            # 非空时再看具体 diff
ls summary/ | tail -20     # 看是否有本地 CC 新提交的问题文件(例如 2026_09_xx_cc_questions_*.md)
```

## 3. 核对 HANDOFF 的说法

- 打开 HANDOFF 引用的文件,确认它们存在且内容与描述一致。
- 运行 HANDOFF 里的"验证命令"。
- 按需只读 `DECISIONS.md`、`RESEARCH_LOG.md` 的相关条目;详细数字在 `summary/results/` 与
  `summary/2026_09_14_interpretability_and_trajectory_report.md`。

## 4. 向用户汇报后再行动

用中文说明:
- 项目目标、当前里程碑、当前任务;
- 已验证的状态、仍不确定的点;
- 上一个会话声称完成了什么,仓库证据是否支持;
- 你打算做的**下一步具体动作**。

HANDOFF 与仓库证据不一致时,以仓库与产物为准,但必须把差异报告出来。
重建结果内部不一致时,不要继续推进。

## 5. 工作纪律

- 证据优先级:源码/配置 > 实验产物/日志 > 可复现命令 > git 历史 > 用户文档 > 已记录的决策 > 会话说法 > 自己的回忆。
- 计划不是结果,假设不是事实,改了代码不等于修好了。
- 不要悄悄改变项目目标;扩大范围先更新 `ACTIVE_PLAN.md`。
- 重要结果改变认识 → 更新 `CURRENT_STATE.md`;持久决策 → `DECISIONS.md`;实验/调查 → `RESEARCH_LOG.md`;
  里程碑变化 → `ACTIVE_PLAN.md`。
- 结束一段实质性工作前:核对仓库状态,更新上述文档与 `HANDOFF.md`。
- 发现证据与文档矛盾:不要悄悄选一个,要写进 `CURRENT_STATE.md` 的 IMPORTANT INCONSISTENCIES 并调查。
- 不经用户同意不启动 Isaac;证据类 `*.txt/*.log/*.png` 被 `.gitignore` 挡住,入库要 `git add -f`(仓库惯例)。
