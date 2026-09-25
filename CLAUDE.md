# CLAUDE.md — isaac_test(Claude 私有规则;Codex 不读本文件,Claude 不读 AGENTS.md)

这是地图与操作规则,不是项目百科。项目知识在 `summary/` 的结构化文档里。

## 会话开始

执行 `summary/REORIENT.md`:依次读 `summary/agent_rule.md` → `PROJECT.md` → `CURRENT_STATE.md` →
`ACTIVE_PLAN.md` → `HANDOFF.md`,再看 `git status / diff / log`,核对 HANDOFF 引用的产物,
向用户汇报重建结果后才动手。

## 证据纪律

- 仓库源码、配置、`summary/results/` 里的产物 > agent 写的总结 > 对话回忆。
- 计划 ≠ 结果;假设 ≠ 事实;改了 ≠ 验证了。每条要进论文的数字都要能指到文件。
- 口径固定:PPO 取最后 6 个检查点(含 last)最大值;线性探针 λ=3e-2(见 `summary/DECISIONS.md`)。

## 范围纪律

- 当前唯一主线与硬约束见 `summary/PROJECT.md`;不悄悄改目标、不加监督臂、不提 Q/advantage。
- 4 物体环境只进海报,不进论文。
- 本会话只负责研究;mutmuas 事务由秘书会话负责(`summary/DECISIONS.md` D-011)。

## 文档纪律

- 认识变了 → `CURRENT_STATE.md`;持久决策 → `DECISIONS.md`;实验/调查 → `RESEARCH_LOG.md`(追加);
  里程碑变了 → `ACTIVE_PLAN.md`;结束实质性工作前 → `HANDOFF.md`。
- 不改写历史文档正文;过时内容在文件头加注。
- 证据与文档矛盾时写进 `CURRENT_STATE.md` 的 IMPORTANT INCONSISTENCIES,不要悄悄择一。

## 操作底线

- **不经用户同意不启动 Isaac / isaaclab.sh。**
- 杀进程先单独查 PID 再 kill 数字,不用 `pkill -f`;CPU 分析用 `CUDA_VISIBLE_DEVICES=""`;一张卡同时只跑一个 Isaac。
- 用户让"看/分析"时只给判断,不顺手改代码或提交。
- conda 环境实际是 `isaac`。提交正文 200–300 字(`agent_rule.md` §2);证据文件用 `git add -f`。
