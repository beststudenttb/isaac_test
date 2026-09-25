# HANDOFF

Last updated: 2026-09-25(会话迁移,由服务器端"研究会话"写)
Current branch: main
Current commit: 本次迁移提交(`git log -1` 标题含"会话迁移");迁移前为 `23849b3`
Working tree state: 迁移提交后应为干净;服务器上 `models/`、`data_*/` 不进 git

## Current milestone

支撑 SI2026 论文(只用 2 物体环境),见 `ACTIVE_PLAN.md`。

## Exact current task

**没有进行中的任务。** 最后一件事是回复本地 CC 的 20 条论文问题(`2026_09_24_cc_questions_for_paper.md`,
提交 `23849b3`,已推送)。之后等待本地 CC 的下一批问题。

## Verified state

- 服务器侧无实验在跑,两张 GPU 空闲(2026-09-25 检查)。
- 2 物体主结果、机制表、各测量的原始产物都在仓库:`summary/results/`、`ppo_table.csv`、`job_logs/`。
- 09-24 回复里的核实结论与证据等级已写入 `CURRENT_STATE.md`。

## Work completed in this session(迁移)

- 新建:`summary/PROJECT.md`、`CURRENT_STATE.md`、`DECISIONS.md`、`RESEARCH_LOG.md`、`ACTIVE_PLAN.md`、
  `REORIENT.md`、本文件;根目录 `CLAUDE.md`。
- 归档(只复制,未改):job 临时目录中 37 个分析脚本 → `scripts/analysis/`;20 份输出 → `summary/results/`;
  流水线脚本与状态日志 → `summary/results/job_logs/`;由 `g4_state.log` 整理出 `summary/results/shape_swap.csv`。
- 在 5 份旧文档文件头加了过时注(正文未改);`agent_rule.md` §0 加一行指向 `REORIENT.md`。
- 记忆目录(`~/.claude/projects/-home-tb-Downloads-isaac-test/memory/`)重组索引、新增 `project-state-lives-in-repo`、
  给 3 条过时记忆加注;未删除任何文件。
- **未**修改任何实现代码;**未**跑任何实验。

## In progress

无。

## Failed / rejected attempts

见 `CURRENT_STATE.md` 的 INVALIDATED 表(本会话历史中被推翻的说法)。

## Hypotheses still being considered

`CURRENT_STATE.md` H1–H4。对论文措辞影响最大的是 **H4**(距离用画面高度,可能来自架构先验)。

## Unknown / needs verification

- 本地论文草稿是否已采纳 09-24 回复里的 11 条更正(服务器看不到草稿)。
- 三格缺失检查点评估(U4)。
- 截止日期只在对话中出现(U5)。

## Known inconsistencies

`CURRENT_STATE.md` 的 IMPORTANT INCONSISTENCIES 1–8(口径、λ、可提取性定义、旧文档里的 last 格数字、conda 环境名)。

## Exact next action

1. `git fetch origin && git log --oneline HEAD..origin/main`,看本地 CC 是否推了新的问题文件。
2. 若有:只读核对后在该文件末尾回复,每条引用仓库产物路径;只提交该文件并 push。
3. 若没有:向用户报告"无待办",并询问 `ACTIVE_PLAN.md` 可选任务 2(补三格评估,需启动 Isaac)与 3(保存 S3/S4 计算脚本)是否要做。

## Verification command / procedure

```bash
cd ~/Downloads/isaac_test && git status --short | wc -l        # 期望 0
python3 -c "
import csv,collections
d=collections.defaultdict(dict)
for r in csv.DictReader(open('summary/results/ppo_table.csv')): d[(r['arm'],r['task'])][r['seed']+r['ckpt']]=float(r['v1_success_pct'])
import itertools
for a in ('A_donor','R_donor'):
  for t in ('red','blue'):
    s=collections.defaultdict(float)
    for k,v in d[(a,t)].items(): s[k[0]]=max(s[k[0]],v)
    print(a,t,[s[x] for x in sorted(s)],round(sum(s.values())/len(s),1))"
# 期望:A red 95.8、A blue 10.4、R red 97.9、R blue 83.3(与 CURRENT_STATE V1 一致)
```

## Relevant files

- `summary/2026_09_24_cc_questions_for_paper.md`(最近一次问答,含全部核实细节)
- `summary/results/ppo_table.csv`、`mech_probe_out.txt`、`mlpladder_out.txt`、`shape_out.txt`、`vzone_out.txt`

## Relevant durable context

- 硬约束与论文范围:`PROJECT.md`「硬约束」
- 口径与不一致:`CURRENT_STATE.md`「口径」「IMPORTANT INCONSISTENCIES」
- λ 与检查点决策:`DECISIONS.md` D-003、D-008
- 可做与不可做:`ACTIVE_PLAN.md`「当前任务」「不要做」
