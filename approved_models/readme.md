# Approved Models

这里保存已经确认值得保留的模型及其说明文件。`models/` 是训练现场输出目录，不入库；只有从 `models/` 中挑选出来的结果才放到这里。

每个模型单独一个文件夹，命名为 `yyyy_mm_dd`。若同一天需要保留多个模型，可以在日期后加简短后缀。

推荐结构：

```text
approved_models/
  readme.md
  yyyy_mm_dd/
    readme.md
    config.py
    model.pt
    policy.zip
    trajectory.csv
```

每个模型文件夹中的 `readme.md` 用来说明模型用途、训练数据、主要指标和人工判断；`config.py` 用来保存当次训练参数；`.pt` / `.zip` 保存模型权重；CSV 保存轨迹或评估结果。

## 兼容性记录

本目录跨越多次 reward、观测和模型结构修改，不能只按文件名混用模型。当前已确认：

| 目录 | 类型 | 策略输入 | teacher / 视觉依赖 |
| --- | --- | ---: | --- |
| `2026_06_16_teacher_ppo` | teacher | 2 | 无 |
| `2026_06_17_student_ppo_no_teacher` | 特权 student | 2 | 无 teacher loss |
| `2026_06_17_student_ppo_teacher_64env_val5_u62` | 特权 student | 2 | `deps/teacher_ppo/policy.zip`，旧 2 维 teacher；原精确 `best_val.zip` 现场已找不到 |
| `2026_06_17_student_ppo_teacher_bc` | 特权 student | 2 | `deps/teacher_ppo/policy.zip`，旧 2 维 teacher；原精确 `best_val.zip` 现场已找不到 |
| `2026_06_18_student_vision_old_xd_student` | 视觉 student | 2 | `deps/cv_old/best.pt` + 旧 2 维 teacher；原精确 teacher `best_val.zip` 现场已找不到 |
| `2026_06_22_student_vision_old_state_compare/old_xd` | 视觉 student | 2 | `deps/cv_old/best.pt`，无 teacher loss |
| `2026_06_22_student_vision_old_state_compare/old_feature` | 视觉 student | 128 | `deps/cv_old/best.pt`，无 teacher loss |
| `2026_06_22_student_vision_old_state_compare/old_shared` | 视觉 student | 516 | `deps/cv_old/best.pt`，无 teacher loss |
| `2026_06_25_mdp_student_update421` | MDP student | 160 | `mdp_state_initial.pt` + `deps/teacher_ppo/best_val.zip`，4 维 goal teacher |
| `2026_06_26_mdp_student_stop_da` | MDP student | 160 | `mdp_state.pt` + `deps/teacher_ppo/best_val.zip`，4 维 goal teacher |
| `2026_06_27_weekend_students/old_xd_student` | 视觉 student | 4 | `deps/cv_old/best.pt` + `deps/teacher_ppo/best_val.zip` |
| `2026_06_27_weekend_students/old_shared_student` | 视觉 student | 518 | `deps/cv_old/best.pt` + `deps/teacher_ppo/best_val.zip` |
| `2026_06_27_weekend_students/mdp_student` | MDP student | 162 | `deps/mdp_state_priv_probe_grad/last.pt` + `deps/teacher_ppo/best_val.zip` |
| `2026_06_29_cv_old` | 视觉预训练 | - | 只保存 `cv_old`，不是 RL policy |
| `2026_07_08_mdp_student_noise` | MDP student | 162 | `deps/mdp_state_priv_probe_grad/last.pt` + `deps/teacher_ppo/best_val.zip`，4 维 goal teacher；noise env |
| `2026_07_10_mdp_student_anneal` | MDP student | 162 | 同上；teacher 退火（250 保持 / 150 线性退到 0）|
| `2026_07_10_spr_student_center` | SPR student | 130 | `stage1_end.pt` + `teacher_best_val.zip`；`spr_loss_k` 去中心化；noise env |
| `2026_07_13_spr_student_bc1` | SPR student | 130 | 同上；`TEACHER_LOSS=1`（反证实验，成绩差，保留是为了结论）|
| `2026_07_14_spr_teacher5` | SPR student | 130 | `stage1_end.pt` + `teacher_best_val.zip`；`TEACHER_LOSS=5`/`STD_MAX=0.1`（反面结果：BC 段 100%，teacher 一撤塌到 0%）|
| `2026_07_14_drqv2_pixels` | DrQ-v2 像素 student（off-policy） | 224×224×3 + 2 | **无**：无特权信息、无 teacher、无 BC；干净 env（非 noise）|

说明：

- 2 维 teacher/student 使用旧观测 `[px, dist]`。
- 4 维 teacher/student 使用 goal 观测 `[px, dist, end_d, end_x]`。
- `516 -> 518`、`160 -> 162` 是因为后续视觉/MDP student 把 `end_d, end_x` 拼进了策略输入。
- 旧包里如果 readme 标注“精确文件现场已找不到”，只能保证保留了同代可用依赖，不能保证 teacher checkpoint 和原训练时逐字节一致。
- 本目录保存模型依赖和配置快照；如果 reward/env 源码之后继续变化，复现历史实验时需要结合当时 commit 或 readme 说明判断。
- `mdp_state_priv_probe_grad` 在 `2026_06_27_weekend_students` 与 `2026_07_08_mdp_student_noise` / `2026_07_10_mdp_student_anneal` 中是**两个不同的 checkpoint**（前者 md5 `1e63006c...`，后两者 `1a177e1d...`），各自目录里的副本才是该实验真正使用的那份，不可互换。
- `2026_07_08_mdp_student_noise` 与 `2026_07_10_mdp_student_anneal` 是同 seed、同 encoder、同 env 的一组对照，唯一变量是 teacher 退火；前 250 个 update 的日志几乎逐行重合。
- SPR student 的 `best_offline.pt` 是 `ppo_*` 类型，**只含策略**，必须配同目录 `stage1_end.pt` 的 SPR/encoder 使用；MDP student 的 `best_offline.pt` 则配 `deps/` 下的 encoder。
- 三条 SPR 线的 `stage1_end.pt` **互不相同**（`2026_07_10_spr_student_center` = `e9464932...`、`2026_07_13_spr_student_bc1` = `ef6d8647...`、`2026_07_14_spr_teacher5` = `696811c7...`），每次 stage1 都重训过；跨实验对比时 encoder 不是单变量。
- `2026_07_13_spr_student_bc1` 与 `2026_07_10_mdp_student_anneal` 的 `TEACHER_LOSS` 都是 1.0，是目前 SPR/MDP 两条线在 BC 权重上唯一对齐的一组：BC-only 段两者离线成绩都是 ~0%，PPO 接手后 MDP 涨到 96%、SPR 塌到 0%。
- `2026_07_14_drqv2_pixels` 的 `best_offline.pt` 是 **eval-only**（`critic`/`critic_target` 已剥离以过 GitHub 100 MB 单文件上限），可推理、不能续训；完整 checkpoint 只在训练现场 `models/` 里。
- **σ 是目前唯一能划分成功/失败的变量**：成功要求三维动作同时 `|a_i| < 0.05`，发现概率 `≈ [2Φ(0.05/σ)−1]³`。两条成功的线（MDP anneal 自由衰减到 0.033、DrQ-v2 调度到 0.10）都在 **σ ≈ 0.12–0.13** 处 success 起飞；两条失败的 SPR PPO 线的 σ 分别被钉死在 0.299 和 0.100（都顶着 `STD_MAX` clamp，梯度还想往上）。
- reward 常数在 06-30（commit `ac1604b`）改过：`R_STOP` 从 3.0 → **0.0**（stop 区内的稠密奖励被关闭），只剩终止时的 `R_SUCCESS * stop_q`。`2026_06_27_weekend_students` 及更早的线跑的是 `R_STOP=3.0`；`2026_07_08` 之后的所有线（含 MDP anneal、SPR、DrQ-v2）跑的是 `R_STOP=0.0`。对比历史实验时必须注意这条。
