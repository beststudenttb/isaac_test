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

说明：

- 2 维 teacher/student 使用旧观测 `[px, dist]`。
- 4 维 teacher/student 使用 goal 观测 `[px, dist, end_d, end_x]`。
- `516 -> 518`、`160 -> 162` 是因为后续视觉/MDP student 把 `end_d, end_x` 拼进了策略输入。
- 旧包里如果 readme 标注“精确文件现场已找不到”，只能保证保留了同代可用依赖，不能保证 teacher checkpoint 和原训练时逐字节一致。
- 本目录保存模型依赖和配置快照；如果 reward/env 源码之后继续变化，复现历史实验时需要结合当时 commit 或 readme 说明判断。
