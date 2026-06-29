# 2026_06_27_weekend_students

周末批量训练结果，来自 `scripts/run_weekend_students.sh` 对照实验。

## 内容

- `old_xd_student/`: `old` CV 模型的 `xd` 输出作为视觉 state。
- `old_shared_student/`: `old` CV 模型的 `shared` 中间特征作为视觉 state。
- `mdp_student/`: 预训练 MDP latent 作为视觉 state。
- `deps/`: 复现本批结果所需的依赖模型。

每个目录保存：

- `config.py`
- `best_offline.pt`
- `best_offline_info.txt`
- `last.pt`
- `log.csv`
- `offline_val.csv`
- `offline_val_config.py`
- 关键轨迹 csv

未保存 `updates/` 和 `tb/`，避免保存大量过程模型和 TensorBoard 文件。

## 依赖模型

- `deps/teacher_ppo/best_val.zip`: student 训练时使用的 teacher。
- `deps/cv_old/best.pt`: `old_xd_student` 和 `old_shared_student` 使用的视觉预训练模型。
- `deps/mdp_state_priv_probe_grad/last.pt`: `mdp_student` 使用的 MDP 预训练模型。

恢复到默认运行路径：

```bash
mkdir -p models/rl/teacher_ppo models/vision/cv_old models/vision/mdp_state_priv_probe_grad
cp approved_models/2026_06_27_weekend_students/deps/teacher_ppo/best_val.zip models/rl/teacher_ppo/best_val.zip
cp approved_models/2026_06_27_weekend_students/deps/cv_old/best.pt models/vision/cv_old/best.pt
cp approved_models/2026_06_27_weekend_students/deps/mdp_state_priv_probe_grad/last.pt models/vision/mdp_state_priv_probe_grad/last.pt
```

## Best 结果

- `old_xd_student`: update 570, success 0.78125.
- `old_shared_student`: update 190, success 1.0.
- `mdp_student`: update 1010, success 0.59375.

## 当前判断

`old_shared_student` 是这一批中表现最好的视觉 student；`old_xd_student` 能学到但 stop 附近更容易受回归误差影响；`mdp_student` 暂时不如人工视觉表征。
