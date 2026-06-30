# 2026_06_30_random_stop_students

随机 stop 目标任务下保存的两组 student 结果。

## 内容

- `old_shared_student/`: `old` CV 模型的 `shared_feature` 作为视觉 state 的 PPO student。
- `mdp_student/`: 预训练 MDP latent 作为视觉 state 的 PPO student。
- `deps/`: 复现所需依赖模型。

本目录未保存 `updates/` 和 TensorBoard 文件，只保存 best/last 模型、配置、日志和关键 trajectory。

## 结果

- `old_shared_student`: best update 1850, offline val success 0.9375, timeout 0.0625。
- `mdp_student`: best update 500, offline val success 0.59375, timeout 0.40625。

注意：保存时当前 reward 已改为 success 使用 `R_SUCCESS * stop_q`，且 `R_STOP = 0.0`。本目录中的模型来自修改前已经跑出的结果；复现旧结果时以目录内 `config.py` 和当前 Git 历史/说明为准。

## 恢复依赖到默认路径

```bash
mkdir -p models/rl/teacher_ppo_random_stop models/vision/cv_old models/vision/mdp_state_priv_probe_grad
cp approved_models/2026_06_30_random_stop_students/deps/teacher_ppo_random_stop/best_val.zip models/rl/teacher_ppo_random_stop/best_val.zip
cp approved_models/2026_06_30_random_stop_students/deps/cv_old/best.pt models/vision/cv_old/best.pt
cp approved_models/2026_06_30_random_stop_students/deps/mdp_state_priv_probe_grad/last.pt models/vision/mdp_state_priv_probe_grad/last.pt
```
