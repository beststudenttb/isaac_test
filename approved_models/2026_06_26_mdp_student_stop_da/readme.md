# 2026_06_26_mdp_student_stop_da

MDP student PPO result saved after changing the stop-stage reward idea.

Saved files:
- `best_offline.pt`: best model selected by offline validation.
- `last.pt`: final model from the training run.
- `mdp_state.pt`: pretrained MDP state model used by the student policy.
- `deps/teacher_ppo/best_val.zip`: teacher model used by teacher loss.
- `traj_train_env0.csv`: env0 full training trajectory.
- `traj_best_offline.csv`: full trajectory for the best offline validation model.
- `traj_offline_env0.csv`: env0 offline validation trajectory.
- `traj_val.csv`, `val.csv`, `offline_val.csv`, `log.csv`: evaluation and training logs.
- `config.py`, `offline_val_config.py`, `best_offline_info.txt`: run metadata.

Not saved:
- `updates/` process checkpoints.
- tensorboard event files.
