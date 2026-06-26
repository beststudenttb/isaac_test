# MDP Student Update 421

保存内容：昨晚 MDP student 实验中当前可用的一组 checkpoint。

- `policy_update_000421.pt`：PPO student policy，来自 `models/rl/mdp_student/updates/update_000421.pt`。
- `mdp_state_initial.pt`：对应的 MDP state 模型，来自 `models/vision/mdp_state_priv_IDM_vicreg/last.pt`。该 policy 早于第一次在线 MDP update，所以使用初始 MDP。
- `config.py`：训练配置快照。
- `train_log.csv`：训练日志。
- `offline_val.csv`：离线评估日志。

离线评估中 `update=421` 的结果：

- `success_rate = 0.515625`
- `fail_rate = 0.0`
- `timeout_rate = 0.484375`
- `mean_return = 11.797098159790039`
- `mean_len = 278.265625`

未保存内容：

- `updates/` 中其他过程 checkpoint。
- `mdp_updates/` 中其他过程 checkpoint。
- 大体积轨迹 CSV。
