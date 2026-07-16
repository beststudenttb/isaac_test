# 2026_07_16 DrQ-v2 像素 student，noise env（off-policy）

**noise env 上第一个 off-policy 满分级结果:离线 deterministic success 99.2%,运动干净(冲→刹→停,零抖)。** 与 `2026_07_14_drqv2_pixels`(干净 env)配对,是能和 SPR/MDP/JSRL 横比的 noise 线。

## 成绩

`best_offline.pt` = `drqv2_062400.pt`（frame 1,996,800），128 env × 1 episode 离线评估：

| 指标 | 值 |
| --- | ---: |
| success_rate | **0.9922** |
| fail_rate | 0.0000 |
| timeout_rate | 0.0078 |
| final_de_mean | 0.027 m |
| final_xe_mean | 2.82 px |
| mean_return | 9.57 |
| mean_len | 99 步 |

离线曲线（`offline_val.csv`，80 个 checkpoint）：早期全 0，σ≈0.18 起飞（noise 版膝盖比干净版靠前），中段 0.3–0.6 震荡（σ 下降期），frame 1.4M 后稳定 0.9+，收官 0.99。

## 配置

- `NoiseStudentEnv`（多一个蓝色干扰球），`STOP_N=1`，固定任务（end_d=1.5, end_x=0）。
- 32 env 训练、batch 128、`UPDATES_PER_TICK=16`、n-step 3、γ=0.99、τ=0.01、σ 调度 0.30→0.02。
- 224×224×3 单帧 + goal(2)；4 层 Conv32 encoder + trunk(50) + hidden 1024。
- checkpoint 命名已对齐 SPR 惯例：`drqv2_{tick:06d}.pt`（update 序号），dict 带 `update`/`step`。

## 一条干净的成功轨迹（best，env0）

```
   t   dist    px_x     a_x    a_w    reward
   1   2.75    27.0   1.000  1.000    1.00   ← 球在最左，满油门冲+转头找
  16   2.46   151.6   1.000  0.512          ← 已把球拉到中间
  36   1.70   105.7   1.000 -0.484    1.00   ← 满速接近，球锁在 ~106
  46   1.47   106.8  -0.047  →0             ← 到 1.5m 一步刹停，动作全归零
  56-70 1.47  107    ≈0     ≈0             ← 稳稳保持，无抖动
  70   1.47   107.6   ...            6.63   ← success
```

满油门冲 → 到 1.5m 一步刹住 → 动作归零稳住。终点 dist=1.47（差 3cm）、px=107.6（差 4px），70 步，零抖动。

## 对照:同一 env,off-policy vs on-policy+课程

| | 本次 DrQ-v2(off-policy) | `2026_07_16_spr_jsrl`(PPO+反向课程) |
| --- | ---: | ---: |
| 离线 success | **0.992** | 0.625 |
| final_de | 0.027m | 1.09m |
| final_xe | 2.8px | 20.6px |
| 运动 | 冲+刹+停,零抖 | 停不准、a_w 抖 |

**同一个 noise env,off-policy 干净利落 99%;on-policy 停在 62% 且抖。** 差别在算法:replay 把稀疏的 +10 沿接近轨迹传回去、把 critic(尤其 a_w 转头维)修对了;on-policy 在这个 reward 下 a_w 的 σ 收不下来(见 spr_jsrl readme)。

## 文件

| 文件 | 说明 |
| --- | --- |
| `best_offline.pt` | **eval-only**：`critic`/`critic_target` 已剥离(223→71.6MB,过 GitHub 100MB 上限)。只含 `encoder`+`actor`,可推理、不能续训。确定性动作与完整版逐位一致(已验证)。 |
| `config.py` / `offline_val_config.py` | 训练 / 评估配置快照 |
| `log.csv` | 训练日志 |
| `offline_val.csv` | 80 个 checkpoint 的离线成绩 |
| `best_offline_info.txt` | 最佳 checkpoint 指标 |
| `traj_best_offline.csv` / `traj_train_env0.csv` | 最佳离线轨迹 / 训练 env0 轨迹 |

## 注意

- val 修过一个 bug 才跑通:`val/drqv2.py` 里 `[INFO] checkpoint=... frame={frame}` 引用了接口重命名后不存在的 `frame`(改 `frame`→`update`/`step` 时漏改一处),第 1 个 checkpoint 写完就 NameError;Kit 的 fastShutdown 把 traceback 吃掉了,查了很久。已修为 `update`/`step`。
- 是 **noise env**,可与 `2026_07_16_spr_jsrl`、`2026_07_10_mdp_student_anneal`、`2026_07_13_spr_student_bc1` 横比;与 `2026_07_14_drqv2_pixels`(干净 env)不能直接横比。
