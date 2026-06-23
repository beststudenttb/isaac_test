# MDP 视觉表征 当前流程图

> 反映 `src/mdp_state.py` + `train/mdp_state.py` 当前结构（自监督转移表征版）。
> ★ = stop-gradient（detach）。所有共享模块（encoder / core / prior_head）在 observe 和 imagine 里复用，不是多份。

## 图 1：模型前向（observe 真实步 + imagine 预测步）

```mermaid
flowchart TB
  classDef perc fill:#d2e3fc,stroke:#1967d2,color:#111;
  classDef core fill:#feefc3,stroke:#e8a000,color:#111;
  classDef state fill:#ceead6,stroke:#1e8e3e,stroke-width:3px,color:#111;
  classDef head fill:#e6f4ea,stroke:#34a853,color:#111;
  classDef io fill:#f1f3f4,stroke:#777,color:#111;

  o(["o_t 图像"]):::io
  prev(["prev: h_t-1, z_t-1, a_t-1"]):::io

  subgraph ENC["感知 encoder（共享）"]
    res["ResNet18 stem+layer1+layer2 (冻结)<br/>→ layer3 (可训, train_layer3)"]:::perc
    attn["coord + attention 热图 → 加权/全局池化"]:::perc
    ihead["img_head → z_img (32维)"]:::perc
    res --> attn --> ihead
  end

  zimg(["z_img_t"]):::perc

  core["core: GRUCell([z_t-1, a_t-1], h_t-1)"]:::core
  belief(["belief h_t"]):::core
  prior["prior_head"]:::core
  post["post_head([h_t, z_img_t])"]:::core
  priorz(["prior_z_t"]):::core
  postz(["post_z_t = z_t"]):::core

  state["STATE s_t = [ h_t , post_z_t ]<br/>(160维, 喂策略)"]:::state

  value["value_head(s_t)"]:::head
  probe["probe_head(s_t.detach ★)<br/>→ px / dist (仅诊断)"]:::head

  %% observe
  o --> res
  ihead --> zimg
  prev --> core --> belief
  belief --> prior --> priorz
  belief --> post
  zimg --> post --> postz
  belief --> state
  postz --> state
  state --> value
  state -. "detach ★" .-> probe

  %% imagine（复用同一 core / prior）
  nbelief["core([post_z_t, a_t], h_t) → h_t+1"]:::core
  nzpred(["next_z_pred = prior_head(h_t+1)"]):::core
  reward["reward_head([s_t, a_t])"]:::head
  done["done_head([s_t, a_t])"]:::head
  at(["动作 a_t"]):::io

  state -- "imagine: s_t, a_t" --> nbelief --> nzpred
  state --> reward
  state --> done
  at --> nbelief
  at --> reward
  at --> done
```

## 图 2：训练损失与梯度（按时序 BPTT，★=目标 detach）

```mermaid
flowchart TB
  classDef shape fill:#d2e3fc,stroke:#1967d2,color:#111;
  classDef loss fill:#fff0f0,stroke:#c5221f,stroke-width:2px,color:#111;
  classDef diag fill:#f3e8fd,stroke:#8430ce,color:#111;
  classDef off fill:#eceff1,stroke:#90a4ae,color:#555;

  zimg_t(["z_img_t"]):::shape
  zimg_n(["z_img_t+1"]):::shape
  nzpred(["next_z_pred_unit"]):::shape
  postn(["post_z_t+1_unit"]):::shape
  state_t(["s_t"]):::shape
  at(["a_t"]):::shape

  %% 塑造表征的损失
  dyn["L_dyn = smooth_l1( next_z_pred_unit , post_z_t+1_unit ★ )<br/>w=DYN_W"]:::loss
  con["L_contrast = InfoNCE( pred vs 真next, 批内负样本 )<br/>跨 B×seq 池化, τ=TAU, w=CONTRAST_W"]:::loss
  idm["L_idm = smooth_l1( inverse([z_img_t, z_img_t+1]) , a_t )<br/>仅用 z_img(无动作泄漏), w=IDM_W"]:::loss
  val["L_value = smooth_l1( value(s_t) , return_t )<br/>w=VALUE_W"]:::loss
  done["L_done = BCE / L_reward<br/>w=DONE_W / REWARD_W"]:::off

  nzpred --> dyn
  postn -. "★" .-> dyn
  nzpred --> con
  postn -. "★ key" .-> con
  zimg_t --> idm
  zimg_n --> idm
  at --> idm
  state_t --> val

  %% 诊断（不回传到表征）
  probe["L_probe = smooth_l1( probe(s_t.detach ★) , [px,dist] )<br/>只训探针头, w=PROBE_W"]:::diag
  ret["retrieval@1 (来自 contrast 矩阵)"]:::diag
  erank["effective_rank(z) / z_std"]:::diag
  state_t -. "detach ★" .-> probe
  con -.-> ret
```

## 当前权重（建议起点，Preset A：纯自监督）

| 损失 | 权重 | 作用 | 塑造表征? |
|---|---|---|---|
| L_contrast | CONTRAST_W=1.0 (τ=0.1) | 防坍缩 + 状态可区分（主引擎） | 是 |
| L_idm | IDM_W=1.0 | 可控因子（bearing/距离变化），忽略噪声 | 是 |
| L_dyn | DYN_W=1.0（可设0，被 contrast 包含） | 转移一致（正样本拉近） | 是(小) |
| L_value | VALUE_W=0.0 | return 信息 | 关 |
| L_done / L_reward | 0.0 | 终止/奖励 | 关 |
| L_probe | PROBE_W=0.1 | px/dist **诊断**（detach，只训头） | 否 |

## 关键点
1. **IDM 用 z_img**（纯感知、无动作）：避免从 belief 直接读出 a_t 的平凡解。
2. **dyn / contrast 目标 detach**：防止 encoder 把自己训成“可预测的常数”而坍缩。
3. **probe 输入 detach**：只训探针头、不 ground 表征 → 是合法的“线性探针”诊断，不污染自监督。
4. **防坍缩靠 contrast（不是方差铰链）**：方差铰链只防常数坍缩、且与 dyn 互拽震荡；contrast 同时防常数/维度坍缩且稳定。
5. **判读顺序**：retrieval@1 / eff_rank 先确认没坍缩 → idm loss 是否下降 → 最后看(事后冻结的)probe_d 能否到 0.2m。
