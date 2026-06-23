# MDP-Consistent Visual Representation for Object-Centric Mobile Robot Docking and Retrieval

## 0. Purpose

This document summarizes the current research direction and implementation plan for a visual robot control project.  
It is intended as a handoff note for Codex / Claude Code.

The current task is:

```text
Robot searches for a ball → approaches the ball → stops at a proper docking distance.
```

The future task may extend to:

```text
Search → approach → dock/stop → grasp with arm → carry/return.
```

The current simulation platform has been moved to Isaac / Isaac Lab. PPO training is already working at an acceptable level. The current research focus is no longer only policy learning, but **learning a visual MDP-like representation** that can be used as the state input for the policy.

---

## 1. Core Research Question

The key question is:

> Can we learn a compact visual representation `h_t = Encoder(o_t)` that behaves like a controllable MDP state?

This representation should be:

```text
Markov-like
action-conditioned
transition-consistent
control-relevant
object-centric
transferable from docking to retrieval
```

For the current ball docking task, `h_t` should preserve information such as:

```text
ball visibility
relative distance
relative bearing
lateral offset
approach velocity
stop feasibility
effect of robot action
```

For future retrieval with a manipulator, the representation may need to extend to:

```text
ball relative pose to mobile base
ball relative pose to gripper
arm configuration
grasp affordance
object possession
return-goal relation
```

---

## 2. Proposed Overall Pipeline

The preferred pipeline is:

```text
1. Train / use privileged teacher policy.
2. Collect many transitions:
   (o_t, a_t, o_{t+1}, s_t*, r_t, done_t)
3. Pretrain an MDP-consistent visual representation.
4. Use h_t = Encoder(o_t) as state input for the policy.
5. Warm-start policy using LBC / imitation learning.
6. Jointly fine-tune:
   - policy updates at high frequency
   - representation updates at low frequency
```

Where:

```text
o_t     : RGB image / visual observation
a_t     : continuous robot action, e.g. [v_x, v_y, omega]
s_t*    : privileged simulator state, only for teacher/training/evaluation
h_t     : learned visual latent state
r_t     : reward
done_t  : episode terminal flag
```

The key design is:

```text
Teacher / privileged policy
        ↓
Collect transition dataset
        ↓
MDP-like representation pretraining
        ↓
h_t = Encoder(o_t)
        ↓
LBC / imitation warm-up policy
        ↓
PPO / RL fine-tuning with two-timescale updates
```

---

## 3. Why Not Pure Reward-Driven Representation?

Directly connecting the encoder to PPO and letting reward train the representation is possible:

```text
image → encoder → latent h → PPO policy → reward
```

But it is not the best main direction for this task.

Problems:

```text
1. Reward is task-specific and may not preserve reusable state structure.
2. PPO on-policy data distribution changes over training.
3. Encoder may learn shortcut features, e.g. ball size or center only.
4. Representation learned for docking may not transfer well to grasp/retrieval.
5. High-frequency encoder-policy co-training may cause latent drift.
```

Therefore, the recommended main direction is:

```text
Representation and policy are first trained separately.
Representation is trained using state-action-transition consistency.
Policy is trained using LBC / imitation learning, then fine-tuned with PPO.
```

Reward-driven representation can still be used as a baseline or auxiliary loss.

---

## 4. Model Components

### 4.1 Encoder

```text
h_t = f_theta(o_t)
```

Input:

```text
RGB image or stacked frames
```

Output:

```text
latent state h_t
```

The latent should encode control-relevant object-centric state, not generic image reconstruction features.

---

### 4.2 Forward Latent Transition Model

```text
h_hat_{t+1} = T_psi(h_t, a_t)
```

Loss:

```math
L_forward = || T_psi(h_t, a_t) - stop_gradient(h_{t+1}) ||^2
```

Purpose:

```text
Ensure h_t contains information needed to predict the next visual state under action a_t.
```

---

### 4.3 Inverse Dynamics Model

```text
a_hat_t = I_eta(h_t, h_{t+1})
```

Loss:

```math
L_inverse = || I_eta(h_t, h_{t+1}) - a_t ||^2
```

Purpose:

```text
Ensure h_t preserves controllable factors:
- relative bearing
- relative distance change
- lateral motion effect
- rotation effect
```

---

### 4.4 Reward / Terminal / Stop Prediction

Optional auxiliary heads:

```text
r_hat_t = R(h_t, a_t)
done_hat_t = D(h_t)
stop_ready_hat_t = S(h_t)
visible_hat_t = V(h_t)
```

Possible losses:

```math
L_reward = || r_hat_t - r_t ||^2
L_done   = BCE(done_hat_t, done_t)
L_stop   = BCE(stop_ready_hat_t, stop_ready_t)
L_visible = BCE(visible_hat_t, visible_t)
```

These losses should be auxiliary, not the main representation objective.

---

### 4.5 Contrastive Transition Loss

Given:

```text
h_hat_{t+1} = T(h_t, a_t)
```

The true next latent `h_{t+1}` should be closer than negative candidates from the same batch.

Possible objective:

```text
InfoNCE / contrastive retrieval loss
```

Purpose:

```text
Prevent representation collapse.
Make predicted next latent distinguishable from wrong next states.
```

---

### 4.6 Suggested Full Representation Loss

```math
L_repr =
  lambda_f * L_forward
+ lambda_i * L_inverse
+ lambda_r * L_reward
+ lambda_d * L_done
+ lambda_s * L_stop
+ lambda_c * L_contrastive
+ lambda_v * L_variance
```

The most important terms are:

```text
L_forward
L_inverse
L_contrastive
L_stop / L_done
```

---

## 5. Two-Timescale Joint Training

After pretraining:

```text
1. Freeze encoder.
2. Train policy with LBC / BC using h_t.
3. Fine-tune policy with PPO.
4. Update encoder slowly and less frequently.
```

Suggested update rhythm:

```text
policy update: every batch
encoder update: every N policy updates
target encoder: EMA update
```

Example:

```text
for each training iteration:
    update policy every batch

    if iteration % N == 0:
        update encoder using representation losses

    update target encoder using EMA
```

Rationale:

```text
The policy needs a stable state distribution.
If encoder changes too fast, h_t drifts and the policy chases a moving target.
```

Compare these variants experimentally:

```text
A. frozen encoder
B. low-frequency encoder update
C. high-frequency end-to-end encoder update
```

Expected result:

```text
low-frequency encoder update should be more stable than high-frequency end-to-end training.
```

---

## 6. Policy Learning

### 6.1 LBC / Imitation Warm Start

Use privileged teacher trajectories:

```text
teacher observes privileged state s_t*
teacher outputs action a_t*
student observes h_t = Encoder(o_t)
student learns pi(a_t | h_t)
```

Behavior cloning loss:

```math
L_BC = || pi(h_t) - a_t^* ||^2
```

For stochastic policy, use negative log likelihood.

---

### 6.2 PPO Fine-Tuning

After BC / LBC warm-up:

```text
Use PPO to improve robustness.
```

PPO helps with:

```text
recovery from off-expert states
overshoot correction
temporary target loss
near-goal perturbations
velocity stopping
OOD initial conditions
```

Recommended final training objective during joint fine-tuning:

```math
L_total =
  L_PPO
+ small_lambda_f * L_forward
+ small_lambda_i * L_inverse
+ small_lambda_s * L_stop
```

Representation losses should have smaller weights during PPO fine-tuning than during representation pretraining.

---

## 7. Dataset Design

Do not train representation only on clean teacher trajectories.

The transition dataset should include:

```text
1. expert / teacher trajectories
2. noisy expert trajectories
3. random exploration
4. recovery trajectories
5. near-goal perturbation
6. failure trajectories
7. target-lost trajectories
8. branching transition data
```

Especially important cases for the docking task:

```text
ball at image edge
ball temporarily invisible
robot too close to ball
large lateral offset
large angular offset
overshoot
orbiting
near-stop instability
```

---

## 8. How to Evaluate the MDP-Consistent Representation

The main issue is not just whether transition loss is low.

The correct question is:

> Does h_t behave like a controllable, task-relevant, transferable MDP state?

Evaluation should be split into four levels:

```text
Level 1: latent transition consistency
Level 2: MDP variable accessibility
Level 3: policy learning usefulness
Level 4: generalization and transfer
```

---

# Level 1: Latent Transition Consistency

## 8.1 One-Step Latent Prediction Error

On held-out episodes:

```text
h_t = Encoder(o_t)
h_{t+1} = Encoder(o_{t+1})
h_hat_{t+1} = T(h_t, a_t)
```

Report:

```text
one-step MSE
one-step cosine error
normalized latent error
```

Important:

```text
Use episode-level train/test split.
Do not randomly split frames, because adjacent frames leak information.
```

---

## 8.2 Multi-Step Latent Rollout Error

Roll out in latent space:

```text
h_hat_{t+1} = T(h_t, a_t)
h_hat_{t+2} = T(h_hat_{t+1}, a_{t+1})
...
h_hat_{t+k} = T(h_hat_{t+k-1}, a_{t+k-1})
```

Compare with real encoded future latents:

```text
h_{t+k} = Encoder(o_{t+k})
```

Report error curve:

```text
k = 1, 2, 3, 5, 10
rollout error = ?
```

For mobile robot docking, evaluate horizons such as:

```text
0.5 s
1.0 s
2.0 s
```

If one-step error is low but multi-step error explodes, latent dynamics is not stable.

---

## 8.3 Action-Conditioned Retrieval Accuracy

Given:

```text
h_hat_{t+1} = T(h_t, a_t)
```

and a candidate batch:

```text
{h_j}_{j=1}^B
```

Measure whether the true `h_{t+1}` is nearest to `h_hat_{t+1}`.

Report:

```text
Retrieval@1
Retrieval@5
MRR
rank of true next state
```

This is more robust than pure MSE and helps detect collapse.

If batch size is 256:

```text
random Retrieval@1 = 1 / 256
```

A good action-conditioned transition model should be far above random.

---

## 8.4 Action Ablation

Compare:

```text
M1: T(h_t, a_t)
M2: T(h_t)
M3: T(h_t, shuffled a_t)
```

If `T(h_t, a_t)` is not better than `T(h_t)`, the model is not truly action-conditioned.

This is a critical ablation.

---

## 8.5 Branching Transition Test

In Isaac, reset to the same initial state multiple times and execute different actions:

```text
same state s_t
same observation o_t
different actions a_t^1, a_t^2, ..., a_t^m
```

Then obtain:

```text
o_{t+1}^1, o_{t+1}^2, ..., o_{t+1}^m
```

Check whether:

```text
T(h_t, a_t^i)
```

is closest to the corresponding:

```text
h_{t+1}^i
```

This tests whether the model can represent counterfactual action effects.

Expected action effects:

```text
forward    → ball appears larger
backward   → ball appears smaller
left shift → ball image position changes
right shift→ ball image position changes
left turn  → bearing changes
right turn → bearing changes
```

---

# Level 2: MDP Variable Accessibility

Use privileged simulator state only for evaluation or training-time guidance, not final policy input.

## 8.6 Linear Probe

Train a small linear head:

```text
s_hat_t = W h_t + b
```

Predict privileged variables:

```text
distance to ball
relative bearing
relative x/y in robot frame
ball visibility
stop_ready
dock_success
```

Report:

```text
distance RMSE
bearing MAE
relative x/y RMSE
visibility accuracy / AUC
stop_ready accuracy / AUC
dock_success accuracy / AUC
```

Interpretation:

```text
If a linear probe can decode these variables, h_t is policy-friendly.
If only a deep MLP can decode them, the information may exist but is not structured well.
```

Recommended:

```text
linear probe
2-layer MLP probe
```

Use both.

---

## 8.7 Pairwise MDP Distance Correlation

Define task-relevant privileged state distance:

```math
D_s(i,j) =
  w_d * |d_i - d_j|
+ w_alpha * |alpha_i - alpha_j|
+ w_x * |x_i - x_j|
+ w_y * |y_i - y_j|
+ w_v * 1[visible_i != visible_j]
+ w_g * 1[stop_i != stop_j]
```

Latent distance:

```math
D_h(i,j) = || normalize(h_i) - normalize(h_j) ||_2
```

Report:

```text
Spearman correlation between D_s and D_h
Kendall rank correlation
triplet accuracy
```

Triplet accuracy:

```text
If D_s(i,j) < D_s(i,k),
then D_h(i,j) should also be < D_h(i,k).
```

This evaluates whether the geometry of h_t matches the geometry of the task MDP.

---

## 8.8 Same-State Different-Appearance Test

In Isaac:

```text
fix robot-ball relative state
change lighting / texture / background / shadow / material / camera noise
```

Generate:

```text
o_t^1, o_t^2, ..., o_t^n
```

Ideally:

```text
f(o_t^1), f(o_t^2), ..., f(o_t^n)
```

should be close.

Report:

```math
intra_state_variance = mean_i || h_i - mean(h) ||^2
```

Also compare with different-state distance:

```math
separation_ratio = inter_state_distance / intra_state_distance
```

Expected:

```text
same state, different appearance → h close
different state, similar appearance → h far
```

---

## 8.9 Phase Classification

Define task phases automatically using privileged conditions:

```text
search       : ball not visible
approach     : visible but far
align        : visible but angular/lateral error large
dock         : near target distance
stop         : distance, angle, and velocity conditions satisfied
failure      : lost target / timeout / collision
```

Train a simple classifier on `h_t`.

Report:

```text
phase classification accuracy
confusion matrix
```

Useful diagnostics:

```text
search vs approach confusion → poor visibility encoding
approach vs dock confusion → poor distance encoding
dock vs stop confusion → poor velocity / terminal encoding
```

---

# Level 3: Policy Learning Usefulness

The final test is whether h_t improves policy learning.

Use the same policy learner and compare different state inputs.

## 8.10 Baselines

Suggested baselines:

```text
1. privileged state + policy         # upper bound
2. raw RGB CNN + policy              # end-to-end visual baseline
3. YOLO bbox / detector feature      # perception baseline
4. autoencoder latent + policy       # reconstruction baseline
5. reward-only latent + policy       # reward-driven baseline
6. transition-only latent + policy
7. inverse-only latent + policy
8. transition + inverse latent
9. full MDP-consistent h + policy
10. full h + LBC + PPO fine-tune
```

---

## 8.11 Docking Policy Metrics

Do not only report success rate.

For current docking task, report:

```text
search success rate
approach success rate
docking success rate
final distance error
final bearing error
final lateral error
stop stability
overshoot rate
orbiting rate
target lost rate
collision rate
average episode length
action smoothness
```

Docking success should require:

```text
d_min < distance < d_max
abs(bearing) < bearing_max
abs(lateral_offset) < lateral_max
linear_velocity < v_max
abs(angular_velocity) < omega_max
conditions held for M consecutive steps
```

This avoids counting unstable or accidental success.

---

## 8.12 Sample Efficiency

Plot:

```text
x-axis: number of demonstrations / transitions / RL steps
y-axis: docking success rate
```

Compare:

```text
raw RGB LBC
YOLO feature LBC
MDP representation LBC
MDP representation LBC + PPO
privileged state upper bound
```

A useful representation should show:

```text
faster learning
higher final success
less demonstration needed
less PPO interaction needed
better stability near stop
```

---

## 8.13 Frozen Representation Test

Compare:

```text
A. frozen encoder
B. low-frequency fine-tuned encoder
C. high-frequency end-to-end encoder
```

Expected:

```text
frozen encoder: stable, maybe lower upper bound
low-frequency encoder update: best tradeoff
high-frequency end-to-end update: unstable due to latent drift
```

This directly evaluates the two-timescale design.

---

# Level 4: Generalization and Transfer

## 8.14 OOD Generalization

Test policy and representation under changes:

```text
ball initial position
robot initial pose
ball size
ball color
background texture
lighting
shadow
camera height
camera FOV
occlusion
distractor objects
friction
wheel noise
control delay
sensor noise
```

Report:

```text
ID success rate
OOD success rate
relative performance drop
```

---

## 8.15 Transfer from Docking to Retrieval

Future extension:

```text
Task A: search and dock
Task B: search, dock, and grasp
Task C: search, grasp, and return
```

Evaluate:

```text
pretrain encoder on Task A
freeze / fine-tune encoder on Task B
compare with training from scratch
```

Good evidence:

```text
Task A representation improves Task B learning speed.
Mobile base docking representation remains useful for grasp precondition.
```

---

## 9. Suggested Main Experiment Table

| Encoder / State | 1-step error ↓ | 5-step error ↓ | Retrieval@1 ↑ | Distance RMSE ↓ | Bearing MAE ↓ | Stop AUC ↑ | Dock success ↑ | OOD success ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Privileged state | - | - | - | 0 | 0 | high | upper bound | upper bound |
| Raw RGB CNN | - | - | - | TBD | TBD | TBD | TBD | TBD |
| YOLO bbox | - | - | - | TBD | TBD | TBD | TBD | TBD |
| Autoencoder latent | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Reward-only latent | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transition-only latent | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Inverse-only latent | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transition + inverse | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Full MDP-consistent h | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Full h + LBC + PPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 10. Critical Ablations

The most important ablations are:

```text
1. without action input: T(h_t)
2. shuffled action input: T(h_t, shuffled a_t)
3. without inverse dynamics loss
4. without terminal / stop prediction
5. without contrastive retrieval loss
6. without non-expert transitions
7. frozen encoder vs low-frequency update vs high-frequency update
8. teacher-only dataset vs mixed dataset
9. LBC only vs LBC + PPO fine-tune
10. MDP representation vs reward-driven representation
```

Especially important:

```text
T(h_t, a_t) vs T(h_t)
```

If action-conditioned transition is not better, the method is not truly MDP-consistent.

---

## 11. Failure Modes to Monitor

Representation failure modes:

```text
latent collapse
transition model ignores action
latent only encodes ball size
latent ignores bearing/lateral offset
latent overfits background texture
good one-step prediction but bad multi-step rollout
good expert-trajectory prediction but poor recovery states
```

Policy failure modes:

```text
oscillatory approach
orbiting around ball
overshoot
stopping too close
stopping too far
losing target near goal
large residual velocity at stop
unstable stop condition
```

---

## 12. Minimum Viable Evaluation Plan

If time is limited, implement these first.

### Dataset

```text
D_train:
  teacher trajectories
  noisy teacher trajectories
  random exploration
  near-goal perturbations

D_test_ID:
  held-out episodes from same distribution

D_test_OOD:
  different lighting / texture / initial pose / ball position

D_branch:
  same initial state, multiple different actions
```

### Representation Metrics

```text
1-step latent prediction error
5-step latent rollout error
transition Retrieval@1 / Retrieval@5
inverse dynamics action MSE
linear probe distance RMSE
linear probe bearing MAE
stop_ready AUC
latent variance / effective dimension
```

### Policy Metrics

```text
dock success
final distance error
final bearing error
overshoot rate
orbiting rate
target lost rate
episode length
OOD success
```

### Baselines

```text
privileged state
raw RGB
YOLO bbox
autoencoder latent
transition-only latent
transition + inverse latent
full MDP-consistent latent
```

---

## 13. Recommended Implementation Checklist

### Data Collection

- [ ] Implement privileged teacher policy.
- [ ] Save transitions:
  - [ ] RGB observation `o_t`
  - [ ] action `a_t`
  - [ ] next RGB observation `o_{t+1}`
  - [ ] privileged state `s_t*`
  - [ ] reward `r_t`
  - [ ] done flag
  - [ ] phase label if available
- [ ] Add noisy teacher data.
- [ ] Add random exploration data.
- [ ] Add near-goal perturbation data.
- [ ] Add branching transition data.

### Representation Pretraining

- [ ] Implement encoder `f_theta`.
- [ ] Implement transition model `T_psi(h, a)`.
- [ ] Implement inverse model `I_eta(h_t, h_{t+1})`.
- [ ] Implement reward / done / stop heads.
- [ ] Implement contrastive transition loss.
- [ ] Add target encoder or stop-gradient path.
- [ ] Add collapse diagnostics.

### Representation Evaluation

- [ ] One-step prediction error.
- [ ] Multi-step rollout error.
- [ ] Retrieval@1 / @5.
- [ ] Action ablation.
- [ ] Branching transition test.
- [ ] Linear probe for distance / bearing / stop.
- [ ] Pairwise MDP distance correlation.
- [ ] Same-state different-appearance test.
- [ ] Phase classification confusion matrix.

### Policy Learning

- [ ] Use `h_t` as state.
- [ ] Train LBC / BC policy from teacher actions.
- [ ] Evaluate docking policy.
- [ ] Fine-tune with PPO.
- [ ] Compare frozen / low-frequency / high-frequency encoder update.

### Paper / Thesis Experiments

- [ ] Compare with privileged state upper bound.
- [ ] Compare with raw RGB end-to-end.
- [ ] Compare with YOLO bbox feature.
- [ ] Compare with autoencoder latent.
- [ ] Compare with reward-driven latent.
- [ ] Compare with MDP-consistent latent.
- [ ] Test OOD generalization.
- [ ] Prepare transfer experiment for retrieval task if time allows.

---

## 14. Recommended Main Claim

A possible research claim:

> We learn an action-conditioned MDP-consistent visual representation for object-centric mobile robot docking. The representation is pretrained from teacher-collected transitions using latent transition consistency, inverse dynamics, and terminal-condition prediction. The learned latent state improves imitation warm-start and PPO fine-tuning, and shows better generalization than raw visual or reward-only representations.

For future extension:

> The learned representation can be reused for object retrieval tasks involving mobile manipulation, where docking becomes a precondition for grasping.

---

## 15. One-Sentence Summary

The target is not simply to make PPO find the ball, but to learn a visual latent state `h_t` that behaves like a controllable MDP state and can support docking now and retrieval later.
