#!/bin/bash
# 2×2 设计(2026-09-14 夜):teacher 观测 × 奖励度量,各 2 种子取优。
#   OBS ∈ {diam=相机坐标(x_c,直径), ang=物理坐标(方位角,距离)}
#   RM  ∈ {cam=相机度量(x_px,直径px), angle=物理度量(deg,m)}   两套尺度在停车点等价
# 每个 teacher -> 12 万帧 -> A/V/R/sup 四个表征(sup 回归该 teacher 自己那两维)+ 未训练 init
# -> 16 条 PPO(4 teacher × 4 臂,含 sup)
L=/home/tb/.claude/jobs/2174f11a/tmp; S=$L/g4_state.txt
cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
gate(){ ( flock 9; sleep 90 ) 9>/tmp/isaac_start.lock; }
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $S; }
COMBOS=(); while [ $# -ge 2 ]; do COMBOS+=("$1 $2"); shift 2; done
[ ${#COMBOS[@]} -gt 0 ] || COMBOS=("diam angle" "ang angle")

teach(){ O=$1; RM=$2; SD=$3; G=$4; N=${O}_${RM}; TD=models/rl/teacher_score_g4_${N}_s${SD}
  [ -f $TD/last.zip ] && return 0
  log "START teacher ${N}_s${SD} gpu$G"
  gate; timeout 5400 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p train/teacher_score.py --no-lateral --obs-mask $O --score-mode $RM --tag _g4_${N}_s${SD} --seed $SD > $L/g4_teacher_${N}_s${SD}.log 2>&1
  log "END   teacher ${N}_s${SD} rc=$?"
}
prep(){ O=$1; RM=$2; G=$3; N=${O}_${RM}; DD=./data_g4_$N; VR=models/vision/g4_$N
  SUPT=cam; [ "$O" = "ang" ] && SUPT=ang
  BEST=""; BV=-1
  for SD in 1 2; do TD=models/rl/teacher_score_g4_${N}_s${SD}; [ -f $TD/last.zip ] || continue
    gate; timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/teacher_score.py --dir $TD --action-k 0.3 --no-lateral --obs-mask $O --score-mode $RM > $L/g4_val_${N}_s${SD}.log 2>&1
    V=$(grep -o "v1成功率 [0-9.]*" $L/g4_val_${N}_s${SD}.log | tail -1 | tr -d '%' | awk '{print $2}')
    Rt=$(grep -o "回报 [0-9.]*" $L/g4_val_${N}_s${SD}.log | tail -1 | awk '{print $2}')
    log "[$N s$SD] v1=$V 回报=$Rt"
    SC=$(awk -v v="${V:-0}" -v r="${Rt:-0}" 'BEGIN{print v*1000+r}')
    if awk -v a="$SC" -v b="$BV" 'BEGIN{exit !(a>b)}'; then BV=$SC; BEST=$SD; fi
  done
  [ -n "$BEST" ] || { log "FAIL 无可用 teacher $N"; return 1; }
  log "[$N] 取 seed $BEST"; echo "$N $BEST" >> $L/g4_best.txt
  TD=models/rl/teacher_score_g4_${N}_s${BEST}
  if [ ! -f $DD/unit_00/transitions.csv ]; then
    log "START collect $N gpu$G (seed $BEST)"
    gate; timeout 3600 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p scripts/collect_teacher_1m.py --no-lateral --obs-mask $O --score-mode $RM --teacher $TD/last.zip --out-dir $DD --transitions 120000 --units 1 --num-envs 64 --seed 1 --angle-deg 45 > $L/g4_collect_$N.log 2>&1
    log "END   collect $N rc=$?"
  fi
  [ -f $DD/unit_00/transitions.csv ] || { log "FAIL collect $N"; return 1; }
  log "START stage1 $N gpu$G (A/V/R/sup, sup-target=$SUPT)"
  CUDA_VISIBLE_DEVICES=$G timeout 7200 python scripts/stage1_bc_heads.py --out-root $VR --data $DD --units 1 --a-target sampled --sup-target $SUPT --donor-steps 4000 > $L/g4_s1_$N.log 2>&1
  log "END   stage1 $N rc=$?"
  for A in A V R sup; do mkdir -p $VR/${A}_donor && cp $VR/$A/donor.pt $VR/${A}_donor/encoder_final.pt; done
}
ppo(){ O=$1; RM=$2; A=$3; G=$4; N=${O}_${RM}; VR=models/vision/g4_$N
  RUN=models/rl/score_k03noy_student_${A}_g4_$N
  [ -f $RUN/eval_score.csv ] && { log "SKIP ppo ${A}_g4_$N"; return 0; }
  log "START ppo ${A}_g4_$N gpu$G"
  gate; timeout 14400 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p train/free_student.py --arm $A --encoder-root $VR --score-mode $RM --tag _g4_$N > $L/g4_ppo_${A}_$N.log 2>&1
  log "END   ppo ${A}_g4_$N rc=$?"
  [ -f $RUN/last.pt ] || { log "FAIL ppo ${A}_g4_$N"; return 1; }
  gate; timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py --arm $A --encoder-root $VR --score-mode $RM --run-tag _g4_$N > $L/g4_eval_${A}_$N.log 2>&1
  grep -h RESULT $L/g4_eval_${A}_$N.log | sed "s/arm=[^ ]* /arm=${A}_g4_$N /" >> $S
  for u in 000200 000210 000220 000230 000240; do
    gate; timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py --arm $A --encoder-root $VR --score-mode $RM --run-tag _g4_$N --ckpt updates/ppo_$u.pt --tag _$u >/dev/null 2>&1
  done
  log "END   eval ${A}_g4_$N"
}
log "=== 阶段1:teacher(每组合 2 种子),两卡按下标奇偶分,互不重叠 ==="
TJOBS=(); for c in "${COMBOS[@]}"; do for SD in 1 2; do TJOBS+=("$c $SD"); done; done
( i=0; for j in "${TJOBS[@]}"; do [ $((i%2)) -eq 0 ] && teach $j 0; i=$((i+1)); done ) &
( i=0; for j in "${TJOBS[@]}"; do [ $((i%2)) -eq 1 ] && teach $j 1; i=$((i+1)); done ) &
wait
log "=== 阶段2:取优 + 采集 + 四臂表征 ==="
( i=0; for c in "${COMBOS[@]}"; do [ $((i%2)) -eq 0 ] && prep $c 0; i=$((i+1)); done ) &
( i=0; for c in "${COMBOS[@]}"; do [ $((i%2)) -eq 1 ] && prep $c 1; i=$((i+1)); done ) &
wait
if [ -n "$NO_PPO" ]; then log "=== 前两阶段完成,PPO 交给队列 worker ==="; exit 0; fi
log "=== 阶段3:16 条 PPO(4 组合 × A/V/R/sup) ==="
JOBS=(); for c in "${COMBOS[@]}"; do for A in R_donor A_donor V_donor sup_donor; do JOBS+=("$c $A"); done; done
( i=0; for j in "${JOBS[@]}"; do [ $((i%2)) -eq 0 ] && ppo $j 0; i=$((i+1)); done ) &
( i=0; for j in "${JOBS[@]}"; do [ $((i%2)) -eq 1 ] && ppo $j 1; i=$((i+1)); done ) &
wait
log "=== 全部完成 ==="
