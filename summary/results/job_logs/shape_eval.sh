#!/bin/bash
# 策略层面的形状对照:干扰物固定为**红色**,只换形状(球 / 立方 / 锥)。
# 表征层面已证形状无关(红立方 0.731 > 红球 0.607,跟投影面积走);这里看策略是否也无关。
# 红球干扰的成绩已有,所以只需补 cube / cone 两格 × A、R × 3 种子。
L=/home/tb/.claude/jobs/2174f11a/tmp; cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/g4_state.txt; }
gate(){ ( flock 9; sleep 45 ) 9>/tmp/isaac_start.lock; }
reap(){ local p; p=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d " ")
  [ -n "$p" ] && for q in $p; do kill -9 $q 2>/dev/null; done; return 0; }
run(){ A=$1; SD=$2; CK=$3; SH=$4
  SARG=""; SFX=""; [ "$SD" != "1" ] && { SARG="--seed $SD"; SFX="_s$SD"; }
  log "START shape-eval $A$SFX ckpt=$CK 干扰=红$SH"
  reap; gate
  timeout 900 env CUDA_VISIBLE_DEVICES=1 ./IsaacLab/isaaclab.sh -p val/score_student.py \
    --arm $A --encoder-root models/vision/g4_diam_angle --score-mode angle $SARG \
    --run-tag _g4_diam_angle_red --ckpt $CK --noise-color red --noise-shape $SH --tag _nzred$SH \
    > $L/sz_${A}${SFX}_${SH}.log 2>&1
  grep -h "RESULT" $L/sz_${A}${SFX}_${SH}.log | sed "s/arm=[^ ]* /arm=${A}${SFX}_nzred${SH} /" >> $L/g4_state.txt
  log "END   shape-eval $A$SFX 红$SH"
}
log "=== gpu1:策略层面的形状对照(干扰物恒为红色,只换形状)==="
for SH in cube cone; do
  run A_donor 1 last.pt               $SH
  run A_donor 2 updates/ppo_000230.pt $SH
  run A_donor 3 updates/ppo_000200.pt $SH
  run R_donor 1 updates/ppo_000220.pt $SH
  run R_donor 2 updates/ppo_000210.pt $SH
  run R_donor 3 updates/ppo_000230.pt $SH
done
log "=== gpu1:形状对照完成 ==="
reap
