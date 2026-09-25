#!/bin/bash
L=/home/tb/.claude/jobs/2174f11a/tmp; cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/g4_state.txt; }
gate(){ ( flock 9; sleep 60 ) 9>/tmp/isaac_start.lock; }
run(){ A=$1; CK=$2; COL=$3
  log "START distract-eval $A ckpt=$CK noise=$COL"
  gate
  timeout 900 env CUDA_VISIBLE_DEVICES=0 ./IsaacLab/isaaclab.sh -p val/score_student.py \
    --arm $A --encoder-root models/vision/g4_diam_angle --score-mode angle \
    --run-tag _g4_diam_angle_red --ckpt $CK --noise-color $COL --tag _nz$COL \
    > $L/dz_${A}_${COL}.log 2>&1
  grep -h "RESULT\|NOISE-COLOR" $L/dz_${A}_${COL}.log | sed "s/arm=[^ ]* /arm=${A}_nz${COL} /" >> $L/g4_state.txt
  log "END   distract-eval $A $COL"
}
log "=== gpu0:换干扰颜色的配对评估(红球任务,各臂 seed1 最佳 ckpt)==="
for COL in red green; do
  run A_donor   last.pt               $COL
  run R_donor   updates/ppo_000220.pt $COL
  run V_donor   updates/ppo_000210.pt $COL
  run init      updates/ppo_000220.pt $COL
  run sup_donor updates/ppo_000220.pt $COL
done
log "=== gpu0:换干扰评估全部完成 ==="
