#!/bin/bash
# 换干扰颜色的配对评估补种子:绿球那条结论(R 只掉 4.6 / A 掉 43.8)目前只有 seed1,补 seed2/3。
L=/home/tb/.claude/jobs/2174f11a/tmp; cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/g4_state.txt; }
gate(){ ( flock 9; sleep 45 ) 9>/tmp/isaac_start.lock; }
run(){ A=$1; SD=$2; CK=$3; COL=$4
  SARG=""; SFX=""; [ "$SD" != "1" ] && { SARG="--seed $SD"; SFX="_s$SD"; }
  log "START dz $A$SFX ckpt=$CK noise=$COL"
  gate
  timeout 900 env CUDA_VISIBLE_DEVICES=0 ./IsaacLab/isaaclab.sh -p val/score_student.py \
    --arm $A --encoder-root models/vision/g4_diam_angle --score-mode angle $SARG \
    --run-tag _g4_diam_angle_red --ckpt $CK --noise-color $COL --tag _nz$COL \
    > $L/dz_${A}${SFX}_${COL}.log 2>&1
  grep -h "RESULT" $L/dz_${A}${SFX}_${COL}.log | sed "s/arm=[^ ]* /arm=${A}${SFX}_nz${COL} /" >> $L/g4_state.txt
  log "END   dz $A$SFX $COL"
}
log "=== gpu0:换干扰评估补 seed2/3(A、R × 红绿)==="
for COL in green red; do
  run A_donor 2 updates/ppo_000230.pt $COL
  run A_donor 3 updates/ppo_000200.pt $COL
  run R_donor 2 updates/ppo_000210.pt $COL
  run R_donor 3 updates/ppo_000230.pt $COL
done
log "=== gpu0:补种子完成 ==="
