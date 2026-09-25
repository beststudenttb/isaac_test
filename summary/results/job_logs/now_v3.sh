#!/bin/bash
L=/home/tb/.claude/jobs/2174f11a/tmp; cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/g4_state.txt; }
gate(){ ( flock 9; sleep 60 ) 9>/tmp/isaac_start.lock; }
job(){ T=$1; TR="--noise"; EV=""; SFX="_red"
  [ "$T" = "blue" ] && { TR="--noise --chase-blue"; EV="--chase-blue"; SFX="_blue"; }
  TAG="V_donor_s3_g4_diam_angle$SFX"; RUN=models/rl/score_k03noy_student_V_donor_s3_g4_diam_angle$SFX
  log "START ppo $TAG gpu1"
  gate; timeout 14400 env CUDA_VISIBLE_DEVICES=1 ./IsaacLab/isaaclab.sh -p train/free_student.py \
    --arm V_donor --encoder-root models/vision/g4_diam_angle --score-mode angle $TR --seed 3 \
    --tag _g4_diam_angle$SFX > $L/g4_ppo_$TAG.log 2>&1
  log "END   ppo $TAG rc=$?"
  [ -f $RUN/last.pt ] || return 1
  gate; timeout 900 env CUDA_VISIBLE_DEVICES=1 ./IsaacLab/isaaclab.sh -p val/score_student.py \
    --arm V_donor --encoder-root models/vision/g4_diam_angle --score-mode angle $EV --seed 3 \
    --run-tag _g4_diam_angle$SFX > $L/g4_eval_$TAG.log 2>&1
  grep -h RESULT $L/g4_eval_$TAG.log | sed "s/arm=[^ ]* /arm=$TAG /" >> $L/g4_state.txt
  for u in 000200 000210 000220 000230 000240; do
    gate; timeout 900 env CUDA_VISIBLE_DEVICES=1 ./IsaacLab/isaaclab.sh -p val/score_student.py \
      --arm V_donor --encoder-root models/vision/g4_diam_angle --score-mode angle $EV --seed 3 \
      --run-tag _g4_diam_angle$SFX --ckpt updates/ppo_$u.pt --tag _$u >/dev/null 2>&1
  done
  log "END   eval $TAG"
}
job red; job blue
log "=== gpu1:V 的 seed3 两条完成 ==="
