#!/bin/bash
# 等任一张卡空出 >=6GB,立刻跑 sup 的解码评估(o -> z -> teacher 观测 -> teacher 策略)
L=/home/tb/.claude/jobs/2174f11a/tmp; S=$L/g4_state.txt
cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $S; }
G=""
while [ -z "$G" ]; do
  for g in 0 1; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 6000 ] && { G=$g; break; }
  done
  [ -z "$G" ] && sleep 60
done
log "DECODE 窗口出现在 gpu$G(已用 ${used}MB),开始"
( flock 9; sleep 90 ) 9>/tmp/isaac_start.lock
timeout 1200 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/decode_policy.py \
  --arm sup --encoder-root models/vision/g4_diam_angle --score-mode angle --run-tag _g4_diam_angle > $L/decode_sup.log 2>&1
log "DECODE rc=$? -> $(grep -h 'DECODE\] arm=' $L/decode_sup.log | tail -1)"
grep -h "DECODE" $L/decode_sup.log >> $S
