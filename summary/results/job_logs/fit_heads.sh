#!/bin/bash
cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
L=/home/tb/.claude/jobs/2174f11a/tmp
for C in diam_cam diam_angle ang_cam ang_angle; do for A in R A V sup init; do
  CUDA_VISIBLE_DEVICES="" python scripts/fit_action_head.py models/vision/g4_$C $A 20000 >> $L/fit_heads.log 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] head fit $C/$A rc=$?" >> $L/g4_state.txt
done; done
echo "[$(date '+%m-%d %H:%M:%S')] === 20 个动作头训练完成 ===" >> $L/g4_state.txt
