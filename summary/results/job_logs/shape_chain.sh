#!/bin/bash
L=/home/tb/.claude/jobs/2174f11a/tmp; cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/g4_state.txt; }
ok=0
while [ $ok -lt 2 ]; do
  sleep 120
  m0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  m1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$m0" -lt 1500 ] && [ "$m1" -lt 1500 ]; then ok=$((ok+1)); else ok=0; fi
done
for g in 0 1; do
  pids=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")
  [ -n "$pids" ] && for p in $pids; do kill -9 $p 2>/dev/null; done
done
sleep 10
log "START 形状实验渲染 gpu0(10 条件 × 6 轮 × 64 env)"
timeout 5400 env CUDA_VISIBLE_DEVICES=0 ./IsaacLab/isaaclab.sh -p scripts/render_shape.py --rounds 6 --out-dir data_shape > $L/render_shape.log 2>&1
log "END   形状实验渲染 rc=$? 帧数=$(ls data_shape/img 2>/dev/null | wc -l)"
CUDA_VISIBLE_DEVICES="" python $L/shape_analyze.py > $L/shape_out.txt 2>&1
log "END   形状实验分析 -> shape_out.txt"
m=0
while IFS= read -r l; do case "$l" in ""|\#*) continue;; esac; set -- $l
  if [ "$5" = "3" ]; then k=$(echo "$l"|tr ' /' '__'); rmdir "$L/g4_claims/$k" 2>/dev/null && m=$((m+1)); fi
done < $L/g4_queue.txt
log "放回 seed3 的 $m 条,worker 继续"
