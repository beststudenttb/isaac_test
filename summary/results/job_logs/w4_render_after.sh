#!/bin/bash
# 等 w4 队列完全排空、两卡都空,再做受控渲染。
# 原因:调度器每次起 Isaac 前 reap 本卡残留,队列跑着的时候没有任何安全窗口。
L=/home/tb/.claude/jobs/2174f11a/tmp; W=$L/w4
cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $W/state.txt; }
TOT=$(grep -cve '^\s*$' $W/queue.txt)
ok=0
while [ $ok -lt 2 ]; do
  sleep 300
  fin=$(( $(ls $W/done 2>/dev/null|wc -l) + $(ls $W/failed 2>/dev/null|wc -l) ))
  m0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  m1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  TOT=$(grep -cve '^\s*$' $W/queue.txt)
  if [ "$fin" -ge "$TOT" ] && [ "$m0" -lt 1500 ] && [ "$m1" -lt 1500 ]; then ok=$((ok+1)); else ok=0; fi
done
log "=== 队列已排空,开始 4 物体受控渲染(gpu0)==="
sleep 60
timeout 3600 env CUDA_VISIBLE_DEVICES=0 ./IsaacLab/isaaclab.sh -p scripts/render_w4.py \
  --rounds 10 --out-dir data_w4_probe > $L/w4_render.log 2>&1
log "=== 受控渲染 rc=$? 帧数=$(ls data_w4_probe/img 2>/dev/null|wc -l) ==="
