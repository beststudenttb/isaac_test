#!/bin/bash
# 队列跑完后自动接上身份实验渲染。等待条件只用文件计数 + 显存,绝不用 ps 匹配字符串。
L=/home/tb/.claude/jobs/2174f11a/tmp; S=$L/g4_state.txt
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $S; }
cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
ok=0
while [ $ok -lt 2 ]; do
  sleep 300
  q=$(grep -cve '^\s*$' -e '^#' $L/g4_queue.txt); c=$(ls $L/g4_claims 2>/dev/null | wc -l)
  m0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  m1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$c" -ge "$q" ] && [ "$m0" -lt 1500 ] && [ "$m1" -lt 1500 ]; then ok=$((ok+1)); else ok=0; fi
done
log "队列判定跑完(认领 $c / 队列 $q,两卡显存空),停 worker,转身份实验渲染"
touch $L/g4_stop; sleep 120
for g in 0 1; do
  pids=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")
  [ -n "$pids" ] && for p in $pids; do kill -9 $p 2>/dev/null; done
done
sleep 10
log "START 身份实验渲染(gpu0,14 条件 × 6 轮 × 64 env = 5376 帧)"
timeout 7200 env CUDA_VISIBLE_DEVICES=0 ./IsaacLab/isaaclab.sh -p scripts/render_identity.py --rounds 6 --out-dir data_identity > $L/render_identity.log 2>&1
log "END   身份实验渲染 rc=$? 帧数=$(ls data_identity/img 2>/dev/null | wc -l)"
CUDA_VISIBLE_DEVICES="" python $L/identity_analyze.py > $L/identity_out.txt 2>&1
log "END   身份实验分析 -> identity_out.txt"
