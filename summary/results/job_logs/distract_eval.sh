#!/bin/bash
# 主结论的鲁棒性检验:已训好的"追红球"策略,把干扰球改色后重新评估(只改外观,不改标签/奖励)。
# 配对比较:同一个 ckpt、同一个表征,只有干扰球颜色不同。红=同色干扰(关键),绿=异色对照。
L=/home/tb/.claude/jobs/2174f11a/tmp; cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/g4_state.txt; }
gate(){ ( flock 9; sleep 90 ) 9>/tmp/isaac_start.lock; }
reap(){ local p; p=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")
  [ -n "$p" ] && for q in $p; do kill -9 $q 2>/dev/null; done; return 0; }
ok=0
while [ $ok -lt 2 ]; do
  sleep 300
  q=$(grep -cve '^\s*$' -e '^#' $L/g4_queue.txt); c=$(ls $L/g4_claims 2>/dev/null | wc -l)
  m0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  m1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$c" -ge "$q" ] && [ "$m0" -lt 1500 ] && [ "$m1" -lt 1500 ]; then ok=$((ok+1)); else ok=0; fi
done
touch $L/g4_stop; sleep 90; reap; sleep 5
log "=== 换干扰颜色的配对评估开始(gpu0,只跑红球任务,用各臂 seed1 的最佳 ckpt)==="
# 臂  最佳ckpt(来自 seed1 红球的末6最好)
run(){ A=$1; CK=$2; COL=$3
  log "START distract-eval $A ckpt=$CK noise=$COL"
  reap; gate
  timeout 900 env CUDA_VISIBLE_DEVICES=0 ./IsaacLab/isaaclab.sh -p val/score_student.py \
    --arm $A --encoder-root models/vision/g4_diam_angle --score-mode angle \
    --run-tag _g4_diam_angle_red --ckpt $CK --noise-color $COL --tag _nz$COL \
    > $L/dz_${A}_${COL}.log 2>&1
  grep -h "RESULT\|NOISE-COLOR" $L/dz_${A}_${COL}.log | sed "s/arm=[^ ]* /arm=${A}_nz${COL} /" >> $L/g4_state.txt
  log "END   distract-eval $A $COL rc=$?"
}
for COL in red green; do
  run A_donor   last.pt                 $COL
  run R_donor   updates/ppo_000220.pt   $COL
  run V_donor   updates/ppo_000210.pt   $COL
  run init      updates/ppo_000220.pt   $COL
  run sup_donor updates/ppo_000220.pt   $COL
done
log "=== 换干扰颜色评估全部完成 ==="
