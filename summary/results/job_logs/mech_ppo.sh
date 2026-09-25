#!/bin/bash
# 机制拆分的行为确认:表征侧已证"擦除量由目标在随机先验上的可提取性决定"(六臂单调零例外),
# 这里看策略是否跟着走。预测:Ax(动作目标但对称易提取,残留 0.559)迁移应像 R(~83%);
#                        Aw(动作目标且反对称难提取,残留 −0.116)应像 A 一样死(~10%)。
# 只跑 seed1 的红/蓝各一条,四条足以判预测真假。
L=/home/tb/.claude/jobs/2174f11a/tmp; cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/g4_state.txt; }
gate(){ ( flock 9; sleep 60 ) 9>/tmp/isaac_start.lock; }
reap(){ local p; p=$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d " ")
  [ -n "$p" ] && for q in $p; do kill -9 $q 2>/dev/null; done; return 0; }
job(){ A=$1; T=$2; G=$3
  TR="--noise"; EV=""; SFX="_red"
  [ "$T" = "blue" ] && { TR="--noise --chase-blue"; EV="--chase-blue"; SFX="_blue"; }
  TAG="${A}_g4_diam_angle$SFX"; RUN=models/rl/score_k03noy_student_${A}_g4_diam_angle$SFX
  [ -f $RUN/eval_score.csv ] && { log "SKIP $TAG"; return 0; }
  log "START ppo $TAG gpu$G"
  reap $G; gate
  timeout 14400 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p train/free_student.py \
    --arm $A --encoder-root models/vision/g4_diam_angle --score-mode angle $TR \
    --tag _g4_diam_angle$SFX > $L/g4_ppo_$TAG.log 2>&1
  log "END   ppo $TAG rc=$?"
  [ -f $RUN/last.pt ] || return 1
  reap $G; gate
  timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py \
    --arm $A --encoder-root models/vision/g4_diam_angle --score-mode angle $EV \
    --run-tag _g4_diam_angle$SFX > $L/g4_eval_$TAG.log 2>&1
  grep -h RESULT $L/g4_eval_$TAG.log | sed "s/arm=[^ ]* /arm=$TAG /" >> $L/g4_state.txt
  for u in 000200 000210 000220 000230 000240; do
    reap $G; gate
    timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py \
      --arm $A --encoder-root models/vision/g4_diam_angle --score-mode angle $EV \
      --run-tag _g4_diam_angle$SFX --ckpt updates/ppo_$u.pt --tag _$u >/dev/null 2>&1
  done
  log "END   eval $TAG"
}
# gpu0 立刻开工;gpu1 等形状对照跑完(用 g4_state.txt 里的完成标记判断,不用 ps 匹配)
( job Ax_donor red 0; job Ax_donor blue 0; log "=== gpu0:Ax 两条完成 ===" ) &
( until grep -q "gpu1:形状对照完成" $L/g4_state.txt; do sleep 120; done
  job Aw_donor red 1; job Aw_donor blue 1; log "=== gpu1:Aw 两条完成 ===" ) &
wait
log "=== 机制拆分的 PPO 行为确认全部完成 ==="
