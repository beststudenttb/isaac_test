#!/bin/bash
# 队列 worker:两张卡各一个循环,从 g4_queue.txt 里抢一行没被认领的活来跑。
# 队列为空就每分钟看一次。写 g4_stop 文件可停。行格式: "<OBS> <RM> <ARM>"(PPO)
L=/home/tb/.claude/jobs/2174f11a/tmp; Q=$L/g4_queue.txt; D=$L/g4_claims; S=$L/g4_state.txt
mkdir -p $D; touch $Q
cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
gate(){ ( flock 9; sleep 90 ) 9>/tmp/isaac_start.lock; }
# timeout 只杀 isaaclab.sh 外壳,里面的 python 会变孤儿继续占显存,导致后续启动静默 OOM。
# 每次起 Isaac 前清掉本卡上的残留进程(另一分支只用另一张卡,所以本卡上的都不是别人的)。
reap(){ local g=$1 pids
  pids=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")
  [ -n "$pids" ] && { log "REAP gpu$g 残留进程 $pids"; for q in $pids; do kill -9 $q 2>/dev/null; done; sleep 5; }
  return 0
}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $S; }
ppo(){ O=$1; RM=$2; A=$3; G=$4; TASK=${5:-red}; SD=${6:-1}; N=${O}_${RM}; VR=models/vision/g4_$N
  # 训练脚本需要 --noise 才带干扰球;评估脚本本来就写死带干扰球,不接受 --noise
  TRARG="--noise"; EVARG=""; TSFX="_red"
  [ "$TASK" = "blue" ] && { TRARG="--noise --chase-blue"; EVARG="--chase-blue"; TSFX="_blue"; }
  SSFX=""; SARG=""; [ "$SD" != "1" ] && { SSFX="_s$SD"; SARG="--seed $SD"; }
  RUN=models/rl/score_k03noy_student_${A}${SSFX}_g4_${N}${TSFX}
  [ -f $RUN/eval_score.csv ] && { log "SKIP ${A}_g4_$N(已有)"; return 0; }
  if [ "$A" = "init" ]; then [ -f $VR/encoder_init.pt ] || return 2; else [ -f $VR/$A/encoder_final.pt ] || return 2; fi   # 2 = 表征还没好
  TAG="${A}${SSFX}_g4_${N}${TSFX}"
  log "START ppo $TAG gpu$G (任务=$TASK seed=$SD)"
  reap $G; gate; timeout 14400 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p train/free_student.py --arm $A --encoder-root $VR --score-mode $RM $TRARG $SARG --tag _g4_${N}${TSFX} > $L/g4_ppo_$TAG.log 2>&1
  log "END   ppo $TAG rc=$?"
  [ -f $RUN/last.pt ] || { log "FAIL ppo $TAG"; return 1; }
  reap $G; gate; timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py --arm $A --encoder-root $VR --score-mode $RM $EVARG $SARG --run-tag _g4_${N}${TSFX} > $L/g4_eval_$TAG.log 2>&1
  grep -h RESULT $L/g4_eval_$TAG.log | sed "s/arm=[^ ]* /arm=$TAG /" >> $S
  for u in 000200 000210 000220 000230 000240; do
    reap $G; gate; timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py --arm $A --encoder-root $VR --score-mode $RM $EVARG $SARG --run-tag _g4_${N}${TSFX} --ckpt updates/ppo_$u.pt --tag _$u >/dev/null 2>&1
  done
  log "END   eval $TAG"
}
bceval(){ O=$1; RM=$2; A=$3; G=$4; N=${O}_${RM}; VR=models/vision/g4_$N
  [ -f $VR/$A/head.pt ] || return 2
  OUT=models/rl/score_k03noy_student_bc_${A}_g4_$N/eval_bc.csv
  [ -f $OUT ] && { log "SKIP bc ${A}_g4_$N"; return 0; }
  log "START bc-eval ${A}_g4_$N gpu$G(不训练,直接拿 stage1 的头当策略)"
  reap $G; gate; timeout 1800 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/bc_policy.py --arm $A --encoder-root $VR --score-mode $RM --run-tag _g4_$N > $L/g4_bc_${A}_$N.log 2>&1
  grep -h "\[BC\]" $L/g4_bc_${A}_$N.log >> $S
  log "END   bc-eval ${A}_g4_$N rc=$?"
}
loop(){ G=$1
  while ps -ef | grep -q "[g]4.sh"; do sleep 60; done      # 前两阶段还在占卡,先别起 PPO
  log "worker gpu$G:前两阶段已结束,开始消费队列"
  while [ ! -f $L/g4_stop ]; do
    got=""
    while IFS= read -r line; do
      case "$line" in ""|\#*) continue;; esac
      key=$(echo "$line" | tr ' /' '__')
      if mkdir "$D/$key" 2>/dev/null; then got="$line"; break; fi
    done < $Q
    if [ -z "$got" ]; then sleep 60; continue; fi
    key=$(echo "$got" | tr ' /' '__')
    case "$got" in
      *" bc:"*) set -- $got; bceval $1 $2 "${3#bc:}" $G; rc=$? ;;
      *) set -- $got; ppo $1 $2 $3 $G $4 $5; rc=$? ;;
    esac
    if [ $rc -ne 0 ] && [ $rc -ne 2 ]; then log "任务失败,休息 60s"; sleep 60; fi
    if [ $rc -eq 2 ]; then log "HOLD $got(表征未就绪,放回队列)"; rmdir "$D/$key" 2>/dev/null; sleep 120; fi
  done
  log "worker gpu$G 退出"
}
log "=== 队列 worker 启动(两卡) ==="
( loop 0 ) & ( loop 1 ) & wait
