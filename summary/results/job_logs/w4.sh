#!/bin/bash
# 周末 4 物体批次(2026-09-19)的**动态派活**调度器。
#
# 不是写死顺序的链条:队列里每行是「优先级 + 任务 id」,两张卡各跑一个 worker,
# 每轮重新读队列,抢**依赖已满足且还没被认领**的最高优先级任务。因此:
#   · 一条任务失败只写 failed 标记,后面的照跑(除非确实依赖它)
#   · 哪张卡先空哪张卡先领活,不按卡号写死分配
#   · queue.txt 随时可以追加新任务,worker 下一轮就看得见
#
# teacher 复用 models/rl/teacher_score_g4_diam_angle_s2(v1=100%、回报 276.2,
# 与现有全部结果同一个 teacher —— 配置完全一致,且让 4 物体结果与 2 物体结果直接可比)。
L=/home/tb/.claude/jobs/2174f11a/tmp; W=$L/w4
mkdir -p $W/claims $W/done $W/failed
S=$W/state.txt; Q=$W/queue.txt
cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac

TEACH=models/rl/teacher_score_g4_diam_angle_s2/last.zip
DATA=./data_w4
VR=models/vision/w4

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $S; }
gate(){ ( flock 9; sleep 90 ) 9>/tmp/isaac_start.lock; }
# timeout 只杀 isaaclab.sh 外壳,里面的 python 会变孤儿继续占显存 -> 后续静默 OOM。
# 起 Isaac 前清掉**本卡**上的残留;另一个 worker 只用另一张卡,所以本卡上的都不是别人的。
reap(){ local g=$1 pids
  pids=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")
  [ -n "$pids" ] && { log "REAP gpu$g 残留 $pids"; for q in $pids; do kill -9 $q 2>/dev/null; done; sleep 5; }
  return 0
}

# ---------- 任务实现 ----------
collect(){ G=$1
  [ -f $DATA/unit_00/transitions.csv ] && { log "SKIP collect(已有)"; return 0; }
  log "START collect gpu$G (4 物体场景,teacher 追红球)"
  reap $G; gate
  timeout 5400 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p scripts/collect_teacher_1m.py \
    --no-lateral --obs-mask diam --score-mode angle --multi --teacher $TEACH \
    --out-dir $DATA --transitions 120000 --units 1 --num-envs 64 --seed 1 --angle-deg 45 \
    > $L/w4_collect.log 2>&1
  local rc=$?; log "END   collect rc=$rc"
  [ -f $DATA/unit_00/transitions.csv ]
}

stage1(){ G=$1
  [ -f $VR/sup/donor.pt ] && { log "SKIP stage1(已有)"; return 0; }
  log "START stage1 gpu$G (A/V/R/sup 四臂 + init)"
  CUDA_VISIBLE_DEVICES=$G timeout 10800 python scripts/stage1_bc_heads.py \
    --out-root $VR --data $DATA --units 1 --a-target sampled --sup-target cam --donor-steps 4000 \
    > $L/w4_stage1.log 2>&1
  local rc=$?; log "END   stage1 rc=$rc"
  for A in A V R sup; do [ -f $VR/$A/donor.pt ] && { mkdir -p $VR/${A}_donor; cp $VR/$A/donor.pt $VR/${A}_donor/encoder_final.pt; }; done
  [ -f $VR/A_donor/encoder_final.pt ] && [ -f $VR/encoder_init.pt ]
}

ppo(){ A=$1; SLOT=$2; SD=$3; G=$4
  local SSFX="" SARG=""
  [ "$SD" != "1" ] && { SSFX="_s$SD"; SARG="--seed $SD"; }
  local TAG="${A}${SSFX}_w4_s${SLOT}"
  local RUN=models/rl/score_k03noy_student_${A}${SSFX}_w4_s${SLOT}
  [ -f $RUN/eval_score.csv ] && { log "SKIP ppo $TAG(已有)"; return 0; }
  log "START ppo $TAG gpu$G (槽位 $SLOT)"
  reap $G; gate
  timeout 14400 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p train/free_student.py \
    --arm $A --encoder-root $VR --score-mode angle --multi --target-slot $SLOT $SARG \
    --tag _w4_s${SLOT} > $L/w4_ppo_$TAG.log 2>&1
  log "END   ppo $TAG rc=$?"
  [ -f $RUN/last.pt ] || { log "FAIL ppo $TAG(没有 last.pt)"; return 1; }
  reap $G; gate
  timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py \
    --arm $A --encoder-root $VR --score-mode angle --multi --target-slot $SLOT $SARG \
    --run-tag _w4_s${SLOT} > $L/w4_eval_$TAG.log 2>&1
  grep -h RESULT $L/w4_eval_$TAG.log | sed "s/arm=[^ ]* /arm=$TAG /" >> $S
  for u in 000200 000210 000220 000230 000240; do
    reap $G; gate
    timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py \
      --arm $A --encoder-root $VR --score-mode angle --multi --target-slot $SLOT $SARG \
      --run-tag _w4_s${SLOT} --ckpt updates/ppo_$u.pt --tag _$u >/dev/null 2>&1
    grep -h RESULT $L/w4_eval_$TAG.log >/dev/null 2>&1
  done
  log "END   eval $TAG"
  return 0
}

# ---------- 依赖 ----------
# 0 = 可以跑;1 = 还没准备好;2 = 依赖已经失败,这条永远跑不了
ready(){ case "$1" in
  collect) return 0 ;;
  stage1)  [ -f $W/failed/collect ] && return 2; [ -f $W/done/collect ] && return 0; return 1 ;;
  ppo:*)   [ -f $W/failed/stage1 ] || [ -f $W/failed/collect ] && return 2
           [ -f $W/done/stage1 ] && return 0; return 1 ;;
  *)       return 0 ;;
esac }

dispatch(){ id=$1; G=$2
  case "$id" in
    collect) collect $G ;;
    stage1)  stage1 $G ;;
    ppo:*)   local _a _s _d; IFS=: read -r _ _a _s _d <<< "$id"; ppo "$_a" "$_s" "$_d" $G ;;
    *) log "未知任务 $id"; return 1 ;;
  esac }

# ---------- worker ----------
loop(){ G=$1
  # 等机制拆分那条链自己跑完(文件标记,不用 ps 匹配,避免自匹配)
  until grep -q "机制拆分的 PPO 行为确认全部完成" $L/g4_state.txt; do sleep 120; done
  log "worker gpu$G:上一条链已结束,开始消费队列"
  local idle=0
  while [ ! -f $W/stop ]; do
    got=""
    while read -r prio id; do
      case "$prio" in ""|\#*) continue;; esac
      [ -d "$W/claims/$id" ] && continue
      ready "$id"; r=$?
      if [ $r -eq 2 ]; then
        mkdir -p "$W/claims/$id"; touch "$W/failed/$id"; log "DEAD $id(依赖已失败,跳过)"; continue
      fi
      [ $r -ne 0 ] && continue
      if mkdir "$W/claims/$id" 2>/dev/null; then got="$id"; break; fi
    done < <(grep -v '^\s*$' $Q | sort -n)
    if [ -z "$got" ]; then
      idle=$((idle+1))
      # 队列里所有行都已认领且都已落定 -> 收工
      tot=$(grep -cve '^\s*$' $Q); fin=$(( $(ls $W/done 2>/dev/null|wc -l) + $(ls $W/failed 2>/dev/null|wc -l) ))
      [ "$fin" -ge "$tot" ] && { log "worker gpu$G:队列已排空($fin/$tot),退出"; break; }
      sleep 60; continue
    fi
    idle=0
    dispatch "$got" $G
    if [ $? -eq 0 ]; then touch "$W/done/$got"; else touch "$W/failed/$got"; log "FAILED $got(不阻塞其他任务)"; fi
  done
  log "worker gpu$G 退出"
}

log "=== 4 物体批次调度器启动:teacher=$TEACH ==="
log "队列 $(grep -cve '^\s*$' $Q) 条"
( loop 0 ) & ( loop 1 ) & wait
log "=== 4 物体批次全部结束 ==="
