#!/bin/bash
# 补种子(2026-09-14):不训 teacher、不采数据、不重训表征,只在已有冻结表征上跑 PPO+评估。
# 轮 3   把周末批次缺的 8 条补齐到 n=3
# 轮 3b  随机表征锚点:另外 4 个 root 的 encoder_init(五份独立抽样)各跑一条
# 轮 4/5 全部 15 个 (teacher,臂) 的 seed 4、seed 5,顺序 R×5 → A×5 → V×5
L=/home/tb/.claude/jobs/2174f11a/tmp
S=$L/seeds_state.txt
cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
gate(){ ( flock 9; sleep 90 ) 9>/tmp/isaac_start.lock; }
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $S; }
ppo(){ # $1 mask $2 arm $3 seed $4 gpu
  M=$1; A=$2; SD=$3; G=$4; R=models/vision/wk_$M
  SFX=""; SARG=""; [ "$SD" != "1" ] && { SFX="_s$SD"; SARG="--seed $SD"; }
  N=${A}${SFX}_wk_$M; RUN=models/rl/score_k03noy_student_$N
  [ -f $RUN/eval_score.csv ] && { log "SKIP $N"; return 0; }
  log "START ppo $N gpu$G"
  gate; timeout 14400 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p train/free_student.py --arm $A --encoder-root $R $SARG --tag _wk_$M > $L/ppo_$N.log 2>&1
  log "END   ppo $N rc=$?"
  [ -f $RUN/last.pt ] || { log "FAIL $N 无 last.pt"; return 1; }
  gate; timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py --arm $A --encoder-root $R $SARG --run-tag _wk_$M > $L/eval_$N.log 2>&1
  grep -h RESULT $L/eval_$N.log | sed "s/arm=[^ ]* /arm=$N /" >> $S
  for u in 000200 000210 000220 000230 000240; do
    gate; timeout 900 env CUDA_VISIBLE_DEVICES=$G ./IsaacLab/isaaclab.sh -p val/score_student.py --arm $A --encoder-root $R $SARG --run-tag _wk_$M --ckpt updates/ppo_$u.pt --tag _$u >/dev/null 2>&1
  done
  log "END   eval $N"
}
round(){ mapfile -t J < $1
  ( i=0; for c in "${J[@]}"; do [ $((i%2)) -eq 0 ] && ppo $c 0; i=$((i+1)); done ) &
  ( i=0; for c in "${J[@]}"; do [ $((i%2)) -eq 1 ] && ppo $c 1; i=$((i+1)); done ) &
  wait; log "ROUND $1 完成"
}
cat > $L/r3.txt <<'X'
xyd R_donor 3
diam R_donor 3
bbox R_donor 3
diam A_donor 3
bbox A_donor 3
diam V_donor 3
bbox V_donor 3
ang init 3
X
cat > $L/r3b.txt <<'X'
diam init 1
bbox init 1
xd init 1
xyd init 1
X
for SD in 4 5; do : > $L/r$SD.txt
  for A in R_donor A_donor V_donor; do for M in ang xd xyd diam bbox; do echo "$M $A $SD" >> $L/r$SD.txt; done; done
done
log "=== 补种子开始:轮3(8) 轮3b(4) 轮4(15) 轮5(15) ==="
round $L/r3.txt; round $L/r3b.txt; round $L/r4.txt; round $L/r5.txt
log "=== 全部完成 ==="
