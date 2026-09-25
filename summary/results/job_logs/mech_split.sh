#!/bin/bash
# 机制拆分:目标的「维数 / 对称性 / 可提取性」在 A 与 R 之间完全共线,用三个新臂拆开。
# 只改 student 回归什么,不碰环境/teacher/数据,所以与已有的 A、R 完全可比。
#   Asym = (a_x, |a_w|)  2维 对称 好提取   -> 若像 R(留住干扰物)=> 对称性/可提取性是主因,维数不是
#   Aw   = a_w           1维 反对称 难提取 -> 若像 A(擦掉)       => 同上
#   Ax   = a_x           1维 对称 好提取   -> 应当像 R
L=/home/tb/.claude/jobs/2174f11a/tmp; cd /home/tb/Downloads/isaac_test
source /home/tb/miniconda3/etc/profile.d/conda.sh; conda activate isaac
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L/g4_state.txt; }
R=models/vision/g4_diam_angle; D=./data_g4_diam_angle
log "=== 机制拆分:stage1 训 Asym / Aw / Ax(gpu0,不碰环境与 teacher)==="
CUDA_VISIBLE_DEVICES=0 timeout 7200 python scripts/stage1_bc_heads.py \
  --out-root $R --data $D --units 1 --a-target sampled --sup-target cam \
  --donor-steps 4000 --arms Asym Aw Ax > $L/mech_stage1.log 2>&1
log "END stage1 rc=$? $(grep INFO $L/mech_stage1.log | grep dim= | tr '\n' ' ')"
for A in Asym Aw Ax; do mkdir -p $R/${A}_donor && cp $R/$A/donor.pt $R/${A}_donor/encoder_final.pt 2>/dev/null; done
CUDA_VISIBLE_DEVICES="" python $L/mech_probe.py > $L/mech_probe_out.txt 2>&1
log "END 机制拆分探针 -> mech_probe_out.txt"
