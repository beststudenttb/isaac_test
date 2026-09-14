#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# 任务 v2「悬停」on-policy 流水线(见 summary/2026_07_30_hover_v2_plan.md):
# train/spr_student.py --hover(悬停 env:只给状态分、无提前终止,跑满 15s)训完接
# val/spr_student.py --hover(v1 env + 切断 wrapper,success 判据与 88%/92% 可比)。
# 除 --hover 外与 scripts/run_spr_student_pipeline.sh 完全一致;输出目录自动带 _hover 后缀
# (models/rl/spr_student_hover),不碰现有 spr_student 结果。
#
#   GPU=1 ./scripts/run_spr_hover_pipeline.sh
STUDENT_ENVS="${STUDENT_ENVS:-128}"
VAL_ENVS="${VAL_ENVS:-32}"   # val/spr_student_cfg.py 注释:64 会在 val 启动时内存崩溃。
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
RANDOM_STOP="${RANDOM_STOP:-0}"
NOISE="${NOISE:-1}"
# Physical GPU for this pipeline. Isaac Sim's usdrt scenegraph only supports
# cuda:0, so we hide other GPUs via CUDA_VISIBLE_DEVICES instead of --device cuda:N.
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"
DEVICE="${DEVICE:-cuda:0}"

ARGS=(--hover)
if [ "${RANDOM_STOP}" = "1" ]; then
  ARGS+=(--random-stop)
fi
if [ "${NOISE}" = "1" ]; then
  ARGS+=(--noise)
fi
if [ -n "${DEVICE}" ]; then
  ARGS+=(--device "${DEVICE}")
fi

# Unsetting DISPLAY makes Kit fall back to pure off-screen rendering, which
# also works on a GPU without a monitor attached (see run_spr_student_pipeline.sh).
env -u DISPLAY ./IsaacLab/isaaclab.sh -p train/spr_student.py \
  --num-envs "${STUDENT_ENVS}" \
  "${ARGS[@]}"

env -u DISPLAY ./IsaacLab/isaaclab.sh -p val/spr_student.py \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  "${ARGS[@]}"
