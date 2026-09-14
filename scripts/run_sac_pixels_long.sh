#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# SAC + HER pixels 长训(8M)流水线:train/sac.py 训完接 val/sac.py 评估。
# 与 scripts/run_sac_pipeline.sh 同一套脚本,只是 cfg 换成 sac_pixels_long_cfg
# (TOTAL_ENV_STEPS=8M,输出 models/rl/sac_pixels_8m,不覆盖 2M 那版结果)。
#
#   GPU=0 ./scripts/run_sac_pixels_long.sh
STUDENT_ENVS="${STUDENT_ENVS:-128}"  # 存图 128x2048x224x224x3 ≈ 39.5GB RAM。
VAL_ENVS="${VAL_ENVS:-128}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"  # 8M / 25k = 320 个 checkpoint,val 约 2.5h;想快就设 2。
OBS_MODE="${OBS_MODE:-pixels}"
CFG="${CFG:-sac_pixels_long_cfg}"
RANDOM_STOP="${RANDOM_STOP:-0}"
NOISE="${NOISE:-1}"
# Physical GPU for this pipeline. Isaac Sim's usdrt scenegraph only supports
# cuda:0, so we hide other GPUs via CUDA_VISIBLE_DEVICES instead of --device cuda:N.
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
DEVICE="${DEVICE:-cuda:0}"

ARGS=(--obs-mode "${OBS_MODE}")
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
# also works on a GPU without a monitor attached (see run_drq_student_pipeline.sh).
env -u DISPLAY ./IsaacLab/isaaclab.sh -p train/sac.py \
  --cfg "${CFG}" \
  --num-envs "${STUDENT_ENVS}" \
  "${ARGS[@]}"

env -u DISPLAY ./IsaacLab/isaaclab.sh -p val/sac.py \
  --cfg "${CFG}" \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  "${ARGS[@]}"
