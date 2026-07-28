#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# spr_z 模式冻结 encoder + 不存图(见 src/replay.py store_images),128 env 也只占 ~0.5GB buffer。
STUDENT_ENVS="${STUDENT_ENVS:-128}"
VAL_ENVS="${VAL_ENVS:-128}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
OBS_MODE="${OBS_MODE:-spr_z}"
CFG="${CFG:-drq_student_cfg}"  # offpolicy.py --cfg 传入的 cfg 模块(spr_z/pixels 用 drq_student_cfg,spr_coadapt 用 spr_coadapt_cfg)。
RANDOM_STOP="${RANDOM_STOP:-1}"
NOISE="${NOISE:-1}"
# Physical GPU for this pipeline. Isaac Sim's usdrt scenegraph only supports
# cuda:0, so we hide other GPUs via CUDA_VISIBLE_DEVICES instead of --device cuda:N.
# Default GPU 0: GPU 1 is usually occupied by the on-policy SPR line.
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
# also works on a GPU without a monitor attached (see run_spr_student_pipeline.sh).
env -u DISPLAY ./IsaacLab/isaaclab.sh -p train/offpolicy.py \
  --cfg "${CFG}" \
  --num-envs "${STUDENT_ENVS}" \
  "${ARGS[@]}"

env -u DISPLAY ./IsaacLab/isaaclab.sh -p val/offpolicy.py \
  --cfg "${CFG}" \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  "${ARGS[@]}"
