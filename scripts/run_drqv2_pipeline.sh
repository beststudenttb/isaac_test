#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

TRAIN_ENVS="${TRAIN_ENVS:-32}"
VAL_ENVS="${VAL_ENVS:-32}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
RANDOM_STOP="${RANDOM_STOP:-0}"
NOISE="${NOISE:-1}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
DEVICE="${DEVICE:-cuda:0}"

ARGS=()
if [ "${RANDOM_STOP}" = "1" ]; then
  ARGS+=(--random-stop)
fi
if [ "${NOISE}" = "1" ]; then
  ARGS+=(--noise)
fi
if [ -n "${DEVICE}" ]; then
  ARGS+=(--device "${DEVICE}")
fi

env -u DISPLAY ./IsaacLab/isaaclab.sh -p train/drqv2.py \
  --num-envs "${TRAIN_ENVS}" \
  "${ARGS[@]}"

env -u DISPLAY ./IsaacLab/isaaclab.sh -p val/drqv2.py \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  "${ARGS[@]}"
