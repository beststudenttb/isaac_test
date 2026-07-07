#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

STUDENT_ENVS="${STUDENT_ENVS:-64}"
VAL_ENVS="${VAL_ENVS:-32}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
RANDOM_STOP="${RANDOM_STOP:-0}"

ARGS=()
if [ "${RANDOM_STOP}" = "1" ]; then
  ARGS+=(--random-stop)
fi
if [ "${NOISE:-0}" = "1" ]; then
  ARGS+=(--noise)
fi

./IsaacLab/isaaclab.sh -p train/spr_student.py \
  --num-envs "${STUDENT_ENVS}" \
  "${ARGS[@]}"

./IsaacLab/isaaclab.sh -p val/spr_student.py \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  "${ARGS[@]}"
