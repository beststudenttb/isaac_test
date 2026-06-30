#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

RANDOM_STOP_FLAG=()
if [[ "${1:-}" == "--random-stop" ]]; then
  RANDOM_STOP_FLAG=(--random-stop)
fi

STUDENT_TEACHER_LOSS="${STUDENT_TEACHER_LOSS:-1.0}"
STUDENT_VAL_ENVS="${STUDENT_VAL_ENVS:-32}"
STUDENT_VAL_EPISODES="${STUDENT_VAL_EPISODES:-1}"
STUDENT_VAL_START="${STUDENT_VAL_START:-0}"
STUDENT_VAL_STRIDE="${STUDENT_VAL_STRIDE:-1}"

./IsaacLab/isaaclab.sh -p train/teacher.py "${RANDOM_STOP_FLAG[@]}"

./IsaacLab/isaaclab.sh -p val/teacher.py "${RANDOM_STOP_FLAG[@]}"

./IsaacLab/isaaclab.sh -p train/student.py \
  "${RANDOM_STOP_FLAG[@]}" \
  --teacher-loss "${STUDENT_TEACHER_LOSS}"

./IsaacLab/isaaclab.sh -p val/student.py \
  "${RANDOM_STOP_FLAG[@]}" \
  --num-envs "${STUDENT_VAL_ENVS}" \
  --num-episodes "${STUDENT_VAL_EPISODES}" \
  --start "${STUDENT_VAL_START}" \
  --stride "${STUDENT_VAL_STRIDE}"
