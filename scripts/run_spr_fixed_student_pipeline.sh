#!/usr/bin/env bash
set -e

STUDENT_ENVS=64
VAL_ENVS=32
VAL_EPISODES=1
VAL_START=0
VAL_STRIDE=10
RANDOM_STOP=0
NOISE="${NOISE:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RANDOM_STOP_ARG=()
if [ "$RANDOM_STOP" = "1" ]; then
  RANDOM_STOP_ARG=(--random-stop)
fi
NOISE_ARG=()
if [ "$NOISE" = "1" ]; then
  NOISE_ARG=(--noise)
fi

./IsaacLab/isaaclab.sh -p train/spr_fixed_student.py --num-envs "$STUDENT_ENVS" "${RANDOM_STOP_ARG[@]}" "${NOISE_ARG[@]}"
./IsaacLab/isaaclab.sh -p val/spr_fixed_student.py --num-envs "$VAL_ENVS" --num-episodes "$VAL_EPISODES" --start "$VAL_START" --stride "$VAL_STRIDE" "${RANDOM_STOP_ARG[@]}" "${NOISE_ARG[@]}"
