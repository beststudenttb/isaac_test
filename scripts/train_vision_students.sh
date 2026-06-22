#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

train_one() {
  local state="$1"
  echo "[INFO] train cv-model=old state=${state}"
  ./IsaacLab/isaaclab.sh -p train/student_vision.py \
    --cv-model old \
    --state "${state}" \
    --student

  echo "[INFO] val cv-model=old state=${state}"
  ./IsaacLab/isaaclab.sh -p val/student_vision.py \
    --cv-model old \
    --state "${state}" \
    --num-envs 32 \
    --stride 1 \
    --student
}

train_one xd
train_one feature
train_one shared
