#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

./IsaacLab/isaaclab.sh -p scripts/collect_cv_dataset.py \
  --samples 50000 \
  --render-mode balanced \
  --aa DLSS \
  --dlss-mode 0

python scripts/train_cv_all.py \
  --train \
  --updates 1000 \
  --batch-size 128 \
  --device cuda

./IsaacLab/isaaclab.sh -p train/student_vision.py \
  --cv-model old \
  --state xd \
  --student
