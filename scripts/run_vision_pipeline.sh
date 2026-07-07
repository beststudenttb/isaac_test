#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

NOISE="${NOISE:-0}"
SAMPLES="${SAMPLES:-50000}"
CV_MODEL="${CV_MODEL:-old}"
CV_STATES="${CV_STATES:-xd shared}"
CV_UPDATES="${CV_UPDATES:-1000}"
CV_BATCH_SIZE="${CV_BATCH_SIZE:-128}"
STUDENT_ENVS="${STUDENT_ENVS:-64}"
VAL_ENVS="${VAL_ENVS:-32}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-10}"
RANDOM_STOP="${RANDOM_STOP:-0}"
NOISE_ARG=()
if [ "${NOISE}" = "1" ]; then
  DATA_DIR="${DATA_DIR:-./data_isaac_noise}"
  CV_OUT_DIR="${CV_OUT_DIR:-./models/vision/cv_old_noise}"
  NOISE_ARG=(--noise)
else
  DATA_DIR="${DATA_DIR:-./data_isaac}"
  CV_OUT_DIR="${CV_OUT_DIR:-./models/vision/cv_old}"
fi
RANDOM_STOP_ARG=()
if [ "${RANDOM_STOP}" = "1" ]; then
  RANDOM_STOP_ARG=(--random-stop)
fi
CV_CKPT="${CV_OUT_DIR}/best.pt"

./IsaacLab/isaaclab.sh -p scripts/collect_cv_dataset.py \
  --samples "${SAMPLES}" \
  --out-dir "${DATA_DIR}" \
  --render-mode balanced \
  --aa DLSS \
  --dlss-mode 0 \
  "${NOISE_ARG[@]}"

python train/cv.py \
  --train \
  --model "${CV_MODEL}" \
  --updates "${CV_UPDATES}" \
  --batch-size "${CV_BATCH_SIZE}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${CV_OUT_DIR}" \
  --device cuda

for STATE in ${CV_STATES}; do
  echo "[INFO] train vision student cv-model=${CV_MODEL} state=${STATE} cv=${CV_CKPT}"
  ./IsaacLab/isaaclab.sh -p train/student_vision.py \
    --cv-model "${CV_MODEL}" \
    --state "${STATE}" \
    --cv-ckpt "${CV_CKPT}" \
    --num-envs "${STUDENT_ENVS}" \
    --student \
    "${RANDOM_STOP_ARG[@]}" \
    "${NOISE_ARG[@]}"

  echo "[INFO] val vision student cv-model=${CV_MODEL} state=${STATE} cv=${CV_CKPT}"
  ./IsaacLab/isaaclab.sh -p val/student_vision.py \
    --cv-model "${CV_MODEL}" \
    --state "${STATE}" \
    --cv-ckpt "${CV_CKPT}" \
    --num-envs "${VAL_ENVS}" \
    --num-episodes "${VAL_EPISODES}" \
    --start "${VAL_START}" \
    --stride "${VAL_STRIDE}" \
    --student \
    "${RANDOM_STOP_ARG[@]}" \
    "${NOISE_ARG[@]}"
done
