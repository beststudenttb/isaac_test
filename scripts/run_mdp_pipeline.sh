#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

NUM_ENVS="${NUM_ENVS:-64}"
TRANSITIONS="${TRANSITIONS:-50000}"
OUT_DIR="${OUT_DIR:-./models/vision/mdp_state_priv_probe_grad}"
UPDATES="${UPDATES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-16}"
NOISE="${NOISE:-0}"
NOISE_ARG=()
if [ "${NOISE}" = "1" ]; then
  DATA_DIR="${DATA_DIR:-./data_mdp_teacher_50k_noise}"
  NOISE_ARG=(--noise)
else
  DATA_DIR="${DATA_DIR:-./data_mdp_teacher_50k}"
fi

./IsaacLab/isaaclab.sh -p scripts/collect_mdp_dataset.py \
  --num-envs "${NUM_ENVS}" \
  --transitions "${TRANSITIONS}" \
  --out-dir "${DATA_DIR}" \
  --render-mode balanced \
  --aa Off \
  "${NOISE_ARG[@]}"

python train/mdp_state.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --updates "${UPDATES}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}"
