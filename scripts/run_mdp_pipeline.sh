#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

NUM_ENVS="${NUM_ENVS:-64}"
TRANSITIONS="${TRANSITIONS:-50000}"
NOISE="${NOISE:-1}"
NOISE_ARG=()
if [ "${NOISE}" = "1" ]; then
  DATA_DIR="./data_mdp_teacher_50k_noise"
  NOISE_ARG=(--noise)
else
  DATA_DIR="./data_mdp_teacher_50k"
fi

./IsaacLab/isaaclab.sh -p scripts/collect_mdp_dataset.py \
  --num-envs "${NUM_ENVS}" \
  --transitions "${TRANSITIONS}" \
  --out-dir "${DATA_DIR}" \
  --render-mode balanced \
  --aa Off \
  "${NOISE_ARG[@]}"

python train/mdp_state.py
