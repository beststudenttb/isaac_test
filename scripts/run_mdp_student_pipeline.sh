#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

STUDENT_ENVS="${STUDENT_ENVS:-64}"
MDP_DIR="${MDP_DIR:-./models/vision/mdp_state_priv_probe_grad}"
MDP_CKPT="${MDP_CKPT:-${MDP_DIR}/last.pt}"
VAL_ENVS="${VAL_ENVS:-64}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
NOISE_ARG=()
if [ "${NOISE:-0}" = "1" ]; then
  NOISE_ARG=(--noise)
fi

./IsaacLab/isaaclab.sh -p train/mdp_student.py \
  --num-envs "${STUDENT_ENVS}" \
  --mdp "${MDP_CKPT}" \
  "${NOISE_ARG[@]}"

./IsaacLab/isaaclab.sh -p val/mdp_student.py \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  --mdp "${MDP_CKPT}" \
  "${NOISE_ARG[@]}"
