#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

STUDENT_ENVS="${STUDENT_ENVS:-128}"
MDP_DIR="${MDP_DIR:-./models/vision/mdp_state_priv_probe_grad}"
MDP_CKPT="${MDP_CKPT:-${MDP_DIR}/last.pt}"
VAL_ENVS="${VAL_ENVS:-128}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
NOISE="${NOISE:-1}"
# Pin this pipeline to one physical GPU (see run_spr_student_pipeline.sh for
# why the process-visible device is always cuda:0).
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
DEVICE="${DEVICE:-cuda:0}"
NOISE_ARG=()
if [ "${NOISE}" = "1" ]; then
  NOISE_ARG=(--noise)
fi
DEVICE_ARG=()
if [ -n "${DEVICE}" ]; then
  DEVICE_ARG=(--device "${DEVICE}")
fi

./IsaacLab/isaaclab.sh -p train/mdp_student.py \
  --num-envs "${STUDENT_ENVS}" \
  --mdp "${MDP_CKPT}" \
  "${NOISE_ARG[@]}" \
  "${DEVICE_ARG[@]}"

./IsaacLab/isaaclab.sh -p val/mdp_student.py \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  --mdp "${MDP_CKPT}" \
  "${NOISE_ARG[@]}" \
  "${DEVICE_ARG[@]}"
