#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

STUDENT_ENVS="${STUDENT_ENVS:-128}"
VAL_ENVS="${VAL_ENVS:-128}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
RANDOM_STOP="${RANDOM_STOP:-0}"
NOISE="${NOISE:-1}"
# Physical GPU for this pipeline. Isaac Sim's usdrt scenegraph only supports
# cuda:0 ("GPUs other than cuda:0 are not currently supported"), so instead of
# passing --device cuda:1 we hide all other GPUs via CUDA_VISIBLE_DEVICES;
# the chosen GPU is then always cuda:0 from the process's point of view.
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"
DEVICE="${DEVICE:-cuda:0}"

ARGS=()
if [ "${RANDOM_STOP}" = "1" ]; then
  ARGS+=(--random-stop)
fi
if [ "${NOISE}" = "1" ]; then
  ARGS+=(--noise)
fi
if [ -n "${DEVICE}" ]; then
  ARGS+=(--device "${DEVICE}")
fi

# GPU1 has no monitor attached, so its X screen can't hand Kit a working
# GLX/Vulkan presenting queue (GLXBadFBConfig). Unsetting DISPLAY makes
# Kit fall back to pure off-screen rendering (see simulation_app.py),
# which sidesteps X entirely instead of relying on it.
env -u DISPLAY ./IsaacLab/isaaclab.sh -p train/spr_student.py \
  --num-envs "${STUDENT_ENVS}" \
  "${ARGS[@]}"

env -u DISPLAY ./IsaacLab/isaaclab.sh -p val/spr_student.py \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  "${ARGS[@]}"
