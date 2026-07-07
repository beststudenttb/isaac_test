#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

CV_MODEL="${CV_MODEL:-old}"
CV_STATES="${CV_STATES:-shared}"
CV_ENVS="${CV_ENVS:-64}"
MDP_ENVS="${MDP_ENVS:-64}"
VAL_ENVS="${VAL_ENVS:-128}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-10}"
MDP_CKPT="${MDP_CKPT:-./models/vision/mdp_state_priv_probe_grad/last.pt}"
RANDOM_STOP_FLAG=(--random-stop)
NOISE_FLAG=()
if [ "${NOISE:-0}" = "1" ]; then
  NOISE_FLAG=(--noise)
fi

for STATE in ${CV_STATES}; do
  echo "[INFO] train vision student cv-model=${CV_MODEL} state=${STATE}"
  ./IsaacLab/isaaclab.sh -p train/student_vision.py \
    --cv-model "${CV_MODEL}" \
    --state "${STATE}" \
    --num-envs "${CV_ENVS}" \
    "${RANDOM_STOP_FLAG[@]}" \
    "${NOISE_FLAG[@]}" \
    --student

  echo "[INFO] val vision student cv-model=${CV_MODEL} state=${STATE}"
  ./IsaacLab/isaaclab.sh -p val/student_vision.py \
    --cv-model "${CV_MODEL}" \
    --state "${STATE}" \
    --num-envs "${VAL_ENVS}" \
    --num-episodes "${VAL_EPISODES}" \
    --start "${VAL_START}" \
    --stride "${VAL_STRIDE}" \
    "${RANDOM_STOP_FLAG[@]}" \
    "${NOISE_FLAG[@]}" \
    --student
done

echo "[INFO] train MDP student mdp=${MDP_CKPT}"
./IsaacLab/isaaclab.sh -p train/mdp_student.py \
  --num-envs "${MDP_ENVS}" \
  --mdp "${MDP_CKPT}" \
  "${RANDOM_STOP_FLAG[@]}" \
  "${NOISE_FLAG[@]}"

echo "[INFO] val MDP student mdp=${MDP_CKPT}"
./IsaacLab/isaaclab.sh -p val/mdp_student.py \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  --mdp "${MDP_CKPT}" \
  "${RANDOM_STOP_FLAG[@]}" \
  "${NOISE_FLAG[@]}"
