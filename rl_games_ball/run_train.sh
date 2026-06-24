#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

NUM_ENVS="${NUM_ENVS:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-10000}"
CFG="${CFG:-rl_games_ball/rl_games_ppo_cnn_lstm_asym.yaml}"

./IsaacLab/isaaclab.sh -p rl_games_ball/train.py \
  --cfg "${CFG}" \
  --num-envs "${NUM_ENVS}" \
  --max-epochs "${MAX_EPOCHS}"
