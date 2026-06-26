#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

NUM_ENVS="${NUM_ENVS:-1}"
GAMES_NUM="${GAMES_NUM:-100}"
CFG="${CFG:-rl_games_ball/rl_games_ppo_cnn_lstm_asym.yaml}"

if [ -z "${CHECKPOINT:-}" ]; then
  echo "Set CHECKPOINT to a rl-games .pth file."
  exit 1
fi

./IsaacLab/isaaclab.sh -p rl_games_ball/train.py \
  --play \
  --cfg "${CFG}" \
  --num-envs "${NUM_ENVS}" \
  --games-num "${GAMES_NUM}" \
  --checkpoint "${CHECKPOINT}"
