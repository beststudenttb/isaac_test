#!/usr/bin/env bash
# 分阶段 SPR-PPO Combo B(jsrl -> forward)训练 + 离线 val 一条龙。对照臂。
# 用法(默认 GPU0、frozen、noise env;并行跑 A/B 时另一张卡):
#     GPU=1 ./scripts/run_spr_phased_b.sh
set -e

cd "$(dirname "$0")/.."

COMBO="B"
ENCODER="${PHASED_ENCODER:-frozen}"
STUDENT_ENVS="${STUDENT_ENVS:-128}"
VAL_ENVS="${VAL_ENVS:-128}"  # 128 env 实测不 OOM(旧注释"64崩"已过时),更多样本。
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
RANDOM_STOP="${RANDOM_STOP:-0}"
NOISE="${NOISE:-1}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
DEVICE="${DEVICE:-cuda:0}"

OUT_DIR="./models/rl/spr_phased_$(echo "${COMBO}" | tr 'A-Z' 'a-z')_${ENCODER}"

ARGS=(--device "${DEVICE}")
if [ "${RANDOM_STOP}" = "1" ]; then
  ARGS+=(--random-stop)
  OUT_DIR="${OUT_DIR}_random_stop"
fi
if [ "${NOISE}" = "1" ]; then
  ARGS+=(--noise)
fi

echo "[pipeline] Combo=${COMBO} encoder=${ENCODER} gpu=${GPU} out=${OUT_DIR}"

PHASED_COMBO="${COMBO}" PHASED_ENCODER="${ENCODER}" \
  env -u DISPLAY ./IsaacLab/isaaclab.sh -p train/spr_phased.py \
    --num-envs "${STUDENT_ENVS}" \
    "${ARGS[@]}"

# val 复用 val/spr_jsrl.py(checkpoint 自包含),VAL_OUT_DIR 指到本次 phased 输出目录。
VAL_OUT_DIR="${OUT_DIR}" \
  env -u DISPLAY ./IsaacLab/isaaclab.sh -p val/spr_jsrl.py \
    --num-envs "${VAL_ENVS}" \
    --num-episodes "${VAL_EPISODES}" \
    --start "${VAL_START}" \
    --stride "${VAL_STRIDE}" \
    "${ARGS[@]}"
