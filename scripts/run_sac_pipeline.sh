#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# SAC + HER off-policy 流水线:train/sac.py 训完接 val/sac.py 评估。
# 逻辑与 scripts/run_drq_student_pipeline.sh 对齐,只是算法换 SAC、buffer 换 HER。
# 只支持 spr_z / pixels(不含 spr_coadapt)。默认固定任务(RANDOM_STOP=0)。
#
# 两张卡分别启动(各开一个 tmux):
#   GPU=0 OBS_MODE=pixels STUDENT_ENVS=64  ./scripts/run_sac_pipeline.sh
#   GPU=1 OBS_MODE=spr_z  STUDENT_ENVS=128 ./scripts/run_sac_pipeline.sh
STUDENT_ENVS="${STUDENT_ENVS:-64}"
VAL_ENVS="${VAL_ENVS:-128}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
OBS_MODE="${OBS_MODE:-spr_z}"
CFG="${CFG:-sac_cfg}"
RANDOM_STOP="${RANDOM_STOP:-0}"  # 先跑固定任务;=1 则随机 stop。
NOISE="${NOISE:-1}"              # 与 arm A(92%)同条件,带干扰球。
# Physical GPU for this pipeline. Isaac Sim's usdrt scenegraph only supports
# cuda:0, so we hide other GPUs via CUDA_VISIBLE_DEVICES instead of --device cuda:N.
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
DEVICE="${DEVICE:-cuda:0}"

ARGS=(--obs-mode "${OBS_MODE}")
if [ "${RANDOM_STOP}" = "1" ]; then
  ARGS+=(--random-stop)
fi
if [ "${NOISE}" = "1" ]; then
  ARGS+=(--noise)
fi
if [ -n "${DEVICE}" ]; then
  ARGS+=(--device "${DEVICE}")
fi

# Unsetting DISPLAY makes Kit fall back to pure off-screen rendering, which
# also works on a GPU without a monitor attached (see run_drq_student_pipeline.sh).
env -u DISPLAY ./IsaacLab/isaaclab.sh -p train/sac.py \
  --cfg "${CFG}" \
  --num-envs "${STUDENT_ENVS}" \
  "${ARGS[@]}"

env -u DISPLAY ./IsaacLab/isaaclab.sh -p val/sac.py \
  --cfg "${CFG}" \
  --num-envs "${VAL_ENVS}" \
  --num-episodes "${VAL_EPISODES}" \
  --start "${VAL_START}" \
  --stride "${VAL_STRIDE}" \
  "${ARGS[@]}"
