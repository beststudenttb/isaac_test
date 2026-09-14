#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# SAC + HER + SPR 共适应流水线:train/sac.py(--obs-mode spr_coadapt)训完接 val/sac.py 评估。
# 与 scripts/run_sac_pipeline.sh 同一套 train/val 脚本,只是 cfg 换成 sac_coadapt_cfg
# (encoder 参数抄 spr_coadapt_cfg、算法参数抄 sac_cfg,双单变量对照)。
#
# 一张卡启动(开 tmux):
#   GPU=0 ./scripts/run_sac_coadapt_pipeline.sh
# 冒烟(8 env / 2000 步,几分钟走完全链路后自然退出,产物落 models/rl/sac_coadapt_smoke):
#   GPU=0 CFG=sac_coadapt_smoke_cfg STUDENT_ENVS=8 VAL_ENVS=8 ./scripts/run_sac_coadapt_pipeline.sh
STUDENT_ENVS="${STUDENT_ENVS:-64}"  # 存图 64x4096x224x224x3 ≈ 39GB RAM。
VAL_ENVS="${VAL_ENVS:-128}"
VAL_EPISODES="${VAL_EPISODES:-1}"
VAL_START="${VAL_START:-0}"
VAL_STRIDE="${VAL_STRIDE:-1}"
OBS_MODE="${OBS_MODE:-spr_coadapt}"
CFG="${CFG:-sac_coadapt_cfg}"
RANDOM_STOP="${RANDOM_STOP:-0}"  # 先跑固定任务;=1 则随机 stop。
NOISE="${NOISE:-1}"              # 与 spr_coadapt(DDPG 对照)同条件,带干扰球。
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
