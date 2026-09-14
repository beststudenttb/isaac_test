"""任务三:用 stage2 长出来的冻结表征,从零训 PPO。

交接文档 2026-09-03 §3 任务三。判据就在这一步:每个冻结表征各训一个 PPO,
比成功率/步数。teacher 拉扯必须彻底关掉,否则测的是蒸馏不是表征。
"""

from pathlib import Path


OUT_ROOT = Path("./models/rl/score_k03noy_student")  # 实际目录会追加臂名,如 free_student_A。
ENCODER_ROOT = Path("./models/vision/score_k03noy")  # stage2 的输出;每臂读 {arm}/encoder_final.pt。
NO_LATERAL = True  # 2026-09-09:动作空间 [a_x, a_w]。teacher 不用 a_y(区内 |mu_y| 0.016),student 却被它拖出平台期/公转。
CLEAR_OUT_DIR = True

NUM_ENVS = 192  # 2026-09-09 实测稳态显存:128→8.4GB(68%) 192→10.5GB(85%) 216→11.4GB(93%) 256→11.7GB(96%) 512 起不来;
                # 吞吐 192 起就平(907 fps)。取 192:独立轨迹数是 128 的 1.5 倍,余量 1.8 GB。
N_STEPS = 176   # 192×176 = 33792 = 33 个整 minibatch(1024),和 teacher 的 32768/次同量级;rollout 覆盖 47% 集长,装得下 λ=0.99 的 50 步视野。
UPDATES = 242   # 总步数 192×176×242 = 8.18M,与 teacher 的 8.19M 对齐。
TOTAL_STEPS = NUM_ENVS * N_STEPS * UPDATES
EPISODE_S = 15.0
STOP_N = 1
SEED = 1
DEVICE = None

ANGLE_DEG = 45.0  # 表征是在 ±45 的数据上长出来的,任务三对齐它。

RENDERING_MODE = "balanced"
ANTIALIASING_MODE = "Off"
DLSS_MODE = None
RERENDER_ON_RESET = 1

END_D_MIN = 1.5  # 固定 stop 目标:goal 退化成常量,与 stage2 的数据设定一致。
END_D_MAX = 1.5
END_X_MIN = 0.0
END_X_MAX = 0.0

# teacher 彻底关掉。stage_teacher_coef 在 TEACHER_LOSS=0 时全程返回 0.0,
# 不存在 Webots §5 那个 max(min_coef, ...) 钳位的坑(本仓库没有 min_coef)。
TEACHER_PATH = "./models/rl/teacher_score_k0.3_noy/last.zip"  # 仅用于轨迹 CSV 的对照列,不进 loss。
TEACHER_LOSS = 0.0
STAGE1_UPDATES = 0
STAGE2_UPDATES = 0
STAGE1_STEPS = 0
STAGE2_STEPS = 0

BATCH_SIZE = 1024  # 对齐 teacher 自己的纯 PPO(无 BC,训到 98.8%)。
                   # 原值 32 抄自 spr_fixed_student_cfg,那些配置都有 teacher BC 拽着;
                   # 无锚时 batch 32 每个 update 走 512 个噪声梯度步,实测 KL 飞到 6.4。
                   # 注:这只是把已收集的 4096 个样本怎么切片,不增加任何渲染。
N_EPOCHS = 10      # 同上,对齐 teacher。
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.99  # 视野 17→50 步,少信 critic 的 bootstrap;N_STEPS=256 装得下。
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
NORMALIZE_ADVANTAGE = True
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

POLICY_NET = [64, 64]
VALUE_NET = [64, 64]
ACTIVATION = "tanh"
STD_INIT = 0.3
STD_MAX = 0.3

SAVE_UPDATE_EVERY = 10
VAL_EVERY = 0
VAL_STEPS = 0
SAVE_BEST = True
BEST_WARMUP = 20
BEST_MARGIN = 0.02
LOG_EVERY = 1
