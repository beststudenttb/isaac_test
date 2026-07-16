"""JSRL + 反向课程 SPR-PPO 配置。

冻结的 stage1_end.pt encoder(与 arm A 逐字节相同的 z)+ PPO + 反向课程。回答:同一个 DrQ-v2
解出 100% 的冻结 z,PPO 配上"teacher 带路、student 只学最后一段"能不能也解出来。
"""

from pathlib import Path


OUT_DIR = Path("./models/rl/spr_jsrl")  # JSRL 输出目录。
RANDOM_STOP_OUT_DIR = Path("./models/rl/spr_jsrl_random_stop")  # --random-stop 时的输出目录。
CLEAR_OUT_DIR = True

# 冻结 encoder 来源。指向保全的副本(md5 cfd52c0f),就是 arm A 用的那一份;
# models/rl/spr_student/ 会被 SPR 跑的 CLEAR_OUT_DIR 清掉,所以不指那里。
SPR_CKPT = Path("./models/rl/_encoders/stage1_sigma_ablation.pt")

TEACHER_PATH = "./models/rl/teacher_ppo/best_val.zip"  # 固定任务 teacher(反向课程的带路者)。
RANDOM_TEACHER_PATH = "./models/rl/teacher_ppo_random_stop/best_val.zip"

NUM_ENVS = 128
N_STEPS = 64  # 每次 update 每个 env 收集多少个 **student** 步(approach B:只算 student)。
UPDATES = 1000
EPISODE_S = 15.0
STOP_N = 1  # 训练时 stop 一次即 success;val 用 3。
SEED = 1
DEVICE = None

RENDERING_MODE = "balanced"
ANTIALIASING_MODE = "Off"
DLSS_MODE = None
RERENDER_ON_RESET = 1

END_D_MIN = 1.5
END_D_MAX = 1.5
END_X_MIN = 0.0
END_X_MAX = 0.0
RANDOM_END_D_MIN = 1.3
RANDOM_END_D_MAX = 1.8
RANDOM_END_X_MIN = -20.0
RANDOM_END_X_MAX = 20.0

# ---- 反向课程 / JSRL ----
# 交棒条件(不使用任何特权信息,只看 teacher 自己的动作幅度):
#     handover = (episode_step >= h) or (max|a_teacher| < STOP_EPS)
# 第二个条件让 teacher 一想停车就撒手 —— 否则 h 大时近距离的简单 episode 会被 teacher 做完,
# 剩给 student 的全是远距离难 episode,课程被系统性偏置(实测干净 teacher h=150 时 52% 被做完)。
# 离散阶梯:课程档位 = f * H_BASE,f ∈ {0, 1/N, …, (N-1)/N}(N=JSRL_N_RUNGS=10)。
# h ∈ {0, 15, 30, …, 135}。从最高档(f=0.9,h=135,teacher 帮最多)起手,学得动降一档、学不动回一档。
# 固定步长阶梯比乘性衰减干净,不会一步跨过断崖(乘性 112→84 恰好跨过 teacher 首次进区中位 86)。
# 注意:f≥0.6(h≥90>86)几档会塌成同一难度(teacher 都在 ~86 步"想停"时交棒),会被快速降过;
# 真正有区分度的课程在 f≤0.5。
JSRL_H_BASE = 150  # 阶梯基准步数;干净 teacher 首次进 stop 区中位 86、90% 分位 140。
JSRL_N_RUNGS = 10  # 档数;f = [0.0, 0.1, …, 0.9]。
JSRL_UP_THRESH = 0.80  # student 成功率 EMA > 此值 -> 降一档(策略够稳才前移,治 h→0 过早退化)。
JSRL_DOWN_THRESH = 0.30  # EMA < 此值 -> 回一档(回火)。
JSRL_EMA = 0.90  # 成功率 EMA;有效窗口 1/(1-α)=10 个 update。
JSRL_COOLDOWN = 10  # 调过档后至少等这么多 update 再调,让 EMA 换掉旧课程样本。
TEACHER_NOISE_STD = 0.0  # guide 阶段 teacher 动作噪声;0=干净带路。
MAX_COLLECT_ITERS = 4000  # 收集循环的硬上限(防 h 过大时永远填不满列时死循环)。

# ---- PPO ----
BATCH_SIZE = 256  # student 步的 minibatch;encoder 冻结,可以开大。
N_EPOCHS = 4
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
NORMALIZE_ADVANTAGE = True
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

POLICY_NET = [64, 64]
VALUE_NET = [64, 64]
ACTIVATION = "tanh"
STD_INIT = 0.2  # log_std 可训练;配 STD_MAX=0.1 会在第一步被 clamp 拽到 0.1,无所谓。
STD_MIN = 0.02
STD_MAX = 0.1  # 强行按住 σ。实际只咬 a_w(转头维,jsrl 里顶死 0.3);a_x/a_y 本就自己收到 0.02。
# 假设:a_w 的 σ 高是训练时 advantage 噪声大、害 μ_w 收不敛 -> 压低 σ 让 μ_w 收敛变准(A)。
# 反面(B):critic 对 a_w 系统性偏,压 σ 只给个安静的错 μ,xe 仍 ~20px。用确定性 val 的 xe/de 分辨:
# 干净且 xe<10px、de<0.2m = A(成);干净但 xe~20px = B,on-policy 受限,停。

SAVE_UPDATE_EVERY = 10  # 每多少 update 存一次 full_{update:06d}.pt。
LOG_EVERY = 1
