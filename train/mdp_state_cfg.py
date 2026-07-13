"""MDP state representation config."""

from pathlib import Path


DATASET_DIR = Path("./data_mdp_teacher_50k_noise")  # 离线 MDP 数据集目录。
OUT_DIR = Path("./models/vision/mdp_state_priv_probe_grad")  # MDP state 模型输出目录。
DEVICE = "auto"  # 训练设备，auto/cpu/cuda。

INPUT_SHAPE = (224, 224, 3)  # 输入图像尺寸，HWC。
Z_DIM = 32  # 图像 latent z 维度。
ACTION_DIM = 3  # 动作维度，x/y/w。
BELIEF_DIM = 128  # GRU 记忆 h 维度。
FEATURE_DIM = 256  # ResNet layer3 输出通道数。
HIDDEN_DIM = 128  # MLP hidden size。
PRETRAINED = True  # ResNet18 是否使用 ImageNet 预训练。
FREEZE_BACKBONE = True  # 是否冻结 ResNet 主干。
TRAIN_LAYER3 = True  # FREEZE_BACKBONE=True 时是否解锁 layer3。

BATCH_SIZE = 16  # 每次更新采样多少条轨迹窗口。
SEQ_LEN = 16  # 每条轨迹窗口长度。
UPDATES = 2000  # 训练 update 数。
LR = 1e-4  # 学习率。
GAMMA = 0.99  # return 折扣因子。

DYN_W = 0.3  # z 动力学预测 loss 权重。
CONTRAST_W = 0.0  # transition contrastive loss 权重（已弃用，保持 0）。
CONTRAST_TAU = 0.1  # contrastive softmax 温度。
IDM_W = 0.1  # 逆动力学 loss 权重。
VAR_W = 0.1  # VICReg 方差项权重（防点坍缩、控离散度）。
COV_W = 0.05  # VICReg 协方差项权重（各维去相关、防维度坍缩）。
VAR_GAMMA = 1.0  # 方差项目标 std 阈值。
REWARD_W = 0.0  # reward 预测 loss 权重。
VALUE_W = 0.0  # value 预测 loss 权重。
DONE_W = 0.0  # done 预测 loss 权重。
PROBE_W = 1.0  # px/dist probe loss 权重。

SAVE_EVERY = 9999990  # 每多少 update 保存一次 checkpoint。
SEED = 0  # 随机种子。
