"""把 BC 预训练的 adapter/head 装成 PPO 格式的 last.pt,好用 val/score_student.py 直接评"RL 之前"的策略。
用法: python make_bc_ckpt.py <bc_dir> <run_dir> <adapter_hidden> <backbone_pt>"""
import sys, torch, torch.nn as nn
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FrozenFreeEncoder
from src.free_ppo import AdapterActorCritic
bc, run, hidden, backbone = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
enc = FrozenFreeEncoder(backbone, **dict(FREE_SPATIAL_CONFIG))
m = AdapterActorCritic(encoder=enc, adapter_hidden=hidden, goal_dim=2, act_dim=2, pi_hidden=[64, 64], vf_hidden=[64, 64], activation=nn.Tanh, init_std=0.3)
m.adapter.load_state_dict(torch.load(bc / "adapter.pt", map_location="cpu")); m.actor.load_state_dict(torch.load(bc / "head.pt", map_location="cpu"))
run.mkdir(parents=True, exist_ok=True)
torch.save({"model": m.state_dict(), "update": 0, "step": 0, "best": -1.0, "teacher_coef": 0.0, "model_cfg": {"adapter": True, "adapter_hidden": hidden}, "info": {}}, run / "last.pt")
print("saved", run / "last.pt")
