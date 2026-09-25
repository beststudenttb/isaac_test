"""静态验证:chase_blue 打开后,project_target 是否真的跟着蓝球。不训练,只建环境看标签。"""
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import argparse
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
a=parser.parse_args(); a.headless=True; a.livestream=0; a.enable_cameras=True
app=AppLauncher(a); sim=app.app
sys.path.insert(0,str(Path(__file__).resolve().parents[0]))
sys.path.insert(0,"/home/tb/Downloads/isaac_test")
import torch, math
import isaaclab.sim as sim_utils
from src.score_env import ScoreNoiseStudentEnvCfg, make_score_noise_student_env
for flag in (True,):
    cfg=ScoreNoiseStudentEnvCfg(); cfg.seed=123; cfg.episode_length_s=600.0; cfg.stop_n=1
    cfg.scene.num_envs=16; cfg.sim.device=a.device; cfg.action_space=2
    cfg.sim.render=sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="Off", enable_dlssg=False)
    cfg.angle_deg=45.0; cfg.end_d_min=cfg.end_d_max=1.5; cfg.end_x_min=cfg.end_x_max=0.0
    cfg.chase_blue=flag
    env=make_score_noise_student_env(cfg); env.reset()
    lab=env.project_target()
    def proj(xy):
        d=xy-env.robot_xy; yaw=env.robot_yaw+env.head_yaw
        c,s=torch.cos(yaw),torch.sin(yaw); f=c*d[:,0]+s*d[:,1]; l=-s*d[:,0]+c*d[:,1]
        fx=112/math.tan(math.radians(40)); return 112-fx*l/torch.clamp(f,min=1e-6), f
    pr,_=proj(env.target_xy); pb,_=proj(env.noise_xy)
    vis=lab["dist"]>0.3
    er=float((lab["px_x"][vis]-pr[vis]).abs().mean()) if vis.any() else float('nan')
    eb=float((lab["px_x"][vis]-pb[vis]).abs().mean()) if vis.any() else float('nan')
    print(f"[VERIFY] chase_blue={flag}  可见 {int(vis.sum())}/16  |标签-红球投影| {er:.2f}px  |标签-蓝球投影| {eb:.2f}px", flush=True)
sim.close()
