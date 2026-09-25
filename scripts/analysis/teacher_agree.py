"""四个 teacher 的行为是不是真的一样?在同一批渲染状态上比它们的确定性动作。"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from stable_baselines3 import PPO
D=Path("data_factorial"); rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
px,d=px[seen],d[seen]
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng=d/np.cos(np.radians(bear))
half=112.0; r_=31.06/d
OBS={"diam":np.stack([px/half-1,np.clip(2*r_/224,0,1),np.full_like(d,1.5/8),np.zeros_like(d)],1),
     "ang":np.stack([np.clip(bear/40,-1,1),np.clip(rng/8,0,1),np.full_like(d,1.5/8),np.zeros_like(d)],1)}
import glob
A={}
for comb in ["diam_cam","diam_angle","ang_cam","ang_angle"]:
    z=sorted(glob.glob(f"models/rl/teacher_score_g4_{comb}_s*/last.zip"))
    pol=PPO.load(z[-1],device="cpu").policy
    o=torch.as_tensor(OBS[comb.split("_")[0]],dtype=torch.float32)
    with torch.no_grad(): a=pol.action_net(pol.mlp_extractor.policy_net(o)).numpy()
    A[comb]=np.clip(a,-1,1)
ks=list(A)
print(f"在 {seen.sum()} 个受控渲染状态上,四个 teacher 的确定性动作两两比较")
print(f"{'':22s}{'a_x 相关':>9s}{'a_w 相关':>9s}{'a_x RMSE':>10s}{'a_w RMSE':>10s}")
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        x,y=A[ks[i]],A[ks[j]]
        c0=np.corrcoef(x[:,0],y[:,0])[0,1]; c1=np.corrcoef(x[:,1],y[:,1])[0,1]
        print(f"{ks[i]+' vs '+ks[j]:22s}{c0:9.3f}{c1:9.3f}{np.sqrt(((x[:,0]-y[:,0])**2).mean()):10.3f}{np.sqrt(((x[:,1]-y[:,1])**2).mean()):10.3f}")
print(f"\n各 teacher 动作的标准差(作为 RMSE 的参照): "+"  ".join(f"{k}: {A[k].std(0).round(3)}" for k in ks))
