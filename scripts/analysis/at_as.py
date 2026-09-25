"""同一批受控状态上,同步比较 teacher 动作 a_T 与 student 动作 a_S(离线,不开仿真器)。
a_S = 冻结表征 + 监督动作头(BC 策略)。顺带缓存 z,后续分析复用。
回答:1) a_S 跟不跟得上 a_T,分区域;2) 换了无关外观后 a_S 变了多少;3) 给定 z 后动作还剩多少歧义。
"""
import sys, csv, math, numpy as np, torch
sys.path.insert(0,"./src"); sys.path.insert(0,".")
from pathlib import Path
from cv_extractor.config import FREE_SPATIAL_CONFIG
from cv_extractor.free_spatial import FreeSpatialFeatureExtractor
from stage1_helpers import make_head
from torchvision.io import read_image
from stable_baselines3 import PPO
import glob, json
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(12)
COMB="diam_angle"; ROOT=Path(f"models/vision/g4_{COMB}"); D=Path("data_factorial")
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx)); rng=d/np.cos(np.radians(bear))
NUIS=np.stack([np.log(g("light_I")), g("light_r")-g("light_b"),
               (g("wall_r")+g("wall_g")+g("wall_b"))/3, (g("floor_r")+g("floor_g")+g("floor_b"))/3,
               g("blue_bear"), g("blue_rng")],1)
TASK=np.stack([bear, rng],1)
pol=PPO.load(sorted(glob.glob(f"models/rl/teacher_score_g4_{COMB}_s*/last.zip"))[-1],device="cpu").policy
r_=31.06/np.maximum(d,1e-6)
obs=np.stack([px/112-1, np.clip(2*r_/224,0,1), np.full_like(d,1.5/8), np.zeros_like(d)],1).astype(np.float32)
with torch.no_grad(): aT=np.clip(pol.action_net(pol.mlp_extractor.policy_net(torch.as_tensor(obs))).numpy(),-1,1)
st=json.load(open(ROOT/"target_stats.json"))["A"]; mu_,sd_=np.array(st["mean"]),np.array(st["std"])
@torch.no_grad()
def z_and_a(arm):
    cz=Path(f"{L}/z_{COMB}_{arm}.npy") if False else Path(f"/home/tb/.claude/jobs/2174f11a/tmp/z_{COMB}_{arm}.npy")
    src=ROOT/"encoder_init.pt" if arm=="init" else ROOT/arm/"donor.pt"
    if cz.exists(): Z=np.load(cz)
    else:
        e=FreeSpatialFeatureExtractor(**FREE_SPATIAL_CONFIG).to(DEV); e.load_state_dict(torch.load(src,map_location=DEV)); e.eval()
        out=[]
        for i in range(0,len(rows),64):
            f=torch.stack([read_image(str(D/r["img"])) for r in rows[i:i+64]]).to(DEV)
            out.append(e(f)["shared_feature"].float().cpu())
        Z=torch.cat(out).numpy().astype(np.float64); np.save(cz,Z)
    h=make_head(256,2); h.load_state_dict(torch.load(ROOT/f"{arm}_act"/"head.pt",map_location="cpu")); h.eval()
    with torch.no_grad(): a=h(torch.tensor(Z,dtype=torch.float32)).numpy()*sd_+mu_
    return Z, np.clip(a,-1,1)
def part_r2(y, X):
    X=np.hstack([X,np.ones((len(X),1))]); w=np.linalg.lstsq(X,y,rcond=None)[0]; p=X@w
    return 1-((y-p)**2).sum()/((y-y.mean(0))**2).sum()
NEAR=seen&(rng<1.9); FAR=seen&(rng>=1.9)
print(f"受控状态 {seen.sum()} 个(近目标 {NEAR.sum()} / 远 {FAR.sum()}),teacher={COMB}\n")
print(f"{'表征':6s}{'a_S vs a_T 相关(a_x/a_w)':>26s}{'RMSE 全体':>11s}{'RMSE 近目标':>12s}{'外观解释的 a_S 方差':>20s}{'任务量解释':>12s}")
for arm in ["init","A","V","R","sup"]:
    if not (ROOT/f"{arm}_act"/"head.pt").exists(): continue
    Z,aS=z_and_a(arm)
    c0=np.corrcoef(aS[seen,0],aT[seen,0])[0,1]; c1=np.corrcoef(aS[seen,1],aT[seen,1])[0,1]
    rm=np.sqrt(((aS[seen]-aT[seen])**2).mean()); rn=np.sqrt(((aS[NEAR]-aT[NEAR])**2).mean())
    # a_S 的方差里,无关外观解释掉多少(先扣掉任务量)
    res=aS[seen]-np.hstack([TASK[seen],np.ones((seen.sum(),1))])@np.linalg.lstsq(np.hstack([TASK[seen],np.ones((seen.sum(),1))]),aS[seen],rcond=None)[0]
    f_nuis=max(0.0,part_r2(res,NUIS[seen])); f_task=part_r2(aS[seen],TASK[seen])
    print(f"{arm:6s}{c0:13.3f}{c1:13.3f}{rm:11.3f}{rn:12.3f}{f_nuis:20.3f}{f_task:12.3f}")
print(f"\n参照:teacher 动作本身的标准差 {aT[seen].std(0).round(3)};任务量(方位角,距离)能解释 teacher 动作 {part_r2(aT[seen],TASK[seen]):.3f}")
