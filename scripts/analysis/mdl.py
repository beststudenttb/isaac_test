"""可访问性(accessibility)轴:同样的信息,要多少样本才能读出来。
动机有两处:Le Lan et al. AISTATS 2022 的界把泛化误差拆成"近似误差 + 估计误差(∝ d_eff/n)",
我们的探针只量了第一项;Voita & Titov 2020 的 MDL probing 也指出"最终能不能读出来"分辨力不足。
用析因渲染的缓存 z,画 held R² 随训练样本数的曲线。CPU。
"""
import sys, csv, math, numpy as np
sys.path.insert(0,".")
from pathlib import Path
D=Path("data_factorial"); C=Path("/home/tb/.claude/jobs/2174f11a/tmp/zfact")
rows=[r for r in csv.DictReader(open(D/"meta.csv")) if r["kind"]=="factorial"]
g=lambda k: np.array([float(r[k]) for r in rows])
px,d=g("px_x"),g("dist"); seen=d>0.3
fx=112/math.tan(math.radians(40)); bear=np.degrees(np.arctan((112-px)/fx))
Y={"红球方位":bear,"蓝球方位":g("blue_bear")}
rd=g("round"); tr=np.flatnonzero((rd%2==0)&seen); te=(rd%2==1)&seen
NS=[50,100,200,500,1000,len(tr)]
rg=np.random.default_rng(0)
def fit(Z,y,idx,lam=3e-2):
    mu,sd=Z[idx].mean(0),Z[idx].std(0)+1e-9
    A=np.hstack([(Z[idx]-mu)/sd,np.ones((len(idx),1))]); B=np.hstack([(Z[te]-mu)/sd,np.ones((te.sum(),1))])
    w=np.linalg.lstsq(A.T@A+lam*len(A)*np.eye(A.shape[1]),A.T@y[idx],rcond=None)[0]
    p=B@w; yy=y[te]; return float(1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum())
for tgt,y in Y.items():
    print(f"\n### {tgt}:held R² 随训练样本数(每格 5 次重采样取中位)",flush=True)
    print(f"{'表征':6s}"+"".join(f"{('N='+str(n)):>10s}" for n in NS),flush=True)
    for arm in ("init","A","V","R","sup"):
        f=C/f"z_{arm}.npy"
        if not f.exists(): continue
        Z=np.load(f).astype(np.float64)
        cells=[]
        for n in NS:
            v=[fit(Z,y,rg.choice(tr,min(n,len(tr)),replace=False)) for _ in range(5)]
            cells.append(f"{np.median(v):10.3f}")
        print(f"{arm:6s}"+"".join(cells),flush=True)
print("\n读法:曲线越早饱和 = 这个量在该表征里越'容易取用';终点相同但起点不同,说明差别在可访问性而不是信息量。",flush=True)
