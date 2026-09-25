import csv,glob,os,statistics as st
ROOT='models/rl/score_k03_student_'
ARMS=['A','V','R','sup']; SEEDS=['','_s2','_s3']
def load(run):
    f=ROOT+run+'/eval_score.csv'
    if not os.path.exists(f): return None
    return list(csv.DictReader(open(f)))
def kl_stats(run):
    log=list(csv.DictReader(open(ROOT+run+'/log.csv')))
    kl=[(int(r['update']),float(r['kl'])) for r in log]
    return sum(1 for u,k in kl if k>5), sum(1 for u,k in kl if u>1900 and k>5)
def ckpt_swing(run):
    rets=[]
    for u in ['001960','001970','001980','001990','002000']:
        f=ROOT+run+f'/eval_score_{u}.csv'
        if not os.path.exists(f): return None
        rets.append(st.mean(float(x['return']) for x in csv.DictReader(open(f))))
    return min(rets),max(rets)
def end_vis(run):
    rows=list(csv.DictReader(open(ROOT+run+'/traj_train_env0.csv')))
    eps=[];cur=[]
    for x in rows:
        cur.append(x)
        if x['done']=='1': eps.append(cur); cur=[]
    seg=eps[9*len(eps)//10:]
    return st.mean(sum(1 for x in e if float(x['dist'])>0)/len(e) for e in seg)
def outcome(r,run):
    ret=st.mean(float(x['return']) for x in r); zone=st.mean(float(x['tail_zone']) for x in r)
    v1=st.mean(float(x['v1_success']) for x in r)
    if v1>=0.9: return '停住'
    if zone>=0.85: return '到位没停'
    if end_vis(run)<0.5: return '跑丢'
    return '训练不稳'
print('== 四臂 n=3 总表(eval: 64 env × 375 步,确定性策略,last.pt) ==')
print(f"{'run':7s}{'v1':>7s}{'回报':>6s}{'区内':>6s}{'末100 de':>9s}{'末100|a|':>9s}{'结局':>8s}{'KL>5':>6s}{'末100 KL>5':>10s}{'末5ckpt回报':>14s}{'训练末段可见':>12s}")
summary={}
for a in ARMS:
    for s in SEEDS:
        run=a+s; r=load(run)
        if r is None: print(f'{run:7s}  (无 eval)'); continue
        v1=st.mean(float(x['v1_success']) for x in r); ret=st.mean(float(x['return']) for x in r)
        zone=st.mean(float(x['zone_dwell']) for x in r); de=st.mean(float(x['tail_de']) for x in r); am=st.mean(float(x['tail_amag']) for x in r)
        nb,nl=kl_stats(run); sw=ckpt_swing(run); o=outcome(r,run)
        summary.setdefault(a,[]).append((v1,ret,o,r))
        print(f'{run:7s}{v1:7.1%}{ret:6.0f}{zone:6.2f}{de:8.2f}m{am:9.3f}{o:>8s}{nb:6d}{nl:10d}{(f"{sw[0]:.0f}-{sw[1]:.0f}" if sw else "-"):>14s}{end_vis(run):12.0%}')
print()
print('== 按臂汇总 ==')
print(f"{'arm':5s}{'n':>3s}{'v1≥90% seed':>12s}{'回报 mean±sd':>16s}{'结局':>30s}")
for a in ARMS:
    L=summary.get(a,[])
    if not L: continue
    rets=[x[1] for x in L]
    print(f"{a:5s}{len(L):3d}{sum(1 for x in L if x[0]>=0.9):>7d}/{len(L):<4d}{st.mean(rets):8.0f}±{(st.pstdev(rets) if len(rets)>1 else 0):<6.0f}{' / '.join(x[2] for x in L):>30s}")
print()
print('== 部署判据:末100步 区内比例≥0.9 且 平均|a|<eps 的 env 占比 ==')
EPS=[0.05,0.10,0.15,0.20,0.30]
print(f"{'run':7s}{'结局':>8s}"+''.join(f'{"eps="+str(e):>9s}' for e in EPS))
allr={e:[] for e in EPS}
for a in ARMS:
    for s in SEEDS:
        run=a+s; r=load(run)
        if r is None: continue
        o=outcome(r,run); row=[]
        for e in EPS:
            ok=[1 if float(x['tail_zone'])>=0.9 and float(x['tail_amag'])<e else 0 for x in r]
            row.append(st.mean(ok)); allr[e].append(st.mean(ok))
        print(f'{run:7s}{o:>8s}'+''.join(f'{v:9.1%}' for v in row))
print(f'{"全部 run 均值":13s}'+''.join(f'{st.mean(allr[e]):9.1%}' for e in EPS))
