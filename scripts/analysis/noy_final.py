import csv,statistics as st,os
ROOT='models/rl/score_k03noy_student_'
ARMS=['A','V','R','sup']; SEEDS=['','_s2','_s3']
def load(run,tag=''):
    f=ROOT+run+f'/eval_score{tag}.csv'
    return list(csv.DictReader(open(f))) if os.path.exists(f) else None
g=lambda L,k: st.mean(float(x[k]) for x in L) if L else float('nan')
def logstats(run):
    log=list(csv.DictReader(open(ROOT+run+'/log.csv')))
    kl=[float(r['kl']) for r in log]; z=[float(r['zone_frac']) for r in log]; sd=[float(r['std_mean']) for r in log]
    w=[(i+20,st.mean(z[i:i+20])) for i in range(0,len(z)-19)]; b=max(w,key=lambda t:t[1])
    return dict(kl_med=st.median(kl),kl5=sum(1 for k in kl if k>5),kl1_last40=sum(1 for k in kl[-40:] if k>1),zpeak=b[1],zpeak_u=b[0],zlast=st.mean(z[-20:]),std_end=sd[-1],std_min=min(sd))
def outcome(r):
    lost=sum(1 for x in r if float(x['return'])<60); v1=g(r,'v1_success'); zone=g(r,'tail_zone'); am=g(r,'tail_amag')
    if v1>=0.85: return '停住'
    if lost>=32: return '多数起点丢球'
    if zone>=0.6 and am<0.1: return '到位没停'
    if zone>=0.6: return '到位没停'
    return '双峰/不稳'
print('== k03noy 批次 12 条:last.pt(64 env,确定性)==')
print(f"{'run':7s}{'v1':>6s}{'回报':>6s}{'丢球env':>8s}{'正常env回报':>11s}{'末100 xe':>9s}{'末100 de':>9s}{'末100|a|':>9s}{'结局':>10s}")
summary={}
for a in ARMS:
    for s in SEEDS:
        run=a+s; r=load(run)
        if r is None: continue
        lost=[x for x in r if float(x['return'])<60]; good=[x for x in r if float(x['return'])>=60]
        o=outcome(r); summary.setdefault(a,[]).append((g(r,'v1_success'),g(r,'return'),o,len(lost)))
        print(f"{run:7s}{g(r,'v1_success'):6.0%}{g(r,'return'):6.0f}{len(lost):8d}{g(good,'return'):11.0f}{g(r,'tail_xe'):7.1f}px{g(r,'tail_de'):8.2f}m{g(r,'tail_amag'):9.3f}{o:>10s}")
print('\n== 按臂(last.pt)==')
print(f"{'arm':5s}{'v1≥85% seed':>12s}{'v1 mean':>9s}{'回报 mean±sd':>16s}  结局")
for a in ARMS:
    L=summary[a]; rets=[x[1] for x in L]
    print(f"{a:5s}{sum(1 for x in L if x[0]>=0.85):>7d}/{len(L):<4d}{st.mean(x[0] for x in L):9.0%}{st.mean(rets):9.0f}±{st.pstdev(rets):<6.0f}  {' / '.join(x[2] for x in L)}")
print('\n== 末尾 5 个 ckpt(u200-u240)+ last 的 v1;"最好"= 六个里最高 ==')
print(f"{'run':7s}{'u200':>6s}{'u210':>6s}{'u220':>6s}{'u230':>6s}{'u240':>6s}{'last':>6s}{'最好':>6s}{'最好回报':>8s}")
bestof={}
for a in ARMS:
    for s in SEEDS:
        run=a+s; row=f"{run:7s}"; cands=[]
        for u in ['000200','000210','000220','000230','000240','']:
            r=load(run,('_'+u) if u else '')
            if r is None: row+=f"{'-':>6s}"; continue
            v=g(r,'v1_success'); cands.append((v,g(r,'return'),u or 'last')); row+=f"{v:6.0%}"
        b=max(cands); bestof.setdefault(a,[]).append(b); row+=f"{b[0]:6.0%}{b[1]:8.0f}"
        print(row)
print('\n== 按臂:last 口径 vs 末 6 点取最好 ==')
for a in ARMS:
    print(f"  {a}: last v1 mean {st.mean(x[0] for x in summary[a]):.0%}  最好 v1 mean {st.mean(x[0] for x in bestof[a]):.0%}  最好回报 mean {st.mean(x[1] for x in bestof[a]):.0f}")
print('\n== 训练动力学:std 与 KL 与结局 ==')
print(f"{'run':7s}{'std 末值':>8s}{'KL 中位':>8s}{'KL>5':>6s}{'末40 KL>1':>9s}{'zone 峰':>8s}{'峰位置':>7s}{'末20 zone':>9s}{'last v1':>8s}{'最好 v1':>8s}")
for a in ARMS:
    for i,s in enumerate(SEEDS):
        run=a+s; L=logstats(run); r=load(run)
        print(f"{run:7s}{L['std_end']:8.2f}{L['kl_med']:8.2f}{L['kl5']:6d}{L['kl1_last40']:9d}{L['zpeak']:8.2f}{'u'+str(L['zpeak_u']):>7s}{L['zlast']:9.2f}{g(r,'v1_success'):8.0%}{bestof[a][i][0]:8.0%}")
