import csv,statistics as st,os,math,glob,sys
fx=112/math.tan(math.radians(40)); cl=lambda v: min(max(v,-1),1)
def nm(v): v=[x for x in v if x==x]; return st.mean(v) if v else float('nan')
def episodes(path, frac=0.9):
    rows=list(csv.DictReader(open(path))); eps=[];cur=[]
    for r in rows:
        cur.append(r)
        if r['done']=='1': eps.append(cur); cur=[]
    return eps[int(len(eps)*frac):]
def shape(eps):
    out=dict(t_align=[],slant=[],ax_appr=[],keep=[],t_arr=[],exits=[],post_a=[],pre_ax=[],n=0)
    for e in eps:
        seen=[float(r['dist'])>0 for r in e]
        if sum(seen)<50 or not seen[0]: continue   # 只算出生就看得到球的起点(用户口径)
        b=[math.degrees(math.atan((112-float(r['px_x']))/fx)) if s else 90 for r,s in zip(e,seen)]
        rg=[float(r['dist'])/math.cos(math.radians(bb)) if s else 99 for r,s,bb in zip(e,seen,b)]
        n=len(e); out['n']+=1
        t=next((i for i,bb in enumerate(b) if abs(bb)<=3),n); out['t_align'].append(t)
        # 斜着走:对准前(|角|>5°)完成的距离缩短占总缩短的比例
        prog_mis=0.0; prog_tot=0.0
        for i in range(1,n):
            if rg[i]<90 and rg[i-1]<90:
                d=rg[i-1]-rg[i]
                if d>0: prog_tot+=d; prog_mis+= d if abs(b[i])>5 else 0
        out['slant'].append(prog_mis/prog_tot if prog_tot>0.3 else float('nan'))
        out['pre_ax'].append(nm([cl(float(e[i]['a_x'])) for i in range(0,t) if rg[i]<90]) if t>3 else float('nan'))
        appr=[i for i in range(t,n) if 1.7<rg[i]<90]
        out['ax_appr'].append(nm([cl(float(e[i]['a_x'])) for i in appr]) if len(appr)>3 else float('nan'))
        out['keep'].append(nm([1.0 if abs(b[i])<=5 else 0.0 for i in appr]) if len(appr)>3 else float('nan'))
        inz=[abs(rg[i]-1.5)<=0.2 and abs(b[i])<=3 for i in range(n)]
        ta=next((i for i,z in enumerate(inz) if z),n); out['t_arr'].append(ta)
        if ta<n:
            ex=sum(1 for i in range(ta+1,n) if inz[i-1] and not inz[i]); out['exits'].append(ex)
            out['post_a'].append(nm([math.hypot(cl(float(e[i]['a_x'])),cl(float(e[i]['a_w']))) for i in range(ta+10,n)]) if n-ta>15 else float('nan'))
    return out
def ev(path):
    r=list(csv.DictReader(open(path))); ok=[x for x in r if x['v1_success']=='1']; s=sorted(float(x['succ_step']) for x in ok)
    return dict(v1=st.mean(float(x['v1_success']) for x in r),succ=nm([float(x['succ_step']) for x in ok]),p90=(s[int(0.9*len(s))-1] if len(s)>=2 else float('nan')),dwell=st.mean(float(x['zone_dwell']) for x in r),amag=st.mean(float(x['tail_amag']) for x in r))
ROOT='models/rl/score_k03noy_student_'
runs=[]
for M in ['ang','xd','xyd','diam','bbox']:
    for A in ['A','V','R']:
        for SD in ['','_s2','_s3']:
            run=f'{A}_donor{SD}_wk_{M}'
            if os.path.exists(ROOT+run+'/eval_score.csv'): runs.append((M,A,SD or '_s1',shape(episodes(ROOT+run+'/traj_train_env0.csv')),ev(ROOT+run+'/eval_score.csv')))
for SD in ['','_s2']:
    run=f'init{SD}_wk_ang'; runs.append(('ang','init',SD or '_s1',shape(episodes(ROOT+run+'/traj_train_env0.csv')),ev(ROOT+run+'/eval_score.csv')))
H=f"{'teacher':7s}{'臂':4s}{'seed':5s}{'n':>3s}|{'对准步':>6s}{'斜走%':>6s}{'对准前a_x':>9s}{'接近a_x':>8s}{'保持%':>6s}{'到区步':>6s}{'出区':>5s}{'停后|a|':>7s}|{'v1':>5s}{'首停':>5s}{'p90':>5s}{'区内':>5s}{'末|a|':>6s}"
def row(lbl,S,E):
    return f"{lbl}{S['n']:3d}|{nm(S['t_align']):6.0f}{nm(S['slant']):6.0%}{nm(S['pre_ax']):9.2f}{nm(S['ax_appr']):8.2f}{nm(S['keep']):6.0%}{nm(S['t_arr']):6.0f}{nm(S['exits']):5.1f}{nm(S['post_a']):7.2f}|{E['v1']:5.0%}{E['succ']:5.0f}{E['p90']:5.0f}{E['dwell']:5.2f}{E['amag']:6.2f}"
def merge(L):
    S={k:sum((r[3][k] for r in L),[]) for k in ['t_align','slant','ax_appr','keep','t_arr','exits','post_a','pre_ax']}; S['n']=sum(r[3]['n'] for r in L)
    E={k:nm([r[4][k] for r in L]) for k in ['v1','succ','p90','dwell','amag']}; return S,E
print("训练末 10% env0 轨迹(随机策略,只算出生可见球的回合)| 确定性 eval(64 起点)\n"+H)
for r in runs: print(row(f"{r[0]:7s}{r[1]:4s}{r[2]:5s}",r[3],r[4]))
print("\n按 (teacher, 臂) 合并:\n"+H)
for M in ['ang','xd','xyd','diam','bbox']:
    for A in ['A','V','R','init']:
        L=[r for r in runs if r[0]==M and r[1]==A]
        if L: S,E=merge(L); print(row(f"{M:7s}{A:4s}{'x'+str(len(L)):5s}",S,E))
print("\n按臂合并:\n"+H)
for A in ['A','V','R','init']:
    L=[r for r in runs if r[1]==A]; S,E=merge(L); print(row(f"{'all':7s}{A:4s}{'x'+str(len(L)):5s}",S,E))
print("\n按 teacher 合并(A/V/R):\n"+H)
for M in ['ang','xd','xyd','diam','bbox']:
    L=[r for r in runs if r[0]==M and r[1]!='init']; S,E=merge(L); print(row(f"{M:7s}{'AVR':4s}{'x'+str(len(L)):5s}",S,E))
print("\nteacher 自身(训练后半段 env0 轨迹,随机策略;eval 无逐步轨迹):")
for M in ['ang','xd','xyd','diam','bbox']:
    for sd in (0,1):
        p=f'models/rl/teacher_score_wk_{M}_s{sd}/traj_train_env0.csv'
        if os.path.exists(p):
            S=shape(episodes(p,0.5)); print(f"{M:7s}s{sd}   {S['n']:3d}|{nm(S['t_align']):6.0f}{nm(S['slant']):6.0%}{nm(S['pre_ax']):9.2f}{nm(S['ax_appr']):8.2f}{nm(S['keep']):6.0%}{nm(S['t_arr']):6.0f}{nm(S['exits']):5.1f}{nm(S['post_a']):7.2f}")
