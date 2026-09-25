import csv,glob,math,os,statistics as st
c=lambda r,k: min(max(float(r[k]),-1),1)
def pos_score(r):
    d=float(r['dist'])
    if d<=0: return 0.0
    return 1/(1+abs(float(r['px_x'])-112)/20+abs(d-1.5)/0.5)
def traj_stats(d):
    rows=list(csv.DictReader(open(d+'/traj_train_env0.csv')))
    eps=[];cur=[]
    for r in rows:
        cur.append(r)
        if r['done']=='1': eps.append(cur); cur=[]
    seg=eps[9*len(eps)//10:]          # 训练末 10% 的 episode
    allr=[r for e in seg for r in e]
    ps=st.mean(pos_score(r) for r in allr)*375            # 去掉动作罚后的位置分回报(满分 375)
    vis=st.mean(1 if float(r['dist'])>0 else 0 for r in allr)
    inz=st.mean(1 if float(r['dist'])>0 and abs(float(r['px_x'])-112)<=10 and abs(float(r['dist'])-1.5)<=0.2 else 0 for r in allr)
    tails=[e[-100:] for e in seg]
    txe=st.mean(abs(float(r['px_x'])-112) for t in tails for r in t if float(r['dist'])>0) if any(float(r['dist'])>0 for t in tails for r in t) else float('nan')
    tde=st.mean(abs(float(r['dist'])-1.5) for t in tails for r in t if float(r['dist'])>0) if any(float(r['dist'])>0 for t in tails for r in t) else float('nan')
    tinz=st.mean(1 if float(r['dist'])>0 and abs(float(r['px_x'])-112)<=10 and abs(float(r['dist'])-1.5)<=0.2 else 0 for t in tails for r in t)
    tam=st.mean(max(abs(c(r,'a_x')),abs(c(r,'a_y')),abs(c(r,'a_w'))) for t in tails for r in t)
    sweeps=[]
    for t in tails:
        angs=[math.atan2(float(r['robot_y'])-float(r['target_y']), float(r['robot_x'])-float(r['target_x'])) for r in t]
        tot=0
        for i in range(1,len(angs)):
            da=angs[i]-angs[i-1]; tot+=(da+math.pi)%(2*math.pi)-math.pi
        sweeps.append(abs(math.degrees(tot)))
    return ps,vis,inz,txe,tde,tinz,tam,st.mean(sweeps)
def log_stats(d):
    log=list(csv.DictReader(open(d+'/log.csv')))
    kl=[(int(r['update']),float(r['kl'])) for r in log]
    return sum(1 for u,k in kl if k>5), sum(1 for u,k in kl if u>1900 and k>5)
def outcome(vis,tinz,tam,sweep,nl):
    if vis<0.5: return '跑丢'
    if tinz>=0.85 and sweep>50: return '公转'
    if tinz>=0.85 and tam<0.35: return '到位停'
    if tinz>=0.85: return '到位没停'
    if nl>=40: return '训练不稳'
    return '没到位'
hdr=f"{'run':9s}{'位置分回报':>10s}{'可见':>6s}{'区内':>6s}{'末100 xe':>9s}{'末100 de':>9s}{'末100区内':>9s}{'末100|a|':>9s}{'扫角':>6s}{'KL>5':>6s}{'末100KL>5':>9s}{'结局':>9s}"
for tag,pat,arms in [('k=0 (周末, 训练 traj 末 10%, 随机策略)','models/rl/score_student_',['A','V','R','rnd','spr']),('k=0.3 (本周, 同口径)','models/rl/score_k03_student_',['A','V','R','sup'])]:
    print(f'== {tag} =='); print(hdr)
    per={}
    for a in arms:
        for s in ['','_s2','_s3','_s4','_s5']:
            d=pat+a+s
            if not os.path.exists(d+'/traj_train_env0.csv'): continue
            ps,vis,inz,txe,tde,tinz,tam,sw=traj_stats(d); nb,nl=log_stats(d); o=outcome(vis,tinz,tam,sw,nl)
            per.setdefault(a,[]).append((ps,tinz,o))
            print(f'{a+s:9s}{ps:10.0f}{vis:6.0%}{inz:6.2f}{txe:8.1f}px{tde:8.2f}m{tinz:9.2f}{tam:9.2f}{sw:5.0f}°{nb:6d}{nl:9d}{o:>9s}')
    print()
    print(f"{'arm':5s}{'n':>3s}{'位置分回报 mean±sd':>18s}{'末100区内 mean':>14s}  结局")
    for a in arms:
        L=per.get(a,[])
        if not L: continue
        ps=[x[0] for x in L]; tz=[x[1] for x in L]
        print(f"{a:5s}{len(L):3d}{st.mean(ps):11.0f}±{(st.pstdev(ps) if len(ps)>1 else 0):<6.0f}{st.mean(tz):14.2f}  {' / '.join(x[2] for x in L)}")
    print()
