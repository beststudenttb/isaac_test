import csv,statistics as st,os,glob,sys
ROOT='models/rl/score_k03noy_student_'
g=lambda L,k: st.mean(float(x[k]) for x in L) if L else float('nan')
def load(run,tag=''):
    f=ROOT+run+f'/eval_score{tag}.csv'
    return list(csv.DictReader(open(f))) if os.path.exists(f) else None
def stats(r):
    good=[x for x in r if float(x['return'])>=60]
    return dict(v1=g(r,'v1_success'),ret=g(r,'return'),found=len(good),v1g=g(good,'v1_success'),retg=g(good,'return'),deg=g(good,'tail_xe'),de=g(good,'tail_de'),amag=g(good,'tail_amag'))
seeds=sys.argv[1:] or ['']
print(f"{'teacher':6s}{'信号':5s}{'seed':>5s}|{'last v1':>8s}{'回报':>6s}{'找到':>5s}{'找到起点 v1':>10s}{'回报':>6s}{'角':>6s}{'距':>7s}|{'最好ckpt':>9s}{'v1':>6s}{'回报':>6s}")
for M in ['ang','xd','xyd','diam','bbox']:
    for A in ['A','V','R']:
        for SD in seeds:
            sfx=f'_s{SD}' if SD else ''; run=f'{A}_donor{sfx}_wk_{M}'; r=load(run)
            if r is None: continue
            L=stats(r); best=None
            for cf in glob.glob(ROOT+run+'/eval_score_*.csv'):
                s=stats(list(csv.DictReader(open(cf)))); u=cf.split('eval_score_')[1][:-4].lstrip('0')
                if best is None or s['v1']>best[1]['v1']: best=(u,s)
            b=best[1] if best else {'v1':float('nan'),'ret':float('nan')}
            print(f"{M:6s}{A:5s}{SD or '1':>5s}|{L['v1']:8.0%}{L['ret']:6.0f}{L['found']:5d}{L['v1g']:10.0%}{L['retg']:6.0f}{L['deg']:5.1f}°{L['de']:6.2f}m|{('u'+best[0]) if best else '-':>9s}{b['v1']:6.0%}{b['ret']:6.0f}")
for SD in seeds:
    sfx=f'_s{SD}' if SD else ''; r=load(f'init{sfx}_wk_ang')
    if r: L=stats(r); print(f"{'init':6s}{'-':5s}{SD or '1':>5s}|{L['v1']:8.0%}{L['ret']:6.0f}{L['found']:5d}{L['v1g']:10.0%}{L['retg']:6.0f}{L['deg']:5.1f}°{L['de']:6.2f}m|")
