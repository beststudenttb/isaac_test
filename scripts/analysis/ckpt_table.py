import csv,statistics as st,os,sys
ROOT='models/rl/score_k03noy_student_'
arms=sys.argv[1:] or ['A','V','R','sup']
def load(run,tag=''):
    f=ROOT+run+f'/eval_score{tag}.csv'
    return list(csv.DictReader(open(f))) if os.path.exists(f) else None
g=lambda L,k: st.mean(float(x[k]) for x in L) if L else float('nan')
print(f"{'ckpt':6s}"+"".join(f"{r+' 丢':>8s}{r+' v1':>8s}{r+' 回报':>8s}{r+' 正常':>8s}" for r in arms))
for u in ['000200','000210','000220','000230','000240','']:
    row=f"{(u[-3:].lstrip('0') if u else 'last'):6s}"
    for run in arms:
        r=load(run,('_'+u) if u else '')
        if r is None: row+=f"{'-':>8s}"*4; continue
        lost=[x for x in r if float(x['return'])<60]; good=[x for x in r if float(x['return'])>=60]
        row+=f"{len(lost):8d}{g(r,'v1_success'):8.0%}{g(r,'return'):8.0f}{g(good,'return'):8.0f}"
    print(row)
