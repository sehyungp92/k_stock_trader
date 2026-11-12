"""Registered C9 label reuse; historical standalone evidence, never cap2 funded labels."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import gzip
import hashlib
import numpy as np
from scipy.stats import spearmanr, rankdata

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
SOURCE=Path('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016/entry_lifecycle_v2')
OLD=ROOT/'candidate_evidence'
OUT=ROOT/'candidate_evidence_v2'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def blocks(n,L):
    rng=np.random.default_rng(41017+L+n);a=rng.integers(0,n,(4000,int(np.ceil(n/L))))
    return ((a[:,:,None]+np.arange(L))%n).reshape(4000,-1)[:,:n]


def stats(rows,feature,calendar):
    rows=[r for r in rows if r.get(feature) is not None and np.isfinite(float(r[feature]))]
    if not rows:return {'feature':feature,'rows':0}
    x=np.array([float(r[feature]) for r in rows]);v=np.array([r['value'] for r in rows]);positive=v>0
    groupx=np.unique(x,return_inverse=True)[1];groupv=np.unique(v,return_inverse=True)[1]
    dayix=np.array([calendar.index(r['day']) for r in rows]);gx=int(groupx.max())+1;gv=int(groupv.max())+1
    def measure(w):
        total=w.sum()
        if not total:return None,None,None
        px=np.bincount(groupx,weights=w*positive,minlength=gx);nx=np.bincount(groupx,weights=w*(~positive),minlength=gx)
        n1=px.sum();n0=nx.sum()
        a=float(np.sum(px*(np.cumsum(nx)-.5*nx))/(n1*n0)) if n1 and n0 else None
        wx=px+nx;wv=np.bincount(groupv,weights=w,minlength=gv)
        rx=(np.cumsum(wx)-.5*wx+.5)[groupx];rv=(np.cumsum(wv)-.5*wv+.5)[groupv]
        center=(total+1)/2;xx=rx-center;yy=rv-center
        denom=np.sqrt(np.sum(w*xx*xx)*np.sum(w*yy*yy));rho=float(np.sum(w*xx*yy)/denom) if denom else None
        return a,rho,float(np.sum(w*v)/total)
    observed=measure(np.ones(len(rows)))
    # The weighted implementation must equal ordinary ranks on a single copy.
    expected=spearmanr(x,v).statistic if np.ptp(x) and np.ptp(v) else None
    assert expected is None or abs(expected-observed[1])<1e-10
    cis={}
    for L in (5,10,20):
        vals=[]
        for draw in blocks(len(calendar),L):
            w=np.bincount(draw,minlength=len(calendar))[dayix];vals.append(measure(w))
        cis[str(L)]={name:np.quantile([r[j] for r in vals if r[j] is not None],[.025,.975]).tolist() if any(r[j] is not None for r in vals) else None
                     for j,name in enumerate(('auc95','spearman95','expectancy95'))}
    dec=np.minimum(9,np.floor((rankdata(x)-.5)/len(x)*10)).astype(int)
    deciles=[dict(decile=j+1,n=int(sum(dec==j)),net_bps=float(v[dec==j].mean())) for j in range(10) if sum(dec==j)]
    sub={}
    for month in sorted({r['day'][:7] for r in rows}):
        w=np.array([int(r['day'].startswith(month)) for r in rows]);a,r,e=measure(w);sub[month]=dict(n=int(w.sum()),auc=a,spearman=r,net_bps=e)
    return dict(feature=feature,rows=len(rows),sessions=len(set(r['day'] for r in rows)),issuer_days=len(set(r['id'] for r in rows)),
        auc=observed[0],spearman=observed[1],net_bps=observed[2],ci=cis,deciles=deciles,months=sub)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    inputs=[Path(__file__),SOURCE/'panel.jsonl.gz',SOURCE/'registration.json',SOURCE/'assessment.json',
            REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-analysis-binding.md']
    write(OUT/'binding.json',{'inputs':[{'path':str(p),'sha256':sha(p)} for p in inputs],
        'scope':'Original standalone q1 native labels only. No current-funded-policy transfer, random-entry or oracle claim.'})
    allcounts=Counter();groups=defaultdict(list);coverage=defaultdict(Counter)
    with gzip.open(SOURCE/'panel.jsonl.gz','rt',encoding='utf-8') as f:
        for index,line in enumerate(f,1):
            r=json.loads(line);allcounts[r['native_status']]+=1
            if r['period'] not in ('is','oos') or r['quantity_scale']!=1:continue
            key=r['period']+'/'+r['scenario'];coverage[key]['rows']+=1
            coverage[key][r['native_status']]+=1
            if not r['label_valid'] or r['entry_value_planned_bps'] is None:
                coverage[key]['invalid_or_null']+=1;continue
            if r['label_available_at'][:10] > ('2026-03-31' if r['period']=='is' else '2026-07-10'):
                coverage[key]['past_partition_maturity']+=1;continue
            if r['deadline_violation']:
                coverage[key]['deadline_violation']+=1;continue
            x=dict(day=r['day'],id=r['base_id'],value=r['entry_value_planned_bps'],source_row=index,
                forecast_core=r.get('forecast_core'),positive_core=r.get('positive_core'),physical_capacity_pass=r.get('physical_capacity_pass'),
                selected_prior100=r['selected_prior100'],actual_prior100=r['actual_prior100'])
            x.update({k:r['features30'].get(k) for k in ('action_quality_votes','first30_ret','first30_rel_volume','first30_cpr_x_rvol')})
            groups[key].append(x)
    output={}
    for key,rows in groups.items():
        p,s=key.split('/');cal=[r['day'] for r in read(Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1/cases')/f'combined_{p}_{s}'/'daily.json')]
        previous=OLD/(p+'_'+s+'.json')
        if previous.exists():
            value=read(previous);write(OUT/previous.name,value)
            write(OUT/(p+'_'+s+'_reuse.json'),{'path':str(previous),'sha256':sha(previous),'economic_statistics_unchanged':True})
            continue
        print('candidate discrimination',key,len(rows),flush=True)
        output[key]=[stats(rows,feature,cal) for feature in ('forecast_core','action_quality_votes','first30_ret','first30_rel_volume','first30_cpr_x_rvol','positive_core','physical_capacity_pass')]
        write(OUT/(p+'_'+s+'.json'),dict(scope='Original historical q1 standalone label basis',coverage=dict(coverage[key]),statistics=output[key]))
    write(OUT/'completion.json',dict(complete=True,all_native_status_counts=dict(allcounts),coverage={k:dict(v) for k,v in coverage.items()},
        input_panel_rows=sum(allcounts.values()),native_replays=0,fits=0,matched_random_entries=False,current_cap2_labels=False))


if __name__=='__main__':main()
