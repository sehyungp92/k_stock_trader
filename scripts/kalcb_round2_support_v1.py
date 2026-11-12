"""Registered C16 decile/regime intervals, full-path DD dispersion and exposure."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import blocks,read,write,bound,csvwrite,SOURCE,SCENARIOS
from scripts.kalcb_round2_candidate_evidence_v2 import stats

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
OUT=ROOT/'support'


def deciles(rows,feature,calendar):
    rows=[r for r in rows if r.get(feature) is not None and np.isfinite(float(r[feature]))]
    x=np.array([r[feature] for r in rows],float);v=np.array([r['value'] for r in rows])
    if not len(x):return []
    dec=np.minimum(9,np.floor((rankdata(x)-.5)/len(x)*10)).astype(int)
    dayix=np.array([calendar.index(r['day']) for r in rows]);output=[]
    for j in sorted(set(dec)):
        use=dec==j
        sums=np.bincount(dayix[use],weights=v[use],minlength=len(calendar))
        counts=np.bincount(dayix[use],minlength=len(calendar));ci={}
        for L in (5,10,20):
            b=blocks(len(calendar),L);den=counts[b].sum(axis=1);ok=den>0
            vals=sums[b].sum(axis=1)[ok]/den[ok]
            ci[str(L)]=dict(ci95=np.quantile(vals,[.025,.975]).tolist(),valid_draws=int(sum(ok)))
        output.append(dict(feature=feature,decile=int(j+1),rows=int(sum(use)),sessions=int(sum(counts>0)),
            mean_net_bps=float(v[use].mean()),x_min=float(x[use].min()),x_max=float(x[use].max()),ci=ci))
    return output


def daily_shape(period,sc):
    folder=SOURCE/'cases'/f'combined_{period}_{sc}'
    eq=read(folder/'equity.json');daily=read(folder/'daily.json');days=[r['day'] for r in daily]
    byday=defaultdict(list)
    for t,v in eq:byday[t[:10]].append(v)
    close=1e8;shapes=[]
    for d in days:
        path=np.array([close]+byday[d])/close
        shapes.append([path[-1],path.max(),path.min(),np.min(path/np.maximum.accumulate(path))])
        close=byday[d][-1]
    shapes=np.array(shapes);out={}
    for L in (5,10,20):
        draws=blocks(len(days),L);wealth=np.ones(len(draws));peak=wealth.copy();mindd=np.ones(len(draws))
        for j in range(len(days)):
            shape=shapes[draws[:,j]]
            # Prior peak versus all current-day values, and ordered intraday DD.
            mindd=np.minimum(mindd,np.minimum(wealth*shape[:,2]/peak,shape[:,3]))
            peak=np.maximum(peak,wealth*shape[:,1]);wealth*=shape[:,0]
        dds=100*(1-mindd)
        out[str(L)]=dict(dd_se_pp=float(dds.std(ddof=1)),dd_ci95=np.quantile(dds,[.025,.975]).tolist(),
            return_se_pp=float((100*(wealth-1)).std(ddof=1)),
            semantics='Resampled complete within-session relative MTM paths; changed chronology, not a strategy replay.')
    return out


def main():
    OUT.mkdir(exist_ok=False)
    write(OUT/'binding.json',dict(program=bound(__file__),protocol=bound(Path(__file__).resolve().parents[1]/'docs/optimisation/kalcb/experiments/kalcb-exp-0041.md'),
        note='Completes already registered decile, calendar/regime, MTM-dispersion and exposure analyses; no new estimand or trading policy.'))
    for period in ('is','oos'):
        for sc in SCENARIOS:
            p=ROOT/'analysis'/f'{period}_{sc}_trade_paths.csv'
            rows=list(csv.DictReader(p.open(encoding='utf-8')))
            cal=read(ROOT/'analysis'/f'{period}_{sc}_account.json')['calendar']
            features=('quality_votes','relvol','opening_return','opening_cpr','atr_fraction','negative_rank')
            for r in rows:
                r['value']=float(r['net_bps']);r['id']=r['day']+'/'+r['symbol']
                for k in features:r[k]=float(r[k]) if r[k] else None
            d=[x for f in features for x in deciles(rows,f,cal)]
            # Full predeclared features in every observed native regime, including sparse groups.
            regimes={}
            for regime in sorted({r['regime'] for r in rows}):
                rr=[r for r in rows if r['regime']==regime]
                regimes[regime or 'missing']=[stats(rr,f,cal) for f in features]
            months={m:[stats([r for r in rows if r['day'].startswith(m)],f,[d for d in cal if d.startswith(m)]) for f in features]
                    for m in sorted({d[:7] for d in cal})}
            write(OUT/f'{period}_{sc}_selection.json',dict(source=bound(p),deciles=d,regimes=regimes,months=months))
            write(OUT/f'{period}_{sc}_mtm_dispersion.json',daily_shape(period,sc))
            print(period,sc,flush=True)
    # Original C9 quantity-one labels; same exclusions as the registered reuse.
    groups=defaultdict(list)
    with gzip.open('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016/entry_lifecycle_v2/panel.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['quantity_scale']!=1 or r['period'] not in ('is','oos') or not r['label_valid'] or r['entry_value_planned_bps'] is None or r['deadline_violation']:continue
            if r['label_available_at'][:10]>('2026-03-31' if r['period']=='is' else '2026-07-10'):continue
            x=dict(day=r['day'],value=r['entry_value_planned_bps'],forecast_core=r.get('forecast_core'),positive_core=r.get('positive_core'),physical_capacity_pass=r.get('physical_capacity_pass'))
            x.update({k:r['features30'].get(k) for k in ('action_quality_votes','first30_ret','first30_rel_volume','first30_cpr_x_rvol')})
            groups[r['period'],r['scenario']].append(x)
    for (period,sc),rows in groups.items():
        cal=read(ROOT/'analysis'/f'{period}_{sc}_account.json')['calendar']
        write(OUT/f'{period}_{sc}_historical_deciles.json',[x for f in ('forecast_core','action_quality_votes','first30_ret','first30_rel_volume','first30_cpr_x_rvol','positive_core','physical_capacity_pass') for x in deciles(rows,f,cal)])
    write(OUT/'completion.json',dict(complete=True,new_native_accounts=0,new_treatment_runs=0))


if __name__=='__main__':main()
