"""Paired, descriptive development evidence for completed recovery trials."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,read,write,bound


def returns(folder):
    data=read(folder/'daily.json')
    eq=np.array([100_000_000.]+[r['equity'] for r in data],dtype=float)
    return [r['day'] for r in data],np.diff(np.log(eq))


def paired(folder,control):
    days,a=returns(folder);other,b=returns(control)
    if days!=other:raise ValueError('Unmatched paired session population')
    delta=a-b;n=len(delta);result={}
    for block in (1,3,5,10):
        rng=np.random.default_rng(20261010+block)
        starts=rng.integers(0,n,size=(4000,(n+block-1)//block))
        indices=((starts[:,:,None]+np.arange(block))%n).reshape(4000,-1)[:,:n]
        gain=np.expm1(delta[indices].sum(axis=1))*100
        result[str(block)]={'relative_wealth_gain_ci95_pct':np.percentile(gain,[2.5,97.5]).tolist(),
                            'fraction_resamples_with_positive_difference':float(np.mean(gain>0))}
    order=np.argsort(delta)[::-1]
    return {'observed_relative_wealth_gain_pct':float(np.expm1(delta.sum())*100),
        'paired_circular_block_resampling':result,
        'largest_incremental_sessions':[{'day':days[i],'incremental_log_return':float(delta[i])} for i in order[:5]],
        'relative_wealth_gain_excluding_top_incremental_sessions_pct':{
            str(k):float(np.expm1(delta.sum()-delta[order[:k]].sum())*100) for k in (1,3)},
        'interpretation':'Descriptive same-history paired sensitivity. Removing session contributions is attribution, not a re-executed counterfactual. Resampling does not undo adaptive selection or establish unseen performance.'}


def run(root):
    summary=read(root/'recovery_results.json');records=[]
    for group,g in summary['groups'].items():
        for trial in g['trials']:
            if trial['recipe']=='control':continue
            record={'group':group,'recipe':trial['recipe'],'metrics':trial['metrics'],'attribution':trial['oos_attribution'],'paired':{}}
            for scenario in ('baseline','combined'):
                folder=Path(trial['cases']['oos_'+scenario]['path']).parent
                record['paired'][scenario]=paired(folder,root/'control'/scenario)
            records.append(record)
    payload={'created_at_utc':datetime.now(timezone.utc).isoformat(),'summary':bound(root/'recovery_results.json'),
        'program':bound(__file__),'trials':records,'trial_count':len(records),
        'independence':'Reused development. The block analysis is descriptive, not adjusted for multiple testing.',
        'findings':[
            'IS/OOS market history continuity passed; the first OOS date does not suffer a new intraday warm-up.',
            'Frozen selected forecasts substantially overestimate matching-target admitted opening action outcomes. A historically rising market does not establish positive executable intraday expectancy.',
            'Causal value/calibration updates materially reduce losses but generally reduce activity. Retained IS references do not validate alternative OOS-only schedules.',
            'ATR sizing improves full OOS DD and losses with a material IS return cost. Half-capping all quantities has an even larger IS cost.',
            'Removing quick exits and locked-core repricing materially worsen complete-window OOS; June-only results were misleading.',
            'All first-stage OOS baseline and combined results remain negative. No replacement is qualified from this stage.',
            'Next: registered finite sizing/exit/calibration interactions, then finalist source qualification or deeper target-alignment investigation.'
        ]}
    write(root/'stage1_review.json',payload)
    lines=['KALCB FULL-WINDOW FIRST-STAGE REVIEW',payload['independence'],f"Distinct noncontrol trials: {len(records)}",'']
    lines+=payload['findings']
    lines+=['','Recipe | baseline return/DD/trades | combined return/DD/trades | 5-day block relative-wealth CI baseline / combined']
    for r in records:
        m=r['metrics'];bits=[r['recipe']]
        for s in ('baseline','combined'):bits.append(f"{m['oos_'+s+'_return']:.3f}/{m['oos_'+s+'_dd']:.3f}/{m['oos_'+s+'_trades']}")
        bits += [json.dumps(r['paired'][s]['paired_circular_block_resampling']['5']['relative_wealth_gain_ci95_pct']) for s in ('baseline','combined')]
        lines.append(' | '.join(bits))
    (root/'stage1_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'trial_count':len(records),'report':str(root/'stage1_review.txt')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);run(p.parse_args().root)
