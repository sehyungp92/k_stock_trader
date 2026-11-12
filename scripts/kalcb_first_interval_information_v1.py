"""Fixed early-renewal information comparison. No native policy qualification."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_lifecycle_readiness_v1 import ROOT, CODE, SCENARIOS, read, write, bound, verify

V49 = Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v49_staged_support')


def features(row, metadata):
    bars = row['raw_prefix']; at = datetime.fromisoformat(row['completion_at'])
    if len(bars) != 7 or any(datetime.fromisoformat(b[0]) + timedelta(minutes=5) > at for b in bars):
        raise ValueError('Future/incomplete first-interval prefix')
    stamps = [datetime.fromisoformat(b[0]) for b in bars]
    if any(b-a != timedelta(minutes=5) for a,b in zip(stamps,stamps[1:])) or (stamps[0].hour, stamps[0].minute) != (9,0):
        raise ValueError('Unsupported session clock')
    atr = float(metadata['daily_atr']); op = bars[:6]; new = bars[6]
    if atr <= 0 or min(b[5] for b in bars) <= 0:
        raise ValueError('Unsupported ATR or volume')
    high=max(b[2] for b in op); low=min(b[3] for b in op); close=op[-1][4]
    vol=sum(b[5] for b in op); vwap=sum((b[2]+b[3]+b[4])/3*b[5] for b in op)/vol
    old=[(close-op[0][1])/atr, (high-low)/atr, (close-vwap)/atr,
         (close-low)/max(high-low,1e-9), math.log1p(float(metadata['first30_rel_volume'])), atr/close]
    extra=[(new[1]-close)/atr, (new[4]-new[1])/atr, (new[2]-high)/atr,
           (new[3]-low)/atr, (new[4]-new[3])/max(new[2]-new[3],1e-9), math.log1p(new[5]/(vol/6))]
    if not all(math.isfinite(x) for x in old+extra):raise ValueError('Nonfinite feature')
    return old+extra


def train(rows):
    if len(rows)<30:raise ValueError('Insufficient training positions for fixed diagnostic')
    X=np.array([r['features'] for r in rows]); y=np.array([r['target_bps'] for r in rows])
    counts=Counter(r['day'] for r in rows); w=np.array([1/counts[r['day']] for r in rows]);w/=w.mean()
    mean=np.average(X,axis=0,weights=w); scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=w));scale=np.maximum(scale,1e-9)
    z=np.clip((X-mean)/scale,-6,6); center=np.average(z,axis=0,weights=w); z-=center
    ym=float(np.average(y,weights=w));zw=z*np.sqrt(w[:,None]);yw=(y-ym)*np.sqrt(w)
    eig_b=np.linalg.svd(zw[:,:6],compute_uv=False)**2;target_df=float(np.sum(eig_b/(eig_b+120)))
    eig_n=np.linalg.svd(zw,compute_uv=False)**2
    lo,hi=1e-8,1e9
    for _ in range(100):
        alpha=(lo+hi)/2
        if np.sum(eig_n/(eig_n+alpha))>target_df:lo=alpha
        else:hi=alpha
    alpha=(lo+hi)/2;models={'I':{'mean':ym}}
    for name,size,penalty,eig in [('B',6,120.,eig_b),('N',12,alpha,eig_n)]:
        xx=zw[:,:size];coef=np.linalg.solve(xx.T@xx+penalty*np.eye(size),xx.T@yw)
        models[name]={'mean':ym,'coef':coef.tolist(),'feature_mean':mean[:size].tolist(),
          'feature_scale':scale[:size].tolist(),'clipped_center':center[:size].tolist(),
          'alpha':float(penalty),'effective_df':float(np.sum(eig/(eig+penalty)))}
    return models


def predict(model, row):
    if 'coef' not in model:return model['mean']
    size=len(model['coef']);x=np.array(row['features'][:size])
    z=np.clip((x-np.array(model['feature_mean']))/np.array(model['feature_scale']),-6,6)-np.array(model['clipped_center'])
    return float(model['mean']+z@np.array(model['coef']))


def summarise(rows):
    result={'positions':len(rows),'days':len({r['day'] for r in rows}), 'models':{}}
    for name in ('I','B','N'):
        days=defaultdict(list);selected=[]
        for r in rows:
            choice=r['predictions'][name]<-r['exit_cost_bps']
            gain=r['optimistic_exit_advantage_account_bps'] if choice else 0.
            days[r['day']].append(gain)
            if choice:selected.append(r)
        result['models'][name]={'actions':len(selected),'action_days':len({r['day'] for r in selected}),
             'mean_evaluation_session_advantage_bps':sum(sum(v)/len(v) for v in days.values())/len(days) if days else None,
             'selected_positive_continuation':sum(r['target_bps']>0 for r in selected),
             'selected_original_winners':sum(r['original_net_krw']>0 for r in selected),
             'best_action_account_bps':max((r['optimistic_exit_advantage_account_bps'] for r in selected),default=None),
             'worst_action_account_bps':min((r['optimistic_exit_advantage_account_bps'] for r in selected),default=None)}
    return result


def physical(fills):
    return [(f['symbol'],f['side'],f['qty'],f['price'],str(datetime.fromisoformat(f['timestamp'])),f['reason'],f['metadata']['fee']) for f in fills]


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    prior=read(V49/'completion.json');forward=read(V49/'forward_v1/completion.json')
    inputs=[bound(V49/'protocol.json'),bound(V49/'forward_v1/review_v1/review.json')]
    for b in prior['outputs']+forward['outputs']:
        verify(b);inputs.append(b)
    if not read(V49/'forward_v1/review_v1/review.json')['passed']:raise ValueError('Unreviewed temporal partition')
    cases={c['key']:c for c in read(V49/'protocol.json')['cases']}
    for period in ('is','oos'):
        for s in SCENARIOS:
            original=Path(cases['control/'+period+'/'+s]['folder']);central=ROOT/'study_v2/cases/control'/period/s
            if physical(read(original/'fills.json'))!=physical(read(central/'fills.json')):raise ValueError('Reference physical account differs')
            for p in (original/'fills.json',central/'fills.json',central/'daily.json'):inputs.append(bound(p))
    registration={'at_utc':datetime.now(timezone.utc).isoformat(),'criteria':'C8-D3','inputs':inputs,
        'program':bound(__file__),'helper':bound(CODE/'scripts/kalcb_lifecycle_readiness_v1.py'),
        'design':bound(CODE/'docs/optimisation/kalcb/experiments/kalcb-exp-0032-first-interval-design.md'),
        'numpy':np.__version__,'native_replays':0,'model_search':False}
    write(out/'registration.json',registration)
    data={};populations={}
    for period in ('is','oos'):
        for s in SCENARIOS:
            account='control/'+period+'/'+s
            original=read(V49/'accounts'/Path(account).with_suffix('.json'))
            targets={r['position_id']:r for r in read(V49/'forward_v1/accounts'/Path(account).with_suffix('.json'))['rows']}
            fills=read(Path(cases[account]['folder'])/'fills.json')
            entry={}
            for f in fills:
                if f['side']=='BUY':entry.setdefault(f['order_id'],f['metadata'])
            rows=[];excluded=Counter()
            for r in original['rows']:
                if not r['position_id']:excluded['unfilled']+=1;continue
                if not r['parent_full_quantity_filled_before35']:excluded['pending_entry']+=1;continue
                if r['parent_sell_intent_at_or_before35'] or r['parent_sale_before35']:excluded['parent_exit']+=1;continue
                t=targets[r['position_id']];held=t['quantity_at_decision']*r['completed_close35']
                equity=r['initial_actual_equity']
                if held<.01*equity:excluded['immaterial']+=1;continue
                if abs(t['observed_net_marked_before35_krw']+t['observed_remaining_net_after35_krw']-r['parent_net_pnl_krw'])>.011:
                    raise ValueError('Changed temporal identity')
                cost=33 if s in ('cost','combined') else 23
                rows.append({'day':r['day'],'symbol':r['symbol'],'position_id':r['position_id'],
                    'features':features(r,entry[r['order_id']]),'target_bps':10000*t['observed_remaining_net_after35_krw']/held,
                    'equity':equity,'held_notional':held,'original_net_krw':r['parent_net_pnl_krw'],'exit_cost_bps':cost,
                    'optimistic_exit_advantage_account_bps':(-t['observed_remaining_net_after35_krw']-held*cost/10000)/equity*10000})
            data[period+'/'+s]=rows;populations[period+'/'+s]={'included':len(rows),'excluded':dict(excluded)}
    outputs={};model_log={}
    days=[r['day'] for r in read(ROOT/'study_v2/cases/control/is/baseline/daily.json')]
    starts=list(range(61,len(days),20))+[len(days)]
    for s in SCENARIOS:
        rows=data['is/'+s];preds=[];fit_records=[]
        for n,j in enumerate(starts):
            training=[r for r in rows if r['day'] in days[:j-1]]
            models=train(training)
            end=starts[n+1] if n+1<len(starts) else None
            evaluation=[r for r in rows if r['day'] in days[j:end]] if j<len(days) else data['oos/'+s]
            predicted=[dict(r,predictions={name:predict(m,r) for name,m in models.items()},training_through=days[j-2]) for r in evaluation]
            if any(r['day']<=days[j-2] for r in predicted):raise ValueError('Future label used')
            fit_records.append({'training_through':days[j-2],'training_rows':len(training), 'training_sessions':j-1,'models':models})
            if j<len(days):preds.extend(predicted)
            else:
                write(out/('oos_'+s+'_predictions.json'),predicted)
                outputs['oos/'+s]={'summary':summarise(predicted),'months':{m:summarise([r for r in predicted if r['day'].startswith(m)]) for m in sorted({r['day'][:7] for r in predicted})}}
        write(out/('is_'+s+'_predictions.json'),preds);write(out/(s+'_models.json'),fit_records)
        outputs['is/'+s]={'summary':summarise(preds),'months':{m:summarise([r for r in preds if r['day'].startswith(m)]) for m in sorted({r['day'][:7] for r in preds})},
                          'unmodelled_initial_sessions':61}
        model_log[s]=len(fit_records)
        print(json.dumps({'scenario':s,'is':outputs['is/'+s]['summary'],'oos':outputs['oos/'+s]['summary']}),flush=True)
    for b in inputs+[registration['program'],registration['helper'],registration['design']]:verify(b)
    write(out/'assessment.json',{'outputs':outputs,'populations':populations,'fit_origins_per_stress':model_log,
         'scope':'Optimistic own-position marked action proxy, no executed candidate, no sequential funded-return claim. All history reused.',
         'economic_objective_achieved':False})
    write(out/'completion.json',{'completed':True,'at_utc':datetime.now(timezone.utc).isoformat(),
          'registration':bound(out/'registration.json'),'assessment':bound(out/'assessment.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.output)
    except Exception as exc:
        if a.output.exists() and not (a.output/'failure.json').exists():write(a.output/'failure.json',{'error':repr(exc)})
        raise
