"""C9-A attribution of the fixed C8-D3 model; no new fits or policies."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.kalcb_first_interval_information_v1 import features, predict, V49
from scripts.kalcb_lifecycle_readiness_v1 import read,write,bound,verify,ROOT

SOURCE=Path('E:/CodexResearch/k_stock_trader/kalcb_lifecycle_readiness_20261016/first_interval_v1')
OUT=Path('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016/d3_support_v1')
NAMES=['opening_return_atr','opening_range_atr','opening_vwap_extension_atr','opening_close_location',
       'log_rel_volume','atr_pct','first_gap_atr','first_body_atr','first_high_above_opening_atr',
       'first_low_above_opening_low_atr','first_close_location','log_interval_volume']


def summary(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['day']].append(r)
    mean=lambda key:sum(sum(r[key] for r in rr)/len(rr) for rr in groups.values())/len(groups) if groups else None
    return {'positions':len(rows),'days':len(groups),'mean_predicted_remaining_notional_bps':mean('prediction'),
            'mean_actual_remaining_notional_bps':mean('target_bps'),
            'prediction_below_liquidation_cost':sum(r['prediction']<-r['exit_cost_bps'] for r in rows),
            'outside_opening':sum(bool(r['outside_opening']) for r in rows),
            'outside_incremental':sum(bool(r['outside_incremental']) for r in rows),
            'beyond_three_training_sd':sum(bool(r['large_z']) for r in rows),
            'clipped':sum(bool(r['clipped']) for r in rows),
            'mean_intercept':mean('intercept'),'mean_opening_contribution':mean('opening_contribution'),
            'mean_incremental_contribution':mean('incremental_contribution'),
            'outside_feature_counts':dict(Counter(n for r in rows for n in r['outside_opening']+r['outside_incremental']))}


def main():
    OUT.mkdir(exist_ok=False)
    registration=read(SOURCE/'registration.json')
    # Exact old implementation/data authentication, without modifying or refitting it.
    for b in registration['inputs']+[registration['program'],registration['helper'],registration['design']]:verify(b)
    completion=read(SOURCE/'completion.json');verify(completion['registration']);verify(completion['assessment'])
    bindings=[bound(p) for p in SOURCE.glob('*.json')]
    write(OUT/'registration.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),
          'design':bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0033.md'),'inputs':bindings,
          'scope':'Raw envelope, standardized support and additive attribution only; no fits or changed action.'})
    cases={r['key']:r for r in read(V49/'protocol.json')['cases']};assessment={};all_rows=[]
    for scenario in ('baseline','participation','cost','combined'):
        source=read(V49/'accounts/control/is'/Path(scenario+'.json'))['rows']
        fills=read(Path(cases['control/is/'+scenario]['folder'])/'fills.json');entry={}
        for f in fills:
            if f['side']=='BUY':entry.setdefault(f['order_id'],f['metadata'])
        targets={r['position_id']:r for r in read(V49/'forward_v1/accounts/control/is'/Path(scenario+'.json'))['rows']}
        training=[]
        for r in source:
            if not r['position_id'] or not r['parent_full_quantity_filled_before35'] or r['parent_sell_intent_at_or_before35'] or r['parent_sale_before35']:continue
            t=targets[r['position_id']];held=t['quantity_at_decision']*r['completed_close35']
            if held<.01*r['initial_actual_equity']:continue
            training.append({'day':r['day'],'features':features(r,entry[r['order_id']]),'target_bps':10000*t['observed_remaining_net_after35_krw']/held})
        models={r['training_through']:r for r in read(SOURCE/(scenario+'_models.json'))}
        for period in ('is','oos'):
            result=[]
            for r in read(SOURCE/(period+'_'+scenario+'_predictions.json')):
                record=models[r['training_through']];m=record['models']['N']
                tr=[x for x in training if x['day']<=r['training_through']]
                if len(tr)!=record['training_rows']:raise ValueError('Changed training support')
                raw=r['features'];low=[min(t['features'][i] for t in tr) for i in range(12)];high=[max(t['features'][i] for t in tr) for i in range(12)]
                z=[(v-a)/b for v,a,b in zip(raw,m['feature_mean'],m['feature_scale'])]
                contributions=[(min(6,max(-6,v))-c)*beta for v,c,beta in zip(z,m['clipped_center'],m['coef'])]
                y=m['mean']+sum(contributions)
                if abs(y-r['predictions']['N'])>1e-9 or abs(y-predict(m,r))>1e-9:raise ValueError('Prediction attribution identity')
                outside=[NAMES[i] for i in range(12) if raw[i]<low[i] or raw[i]>high[i]]
                row={k:r[k] for k in ('day','symbol','position_id','target_bps','exit_cost_bps','training_through')}
                row.update(scenario=scenario,period=period,prediction=y,intercept=m['mean'],
                    opening_contribution=sum(contributions[:6]),incremental_contribution=sum(contributions[6:]),
                    contributions=dict(zip(NAMES,contributions)),outside_opening=[x for x in outside if x in NAMES[:6]],
                    outside_incremental=[x for x in outside if x in NAMES[6:]],
                    large_z=[NAMES[i] for i,x in enumerate(z) if abs(x)>3],clipped=[NAMES[i] for i,x in enumerate(z) if abs(x)>6])
                result.append(row);all_rows.append(row)
            assessment[period+'/'+scenario]={'all':summary(result),'inside_raw_envelope':summary([r for r in result if not r['outside_opening'] and not r['outside_incremental']]),
                'outside_raw_envelope':summary([r for r in result if r['outside_opening'] or r['outside_incremental']]),
                'months':{m:summary([r for r in result if r['day'].startswith(m)]) for m in sorted({r['day'][:7] for r in result})}}
    write(OUT/'rows.json',all_rows);write(OUT/'assessment.json',assessment)
    write(OUT/'completion.json',{'complete':True,'at_utc':datetime.now(timezone.utc).isoformat(),'registration':bound(OUT/'registration.json'),
          'rows':bound(OUT/'rows.json'),'assessment':bound(OUT/'assessment.json'),'fits':0})
    print(json.dumps({k:v['all'] for k,v in assessment.items() if k.startswith('oos/')}),flush=True)


if __name__=='__main__':
    main()
