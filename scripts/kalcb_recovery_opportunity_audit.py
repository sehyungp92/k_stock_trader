"""Compare frozen expectations with matching-target native OOS action outcomes."""
from collections import defaultdict
import json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,FULL,read,write,bound,digest,rows,POLICY
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.portfolio_frozen_inference import predictions,calibrated

def main():
    import numpy as np
    from scipy.stats import spearmanr
    from threadpoolctl import threadpool_limits
    if not read(ROOT/'labels/completion.json')['complete']:raise ValueError('Incomplete target grid')
    labels=defaultdict(dict)
    for folder in (ROOT/'labels').iterdir():
        if not folder.is_dir() or not folder.name.startswith('opening_'):continue
        m=read(folder/'grid_manifest.json')
        for name,h in m['files'].items():
            if not name.endswith('.jsonl'):continue
            if digest(folder/name)!=h:raise ValueError('Changed native labels')
            for r in rows(folder/name):
                if r['opportunity_id'] in labels[m['scenario']]:raise ValueError('Duplicate action')
                labels[m['scenario']][r['opportunity_id']]=r
    raw_rows=[]
    for r in rows(FULL/'forward_features/continuation/opportunities.jsonl'):
        if r['trade_date']<'2026-04-01':continue
        compact={k:r[k] for k in ('opportunity_id','trade_date','features')}
        compact.update(candidate={'metadata':{}},action={'quantity':r['action']['quantity']})
        raw_rows.extend(scaled_row(compact,q) for q in (.25,.5,1.))
    directory=Path(read(FULL/'frozen_inference_bridge.json')['models'][0]['source']['path']).parent
    with threadpool_limits(limits=1):predicted,proof=predictions(directory,raw_rows,'opening')
    cals=read(ROOT/'policies/frozen/forward_lineage.json')['calibrations'];core=calibrated(predicted,cals['frozen_core']);support=calibrated(predicted,cals['new_forecast'])
    admissions=rows(ROOT/'policies/frozen'/(POLICY+'.admission.jsonl'))
    def stats(keys,scenario,forecasts):
        all_labels=[labels[scenario][k] for k in keys];valid=[r for r in all_labels if r['label_valid']]
        if not valid:return {'attempts':len(all_labels),'valid':0}
        actual=np.array([r['net_value_equity_bps'] for r in valid]);forecast=np.array([forecasts[r['opportunity_id']][scenario+'_value_bps'] for r in valid])
        return {'attempts':len(all_labels),'valid':len(valid),'physical_or_unresolved_invalid':len(all_labels)-len(valid),
            'positive_share_pct':float(np.mean(actual>0)*100),'mean_actual_reference_equity_bps':float(actual.mean()),
            'median_actual_reference_equity_bps':float(np.median(actual)),'mean_forecast_reference_equity_bps':float(forecast.mean()),
            'mean_forecast_error_bps':float((forecast-actual).mean()),
            'spearman':float(spearmanr(forecast,actual).statistic) if len(actual)>2 and np.ptp(forecast)>0 and np.ptp(actual)>0 else None}
    groups={}
    for period,first,last in [('full','2026-04-01','2026-07-10'),('Apr-May','2026-04-01','2026-05-31'),('Jun-Jul','2026-06-01','2026-07-10')]:
        values={}
        for q in (.25,.5,1.):
            keys=[r['opportunity_id'] for r in raw_rows if r['features']['action_quantity_scale']==q and first<=r['trade_date']<=last]
            values['all_ready_q'+str(q)]={s:stats(keys,s,predicted) for s in ('baseline','combined')}
        for origin,forecasts in [('frozen_core',core),('new_forecast',support)]:
            keys=[r['action_id'] for r in admissions if r['forecast_origin']==origin and first<=r['trade_date']<=last]
            values['admitted_'+origin]={s:stats(keys,s,forecasts) for s in ('baseline','combined')}
        groups[period]=values
    payload={'scope':'Isolated native action labels at fixed KRW100m reference equity, original training-target management. These are not portfolio returns and do not include current selected profit-retention/late-extension changes.',
        'purpose':'Distinguish matching-target forecast degradation from portfolio sizing and management effects.',
        'groups':groups,'models':proof,'label_completion':bound(ROOT/'labels/completion.json'),'program':bound(__file__)}
    write(ROOT/'opportunity_audit.json',payload)
    print(json.dumps(groups['full'],indent=2),flush=True)

if __name__=='__main__':main()
