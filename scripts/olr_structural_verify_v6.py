"""Independent scalar arithmetic and source-binding audit of stage-1 forecasts."""
from collections import defaultdict
from datetime import date,timedelta
from pathlib import Path
from math import exp,log1p
import json,hashlib
from scripts.olr_structural_v6 import ROOT,OUT,REPO,PARENT,read,write,sha,utc

def midpoint(values):
    values=sorted(values);n=len(values)
    if not n:raise ValueError('Empty median')
    return values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2

def verify():
    receipt=read(ROOT/'train_panel_receipt.json');result=read(OUT/'forecast_calibration.json')
    assert receipt['panel_sha256']==result['panel_sha256']==sha(ROOT/'train_panel.json')
    assert result['predictions_sha256']==sha(ROOT/'forecast_predictions_train.json')
    assert result['model_sha256']==sha(REPO/'strategy_olr/core/liquidity_v6.py')
    assert result['registration_sha256']==sha(ROOT/'registration.json')
    assert receipt['worker_sha256']==sha(REPO/'scripts/olr_structural_source_v6.py')
    for path,digest in receipt['bindings'].items():assert sha(path)==digest,path
    parent_out=REPO/'data/backtests/analysis/olr_minute_exit_20261011/clockfix_generation_1'
    frozen=read(parent_out/'research_harness.json')
    assert sha(parent_out/'research_harness.json')=='f101cfeb297995ffebcd03126456bacbddaf641c3f9ee6649056aff3c86cf5ac'
    proof_path=str(PARENT/'native_source_preflight_train.json')
    assert frozen['artifacts'][proof_path]==sha(proof_path)
    proof=read(proof_path)
    assert proof['original_receipt_sha256']==sha(proof['original_receipt'])
    files=Path(read(PARENT/'native_engines/train/ready.json')['files'])
    for path,digest in receipt['native_imports'].items():assert sha(files/path)==digest,path
    panel=read(ROOT/'train_panel.json');predictions=read(ROOT/'forecast_predictions_train.json')
    indexed={(r['day'],r['symbol']):r for r in panel['rows']}
    assert len(indexed)==len(panel['rows'])==len(panel['universe'])*len(panel['sessions'])
    assert len(panel['universe'])==103 and max(panel['sessions'])=='2026-03-31'
    pair_map={};own=defaultdict(list)
    for (day,symbol),r in indexed.items():
        if r['auction_volume'] is not None:own[symbol].append(r)
    for values in own.values():values.sort(key=lambda r:r['day'])
    for prior,target in zip(panel['sessions'],panel['sessions'][1:]):
        pair_map[target]={}
        for symbol in panel['universe']:
            a,z=indexed[(prior,symbol)],indexed[(target,symbol)]
            if a['prefix_volume'] and z['auction_volume'] is not None:
                pair_map[target][symbol]=log1p(z['auction_volume'])-log1p(a['prefix_volume'])
    by_day=defaultdict(list)
    for r in predictions:by_day[r['day']].append(r)
    checks=0
    for day,rows in by_day.items():
        earliest=(date.fromisoformat(day)-timedelta(days=60)).isoformat()
        pairs={d:v for d,v in pair_map.items() if earliest<=d<day and v}
        prior=midpoint([midpoint(list(v.values())) for v in pairs.values()]) if pairs else None
        for r in rows:
            symbol=r['symbol'];observed=[x['auction_volume'] for x in own[symbol] if earliest<=x['day']<day][-20:]
            broad=midpoint(observed) if observed else None
            ratios=[values[symbol] for d,values in pairs.items() if symbol in values][-20:]
            prefix=indexed[(day,symbol)]['prefix_volume'];normalized=None
            if prefix and prior is not None:
                center=midpoint(ratios) if ratios else prior;n=len(ratios)
                normalized=max(0.,exp(log1p(prefix)+(n*center+5*prior)/(n+5))-1)
            assert r['actual']==indexed[(r['target_day'],symbol)]['auction_volume']
            for model,expected in (('broad_median',broad),('normalized_shrinkage',normalized)):
                evidence=r['predictions'][model];actual=evidence['predicted_volume']
                assert (expected is None and actual is None) or (expected is not None and actual is not None and abs(actual-expected)<=max(1e-7,abs(expected)*1e-12))
                assert evidence['latest_own_day'] is None or evidence['latest_own_day']<day
                if model=='normalized_shrinkage':assert evidence['latest_label_day'] is None or evidence['latest_label_day']<day
                if r['paired']:
                    expected_error=abs(log1p(expected)-log1p(r['actual']))
                    assert abs(r['errors'][model]['absolute_log_error']-expected_error)<1e-10
                    assert r['errors'][model]['oversized']==(expected>r['actual'])
                checks+=1
    out=dict(utc=utc(),passed=True,independent_scalar_forecasts=checks,source_bindings=len(receipt['bindings']),
             native_imports=len(receipt['native_imports']),forecasts_sha256=sha(ROOT/'forecast_predictions_train.json'),
             calibration_sha256=sha(OUT/'forecast_calibration.json'),verifier_sha256=sha(__file__),pnl_or_oos_replayed=False)
    write(OUT/'forecast_verification.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':verify()
