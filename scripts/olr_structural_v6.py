"""Bounded structural-research registration and chronology-first controller.

The completed studies remain immutable comparators. No live configuration is
modified. Forecast development does not consume trading P&L or April-July rows.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, tempfile

REPO = Path(__file__).resolve().parents[1]
PARENT = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/minute_exit_v5/clockfix_generation_1')
ROOT = PARENT.parents[1]/'structural_v6'
OUT = REPO/'data/backtests/analysis/olr_structural_20261012'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
def utc(): return datetime.now(timezone.utc).isoformat()
def temp():
    path=ROOT/'runtime_tmp';path.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(path);tempfile.tempdir=str(path)

def register():
    temp();ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    previous=REPO/'data/backtests/analysis/olr_minute_exit_20261011/clockfix_generation_1'
    assert sha(previous/'results.json')=='883f3d76d8cf1a0fea0c22660f104c8466c8840623ca7934bd0f8590337a8415'
    assert sha(previous/'selected_research_policy.json')=='aa2d612718023b915e1b87f65ae9f3a8cf5575edaf30e564b4aadbd9ece3c9a9'
    protocol=dict(version='structural-v6-stage1',registered_utc=utc(),parent=str(PARENT),
        user_objective='Higher robust net return and economically meaningful frequency, lower maximum drawdown; aggressive exposure with bounded risk.',
        problem='Policy-dependent auction observation plus an under-three-samples unconstrained request; earlier broad-history substitution was uncalibrated.',
        immutable_comparators={str(previous/n):sha(previous/n) for n in ('results.json','selected_research_policy.json','delivery_checks.json')},
        design=['Replace routed observation in liquidity forecasting with a source-qualified fixed-universe panel.',
                'Predict next-session executable auction volume from information completed by the current decision time.',
                'Remove sample-count bypass: zero request if history cannot support a finite forecast.',
                'Compare broad median and a fixed five-observation empirical shrinkage prior normalized by completed prefix volume.',
                'Estimate forecast skill chronologically before account performance. No P&L calibration, rank tuning, or OOS selection.',
                'Only after mechanism tests and independent source/arithmetic verification freeze the native integration and bounded account protocol.',
                'Require a causal opportunity ledger and positive chronological incremental economics before advancing any new entry family. F is not reused as an eligible family.'],
        forecast=dict(models=['broad_median','normalized_shrinkage'],lookback_calendar_days=60,own_max_observations=20,
                      prior_strength=5,zero_liquidity='observed executable zero is retained',
                      unknown_liquidity='missing source target is not imputed as zero or skipped to a later day; unknown prediction means no new request',
                      availability='Only labels from a strictly earlier session; prefix must be complete at the exchange-adjusted 14:30 cutoff.',
                      training_end='2026-03-31',score_start='2025-08-07',blocks=[['2025-08-07','2025-10-31'],['2025-11-01','2025-12-31'],['2026-01-01','2026-03-31']]),
        advancement=dict(min_days=30,min_targets=300,min_selected_targets=20,
            max_date_mean_absolute_log_error_ratio=0.90,min_blocks_improved=2,max_oversizing_rate_increase=0.02,
            selection='Normalized shrinkage advances only if every registered forecast gate passes on paired targets; broad median is the structural migration control, not a promoted policy.'),
        budget=dict(forecast_models=2,max_native_policy_variants=3,max_native_full_window_cases=18,
                    native_variants='Legacy clock control; broad coherent migration; normalized forecast only if chronology gates pass. Same nominal/stress/capped IS/OOS conditions.',
                    new_entry_variants=0,entry_followon='An additional bounded protocol must be frozen before outcome evaluation, using training opportunity economics; do not invent entry claims from these forecast models.'),
        objective_gates='All original/guard/retained return/DD/frequency/notional/stress/capped gates remain required for a full successor. Migration correctness is a separate engineering result; no automatic promotion.',
        no_live_activation=True,independent_validation=False,scope='April-July repeatedly inspected development; future unobserved validation still required.',
        implementation_constraints=['New generation on E; old sources, engines, repairs, profiles and completed studies unchanged.',
            'Keep full native minute orders/cancels/acks/partials/costs/source/capacity/ledger audits and both DD grids.',
            'Restore matching actual IS market state into OOS, and independently verify its prefix.',
            'No claimed fill liquidity from a forecast, no hindsight exclusions, no broad parameter search.'])
    write(ROOT/'registration.json',protocol);write(OUT/'registration.json',protocol)
    print(json.dumps(dict(registered=str(ROOT/'registration.json'),sha256=sha(ROOT/'registration.json'))),flush=True)

def launch_extract():
    temp();assert (ROOT/'registration.json').exists()
    if (ROOT/'train_panel.json').exists():raise RuntimeError('Authenticate completed extraction; do not overwrite')
    command=[sys.executable,'-B','-u',str(REPO/'scripts/olr_structural_source_v6.py'),str(ROOT),str(PARENT),'train']
    with (ROOT/'train_panel.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        query="Get-CimInstance Win32_Process -Filter 'ProcessId="+str(p.pid)+"' | Select-Object CommandLine,@{n='creation_time_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
        identity=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',query],text=True))
        write(ROOT/'train_panel_process.json',dict(pid=p.pid,**identity,launch_utc=utc(),command=command))
        print(json.dumps(dict(pid=p.pid,phase='train_source_panel')),flush=True)
        rc=p.wait()
    if rc:raise RuntimeError('Preserve failed extraction before repair')

def calibrate():
    from strategy_olr.core.liquidity_v6 import CausalLiquidityPanel, MODELS
    from collections import defaultdict
    from math import log1p
    from statistics import mean
    temp();protocol=read(ROOT/'registration.json');proof=read(ROOT/'train_panel_receipt.json')
    assert proof['panel_sha256']==sha(ROOT/'train_panel.json')
    assert proof['worker_sha256']==sha(REPO/'scripts/olr_structural_source_v6.py')
    data=read(ROOT/'train_panel.json');panel=CausalLiquidityPanel(data['rows'],data['sessions'],data['universe'])
    assert max(data['sessions'])<='2026-03-31'
    predictions=[];coverage=defaultdict(int)
    for day,target in zip(data['sessions'],data['sessions'][1:]):
        if day<protocol['forecast']['score_start']:continue
        for symbol in data['universe']:
            actual=panel.rows.get((target,symbol),{}).get('auction_volume')
            pred={m:panel.forecast(day,symbol,m) for m in MODELS}
            selected=symbol in data['selected'].get(day,[])
            row=dict(day=day,symbol=symbol,target_day=target,actual=actual,selected=selected,predictions=pred)
            coverage['decisions']+=1;coverage['missing_target']+=actual is None
            row['paired']=actual is not None and all(p['predicted_volume'] is not None for p in pred.values())
            if row['paired']:
                row['errors']={m:dict(absolute_log_error=abs(log1p(p['predicted_volume'])-log1p(actual)),
                    oversized=p['predicted_volume']>actual) for m,p in pred.items()}
            predictions.append(row)
    def summarize(rows):
        rows=[r for r in rows if r['paired']];result=dict(targets=len(rows),days=len({r['day'] for r in rows}))
        for model in MODELS:
            daily=defaultdict(list)
            for r in rows:daily[r['day']].append(r['errors'][model])
            result[model]=dict(date_mean_absolute_log_error=mean(mean(x['absolute_log_error'] for x in values) for values in daily.values()) if daily else None,
                oversized_rate=mean(mean(x['oversized'] for x in values) for values in daily.values()) if daily else None)
        return result
    all_rows=summarize(predictions);selected=summarize([r for r in predictions if r['selected']])
    blocks=[summarize([r for r in predictions if lo<=r['day']<=hi]) for lo,hi in protocol['forecast']['blocks']]
    a,z=all_rows['broad_median'],all_rows['normalized_shrinkage'];g=protocol['advancement']
    checks=dict(enough_days=all_rows['days']>=g['min_days'],enough_targets=all_rows['targets']>=g['min_targets'],
        enough_selected_targets=selected['targets']>=g['min_selected_targets'],
        forecast_error=z['date_mean_absolute_log_error']<=g['max_date_mean_absolute_log_error_ratio']*a['date_mean_absolute_log_error'],
        block_consistency=sum(b['normalized_shrinkage']['date_mean_absolute_log_error']<b['broad_median']['date_mean_absolute_log_error'] for b in blocks)>=g['min_blocks_improved'],
        oversizing=z['oversized_rate']<=a['oversized_rate']+g['max_oversizing_rate_increase'],
        selected_error=selected['normalized_shrinkage']['date_mean_absolute_log_error']<=selected['broad_median']['date_mean_absolute_log_error'])
    write(ROOT/'forecast_predictions_train.json',predictions)
    result=dict(utc=utc(),registration_sha256=sha(ROOT/'registration.json'),panel_sha256=proof['panel_sha256'],
        predictions_sha256=sha(ROOT/'forecast_predictions_train.json'),model_sha256=sha(REPO/'strategy_olr/core/liquidity_v6.py'),
        coverage=dict(coverage),all_targets=all_rows,selected_targets=selected,blocks=blocks,advancement_checks=checks,
        normalized_model_advances=all(checks.values()),performance_evaluated=False,oos_consumed=False)
    write(OUT/'forecast_calibration.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['register','extract','calibrate']);args=p.parse_args()
    {'register':register,'extract':launch_extract,'calibrate':calibrate}[args.phase]()
