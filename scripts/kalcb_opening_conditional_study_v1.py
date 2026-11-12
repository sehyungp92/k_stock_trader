"""Finite data-only forward economic-model contrast; no native policy producer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb import opening_conditional_value_v1 as model

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
DEFAULT = ROOT/'v27_opening_conditional_value'
AUDIT = ROOT/'v26_opening_decision_audit_v2'
CALENDAR = ROOT.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def bound(path): return {'path':str(Path(path).absolute()),'sha256':sha(path)}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2,allow_nan=False)


def verify(items):
    for item in items:
        if sha(item['path'])!=item['sha256']: raise ValueError('Changed registered input: '+item['path'])


def register(root, tests):
    receipt = read(tests)
    if not receipt['passed']: raise ValueError('Failed focused tests')
    audit=read(AUDIT/'completion.json')
    if not audit['complete'] or audit['fitted_models'] or audit['native_cases']: raise ValueError('Unexpected audit')
    verify(audit['sources']+audit['artifacts']+[audit['program']])
    if read(AUDIT/'assessment.json')['management_field_differences']: raise ValueError('Label management mismatch')
    programs=[bound(__file__),bound(model.__file__),bound(REPO/'tests/backtests/strategies/test_kalcb_opening_conditional_value_v1.py')]
    inputs=[bound(AUDIT/'completion.json'),bound(AUDIT/'decisions.jsonl.gz'),bound(AUDIT/'assessment.json'),bound(CALENDAR),bound(tests)]
    root.mkdir(exist_ok=False)
    write(root/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'kind':'finite_data_only_economic_model_experiment', 'recipes':['unit_mean','conditional'],
        'parents':['prior100','prior125'],'programs':programs,'bindings':inputs,
        'features':model.FEATURE_NAMES, 'model':'Weighted ridge residual in exact planned-notional bps. Three quantity intercepts penalty100; conditional standardized features penalty120. Zero-correction prior. No parameter search.',
        'population':'Historical physical/capacity eligible actions with positive baseline AND combined core forecasts, including original nonselected actions. Current labels never select the forecast population. Both scenario labels must be mature and valid to fit. Retain invalid/failure/no-fill rows in immutable audit.',
        'weights':'Each issuer-day has one total weight across dependent quantities; equal session weighting, normalized to number of issuer-days. Scenarios are fitted separately, never counted as independent exposure.',
        'chronology':'Expanding genuine-forward residual history; min60 mature distinct sessions and100 issuer-days; one-session embargo; exact inherited20-session base window dates. No pre-IS residual backfill or April reset. Under-supported maps retain unchanged parent.',
        'quantity_support':{'issuer_days':20,'sessions':8},
        'domain':'Both components share prior-only feature min/max envelope; no extrapolation or silent missing features. Existing physical threshold/capacity gates unchanged.',
        'action_scope':'Read-only hypothetical selection on exact original pre-population selected action/quantity. Corrected both-scenario positive required for restored admissions; supported nonpositive removes current admission. No quantity/rank/management/timing changes, no new opportunity geometry, no active policy files.',
        'component_control':'Identical samples/weights/quantity intercepts/domain with no contextual regressors, to test added contextual information rather than pooled mean correction.',
        'readiness_before_any_native_registration':[
            'At least20 distinct changed issuer-days across8 sessions in each IS/OOS period for a parent; dependent scenarios do not inflate support.',
            'Context correction must improve IS session-balanced mean squared forecast error against unit_mean on common supported original-selected actions in BOTH scenarios.',
            'IS sum of isolated exact-action changes must be nonnegative in BOTH scenarios. Report removed/restored positive payoff mass separately, including top-tail costs; individual winner sacrifice is not an automatic veto. These are research off-ramps, not portfolio or statistical guarantees.',
            'Report every OOS outcome, concentrated gain and component; do not retune after failure. Any readiness pass still requires real native guards, funding and exact-parent bridges before portfolio performance claims.'
        ],
        'readiness_failure':'No native dispatch and no hidden parameter/feature search. Preserve the failed mechanism and reassess which structural hypothesis has new evidence.',
        'native_experiments_registered':0,'policies_written':0,
        'estimate':'Data-only fit/review under10min after registration; native experiment count/runtime requires a separate evidence-supported protocol.',
        'limitations':'April-July is reused development; labels are isolated account outcomes and cannot be added as portfolio returns. No selection claim for a statistically independent OOS. No promotion/live changes.'})
    print(json.dumps({'registered':str(root),'recipes':2,'native_cases':0}),flush=True)


def assess(rows, forecasts):
    bykey={r['action_id']:r for r in rows}
    result={}
    for recipe in ('unit_mean','conditional'):
        for parent in ('control','control_prior125'):
            for period in ('is','oos'):
                items=[]
                for action_id, forecast in forecasts[recipe].items():
                    row=bykey[action_id]
                    if row['period']!=period: continue
                    before=parent in row['selected']
                    # No new ranks/quantities are invented. Only original common selected actions.
                    after=before
                    if row['original_parent_selected'] and forecast['supported']:
                        after=forecast['positive_both']
                    if before==after: continue
                    item={'action_id':action_id,'base_id':row['base_id'],'day':row['day'],
                          'before':before,'after':after,'direction':'restored' if after else 'removed',
                          'delta_isolated_equity_bps':{},'positive_payoff_removed_or_restored':{}}
                    for scenario in model.SCENARIOS:
                        label=row['labels'][scenario]
                        item['delta_isolated_equity_bps'][scenario] = (1 if after else -1)*label['net_value_equity_bps'] if label['label_valid'] else None
                        item['positive_payoff_removed_or_restored'][scenario] = max(0.,label['net_value_equity_bps']) if label['label_valid'] else None
                    items.append(item)
                score={s:{'delta_isolated_equity_bps':sum(x['delta_isolated_equity_bps'][s] or 0 for x in items),
                    'unknown_labels':sum(x['delta_isolated_equity_bps'][s] is None for x in items),
                    'removed_positive_mass':sum(x['positive_payoff_removed_or_restored'][s] or 0 for x in items if not x['after']),
                    'restored_positive_mass':sum(x['positive_payoff_removed_or_restored'][s] or 0 for x in items if x['after'])} for s in model.SCENARIOS}
                result[f'{recipe}/{parent}/{period}']={'changes':items,'distinct_issuer_days':len({r['base_id'] for r in items}),
                    'sessions':len({r['day'] for r in items}), 'restored':sum(r['after'] for r in items),
                    'removed':sum(not r['after'] for r in items),'isolated_descriptive_only':score}
    errors={}
    for period in ('is','oos'):
        for scenario in model.SCENARIOS:
            sessions=defaultdict(lambda:defaultdict(list))
            for row in rows:
                key=row['action_id']
                if row['period']!=period or not row['original_parent_selected'] or not row['labels'][scenario]['label_valid']: continue
                if any(not forecasts[recipe][key]['supported'] for recipe in ('unit_mean','conditional')): continue
                y=row['labels'][scenario]['net_value_equity_bps']/row['notional_equity']
                for recipe in ('unit_mean','conditional'):
                    e=forecasts[recipe][key]['values'][scenario]['corrected_notional_bps']-y
                    sessions[row['day']][recipe].append(e*e)
            errors[period+'/'+scenario]={'sessions':len(sessions),**{recipe:sum(sum(v[recipe])/len(v[recipe]) for v in sessions.values())/len(sessions) if sessions else None for recipe in ('unit_mean','conditional')}}
    readiness={}
    for parent in ('control','control_prior125'):
        support=all(result[f'conditional/{parent}/{p}']['distinct_issuer_days']>=20 and result[f'conditional/{parent}/{p}']['sessions']>=8 for p in ('is','oos'))
        discrimination=all(errors['is/'+s]['conditional'] is not None and errors['is/'+s]['conditional'] < errors['is/'+s]['unit_mean'] for s in model.SCENARIOS)
        economics=all(x['unknown_labels']==0 and x['delta_isolated_equity_bps']>=0 for x in result[f'conditional/{parent}/is']['isolated_descriptive_only'].values())
        readiness[parent]={'support':support,'is_context_error_improves':discrimination,'is_net_isolated_delta_nonnegative':economics,
            'eligible_for_separate_native_protocol':support and discrimination and economics}
    return {'hypothetical_changes':result,'common_support_session_balanced_mse':errors,'readiness':readiness,
        'portfolio_performance_measured':False,'study_complete':False,'no_automatic_native_dispatch':True}


def run(root):
    protocol=read(root/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    if (root/'supervisor_start.json').exists(): raise ValueError('Already launched; do not restart registered fit')
    import psutil
    process=psutil.Process()
    write(root/'supervisor_start.json',{'pid':process.pid,'create_time':process.create_time(),'cmdline':process.cmdline(),'at_utc':datetime.now(timezone.utc).isoformat()})
    calendar=read(CALENDAR);sessions=calendar['sessions'];windows=calendar['windows']
    assert len(sessions)==286 and sessions.index('2026-04-01')==217
    with gzip.open(AUDIT/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:rows=[json.loads(l) for l in stream]
    proofs=[]; forecasts={'unit_mean':{},'conditional':{}}
    for first,last in windows:
        proof=model.fit(rows,sessions,first,last);proofs.append(proof)
        for row in rows:
            if first<=row['day']<=last:
                for recipe in forecasts: forecasts[recipe][row['action_id']]=model.forecast(row,proof,recipe)
        print(json.dumps({'through':last,'status':proof['status'],'sessions':proof['training_sessions'],'issuer_days':proof['training_issuer_days']}),flush=True)
    if any(len(forecasts[recipe])!=len(rows) for recipe in forecasts): raise ValueError('Incomplete forecast coverage')
    write(root/'chronology.json',proofs)
    with gzip.open(root/'forecasts.json.gz','xt',encoding='utf-8') as stream:json.dump(forecasts,stream,allow_nan=False)
    assessment=assess(rows,forecasts)
    write(root/'assessment.json',assessment)
    verify(protocol['programs']+protocol['bindings'])
    write(root/'completion.json',{'complete':True,'study_complete':False,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'native_cases':0,'policies_written':0,'recipes':2,'fits':len(proofs),
        'artifacts':[bound(root/f) for f in ('chronology.json','forecasts.json.gz','assessment.json')],
        'support_counts':{recipe:dict(Counter('supported' if v['supported'] else v['reason'] for v in predictions.values())) for recipe,predictions in forecasts.items()},
        'readiness':assessment['readiness']})
    print(json.dumps({'complete':True,'readiness':assessment['readiness']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','run'));parser.add_argument('--root',type=Path,default=DEFAULT);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.command=='register':register(args.root,args.tests)
    else:run(args.root)
