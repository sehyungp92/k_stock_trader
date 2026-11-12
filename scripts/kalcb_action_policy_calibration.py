"""Choose complete action policies using only earlier full-account observations.

The fixed rule uses up to 60 earlier sessions, at least 40 sessions and 20
completed trades in each scenario, positive baseline and combined-stress growth,
and no observed physical/deadline failures in that window. Cash is an explicit
alternative. This is development calibration, not a new independent test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def observed_statistics(case, allowed):
    """Ignore all outcomes outside the allowed completed-session window."""
    daily = case['daily']
    selected = [i for i, r in enumerate(daily) if r['day'] in allowed]
    failures = sorted(set(case['invalid_execution_days']) & allowed)
    deadlines = sorted(d for d, q in case['residual_sessions'].items() if d in allowed and q)
    growth = 0.0
    clean = True
    for i in selected:
        previous = daily[i-1]['equity'] if i else ev.REFERENCE_EQUITY
        if previous <= 0 or daily[i]['equity'] <= 0:
            clean = False
        else:
            growth += math.log(daily[i]['equity']/previous)
    if selected:
        begin, end = selected[0], selected[-1]
        clean &= not daily[end]['inventory'] and not daily[end]['pending']
        if begin:
            clean &= not daily[begin-1]['inventory'] and not daily[begin-1]['pending']
    trades = sum(t['entry_fill_time'][:10] in allowed and t['exit_fill_time'][:10] in allowed
                 for t in case['trades'])
    return {'sessions': len(selected), 'trades': trades, 'log_growth': growth,
            'return_pct': 100*math.expm1(growth), 'clean_account_boundary': bool(clean),
            'physical_failure_days': failures, 'deadline_failure_days': deadlines}


def select_policy(cases, sessions, first, *, lookback=60, minimum_sessions=40, minimum_trades=20):
    prior = sessions[:max(0, sessions.index(first)-1)]  # one session embargo
    allowed = set(prior[-lookback:])
    trials = []
    for name, scenarios in sorted(cases.items()):
        stats = {s: observed_statistics(scenarios[s], allowed) for s in ('baseline','combined')}
        feasible = all(s['sessions'] >= minimum_sessions and s['trades'] >= minimum_trades
                       and s['log_growth'] > 0 and s['clean_account_boundary']
                       and not s['physical_failure_days'] and not s['deadline_failure_days']
                       for s in stats.values())
        trials.append({'policy': name, 'feasible': feasible, 'scenarios': stats})
    candidates = [t for t in trials if t['feasible']]
    best = max(candidates, key=lambda t: (t['scenarios']['baseline']['log_growth'],
               t['scenarios']['baseline']['trades'], t['policy'])) if candidates else None
    return (best['policy'] if best else 'cash'), {
        'valid_from': first, 'observed_from': min(allowed) if allowed else None,
        'observed_through': max(allowed) if allowed else None,
        'latest_label_available_at': max(allowed)+'T23:59:59+09:00' if allowed else None,
        'trials': trials, 'selected': best['policy'] if best else 'cash',
        'rule': {'lookback_sessions':lookback,'minimum_sessions':minimum_sessions,
                 'minimum_completed_trades_per_scenario':minimum_trades,'embargo_sessions':1,
                 'objective':'largest earlier baseline log growth subject to positive combined growth and observed execution constraints; frequency breaks ties'}}


def load_case(directory, name, scenario, policy_manifest_hash):
    def read(file): return json.loads((directory/file).read_text())
    result, request, completion = read('result.json'), read('request.json'), read('completion.json')
    if (not completion['complete'] or result['policy'] != name or result['scenario'] != scenario
            or len({r['source_generation_hash'] for r in (result,request,completion)}) != 1):
        raise ValueError('Incomplete or inconsistent full-account observation')
    audit = read('execution_audit.json')['audit_receipt']
    if audit['identity']['candidate_policy_manifest_sha256'] != policy_manifest_hash:
        raise ValueError('Full-account observation belongs to a different policy')
    errors = set(audit['errors'])
    if errors-{'shared_bar_volume_exceeded'}:
        raise ValueError(f'Unsupported execution observation errors: {errors}')
    failures = ev.physical_fill_failures(read('fills.json'))
    if bool(failures) != ('shared_bar_volume_exceeded' in errors):
        raise ValueError('Physical-volume checks disagree')
    # Keep only the fields used by chronology/count calibration. Full trade
    # metadata remains in the separately hash-bound evidence file.
    trades=[{k:t[k] for k in ('entry_fill_time','exit_fill_time')} for t in read('trades.json')]
    return {'daily':read('daily.json'), 'trades':trades,
            'residual_sessions':result['residual_sessions'],
            'invalid_execution_days':sorted({f['timestamp'][:10] for f in failures})}


def build(catalogs, prepared, output):
    from scripts.kalcb_executed_value_replay import verify_policy
    from scripts.kalcb_executed_value_study import emit
    provenance = json.loads((prepared/'prepared.json').read_text())
    cases, snapshots, proofs, sources = {}, {}, [], {}
    bounds = set()
    for catalog in catalogs:
        policies, replays = Path(catalog['policies']), Path(catalog['replays'])
        manifest_hash = ev.digest(policies/'policies_manifest.json')
        manifest = json.loads((policies/'policies_manifest.json').read_text())
        bounds.add((manifest['evaluation_from'],manifest['evaluation_through']))
        proofs.extend(json.loads((policies/'chronology.json').read_text())['folds'])
        sources[str(policies/'policies_manifest.json')] = manifest_hash
        for name in catalog['names']:
            if name in cases: raise ValueError('Duplicate calibration policy')
            _, parsed = verify_policy(policies,name)
            snapshots[name] = {d.isoformat():s.to_json_dict() for d,s in parsed.items()}
            cases[name] = {}
            for scenario in ('baseline','combined'):
                directory = replays/(name+'_'+scenario)
                cases[name][scenario] = load_case(directory,name,scenario,manifest_hash)
                for file in ('result.json','request.json','completion.json','execution_audit.json',
                             'daily.json','trades.json','fills.json'):
                    sources[str(directory/file)] = ev.digest(directory/file)
    if len(bounds) != 1: raise ValueError('Unmatched evaluation periods')
    first, last = next(iter(bounds))
    sessions = [d for d in provenance['sessions'] if first <= d <= last]
    name = 'action_portfolio_calibrated'
    selected_snapshots, calibration, decisions = [], [], []
    for offset in range(0,len(sessions),20):
        begin = sessions[offset]
        chosen, detail = select_policy(cases,sessions,begin)
        detail.update(valid_through=sessions[min(offset+19,len(sessions)-1)],policy=name)
        calibration.append(detail)
        for day in sessions[offset:offset+20]:
            raw = snapshots[chosen][day] if chosen != 'cash' else None
            items = [({'candidate':c, 'feature_available_at':c['metadata'].get('feature_available_at')},
                      c['selection_score']) for c in raw['candidates']] if raw else []
            forecast = {'policy':name,'selected_policy':chosen,'valid_from':begin}
            selected_snapshots.append(ev.snapshot_for(day,items,source=provenance['opportunities_sha256'],
                                                     forecast=forecast).to_json_dict())
            decisions.extend({'opportunity_id':day+'/'+r['candidate']['symbol'],'trade_date':day,
                              'score':score,'reason':'admitted',**forecast} for r,score in items)
        print(json.dumps({'calibration_from':begin,'selected_policy':chosen}),flush=True)
    ev.write_json(output/(name+'.snapshots.json'),selected_snapshots)
    with (output/(name+'.admission.jsonl')).open('x',encoding='utf-8') as stream:
        for row in decisions: emit(stream,row)
    ev.write_json(output/'chronology.json',{'folds':proofs,'calibration':calibration,
        'scope':'earlier OOF full-account policy selection; reused development history'})
    ev.write_json(output/'source_evidence.json',sources)
    ev.write_json(output/'policies_manifest.json',{'version':'kalcb-action-portfolio-calibration-v1',
        'policies':[name],'evaluation_from':first,'evaluation_through':last,
        'contract':{'budget':8,'production_promotion':False,'quantity_flag_required':True},
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','prepared','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=source_generation_hash()
    a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source,
                                        'catalog_sha256':ev.digest(a.catalog)})
    build(json.loads(a.catalog.read_text()),a.prepared,a.output)
    if source_generation_hash()!=source: raise RuntimeError('Source changed during calibration')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__': main()
