"""C15 all-episode native lifecycle reconciliation; no financial replay."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from scripts.kalcb_opportunity_closeout_v1 import journal,first_submission


def run(source,out):
    done=read(source/'completion.json');verify([done['registration'],done['assessment']])
    p=read(source/'protocol.json');verify(p['inputs']+p['programs']+p['database_bindings'])
    ledger=read(source/'completed_cases.json');assessment=read(source/'assessment.json')
    c14=next(Path(b['path']) for b in p['inputs'] if 'readiness_v2' in b['path'] and Path(b['path']).name=='assessment.json')
    rows=read(c14)['rows'];bindings=[bound(__file__),bound(source/'protocol.json'),bound(source/'completion.json'),
        done['registration'],done['assessment'],bound(source/'completed_cases.json'),bound(c14)]
    for row in ledger:
        folder=source/'cases'/row['case']['id'];verify([row['completion']]);bindings.append(row['completion'])
        for name in ('activation.json','orders.jsonl.gz','decisions.jsonl.gz','trades.json','fills.json','reconsideration_actions.json'):
            bindings.append(bound(folder/name))
        if row['verdict']['qualified']:
            if not read(folder/'independent_account_verification.json')['passed'] or not read(folder/'funding_review.json')['verified']:
                raise ValueError('Unaccepted account')
            bindings.extend(bound(folder/n) for n in ('independent_account_verification.json','funding_review.json'))
        if row['case'].get('bridge'):
            if not read(folder/'identity_bridge.json')['passed']:raise ValueError('Unbridged account')
            bindings.append(bound(folder/'identity_bridge.json'))
    out.mkdir(parents=True,exist_ok=False)
    write(out/'registration.json',dict(bindings=bindings,scope='C15 declared all-episode trace and account reconciliation, no treatment selection.'))
    results={}
    for row in ledger:
        c=row['case'];name=c['id']
        if c['mode']=='disabled' or not row['verdict']['qualified']:continue
        folder=source/'cases'/name;targets={r['base_id'] for r in rows if r['period']==c['period']};negative_count=len(targets)
        trace=read(source/'traces'/(name+'.json'))
        # Existing helper declares extreme/median paths; the entire rejected population remains included.
        targets.update(trace['traces'])
        actual=journal(folder,targets);references={}
        for arm,field in [('veto','controls'),('prior','prior_controls')]:
            control=Path(p[field][c['period']+'/'+c['scenario']]);parent=journal(control,targets)
            metric=assessment['cells' if arm=='veto' else 'versus_prior100'][name]
            delta=actual['net']-parent['net']
            if abs(delta-metric['net_delta_krw'])>.02:raise ValueError('Full campaign/account delta mismatch')
            attribution=metric['activity_delta']
            decomposed=(attribution['shared_quantity_at_control_unit_pnl_krw']+
                attribution['shared_unit_outcome_at_treatment_qty_krw']+attribution['added_pnl_krw']-attribution['removed_control_pnl_krw'])
            if abs(decomposed-delta)>.02:raise ValueError('Attribution reconciliation failed')
            monthly=defaultdict(float)
            for k,value in actual['affected'].items():monthly[k[:7]]+=value['net']-parent['affected'][k]['net']
            references[arm]=dict(net_delta_krw=delta,first_submission=first_submission(actual['submits'],parent['submits']),
                targets={k:dict(actual=actual['affected'][k],control=parent['affected'][k],
                    delta=actual['affected'][k]['net']-parent['affected'][k]['net']) for k in sorted(targets)},
                target_monthly_delta=dict(monthly),target_sums_are_not_full_policy_effects=True,
                opening_and_later_books={day:dict(actual=[s for s in actual['submits'] if s['episode'].startswith(day)],
                    control=[s for s in parent['submits'] if s['episode'].startswith(day)]) for day in sorted({k[:10] for k in targets})})
        results[name]=dict(activation=read(folder/'activation.json'),actions=read(folder/'reconsideration_actions.json'),
            all_negative_population=negative_count,source_confirmations=sorted({r['base_id'] for r in rows if r['period']==c['period'] and r['states']['above_opening_high']}),
            comparisons=references)
        write(out/'cases'/(name+'.json'),results[name])
    verify(bindings+p['inputs']+p['programs'])
    write(out/'review.json',dict(cases={k:dict(activation=v['activation'],all_negative_population=v['all_negative_population'],
            source_confirmations=v['source_confirmations'],comparisons={arm:dict(net_delta_krw=d['net_delta_krw'],
            first_submission=d['first_submission'],target_monthly_delta=d['target_monthly_delta']) for arm,d in v['comparisons'].items()}) for k,v in results.items()},
            economic_objective_achieved=False,human_assessment_required=True))
    write(out/'completion.json',dict(complete=True,registration=bound(out/'registration.json'),review=bound(out/'review.json'),
          cases=[bound(out/'cases'/(name+'.json')) for name in results]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.source,a.output)
    except Exception as error:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(error)))
        raise
