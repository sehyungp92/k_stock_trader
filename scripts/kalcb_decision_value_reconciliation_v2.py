"""Finite, existing-data decision reconciliation; never fits or launches native work."""
from collections import defaultdict, Counter
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader')
ROOT = BASE/'kalcb_late_entry_20261011'
OLD = BASE/'kalcb_selective_recovery_20261010'
HELPER = REPO/'backtests/strategies/kalcb/decision_value_contract_v1.py'


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,d):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(d,f,indent=2,allow_nan=False)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8') as f: return [json.loads(s) for s in f if s.strip()]


def register(out, tests):
    out.mkdir(exist_ok=False)
    inputs=[ROOT/'v26_opening_decision_audit_v2/completion.json',
            ROOT/'v26_opening_decision_audit_v2/decisions.jsonl.gz',
            ROOT/'v25_pending_funding_screen/assessment_v1/assessment.json',
            OLD/'v2/build/decision_audit.jsonl.gz',
            OLD/'v5_population_setup/build_v1/selection_changes.jsonl',
            OLD/'v5_population/ablation_ledger.json',
            OLD/'v6_qualification/build/completion.json',OLD/'v6_qualification/build/proofs.json',
            OLD/'v6_qualification/build/decisions.jsonl',Path(tests)]
    for parent in ('control','control_prior125'):
        for period in ('is','oos'):
            profile=ROOT/f'v25_pending_funding_screen/cases/{parent}/{period}/input_profile.json'
            inputs.append(profile);d=read(profile)
            inputs.extend(Path(d[k]['path']) for k in ('manifest','policy','plan'))
            policy=Path(d['policy']['path']).parent
            inputs.extend(policy/n for n in ('chronology.json','interaction_cost_margin.admission.jsonl'))
    programs=[Path(__file__).resolve(),HELPER,REPO/'tests/test_kalcb_decision_value_contract_v1.py']
    save(out/'protocol.json',{'version':'decision-value-reconciliation-v2','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Finite existing-data reconciliation; zero fits, new labels, policy changes or native cases.',
        'parents':['control','control_prior125'],'periods':['is','oos'],'scenarios':['baseline','combined'],
        'work':'Reconstruct exact integer feature transforms, prior-specific calibration/veto, preserved score, one-to-one native lifetimes, level errors and foregone economic value. No outcome-selected buckets.',
        'runtime_estimate':'1-5 minutes after bounded source inspection; no full source/native re-audit claimed.',
        'rules':['Unknown or unmatched funded state is not interchangeable with isolated labels.',
                 'No summed isolated PnL presented as portfolio return. All periods reused development.',
                 'No repeated full-ranking correction: inspect historical full-policy failure.',
                 'Stop before any further experiment and freeze its separate finite mechanism.'],
        'bindings':[{'path':str(p),'sha256':sha(p)} for p in inputs],
        'programs':[{'path':str(p),'sha256':sha(p)} for p in programs]})


def run(out):
    if (out/'completion.json').exists() or (out/'run_receipt.json').exists():
        raise ValueError('Already run; preserve evidence and use a versioned successor on failure')
    protocol=read(out/'protocol.json')
    bindings=protocol['bindings']+protocol['programs']
    def verify():
        for b in bindings:
            if sha(b['path'])!=b['sha256']:raise ValueError('Changed binding: '+b['path'])
    verify()
    save(out/'run_receipt.json',{'argv':sys.argv,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    spec=importlib.util.spec_from_file_location('kalcb_decision_value_contract_registered',HELPER)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    completion=read(ROOT/'v26_opening_decision_audit_v2/completion.json')
    if not completion['complete']:raise ValueError('Incomplete source')
    def expected(path, collection):
        b=next(b for b in collection if Path(b['path']).resolve()==Path(path).resolve())
        if sha(path)!=b['sha256']:raise ValueError('Historical source hash mismatch')
    source=ROOT/'v26_opening_decision_audit_v2/decisions.jsonl.gz'
    expected(source,completion['artifacts'])
    records=rows(source);lookup={r['action_id']:r for r in records}
    if len(lookup)!=17445 or len(records)!=17445:raise ValueError('Incomplete/duplicate audit')
    oldaudit=OLD/'v2/build/decision_audit.jsonl.gz';expected(oldaudit,completion['sources'])
    other={r['action_id']:r for r in rows(oldaudit) if r['family']=='decay60'}
    changes={r['day']:r for r in rows(OLD/'v5_population_setup/build_v1/selection_changes.jsonl')
             if r['variant']=='expanding_population_removals_only'}
    parent_actions={x['action']:(x['origin'],x['score']) for r in changes.values() for x in r['parent']}
    prior_decisions={(int(r['prior']),r['day']):r for r in rows(OLD/'v6_qualification/build/decisions.jsonl')}
    build=read(OLD/'v6_qualification/build/completion.json')
    expected(OLD/'v6_qualification/build/proofs.json',[build['proofs']])
    expected(OLD/'v6_qualification/build/decisions.jsonl',[build['decisions']])
    proofs=read(OLD/'v6_qualification/build/proofs.json')
    accounts=read(ROOT/'v25_pending_funding_screen/assessment_v1/assessment.json')['accounts']
    # Full-quantity reference restores the exact integer ratio for lower doses.
    features={}
    for r in records:
        full=lookup[r['base_id']+'/q1']['reference_quantity']
        features[r['action_id']]=m.action_features(r,full)
    output=[];seen=set();bridges=[];stats={};held={}
    for parent,prior in [('control',100),('control_prior125',125)]:
      for period in ('is','oos'):
        profile=read(ROOT/f'v25_pending_funding_screen/cases/{parent}/{period}/input_profile.json')
        for k in ('manifest','policy','plan'):expected(profile[k]['path'],[profile[k]])
        policy_path=Path(profile['policy']['path']);policy=read(policy_path)
        for name in ('chronology.json','interaction_cost_margin.admission.jsonl'):
            if sha(policy_path.parent/name)!=policy['files'][name]:raise ValueError('Policy content drift')
        admissions={a['action_id']:a for a in rows(policy_path.parent/'interaction_cost_margin.admission.jsonl')}
        chron=read(policy_path.parent/'chronology.json')
        actualproof={p['valid_from']:p for p in chron['calibration'] if p.get('family')=='expanding' and p.get('prior_count')==float(prior)}
        originalproof={p['valid_from']:p for p in proofs[str(float(prior))]}
        for day,p in actualproof.items():
            if p!=originalproof[day]:raise ValueError('Actual parent calibration differs')
        subset=[r for r in records if r['period']==period]
        decisions=[]
        for r in subset:
            if r['action_id'] not in parent_actions:continue
            origin,score=parent_actions[r['action_id']]
            d=prior_decisions[prior,r['day']]
            proof=actualproof[d['calibration_valid_from']]
            value=m.reconcile_value(r,other[r['action_id']],origin,proof,score)
            actual_selected=r['action_id'] in admissions
            if value.retained!=actual_selected or actual_selected!=(parent in r['selected']):
                raise ValueError('Final admission reconstruction mismatch: '+r['action_id'])
            if actual_selected and not math.isclose(score,admissions[r['action_id']]['score'],abs_tol=1e-12):
                raise ValueError('Preserved ranking score mismatch')
            if value.retained!=(r['action_id'] in d['retained_actions']):raise ValueError('Historical selection mismatch')
            positions={s:m.unique_position_links(r,parent,s,seen) for s in m.SCENARIOS}
            item={'parent':parent,'period':period,'day':r['day'],'value':asdict(value),
                  'raw':r['raw'],'labels':r['labels'],'positions':positions,
                  'actual_fills_prove_reference_quantity_exchangeability':False}
            output.append(item);decisions.append(item)
        if {x['value']['action_id'] for x in decisions if x['value']['retained']}!=set(admissions):
            raise ValueError('Missing native admission')
        bridges.append({'parent':parent,'period':period,'prior':prior,'parent_actions':len(decisions),
                        'retained':len(admissions),'proofs':len(actualproof),'all_veto_quantity_score_identities_match':True})
        for scenario in m.SCENARIOS:
            positions=[p for x in decisions for p in x['positions'][scenario]]
            original=[p for p in accounts[f'{parent}/{period}/{scenario}']['positions'] if p['route']=='first30_open_anchor']
            if {p['position_id'] for p in positions}!={p['position_id'] for p in original}:
                raise ValueError('Incomplete unique opening lifetimes')
            for field in ('net_pnl_krw','fees_krw','entry_cost_krw'):
                if not math.isclose(sum(p[field] for p in positions),sum(p[field] for p in original),abs_tol=.01):
                    raise ValueError('Lifetime account sum mismatch')
            key=f'{parent}/{period}/{scenario}'
            held[key]={'lifetimes':len(positions),'net_pnl_krw':sum(p['net_pnl_krw'] for p in positions),
                       'fees_krw':sum(p['fees_krw'] for p in positions),
                       'same_reference_quantity':sum(p['reference_quantity_equal'] for p in positions),
                       'top10_winners_krw':sum(sorted((p['net_pnl_krw'] for p in positions),reverse=True)[:10]),
                       'scope':'Actual native unique opening lifetimes; quantity equality alone does not prove state/label transfer.'}
            for group in ('before_veto','retained','vetoed','filled'):
                selected=[x for x in decisions if group=='before_veto' or group=='retained' and x['value']['retained']
                          or group=='vetoed' and not x['value']['retained'] or group=='filled' and x['positions'][scenario]]
                valid=[x for x in selected if x['labels'][scenario]['label_valid']]
                labels=[x['labels'][scenario]['net_value_equity_bps']/x['value']['notional_equity'] for x in valid]
                result={'actions':len(selected),'valid_labels':len(valid),'sessions':len({x['day'] for x in valid}),
                        'positive_labels':sum(y>0 for y in labels),'mean_isolated_net_notional_bps':statistics.fmean(labels) if labels else None,
                        'supported':sum(x['value']['supported'] for x in selected),'forecasts':{}}
                for field in ('raw','core','final'):
                    preds=[(x['raw'] if field=='raw' else x['value'][field])[scenario]/x['value']['notional_equity'] for x in valid]
                    result['forecasts'][field]={'mean_notional_bps':statistics.fmean(preds) if preds else None,
                        'mse':statistics.fmean((p-y)**2 for p,y in zip(preds,labels)) if preds else None,
                        'bias_notional_bps':statistics.fmean(p-y for p,y in zip(preds,labels)) if preds else None}
                stats[key+'/'+group]=result
        print(json.dumps(bridges[-1]),flush=True)
    old=read(OLD/'v5_population/ablation_ledger.json')
    previous={x['recipe']:x['metrics'] for x in old['trials'] if x['recipe'] in
              ('control','expanding_population_full','expanding_population_additions_only','expanding_population_removals_only')}
    with gzip.open(out/'reconciled_decisions.jsonl.gz','xt',encoding='utf-8') as f:
        for x in output:f.write(json.dumps(x,allow_nan=False)+'\n')
    with gzip.open(out/'exact_model_features.jsonl.gz','xt',encoding='utf-8') as f:
        for key,x in features.items():f.write(json.dumps({'action_id':key,'features':x},allow_nan=False)+'\n')
    save(out/'assessment.json',{'complete':True,'bridges':bridges,'metrics':stats,'native_opening':held,
        'historical_full_correction':previous,'transformed_action_rows':len(features),'fitted_models':0,'native_cases':0,
        'scope':'Read-only conditional diagnostics. No isolated result is an attainable portfolio counterfactual. No new improvement proven.'})
    verify()
    outputs=[out/n for n in ('reconciled_decisions.jsonl.gz','exact_model_features.jsonl.gz','assessment.json')]
    save(out/'completion.json',{'complete':True,'study_complete':False,'programs':protocol['programs'],
        'bindings':protocol['bindings'],'outputs':[{'path':str(p),'sha256':sha(p)} for p in outputs],
        'created_at_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','run']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.command=='register':register(args.root,args.tests)
    else:run(args.root)
