"""0039 exhaustive, bound causal follow-through. No new trading policy or replay."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from statistics import mean
import traceback
from olr_selective_half_continuation_v1 import read,write,sha,now,status,frozen_imports
from olr_opportunity_continuity_v1 import ledger


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    design=Path(__file__).resolve().parents[1]/'docs/optimisation/olr/experiments/olr-exp-0039.md'
    ready=read(root/'native_engines/train/ready.json');files=Path(ready['files'])
    features_path=files/ready['study']['relative_context_features']
    census_path=root.parent/'opportunity_continuity_v2/train_rows.json'
    full_parent=root.parent/'relative_context_v2/assessment.json'
    inputs=[Path(__file__),Path(__file__).with_name('olr_opportunity_continuity_v1.py'),
        root/'assessment.json',root/'causal_review_v1/assessment.json',features_path,census_path,full_parent]
    write(out/'protocol.json',dict(at=now(),inputs={str(p):sha(p) for p in inputs},design_sha256=sha(design),
        method='All seven extra lifecycles,24 source outside-slot candidates, first divergence and largest signed matched allocations; existing09:30 predicate only. No new financial accounts.',
        already_seen='Final0039 economic summaries and seven source/order/trade summaries. Additional decomposition and contemporaneous interaction checks are diagnostic development evidence.'))
    (out/'design_at_freeze.md').write_bytes(design.read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    status(out,'authenticating_completed_accounts')
    source_hash,completed_case,_=frozen_imports(root)
    a=read(root/'assessment.json');review=read(root/'causal_review_v1/assessment.json')
    artifacts=[];parity={};restarts={};trades={}
    for key,s in a['summaries'].items():
        case=Path(s['case']);assert completed_case(case)['valid']
        assert sha(case/'completion.json')==s['completion_sha256']==read(root/'accepted'/(key+'.json'))['completion_sha256']
        for name,h in read(case/'completion.json')['artifacts'].items():
            assert sha(case/name)==h;artifacts.append(dict(case=key,name=name,sha256=h))
        trades[key]={(t['symbol'],t['entry_decision_time'][:10]):t for t in read(case/'trades.json')}
    assert len(artifacts)==76
    for w in ('train','oos'):
        rr=read(root/'native_engines'/w/'ready.json');assert source_hash(Path(rr['files']))==rr['study']['source_generation_hash']
        p=read(root/(w+'_disabled_parity.json'));assert p['passed'] and all(p['checks'].values());parity[w]=p
        if w=='oos':
            for policy in ('control','access'):
                s=read(root/'runtime/oos'/policy/'study.json');case=root/'cases/oos'/policy
                checks={k:sha(Path(rr['files'])/s[k])==sha(root/'cases/train'/policy/n) for k,n in [
                    ('capacity_checkpoint','capacity_terminal_state.json'),('capacity_checkpoint_completion','completion.json'),('capacity_checkpoint_result','result.json')]}
                restart=read(case/'capacity_restart_receipt.json')
                assert all(checks.values()) and s['configuration']['initial_equity']==1e8
                assert restart['same_configuration_and_scenario'] and restart['all_samples_verified_against_oos_retained_ohlcv']
                restarts[policy]=dict(checks=checks,initial_equity=s['configuration']['initial_equity'],receipt_sha256=sha(case/'capacity_restart_receipt.json'))
    features=read(features_path)
    assert sha(features_path)==ready['manifest']['files'][ready['study']['relative_context_features']]
    assert features['source_database_sha256']==ready['manifest']['files'][ready['study']['database']]
    v=review['windows']['train'];extras=v['extra_traces'];extra_keys={(x['symbol'],x['day']) for x in extras}
    assert len(extra_keys)==7
    census=read(census_path);by_key={(x['symbol'],x['day']):x for x in census};by_day=defaultdict(list)
    for x in census:by_day[x['day']].append(x)
    outside=[]
    for x in census:
        if x['disposition']!='outside_slots':continue
        final=x['timeline'][-1];original=[o for o in by_day[x['day']] if o['rank']<=6]
        expired=[]
        for o in original:
            z=o['timeline'][-1];book=z['book']
            if z['source_complete'] and not o['first_confirmation'] and not o['proposals'] and not book['quantities'].get(o['symbol']) and not any(p['symbol']==o['symbol'] and p['side']=='BUY' for p in book['pending']):expired.append(o['symbol'])
        actual=(x['symbol'],x['day']) in extra_keys
        reason='actual_extra' if actual else 'no_final_confirmation' if not final['passes'] else 'no_expired_original_priority' if not expired else 'other_native_state_or_rank_priority'
        outside.append(dict(symbol=x['symbol'],day=x['day'],rank=x['rank'],first_confirmation=x['first_confirmation'],final_passes=final['passes'],expired_originals=expired,reason=reason))
    assert len(outside)==24
    contexts={}
    for policy in ('control','access'):
        contexts[policy]={}
        for batch in read(Path(a['summaries']['train_'+policy]['case'])/'funded_population_receipt.json')['entry_contexts']:
            for decision in batch['decisions']:
                p=batch['proposals'][decision['intent_id']];key=(p['symbol'],batch['at'])
                item=dict(proposal=p,decision=decision,account=batch['account'],cash=batch['actual_deployable_cash'])
                if key in contexts[policy]:assert contexts[policy][key]==item
                contexts[policy][key]=item
    points=set();morning_keys=[]
    for x in extras:
        points.add(datetime.fromisoformat(x['decision']['at']))
        t=x['trade']
        first_day=next(day for day in features['evaluation_days'] if day>x['day'])
        days=[day for day in features['evaluation_days'] if x['day']<day<=t['exit_fill_time'][:10]]
        assert first_day in days
        for day in days:
            at=day+'T09:30:00+09:00';points.add(datetime.fromisoformat(at));morning_keys.append((x,at))
    status(out,'reconstructing_exact_held_and_funding_states')
    books={p:ledger(Path(a['summaries']['train_'+p]['case']),sorted(points)) for p in ('control','access')}
    full=read(full_parent)['summaries']['full'];full_case=Path(full['case'])
    assert completed_case(full_case)['valid'] and sha(full_case/'completion.json')==full['completion_sha256']
    full_checks={n:sha(full_case/n)==h for n,h in read(full_case/'completion.json')['artifacts'].items()};assert all(full_checks.values())
    full_book=ledger(full_case,sorted(points))
    detailed=[];dates=defaultdict(float);original_same_boundary=[]
    for x in extras:
        t=x['trade'];d=x['decision'];p=d['proposal'];m=p['metadata'];source=by_key[(x['symbol'],x['day'])]
        first=next(z for z in source['timeline'] if z['passes']);final=source['timeline'][-1]
        row=dict(symbol=x['symbol'],day=x['day'],rank=m['candidate_rank'],first_confirmation=first['at'],final_available=final['at'],
            first_close=first['close'],final_close=final['close'],delay_minutes=(datetime.fromisoformat(final['at'])-datetime.fromisoformat(first['at'])).total_seconds()/60,
            observed_close_drift_bps=1e4*(final['close']/first['close']-1),requested=p['qty'],approved=d['decision']['approved_qty'],binding=d['decision']['binding_limit'],
            actual_notional=t['qty']*t['entry_price'],net_pnl=t['net_pnl'],net_bps=1e4*t['net_pnl']/(t['qty']*t['entry_price']),
            entry_time=t['entry_fill_time'],exit_time=t['exit_fill_time'],exit_reason=t['exit_reason'],risk_per_share=m['risk_per_share'],
            mfe_r=t['mfe']/m['risk_per_share'],mae_r=t['mae']/m['risk_per_share'],order_count=len(x['orders']),
            full_context_comparator_book=full_book['snapshots'][d['at']],actual_access_book=books['access']['snapshots'][d['at']],
            scope='Close drift is observed before entry, not an executable earlier-entry payoff. Full-context book shows a dependency only; its capital cannot be spliced into this account.')
        detailed.append(row);dates[x['day']]+=t['net_pnl']
        for key,c in contexts['control'].items():
            if key[1]!=d['at']:continue
            z=contexts['access'].get(key)
            item=dict(symbol=key[0],at=key[1],control_qty=c['decision']['approved_qty'],access_qty=z['decision']['approved_qty'] if z else None,
                original_capacity_equal=bool(z and c['proposal']['metadata']['capacity_forecast']==z['proposal']['metadata']['capacity_forecast']),
                equity_delta=z['account']['equity']-c['account']['equity'] if z else None)
            if item not in original_same_boundary:original_same_boundary.append(item)
    mornings=[]
    for x,at in morning_keys:
        t=x['trade'];s=x['symbol'];book=books['access']['snapshots'][at];f=features['rows'].get(at,{}).get(s)
        quantity=book['quantities'].get(s,0);pending=[p for p in book['pending'] if p['symbol']==s]
        predicate=bool(f and f['peer_count']>=68 and f['own_return']<0 and f['market_median']>=0)
        eligible=bool(quantity and not pending and predicate)
        legs=[e for o in x['orders'] if o['side']=='SELL' for e in o['events'] if e['kind']=='fill' and e['at']>at]
        if quantity:assert sum(e['fill_qty'] for e in legs)==quantity,(s,at,quantity,legs)
        proceeds=sum(e['fill_qty']*e['price']-e['fee'] for e in legs)
        proxy=proceeds-quantity*f['close']*(1-.0027) if f and quantity else None
        mornings.append(dict(symbol=s,entry_day=x['day'],at=at,quantity=quantity,pending=pending,features=f,
            predicate=predicate,actionable_in_saved_policy=eligible,remaining_net_sale_proceeds=proceeds,
            hold_minus_instant_mark_sale_proxy=proxy,
            scope='Existing09:30 information and priority only. Instant marked sale is not an executable funded counterfactual. Flat, native pending and unsupported late-open cases remain explicit.'))
    matched=[z for z in a['comparisons']['train']['accounting']['per_opportunity'] if z['baseline_qty'] and z['treatment_qty']]
    selected=sorted(matched,key=lambda x:x['delta_pnl'])[:1]+sorted(matched,key=lambda x:x['delta_pnl'],reverse=True)[:1]
    attribution=[]
    for z in selected:
        keys=sorted(k for k in contexts['control'].keys()|contexts['access'].keys() if k[0]==z['symbol'] and k[1][:10]==z['day'])
        attribution.append(dict(account_row=z,contexts=[dict(key=k,control=contexts['control'].get(k),access=contexts['access'].get(k)) for k in keys]))
    first=v['first_changed_original_allocation'];ca,cb=first['control'],first['access']
    first_summary=dict(symbol=first['symbol'],at=first['at'],control_quantity=ca['decision']['approved_qty'],access_quantity=cb['decision']['approved_qty'],
        equity_delta=cb['account']['equity']-ca['account']['equity'],cash_delta=cb['cash']-ca['cash'],
        capacity_forecast_equal=ca['proposal']['metadata']['capacity_forecast']==cb['proposal']['metadata']['capacity_forecast'],
        risk_equal=ca['proposal']['metadata']['risk_per_share']==cb['proposal']['metadata']['risk_per_share'],
        entry_close_equal=ca['proposal']['metadata']['entry_submission_close']==cb['proposal']['metadata']['entry_submission_close'])
    totals=a['comparisons']['train']['accounting']['additive_pnl_accounting'];assert math.isclose(totals['total'],a['comparisons']['train']['delta']['return_pct']*1e6,abs_tol=.01)
    new_removed=[z for z in a['comparisons']['train']['accounting']['per_opportunity'] if not z['baseline_qty'] or not z['treatment_qty']]
    for z in new_removed:
        for policy,k in [('control','baseline_qty'),('access','treatment_qty')]:
            t=trades['train_'+policy].get((z['symbol'],z['day']));z[policy+'_notional']=t['qty']*t['entry_price'] if t else 0
    checks=dict(pooled_return_positive=sum(v['delta']['return_pct'] for v in a['comparisons'].values())>0,
        net_meaningful_at_least_two=sum(v['delta']['meaningful'] for v in a['comparisons'].values())>=2,
        positive_changed_meaningful_value=sum(v['changed_meaningful_pnl'] for v in review['windows'].values())>0)
    write(out/'assessment.json',dict(at=now(),artifact_checks=artifacts,disabled_parity=parity,restarts=restarts,
        full_context_comparator_checks=full_checks,extra_rows=detailed,extra_date_pnl=dict(dates),
        outside_census=outside,outside_reasons=dict(Counter(z['reason'] for z in outside)),
        same_boundary_original_allocations=original_same_boundary,first_original_divergence=first_summary,
        largest_signed_matched_cases=attribution,morning_context=mornings,added_removed=new_removed,
        additive_accounting=totals,prospective_support_checks=checks,economic_objective_achieved=False,
        limits=['All data reused development; six intervention dates, no OOS action.',
            'No earlier entry, resized request, context exit or combined funded policy was simulated. Fixed states and additive attribution do not establish joint returns.']))
    for p,h in read(out/'protocol.json')['inputs'].items():assert sha(Path(p))==h
    status(out,'complete_awaiting_disposition',new_financial_accounts=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    try:run(a.root,a.out)
    except Exception:
        if a.out.exists():status(a.out,'failed_preserve',error=traceback.format_exc())
        raise
