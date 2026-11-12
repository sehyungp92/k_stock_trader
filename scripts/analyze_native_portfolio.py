"""Post-selection attribution. Shadow outcomes are diagnostics, never rank inputs."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.portfolio_native_budget_study import read, checked, SCENARIOS
from backtests.strategies.kalcb.executed_value import write_json, digest


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream: yield json.loads(line)


def key(t): return (t['strategy_id'], t['symbol'], t['entry_decision_time'])


def distribution(trades):
    return {'entries':len(trades),'wins':sum(t['net_pnl']>0 for t in trades),
            'losses':sum(t['net_pnl']<0 for t in trades),
            'net_pnl_krw':sum(t['net_pnl'] for t in trades)}


def describe(case):
    import numpy as np
    result=checked(case);trades=read(case/'trades.json');daily=read(case/'daily.json')
    request=read(case/'request.json'); study=read(request['arguments']['study'])
    initial=study['initial_equity'];nav=np.asarray([initial]+[d['equity'] for d in daily])
    daily_returns=nav[1:]/nav[:-1]-1
    pnl=defaultdict(float);entry_month=defaultdict(float);exit_month=defaultdict(float)
    for t in trades:
        pnl[t['strategy_id']]+=t['net_pnl']
        entry_month[t['entry_fill_time'][:7]]+=t['net_pnl']
        exit_month[t['exit_fill_time'][:7]]+=t['net_pnl']
    if abs(sum(pnl.values())-(nav[-1]-initial))>1.:raise ValueError('Closed PNL does not reconcile to NAV')
    kalcb_daily=defaultdict(float)
    for t in trades:
        if t['strategy_id']=='KALCB': kalcb_daily[t['exit_fill_time'][:10]]+=t['net_pnl']
    k=np.asarray([kalcb_daily[d['day']] for d in daily])/nav[:-1]
    o=daily_returns-k
    day_index={d['day']:i for i,d in enumerate(daily)}
    olr_holding=[day_index[t['exit_fill_time'][:10]]-day_index[t['entry_fill_time'][:10]] for t in trades if t['strategy_id']=='OLR']
    overnight=[max(0.,(d['equity']-d['cash'])/d['equity']) for d in daily]
    admissions=list(rows(case/'admissions.jsonl.gz'))
    native=list(rows(case/'native_budget_admissions.jsonl.gz')) if (case/'native_budget_admissions.jsonl.gz').exists() else []
    constrained=[r for r in native if r['approved_qty']<r['offered_qty']]
    binding=Counter(reason for r in constrained for reason in r['reservation']['binding_constraints'])
    physical_blocked={r['order']['order_id'] for r in admissions if r['admitted_qty']==0}
    funding_blocked={r['order']['order_id'] for r in native if r['approved_qty']==0 and r['offered_qty']>0}
    fills=read(case/'fills.json') if (case/'fills.json').exists() else []
    actual_filled={f['order_id'] for f in fills if f['side']=='BUY'}
    groups=defaultdict(list)
    for fill in fills:groups[(fill['symbol'],fill['timestamp'],fill['metadata']['execution_phase'])].append(fill)
    crosses=[]
    for group,events in groups.items():
        buyers={f['strategy_id'] for f in events if f['side']=='BUY'}
        sellers={f['strategy_id'] for f in events if f['side']=='SELL'}
        if any(b!=s for b in buyers for s in sellers):
            crosses.append({'symbol':group[0],'timestamp':group[1],'phase':group[2],
                'buy_qty':sum(f['qty'] for f in events if f['side']=='BUY'),
                'sell_qty':sum(f['qty'] for f in events if f['side']=='SELL')})
    submissions=list(rows(case/'allocation_events.jsonl.gz')) if (case/'allocation_events.jsonl.gz').exists() else []
    submission_times=defaultdict(set)
    for e in submissions: submission_times[e['strategy_id']].add(e['timestamp'])
    recycled=[e for e in submissions if e['native_metadata'].get('allocation_redistributed_qty',0)>0]
    recycling=[]
    for e in recycled:
        oid=e['submitted_order_id']; meta=e['native_metadata']
        filled=sum(f['qty'] for f in fills if f['order_id']==oid)
        base=meta['allocation_before_redistribution']
        recycling.append({'date':e['timestamp'][:10],'symbol':e['symbol'],'order_id':oid,
            'native_rank':meta.get('candidate_rank'),'before':base,'extra_assigned':meta['allocation_redistributed_qty'],
            'filled':filled,'extra_filled':max(0,filled-base)})
    return {'result':{k:result[k] for k in ('recipe','scenario','return_pct','drawdown_pct','trades','strategy_counts','trades_per_21_sessions','audit_status')},
        'frequency_detail':{'collapsed_strategy_entries':len(trades),
            'unique_symbol_fill_timestamps':len({(t['symbol'],t['entry_fill_time']) for t in trades}),
            'unique_issuer_entry_days':len({(t['symbol'],t['entry_fill_time'][:10]) for t in trades}),
            'entry_notional_krw':sum(t['qty']*t['entry_price'] for t in trades)},
        'pnl_by_strategy':dict(pnl),'entry_month_pnl':dict(entry_month),'exit_month_pnl':dict(exit_month),
        'daily_marked_sleeve_contributions':{'correlation':float(np.corrcoef(k,o)[0,1]) if np.std(k)>0 and np.std(o)>0 else None,
            'both_negative_sessions':int(sum((k<0)&(o<0))),
            'basis':'KALCB is flat at every audited session end, so its realized daily net PNL is its marked contribution; OLR is the reconciled account residual.'},
        'olr_liquidation_delay':{'entries':len(olr_holding),'final_exit_later_than_next_session':sum(x>1 for x in olr_holding),
            'max_session_distance':max(olr_holding,default=0)},
        'january_february_exit_profit_share':sum(exit_month.get(m,0) for m in ('2026-01','2026-02'))/sum(pnl.values()) if sum(pnl.values()) else None,
        'worst_session_pct':float(100*min(daily_returns)),
        'overnight_exposure':{'mean_nav_fraction':float(np.mean(overnight)),'max_nav_fraction':max(overnight),
            'max_10pct_gap_nav_loss_pct':max(overnight)*10,'max_20pct_gap_nav_loss_pct':max(overnight)*20,
            'interpretation':'Static first-order exposure shock; excludes liquidation costs, correlated intraday losses and changes in position management. Not a replay or loss bound.'},
        'admission_diagnostics':{'physical_zero_admission_orders':len(physical_blocked),
            'additional_budget_zero_admission_orders':len(funding_blocked),
            'zero_admission_orders_without_any_entry_fill':len((physical_blocked|funding_blocked)-actual_filled),
            'budget_resize_orders':len({r['order']['order_id'] for r in native if 0<r['approved_qty']<r['offered_qty']}),
            'binding_events':dict(binding),
            'max_planned_kalcb_stop_risk_nav_fraction':max((r['ledger']['committed_planned_kalcb_risk']/r['ledger']['equity'] for r in native if r['ledger']['equity']>0),default=0),
            'scope':'Admission observations; repeated attempts are collapsed by order ID. A zero admission may cancel a partly filled order and is not automatically a missing economic trade.'},
        'observed_funding_exposure':{
            'max_gross_nav_at_buy_admission': max((r['exposure']['gross']/r['equity'] for r in admissions if r['equity']>0), default=0),
            'max_negative_cash_nav_at_buy_admission': max((max(0., -r['cash']/r['equity']) for r in admissions if r['equity']>0), default=0),
            'negative_cash_session_ends': sum(d['cash'] < 0 for d in daily),
            'minimum_session_end_cash_nav': min(d['cash']/d['equity'] for d in daily if d['equity']>0),
            'scope': 'Observed before-buy admission snapshots, not a bound or an exhaustive post-fill exposure maximum. Native KALCB buying power permits negative intraday cash; live availability and financing charges require broker evidence.'},
        'same_phase_opposite_strategy_flows':crosses,
        'flow_interpretation':'Observed coincident buys/sells only; no assumed internal crossing, avoided tax, or executable profit is credited.',
        'recycling_interventions':recycling,
        'submission_timing':{
            'clock_ranges': {sid: [min(t[11:19] for t in values), max(t[11:19] for t in values)] for sid, values in submission_times.items()},
            'simultaneous_cross_strategy_timestamps': len(submission_times['KALCB'] & submission_times['OLR']),
            'scope': 'Native request submission clocks, including special-session schedule shifts; not fill clocks. Pending orders and existing holdings can still overlap.'},
        'recycling_support':{'assigned_issuer_days':len({(r['date'],r['symbol']) for r in recycling}),
            'extra_filled_issuer_days':len({(r['date'],r['symbol']) for r in recycling if r['extra_filled']>0})},
        'files':{p.name:digest(p) for p in case.iterdir() if p.is_file()}}


def shadow_comparison(case, reference):
    actual=read(case/'trades.json');shadow=read(reference/'trades.json')
    sid=read(reference/'result.json')['standalone']
    a={key(t):t for t in actual if t['strategy_id']==sid};b={key(t):t for t in shadow}
    if len(a)!=sum(t['strategy_id']==sid for t in actual) or len(b)!=len(shadow):
        raise ValueError('Ambiguous economic entry identity in shadow comparison')
    missing=[t for k,t in b.items() if k not in a];added=[t for k,t in a.items() if k not in b]
    submissions=list(rows(case/'allocation_events.jsonl.gz'))
    fills=read(case/'fills.json')
    admitted=defaultdict(list)
    for r in rows(case/'native_budget_admissions.jsonl.gz'):admitted[r['order']['order_id']].append(r)
    classified=[]
    for t in missing:
        attempts=[e for e in submissions if (e['strategy_id'],e['symbol'],e['timestamp'])==key(t)]
        decision=datetime.fromisoformat(t['entry_decision_time'])
        held=sum((1 if fill['side']=='BUY' else -1)*fill['qty'] for fill in fills
                 if fill['strategy_id']==sid and fill['symbol']==t['symbol']
                 and datetime.fromisoformat(fill['timestamp'])<=decision)
        evidence=[r for e in attempts for r in admitted[e['submitted_order_id']]]
        classification=('not_submitted_in_shared_native_path' if not attempts else
            'budget_zero_admission_observed' if any(r['approved_qty']==0 and r['offered_qty']>0 for r in evidence) else
            'physical_zero_admission_observed' if any(r['offered_qty']==0 for r in evidence) else
            'submitted_but_no_matching_economic_entry')
        classified.append({'key':key(t),'reference_pnl':t['net_pnl'],'reason':classification,
            'held_quantity_at_reference_decision':held,
            'existing_position_observed_without_new_submission':not attempts and held>0})
    return {'reference':str(reference),'strategy':sid,
        'shared':distribution(list(a.values())),'standalone':distribution(list(b.values())),
        'matched_entries':len(set(a)&set(b)),'missing_reference':distribution(missing),'new_shared':distribution(added),
        'missing_details':classified,
        'interpretation':'Matched decision timestamp/symbol, not hindsight ranking. Reference PNL is not recoverable incremental portfolio PNL: funding, sizing, exits, liquidity and later state differ.'}


def paired_bootstrap(case, reference, block=5, samples=5000):
    import numpy as np
    a=read(case/'daily.json');b=read(reference/'daily.json')
    if [d['day'] for d in a]!=[d['day'] for d in b]:raise ValueError('Unmatched session schedule')
    def returns(path, days):
        req=read(path/'request.json');initial=read(req['arguments']['study'])['initial_equity']
        nav=np.asarray([initial]+[r['equity'] for r in days]);return nav[1:]/nav[:-1]-1
    x,y=returns(case,a),returns(reference,b);n=len(x)
    rng=np.random.default_rng(20261009);starts=rng.integers(0,n-block+1,(samples,(n+block-1)//block))
    ix=(starts[:,:,None]+np.arange(block)).reshape(samples,-1)[:,:n]
    diff=100*(np.prod(1+x[ix],axis=1)-np.prod(1+y[ix],axis=1))
    return {'reference':str(reference),'paired_5day_bootstrap_return_difference_pp_95pct':list(np.quantile(diff,[.025,.975])),
        'samples':samples,'seed':20261009,'interpretation':'Conditional development resampling only. Does not correct selection bias or establish independent expected return.'}


def analyze(root, controls):
    f=read(root/'finalist.json');name=f['winner']
    if not name:raise ValueError('No selected candidate')
    k=f['recipe']['native_budget']['kalcb_sizing_fraction'];o=f['recipe']['native_budget']['olr_fraction']
    recycle=int(f['recipe']['olr_capacity'].get('capacity_redistribute',False))
    references=[f'control_k{int(k*100):03}', f'control_o{int(o*100):03}_r{recycle}']
    result={'selected':name,'source_generation_hash':f['source_generation_hash'],
        'scope':'reused_development','scenarios':{},'control_root':str(controls)}
    for scenario in SCENARIOS:
        case=root/'qualified'/(name+'_'+scenario)
        if not case.exists():case=root/'cases'/(name+'_'+scenario)
        item=describe(case)
        item['matched_sleeve_comparisons']=[shadow_comparison(case,controls/'cases'/(r+'_'+scenario)) for r in references]
        item['bootstrap_vs_native_olr']=paired_bootstrap(case,controls/'cases'/('control_o100_r1_'+scenario))
        result['scenarios'][scenario]=item
    write_json(root/'attribution.json',result)
    print(json.dumps({'written':str(root/'attribution.json'),'selected':name}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--controls',type=Path,required=True)
    args=p.parse_args();analyze(args.root.resolve(),args.controls.resolve())
