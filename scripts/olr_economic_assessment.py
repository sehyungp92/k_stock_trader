"""Authenticate funded cases and attribute the completed bounded study.

Accounting contrasts describe whole-policy replays. They are not single-action
causal estimates, new forecasts, or a license to qualify invalid execution.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.olr.economic_evidence import completed_case,compare,opportunity_key,read,sha


def paths(root):
    return dict(baseline=root/'cases/train/baseline_baseline',
        legacy_defective=root/'cases/train/combined_baseline_attempt1',
        **{m:root/'acquisition_contract_v3/cases/train'/f'{m}_baseline'
           for m in ('lifecycle','allocation','management','combined')})


def keyed(trades):
    result={opportunity_key(t):t for t in trades}
    if len(result)!=len(trades):raise ValueError('Ambiguous opportunity lineage')
    return result


def contrast(base,treatment):
    a,b=keyed(base),keyed(treatment);shared=sorted(a.keys()&b.keys())
    added=sorted(b.keys()-a.keys());removed=sorted(a.keys()-b.keys())
    quantity=sum((b[k]['qty']-a[k]['qty'])*a[k]['net_pnl']/a[k]['qty'] for k in shared)
    unit=sum(b[k]['qty']*(b[k]['net_pnl']/b[k]['qty']-a[k]['net_pnl']/a[k]['qty']) for k in shared)
    new=sum(b[k]['net_pnl'] for k in added);lost=-sum(a[k]['net_pnl'] for k in removed)
    total=sum(t['net_pnl'] for t in treatment)-sum(t['net_pnl'] for t in base)
    assert math.isclose(quantity+unit+new+lost,total,abs_tol=1e-5)
    top=sorted(a,key=lambda k:a[k]['net_pnl'],reverse=True)[:10]
    original=sum(a[k]['net_pnl'] for k in top)
    capture=sum(b[k]['net_pnl'] for k in top if k in b)
    details=[]
    for k in sorted(a.keys()|b.keys()):
        old=a.get(k);t=b.get(k);delta=(t['net_pnl'] if t else 0)-(old['net_pnl'] if old else 0)
        details.append(dict(symbol=k[0],day=k[1],delta_pnl=delta,
            baseline_qty=old['qty'] if old else 0,treatment_qty=t['qty'] if t else 0,
            baseline_pnl=old['net_pnl'] if old else 0,treatment_pnl=t['net_pnl'] if t else 0,
            baseline_exit=old['exit_reason'] if old else None,treatment_exit=t['exit_reason'] if t else None))
    return dict(shared=len(shared),added=len(added),removed=len(removed),
        quantity_changes=sum(a[k]['qty']!=b[k]['qty'] for k in shared),
        entry_clock_changes=sum(a[k]['entry_decision_time']!=b[k]['entry_decision_time'] for k in shared),
        additive_pnl_accounting=dict(matched_quantity=quantity,matched_net_unit_payoff=unit,
            added_opportunities=new,removed_opportunities=lost,total=total),
        top10_baseline_winners=dict(baseline_pnl=original,treatment_same_opportunities_pnl=capture,
            signed_capture_ratio=capture/original,missing=sum(k not in b for k in top)),
        added_notional=[b[k]['qty']*b[k]['entry_price'] for k in added],
        added_net_pnl=new,removed_baseline_net_pnl=-lost,
        worst_contributors=sorted(details,key=lambda x:x['delta_pnl'])[:12],
        best_contributors=sorted(details,key=lambda x:x['delta_pnl'],reverse=True)[:8],
        per_opportunity=details)


def drawdown_book(path,expected):
    """Reconcile full marked events only when the saved journal includes them."""
    with gzip.open(path/'minute_events.jsonl.gz','rt',encoding='utf-8') as f:
        event_counts=Counter(json.loads(line)['event'] for line in f)
    if not event_counts['opening_marks'] or not event_counts['closing_marks']:
        # Legacy exporter retains orders/fills but drops marks. Never invent
        # a full DD attribution from exit PNL or neighboring fill marks.
        return dict(full_dd_pct=expected['all_event_drawdown_pct'],**expected['maximum'],
            independent_symbol_attribution_available=False,
            limitation='Saved minute journal omits opening/closing marks; source-bound native full-event DD is available but per-symbol peak/trough attribution is not reconstructed.',
            saved_event_counts=dict(event_counts))
    cash=100_000_000.;cash_by=defaultdict(float);q=defaultdict(int);marks={}
    peak=cash;peak_at=None;peak_book={};worst=0.;out=None
    def book():return {s:cash_by[s]+q[s]*marks.get(s,0) for s in cash_by.keys()|q.keys()}
    with gzip.open(path/'minute_events.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);kind=row['event']
            if kind in ('opening_marks','closing_marks'):
                marks.update(row['prices']);stamp=row['timestamp']
            elif kind=='fill':
                fill=row['fill'];s=fill['symbol'];sign=1 if fill['side']=='BUY' else -1
                delta=-sign*fill['qty']*fill['price']-fill['metadata']['fee']
                cash+=delta;cash_by[s]+=delta;q[s]+=sign*fill['qty'];stamp=fill['timestamp']
                assert q[s]>=0
            else:continue
            nav=cash+sum(n*marks[s] for s,n in q.items() if n)
            if nav>peak:peak=nav;peak_at=stamp;peak_book=book()
            dd=100*(peak-nav)/peak
            if dd>worst:
                worst=dd;now=book()
                changes={s:now.get(s,0)-peak_book.get(s,0) for s in now.keys()|peak_book.keys()}
                assert math.isclose(sum(changes.values()),nav-peak,abs_tol=1e-5)
                out=dict(peak_at=peak_at,trough_at=stamp,peak_equity=peak,trough_equity=nav,
                    dollar_decline=peak-nav,by_symbol=sorted(changes.items(),key=lambda x:x[1]))
    assert math.isclose(worst,expected['all_event_drawdown_pct'],abs_tol=1e-8)
    assert math.isclose(nav,expected['final_equity'],abs_tol=1e-5)
    return dict(full_dd_pct=worst,independent_symbol_attribution_available=True,**out)


def trace_stats(path):
    if not (path/'economic_policy_trace.json').exists():return {}
    trace=read(path/'economic_policy_trace.json');fits=read(path/'economic_policy_fits.json')
    for fit in fits:
        assert datetime.fromisoformat(fit['label_max_known_at']).date().isoformat()<fit['day']
    entries={r['at']:r for r in trace if r['kind']=='entry'}
    allocations=[r for r in trace if r['kind']=='allocation_result']
    reasons=Counter(r['outcome']['reason'] for r in allocations)
    zero=[];recovery=[];changes=Counter()
    for r in allocations:
        selected=r['selected_quantities'];native=r['native_quantities']
        changes.update('increase' if selected.get(k,0)>v else 'decrease' if selected.get(k,0)<v else 'same'
                       for k,v in native.items())
        if sum(selected.values())==0:zero.append(r['at'])
        if r['outcome']['reason']=='bounded_risk_recovery':recovery.append(r['at'])
    management=[r for r in trace if r['kind']=='management'];retire=[r for r in management if r['retire']]
    supported={'09:05','09:30','10:00'}
    out=dict(fit_count=len(fits),fit_maturity_pass=True,allocation_cohorts=len(allocations),
        allocation_reasons=dict(reasons),quantity_comparisons=dict(changes),
        zero_cohorts=len(zero),zero_first=min(zero) if zero else None,zero_last=max(zero) if zero else None,
        recovery_cohorts=len(recovery),recovery_first=min(recovery) if recovery else None,
        recovery_last=max(recovery) if recovery else None,
        observed_entry_max_dd=max([e['current_drawdown'] for e in entries.values()],default=0),
        management_quotes=len(management),retire_requests=len(retire),
        quotes_outside_label_clocks=sum(r['at'][11:16] not in supported for r in management),
        retire_outside_label_clocks=sum(r['at'][11:16] not in supported for r in retire),
        retire_hour_counts=dict(Counter(r['at'][11:13] for r in retire)))
    by_month=defaultdict(Counter)
    for r in allocations:
        by_month[r['at'][:7]][r['outcome']['reason']]+=1
        if sum(r['selected_quantities'].values())==0:by_month[r['at'][:7]]['all_zero']+=1
    out['allocation_by_month']={k:dict(v) for k,v in by_month.items()}
    return out


def assess(root):
    cases=paths(root);results={};trades={}
    for mode,p in cases.items():
        verified=completed_case(p);ts=read(p/'trades.json');trades[mode]=ts
        assert math.isclose(sum(t['net_pnl'] for t in ts),verified['return_pct']*1_000_000,abs_tol=.01)
        nav=read(p/'minute_nav_evidence.json');audit=read(p/'execution_audit.json')
        notional=sum(t['qty']*t['entry_price'] for t in ts)
        results[mode]=dict(evidence={k:v for k,v in verified.items() if k!='acquired'},
            acquisition_notional=notional,median_entry_notional=median(t['qty']*t['entry_price'] for t in ts),
            entry_count_at_least_1m=sum(t['qty']*t['entry_price']>=1_000_000 for t in ts),
            cash_fees=sum(t['commission'] for t in ts),
            acquisition_weighted_calendar_hold_hours=sum(t['qty']*t['entry_price']*
                (datetime.fromisoformat(t['exit_fill_time'])-datetime.fromisoformat(t['entry_fill_time'])).total_seconds()/3600 for t in ts)/notional,
            exit_reason_counts=dict(Counter(t['exit_reason'] for t in ts)),
            drawdown_book=drawdown_book(p,nav),policy_trace=trace_stats(p),
            impossible_volume_events=audit['capacity_model_audit'].get('impossible_volume_events',[]))
        if mode not in ('baseline','legacy_defective'):
            results[mode]['gates']=compare(completed_case(cases['baseline']),verified)
    comparisons={}
    for mode in ('lifecycle','allocation','management','combined','legacy_defective'):
        comparisons[mode+'_vs_baseline']=contrast(trades['baseline'],trades[mode])
    for mode in ('allocation','management','combined'):
        comparisons[mode+'_vs_lifecycle']=contrast(trades['lifecycle'],trades[mode])
    comparisons['combined_vs_allocation']=contrast(trades['allocation'],trades['combined'])
    assert not any(results[m]['gates']['full_objective_cell_pass'] for m in ('lifecycle','allocation','management','combined'))
    return dict(cases=results,contrasts=comparisons,assessment_complete=True,
        funded_successor_qualified=False,objective_achieved=False,
        new_oos_replays=0,new_stress_or_capped_replays=0,
        advancement='All corrected nominal candidates fail required gates; no stress/capped or OOS promotion stage is justified.',
        interpretation='Valid full-policy replay contrasts are funded portfolio evidence. Per-opportunity quantity/unit decompositions are accounting, not individual causal actions. Management-only metrics are invalid execution diagnostics.',
        assessor_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=assess(args.root)
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({m:{'valid':x['evidence']['valid'],'acquired':x['acquisition_notional'],
        'trace':x['policy_trace']} for m,x in result['cases'].items()},indent=2))
