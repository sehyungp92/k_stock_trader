"""Exp0031 saved-evidence population and source preflight; no new targets."""
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb.fixed_allocation_reference_v1 import verify_identity_bridge

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_opening_marginal_20261016')
OLD=Path('E:/CodexResearch/k_stock_trader/opening_account_20261013')
FRONTIER=Path('E:/CodexResearch/k_stock_trader/kalcb_frontier_completion_20261016/v3')


def main():
    if (ROOT/'preflight_v1.json').exists():raise ValueError('Preserve prior preflight')
    profile_path=OLD/'study_v2/runtime/control/is/profile.json';profile=read(profile_path)
    verify([profile[k] for k in ('manifest','plan','policy')])
    index=read(FRONTIER/'account_index.json');rows={};inputs=[bound(profile_path),bound(FRONTIER/'account_index.json')]
    for scenario in ('baseline','combined'):
        folder=OLD/'study_v2/cases/control/is'/scenario
        validation=validate_native(folder,profile,scenario,save=False)
        bridge=verify_identity_bridge(folder,index['compound_prior100/is/'+scenario])
        orders=[]
        with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                event=json.loads(line);order=event.get('order',{})
                if event['event']=='submit' and order.get('side')=='BUY' and order.get('metadata',{}).get('entry_route')=='first30_open_anchor':
                    orders.append(order)
        units={};clock=Counter()
        for order in orders:
            meta=order['metadata'];available=meta['decision_available_at'];day=available[:10]
            key=day+'/'+order['symbol'];clock[available[11:16]]+=1
            if key in units:raise ValueError('Repeated first30 submission needs explicit lifetime semantics')
            units[key]=dict(day=day,symbol=order['symbol'],available_at=available,qty=order['qty'],order_id=order['order_id'],
                score=meta.get('frontier_selection_score'),reference_price=meta.get('entry_price_ref'))
        if set(clock)!={'09:30'}:raise ValueError('Nonstandard source decision clock requires a revised observer')
        days=sorted({row['day'] for row in units.values()});counts=Counter(row['day'] for row in units.values())
        trades=read(folder/'trades.json');payoffs=defaultdict(float)
        for trade in trades:
            if trade['route_metadata'].get('entry_route')=='first30_open_anchor':
                payoffs[trade['entry_decision_time'][:10]+'/'+trade['symbol']]+=trade['net_pnl']
        for key,value in payoffs.items():
            if key not in units:raise ValueError('Opening funded trade missing from emitted population')
        rows[scenario]=dict(folder=str(folder),validation=validation,bridge=bridge,emitted_opportunities=len(units),
            distinct_sessions=len(days),maximum_session_intents=max(counts.values()),availability_clocks=dict(clock),
            monthly_sessions=dict(Counter(day[:7] for day in days)),funded_opportunities=len(payoffs),
            profitable_funded=sum(x>0 for x in payoffs.values()),losing_funded=sum(x<0 for x in payoffs.values()),
            first30_net_pnl_krw=sum(payoffs.values()),funded_positive_pnl_krw=sum(x for x in payoffs.values() if x>0),
            funded_negative_pnl_krw=sum(x for x in payoffs.values() if x<0),units=units)
        inputs.extend(bound(folder/f) for f in ('request.json','result.json','execution_audit.json','completion.json',
            'orders.jsonl.gz','fills.json','trades.json','daily.json','equity.json','opening_decisions.json','independent_account_verification.json'))
    code=Path(profile['manifest']['path']).parent/'files'
    helpers=('strategy_kalcb/core/logic.py','strategy_kalcb/core/account_allocation_v1.py',
        'backtests/strategies/kalcb/account_decision_v1.py','backtests/strategies/kalcb/account_history_v1.py',
        'backtests/strategies/kalcb/pending_entry_account_pilot_v1.py','backtests/strategies/kalcb/mechanism_observer_v2.py')
    inputs.extend(bound(code/p) for p in helpers)
    verify(inputs)
    write(ROOT/'preflight_v1.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0031',
        program=bound(__file__),design=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0031.md'),
        parent_profile=profile,profile_binding=bound(profile_path),cases=rows,bindings=inputs,
        scope='Saved source-qualified control population only; no new branch target or fitted decision.',
        observer_clock='09:25 completed bar, available09:30. Every original first30 submission in both full156-session controls has that clock. Nonstandard sessions without such original intents have no omission; do not invent shifted candidate information.'))
    print(json.dumps({key:{k:v for k,v in row.items() if k not in ('units','validation','bridge')} for key,row in rows.items()},indent=2))


if __name__=='__main__':main()
