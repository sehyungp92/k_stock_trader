"""Native response evidence with explicit cash outcomes and label maturity.

Portfolio decisions and an opportunity's alpha label are different estimands:
a supported expired no-fill order has a known zero cash outcome, but no
observed acquired-position return. Unresolved inventory is neither one.
"""
from collections import defaultdict
from datetime import datetime, timedelta
from math import isclose, isfinite
from statistics import mean


def dt(value):
    result=datetime.fromisoformat(value) if isinstance(value,str) else value
    if result.tzinfo is None:raise ValueError('An aware source clock is required')
    return result


def reconstruct(result, journal):
    cash=0.;holdings={};marks={};peak=0.;dd=0.;costs=0.;acquired=0.;minimum_cash=100_000_000.
    points=[];last_stamp=None;buy_quantity=0;fill_ids=set()
    for sequence,event in enumerate(journal):
        kind=event['event']
        if kind in ('opening_marks','closing_marks'):
            stamp=dt(event['timestamp']);marks.update(event['prices'])
            # Closing marks are known after the completed interval; ordered
            # events retain the original label/phase separately for exact DD.
            known=stamp+timedelta(minutes=1) if kind=='closing_marks' else stamp
        elif kind=='fill':
            fill=event['fill'];stamp=dt(fill['timestamp']);known=stamp
            identity=fill['metadata']['execution_id']
            if identity in fill_ids:raise ValueError('Duplicate native execution')
            fill_ids.add(identity)
            q,p,fee=fill['qty'],fill['price'],fill['metadata']['fee']
            if type(q) is not int or q<=0 or any(not isfinite(v) for v in (p,fee)) or p<=0 or fee<0:raise ValueError('Invalid fill arithmetic')
            if fill['side'] not in ('BUY','SELL'):raise ValueError('Invalid native fill side')
            sign=1 if fill['side']=='BUY' else -1
            expected_before=100_000_000+cash
            if not isclose(expected_before,event['cash_before'],abs_tol=1e-5):raise ValueError('Cash-before mismatch')
            cash-=sign*q*p+fee;costs+=fee
            if not isclose(100_000_000+cash,event['cash_after'],abs_tol=1e-5):raise ValueError('Cash-after mismatch')
            if sign==1:acquired+=q*p;buy_quantity+=q
            holdings[fill['symbol']]=holdings.get(fill['symbol'],0)+sign*q
            if holdings[fill['symbol']]<0:raise ValueError('Unfunded short inventory')
            minimum_cash=min(minimum_cash,100_000_000+cash)
        else:continue
        if last_stamp is not None and stamp<last_stamp:raise ValueError('Reversed native event clock')
        last_stamp=stamp
        if any(q and symbol not in marks for symbol,q in holdings.items()):raise ValueError('Missing mark on held inventory')
        pnl=cash+sum(q*marks.get(symbol,0) for symbol,q in holdings.items())
        if not isfinite(pnl):raise ValueError('Invalid portfolio mark')
        peak=max(peak,pnl);dd=max(dd,peak-pnl)
        point=[stamp.isoformat(),sequence,kind,pnl,cash,sum(holdings.values()),known.isoformat()]
        # Preserve every change including same-time opening/fill/closing. The
        # final point is retained below even when the value has not changed.
        if not points or point[3:6]!=points[-1][3:6]:points.append(point)
        terminal=point
    if not points:raise ValueError('No sourced native observations')
    if points[-1]!=terminal:points.append(terminal)
    if minimum_cash < -1e-5:raise ValueError('Negative native cash')
    if sum(holdings.values())!=result['residual_quantity'] or buy_quantity!=result['filled_quantity']:raise ValueError('Inventory mismatch')
    for computed,recorded in [(dd,result['fixed_target_risk']['max_drawdown_krw']),(acquired,result['filled_notional']),(costs,result['total_fees'])]:
        if not isclose(computed,recorded,rel_tol=1e-10,abs_tol=1e-5):raise ValueError('Native response arithmetic mismatch')
    status=result['status']
    if status not in ('matured','unfilled','right_censored','unsupported_source','invalid_execution'):
        raise ValueError('Unknown native response disposition')
    if not result.get('execution_arithmetic_pass',True) and status!='invalid_execution':
        raise ValueError('Failed native execution cannot be relabelled trainable')
    if status=='invalid_execution' and (result['net_pnl'] is not None or result['net_bps'] is not None):
        raise ValueError('Invalid execution cannot supply a payable outcome')
    closed=status in ('matured','unfilled') and not result['pending_orders'] and not result['residual_quantity']
    cash_outcome=cash if closed else None
    if status=='matured' and (not closed or not isclose(cash,result['net_pnl'],rel_tol=1e-10,abs_tol=1e-5)):raise ValueError('Matured cash payoff mismatch')
    if status=='unfilled' and (not closed or buy_quantity or abs(cash)>1e-8):raise ValueError('Unfilled does not mean zero when orders/costs remain')
    last_day=max(s.split(':')[1] for s in result['source_sessions'])
    # Training consumes labels only after the entire observed final session.
    mature_at=last_day+'T23:59:59+09:00' if closed else None
    return dict(status=status,policy_cash_pnl=cash_outcome,alpha_net_bps=result['net_bps'],known_at=mature_at,
                actual_acquired=acquired,fees=costs,full_event_dd_krw=dd,minimum_cash=minimum_cash,
                residual_quantity=result['residual_quantity'],pending_orders=result['pending_orders'],
                path_columns=['label','event_sequence','phase','net_pnl','cash_delta','open_quantity','known_at'],path=points)


def available_records(records,at):
    at=dt(at)
    return tuple(r for r in records if r['known_at'] is not None and dt(r['known_at'])<at)


def paired_cells(records,blocks,budgets,scenarios):
    """Date-equal *request* outcomes and marginal quantities; never account PNL."""
    keyed={(r['day'],r['symbol'],r['budget'],r['scenario']):r for r in records}
    if len(keyed)!=len(records):raise ValueError('Duplicate response key')
    opportunities=sorted({(r['day'],r['symbol']) for r in records})
    output=[]
    for start,end in blocks:
        requested=[k for k in opportunities if start<=k[0]<=end];paired=[]
        for day,symbol in requested:
            rows=[keyed.get((day,symbol,b,s)) for b in budgets for s in scenarios]
            if all(r is not None and r['known_at'] is not None and r['known_at'][:10]<=end for r in rows):paired.append((day,symbol))
        def avg(values):
            by_day=defaultdict(list)
            for day,value in values:by_day[day].append(value)
            return mean(mean(v) for v in by_day.values()) if by_day else None
        cells=[]
        for s in scenarios:
            for b in budgets:
                rows=[keyed[(d,sym,b,s)] for d,sym in paired]
                cells.append(dict(scenario=s,budget=b,n=len(rows),date_mean_net_pnl=avg((r['day'],r['policy_cash_pnl']) for r in rows),
                    date_mean_acquired=avg((r['day'],r['actual_acquired']) for r in rows),
                    date_mean_dollar_dd=avg((r['day'],r['full_event_dd_krw']) for r in rows),
                    observed_unfilled=sum(r['status']=='unfilled' for r in rows)))
        output.append(dict(start=start,end=end,opportunities=len(requested),paired=len(paired),dates=len({d for d,_ in paired}),
                           coverage=len(paired)/len(requested) if requested else 0,cells=cells))
    return output
