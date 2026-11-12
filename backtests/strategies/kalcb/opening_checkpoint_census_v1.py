"""Fixed minute45 information census, never an exit policy or action label."""
from datetime import datetime, timedelta


def features(day, prefix, held_qty, gross_cash):
    start=datetime.fromisoformat(day+'T09:00:00+09:00')
    cutoff=start+timedelta(minutes=45)
    visible=[b for b in prefix if datetime.fromisoformat(b['timestamp'])+timedelta(minutes=5)<=cutoff]
    expected=[(start+timedelta(minutes=5*i)).isoformat() for i in range(9)]
    if [datetime.fromisoformat(b['timestamp']).isoformat() for b in visible]!=expected:
        raise ValueError('Incomplete/duplicated/nonconsecutive causal nine-bar prefix')
    if held_qty<=0:
        raise ValueError('No held position')
    for b in visible:
        if not(b['volume']>0 and b['low']>0 and b['low']<=min(b['open'],b['close'])<=max(b['open'],b['close'])<=b['high']):
            raise ValueError('Invalid/zero-volume source bar')
    vwap=sum((b['high']+b['low']+b['close'])/3*b['volume'] for b in visible)/sum(b['volume'] for b in visible)
    price=visible[-1]['close'];gross=gross_cash+held_qty*price
    negative=gross<0;below=price<vwap
    return {'available_at':cutoff.isoformat(),'consumed_bars':9,'completed_close':price,'session_vwap':vwap,
            'gross_marked_pnl_krw':gross,'gross_negative':negative,'below_own_vwap':below,
            'state':('negative' if negative else 'nonnegative')+'/'+('below_vwap' if below else 'at_or_above_vwap')}


def inventory_before(fills, cutoff):
    clock=datetime.fromisoformat(cutoff);qty=0;cash=0.;fees=0.
    for f in fills:
        if datetime.fromisoformat(f['timestamp'])>=clock:continue
        sign=1 if f['side']=='BUY' else -1
        qty+=sign*f['qty'];cash-=sign*f['qty']*f['price'];fees+=f['metadata']['fee']
    return qty,cash,fees


def summarize(rows):
    # Groups are accounting descriptions, never pooled independent replications.
    return {'positions':len(rows),'issuer_days':len({(r['symbol'],r['day']) for r in rows}),
            'sessions':len({r['day'] for r in rows}),'issuers':len({r['symbol'] for r in rows}),
            'terminal_profitable_positions':sum(r['terminal_net_krw']>0 for r in rows),
            'negative_actual_continuation_positions':sum(r['actual_remaining_net_krw']<0 for r in rows),
            'actual_remaining_net_krw':sum(r['actual_remaining_net_krw'] for r in rows),
            'actual_remaining_positive_krw':sum(max(0,r['actual_remaining_net_krw']) for r in rows),
            'actual_remaining_negative_krw':sum(min(0,r['actual_remaining_net_krw']) for r in rows),
            'terminal_net_krw':sum(r['terminal_net_krw'] for r in rows),
            'future_entry_fill_positions':sum(r['entry_fills_at_or_after_checkpoint']>0 for r in rows),
            'largest_positive_continuation_krw':max([r['actual_remaining_net_krw'] for r in rows],default=None),
            'scope':'Observed remaining physical cashflows/marks/fees on unchanged path, not executable exit savings or marginal alpha.'}
