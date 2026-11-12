"""Lagged, quantity-dependent adverse liquidation scenarios. No future I/O."""
from datetime import date, datetime, timedelta
from math import ceil, floor, isfinite
from statistics import mean
from .opening_tail import tail_mean

KEY='liquidation_tail_history_v3'
SESSIONS=60
PRIOR=20


def empty():return {'version':3,'through':'','sessions':[],'observations':[]}


def advance(history, rows, through, session_days):
    date.fromisoformat(through)
    if history.get('version')!=3 or history['through'] and through<history['through']:
        raise ValueError('Liquidation history clock moved backwards')
    days=sorted(set(history['sessions'])|{str(x) for x in session_days if str(x)<=through})[-SESSIONS:]
    if not days or days[-1]!=through:raise ValueError('Missing completed session clock')
    previous={(r['day'],r['symbol']):r for r in history['observations']}
    for row in rows:
        if row['day']>through or row['previous_day']>=row['day'] or row['day'] not in session_days:
            raise ValueError('Future or reversed liquidation history')
        ref=datetime.fromisoformat(row['reference_at']);available=datetime.fromisoformat(row['available_at'])
        if ref.tzinfo is None or available.tzinfo is None or ref.date().isoformat()!=row['previous_day'] or available.date().isoformat()!=row['day']:
            raise ValueError('Invalid availability clock')
        if not isfinite(row['reference_price']) or row['reference_price']<=0:raise ValueError('Invalid reference price')
        last=None
        for at, price, vol, capacity_bar in row['liquidation']:
            ts=datetime.fromisoformat(at);prev=datetime.fromisoformat(capacity_bar)
            if (ts.tzinfo is None or prev.tzinfo is None or ts.date().isoformat()!=row['day'] or prev.date()!=ts.date()
                    or prev+timedelta(minutes=5)!=ts or ts>=available or last is not None and ts<=last):
                raise ValueError('Liquidation path has unsupported observation time')
            if not isfinite(price) or price<=0 or not isfinite(vol) or vol<0:raise ValueError('Invalid liquidation prices/capacity')
            last=ts
        key=row['day'],row['symbol']
        if key in previous and previous[key]!=row:raise ValueError('Published liquidation history changed')
        previous[key]=row
    return {'version':3,'through':through,'sessions':days,
            'observations':[r for key,r in sorted(previous.items()) if key[0] in days]}


def scenario(row, notional, participation):
    """Adverse average-price envelope monotone in the requested share count.

    No assumed closing-auction liquidity. Shares not liquidated by the final
    eligible continuous open are impaired to zero in this sizing scenario.
    Costs are added separately by the unchanged loss-budget module.
    """
    if not all(isfinite(x) and x>0 for x in (notional,participation)) or participation>1:
        raise ValueError('Invalid scenario notional or participation')
    ref=row['reference_price'];q=max(1,ceil(notional/ref));remaining=q;proceeds=0.;filled=0;worst=0.;end=None
    for at,price,volume,_ in row['liquidation']:
        qty=min(remaining,floor(participation*volume))
        if not qty:continue
        remaining-=qty;filled+=qty;proceeds+=qty*price;end=at
        worst=max(worst,1-proceeds/(filled*ref))
        if not remaining:break
    if remaining:worst=max(worst,1-proceeds/(q*ref))
    return {'fraction':max(0.,worst),'requested_shares':q,'unfilled_shares':remaining,'last_fill':end}


def evidence(state,symbol,sector,at,notional,participation):
    h=state.meta.get(KEY)
    if at.tzinfo is None or not h or not h['through'] or h['through']>=at.date().isoformat():
        raise ValueError('Liquidation sizing needs strictly prior completed-session history')
    if any(r['day']>=at.date().isoformat() for r in h['observations']):raise ValueError('Future liquidation row')
    own=[r for r in h['observations'] if r['symbol']==symbol]
    peers=[r for r in h['observations'] if r['sector']==sector and r['symbol']!=symbol];pool='sector_peers'
    if len(peers)<PRIOR:
        peers=[r for r in h['observations'] if r['symbol']!=symbol];pool='declared_universe_peers'
    if len(peers)<PRIOR:raise ValueError('Insufficient liquidation peer history')
    oq=[scenario(r,notional,participation) for r in own];pq=[scenario(r,notional,participation) for r in peers]
    weight=len(own)/(len(own)+PRIOR);a=tail_mean([q['fraction'] for q in oq]);z=tail_mean([q['fraction'] for q in pq])
    return {'version':'entry-liquidation-es90-shrink20-v3','fraction':weight*a+(1-weight)*z,
      'through':h['through'],'sessions':len(h['sessions']),'notional_upper':notional,'participation':participation,
      'symbol_observations':len(own),'peer_observations':len(peers),'pool':pool,'symbol_tail_mean':a,'peer_tail_mean':z,
      'symbol_weight':weight,'own_incomplete_scenarios':sum(q['unfilled_shares']>0 for q in oq),
      'peer_incomplete_scenarios':sum(q['unfilled_shares']>0 for q in pq),
      'aligned_own_loss_fractions':{r['day']:q['fraction'] for r,q in zip(own,oq)},
      'interpretation':'Conservative capacity/price scenario at upper requested notional, not a fill prediction or loss guarantee.'}
