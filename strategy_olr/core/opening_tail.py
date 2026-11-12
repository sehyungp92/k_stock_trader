"""Strictly lagged opening-loss scenarios, independent of candidate routing.

The adapter supplies sourced observations and an explicit observation clock.
This module never reads files, future bars, trade outcomes or account P&L.
"""
from datetime import date, datetime
from math import ceil, isfinite
from statistics import mean

KEY = 'opening_tail_history_v1'
SESSIONS = 60
QUANTILE = .90
PRIOR_STRENGTH = 20


def empty():
    return {'version': 1, 'through': '', 'sessions': [], 'observations': []}


def advance(history, rows, through, session_days):
    """Advance only after a completed session; retain a 60-session panel.

    Rows are dated outcomes with previous-session closes and next eligible
    morning opens. They become usable for sizing on the FOLLOWING session.
    """
    target = date.fromisoformat(str(through))
    if history.get('version') != 1 or history['through'] and str(through) < history['through']:
        raise ValueError('Opening-tail clock cannot move backwards')
    days = sorted(set(history['sessions']) | {str(d) for d in session_days if str(d) <= str(through)})[-SESSIONS:]
    if not days or days[-1] != str(through):
        raise ValueError('Opening-tail clock needs a declared completed session')
    previous = {(r['day'], r['symbol']): r for r in history['observations']}
    for row in rows:
        if date.fromisoformat(row['day']) > target or row['previous_day'] >= row['day']:
            raise ValueError('Opening-tail observation contains future or reversed dates')
        if row['day'] not in session_days:
            raise ValueError('Opening-tail observation is outside declared sessions')
        available=datetime.fromisoformat(row['available_at'])
        if available.tzinfo is None or available.date().isoformat() != row['day']:
            raise ValueError('Opening-tail availability is on the wrong date')
        for field in ('previous_close','eligible_open'):
            if not isfinite(row[field]) or row[field] <= 0:
                raise ValueError('Opening-tail prices must be finite and positive')
        loss=max(0.,1-row['eligible_open']/row['previous_close'])
        if not isfinite(row['loss_fraction']) or abs(loss-row['loss_fraction'])>1e-12:
            raise ValueError('Opening-tail loss does not reconcile to prices')
        key=row['day'],row['symbol']
        if key in previous and previous[key] != row:
            raise ValueError('Opening-tail observation changed after publication')
        previous[key]=dict(row)
    return {'version':1,'through':str(through),'sessions':days,
            'observations':[r for key,r in sorted(previous.items()) if key[0] in days]}


def tail_mean(values):
    if not values: return 0.
    values=sorted(values,reverse=True)
    return mean(values[:max(1,ceil(len(values)*(1-QUANTILE)-1e-10))])


def evidence(state, symbol, sector, at):
    if at.tzinfo is None:
        raise ValueError('Opening-tail decision clock must be aware')
    history=state.meta.get(KEY)
    if not history or not history['through'] or history['through'] >= at.date().isoformat():
        raise ValueError('Opening-tail sizing requires completed strictly prior-session history')
    rows=history['observations']
    if any(r['day']>=at.date().isoformat() for r in rows):
        raise ValueError('Opening-tail history leaks the current or future session')
    own=[r['loss_fraction'] for r in rows if r['symbol']==symbol]
    # Do not count the same symbol twice in the shrinkage prior.
    peers=[r['loss_fraction'] for r in rows if r['sector']==sector and r['symbol']!=symbol]
    pool='sector_peers'
    if len(peers)<PRIOR_STRENGTH:
        peers=[r['loss_fraction'] for r in rows if r['symbol']!=symbol];pool='declared_universe_peers'
    if len(peers)<PRIOR_STRENGTH:
        raise ValueError('Opening-tail pooled history is insufficient')
    weight=len(own)/(len(own)+PRIOR_STRENGTH)
    own_es=tail_mean(own);peer_es=tail_mean(peers)
    fraction=weight*own_es+(1-weight)*peer_es
    return {'version':'opening-tail-es90-shrink20-v1','fraction':fraction,
            'symbol_observations':len(own),'peer_observations':len(peers),'pool':pool,
            'symbol_tail_mean':own_es,'peer_tail_mean':peer_es,'symbol_weight':weight,
            'through':history['through'],'first_session':history['sessions'][0],
            'sessions':len(history['sessions']),'quantile':QUANTILE,
            'interpretation':'Sizing scenario, not a guaranteed stop fill or realized-loss bound.'}
