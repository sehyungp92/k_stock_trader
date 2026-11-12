"""Causal, outcome-free minute45 context census. No policy or model.

Peers are the ORIGINAL opening opportunity population, not a broad-market
index. Both endpoint breadths use the same complete-by-minute45 peer set.
All values become available only at minute45; none are backdated to minute30.
"""
from datetime import datetime, timedelta
import math


CONTEXT_FEATURES = (
    'peer_market_vwap_breadth_change', 'peer_sector_vwap_breadth_change',
    'relative_post_entry_market_return_atr', 'relative_post_entry_sector_return_atr',
)
OWN_FEATURES = ('post_entry_return_atr', 'vwap_extension_atr',
                'post_entry_range_atr', 'post_entry_close_location',
                'post_entry_path_efficiency', 'post_entry_volume_ratio')


def prefix_statistics(row, bars):
    clock30 = datetime.fromisoformat(row['feature_available_at'])
    opening = clock30 - timedelta(minutes=30)
    clock45 = clock30 + timedelta(minutes=15)
    if opening.date().isoformat() != row['trade_date']:
        raise ValueError('Opening date/clock mismatch')
    if row['candidate']['sector'] != row['sector'] or row['candidate']['trade_date'] != row['trade_date']:
        raise ValueError('Original causal sector/date mismatch')
    visible = [b for b in bars if opening <= datetime.fromisoformat(b['timestamp'])
               and datetime.fromisoformat(b['timestamp']) + timedelta(minutes=5) <= clock45]
    stamps = [opening + timedelta(minutes=5*i) for i in range(9)]
    if [datetime.fromisoformat(b['timestamp']) for b in visible] != stamps:
        raise ValueError('Missing/duplicate/nonconsecutive completed prefix')
    for b in visible:
        if b['symbol'] != row['symbol'] or not all(math.isfinite(float(b[k])) for k in ('open','high','low','close','volume')):
            raise ValueError('Invalid raw symbol/value')
        if b['volume'] <= 0 or b['low'] <= 0 or not b['low'] <= min(b['open'],b['close']) <= max(b['open'],b['close']) <= b['high']:
            raise ValueError('Invalid raw OHLC/volume')
    original = row['opening_bars']
    if len(original) != 6:
        raise ValueError('Original opening contract is not six bars')
    for a,b in zip(original,visible[:6]):
        if datetime.fromisoformat(a['timestamp']) != datetime.fromisoformat(b['timestamp']) or any(a[k] != b[k] for k in ('symbol','open','high','low','close','volume')):
            raise ValueError('Original opening/raw source mismatch')
    atr = float(row['features']['daily_atr_pct'])
    if not math.isfinite(atr) or atr <= 0:
        raise ValueError('Invalid prior ATR')
    vwaps = []
    cash = volume = 0.
    for b in visible:
        cash += (b['high']+b['low']+b['close'])/3*b['volume']
        volume += b['volume']
        vwaps.append(cash/volume)
    c30,c45 = visible[5]['close'],visible[8]['close']
    post=visible[6:]
    hi,lo=max(b['high'] for b in post),min(b['low'] for b in post)
    path=[c30]+[b['close'] for b in post]
    travel=sum(abs(b-a) for a,b in zip(path,path[1:]))
    own={
        'post_entry_return_atr':(c45/c30-1)/atr,
        'vwap_extension_atr':(c45/vwaps[-1]-1)/atr,
        'post_entry_range_atr':(hi-lo)/c30/atr,
        'post_entry_close_location':(c45-lo)/(hi-lo) if hi>lo else .5,
        'post_entry_path_efficiency':(c45-c30)/travel if travel else 0.,
        'post_entry_volume_ratio':sum(b['volume'] for b in post)/(sum(b['volume'] for b in visible[:6])/2),
    }
    return {'symbol':row['symbol'],'day':row['trade_date'],'sector':row['sector'],
            'available_at':clock45.isoformat(),'opening_at':opening.isoformat(),
            'completed_close':c45,'daily_atr_pct':atr,'return_30_45':c45/c30-1,
            'above_vwap30':c30>vwaps[5],'above_vwap45':c45>vwaps[-1],
            'own_features':own,'consumed_bars':visible}


def context_rows(original, bars_by_symbol):
    if not original: return [],[]
    if len({r['trade_date'] for r in original}) != 1:
        raise ValueError('Mixed causal population dates')
    if len({r['symbol'] for r in original}) != len(original):
        raise ValueError('Duplicate original population member')
    valid,excluded={},[]
    for row in original:
        try: value=prefix_statistics(row,bars_by_symbol.get(row['symbol'],()))
        except ValueError as error:
            if any(token in str(error) for token in ('mismatch','Original opening contract')):
                raise
            excluded.append({'symbol':row['symbol'],'day':row['trade_date'],'reason':str(error)})
            continue
        valid[row['symbol']]=value
    result=[]
    for symbol,row in sorted(valid.items()):
        peers=[v for k,v in valid.items() if k!=symbol]
        sector=[v for v in peers if v['sector']==row['sector']]
        features=None
        if peers and sector:
            def breadth_change(values):
                return sum(int(v['above_vwap45'])-int(v['above_vwap30']) for v in values)/len(values)
            def relative(values):
                return (row['return_30_45']-sum(v['return_30_45'] for v in values)/len(values))/row['daily_atr_pct']
            features=dict(zip(CONTEXT_FEATURES,(breadth_change(peers),breadth_change(sector),relative(peers),relative(sector))))
        result.append({**row,'context_features':features,'context_supported':features is not None,
                       'market_peer_symbols':sorted(v['symbol'] for v in peers),
                       'sector_peer_symbols':sorted(v['symbol'] for v in sector),
                       'unsupported_reason':None if features is not None else 'no_same_sector_peer' if peers else 'no_market_peer'})
    return result,excluded


def chronological_support(rows,sessions,windows):
    result=[]
    for first,last in windows:
        allowed=set(sessions[:max(0,sessions.index(first)-1)])
        prior=[r for r in rows if r['day'] in allowed and r['context_supported']]
        days={r['day'] for r in prior}
        result.append({'valid_from':first,'valid_through':last,'prior_source_sessions':len(days),
                       'prior_issuer_days':len(prior),'observed_through':max(days,default=None),
                       'source_60_session_floor':len(days)>=60,
                       'mature_action_label_support_verified':False})
    return result
