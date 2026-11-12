"""Additive, independent minute90 raw-prefix/population proof.

This does not replace the mandatory original minute45 native source verifier.
The label producer invokes it on immutable raw database observations; a future
portfolio integration must retain the original source obligations as well.
"""
from collections import defaultdict
from datetime import date, datetime, timedelta
import math


def verify_rows(original, features, indexed, requested_days, snapshots=()):
    from kis_core.trading_calendar import get_trading_calendar
    from strategy_common.clock import KST
    calendar = get_trading_calendar()
    population = {}
    for row in original:
        day = date.fromisoformat(row['trade_date'])
        if day not in requested_days:
            continue
        key = row['symbol'], day
        if key in population:
            raise ValueError('Duplicate original reclaim population')
        population[key] = row
    valid, excluded, sectors = {}, [], defaultdict(list)
    for (symbol, day), row in population.items():
        opening = datetime.combine(day, calendar.session_hours(day)[0], KST)
        stamps = [opening + timedelta(minutes=5*i) for i in range(18)]
        values = [indexed.get((symbol, stamp.isoformat(), '5m')) for stamp in stamps]
        if any(v is None for v in values):
            excluded.append((symbol, str(day), 'missing'))
            continue
        if any(len(v) != 6 or v[5] is not True for v in values):
            raise ValueError('Uncompleted reclaim observation')
        if any(v[4] <= 0 or v[2] <= 0 or v[2] > min(v[0], v[3]) or v[1] < max(v[0], v[3]) for v in values):
            excluded.append((symbol, str(day), 'invalid_or_zero_volume'))
            continue
        valid[(symbol, day)] = stamps, values
        sectors[(day, row['sector'])].append(values[-1][3]/values[0][0]-1)
    seen = set()
    for row in features:
        day = date.fromisoformat(row['trade_date']); key = row['symbol'], day
        if day not in requested_days:
            continue
        if key not in valid or key in seen:
            raise ValueError('Incomplete or duplicate consumed reclaim prefix')
        seen.add(key); stamps, values = valid[key]
        if row['sector'] != population[key]['sector']:
            raise ValueError('Changed reclaim sector identity')
        cutoff = stamps[0] + timedelta(minutes=90)
        if datetime.fromisoformat(row['feature_available_at']) != cutoff:
            raise ValueError('Wrong reclaim feature clock')
        bars = row['opening_bars']
        if len(bars) != 18:
            raise ValueError('Incomplete consumed reclaim bars')
        for bar, stamp, value in zip(bars, stamps, values):
            if (bar['symbol'] != key[0] or datetime.fromisoformat(bar['timestamp']) != stamp
                    or tuple(bar[k] for k in ('open','high','low','close','volume')) != tuple(value[:5])):
                raise ValueError('Reclaim source values differ')
        # Independent scalar implementation, not the producer's predicate.
        cash = volume = 0.; vwap = []
        for o,h,l,c,v,_ in values:
            cash += (h+l+c)/3*v; volume += v; vwap.append(cash/volume)
        geometry = (any(values[i][3] > vwap[i] for i in range(6,12))
            and any(values[i][3] < vwap[i] for i in range(12,16))
            and all(values[i][3] > vwap[i] and values[i][3] >= values[i][0] for i in (16,17)))
        if row['guard_at_rank1'] is not geometry:
            raise ValueError('Reclaim geometry differs from raw source')
        sector = sectors[(day,row['sector'])]
        expected = {'checkpoint_sector_return':sum(sector)/len(sector),
                    'checkpoint_sector_breadth':sum(v>0 for v in sector)/len(sector)}
        if any(not math.isclose(row['features'][k], v, rel_tol=1e-12, abs_tol=1e-12) for k,v in expected.items()):
            raise ValueError('Reclaim sector consumed an invalid population')
    if seen != set(valid):
        raise ValueError('Reclaim population differs from complete raw sources')
    for snap in snapshots:
        day=date.fromisoformat(snap['trade_date'])
        opening=datetime.combine(day,calendar.session_hours(day)[0],KST)
        for candidate in snap['candidates']:
            clock=candidate.get('metadata',{}).get('entry_checkpoint_at')
            if clock and datetime.fromisoformat(clock) >= opening+timedelta(minutes=90):
                if datetime.fromisoformat(clock)!=opening+timedelta(minutes=90) or (candidate['symbol'],day) not in seen:
                    raise ValueError('Reclaim candidate lacks declared complete source')
    return {'considered':len(population),'complete_consumed_prefixes':len(seen),
            'excluded':sorted(excluded),'sector_membership_verified':True,
            'geometry_verified':True,'scope':'Additive minute90 proof; mandatory minute45 native obligations remain'}
