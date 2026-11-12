"""Independent source-population proof for completed first-hour features.

The inherited 45-minute source obligations remain mandatory. New 60-minute
features may use only the complete prefixes independently reconstructed here.
This verifier has no date/symbol exceptions and never rewrites audit outcomes.
"""
from collections import defaultdict
from datetime import date, datetime, timedelta
import io
import json
import math


def verified_rows(generation, relative):
    with generation.open_verified(generation.root / relative) as raw:
        with io.TextIOWrapper(raw, encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def verify_rows(original, features, indexed, requested_days, snapshots):
    """Reconstruct membership and sector inputs independently of the builder."""
    from kis_core.trading_calendar import get_trading_calendar
    from strategy_common.clock import KST
    calendar = get_trading_calendar()
    population = {}
    for row in original:
        day = date.fromisoformat(row['trade_date'])
        if day not in requested_days:
            continue
        key = (row['symbol'], day)
        if key in population:
            raise ValueError('Duplicate original first-hour population')
        population[key] = row
    valid, excluded, sectors = {}, [], defaultdict(list)
    for (symbol, day), row in population.items():
        opening = datetime.combine(day, calendar.session_hours(day)[0], KST)
        stamps = [opening + timedelta(minutes=5*i) for i in range(12)]
        values = [indexed.get((symbol, stamp.isoformat(), '5m')) for stamp in stamps]
        if any(value is None for value in values):
            excluded.append((symbol, str(day)))
            continue
        if any(len(v) != 6 or v[5] is not True for v in values):
            raise ValueError('Uncompleted first-hour source observation')
        valid[(symbol, day)] = (stamps, values)
        sectors[(day, row['sector'])].append(values[-1][3]/values[0][0]-1)
    seen = set()
    for row in features:
        day = date.fromisoformat(row['trade_date'])
        if day not in requested_days:
            continue
        key = (row['symbol'], day)
        if key not in valid or key in seen:
            raise ValueError('Incomplete or duplicate consumed first-hour prefix')
        seen.add(key)
        stamps, values = valid[key]
        if row['sector'] != population[key]['sector']:
            raise ValueError('Changed first-hour sector identity')
        cutoff = stamps[0] + timedelta(minutes=60)
        if datetime.fromisoformat(row['feature_available_at']) != cutoff:
            raise ValueError('Wrong first-hour feature availability')
        bars = row['opening_bars']
        if len(bars) != 12:
            raise ValueError('Incomplete consumed first-hour bars')
        for bar, stamp, value in zip(bars, stamps, values):
            if (bar['symbol'] != key[0] or datetime.fromisoformat(bar['timestamp']) != stamp
                    or tuple(bar[k] for k in ('open','high','low','close','volume')) != tuple(value[:5])):
                raise ValueError('First-hour source values differ')
        sector = sectors[(day, row['sector'])]
        expected = {'checkpoint_sector_return':sum(sector)/len(sector),
                    'checkpoint_sector_breadth':sum(v > 0 for v in sector)/len(sector)}
        if any(not math.isclose(row['features'][k], v, rel_tol=1e-12, abs_tol=1e-12) for k,v in expected.items()):
            raise ValueError('First-hour sector consumed an invalid population')
    if seen != set(valid):
        raise ValueError('First-hour feature population differs from complete sources')
    for snap in snapshots:
        day = date.fromisoformat(snap['trade_date'])
        opening = datetime.combine(day, calendar.session_hours(day)[0], KST)
        for candidate in snap['candidates']:
            clock = candidate.get('metadata', {}).get('entry_checkpoint_at')
            if clock and datetime.fromisoformat(clock) >= opening+timedelta(minutes=60):
                if (candidate['symbol'], day) not in seen:
                    raise ValueError('First-hour candidate lacks complete source features')
    return {'considered':len(population), 'complete_consumed_prefixes':len(seen),
            'incomplete_excluded':sorted(excluded), 'sector_membership_verified':True}


def verify_first_hour(generation, indexed, required, broker):
    binding = generation.payload['evidence'].get('first_hour_feature_sources')
    if not binding or binding.get('version') != 'complete-first-hour-population-v1':
        raise ValueError('Unbound first-hour feature population')
    from itertools import chain
    original = chain.from_iterable(verified_rows(generation, p) for p in binding['original_populations'])
    return verify_rows(original, verified_rows(generation, binding['features']), indexed,
                       {day for _,day,_ in required}, broker.qualification_snapshots)
