"""Memoize the exact native complete-prefix result for identical timestamps.

This never approximates coverage or supplies bars. A miss executes the original
native rule. A hit returns the caller's own bars for exactly the keys previously
accepted by that rule. Native replay/audit runs in a separate unpatched process.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def memoized_prefix(original, bars_by_key, cutoff, cache_root, contract, counters):
    days = {day for day, symbol in bars_by_key}
    if len(days) != 1:
        return original(bars_by_key, cutoff)
    keys = list(bars_by_key)
    records = [(day.isoformat(), symbol, [bar.timestamp.isoformat() for bar in bars_by_key[day,symbol]])
               for day,symbol in keys]
    identity = signature({'contract':contract, 'cutoff':cutoff.isoformat(), 'timestamps':records})
    path = Path(cache_root)/(identity+'.json')
    if path.exists():
        row = json.loads(path.read_text(encoding='utf-8'))
        checksum = row.pop('sha256')
        if checksum != signature(row) or row['input_identity'] != identity:
            raise ValueError('Native prefix cache authentication failed')
        accepted = set(row['accepted_indices'])
        if any(not isinstance(i,int) or not 0 <= i < len(keys) for i in accepted):
            raise ValueError('Invalid native prefix cache keys')
        counters['hits'] = counters.get('hits',0)+1
        return {key:bars_by_key[key] for i,key in enumerate(keys) if i in accepted}, row['excluded']
    filtered, excluded = original(bars_by_key, cutoff)
    if not set(filtered).issubset(keys) or any(filtered[k] is not bars_by_key[k] for k in filtered):
        raise ValueError('Native prefix contract changed')
    row = {'input_identity':identity, 'accepted_indices':[i for i,k in enumerate(keys) if k in filtered],
           'excluded':excluded}
    row['sha256'] = signature(row)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row,sort_keys=True),encoding='utf-8')
    temporary.replace(path)
    counters['misses'] = counters.get('misses',0)+1
    return filtered, excluded


@contextmanager
def cached_prefix(cache_root, source_hash, counters):
    from strategy_common import source_quality
    from kis_core.trading_calendar import get_trading_calendar
    original = source_quality.complete_feature_bars
    calendar = get_trading_calendar()
    contract = {'native_source_hash':source_hash, 'schedule_hash':calendar.session_schedule_hash,
                'policy':source_quality.POLICY, 'memo_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    source_quality.complete_feature_bars = lambda bars, cutoff: memoized_prefix(
        original,bars,cutoff,cache_root,contract,counters)
    try:
        yield
    finally:
        source_quality.complete_feature_bars = original
