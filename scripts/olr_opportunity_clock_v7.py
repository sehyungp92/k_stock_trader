"""Scope OLR completeness to the same close-anchored cutoff as its features."""
from contextlib import contextmanager
from datetime import datetime


def close_anchored_feature_bars(bars_by_key, cutoff):
    from strategy_common.clock import KST
    from strategy_common.execution_session import session_cutoff
    from strategy_common.source_quality import missing_prefix
    result, excluded = {}, []
    for (day, symbol), bars in bars_by_key.items():
        available = datetime.combine(day, session_cutoff(day, cutoff, anchor='close'), KST)
        if missing_prefix(symbol, day, (bar.timestamp for bar in bars), available):
            excluded.append(f'{symbol}:{day}')
        else:
            result[day, symbol] = bars
    return result, sorted(excluded)


@contextmanager
def olr_completeness_clock():
    from strategy_common import source_quality
    original = source_quality.complete_feature_bars
    source_quality.complete_feature_bars = close_anchored_feature_bars
    try:
        yield
    finally:
        source_quality.complete_feature_bars = original
