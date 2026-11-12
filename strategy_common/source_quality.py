"""Causal missing-observation fallback; never fill gaps with invented bars."""
from datetime import datetime, timedelta
from functools import lru_cache

from strategy_common.clock import ensure_kst
from strategy_common.session_coverage import session_bar_coverage

POLICY = "completed-prefix-new-risk-v1"


def expected_labels(symbol, day):
    from kis_core.trading_calendar import get_trading_calendar
    calendar = get_trading_calendar()
    return _expected_labels(symbol, day, calendar.session_schedule_hash)


@lru_cache(maxsize=32768)
def _expected_labels(symbol, day, schedule_hash):
    del schedule_hash  # Identity binds cached labels to the sourced calendar.
    result = session_bar_coverage((), day, 5, symbol=symbol)
    if result.get("reason") in {"calendar_coverage", "observations_during_suspension"}:
        raise ValueError("Source-quality policy requires a sourced session calendar")
    return frozenset(datetime.fromisoformat(value) for value in result.get("missing_intervals", ()))


def missing_prefix(symbol, day, timestamps, available_at):
    """Only intervals completed by available_at can be identified as missing."""
    available_at = ensure_kst(available_at)
    observed = {ensure_kst(stamp) for stamp in timestamps if ensure_kst(stamp) + timedelta(minutes=5) <= available_at}
    return sorted(stamp for stamp in expected_labels(symbol, day)
                  if stamp + timedelta(minutes=5) <= available_at and stamp not in observed)


def complete_feature_bars(bars_by_key, cutoff):
    """Drop a symbol-day from features, preserving the separate execution feed."""
    from strategy_common.clock import KST
    from strategy_common.execution_session import session_cutoff
    result, excluded = {}, []
    for (day, symbol), bars in bars_by_key.items():
        available = datetime.combine(day, session_cutoff(day, cutoff, anchor="open"), KST)
        missing = missing_prefix(symbol, day, (bar.timestamp for bar in bars), available)
        if missing:
            excluded.append(f"{symbol}:{day}")
        else:
            result[(day, symbol)] = bars
    return result, sorted(excluded)
