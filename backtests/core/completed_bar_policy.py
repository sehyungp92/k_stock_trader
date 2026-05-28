from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable

from strategy_common.clock import KST, ensure_kst
from strategy_common.market import MarketBar

KRX_OPEN = time(9, 0)
KRX_CLOSE = time(15, 30)
_TRADING_CALENDAR_LOADED = False
_TRADING_CALENDAR = None


def _get_trading_calendar():
    global _TRADING_CALENDAR_LOADED, _TRADING_CALENDAR
    if _TRADING_CALENDAR_LOADED:
        return _TRADING_CALENDAR
    _TRADING_CALENDAR_LOADED = True
    try:
        from kis_core.trading_calendar import get_trading_calendar

        _TRADING_CALENDAR = get_trading_calendar()
    except Exception:  # pragma: no cover - defensive import for isolated tooling
        _TRADING_CALENDAR = None
    return _TRADING_CALENDAR


def is_trading_day(value: date) -> bool:
    calendar = _get_trading_calendar()
    if calendar is None:
        return value.weekday() < 5
    return calendar.is_trading_day(value)


def is_regular_session(value: datetime) -> bool:
    ts = ensure_kst(value)
    return is_trading_day(ts.date()) and KRX_OPEN <= ts.time() <= KRX_CLOSE


def timeframe_delta(timeframe: str) -> timedelta | None:
    normalized = timeframe.strip().lower()
    if normalized in {"d", "1d", "day", "daily"}:
        return None
    if normalized.endswith("m"):
        return timedelta(minutes=int(normalized[:-1] or "1"))
    if normalized.endswith("h"):
        return timedelta(hours=int(normalized[:-1] or "1"))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def session_close_for(value: datetime | date) -> datetime:
    day = value if isinstance(value, date) and not isinstance(value, datetime) else ensure_kst(value).date()
    return datetime.combine(day, KRX_CLOSE, tzinfo=KST)


def bar_availability_time(bar_or_timestamp: MarketBar | datetime, timeframe: str | None = None) -> datetime:
    if isinstance(bar_or_timestamp, MarketBar):
        timestamp = ensure_kst(bar_or_timestamp.timestamp)
        timeframe = bar_or_timestamp.timeframe
    else:
        timestamp = ensure_kst(bar_or_timestamp)
        if timeframe is None:
            raise ValueError("timeframe is required when passing a timestamp")

    delta = timeframe_delta(timeframe)
    if delta is None:
        return session_close_for(timestamp)
    return timestamp + delta


def is_bar_visible_at(bar: MarketBar, replay_time: datetime) -> bool:
    if not bar.is_completed:
        return False
    return bar_availability_time(bar) <= ensure_kst(replay_time)


def visible_bars_at(bars: Iterable[MarketBar], replay_time: datetime) -> list[MarketBar]:
    return [bar for bar in bars if is_bar_visible_at(bar, replay_time)]


def assert_no_lookahead(lower_time: datetime, higher_bar: MarketBar) -> None:
    if not is_bar_visible_at(higher_bar, lower_time):
        raise ValueError(
            "Higher-timeframe bar is not yet available: "
            f"{higher_bar.symbol} {higher_bar.timeframe} {higher_bar.timestamp}"
        )
