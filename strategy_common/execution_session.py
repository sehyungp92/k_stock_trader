"""Shared execution and dated-reference checks; coverage is explicit."""
from datetime import date, time, datetime, timedelta
import math

from kis_core.trading_calendar import get_trading_calendar
from strategy_common.clock import ensure_kst


def session_cutoff(day, regular_time, *, anchor="close"):
    opening, closing = get_trading_calendar().session_hours(day)
    reference, actual = (time(9), opening) if anchor == "open" else (time(15, 30), closing)
    return (datetime.combine(day, regular_time) + (datetime.combine(day, actual) - datetime.combine(day, reference))).time()


def kalcb_session_config(config, day):
    from dataclasses import fields, replace
    opening, closing = get_trading_calendar().session_hours(day)
    if (opening, closing) == (time(9), time(15, 30)):
        return config
    values = {field.name: session_cutoff(day, getattr(config, field.name),
        anchor="close" if field.name in {"session_close", "flatten_time"} else "open")
        for field in fields(config) if isinstance(getattr(config, field.name), time)}
    return replace(config, **values)


def bar_execution_issue(bar) -> str | None:
    if not bar.is_completed:
        return "incomplete_bar"
    values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
    if not all(math.isfinite(value) for value in values):
        return "nonfinite_bar"
    if not (0 < bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high):
        return "incoherent_ohlc"
    timestamp = ensure_kst(bar.timestamp)
    calendar = get_trading_calendar()
    if not calendar.is_trading_day(timestamp.date()):
        return "non_trading_session"
    if calendar.suspension_source(bar.symbol, timestamp.date()):
        return "suspended_symbol"
    opening, closing = calendar.session_hours(timestamp.date())
    if not opening <= timestamp.time().replace(tzinfo=None) <= closing and not bar_is_delayed_closing_auction(bar):
        return "outside_session"
    if bar.volume <= 0:
        return "zero_volume"
    if bar.timeframe.endswith("m"):
        end = timestamp + timedelta(minutes=int(bar.timeframe[:-1]))
        for first, last, _ in calendar.intraday_halts_for(bar.symbol, timestamp.date()):
            first, last = (datetime.combine(timestamp.date(), t, tzinfo=timestamp.tzinfo) for t in (first, last))
            if first <= timestamp and end <= last:
                return "halted_interval"
    return None


def bar_is_delayed_closing_auction(bar) -> bool:
    stamp = ensure_kst(bar.timestamp)
    calendar = get_trading_calendar()
    closing = calendar.session_hours(stamp.date())[1]
    match = calendar.closing_match_time(bar.symbol, stamp.date())
    return match > closing and stamp.time().replace(tzinfo=None) == match


def bar_reopens_after_label(bar) -> bool:
    """A reopening OHLC bar cannot execute or mark its open at its earlier label."""
    stamp = ensure_kst(bar.timestamp)
    return any(first <= stamp.time().replace(tzinfo=None) < last
               for first, last, _ in get_trading_calendar().intraday_halts_for(bar.symbol, stamp.date()))


def bar_has_periodic_auction(bar) -> bool:
    """A sparse auction print is not evidence of continuous matching."""
    stamp = ensure_kst(bar.timestamp)
    return bool(get_trading_calendar().periodic_auction_sources(bar.symbol, stamp.date()))


def prior_session_issue(trade_date: date, reference_date: date | None) -> str | None:
    calendar = get_trading_calendar()
    expected = calendar.previous_trading_day(trade_date)
    if reference_date != expected:
        return "missing_previous_session"
    if not calendar.has_coverage(trade_date) or not calendar.has_coverage(expected):
        return "unverified_calendar"
    return None


def snapshot_reference_issues(snapshot) -> dict[str, str]:
    """Validate the existing dated reference after snapshot hydration."""
    issues = {}
    for candidate in snapshot.candidates:
        raw = candidate.metadata.get("prior_day_date")
        try:
            reference = date.fromisoformat(str(raw)) if raw else None
        except ValueError:
            reference = None
        issue = ("candidate_session_mismatch" if candidate.trade_date != snapshot.trade_date
                 else prior_session_issue(snapshot.trade_date, reference))
        if issue:
            issues[candidate.symbol] = issue
    return issues
