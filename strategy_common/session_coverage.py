"""Conservative bar coverage: unexplained missing intervals remain unverified."""
from datetime import datetime, time, timedelta

from kis_core.trading_calendar import get_trading_calendar
from strategy_common.clock import KST, ensure_kst


def session_bar_coverage(timestamps, day, minutes=1, *, symbol=None):
    calendar = get_trading_calendar()
    if not calendar.has_coverage(day) or not calendar.has_session_coverage(day):
        return {"status": "unverified", "reason": "calendar_coverage"}
    if not calendar.is_trading_day(day):
        return {"status": "not_applicable", "reason": "closed_session"}
    if minutes <= 0:
        raise ValueError("Bar duration must be positive")
    observed = {ensure_kst(ts) for ts in timestamps if ensure_kst(ts).date() == day}
    suspension = calendar.suspension_source(symbol, day)
    if suspension:
        return {"status": "unverified" if observed else "pass", "date": day.isoformat(),
                "reason": "observations_during_suspension" if observed else "sourced_symbol_suspension",
                "source": suspension, "missing_intervals": [], "expected_intervals": 0,
                "calendar": calendar.session_schedule_hash}
    opening, closing = calendar.session_hours(day)
    start = datetime.combine(day, opening, tzinfo=KST)
    end = datetime.combine(day, closing, tzinfo=KST)
    auction = end - timedelta(minutes=calendar.closing_auction_minutes)
    midnight = datetime.combine(day, time(), tzinfo=KST)

    def label(value):
        return midnight + timedelta(minutes=int((value - midnight).total_seconds() // (60 * minutes)) * minutes)

    match = calendar.closing_match_time(symbol, day)
    # An extended auction is one event at its actual time, not an earlier bin.
    expected = {datetime.combine(day, match, tzinfo=KST) if match != closing else label(end)}
    cursor = start
    periodic_sources = calendar.periodic_auction_sources(symbol, day)
    while cursor < (end if periodic_sources else auction):
        expected.add(label(cursor))
        cursor += timedelta(minutes=30 if periodic_sources else 1)
    halted = set()
    for first, last, _ in calendar.intraday_halts_for(symbol, day):
        first = datetime.combine(day, first, tzinfo=KST)
        last = datetime.combine(day, last, tzinfo=KST)
        halted.update(ts for ts in expected if first <= ts and ts + timedelta(minutes=minutes) <= last)
    expected -= halted
    contradictory = sorted(observed & halted)
    off_auction = sorted(observed - expected) if periodic_sources else []
    missing = sorted(expected - observed)
    return {"status": "unverified" if missing or contradictory or off_auction else "pass", "date": day.isoformat(),
            "missing_intervals": [ts.isoformat() for ts in missing],
            "sourced_halt_intervals": [ts.isoformat() for ts in sorted(halted)],
            "observations_during_halt": [ts.isoformat() for ts in contradictory],
            "periodic_auction_sources": list(periodic_sources),
            "observations_outside_periodic_auctions": [ts.isoformat() for ts in off_auction],
            "expected_intervals": len(expected), "calendar": calendar.session_schedule_hash}
