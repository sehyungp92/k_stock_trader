"""Day 15 Time Exit."""

from datetime import date, timedelta
from typing import Optional, Callable
from loguru import logger

from .manager import PCIMPosition
from ..config.constants import EXITS


def trading_days_between(start: date, end: date, is_trading_day: Optional[Callable[[date], bool]] = None) -> int:
    """
    Count trading days between dates.

    Args:
        start: Start date (exclusive - entry date)
        end: End date (inclusive - today)
        is_trading_day: Optional function to check if date is a trading day.
                       Falls back to weekday check if not provided.
    """
    count = 0
    current = start + timedelta(days=1)  # Start day after entry
    while current <= end:
        if is_trading_day:
            if is_trading_day(current):
                count += 1
        elif current.weekday() < 5:  # Fallback: weekday only
            count += 1
        current = current + timedelta(days=1)
    return count


def check_time_exit(pos: PCIMPosition, today: date, is_trading_day: Optional[Callable[[date], bool]] = None) -> bool:
    """
    Check if Day 15 time exit triggered.

    Uses KRX trading calendar if is_trading_day function provided,
    otherwise falls back to KIS trading calendar or weekday-only counting.
    """
    if is_trading_day is None:
        try:
            from kis_core.trading_calendar import get_trading_calendar
            cal = get_trading_calendar()
            is_trading_day = cal.is_trading_day
        except ImportError:
            pass  # Fall through to weekday-only counting
    days_held = trading_days_between(pos.entry_date, today, is_trading_day)

    if days_held >= EXITS["TIME_EXIT_DAY"]:
        logger.info(f"{pos.symbol}: Day {days_held} time exit triggered (entry={pos.entry_date})")
        return True
    return False
