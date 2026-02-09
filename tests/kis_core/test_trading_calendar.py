"""Tests for KRX Trading Calendar."""
import pytest
from datetime import date
from kis_core.trading_calendar import KRXTradingCalendar


class TestKRXTradingCalendar:
    """Tests for KRXTradingCalendar holiday and trading day logic."""

    @pytest.fixture
    def calendar(self):
        holidays = {
            date(2024, 1, 1),   # New Year
            date(2024, 3, 1),   # Independence Movement Day
            date(2024, 5, 5),   # Children's Day (Sunday in 2024, but kept for testing)
            date(2024, 5, 6),   # Substitute holiday for Children's Day
        }
        return KRXTradingCalendar(holidays=holidays)

    # -----------------------------------------------------------------
    # is_trading_day
    # -----------------------------------------------------------------

    def test_weekday_is_trading(self, calendar):
        # Monday, Jan 15, 2024
        assert calendar.is_trading_day(date(2024, 1, 15)) is True

    def test_tuesday_is_trading(self, calendar):
        assert calendar.is_trading_day(date(2024, 1, 16)) is True

    def test_wednesday_is_trading(self, calendar):
        assert calendar.is_trading_day(date(2024, 1, 17)) is True

    def test_thursday_is_trading(self, calendar):
        assert calendar.is_trading_day(date(2024, 1, 18)) is True

    def test_friday_is_trading(self, calendar):
        assert calendar.is_trading_day(date(2024, 1, 19)) is True

    def test_saturday_not_trading(self, calendar):
        assert calendar.is_trading_day(date(2024, 1, 13)) is False

    def test_sunday_not_trading(self, calendar):
        assert calendar.is_trading_day(date(2024, 1, 14)) is False

    def test_holiday_not_trading(self, calendar):
        # New Year
        assert calendar.is_trading_day(date(2024, 1, 1)) is False

    def test_march_1_holiday(self, calendar):
        # Independence Movement Day (Friday in 2024)
        assert calendar.is_trading_day(date(2024, 3, 1)) is False

    def test_substitute_holiday(self, calendar):
        # May 6 substitute holiday
        assert calendar.is_trading_day(date(2024, 5, 6)) is False

    # -----------------------------------------------------------------
    # previous_trading_day
    # -----------------------------------------------------------------

    def test_previous_trading_day_simple(self, calendar):
        # Tuesday -> Monday
        result = calendar.previous_trading_day(date(2024, 1, 16))
        assert result == date(2024, 1, 15)

    def test_previous_trading_day_skips_weekend(self, calendar):
        # Monday -> previous Friday
        result = calendar.previous_trading_day(date(2024, 1, 15))
        assert result == date(2024, 1, 12)

    def test_previous_trading_day_skips_holiday(self, calendar):
        # Jan 2 -> Dec 29 (Friday), skipping Jan 1 holiday
        result = calendar.previous_trading_day(date(2024, 1, 2))
        assert result == date(2023, 12, 29)

    def test_previous_trading_day_skips_holiday_and_weekend(self, calendar):
        # Mar 4 (Monday) -> previous trading day skips Mar 1 (holiday, Friday)
        # so goes to Feb 29 (Thursday)
        result = calendar.previous_trading_day(date(2024, 3, 4))
        assert result == date(2024, 2, 29)

    # -----------------------------------------------------------------
    # next_trading_day
    # -----------------------------------------------------------------

    def test_next_trading_day_simple(self, calendar):
        # Monday -> Tuesday
        result = calendar.next_trading_day(date(2024, 1, 15))
        assert result == date(2024, 1, 16)

    def test_next_trading_day_skips_weekend(self, calendar):
        # Friday -> Monday
        result = calendar.next_trading_day(date(2024, 1, 12))
        assert result == date(2024, 1, 15)

    def test_next_trading_day_skips_holiday(self, calendar):
        # Feb 29 (Thursday) -> next trading day skips Mar 1 (holiday)
        # Mar 1 is a Friday holiday, Mar 2-3 are weekend, so next is Mar 4
        result = calendar.next_trading_day(date(2024, 2, 29))
        assert result == date(2024, 3, 4)

    def test_next_trading_day_from_saturday(self, calendar):
        # Saturday -> Monday
        result = calendar.next_trading_day(date(2024, 1, 13))
        assert result == date(2024, 1, 15)

    def test_next_trading_day_from_sunday(self, calendar):
        # Sunday -> Monday
        result = calendar.next_trading_day(date(2024, 1, 14))
        assert result == date(2024, 1, 15)

    # -----------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------

    def test_empty_holidays(self):
        cal = KRXTradingCalendar(holidays=set())
        # Weekday with no holidays -> trading
        assert cal.is_trading_day(date(2024, 1, 1)) is True  # Only weekday check
        # Weekend still not trading
        assert cal.is_trading_day(date(2024, 1, 6)) is False

    def test_many_consecutive_holidays(self):
        # Long holiday block (e.g., Lunar New Year extended)
        holidays = {
            date(2024, 2, 9),   # Friday
            date(2024, 2, 12),  # Monday (substitute)
        }
        cal = KRXTradingCalendar(holidays=holidays)
        # Feb 8 (Thursday) -> next trading day skips Feb 9 (holiday),
        # Feb 10-11 (weekend), Feb 12 (holiday) -> Feb 13
        result = cal.next_trading_day(date(2024, 2, 8))
        assert result == date(2024, 2, 13)

    def test_previous_and_next_are_inverse(self, calendar):
        # For a regular weekday, next(prev(d)) should return d
        d = date(2024, 1, 16)  # Tuesday
        prev = calendar.previous_trading_day(d)
        next_after_prev = calendar.next_trading_day(prev)
        assert next_after_prev == d
