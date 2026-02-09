"""Tests for PCIM position management: stops, profit_taking, trailing, time_exit."""

import pytest
from datetime import date, timedelta

from strategy_pcim.positions.manager import PCIMPosition
from strategy_pcim.positions.stops import check_stop_hit
from strategy_pcim.positions.profit_taking import check_take_profit
from strategy_pcim.positions.trailing import update_trailing_stop_eod
from strategy_pcim.positions.time_exit import trading_days_between, check_time_exit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(**overrides) -> PCIMPosition:
    """Create a PCIMPosition with sensible defaults; override any field."""
    defaults = dict(
        symbol="005930",
        entry_date=date(2024, 1, 2),
        entry_price=72000,
        qty=100,
        atr_at_entry=2000,
    )
    defaults.update(overrides)
    return PCIMPosition(**defaults)


# ===========================================================================
# PCIMPosition __post_init__
# ===========================================================================

class TestPCIMPositionInit:
    """Tests for PCIMPosition initialization via __post_init__."""

    def test_remaining_qty_set(self):
        """remaining_qty is set to qty on init."""
        pos = _make_position()
        assert pos.remaining_qty == 100

    def test_initial_stop_computed(self):
        """initial_stop = entry_price - STOP_ATR_MULT * atr_at_entry."""
        pos = _make_position()
        # STOP_ATR_MULT = 1.5, so stop = 72000 - 1.5*2000 = 69000
        assert pos.initial_stop == 72000 - 1.5 * 2000
        assert pos.initial_stop == 69000

    def test_current_stop_equals_initial(self):
        """current_stop starts equal to initial_stop."""
        pos = _make_position()
        assert pos.current_stop == pos.initial_stop

    def test_max_price_equals_entry(self):
        """max_price starts at entry_price."""
        pos = _make_position()
        assert pos.max_price == pos.entry_price

    def test_stop_below_entry(self):
        """initial_stop must be below entry_price."""
        pos = _make_position()
        assert pos.initial_stop < pos.entry_price


# ===========================================================================
# Stop Hit
# ===========================================================================

class TestStopHit:
    """Tests for check_stop_hit price vs stop level."""

    def test_below_stop(self):
        """Price below stop -> hit."""
        pos = _make_position()
        assert check_stop_hit(pos, pos.current_stop - 100) is True

    def test_above_stop(self):
        """Price above stop -> not hit."""
        pos = _make_position()
        assert check_stop_hit(pos, pos.current_stop + 100) is False

    def test_at_stop(self):
        """Price exactly at stop -> hit (<= check)."""
        pos = _make_position()
        assert check_stop_hit(pos, pos.current_stop) is True

    def test_far_above_stop(self):
        """Price far above stop -> not hit."""
        pos = _make_position()
        assert check_stop_hit(pos, pos.entry_price + 10000) is False


# ===========================================================================
# Take Profit
# ===========================================================================

class TestTakeProfit:
    """Tests for check_take_profit at +2.5 ATR level."""

    def test_not_reached(self):
        """Price below TP level -> no action."""
        pos = _make_position()
        should, qty = check_take_profit(pos, pos.entry_price + 100)
        assert should is False
        assert qty == 0

    def test_reached(self):
        """Price at TP level (+2.5 ATR) -> take profit triggered."""
        pos = _make_position(atr_at_entry=2000)
        tp_price = pos.entry_price + 2.5 * 2000  # 72000 + 5000 = 77000
        should, qty = check_take_profit(pos, tp_price)
        assert should is True
        assert qty > 0
        # qty = int(100 * 0.60) = 60
        assert qty == 60

    def test_above_tp_level(self):
        """Price well above TP level -> still triggers."""
        pos = _make_position(atr_at_entry=2000)
        should, qty = check_take_profit(pos, 90000)
        assert should is True
        assert qty == 60

    def test_already_done(self):
        """tp_done=True -> no action even at high price."""
        pos = _make_position()
        pos.tp_done = True
        should, qty = check_take_profit(pos, 999999)
        assert should is False
        assert qty == 0

    def test_qty_is_60_pct_of_remaining(self):
        """TP sells 60% of remaining_qty."""
        pos = _make_position(qty=200)
        tp_price = pos.entry_price + 2.5 * pos.atr_at_entry
        should, qty = check_take_profit(pos, tp_price)
        assert qty == int(200 * 0.60)  # 120


# ===========================================================================
# Trailing Stop
# ===========================================================================

class TestTrailingStop:
    """Tests for update_trailing_stop_eod ratchet logic."""

    def test_trailing_ratchets_up(self):
        """Higher close moves trailing stop up."""
        pos = _make_position()
        old_stop = pos.current_stop
        update_trailing_stop_eod(pos, close_today=80000, atr20_today=1500)
        assert pos.current_stop >= old_stop
        assert pos.max_price == 80000

    def test_trailing_never_decreases(self):
        """Trailing stop never decreases even when price drops."""
        pos = _make_position()
        update_trailing_stop_eod(pos, close_today=80000, atr20_today=1500)
        first_stop = pos.current_stop
        update_trailing_stop_eod(pos, close_today=75000, atr20_today=2000)
        assert pos.current_stop >= first_stop

    def test_trailing_stop_formula(self):
        """trail_level = close - 1.5 * ATR20, stop = max(initial, trail)."""
        pos = _make_position()
        update_trailing_stop_eod(pos, close_today=80000, atr20_today=1500)
        expected_trail = 80000 - 1.5 * 1500  # 77750
        assert pos.trailing_stop == expected_trail
        assert pos.current_stop == max(pos.initial_stop, expected_trail)

    def test_max_price_tracked(self):
        """max_price tracks the highest close seen."""
        pos = _make_position()
        update_trailing_stop_eod(pos, close_today=80000, atr20_today=1500)
        assert pos.max_price == 80000
        update_trailing_stop_eod(pos, close_today=85000, atr20_today=1500)
        assert pos.max_price == 85000
        update_trailing_stop_eod(pos, close_today=82000, atr20_today=1500)
        assert pos.max_price == 85000  # Doesn't decrease

    def test_initial_stop_is_floor(self):
        """current_stop never goes below initial_stop."""
        pos = _make_position()
        # Close below entry -> trail would be below initial_stop
        update_trailing_stop_eod(pos, close_today=70000, atr20_today=3000)
        # trail = 70000 - 4500 = 65500, initial = 69000
        assert pos.current_stop == pos.initial_stop


# ===========================================================================
# Trading Days Between
# ===========================================================================

class TestTradingDaysBetween:
    """Tests for trading_days_between date counting."""

    def test_weekdays_only(self):
        """Mon to Fri -> 4 trading days (Tue-Fri)."""
        start = date(2024, 1, 1)  # Monday
        end = date(2024, 1, 5)    # Friday
        assert trading_days_between(start, end) == 4

    def test_includes_weekend(self):
        """Fri to Mon -> 1 trading day (Mon only)."""
        start = date(2024, 1, 5)  # Friday
        end = date(2024, 1, 8)    # Monday
        assert trading_days_between(start, end) == 1

    def test_custom_calendar(self):
        """Custom calendar (Mon-Thu only) counts fewer days."""
        start = date(2024, 1, 1)  # Monday
        end = date(2024, 1, 5)    # Friday
        is_trading = lambda d: d.weekday() < 4  # Mon-Thu only
        assert trading_days_between(start, end, is_trading) == 3

    def test_same_day(self):
        """Start == end -> 0 trading days (start is exclusive)."""
        d = date(2024, 1, 3)
        assert trading_days_between(d, d) == 0

    def test_next_day_weekday(self):
        """Start to next weekday -> 1 trading day."""
        start = date(2024, 1, 2)  # Tuesday
        end = date(2024, 1, 3)    # Wednesday
        assert trading_days_between(start, end) == 1

    def test_two_full_weeks(self):
        """Two full weeks: Mon to next-next Mon -> 10 trading days."""
        start = date(2024, 1, 1)  # Monday
        end = date(2024, 1, 15)   # Monday, 2 weeks later
        assert trading_days_between(start, end) == 10


# ===========================================================================
# Time Exit (Day 15)
# ===========================================================================

class TestTimeExit:
    """Tests for check_time_exit Day 15 forced exit."""

    def test_before_day_15(self):
        """Held < 15 trading days -> no exit."""
        pos = _make_position(entry_date=date(2024, 1, 2))
        assert check_time_exit(pos, date(2024, 1, 10)) is False

    def test_at_day_15(self):
        """Held >= 15 trading days -> exit triggered."""
        pos = _make_position(entry_date=date(2024, 1, 2))
        # Jan 2 + 22 calendar days = Jan 24 -> 16 trading days >= 15
        target = date(2024, 1, 24)
        result = check_time_exit(pos, target)
        assert result is True

    def test_exactly_day_14(self):
        """Held exactly 14 trading days -> no exit yet."""
        pos = _make_position(entry_date=date(2024, 1, 2))
        # Find a date with exactly 14 trading days from Jan 2
        # Jan 2 (Tue) -> 14 weekdays = Jan 22 (Mon)
        target = date(2024, 1, 22)
        result = check_time_exit(pos, target)
        assert result is False

    def test_custom_calendar(self):
        """Time exit with custom trading calendar."""
        pos = _make_position(entry_date=date(2024, 1, 2))
        # With a restricted calendar, 15 days takes longer
        is_trading = lambda d: d.weekday() < 4  # Mon-Thu only (4 days/week)
        # Won't trigger as quickly with fewer trading days per week
        assert check_time_exit(pos, date(2024, 1, 15), is_trading) is False
