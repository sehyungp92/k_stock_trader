"""Tests for Nulrimok Exit Management."""

import pytest
from datetime import datetime
from strategy_nulrimok.iepe.exits import classify_setup, check_avwap_breakdown, PositionState, SetupType


class TestClassifySetup:
    """Tests for classify_setup: determines MOMENTUM, FLOW_GRIND, MEAN_REVERSION, FAILED, or UNKNOWN."""

    def test_too_early(self):
        """bars_since_entry < 3 -> UNKNOWN (not enough data)."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95, bars_since_entry=2,
        )
        bar = {'high': 105, 'close': 103}
        assert classify_setup(pos, bar, avwap=100) == SetupType.UNKNOWN

    def test_already_classified(self):
        """If setup != UNKNOWN, return existing classification."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95, bars_since_entry=5,
            setup=SetupType.MOMENTUM,
        )
        bar = {'high': 105, 'close': 103}
        assert classify_setup(pos, bar, avwap=100) == SetupType.MOMENTUM

    def test_failed_after_2_sessions(self):
        """sessions_held >= 2 with no breakout -> FAILED."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95, bars_since_entry=5,
            sessions_held=2,
        )
        bar = {'high': 101, 'close': 100}
        assert classify_setup(pos, bar, avwap=100) == SetupType.FAILED

    def test_momentum_breakout(self):
        """high > max_price * 1.005 -> MOMENTUM."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95, bars_since_entry=5,
            max_price=102,
        )
        bar = {'high': 105, 'close': 104}  # high 105 > 102 * 1.005 = 102.51
        assert classify_setup(pos, bar, avwap=100) == SetupType.MOMENTUM

    def test_flow_grind(self):
        """close > entry * 1.002 AND close <= max * 0.998 -> FLOW_GRIND."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95, bars_since_entry=5,
            max_price=102,
        )
        # close=101 > 100*1.002=100.2 True; close=101 <= 102*0.998=101.796 True
        # high=101, not > 102*1.005=102.51 so not momentum
        bar = {'high': 101, 'close': 101}
        assert classify_setup(pos, bar, avwap=100) == SetupType.FLOW_GRIND

    def test_mean_reversion_fallthrough(self):
        """No momentum, no flow grind criteria met -> MEAN_REVERSION."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95, bars_since_entry=5,
            max_price=102,
        )
        # close=100 -> 100 > 100*1.002=100.2 is False -> not FLOW_GRIND
        # high=100 -> 100 > 102*1.005=102.51 is False -> not MOMENTUM
        bar = {'high': 100, 'close': 100}
        assert classify_setup(pos, bar, avwap=100) == SetupType.MEAN_REVERSION

    def test_failed_takes_priority_over_momentum(self):
        """sessions_held >= 2 is checked before momentum breakout."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95, bars_since_entry=5,
            sessions_held=2, max_price=102,
        )
        bar = {'high': 110, 'close': 108}  # Would be momentum, but failed check first
        assert classify_setup(pos, bar, avwap=100) == SetupType.FAILED

    def test_already_classified_not_overridden(self):
        """Already-classified FLOW_GRIND is not overridden even with momentum bar."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95, bars_since_entry=10,
            max_price=102, setup=SetupType.FLOW_GRIND,
        )
        bar = {'high': 110, 'close': 108}
        assert classify_setup(pos, bar, avwap=100) == SetupType.FLOW_GRIND


class TestCheckAvwapBreakdown:
    """Tests for check_avwap_breakdown: close < avwap*(1-0.007) AND vol_ratio > 1.5."""

    def test_breakdown(self):
        """close below avwap threshold with elevated volume -> True."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95,
        )
        # close=98 < 100*(1-0.007)=99.3 True; vol_ratio=200/100=2.0 > 1.5 True
        bar = {'close': 98, 'volume': 200}
        assert check_avwap_breakdown(pos, bar, avwap=100, vol_avg=100) is True

    def test_no_breakdown_price_above_threshold(self):
        """close above avwap threshold -> False."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95,
        )
        bar = {'close': 100, 'volume': 100}
        assert check_avwap_breakdown(pos, bar, avwap=100, vol_avg=100) is False

    def test_no_breakdown_low_volume(self):
        """close below threshold but volume not elevated -> False."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95,
        )
        # close=98 < 99.3 True; vol_ratio=100/100=1.0 not > 1.5 False
        bar = {'close': 98, 'volume': 100}
        assert check_avwap_breakdown(pos, bar, avwap=100, vol_avg=100) is False

    def test_breakdown_with_zero_vol_avg(self):
        """vol_avg=0 -> vol_ratio defaults to 1.0 which is not > 1.5 -> False."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95,
        )
        bar = {'close': 98, 'volume': 200}
        assert check_avwap_breakdown(pos, bar, avwap=100, vol_avg=0) is False


class TestPositionState:
    """Tests for PositionState dataclass."""

    def test_remaining_qty_initialized(self):
        """remaining_qty defaults to qty via __post_init__."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=50, stop=95,
        )
        assert pos.remaining_qty == 50

    def test_remaining_qty_explicit(self):
        """Explicitly set remaining_qty is preserved if non-zero."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=50, stop=95, remaining_qty=30,
        )
        assert pos.remaining_qty == 30

    def test_defaults(self):
        """Default field values."""
        pos = PositionState(
            ticker="005930", entry_time=datetime.now(),
            entry_price=100, qty=10, stop=95,
        )
        assert pos.sessions_held == 0
        assert pos.bars_since_breakdown == 0
        assert pos.in_breakdown is False
        assert pos.max_price == 0.0
        assert pos.setup == SetupType.UNKNOWN
        assert pos.bars_since_entry == 0
        assert pos.entry_low == float('inf')
        assert pos.close_history == []
        assert pos.partial_taken is False
        assert pos.atr30m == 0.0
        assert pos.flow_grind_bars_below_avwap == 0

    def test_setup_type_enum(self):
        """SetupType enum has expected members."""
        assert SetupType.UNKNOWN is not None
        assert SetupType.MOMENTUM is not None
        assert SetupType.MEAN_REVERSION is not None
        assert SetupType.FLOW_GRIND is not None
        assert SetupType.FAILED is not None
