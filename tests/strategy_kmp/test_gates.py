"""Tests for KMP entry gates."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import time
import math

from strategy_kmp.core.state import SymbolState, State
from strategy_kmp.core.gates import (
    lock_or_and_filter,
    spread_ok,
    rvol_ok,
    vi_blocked,
    minutes_since_0916,
    min_surge_threshold,
    min_surge_threshold_strict,
    size_time_multiplier,
    is_past_entry_cutoff,
    is_past_flatten_time,
    is_in_or_window,
)


class TestMinutesSince0916:
    """Tests for minutes_since_0916 function."""

    def test_at_0916(self):
        """Test at exactly 09:16."""
        now = datetime(2024, 1, 15, 9, 16, 0)
        assert minutes_since_0916(now) == 0.0

    def test_at_0930(self):
        """Test at 09:30 (14 minutes after)."""
        now = datetime(2024, 1, 15, 9, 30, 0)
        assert minutes_since_0916(now) == 14.0

    def test_at_1000(self):
        """Test at 10:00 (44 minutes after)."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        assert minutes_since_0916(now) == 44.0

    def test_before_0916(self):
        """Test before 09:16 returns 0."""
        now = datetime(2024, 1, 15, 9, 0, 0)
        assert minutes_since_0916(now) == 0.0


class TestMinSurgeThreshold:
    """Tests for min_surge_threshold function."""

    def test_at_0916(self):
        """Test threshold at 09:16."""
        # Base is 3.0
        mock_switches = MagicMock()
        mock_switches.min_surge_slope = 0.03
        assert min_surge_threshold(0, mock_switches) == 3.0

    def test_at_10_minutes(self):
        """Test threshold at 10 minutes after."""
        mock_switches = MagicMock()
        mock_switches.min_surge_slope = 0.03
        # 3.0 + 0.03 * 10 = 3.3
        assert min_surge_threshold(10, mock_switches) == pytest.approx(3.3, abs=0.01)

    def test_at_44_minutes(self):
        """Test threshold capped at 44 minutes."""
        mock_switches = MagicMock()
        mock_switches.min_surge_slope = 0.03
        # 3.0 + 0.03 * 44 = 4.32
        assert min_surge_threshold(44, mock_switches) == pytest.approx(4.32, abs=0.01)

    def test_beyond_44_minutes(self):
        """Test threshold doesn't increase beyond 44 minutes."""
        mock_switches = MagicMock()
        mock_switches.min_surge_slope = 0.03
        threshold_44 = min_surge_threshold(44, mock_switches)
        threshold_60 = min_surge_threshold(60, mock_switches)
        assert threshold_44 == threshold_60


class TestMinSurgeThresholdStrict:
    """Tests for strict surge threshold."""

    def test_at_0916(self):
        """Test strict threshold at 09:16."""
        assert min_surge_threshold_strict(0) == 3.0

    def test_at_10_minutes(self):
        """Test strict threshold at 10 minutes."""
        # 3.0 + 0.04 * 10 = 3.4
        assert min_surge_threshold_strict(10) == pytest.approx(3.4, abs=0.01)

    def test_strict_higher_than_permissive(self):
        """Test strict is higher than permissive at same time."""
        mock_switches = MagicMock()
        mock_switches.min_surge_slope = 0.03

        for minutes in [10, 20, 30]:
            permissive = min_surge_threshold(minutes, mock_switches)
            strict = min_surge_threshold_strict(minutes)
            assert strict > permissive


class TestSizeTimeMultiplier:
    """Tests for size_time_multiplier function."""

    def test_at_0916(self):
        """Test size multiplier at 09:16."""
        assert size_time_multiplier(0) == 1.0

    def test_at_20_minutes(self):
        """Test size multiplier at 20 minutes."""
        # 1.0 - 0.012 * 20 = 0.76
        assert size_time_multiplier(20) == pytest.approx(0.76, abs=0.01)

    def test_floor_at_45_minutes(self):
        """Test size multiplier at 45+ minutes (capped at 44 min)."""
        # Minutes capped at 44, so result = max(0.45, 1.0 - 0.012 * 44) = max(0.45, 0.472) = 0.472
        assert size_time_multiplier(45) == pytest.approx(0.472, abs=0.001)

    def test_floor_at_100_minutes(self):
        """Test floor is maintained beyond 44 minutes."""
        # Same as 44 minutes due to cap
        assert size_time_multiplier(100) == pytest.approx(0.472, abs=0.001)


class TestLockOrAndFilter:
    """Tests for lock_or_and_filter function."""

    @pytest.fixture
    def state(self):
        """Create symbol state for testing."""
        s = SymbolState(code="005930")
        s.or_high = 72000
        s.or_low = 71000
        return s

    def test_locks_or(self, state):
        """Test OR is locked after call."""
        mock_switches = MagicMock()
        mock_switches.or_range_max = 0.07
        mock_switches.log_would_block = MagicMock()

        lock_or_and_filter(state, mock_switches)

        assert state.or_locked is True
        assert state.or_mid == 71500

    def test_valid_range(self, state):
        """Test valid OR range passes."""
        mock_switches = MagicMock()
        mock_switches.or_range_max = 0.07
        mock_switches.log_would_block = MagicMock()

        # Range = (72000 - 71000) / 71500 = 1.4%
        result = lock_or_and_filter(state, mock_switches)

        assert result is True

    def test_range_too_narrow(self, state):
        """Test narrow OR range fails."""
        state.or_high = 71100
        state.or_low = 71000
        # Range = 100 / 71050 = 0.14% (< 1.2% minimum)

        mock_switches = MagicMock()
        mock_switches.or_range_max = 0.07
        mock_switches.log_would_block = MagicMock()

        result = lock_or_and_filter(state, mock_switches)

        assert result is False

    def test_range_too_wide(self, state):
        """Test wide OR range fails with strict max."""
        state.or_high = 75000
        state.or_low = 70000
        # Range = 5000 / 72500 = 6.9% (> 5.5% max)

        mock_switches = MagicMock()
        mock_switches.or_range_max = 0.055  # Strict max
        mock_switches.log_would_block = MagicMock()

        result = lock_or_and_filter(state, mock_switches)

        assert result is False

    def test_zero_mid_fails(self):
        """Test zero or_mid fails."""
        state = SymbolState(code="005930")
        state.or_high = 0
        state.or_low = 0

        mock_switches = MagicMock()
        mock_switches.or_range_max = 0.07
        mock_switches.log_would_block = MagicMock()

        result = lock_or_and_filter(state, mock_switches)

        assert result is False

    def test_logs_would_block(self, state):
        """Test would-block logging when using permissive max."""
        state.or_high = 74000
        state.or_low = 70000
        # Range = 4000 / 72000 = 5.6% (passes 7% but would fail 5.5%)

        mock_switches = MagicMock()
        mock_switches.or_range_max = 0.07  # Permissive
        mock_switches.log_would_block = MagicMock()

        result = lock_or_and_filter(state, mock_switches)

        assert result is True
        mock_switches.log_would_block.assert_called_once()


class TestSpreadOk:
    """Tests for spread_ok function."""

    def test_spread_below_max(self):
        """Test spread below max passes."""
        state = SymbolState(code="005930")
        state.spread_pct = 0.003  # 0.3% < 0.4% max

        assert spread_ok(state) is True

    def test_spread_at_max(self):
        """Test spread at max passes."""
        state = SymbolState(code="005930")
        state.spread_pct = 0.004  # 0.4% = max

        assert spread_ok(state) is True

    def test_spread_above_max(self):
        """Test spread above max fails."""
        state = SymbolState(code="005930")
        state.spread_pct = 0.005  # 0.5% > 0.4% max

        assert spread_ok(state) is False


class TestRvolOk:
    """Tests for rvol_ok function."""

    def test_rvol_above_min(self):
        """Test RVol above minimum passes."""
        state = SymbolState(code="005930")
        state.rvol_1m = 2.5  # > 2.0 min

        assert rvol_ok(state) is True

    def test_rvol_at_min(self):
        """Test RVol at minimum passes."""
        state = SymbolState(code="005930")
        state.rvol_1m = 2.0

        assert rvol_ok(state) is True

    def test_rvol_below_min(self):
        """Test RVol below minimum fails."""
        state = SymbolState(code="005930")
        state.rvol_1m = 1.5

        assert rvol_ok(state) is False


class TestViBlocked:
    """Tests for vi_blocked function."""

    def test_no_vi_ref_not_blocked(self):
        """Test missing VI reference does not block (fail-open: no VI data = no VI concern)."""
        state = SymbolState(code="005930")
        state.vi_ref = 0

        assert vi_blocked(state, 72000, 5) is False

    def test_within_cooldown_blocked(self):
        """Test within VI cooldown is blocked."""
        state = SymbolState(code="005930")
        state.vi_ref = 70000
        state.last_vi_ts = time.time() - 300  # 5 minutes ago (< 10 min cooldown)

        assert vi_blocked(state, 72000, 5) is True

    def test_past_cooldown_not_blocked(self):
        """Test past VI cooldown is not blocked when below wall."""
        state = SymbolState(code="005930")
        state.vi_ref = 70000
        state.last_vi_ts = time.time() - 700  # 11+ minutes ago (> 10 min cooldown)

        # Static up = 70000 * 1.02 = 71400
        # Wall = 71400 - (10 * 5) = 71350
        # Entry at 71000 should not be blocked (below wall)
        assert vi_blocked(state, 71000, 5) is False

    def test_near_vi_wall_blocked(self):
        """Test entry near VI wall is blocked."""
        state = SymbolState(code="005930")
        state.vi_ref = 70000
        state.last_vi_ts = time.time() - 700  # Past cooldown

        # Static up = 70000 * 1.02 = 71400
        # Wall = 71400 - (10 * 5) = 71350
        # Entry at 71350 should be blocked

        assert vi_blocked(state, 71350, 5) is True

    def test_below_vi_wall_not_blocked(self):
        """Test entry below VI wall is not blocked."""
        state = SymbolState(code="005930")
        state.vi_ref = 70000
        state.last_vi_ts = time.time() - 700  # Past cooldown

        # Static up = 70000 * 1.02 = 71400
        # Wall = 71400 - (10 * 5) = 71350
        # Entry at 71000 should not be blocked

        assert vi_blocked(state, 71000, 5) is False


class TestTimeChecks:
    """Tests for time check functions."""

    def test_is_in_or_window_at_0900(self):
        """Test 09:00 is in OR window."""
        ts = datetime(2024, 1, 15, 9, 0, 0)
        assert is_in_or_window(ts) is True

    def test_is_in_or_window_at_0914(self):
        """Test 09:14 is in OR window."""
        ts = datetime(2024, 1, 15, 9, 14, 59)
        assert is_in_or_window(ts) is True

    def test_is_in_or_window_at_0915(self):
        """Test 09:15 is not in OR window."""
        ts = datetime(2024, 1, 15, 9, 15, 0)
        assert is_in_or_window(ts) is False

    def test_is_past_entry_cutoff_before(self):
        """Test before entry cutoff."""
        ts = datetime(2024, 1, 15, 9, 30, 0)
        assert is_past_entry_cutoff(ts) is False

    def test_is_past_entry_cutoff_at(self):
        """Test at entry cutoff."""
        ts = datetime(2024, 1, 15, 10, 0, 0)
        assert is_past_entry_cutoff(ts) is True

    def test_is_past_flatten_time_before(self):
        """Test before flatten time."""
        ts = datetime(2024, 1, 15, 14, 0, 0)
        assert is_past_flatten_time(ts) is False

    def test_is_past_flatten_time_at(self):
        """Test at flatten time."""
        ts = datetime(2024, 1, 15, 14, 30, 0)
        assert is_past_flatten_time(ts) is True
