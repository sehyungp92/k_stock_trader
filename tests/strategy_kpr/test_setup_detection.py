"""Tests for KPR setup detection."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import math

from strategy_kpr.core.state import SymbolState, FSMState
from strategy_kpr.core.setup_detection import (
    check_vwap_depth,
    detect_panic_flush,
    detect_drift,
    detect_setup,
)


class TestCheckVwapDepth:
    """Tests for VWAP depth check."""

    @pytest.fixture
    def mock_switches(self):
        """Create mock switches with default values."""
        switches = MagicMock()
        switches.vwap_depth_min = 0.015  # Permissive 1.5%
        switches.vwap_depth_max = 0.06   # Permissive 6%
        switches.log_would_block = MagicMock()
        return switches

    def test_in_band(self, mock_switches):
        """Test price within VWAP band."""
        # VWAP = 72000, price = 70000
        # Depth = (72000 - 70000) / 72000 = 2.78%
        in_band, depth = check_vwap_depth(70000, 72000, mock_switches)

        assert in_band is True
        assert depth == pytest.approx(0.0278, abs=0.001)

    def test_below_min_depth(self, mock_switches):
        """Test price too close to VWAP."""
        # VWAP = 72000, price = 71500
        # Depth = (72000 - 71500) / 72000 = 0.69%
        in_band, depth = check_vwap_depth(71500, 72000, mock_switches)

        assert in_band is False
        assert depth == pytest.approx(0.0069, abs=0.001)

    def test_above_max_depth(self, mock_switches):
        """Test price too far from VWAP."""
        # VWAP = 72000, price = 65000
        # Depth = (72000 - 65000) / 72000 = 9.72%
        in_band, depth = check_vwap_depth(65000, 72000, mock_switches)

        assert in_band is False
        assert depth == pytest.approx(0.0972, abs=0.001)

    def test_zero_vwap(self, mock_switches):
        """Test handles zero VWAP."""
        in_band, depth = check_vwap_depth(70000, 0, mock_switches)

        assert in_band is False
        assert depth == 0.0

    def test_logs_would_block_strict(self, mock_switches):
        """Test logs would-block for strict thresholds."""
        mock_switches.vwap_depth_min = 0.01  # Very permissive
        mock_switches.vwap_depth_max = 0.10  # Very permissive

        # Depth 1.5% would pass permissive but fail strict (2%)
        in_band, _ = check_vwap_depth(70920, 72000, mock_switches, symbol="005930")

        assert in_band is True


class TestDetectPanicFlush:
    """Tests for panic flush detection."""

    def test_panic_detected(self):
        """Test panic flush is detected."""
        state = SymbolState(code="005930")
        state.hod = 72000
        state.hod_time = datetime.now() - timedelta(minutes=10)  # Within 15 min max age

        # 3%+ drop from HOD within PANIC_MAX_AGE_MIN (15 min)
        price = 69840  # 3% drop (meets PANIC_DROP_PCT = 0.03)
        bar_time = datetime.now()

        result = detect_panic_flush(state, price, bar_time)

        assert result is True

    def test_panic_not_detected_drop_too_small(self):
        """Test panic not detected with small drop."""
        state = SymbolState(code="005930")
        state.hod = 72000
        state.hod_time = datetime.now() - timedelta(minutes=10)

        # Only 2% drop (below 3% threshold PANIC_DROP_PCT)
        price = 70560  # 2% drop
        bar_time = datetime.now()

        result = detect_panic_flush(state, price, bar_time)

        assert result is False

    def test_panic_not_detected_too_old(self):
        """Test panic not detected when HOD too old."""
        state = SymbolState(code="005930")
        state.hod = 72000
        state.hod_time = datetime.now() - timedelta(minutes=20)  # > 15 minutes (PANIC_MAX_AGE_MIN)

        price = 69840  # 3% drop
        bar_time = datetime.now()

        result = detect_panic_flush(state, price, bar_time)

        assert result is False

    def test_panic_no_hod(self):
        """Test panic not detected without HOD."""
        state = SymbolState(code="005930")
        state.hod = 0
        state.hod_time = None

        result = detect_panic_flush(state, 70000, datetime.now())

        assert result is False


class TestDetectDrift:
    """Tests for drift detection."""

    def test_drift_detected(self):
        """Test drift is detected."""
        state = SymbolState(code="005930")
        state.hod = 72000
        state.hod_time = datetime.now() - timedelta(minutes=90)  # 90 minutes ago (>= DRIFT_MIN_AGE_MIN=60)

        # 2%+ drop from HOD over 60+ minutes (DRIFT_DROP_PCT = 0.02)
        price = 70560  # 2% drop
        bar_time = datetime.now()

        result = detect_drift(state, price, bar_time)

        assert result is True

    def test_drift_not_detected_drop_too_small(self):
        """Test drift not detected with small drop."""
        state = SymbolState(code="005930")
        state.hod = 72000
        state.hod_time = datetime.now() - timedelta(minutes=90)

        # Only 1% drop (below 2% threshold DRIFT_DROP_PCT)
        price = 71280  # 1% drop
        bar_time = datetime.now()

        result = detect_drift(state, price, bar_time)

        assert result is False

    def test_drift_not_detected_too_recent(self):
        """Test drift not detected when HOD too recent."""
        state = SymbolState(code="005930")
        state.hod = 72000
        state.hod_time = datetime.now() - timedelta(minutes=30)  # < 60 minutes (DRIFT_MIN_AGE_MIN)

        price = 70560  # 2% drop
        bar_time = datetime.now()

        result = detect_drift(state, price, bar_time)

        assert result is False


class TestDetectSetup:
    """Tests for setup detection."""

    @pytest.fixture
    def state(self):
        """Create symbol state for testing."""
        s = SymbolState(code="005930")
        s.hod = 72000
        s.hod_time = datetime.now() - timedelta(minutes=10)  # Within PANIC_MAX_AGE_MIN (15 min)
        s.lod = 68000
        return s

    @pytest.fixture
    def mock_switches(self):
        """Create mock switches for VWAP depth check."""
        switches = MagicMock()
        switches.vwap_depth_min = 0.02  # 2% min (VWAP_DEPTH_MIN)
        switches.vwap_depth_max = 0.05  # 5% max (VWAP_DEPTH_MAX)
        switches.log_would_block = MagicMock()
        return switches

    def test_panic_setup_detected(self, state, mock_switches):
        """Test panic setup is detected."""
        # Price = 69840 (3% drop from HOD 72000, within PANIC_DROP_PCT)
        # VWAP = 71500, depth = (71500 - 69840) / 71500 = 2.3% (within 2-5% band)
        bar = {
            "close": 69840,
            "high": 69900,
            "low": 69700,
        }
        vwap = 71500
        bar_time = datetime.now()

        with patch("strategy_kpr.core.setup_detection.kpr_switches", mock_switches):
            result = detect_setup(state, bar, vwap, bar_time)

        assert result is True
        assert state.setup_type == "panic"
        assert state.setup_low == 68000
        assert state.reclaim_level is not None
        assert state.stop_level is not None
        assert state.setup_time == bar_time

    def test_drift_setup_detected(self, state, mock_switches):
        """Test drift setup is detected."""
        state.hod_time = datetime.now() - timedelta(minutes=90)  # Old HOD (>= DRIFT_MIN_AGE_MIN)

        # 2% drop from HOD (meets DRIFT_DROP_PCT), within VWAP band
        bar = {
            "close": 70560,  # 2% drop
            "high": 70600,
            "low": 70500,
        }
        vwap = 72000  # Depth = (72000 - 70560) / 72000 = 2.0% (at min boundary)
        bar_time = datetime.now()

        with patch("strategy_kpr.core.setup_detection.kpr_switches", mock_switches):
            result = detect_setup(state, bar, vwap, bar_time)

        assert result is True
        assert state.setup_type == "drift"

    def test_no_setup_without_drop(self, state, mock_switches):
        """Test no setup without sufficient drop."""
        bar = {
            "close": 71280,  # Only 1% drop (below both thresholds)
            "high": 71300,
            "low": 71200,
        }
        vwap = 72700  # Depth = (72700 - 71280) / 72700 = 2.0%
        bar_time = datetime.now()

        with patch("strategy_kpr.core.setup_detection.kpr_switches", mock_switches):
            result = detect_setup(state, bar, vwap, bar_time)

        assert result is False

    def test_no_setup_outside_vwap_band(self, state, mock_switches):
        """Test no setup when outside VWAP band."""
        bar = {
            "close": 69840,  # 3% drop (would trigger panic)
            "high": 69900,
            "low": 69700,
        }
        vwap = 75000  # Too far - depth = (75000 - 69840) / 75000 = 6.9% (> 5% max)
        bar_time = datetime.now()

        with patch("strategy_kpr.core.setup_detection.kpr_switches", mock_switches):
            result = detect_setup(state, bar, vwap, bar_time)

        assert result is False

    def test_setup_sets_reclaim_and_stop(self, state, mock_switches):
        """Test setup sets reclaim and stop levels."""
        bar = {
            "close": 69840,
            "high": 69900,
            "low": 69700,
        }
        vwap = 71500
        bar_time = datetime.now()

        with patch("strategy_kpr.core.setup_detection.kpr_switches", mock_switches):
            detect_setup(state, bar, vwap, bar_time)

        # Reclaim = LOD * (1 + offset)
        assert state.reclaim_level > state.setup_low
        # Stop = LOD * (1 - buffer)
        assert state.stop_level < state.setup_low

    def test_setup_clears_accept_closes(self, state, mock_switches):
        """Test setup clears accept_closes."""
        state.accept_closes = 5

        bar = {
            "close": 69840,
            "high": 69900,
            "low": 69700,
        }
        vwap = 71500
        bar_time = datetime.now()

        with patch("strategy_kpr.core.setup_detection.kpr_switches", mock_switches):
            detect_setup(state, bar, vwap, bar_time)

        assert state.accept_closes == 0
