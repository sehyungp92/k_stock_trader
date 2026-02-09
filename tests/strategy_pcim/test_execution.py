"""Tests for PCIM execution modules: bucket_a trigger, vetoes, gap_reversal, trend_gate."""

import pytest

from strategy_pcim.execution.bucket_a import check_bucket_a_trigger
from strategy_pcim.execution.vetoes import check_execution_veto
from strategy_pcim.pipeline.gap_reversal import compute_gap_reversal_rate
from strategy_pcim.pipeline.trend_gate import check_trend_gate
from strategy_pcim.config.switches import PCIMSwitches


# ===========================================================================
# Bucket A Trigger
# ===========================================================================

class TestBucketATrigger:
    """Tests for check_bucket_a_trigger opening range bar logic."""

    def test_zero_range_not_triggered(self):
        """Zero range bar (high == low) -> ZERO_RANGE."""
        bar = {"high": 100, "low": 100, "close": 100, "volume": 1000}
        result = check_bucket_a_trigger(bar, baseline_volume=500)
        assert result.triggered is False
        assert result.reason == "ZERO_RANGE"

    def test_close_not_strong_enough(self):
        """Close in bottom of range -> CLOSE_NOT_STRONG."""
        bar = {"high": 110, "low": 90, "close": 92, "volume": 1000}
        result = check_bucket_a_trigger(bar, baseline_volume=500)
        assert result.triggered is False
        assert "CLOSE_NOT_STRONG" in result.reason

    def test_volume_too_low(self):
        """Strong close but volume below threshold -> VOLUME_LOW."""
        bar = {"high": 110, "low": 90, "close": 108, "volume": 400}
        result = check_bucket_a_trigger(bar, baseline_volume=500)
        assert result.triggered is False
        assert "VOLUME_LOW" in result.reason

    def test_triggered(self):
        """Strong close + sufficient volume -> TRIGGERED."""
        bar = {"high": 110, "low": 90, "close": 108, "volume": 700}
        result = check_bucket_a_trigger(bar, baseline_volume=500)
        assert result.triggered is True
        assert result.reason == "TRIGGERED"
        assert result.bar == bar
        assert result.vol_ratio > 0

    def test_custom_vol_threshold(self):
        """Custom vol_threshold overrides config default."""
        bar = {"high": 110, "low": 90, "close": 108, "volume": 550}
        # Default threshold 1.20: 550/500=1.1 < 1.2 -> would fail
        # Custom threshold 1.0: 1.1 >= 1.0 -> passes
        result = check_bucket_a_trigger(bar, baseline_volume=500, vol_threshold=1.0)
        assert result.triggered is True

    def test_close_at_boundary(self):
        """Close exactly at top 30% boundary -> passes close check."""
        # Range 90-110, top 30% starts at 104 (i.e. close_pos >= 0.70)
        bar = {"high": 110, "low": 90, "close": 104, "volume": 700}
        result = check_bucket_a_trigger(bar, baseline_volume=500)
        assert result.triggered is True

    def test_close_just_below_boundary(self):
        """Close just below top 30% boundary -> fails close check."""
        # close_pos = (103 - 90) / 20 = 0.65 < 0.70
        bar = {"high": 110, "low": 90, "close": 103, "volume": 700}
        result = check_bucket_a_trigger(bar, baseline_volume=500)
        assert result.triggered is False
        assert "CLOSE_NOT_STRONG" in result.reason

    def test_vol_ratio_stored(self):
        """vol_ratio is stored on the result when volume is low."""
        bar = {"high": 110, "low": 90, "close": 108, "volume": 400}
        result = check_bucket_a_trigger(bar, baseline_volume=500)
        assert result.vol_ratio == pytest.approx(0.8)


# ===========================================================================
# Execution Vetoes
# ===========================================================================

class TestExecutionVeto:
    """Tests for check_execution_veto: VI, limit, spread checks."""

    def test_vi_veto(self):
        """Stock in VI -> IN_VI veto."""
        result = check_execution_veto(
            {"bid": 100, "ask": 101, "last": 100},
            upper_limit_price=110,
            tick_size=1,
            is_in_vi=True,
        )
        assert result == "IN_VI"

    def test_near_upper_limit(self):
        """Price within 2 ticks of upper limit -> NEAR_UPPER_LIMIT veto."""
        result = check_execution_veto(
            {"bid": 100, "ask": 101, "last": 109},
            upper_limit_price=110,
            tick_size=1,
            is_in_vi=False,
        )
        assert result is not None
        assert "NEAR_UPPER_LIMIT" in result

    def test_spread_too_wide(self):
        """Spread exceeding threshold -> SPREAD_TOO_WIDE veto."""
        switches = PCIMSwitches(spread_veto_pct=0.007)
        result = check_execution_veto(
            {"bid": 100, "ask": 101, "last": 100},
            upper_limit_price=200,
            tick_size=1,
            is_in_vi=False,
            switches=switches,
        )
        assert result is not None
        assert "SPREAD" in result

    def test_no_veto(self):
        """Normal conditions -> no veto (returns None)."""
        switches = PCIMSwitches(spread_veto_pct=0.02)
        result = check_execution_veto(
            {"bid": 99.5, "ask": 100.5, "last": 100},
            upper_limit_price=200,
            tick_size=1,
            is_in_vi=False,
            switches=switches,
        )
        assert result is None

    def test_vi_takes_priority(self):
        """VI check is done first, regardless of other conditions."""
        result = check_execution_veto(
            {"bid": 100, "ask": 200, "last": 109},  # Wide spread + near limit
            upper_limit_price=110,
            tick_size=1,
            is_in_vi=True,
        )
        assert result == "IN_VI"

    def test_upper_limit_before_spread(self):
        """Upper limit check is done before spread check."""
        switches = PCIMSwitches(spread_veto_pct=0.007)
        result = check_execution_veto(
            {"bid": 100, "ask": 200, "last": 109},  # Both near limit + wide spread
            upper_limit_price=110,
            tick_size=1,
            is_in_vi=False,
            switches=switches,
        )
        assert "NEAR_UPPER_LIMIT" in result

    def test_spread_narrow_passes(self):
        """Narrow spread passes even with strict threshold."""
        switches = PCIMSwitches(spread_veto_pct=0.006)
        result = check_execution_veto(
            {"bid": 99900, "ask": 100000, "last": 100000},  # 0.1% spread
            upper_limit_price=130000,
            tick_size=100,
            is_in_vi=False,
            switches=switches,
        )
        assert result is None


# ===========================================================================
# Gap Reversal Rate
# ===========================================================================

class TestGapReversalRate:
    """Tests for compute_gap_reversal_rate from daily bars."""

    def test_no_gaps(self):
        """No gap events -> event_count=0, rate=0.0, insufficient."""
        bars = [{"open": 100, "close": 100}] * 20
        result = compute_gap_reversal_rate(bars)
        assert result.event_count == 0
        assert result.rate == 0.0
        assert result.insufficient_sample is True

    def test_all_reversals(self):
        """All gap-up events reverse (close < open) -> rate=1.0."""
        bars = [{"open": 100, "close": 100}]  # prev
        for _ in range(15):
            bars.append({"open": 102, "close": 99})  # gap up + reversal
        result = compute_gap_reversal_rate(bars)
        assert result.event_count == 15
        assert result.reversal_count == 15
        assert result.rate == pytest.approx(1.0)
        assert result.insufficient_sample is False

    def test_no_reversals(self):
        """All gap-up events continue up (close > open) -> rate=0.0."""
        bars = [{"open": 100, "close": 100}]
        prev_close = 100
        for _ in range(15):
            gap_open = prev_close * 1.02  # 2% gap
            close = gap_open + 3  # continuation: close above open
            bars.append({"open": gap_open, "close": close})
            prev_close = close
        result = compute_gap_reversal_rate(bars)
        assert result.event_count == 15
        assert result.reversal_count == 0
        assert result.rate == pytest.approx(0.0)

    def test_insufficient_sample(self):
        """Fewer than MIN_EVENTS (10) gap events -> insufficient_sample=True."""
        bars = [{"open": 100, "close": 100}]
        for _ in range(5):  # Only 5 gap events
            bars.append({"open": 102, "close": 99})
        result = compute_gap_reversal_rate(bars)
        assert result.event_count == 5
        assert result.insufficient_sample is True

    def test_mixed_reversals(self):
        """Mix of reversals and continuations with properly chained bars."""
        bars = [{"open": 100, "close": 100}]
        prev_close = 100
        for i in range(20):
            gap_open = prev_close * 1.02  # 2% gap
            if i % 2 == 0:
                close = gap_open - 3   # reversal: close below open
            else:
                close = gap_open + 3   # continuation: close above open
            bars.append({"open": gap_open, "close": close})
            prev_close = close
        result = compute_gap_reversal_rate(bars)
        assert result.event_count == 20
        assert result.reversal_count == 10
        assert result.rate == pytest.approx(0.5)
        assert result.insufficient_sample is False

    def test_gap_below_threshold_ignored(self):
        """Gap < 1% is not counted as a gap event."""
        bars = [{"open": 100, "close": 100}]
        prev_close = 100
        for _ in range(15):
            gap_open = prev_close * 1.005  # 0.5% gap, below 1% threshold
            close = gap_open  # flat close
            bars.append({"open": gap_open, "close": close})
            prev_close = close
        result = compute_gap_reversal_rate(bars)
        assert result.event_count == 0


# ===========================================================================
# Trend Gate (20DMA)
# ===========================================================================

class TestTrendGate:
    """Tests for check_trend_gate 20DMA filter."""

    def test_above_sma20_passes(self):
        """Last close well above SMA20 -> passes."""
        closes = list(range(80, 100)) + [120]  # 21 values, last way above SMA20
        assert check_trend_gate(closes) is True

    def test_below_sma20_fails(self):
        """Last close below SMA20 -> fails."""
        closes = list(range(100, 120)) + [80]  # Last below SMA20
        assert check_trend_gate(closes) is False

    def test_insufficient_data(self):
        """< 20 data points -> fails (returns False)."""
        closes = [100] * 10
        assert check_trend_gate(closes) is False

    def test_exactly_20_data_points(self):
        """Exactly 20 data points should compute (not fail for insufficient)."""
        closes = list(range(80, 100))  # 20 values, rising
        # SMA20 of [80..99] = mean(80..99) = 89.5, last close = 99 > 89.5
        assert check_trend_gate(closes) is True

    def test_flat_prices_at_boundary(self):
        """All identical prices -> close == SMA20 -> fails (not strictly >)."""
        closes = [100.0] * 25
        assert check_trend_gate(closes) is False
