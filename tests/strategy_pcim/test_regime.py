"""Tests for PCIM regime calculation -- calls actual source functions."""

import pytest

from strategy_pcim.premarket.regime import compute_regime, RegimeResult


class TestComputeRegime:
    """Tests for compute_regime using KOSPI-based regime classification."""

    def test_insufficient_data_defaults_normal(self):
        """< 50 data points defaults to NORMAL regime."""
        result = compute_regime([100.0] * 30)
        assert result.name == "NORMAL"
        assert result.max_exposure == 0.80
        assert result.disable_bucket_a is False
        assert result.value == 0.0

    def test_strong_regime(self):
        """Consistently rising prices -> regime_value >> 2 -> STRONG."""
        closes = [100 + i * 2 for i in range(100)]
        result = compute_regime(closes)
        assert result.name == "STRONG"
        assert result.max_exposure == 1.0
        assert result.disable_bucket_a is False
        assert result.value > 2.0

    def test_crisis_regime(self):
        """Sharp declining prices -> regime_value << -2 -> CRISIS."""
        closes = [200 - i * 3 for i in range(100)]
        result = compute_regime(closes)
        assert result.name == "CRISIS"
        assert result.max_exposure == 0.20
        assert result.disable_bucket_a is True
        assert result.value < -2.0

    def test_normal_regime(self):
        """Flat market with small noise -> regime_value near 0-2."""
        import random
        random.seed(42)
        closes = [100 + random.uniform(-0.5, 0.5) for _ in range(100)]
        result = compute_regime(closes)
        # Flat market gives regime_value ~1.12, which is >= 0 and < 2 -> NORMAL
        assert result.name == "NORMAL"
        assert result.max_exposure == 0.80
        assert result.disable_bucket_a is False

    def test_weak_regime(self):
        """Mildly declining market -> regime_value between -2 and 0 -> WEAK."""
        import random
        random.seed(3)
        closes = [100.0]
        for _ in range(99):
            closes.append(closes[-1] + random.gauss(-0.02, 0.5))
        result = compute_regime(closes)
        assert result.name == "WEAK"
        assert result.max_exposure == 0.50
        assert result.disable_bucket_a is True
        assert -2.0 <= result.value < 0.0

    def test_exactly_50_data_points(self):
        """Exactly 50 data points should compute regime (not default)."""
        closes = [100 + i for i in range(50)]
        result = compute_regime(closes)
        # Should not default -- actually computes
        assert isinstance(result, RegimeResult)
        assert result.value != 0.0 or result.name != "NORMAL"  # Computed, not defaulted

    def test_regime_result_fields(self):
        """RegimeResult has all expected fields."""
        result = compute_regime([100.0] * 60)
        assert hasattr(result, "name")
        assert hasattr(result, "value")
        assert hasattr(result, "max_exposure")
        assert hasattr(result, "disable_bucket_a")

    def test_constant_prices_near_zero_regime(self):
        """All identical prices -> ATR approximated, regime near zero."""
        closes = [100.0] * 60
        result = compute_regime(closes)
        # ATR would be 0, falls back to 1% of last_close
        # SMA50 == last_close -> regime_value = 0 -> classified as NORMAL or WEAK
        assert result.name in ("NORMAL", "WEAK")
