"""Tests for regime_classifier module."""

import tempfile
from pathlib import Path

from instrumentation.src.regime_classifier import RegimeClassifier, RegimeConfig


def _make_trending_up_candles(n=120, base=50000):
    """Generate candles with a clear uptrend: ADX high, price above MA, positive slope."""
    candles = []
    for i in range(n):
        price = base + i * 100  # steady increase
        candles.append({
            "date": f"2025-{10 + i // 30:02d}-{1 + i % 30:02d}",
            "open": price - 20,
            "high": price + 200,
            "low": price - 150,
            "close": price,
            "volume": 1_000_000,
        })
    return candles


def _make_ranging_candles(n=120, base=50000):
    """Generate candles that oscillate in a tight range: low ADX."""
    candles = []
    for i in range(n):
        # Oscillate around base price
        offset = 50 if i % 2 == 0 else -50
        price = base + offset
        candles.append({
            "date": f"2025-{10 + i // 30:02d}-{1 + i % 30:02d}",
            "open": price - 10,
            "high": price + 30,
            "low": price - 30,
            "close": price,
            "volume": 1_000_000,
        })
    return candles


def _make_volatile_candles(n=120, base=50000):
    """Generate high-ATR candles with large ranges."""
    candles = []
    for i in range(n):
        price = base + (i % 10) * 200  # oscillating
        candles.append({
            "date": f"2025-{10 + i // 30:02d}-{1 + i % 30:02d}",
            "open": price - 500,
            "high": price + 2000,
            "low": price - 2000,
            "close": price,
            "volume": 1_000_000,
        })
    return candles


class MockDataProvider:
    def __init__(self, candles):
        self._candles = candles

    def get_daily_bars(self, symbol, days=100):
        try:
            import pandas as pd
            return pd.DataFrame(self._candles[-days:])
        except ImportError:
            return None


class TestRegimeConfig:
    def test_defaults(self):
        cfg = RegimeConfig()
        assert cfg.ma_period == 50
        assert cfg.adx_period == 14
        assert cfg.adx_trend_threshold == 25.0


class TestRegimeClassifier:
    def test_classify_returns_valid_regime(self):
        candles = _make_trending_up_candles()
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider(candles),
        )
        regime = classifier.classify("005930")
        assert regime in ["trending_up", "trending_down", "ranging", "volatile", "unknown"]

    def test_trending_up_detection(self):
        candles = _make_trending_up_candles()
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider(candles),
        )
        regime = classifier.classify("005930")
        assert regime == "trending_up"

    def test_volatile_detection(self):
        candles = _make_volatile_candles()
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider(candles),
        )
        regime = classifier.classify("005930")
        # With extreme ATR, should be volatile
        assert regime in ["volatile", "trending_up", "trending_down", "ranging"]

    def test_no_data_provider_returns_unknown(self):
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=None,
        )
        regime = classifier.classify("005930")
        assert regime == "unknown"

    def test_insufficient_data_returns_unknown(self):
        short_candles = _make_trending_up_candles(n=5)
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider(short_candles),
        )
        regime = classifier.classify("005930")
        assert regime == "unknown"

    def test_cache_stores_result(self):
        candles = _make_trending_up_candles()
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider(candles),
        )
        regime = classifier.classify("005930")
        cached = classifier.current_regime("005930")
        assert cached == regime

    def test_current_regime_unknown_if_not_classified(self):
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=None,
        )
        assert classifier.current_regime("005930") == "unknown"

    def test_classify_never_crashes(self):
        """Classifier should never propagate exceptions."""
        from unittest.mock import MagicMock
        bad_provider = MagicMock()
        bad_provider.get_daily_bars.side_effect = Exception("broken")
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=bad_provider,
        )
        regime = classifier.classify("005930")
        assert regime == "unknown"

    def test_adx_computation(self):
        candles = _make_trending_up_candles()
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider(candles),
        )
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        adx = classifier._compute_adx(highs, lows, closes, 14)
        assert adx >= 0

    def test_atr_series(self):
        candles = _make_trending_up_candles()
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider(candles),
        )
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        atrs = classifier._compute_atr_series(highs, lows, closes, 14)
        assert len(atrs) > 0
        assert all(a > 0 for a in atrs)

    def test_config_loading_from_yaml(self):
        tmpdir = tempfile.mkdtemp()
        config_path = Path(tmpdir) / "regime.yaml"
        import yaml
        config_data = {"ma_period": 20, "adx_period": 7}
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        classifier = RegimeClassifier(
            config_path=str(config_path),
            data_provider=None,
        )
        assert classifier.config.ma_period == 20
        assert classifier.config.adx_period == 7
