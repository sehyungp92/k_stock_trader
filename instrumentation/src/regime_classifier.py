"""
Regime Classifier — deterministic market regime tagger adapted for KRX/KIS.

Classifies the current market regime for a given symbol using standard
technical indicators (MA, ADX, ATR). Not a machine learning model -- just
rules-based classification for consistent, explainable labels.

Regime definitions:
    trending_up:    Price above 50-period MA, ADX > 25, MA slope positive
    trending_down:  Price below 50-period MA, ADX > 25, MA slope negative
    ranging:        ADX < 20, ATR percentile below 60th
    volatile:       ATR percentile above 80th, regardless of trend

ADAPTATION for KIS API:
    KIS returns daily OHLCV bars via ``data_provider.get_daily_bars(symbol, days=N)``,
    which yields a pandas DataFrame with columns [date, open, high, low, close, volume]
    sorted ascending by date.  Arbitrary intraday timeframes are not available from
    the KIS REST API, so this classifier always uses daily bars.  The ``timeframe``
    parameter is accepted for interface compatibility but ignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # graceful degradation

try:
    import pandas as pd
except ImportError:
    pd = None


logger = logging.getLogger("regime_classifier")


@dataclass
class RegimeConfig:
    """Thresholds for regime classification. Loaded from YAML."""
    ma_period: int = 50
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    atr_period: int = 14
    atr_volatile_percentile: float = 80.0
    atr_lookback_bars: int = 100     # bars to compute ATR percentile over
    slope_lookback_bars: int = 5      # bars to compute MA slope


class RegimeClassifier:
    """
    Deterministic market regime classifier adapted for KRX/KIS.

    Usage:
        from kis_core import KoreaInvestAPI
        api = KoreaInvestAPI(config)
        classifier = RegimeClassifier(
            config_path="instrumentation/config/regime_classifier_config.yaml",
            data_provider=api,
        )
        regime = classifier.classify("005930")  # Samsung Electronics
        # Returns: "trending_up" | "trending_down" | "ranging" | "volatile" | "unknown"
    """

    def __init__(
        self,
        config_path: str = "instrumentation/config/regime_classifier_config.yaml",
        data_provider=None,
    ):
        """
        Args:
            config_path: path to regime classifier config YAML
            data_provider: KIS API client instance (KoreaInvestAPI).
                Must support: get_daily_bars(symbol, days=N) -> pd.DataFrame
                with columns [date, open, high, low, close, volume].
        """
        self.data_provider = data_provider
        self.config = self._load_config(config_path)
        self._cache: dict[str, str] = {}  # symbol -> regime

    def _load_config(self, path: str) -> RegimeConfig:
        config_file = Path(path)
        if config_file.exists() and yaml is not None:
            try:
                with open(config_file) as f:
                    raw = yaml.safe_load(f) or {}
                return RegimeConfig(**{k: v for k, v in raw.items() if k in RegimeConfig.__dataclass_fields__})
            except Exception as e:
                logger.warning("Failed to load regime config from %s: %s — using defaults", path, e)
        return RegimeConfig()

    def classify(self, symbol: str, timeframe: str = "1h") -> str:
        """
        Classify the current market regime for a symbol.

        Args:
            symbol: KRX stock code (e.g. "005930")
            timeframe: accepted for interface compatibility but ignored.
                KIS API only provides daily bars, so daily data is always used.

        Returns one of: "trending_up", "trending_down", "ranging", "volatile", "unknown"
        """
        try:
            candles = self._fetch_candles(symbol, timeframe)
            if not candles or len(candles) < self.config.atr_lookback_bars:
                return "unknown"

            closes = [c["close"] for c in candles]
            highs = [c["high"] for c in candles]
            lows = [c["low"] for c in candles]

            # Current price vs MA
            ma = sum(closes[-self.config.ma_period:]) / self.config.ma_period
            current_price = closes[-1]
            above_ma = current_price > ma

            # MA slope (positive = up, negative = down)
            ma_recent = sum(closes[-self.config.slope_lookback_bars:]) / self.config.slope_lookback_bars
            ma_prior = sum(closes[-(self.config.slope_lookback_bars + 5):-5]) / self.config.slope_lookback_bars
            ma_slope_positive = ma_recent > ma_prior

            # ADX (simplified: using directional movement)
            adx = self._compute_adx(highs, lows, closes, self.config.adx_period)

            # ATR percentile
            atrs = self._compute_atr_series(highs, lows, closes, self.config.atr_period)
            current_atr = atrs[-1] if atrs else 0
            atr_percentile = self._percentile_rank(atrs, current_atr)

            # Classification logic
            if atr_percentile >= self.config.atr_volatile_percentile:
                regime = "volatile"
            elif adx >= self.config.adx_trend_threshold:
                if above_ma and ma_slope_positive:
                    regime = "trending_up"
                elif not above_ma and not ma_slope_positive:
                    regime = "trending_down"
                else:
                    regime = "ranging"  # conflicting signals
            elif adx < self.config.adx_range_threshold:
                regime = "ranging"
            else:
                # ADX between range and trend thresholds
                if above_ma and ma_slope_positive:
                    regime = "trending_up"
                elif not above_ma and not ma_slope_positive:
                    regime = "trending_down"
                else:
                    regime = "ranging"

            self._cache[symbol] = regime
            return regime

        except Exception as e:
            logger.warning("Regime classification failed for %s: %s", symbol, e)
            return self._cache.get(symbol, "unknown")

    def current_regime(self, symbol: str) -> str:
        """Get the most recently computed regime (cached)."""
        return self._cache.get(symbol, "unknown")

    def classify_multi_tf(self, symbol: str) -> dict:
        """Classify regime across multiple timeframes.

        Returns dict with primary (daily 50-MA), higher_tf (200-MA proxy for
        weekly trend), and sector_regime (placeholder).
        """
        primary = self.classify(symbol)
        higher_tf = "unknown"
        try:
            candles = self._fetch_candles(symbol, "1d")
            if candles and len(candles) >= 200:
                closes = [c["close"] for c in candles]
                ma200 = sum(closes[-200:]) / 200
                ma200_prev = sum(closes[-206:-6]) / 200
                price = closes[-1]
                slope_positive = ma200 > ma200_prev
                if price > ma200 and slope_positive:
                    higher_tf = "trending_up"
                elif price < ma200 and not slope_positive:
                    higher_tf = "trending_down"
                else:
                    higher_tf = "ranging"
        except Exception:
            pass

        return {
            "primary_regime": primary,
            "higher_tf_regime": higher_tf,
            "sector_regime": "unknown",
        }

    def _fetch_candles(self, symbol: str, timeframe: str) -> list:
        """
        Fetch daily OHLCV bars from the KIS API.

        ADAPTATION: KIS API returns daily bars via get_daily_bars(symbol, days=N).
        The timeframe parameter is accepted for interface compatibility but always
        uses daily data since KIS doesn't provide arbitrary timeframe OHLCV.

        Returns list of dicts with keys: date, open, high, low, close, volume.
        """
        if self.data_provider is None:
            logger.warning("No data_provider configured for regime classifier")
            return []

        limit = max(self.config.atr_lookback_bars, self.config.ma_period) + 20

        try:
            df = self.data_provider.get_daily_bars(symbol, days=limit)

            # Handle both DataFrame and None/empty returns
            if df is None:
                return []
            if pd is not None and isinstance(df, pd.DataFrame):
                if df.empty:
                    return []
                return df.to_dict("records")
            # If pandas not available but data_provider returned something else
            if isinstance(df, list):
                return df
            return []
        except Exception as e:
            logger.warning("Failed to fetch candles for %s: %s", symbol, e)
            return []

    def _compute_adx(self, highs: list, lows: list, closes: list, period: int) -> float:
        """Simplified ADX calculation using Wilder's directional movement."""
        if len(highs) < period + 1:
            return 0

        plus_dm = []
        minus_dm = []
        tr = []

        for i in range(1, len(highs)):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]

            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)

            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr.append(max(hl, hc, lc))

        # Smoothed averages (Wilder's method)
        def wilder_smooth(data: list, n: int) -> list:
            if len(data) < n:
                return [0]
            result = [sum(data[:n]) / n]
            for i in range(n, len(data)):
                result.append((result[-1] * (n - 1) + data[i]) / n)
            return result

        smoothed_tr = wilder_smooth(tr, period)
        smoothed_plus = wilder_smooth(plus_dm, period)
        smoothed_minus = wilder_smooth(minus_dm, period)

        if not smoothed_tr or smoothed_tr[-1] == 0:
            return 0

        plus_di = smoothed_plus[-1] / smoothed_tr[-1] * 100
        minus_di = smoothed_minus[-1] / smoothed_tr[-1] * 100

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0

        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx  # single-point DX as ADX approximation

    def _compute_atr_series(self, highs: list, lows: list, closes: list, period: int) -> list:
        """Compute ATR series using Wilder's smoothing."""
        if len(highs) < period + 1:
            return []
        trs = []
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            trs.append(max(hl, hc, lc))

        atrs = []
        if len(trs) >= period:
            atr = sum(trs[:period]) / period
            atrs.append(atr)
            for i in range(period, len(trs)):
                atr = (atr * (period - 1) + trs[i]) / period
                atrs.append(atr)
        return atrs

    def _percentile_rank(self, series: list, value: float) -> float:
        """Compute percentile rank of value within the lookback window."""
        if not series:
            return 50.0
        lookback = series[-self.config.atr_lookback_bars:]
        count_below = sum(1 for v in lookback if v < value)
        return (count_below / len(lookback)) * 100
