"""Technical indicators for KIS strategies."""

from __future__ import annotations
from typing import List, Sequence
from collections import deque
import math


def sma(values: Sequence[float], period: int) -> List[float]:
    """Simple Moving Average."""
    if len(values) < period:
        return []
    result = []
    window_sum = sum(values[:period])
    result.append(window_sum / period)
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        result.append(window_sum / period)
    return result


def ema(values: Sequence[float], period: int) -> List[float]:
    """Exponential Moving Average."""
    if not values:
        return []
    multiplier = 2 / (period + 1)
    result = [values[0]]
    for i in range(1, len(values)):
        result.append((values[i] - result[-1]) * multiplier + result[-1])
    return result


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> List[float]:
    """Average True Range."""
    if len(highs) < 2:
        return []
    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        true_ranges.append(tr)
    return sma(true_ranges, period)


def zscore(values: Sequence[float], lookback: int = 20) -> List[float]:
    """Z-score normalization."""
    if len(values) < lookback:
        return []
    result = []
    for i in range(lookback - 1, len(values)):
        window = values[i - lookback + 1:i + 1]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        result.append((values[i] - mean) / std)
    return result


def percentile_rank(value: float, distribution: Sequence[float]) -> float:
    """Calculate percentile rank of value in distribution."""
    if not distribution:
        return 50.0
    count_below = sum(1 for x in distribution if x < value)
    return (count_below / len(distribution)) * 100


class RollingSMA:
    """Efficient rolling SMA for streaming updates."""
    def __init__(self, period: int):
        self.period = period
        self.values: deque = deque(maxlen=period)
        self._sum = 0.0

    def update(self, value: float) -> float | None:
        if len(self.values) >= self.period:
            self._sum -= self.values[0]
        self.values.append(value)
        self._sum += value
        return self._sum / self.period if len(self.values) >= self.period else None


class RollingATR:
    """Rolling Average True Range."""
    def __init__(self, period: int = 14):
        self.period = period
        self.prev_close = None
        self.tr_values: deque = deque(maxlen=period)

    def update_bar(self, high: float, low: float, close: float) -> float | None:
        if self.prev_close is not None:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        else:
            tr = high - low
        self.prev_close = close
        self.tr_values.append(tr)
        return sum(self.tr_values) / len(self.tr_values) if len(self.tr_values) >= self.period else None
