"""Bar aggregation utilities."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, List, Optional
from collections import deque


@dataclass
class Bar:
    """OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarAggregator:
    """Aggregates ticks or smaller bars into larger timeframe bars."""

    def __init__(self, interval_minutes: int = 1, max_bars: int = 500):
        self.interval = timedelta(minutes=interval_minutes)
        self.max_bars = max_bars
        self._current_bar: Optional[Bar] = None
        self._completed_bars: Deque[Bar] = deque(maxlen=max_bars)
        self._bar_start: Optional[datetime] = None

    def update_tick(self, ts: datetime, price: float, volume: float) -> Optional[Bar]:
        interval_min = self.interval.seconds // 60
        bar_start = ts.replace(
            minute=(ts.minute // interval_min) * interval_min,
            second=0, microsecond=0
        )
        completed = None

        if self._bar_start is None or bar_start > self._bar_start:
            if self._current_bar is not None:
                self._completed_bars.append(self._current_bar)
                completed = self._current_bar
            self._bar_start = bar_start
            self._current_bar = Bar(bar_start, price, price, price, price, volume)
        else:
            if self._current_bar:
                self._current_bar.high = max(self._current_bar.high, price)
                self._current_bar.low = min(self._current_bar.low, price)
                self._current_bar.close = price
                self._current_bar.volume += volume
        return completed

    def get_completed_bars(self, n: int = 0) -> List[Bar]:
        bars = list(self._completed_bars)
        return bars[-n:] if n > 0 else bars


def aggregate_bars(bars: List[dict], target_minutes: int) -> List[Bar]:
    """Aggregate smaller bars into larger timeframe."""
    if not bars:
        return []

    result, current, current_start = [], None, None

    for bar in bars:
        ts = bar.get('timestamp')
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        bar_start = ts.replace(
            minute=(ts.minute // target_minutes) * target_minutes,
            second=0, microsecond=0
        )

        if current_start is None or bar_start > current_start:
            if current:
                result.append(current)
            current_start = bar_start
            current = Bar(bar_start, float(bar['open']), float(bar['high']),
                          float(bar['low']), float(bar['close']), float(bar['volume']))
        else:
            if current:
                current.high = max(current.high, float(bar['high']))
                current.low = min(current.low, float(bar['low']))
                current.close = float(bar['close'])
                current.volume += float(bar['volume'])

    if current:
        result.append(current)
    return result
