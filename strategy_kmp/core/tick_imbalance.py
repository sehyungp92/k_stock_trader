"""Tick imbalance calculation with bucketed storage.

Uses 1-second buckets for O(window_sec) memory instead of O(ticks).
Implements tick-rule classification: uptick=buy, downtick=sell.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImbalanceBucket:
    """One-second bucket of buy/sell value."""

    ts_sec: int
    buy_val: float = 0.0
    sell_val: float = 0.0


class TickImbalance:
    """Tick-rule imbalance calculator with 1-second buckets.

    Efficient implementation that aggregates ticks into second-buckets
    rather than storing every individual tick.

    Attributes:
        window_sec: Rolling window size in seconds (60-120 per spec).
        buckets: Deque of ImbalanceBucket for each second.
        last_px: Last tick price for direction classification.
        last_dir: Last non-zero direction (+1 buy, -1 sell).
    """

    def __init__(self, window_sec: int = 90):
        """Initialize tick imbalance calculator.

        Args:
            window_sec: Rolling window in seconds. Spec allows 60-120s.
        """
        self.window_sec = window_sec
        self.buckets: deque[ImbalanceBucket] = deque(maxlen=300)
        self.last_px: Optional[float] = None
        self.last_dir: int = 0

    def update(self, ts: float, price: float, volume: float) -> None:
        """Update from a tick event.

        Args:
            ts: Unix timestamp of tick.
            price: Trade price.
            volume: Trade volume (shares).
        """
        if price <= 0 or volume <= 0:
            return

        ts_sec = int(ts)
        val = price * volume

        # Tick rule classification
        if self.last_px is None:
            d = 0
        elif price > self.last_px:
            d = +1  # uptick = buy-initiated
        elif price < self.last_px:
            d = -1  # downtick = sell-initiated
        else:
            d = self.last_dir  # zero-tick inherits last direction

        if d != 0:
            self.last_dir = d
        self.last_px = price

        # Add to current bucket or create new
        if not self.buckets or self.buckets[-1].ts_sec != ts_sec:
            self.buckets.append(ImbalanceBucket(ts_sec=ts_sec))

        b = self.buckets[-1]
        if d > 0:
            b.buy_val += val
        elif d < 0:
            b.sell_val += val

    def compute(self, now_ts: float) -> float:
        """Compute imbalance ratio over rolling window.

        Args:
            now_ts: Current unix timestamp.

        Returns:
            Imbalance ratio in [-1, +1]. Positive = buy pressure.
        """
        cutoff = int(now_ts) - self.window_sec
        buy = sell = 0.0

        for b in reversed(self.buckets):
            if b.ts_sec < cutoff:
                break
            buy += b.buy_val
            sell += b.sell_val

        total = buy + sell
        if total <= 0:
            return 0.0
        return (buy - sell) / total

    def reset(self) -> None:
        """Clear all buckets."""
        self.buckets.clear()
        self.last_px = None
        self.last_dir = 0
