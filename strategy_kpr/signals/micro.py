"""KPR MicroPressure: tick-level uptick/downtick for HOT, bar proxy for WARM/COLD."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict

from ..config.constants import (
    VOL_SURGE_THRESHOLD, BAR_STRENGTH_BULL, BAR_STRENGTH_BEAR,
    MICRO_LOOKBACK_BARS,
)

# Tick-level classification thresholds
TICK_IMBALANCE_BULL = 0.15   # Net uptick ratio for ACCUMULATE
TICK_IMBALANCE_BEAR = -0.15  # Net uptick ratio for DISTRIBUTE
TICK_WINDOW = 60             # Ticks to keep in rolling window


class MicroSignal(Enum):
    ACCUMULATE = auto()
    NEUTRAL = auto()
    DISTRIBUTE = auto()


@dataclass
class _TickAccumulator:
    """Per-symbol tick-level uptick/downtick tracker."""
    prev_price: float = 0.0
    upticks: int = 0
    downticks: int = 0
    uptick_vol: float = 0.0
    downtick_vol: float = 0.0
    # Rolling window of (direction, volume) pairs for recent bias
    window: deque = field(default_factory=lambda: deque(maxlen=TICK_WINDOW))

    def add_tick(self, price: float, volume: float) -> None:
        if self.prev_price > 0 and price != self.prev_price:
            if price > self.prev_price:
                self.upticks += 1
                self.uptick_vol += volume
                self.window.append((1, volume))
            else:
                self.downticks += 1
                self.downtick_vol += volume
                self.window.append((-1, volume))
        self.prev_price = price

    def imbalance(self) -> float:
        """Volume-weighted imbalance in [-1, 1]. Positive = buying pressure."""
        up_v = sum(v for d, v in self.window if d == 1)
        dn_v = sum(v for d, v in self.window if d == -1)
        total = up_v + dn_v
        if total <= 0:
            return 0.0
        return (up_v - dn_v) / total

    def classify(self, vol_surging: bool) -> MicroSignal:
        """Classify from tick imbalance."""
        imb = self.imbalance()
        if imb >= TICK_IMBALANCE_BULL:
            return MicroSignal.ACCUMULATE
        if imb <= TICK_IMBALANCE_BEAR and vol_surging:
            return MicroSignal.DISTRIBUTE
        return MicroSignal.NEUTRAL

    def reset(self) -> None:
        self.prev_price = 0.0
        self.upticks = 0
        self.downticks = 0
        self.uptick_vol = 0.0
        self.downtick_vol = 0.0
        self.window.clear()


class MicroPressureProvider:
    """Tick-level micro-pressure for HOT tier, bar proxy for WARM/COLD."""

    def __init__(self):
        self._vol_history: Dict[str, deque] = {}
        self._tick_accums: Dict[str, _TickAccumulator] = {}
        self._hot_tickers: set = set()  # Tickers with active tick feeds

    def on_tick(self, ticker: str, price: float, volume: float) -> None:
        """Process a real-time tick for HOT tier symbols."""
        acc = self._tick_accums.get(ticker)
        if acc is None:
            acc = _TickAccumulator()
            self._tick_accums[ticker] = acc
        acc.add_tick(price, volume)
        self._hot_tickers.add(ticker)

    def update(self, ticker: str, bar: dict) -> MicroSignal:
        """Classify micro pressure from a completed 1m bar.

        For HOT tier symbols with tick data, uses tick-level imbalance.
        For WARM/COLD tier, falls back to bar-strength proxy.
        """
        volume = float(bar.get('volume', 0))

        # Volume surge vs recent average (shared by both paths)
        if ticker not in self._vol_history:
            self._vol_history[ticker] = deque(maxlen=MICRO_LOOKBACK_BARS)
        hist = self._vol_history[ticker]
        avg_vol = sum(hist) / len(hist) if hist else volume
        vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0
        hist.append(volume)
        surging = vol_ratio >= VOL_SURGE_THRESHOLD

        # HOT tier: use tick-level imbalance if we have tick data
        acc = self._tick_accums.get(ticker)
        if ticker in self._hot_tickers and acc and len(acc.window) >= 10:
            signal = acc.classify(surging)
            # Reset accumulator for next bar window
            acc.reset()
            return signal

        # WARM/COLD fallback: bar-strength proxy
        high = float(bar.get('high', 0))
        low = float(bar.get('low', 0))
        open_price = float(bar.get('open', 0))
        close = float(bar.get('close', 0))

        if high <= low or volume <= 0:
            return MicroSignal.NEUTRAL

        bar_strength = (close - low) / (high - low)
        bullish_bar = bar_strength >= BAR_STRENGTH_BULL
        bearish_bar = bar_strength <= BAR_STRENGTH_BEAR

        if surging and bearish_bar:
            return MicroSignal.DISTRIBUTE
        if bullish_bar and (surging or close > open_price):
            return MicroSignal.ACCUMULATE
        return MicroSignal.NEUTRAL

    def demote(self, ticker: str) -> None:
        """Called when a symbol is demoted from HOT tier."""
        self._hot_tickers.discard(ticker)
        self._tick_accums.pop(ticker, None)
