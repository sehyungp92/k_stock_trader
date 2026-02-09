"""Intraday VWAP Computation."""

from dataclasses import dataclass
from typing import List

from ..config.constants import BUCKET_B


@dataclass
class VWAPState:
    """VWAP computation state."""
    cum_value: float = 0.0
    cum_volume: float = 0.0

    @property
    def vwap(self) -> float:
        if self.cum_volume <= 0:
            return 0.0
        return self.cum_value / self.cum_volume

    def update(self, bar: dict) -> float:
        """Update VWAP with new bar. Returns current VWAP."""
        high = float(bar.get('high', 0))
        low = float(bar.get('low', 0))
        close = float(bar.get('close', 0))
        volume = float(bar.get('volume', 0))

        if volume <= 0:
            return self.vwap

        typical = (high + low + close) / 3.0
        self.cum_value += typical * volume
        self.cum_volume += volume
        return self.vwap


def compute_vwap_series(bars_1m: List[dict]) -> List[float]:
    """Compute VWAP series from 1-minute bars."""
    state = VWAPState()
    return [state.update(bar) for bar in bars_1m]


def check_vwap_touch(bar: dict, vwap: float) -> bool:
    """Check if bar touched VWAP within tolerance."""
    tol = BUCKET_B["VWAP_TOUCH_TOL"]
    low = float(bar.get('low', 0))
    high = float(bar.get('high', 0))
    return low <= vwap * (1 + tol) and high >= vwap * (1 - tol)


def check_vwap_reclaim(
    prev_bar: dict, prev_vwap: float,
    curr_bar: dict, curr_vwap: float,
) -> bool:
    """
    Check if price reclaimed VWAP.
    Reclaim: prev_close < prev_VWAP AND curr_close > curr_VWAP + buffer
    """
    buffer = BUCKET_B["VWAP_RECLAIM_BUFFER"]
    prev_close = float(prev_bar.get('close', 0))
    curr_close = float(curr_bar.get('close', 0))

    was_below = prev_close < prev_vwap
    is_above = curr_close > curr_vwap
    has_buffer = curr_close >= curr_vwap * (1 + buffer)
    return was_below and is_above and has_buffer
