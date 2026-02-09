"""VWAP calculations for KIS strategies."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional


@dataclass
class VWAPLedger:
    """Maintains cumulative VWAP that survives tier transitions."""
    cum_vol: float = 0.0
    cum_pv: float = 0.0
    anchor_date: Optional[date] = None

    @property
    def vwap(self) -> float:
        if self.cum_vol <= 0:
            return 0.0
        return self.cum_pv / self.cum_vol

    def update_from_tick(self, price: float, volume: float) -> None:
        if volume <= 0 or price <= 0:
            return
        self.cum_vol += volume
        self.cum_pv += price * volume

    def update_from_bar(self, bar: dict) -> None:
        high = float(bar.get('high', 0))
        low = float(bar.get('low', 0))
        close = float(bar.get('close', 0))
        volume = float(bar.get('volume', 0))
        if volume <= 0:
            return
        typical = (high + low + close) / 3.0
        self.cum_vol += volume
        self.cum_pv += typical * volume

    def reset(self, anchor: Optional[date] = None) -> None:
        self.cum_vol = 0.0
        self.cum_pv = 0.0
        self.anchor_date = anchor


def compute_anchored_daily_vwap(daily_bars: List[dict], anchor_date: date) -> float:
    """Compute anchored VWAP from daily bars (v1 approximation)."""
    cum_vol, cum_pv = 0.0, 0.0

    for bar in daily_bars:
        bar_date = bar.get('date')
        if isinstance(bar_date, str):
            bar_date = datetime.strptime(bar_date, "%Y%m%d").date()
        if bar_date < anchor_date:
            continue

        high = float(bar.get('high', bar.get('stck_hgpr', 0)))
        low = float(bar.get('low', bar.get('stck_lwpr', 0)))
        close = float(bar.get('close', bar.get('stck_clpr', 0)))
        volume = float(bar.get('volume', bar.get('acml_vol', 0)))

        if volume <= 0:
            continue
        typical = (high + low + close) / 3.0
        cum_vol += volume
        cum_pv += typical * volume

    return cum_pv / cum_vol if cum_vol > 0 else 0.0


def vwap_band(vwap: float, band_pct: float = 0.005) -> tuple:
    """Calculate VWAP band. Returns (lower, upper)."""
    return (vwap * (1 - band_pct), vwap * (1 + band_pct))
