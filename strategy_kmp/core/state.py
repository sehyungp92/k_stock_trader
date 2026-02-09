"""
KMP Symbol State and FSM.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import math
from typing import Optional

from kis_core import BarAggregator, RollingATR

from .tick_imbalance import TickImbalance


class State(Enum):
    """KMP FSM states."""
    IDLE = auto()
    CANDIDATE = auto()
    WATCH_BREAK = auto()
    WAIT_ACCEPTANCE = auto()
    ARMED = auto()
    IN_POSITION = auto()
    PENDING_EXIT = auto()  # Exit order submitted, awaiting fill confirmation
    DONE = auto()


@dataclass
class SymbolState:
    """Per-symbol state for KMP strategy."""
    code: str
    fsm: State = State.IDLE

    # Sector metadata (loaded from universe_meta)
    sector: str = ""
    skip_reason: str = ""

    # Daily trend anchor
    sma20: float = 0.0
    sma60: float = 0.0
    prev_close: float = 0.0
    trend_ok: bool = False

    # Opening Range (09:00-09:15)
    or_high: float = -math.inf
    or_low: float = math.inf
    or_mid: float = 0.0
    or_locked: bool = False

    # VWAP (cumulative)
    cum_vol: float = 0.0
    cum_val: float = 0.0
    vwap: float = 0.0

    # Scan features
    value15: float = 0.0
    surge: float = 0.0

    # 1-minute volume
    avg_1m_vol: float = 0.0
    curr_1m_vol: float = 0.0
    rvol_1m: float = 0.0

    # Bid/Ask spread
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0

    # Acceptance tracking
    break_ts: float = 0.0
    retest_low: float = math.inf

    # VI tracking
    vi_ref: float = 0.0
    last_vi_ts: float = -math.inf

    # Tick imbalance (bucketed for efficiency)
    imb_calc: TickImbalance = field(default_factory=TickImbalance)
    imb: float = 0.0
    _prev_cum_vol: float = 0.0  # For computing trade_vol from cumulative delta

    # Position tracking
    entry_px: float = 0.0
    entry_ts: float = 0.0
    qty: int = 0
    structure_stop: float = 0.0
    hard_stop: float = 0.0
    max_fav: float = 0.0
    trail_px: float = 0.0
    pgm_regime_at_entry: str = "mixed"

    # Bar aggregation (populated by tick dispatch)
    bar_1m: BarAggregator = field(default_factory=lambda: BarAggregator(1))
    bar_5m: BarAggregator = field(default_factory=lambda: BarAggregator(5))
    rolling_atr: RollingATR = field(default_factory=lambda: RollingATR(period=14))
    atr_1m: Optional[float] = None
    last_5m_value: float = 0.0

    # Order tracking
    entry_order_id: str | None = None
    entry_armed_ts: float = 0.0

    def update_vwap(self, price: float, volume: float) -> None:
        """Update VWAP from tick."""
        self.cum_vol += volume
        self.cum_val += price * volume
        if self.cum_vol > 0:
            self.vwap = self.cum_val / self.cum_vol

    def update_spread(self) -> None:
        """Update spread from bid/ask."""
        if self.bid > 0 and self.ask > 0:
            self.spread = max(0.0, self.ask - self.bid)
            mid = (self.ask + self.bid) / 2
            self.spread_pct = self.spread / max(mid, 1e-9)

    def reset_for_new_day(self) -> None:
        """Reset state for new trading day."""
        self.fsm = State.IDLE
        self.or_high = -math.inf
        self.or_low = math.inf
        self.or_mid = 0.0
        self.or_locked = False
        self.cum_vol = 0.0
        self.cum_val = 0.0
        self.vwap = 0.0
        self.value15 = 0.0
        self.surge = 0.0
        self.break_ts = 0.0
        self.retest_low = math.inf
        self.imb_calc.reset()
        self.imb = 0.0
        self._prev_cum_vol = 0.0
        self.entry_px = 0.0
        self.entry_ts = 0.0
        self.qty = 0
        self.entry_order_id = None
