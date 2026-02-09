"""
Tick dispatch: WS tick/ASP events -> SymbolState updates.

Handles H0STCNT0 (tick) and H0STASP0 (bid/ask) messages,
updating VWAP, OR, tick imbalance, 1m/5m bars, rvol, and bid/ask.
"""

from __future__ import annotations
import time
from datetime import datetime
from typing import Dict
from loguru import logger

from ..core.state import SymbolState, State
from ..core.tick_table import tick_size


def on_tick(
    s: SymbolState,
    price: float,
    volume: float,
    cum_vol: float,
    cum_val: float,
    vi_ref: float,
    ts: datetime,
    or_locked: bool,
) -> None:
    """
    Process a single H0STCNT0 tick event.

    Args:
        s: Symbol state to update.
        price: Last traded price from tick.
        volume: Tick volume (shares in this tick).
        cum_vol: Cumulative volume from open (from WS field).
        cum_val: Cumulative value from open (from WS field).
        vi_ref: VI reference price (0 if not provided).
        ts: Tick timestamp as datetime (KST).
        or_locked: Whether OR is already locked.
    """
    if price <= 0:
        return

    # --- VWAP (prefer cumulative fields from exchange) ---
    if cum_vol > 0 and cum_val > 0:
        s.cum_vol = cum_vol
        s.cum_val = cum_val
        s.vwap = cum_val / cum_vol
    elif volume > 0:
        s.update_vwap(price, volume)

    # --- Opening Range (only before lock) ---
    if not or_locked and not s.or_locked:
        s.or_high = max(s.or_high, price)
        s.or_low = min(s.or_low, price)

    # --- VI reference (only mark event when ref changes) ---
    if vi_ref > 0 and vi_ref != s.vi_ref:
        s.vi_ref = vi_ref
        s.last_vi_ts = time.time()

    # --- Tick imbalance (bucketed, uses tick-rule classification) ---
    # Compute trade_vol from cumulative delta when WS provides cumulative only
    if cum_vol > 0:
        trade_vol = cum_vol - s._prev_cum_vol if s._prev_cum_vol > 0 else volume
        s._prev_cum_vol = cum_vol
    else:
        trade_vol = volume

    now_ts = time.time()
    if trade_vol > 0:
        s.imb_calc.update(now_ts, price, trade_vol)
    s.imb = s.imb_calc.compute(now_ts)

    # --- Bar aggregation ---
    completed_1m = s.bar_1m.update_tick(ts, price, volume)
    s.bar_5m.update_tick(ts, price, volume)

    if completed_1m is not None:
        # Update rolling ATR from completed 1m bar
        atr_val = s.rolling_atr.update_bar(
            completed_1m.high, completed_1m.low, completed_1m.close,
        )
        if atr_val is not None:
            s.atr_1m = atr_val

        # Update RVol (current 1m volume vs historical average)
        s.curr_1m_vol = completed_1m.volume
        if s.avg_1m_vol > 0:
            s.rvol_1m = completed_1m.volume / s.avg_1m_vol

    # --- 5m value from completed 5m bars ---
    bars_5m = s.bar_5m.get_completed_bars(1)
    if bars_5m:
        last_bar = bars_5m[-1]
        # Approximate value = close * volume (conservative proxy)
        s.last_5m_value = last_bar.close * last_bar.volume


def on_ask_bid(s: SymbolState, bid: float, ask: float) -> None:
    """Process H0STASP0 (orderbook top-of-book) event."""
    if bid > 0:
        s.bid = bid
    if ask > 0:
        s.ask = ask
    s.update_spread()


