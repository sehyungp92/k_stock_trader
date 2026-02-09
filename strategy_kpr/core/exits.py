"""KPR Exit Engine: hard stop, partial, full, time stop, flow deterioration."""

from __future__ import annotations
from datetime import datetime
from loguru import logger

from .state import SymbolState, FSMState
from ..signals.investor import InvestorSignal
from ..signals.micro import MicroSignal
from ..config.constants import (
    TIME_STOP_MINUTES, PARTIAL_R_TARGET,
    FULL_R_TARGET_HIGH_VOL, FULL_R_TARGET_NORMAL, FULL_R_TARGET_LOW_VOL,
    TRAIL_START_R, TRAIL_FACTOR, FLOW_DETERIORATE_TRAIL,
)


def current_r(s: SymbolState, price: float) -> float:
    risk = max(s.entry_px - (s.stop_level or s.entry_px * 0.98), 1e-9)
    return (price - s.entry_px) / risk


def adaptive_full_r_target(s: SymbolState) -> float:
    """Return adaptive full R target based on volatility (ATR / entry_px)."""
    if s.entry_px <= 0:
        return FULL_R_TARGET_NORMAL
    atr = getattr(s, 'atr', 0.0) or s.entry_px * 0.02
    atr_pct = atr / s.entry_px
    if atr_pct >= 0.03:  # high vol >= 3%
        return FULL_R_TARGET_HIGH_VOL
    if atr_pct <= 0.015:  # low vol <= 1.5%
        return FULL_R_TARGET_LOW_VOL
    return FULL_R_TARGET_NORMAL


def check_exits(
    s: SymbolState,
    price: float,
    now: datetime,
    investor_sig=None,
    micro_sig=None,
) -> tuple[bool, str, int]:
    """
    Check all exit conditions for an IN_POSITION symbol.

    Returns (should_exit, reason, exit_qty).
    exit_qty == 0 means no exit.
    exit_qty < s.remaining_qty means partial.
    exit_qty == s.remaining_qty means full exit.
    """
    if s.fsm != FSMState.IN_POSITION or s.remaining_qty <= 0:
        return False, "", 0

    stop = s.stop_level or s.entry_px * 0.98

    # 1. Hard stop
    if price <= stop:
        return True, "hard_stop", s.remaining_qty

    r = current_r(s, price)

    # 2. Update max price and trailing stop
    s.max_price = max(s.max_price, price)
    if r >= TRAIL_START_R:
        gain = s.max_price - s.entry_px
        trail = s.entry_px + gain * TRAIL_FACTOR
        s.trail_stop = max(s.trail_stop, trail, stop)

    # 3. Flow deterioration: tighten trail
    if investor_sig == InvestorSignal.DISTRIBUTE or micro_sig == MicroSignal.DISTRIBUTE:
        if r >= TRAIL_START_R:
            gain = s.max_price - s.entry_px
            tight_trail = s.entry_px + gain * FLOW_DETERIORATE_TRAIL
            s.trail_stop = max(s.trail_stop, tight_trail)

    # 4. Trailing stop hit
    if s.trail_stop > 0 and price <= s.trail_stop:
        return True, "trailing_stop", s.remaining_qty

    # 5. Partial at 1.5R (state updates happen in main.py after OMS confirms)
    if not s.partial_filled and r >= PARTIAL_R_TARGET:
        partial_qty = s.remaining_qty // 2
        if partial_qty > 0:
            logger.info(f"{s.code}: Partial exit at {PARTIAL_R_TARGET}R, qty={partial_qty}")
            return True, "partial_target", partial_qty

    # 6. Full target at adaptive R
    full_r = adaptive_full_r_target(s)
    if r >= full_r:
        return True, "full_target", s.remaining_qty

    # 7. Time stop
    if s.entry_ts:
        held_min = (now - s.entry_ts).total_seconds() / 60.0
        if held_min >= TIME_STOP_MINUTES and r < TRAIL_START_R:
            return True, "time_stop", s.remaining_qty

    return False, "", 0
