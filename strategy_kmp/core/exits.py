"""
KMP Exit Engine: Hard stop, stall scratch, adaptive trailing.
"""

from __future__ import annotations
import time
from loguru import logger

from .state import SymbolState, State
from ..config.constants import STALL_MIN_MINUTES, STALL_R_MIN


def current_r(s: SymbolState, last_px: float) -> float:
    """Calculate current R-multiple."""
    risk = max(s.entry_px - s.structure_stop, 1e-9)
    return (last_px - s.entry_px) / risk


def retracement_factor(
    minutes_held: float,
    prog_regime: str,
    imb: float,
) -> float:
    """
    Calculate adaptive retracement factor for trailing stop.

    Tightens over time and with adverse flow.
    """
    # Base factor: starts at 0.5, ramps to 0.75
    if minutes_held <= 15:
        f = 0.5
    else:
        f = 0.5 + min(0.25, (minutes_held - 15) * 0.0167)

    # Tighten on outflow regime
    if prog_regime == "outflow":
        f = max(f, 0.7)

    # Tighten on negative tick imbalance
    if imb < 0:
        f = max(f, 0.7)

    return f


def update_trail(
    s: SymbolState,
    last_px: float,
    prog_regime: str,
) -> None:
    """Update trailing stop based on price action."""
    s.max_fav = max(s.max_fav, last_px)

    gain = s.max_fav - s.entry_px
    if gain <= 0:
        return

    minutes_held = (time.time() - s.entry_ts) / 60.0
    f = retracement_factor(minutes_held, prog_regime, s.imb)

    trail = s.entry_px + gain * f
    s.trail_px = max(s.trail_px, trail, s.structure_stop)


def check_exit_conditions(
    s: SymbolState,
    last_px: float,
    prog_regime: str,
    risk_off: bool = False,
) -> tuple[bool, str]:
    """
    Check all exit conditions.

    Returns (should_exit, reason).
    """
    now = time.time()

    # Portfolio/regime exit
    if risk_off:
        return True, "risk_off"

    # Hard stop
    if last_px <= s.hard_stop:
        return True, "hard_stop"

    # Acceptance failure (first 15 min)
    minutes_held = (now - s.entry_ts) / 60.0
    if minutes_held < 15:
        if last_px < s.or_high and last_px < s.vwap:
            return True, "acceptance_failure"

    # Stall scratch (R-based)
    if minutes_held >= STALL_MIN_MINUTES:
        r = current_r(s, last_px)
        if r < STALL_R_MIN:
            return True, "stall_scratch"

    # Trailing stop
    update_trail(s, last_px, prog_regime)
    if last_px <= s.trail_px and s.max_fav > s.entry_px:
        return True, "trailing_stop"

    return False, ""
