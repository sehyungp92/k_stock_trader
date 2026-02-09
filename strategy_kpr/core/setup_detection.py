"""KPR Setup Detection."""

from datetime import datetime
from .state import SymbolState
from ..config.constants import (
    VWAP_DEPTH_MIN, VWAP_DEPTH_MAX, PANIC_DROP_PCT, PANIC_MAX_AGE_MIN,
    DRIFT_DROP_PCT, DRIFT_MIN_AGE_MIN, RECLAIM_OFFSET_PCT, STOP_BUFFER_PCT,
)
from ..config.switches import kpr_switches


def check_vwap_depth(price: float, vwap: float, switches=None, symbol: str = "") -> tuple[bool, float]:
    """
    Check if price is within acceptable VWAP depth range.

    Args:
        price: Current price
        vwap: VWAP price
        switches: Optional KPRSwitches instance (defaults to global)
        symbol: Stock code for logging

    Returns:
        Tuple of (is_in_band, depth_pct)
    """
    if switches is None:
        switches = kpr_switches

    if vwap <= 0:
        return False, 0.0

    depth = (vwap - price) / vwap
    depth_min = switches.vwap_depth_min
    depth_max = switches.vwap_depth_max

    in_band = depth_min <= depth <= depth_max

    # Log would-block for setups that pass permissive but would fail strict
    if in_band and symbol:
        # Check if would fail strict depth minimum (0.02)
        if depth < VWAP_DEPTH_MIN:
            kpr_switches.log_would_block(
                symbol,
                "VWAP_DEPTH_MIN",
                depth,
                VWAP_DEPTH_MIN,
            )
        # Check if would fail strict depth maximum (0.05)
        if depth > VWAP_DEPTH_MAX:
            kpr_switches.log_would_block(
                symbol,
                "VWAP_DEPTH_MAX",
                depth,
                VWAP_DEPTH_MAX,
            )

    return in_band, depth


def detect_panic_flush(s: SymbolState, price: float, bar_time: datetime) -> bool:
    if s.hod <= 0 or s.hod_time is None:
        return False
    drop_pct = (s.hod - price) / s.hod
    hod_age = (bar_time - s.hod_time).total_seconds() / 60.0
    return drop_pct >= PANIC_DROP_PCT and hod_age <= PANIC_MAX_AGE_MIN


def detect_drift(s: SymbolState, price: float, bar_time: datetime) -> bool:
    if s.hod <= 0 or s.hod_time is None:
        return False
    drop_pct = (s.hod - price) / s.hod
    hod_age = (bar_time - s.hod_time).total_seconds() / 60.0
    return drop_pct >= DRIFT_DROP_PCT and hod_age >= DRIFT_MIN_AGE_MIN


def detect_setup(s: SymbolState, bar: dict, vwap: float, bar_time: datetime) -> bool:
    price = float(bar.get('close', 0))
    in_band, _ = check_vwap_depth(price, vwap, symbol=s.code)
    if not in_band:
        return False

    is_panic = detect_panic_flush(s, price, bar_time)
    is_drift = detect_drift(s, price, bar_time)
    if not (is_panic or is_drift):
        return False

    s.setup_low = s.lod
    s.reclaim_level = s.setup_low * (1 + RECLAIM_OFFSET_PCT)
    s.stop_level = s.setup_low * (1 - STOP_BUFFER_PCT)
    s.setup_time = bar_time
    s.setup_type = "panic" if is_panic else "drift"
    s.accept_closes = 0
    return True
