"""
KMP Entry Gates: VI, spread, RVol, time decay.
"""

from __future__ import annotations
import time

from .state import SymbolState
from ..config.constants import (
    RVOL_MIN, SPREAD_MAX_PCT, VI_WALL_TICKS, VI_COOLDOWN_MIN,
    OR_RANGE_MIN, OR_RANGE_MAX, MIN_SURGE_BASE, MIN_SURGE_SLOPE,
    SIZE_DECAY_SLOPE, SIZE_DECAY_FLOOR,
)
from ..config.switches import kmp_switches


def minutes_since_0916(now_kst) -> float:
    """Calculate minutes since 09:16 KST."""
    target = now_kst.replace(hour=9, minute=16, second=0, microsecond=0)
    delta = (now_kst - target).total_seconds() / 60.0
    return max(0.0, delta)


def min_surge_threshold(minutes: float, switches=None) -> float:
    """
    Calculate minimum surge threshold with time decay.

    Args:
        minutes: Minutes since 09:16
        switches: Optional KMPSwitches instance (defaults to global)

    Starts at 3.0, increases by slope per minute.
    Default slope 0.03 (permissive), conservative 0.04.
    """
    if switches is None:
        switches = kmp_switches

    m = max(0.0, min(44.0, minutes))
    slope = switches.min_surge_slope

    return MIN_SURGE_BASE + slope * m


def min_surge_threshold_strict(minutes: float) -> float:
    """Calculate minimum surge threshold using conservative (strict) slope."""
    m = max(0.0, min(44.0, minutes))
    return MIN_SURGE_BASE + 0.04 * m


def size_time_multiplier(minutes: float) -> float:
    """
    Calculate size multiplier with time decay.

    Starts at 1.0, decreases to 0.45 floor.
    """
    m = max(0.0, min(44.0, minutes))
    return max(SIZE_DECAY_FLOOR, 1.0 - SIZE_DECAY_SLOPE * m)


def lock_or_and_filter(s: SymbolState, switches=None) -> bool:
    """
    Lock opening range and filter by range size.

    Args:
        s: Symbol state
        switches: Optional KMPSwitches instance (defaults to global)

    Returns True if OR range is valid.
    Default 1.2% - 7.0% (permissive), conservative 1.2% - 5.5%.
    """
    if switches is None:
        switches = kmp_switches

    s.or_locked = True
    s.or_mid = (s.or_high + s.or_low) / 2

    if s.or_mid <= 0:
        return False

    or_pct = (s.or_high - s.or_low) / s.or_mid
    or_max = switches.or_range_max
    result = OR_RANGE_MIN <= or_pct <= or_max

    # Log would-block: passed permissive max but would fail strict (5.5%)
    if result and or_pct > OR_RANGE_MAX:
        switches.log_would_block(
            s.code,
            "OR_RANGE_MAX",
            or_pct,
            OR_RANGE_MAX,
            {"or_high": s.or_high, "or_low": s.or_low},
        )

    return result


def spread_ok(s: SymbolState) -> bool:
    """Check if spread is within acceptable range."""
    return s.spread_pct <= SPREAD_MAX_PCT


def rvol_ok(s: SymbolState) -> bool:
    """Check if relative volume is sufficient."""
    return s.rvol_1m >= RVOL_MIN


def vi_blocked(s: SymbolState, entry_trigger_px: float, tick_size: float) -> bool:
    """
    Check if entry is blocked by VI proximity.

    Returns True if within VI cooldown or entry price is near VI wall.
    Missing VI reference blocks entry (per spec: vi_ref missing → DONE).
    """
    if s.vi_ref <= 0:
        return True

    # Cooldown: block during post-VI churn period
    now = time.time()
    if (now - s.last_vi_ts) < VI_COOLDOWN_MIN * 60:
        return True

    # Wall: block if entry is too close to static VI trigger
    static_up = s.vi_ref * 1.02
    wall = static_up - (VI_WALL_TICKS * tick_size)

    return entry_trigger_px >= wall


def is_in_or_window(ts_kst) -> bool:
    """Check if timestamp is within OR window (09:00-09:15)."""
    h, m = ts_kst.hour, ts_kst.minute
    if h == 9 and m < 15:
        return True
    return False


def is_past_entry_cutoff(now_kst, cutoff=(10, 0)) -> bool:
    """Check if past entry cutoff time."""
    return (now_kst.hour, now_kst.minute) >= cutoff


def is_past_flatten_time(now_kst, flatten=(14, 30)) -> bool:
    """Check if past flatten time."""
    return (now_kst.hour, now_kst.minute) >= flatten
