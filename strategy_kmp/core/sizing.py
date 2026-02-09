"""
KMP Position Sizing: Risk parity + quality score + overlays.
"""

from __future__ import annotations
from .state import SymbolState
from ..config.constants import (
    BASE_RISK_PCT, LIQ_CAP_PCT_5M_VALUE,
    QUALITY_THRESHOLD_LOW, QUALITY_THRESHOLD_MED, QUALITY_THRESHOLD_HIGH,
    RVOL_MIN,
)
from ..config.switches import kmp_switches
from .gates import min_surge_threshold, minutes_since_0916

NAV_CAP_PCT = 0.20  # Max 20% of NAV per position


def compute_qty(
    s: SymbolState,
    equity: float,
    entry_px: float,
    stop_px: float,
    program_mult: float,
    time_mult: float,
    now_kst,
    regime_breadth_ok: bool = True,
    not_chop: bool = True,
) -> int:
    """
    Compute position size using risk parity + quality + overlays.

    Args:
        s: Symbol state
        equity: Account equity
        entry_px: Entry price
        stop_px: Stop price
        program_mult: Program regime multiplier
        time_mult: Time decay multiplier
        now_kst: Current KST time
        regime_breadth_ok: Whether leader breadth >= 8
        not_chop: Whether market is NOT in chop

    Returns:
        Position size in shares
    """
    risk_krw = equity * BASE_RISK_PCT
    risk_per_share = max(entry_px - stop_px, 0.0)

    if risk_per_share <= 0:
        return 0

    qty_base = int(risk_krw / risk_per_share)

    qmult = quality_multiplier(s, now_kst, regime_breadth_ok=regime_breadth_ok, not_chop=not_chop)
    if qmult <= 0:
        return 0

    qty = int(qty_base * qmult * time_mult * program_mult)

    return max(0, qty)


def quality_multiplier(
    s: SymbolState,
    now_kst,
    switches=None,
    regime_breadth_ok: bool = True,
    not_chop: bool = True,
) -> float:
    """
    Calculate quality-based size multiplier.

    Args:
        s: Symbol state
        now_kst: Current KST time
        switches: Optional KMPSwitches instance (defaults to global)
        regime_breadth_ok: Whether leader breadth >= 8
        not_chop: Whether market is NOT in chop

    Returns 0.0, 0.5, 1.0, or 1.5 based on score.
    """
    if switches is None:
        switches = kmp_switches

    score = quality_score(s, now_kst, regime_breadth_ok=regime_breadth_ok, not_chop=not_chop)
    min_threshold = switches.quality_min_threshold

    # Use switch-configurable minimum threshold
    if score < min_threshold:
        return 0.0

    # Log would-block: score passed permissive threshold but would fail strict (40)
    if min_threshold < QUALITY_THRESHOLD_LOW and score < QUALITY_THRESHOLD_LOW:
        switches.log_would_block(
            s.code,
            "QUALITY_SCORE",
            score,
            QUALITY_THRESHOLD_LOW,
        )

    if score < QUALITY_THRESHOLD_MED:
        return 0.5
    if score < QUALITY_THRESHOLD_HIGH:
        return 1.0
    return 1.5


def quality_score(
    s: SymbolState,
    now_kst,
    regime_breadth_ok: bool = True,
    not_chop: bool = True,
) -> float:
    """
    Calculate quality score (0-100).

    Components:
    - Surge vs minimum threshold (0-20 pts)
    - Relative volume (0-15 pts)
    - Tick imbalance (sponsorship proxy) (0-15 pts)
    - Spread/liquidity (0-10 pts)
    - Acceptance-cleanliness (0-10 pts)
    - Regime breadth (0 or 15 pts, binary)
    - Not-chop (0 or 15 pts, binary)
    """
    score = 0.0

    # Surge component (0-20 points)
    m = minutes_since_0916(now_kst)
    min_surge = min_surge_threshold(m)
    surge_excess = s.surge - min_surge
    score += max(0, min(20, surge_excess * 10))

    # RVol component (0-15 points)
    rvol_excess = s.rvol_1m - RVOL_MIN
    score += max(0, min(15, rvol_excess * 10))

    # Tick imbalance component (0-15 points)
    imb_score = (s.imb + 0.1) * 50
    score += max(0, min(15, imb_score))

    # Spread component (0-10 points)
    spread_score = 10 - (s.spread_pct * 500)
    score += max(0, min(10, spread_score))

    # Acceptance-cleanliness component (0-10 points)
    # Clean = pullback held support tightly (retest_low close to or_high)
    # Shallow pullback (<1%) = max points, deep (>3%) = 0
    if s.or_high > 0 and s.retest_low > 0 and s.retest_low < s.or_high:
        pullback_depth = (s.or_high - s.retest_low) / s.or_high
        cleanliness = max(0, 10 - pullback_depth * 400)
        score += min(10, cleanliness)

    # Regime breadth component (0 or 15 points, binary)
    if regime_breadth_ok:
        score += 15

    # Not-chop component (0 or 15 points, binary)
    if not_chop:
        score += 15

    return max(0, min(100, score))


def apply_liquidity_cap(qty: int, entry_px: float, last_5m_value: float) -> int:
    """
    Apply liquidity cap to position size.

    Limits to 5% of last 5-minute traded value.
    """
    if last_5m_value <= 0:
        return qty

    max_notional = LIQ_CAP_PCT_5M_VALUE * last_5m_value
    max_qty = int(max_notional / max(entry_px, 1.0))

    return min(qty, max_qty)


def apply_nav_cap(qty: int, entry_px: float, equity: float) -> int:
    """
    Apply NAV cap to position size.

    Limits to 20% of NAV.
    """
    if equity <= 0 or entry_px <= 0:
        return qty

    max_notional = NAV_CAP_PCT * equity
    max_qty = int(max_notional / entry_px)

    return min(qty, max_qty)
