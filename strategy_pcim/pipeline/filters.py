"""PCIM Hard and Soft Filters."""

from typing import Optional
from loguru import logger

from .candidate import Candidate
from ..config.constants import HARD_FILTERS, SOFT_FILTERS, GAP_REVERSAL
from ..config.switches import pcim_switches


def apply_hard_filters(c: Candidate, has_earnings_soon: bool) -> Optional[str]:
    """Apply hard filters. Returns reject reason or None."""
    if c.adtv_20d < HARD_FILTERS["ADTV_MIN"]:
        logger.info(
            f"{c.symbol}: REJECTED ADTV_LT_5B "
            f"(actual={c.adtv_20d/1e9:.1f}B < {HARD_FILTERS['ADTV_MIN']/1e9:.0f}B)"
        )
        return "ADTV_LT_5B"
    if c.market_cap < HARD_FILTERS["MCAP_MIN"]:
        logger.info(
            f"{c.symbol}: REJECTED MCAP_LT_30B "
            f"(actual={c.market_cap/1e9:.1f}B < {HARD_FILTERS['MCAP_MIN']/1e9:.0f}B)"
        )
        return "MCAP_LT_30B"
    if c.market_cap > HARD_FILTERS["MCAP_MAX"]:
        logger.info(
            f"{c.symbol}: REJECTED MCAP_GT_15T "
            f"(actual={c.market_cap/1e12:.2f}T > {HARD_FILTERS['MCAP_MAX']/1e12:.0f}T)"
        )
        return "MCAP_GT_15T"
    if has_earnings_soon:
        logger.info(f"{c.symbol}: REJECTED EARNINGS_WINDOW (within 5 days)")
        return "EARNINGS_WINDOW"
    return None


def apply_gap_reversal_filter(c: Candidate, switches=None) -> Optional[str]:
    """
    Apply gap reversal rate filter. Returns reject reason or None.

    Args:
        c: Candidate to filter
        switches: Optional PCIMSwitches instance (defaults to global)
    """
    if switches is None:
        switches = pcim_switches

    if c.gap_rev_insufficient:
        return None

    threshold = switches.gap_reversal_threshold
    strict_threshold = GAP_REVERSAL["THRESHOLD"]

    if c.gap_rev_rate > threshold:
        logger.info(
            f"{c.symbol}: REJECTED GAP_REV_GT_THRESHOLD "
            f"(rate={c.gap_rev_rate:.1%} > {threshold:.0%}, events={c.gap_rev_events})"
        )
        return f"GAP_REV_GT_{int(threshold*100)}PCT_{c.gap_rev_rate:.1%}"

    # Log would-block: passed permissive but would fail strict (0.60)
    if c.gap_rev_rate > strict_threshold:
        switches.log_would_block(
            c.symbol,
            "GAP_REVERSAL",
            c.gap_rev_rate,
            strict_threshold,
            {"events": c.gap_rev_events},
        )

    return None


def compute_soft_multiplier(c: Candidate, five_day_return: float, switches=None) -> float:
    """
    Compute soft filter multiplier.

    Args:
        c: Candidate to evaluate
        five_day_return: 5-day price return
        switches: Optional PCIMSwitches instance (defaults to global)

    Returns:
        Soft filter multiplier (1.0 = no penalty)
    """
    if switches is None:
        switches = pcim_switches

    mult = 1.0

    # ADTV soft penalty (optional - redundant with tier sizing for T3)
    if switches.enable_adtv_soft_penalty:
        if SOFT_FILTERS["ADTV_SOFT_LOW"] <= c.adtv_20d < SOFT_FILTERS["ADTV_SOFT_HIGH"]:
            mult *= SOFT_FILTERS["ADTV_SOFT_MULT"]
            logger.debug(f"{c.symbol}: Soft mult {SOFT_FILTERS['ADTV_SOFT_MULT']} (low ADTV)")
    else:
        # Log would-block if ADTV is in soft penalty range
        if SOFT_FILTERS["ADTV_SOFT_LOW"] <= c.adtv_20d < SOFT_FILTERS["ADTV_SOFT_HIGH"]:
            switches.log_would_block(
                c.symbol,
                "ADTV_SOFT_PENALTY",
                1.0,
                SOFT_FILTERS["ADTV_SOFT_MULT"],
                {"adtv": c.adtv_20d, "note": "Tier sizing still applies"},
            )

    # 5-day return penalty (always applied)
    if five_day_return > SOFT_FILTERS["FIVEDAY_UP_PCT"]:
        mult *= SOFT_FILTERS["FIVEDAY_MULT"]
        logger.debug(f"{c.symbol}: Soft mult {SOFT_FILTERS['FIVEDAY_MULT']} (5d up {five_day_return:.1%})")

    return mult
