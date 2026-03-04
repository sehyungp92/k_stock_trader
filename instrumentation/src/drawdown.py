"""Drawdown state computation for trade context."""
from __future__ import annotations

# Drawdown tiers aligned with momentum_trader convention
_TIERS = [
    (-1.0, "full", 1.0),           # Above -1%: full sizing
    (-2.0, "half", 0.5),           # -1% to -2%: half sizing
    (-3.0, "quarter", 0.25),       # -2% to -3%: quarter sizing
    (float("-inf"), "halt", 0.0),  # Below -3%: halt new entries
]


def compute_drawdown_context(daily_pnl_pct: float) -> dict:
    """Compute drawdown context from daily P&L percentage.

    Args:
        daily_pnl_pct: Current session P&L as percentage (e.g., -1.5 for -1.5%).

    Returns:
        Dict with drawdown_pct, drawdown_tier, drawdown_size_mult.
    """
    dd_pct = min(daily_pnl_pct, 0.0)

    for threshold, tier, mult in _TIERS:
        if dd_pct >= threshold:
            return {
                "drawdown_pct": round(dd_pct, 4),
                "drawdown_tier": tier,
                "drawdown_size_mult": mult,
            }

    return {"drawdown_pct": round(dd_pct, 4), "drawdown_tier": "halt", "drawdown_size_mult": 0.0}
