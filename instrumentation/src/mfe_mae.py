"""Shared MFE/MAE context builder for all strategies."""
from __future__ import annotations
import math


def build_mfe_mae_context(
    entry_price: float,
    stop_price: float,
    max_fav_price: float,
    min_adverse_price: float,
) -> dict:
    """Build MFE/MAE context dict for TradeEvent.

    Args:
        entry_price: Trade entry price.
        stop_price: Stop loss price (for R-multiple computation).
        max_fav_price: Highest price seen during trade (LONG). 0 if not tracked.
        min_adverse_price: Lowest price seen during trade (LONG). inf if not tracked.

    Returns:
        Dict with mfe_price, mae_price, mfe_pct, mae_pct, mfe_r, mae_r.
    """
    risk = max(abs(entry_price - stop_price), 1e-9)
    entry = entry_price if entry_price > 0 else 1e-9

    has_mfe = max_fav_price > 0 and not math.isinf(max_fav_price)
    has_mae = min_adverse_price < float("inf") and min_adverse_price > 0

    return {
        "mfe_price": max_fav_price if has_mfe else None,
        "mae_price": min_adverse_price if has_mae else None,
        "mfe_pct": round((max_fav_price - entry) / entry * 100, 4) if has_mfe else None,
        "mae_pct": round((entry - min_adverse_price) / entry * 100, 4) if has_mae else None,
        "mfe_r": round((max_fav_price - entry) / risk, 4) if has_mfe else None,
        "mae_r": round((entry - min_adverse_price) / risk, 4) if has_mae else None,
    }
