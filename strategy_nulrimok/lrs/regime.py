"""Nulrimok Composite Market Regime."""

import math
from dataclasses import dataclass

from kis_core import sma, percentile_rank
from .db import LRSDatabase
from ..config.constants import (
    REGIME_PRICE_WEIGHT, REGIME_BREADTH_WEIGHT, REGIME_VOL_WEIGHT, REGIME_FX_WEIGHT,
    REGIME_TIER_A_THRESHOLD, REGIME_TIER_B_THRESHOLD, BREADTH_THRESHOLD,
    VOL_PERCENTILE_CAP, FX_CHANGE_CAP,
)
from ..config.switches import nulrimok_switches


@dataclass
class RegimeResult:
    tier: str
    score: float
    risk_mult: float
    price_ok: bool
    breadth_ok: bool
    vol_ok: bool
    fx_ok: bool
    breadth_value: float
    vol_value: float
    fx_change: float


def _compute_breadth(lrs: LRSDatabase) -> float:
    """Breadth = % of universe members with close > SMA20."""
    tickers = lrs.get_all_tickers()
    if not tickers:
        return 0.0
    above, total = 0, 0
    for t in tickers:
        closes = lrs.get_closes(t, 30)
        if len(closes) < 20:
            continue
        total += 1
        sma20 = sum(closes[-20:]) / 20
        if closes[-1] > sma20:
            above += 1
    return above / total if total > 0 else 0.0


def compute_regime(lrs: LRSDatabase) -> RegimeResult:
    # Fetch ~1Y of trading days (280) for proper vol percentile
    index_data = lrs.get_index_series("KOSPI", days=280)
    if len(index_data) < 50:
        return RegimeResult("C", 0.0, 0.0, False, False, False, False, 0.0, 0.0, 0.0)

    closes = [d['close'] for d in index_data]
    ma50_values = sma(closes, 50)
    price_ok = closes[-1] > ma50_values[-1] if ma50_values else False

    breadth = _compute_breadth(lrs)
    breadth_ok = breadth > BREADTH_THRESHOLD

    # Compute vol percentile against rolling 1Y distribution of 20-day vols
    returns = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
    vol_20d = math.sqrt(sum(r ** 2 for r in returns[-20:]) / 20) * math.sqrt(252) if len(returns) >= 20 else 0.2
    # Build rolling 20-day vol series from all available data
    vol_series = [math.sqrt(sum(r ** 2 for r in returns[i:i + 20]) / 20) * math.sqrt(252)
                  for i in range(max(0, len(returns) - 252), len(returns) - 19)]
    vol_pct = percentile_rank(vol_20d, vol_series) / 100 if vol_series else 0.5
    vol_ok = vol_pct < VOL_PERCENTILE_CAP

    fx_series = lrs.get_fx_series("KRWUSD", days=10)
    fx_change = (fx_series[-1] / fx_series[-6]) - 1 if len(fx_series) >= 6 else 0.0
    fx_ok = fx_change < FX_CHANGE_CAP

    score = (REGIME_PRICE_WEIGHT * price_ok + REGIME_BREADTH_WEIGHT * breadth_ok +
             REGIME_VOL_WEIGHT * vol_ok + REGIME_FX_WEIGHT * fx_ok)

    if score > REGIME_TIER_A_THRESHOLD:
        tier, risk_mult = "A", 1.0
    elif score >= REGIME_TIER_B_THRESHOLD:
        tier, risk_mult = "B", 0.5
    else:
        # Tier C handling with switch
        tier = "C"
        if nulrimok_switches.allow_tier_c_reduced:
            # Permissive: allow Tier C with reduced 0.25x sizing
            risk_mult = 0.25
            nulrimok_switches.log_would_block(
                "REGIME",
                "TIER_C_REDUCED",
                0.25,
                0.0,
                {"score": score, "tier": tier},
            )
        else:
            # Conservative: block Tier C completely
            risk_mult = 0.0

    return RegimeResult(tier, score, risk_mult, price_ok, breadth_ok, vol_ok, fx_ok,
                        breadth, vol_20d, fx_change)
