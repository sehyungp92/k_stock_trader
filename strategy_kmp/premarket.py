"""
KMP Premarket Baseline Computation.

Estimates intraday baselines from daily OHLCV data since the KIS API
does not support historical intraday date-range queries.

baseline_15m_value: Estimated 09:00-09:15 traded value (close * volume).
    Uses daily traded value scaled by an opening-concentration factor.
    The first 15 minutes of KRX trading typically account for ~8% of
    daily value (empirically 7-10% depending on market regime).

baseline_1m_vol: Estimated average 1-minute volume during 09:00-10:00.
    The first hour accounts for ~20% of daily volume across 60 minutes.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

from loguru import logger

# Opening 15 minutes concentration: ~8% of daily value
_OPEN_15M_FRACTION = 0.08

# First-hour concentration: ~20% of daily volume across 60 one-minute bars
_FIRST_HOUR_FRACTION = 0.20
_FIRST_HOUR_MINUTES = 60


def compute_baselines(
    daily_data: Dict[str, list],
    lookback_15m: int = 14,
    lookback_1m: int = 20,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute baseline_15m_value and baseline_1m_vol from daily bars.

    Args:
        daily_data: {ticker: [list of daily bar dicts]} with keys
                    'close' and 'volume'.  Sorted ascending by date.
        lookback_15m: Number of recent trading days for 15m value baseline.
        lookback_1m:  Number of recent trading days for 1m volume baseline.

    Returns:
        (baseline_15m_value, baseline_1m_vol) dicts keyed by ticker.
    """
    baseline_15m: Dict[str, float] = {}
    baseline_1m_vol: Dict[str, float] = {}
    computed = 0

    for ticker, bars in daily_data.items():
        if not bars:
            continue

        b15, b1m = _estimate_ticker_baselines(
            bars, lookback_15m, lookback_1m,
        )

        if b15 > 0:
            baseline_15m[ticker] = b15
        if b1m > 0:
            baseline_1m_vol[ticker] = b1m
        if b15 > 0 or b1m > 0:
            computed += 1

    logger.info(
        f"Premarket baselines computed: {computed}/{len(daily_data)} tickers "
        f"(15m_value={len(baseline_15m)}, 1m_vol={len(baseline_1m_vol)})"
    )
    return baseline_15m, baseline_1m_vol


def _estimate_ticker_baselines(
    bars: list,
    lookback_15m: int,
    lookback_1m: int,
) -> Tuple[float, float]:
    """Estimate baselines for a single ticker from daily bars."""
    max_lookback = max(lookback_15m, lookback_1m)
    recent = bars[-max_lookback:] if len(bars) >= max_lookback else bars

    # Compute daily traded values and volumes
    daily_values: List[float] = []
    daily_volumes: List[float] = []
    for b in recent:
        close = float(b.get('close', 0))
        volume = float(b.get('volume', 0))
        if close > 0 and volume > 0:
            daily_values.append(close * volume)
            daily_volumes.append(volume)

    # baseline_15m_value: median of estimated 15-minute opening values
    b15 = 0.0
    vals_15m = daily_values[-lookback_15m:] if len(daily_values) >= lookback_15m else daily_values
    if vals_15m:
        estimated = [v * _OPEN_15M_FRACTION for v in vals_15m]
        b15 = statistics.median(estimated)

    # baseline_1m_vol: median of estimated average 1-minute volume (first hour)
    b1m = 0.0
    vols_1m = daily_volumes[-lookback_1m:] if len(daily_volumes) >= lookback_1m else daily_volumes
    if vols_1m:
        estimated = [v * _FIRST_HOUR_FRACTION / _FIRST_HOUR_MINUTES for v in vols_1m]
        b1m = statistics.median(estimated)

    return b15, b1m
