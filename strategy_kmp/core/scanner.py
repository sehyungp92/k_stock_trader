"""
KMP 09:15 Value Surge Scanner.
"""

from __future__ import annotations
import asyncio
from typing import Dict, List, Optional, Tuple
from loguru import logger

from kis_core import RateBudget

from .state import SymbolState, State
from ..config.constants import GAP_SKIP

# REST call spacing when rate limited (seconds)
_RATE_LIMIT_SLEEP = 0.5


async def _rate_limited_call(budget: Optional[RateBudget], endpoint: str, fn, *args, **kwargs):
    """Execute REST call with rate limiting. Retries with backoff if limited."""
    for attempt in range(3):
        if budget is None or budget.try_consume(endpoint):
            return fn(*args, **kwargs)
        await asyncio.sleep(_RATE_LIMIT_SLEEP * (attempt + 1))
    # Final attempt without budget check
    return fn(*args, **kwargs)


async def scan_at_0915(
    api,  # KoreaInvestAPI
    universe: List[str],
    baseline_15m: Dict[str, float],
    states: Dict[str, SymbolState],
    min_surge: float = 3.0,
    top_n: int = 40,
    rate_budget: Optional[RateBudget] = None,
) -> List[str]:
    """
    Scan for value surge leaders at 09:15.

    Also seeds Opening Range and early RVol/ATR from REST 1m bars.

    Args:
        rate_budget: Shared RateBudget for cross-strategy REST coordination.
    """
    scored: List[Tuple[str, float]] = []

    for ticker in universe:
        state = states.get(ticker)
        if not state:
            continue

        if not state.trend_ok:
            continue

        try:
            bars = await _rate_limited_call(
                rate_budget, "CHART", api.get_minute_bars, ticker, minutes=15
            )
            if bars is None or bars.empty:
                continue

            value15 = (bars['close'] * bars['volume']).sum()

            base = baseline_15m.get(ticker, 0.0)
            if base <= 0:
                continue

            surge = value15 / base
            if surge < min_surge:
                continue

            # Gap skip: reject if open gapped >= 5% from prev close
            if state.sma20 > 0:
                open_px = bars.iloc[0]['open'] if 'open' in bars.columns else 0.0
                prev_close = state.prev_close
                if prev_close > 0 and open_px > 0:
                    gap_pct = abs(open_px - prev_close) / prev_close
                    if gap_pct >= GAP_SKIP:
                        logger.debug(f"{ticker}: Gap {gap_pct:.1%} >= {GAP_SKIP:.0%}, skip")
                        continue

            # Seed Opening Range from REST 1m bars (09:00-09:15)
            state.or_high = bars['high'].max()
            state.or_low = bars['low'].min()

            # Seed early RVol from last completed 1m bar
            if state.avg_1m_vol > 0 and not bars.empty:
                last_bar_vol = bars.iloc[-1]['volume']
                if last_bar_vol > 0:
                    state.rvol_1m = last_bar_vol / state.avg_1m_vol

            # Seed last_5m_value from last 5 bars (approx 5m of 1m bars)
            tail = bars.tail(5)
            state.last_5m_value = (tail['close'] * tail['volume']).sum()

            state.value15 = value15
            state.surge = surge
            scored.append((ticker, value15))

        except Exception as e:
            logger.debug(f"Scan error for {ticker}: {e}")
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [t for t, _ in scored[:top_n]]

    for ticker in top:
        states[ticker].fsm = State.CANDIDATE
        logger.info(f"KMP candidate: {ticker} surge={states[ticker].surge:.1f}x")

    return top


def apply_trend_anchor(
    states: Dict[str, SymbolState],
    daily_data: Dict[str, list],
) -> None:
    """
    Apply daily trend anchor.

    Requires: close > SMA20, SMA20 slope >= 0, SMA20 >= SMA60.
    Also stores prev_close for gap-skip filter.
    """
    for ticker, bars in daily_data.items():
        if ticker not in states:
            continue

        if len(bars) < 60:
            continue

        state = states[ticker]

        closes = [b['close'] for b in bars[-60:]]
        state.sma60 = sum(closes) / 60
        state.sma20 = sum(closes[-20:]) / 20
        state.prev_close = closes[-1]

        # SMA20 slope: compare current SMA20 to SMA20 one bar ago
        sma20_prev = sum(closes[-21:-1]) / 20
        slope_ok = state.sma20 >= sma20_prev

        state.trend_ok = (
            closes[-1] > state.sma20
            and slope_ok
            and state.sma20 >= state.sma60
        )
