from __future__ import annotations

from statistics import fmean
from typing import Iterable

from strategy_common.market import MarketBar

from .config import KALCBConfig
from .models import EntryType


def close_location_value(bar: MarketBar) -> float:
    width = max(float(bar.high) - float(bar.low), 1e-9)
    return (float(bar.close) - float(bar.low)) / width


def compute_opening_range(bars_5m: list[MarketBar], n_bars: int) -> tuple[float, float, float]:
    if len(bars_5m) < n_bars:
        return 0.0, 0.0, 0.0
    window = bars_5m[:n_bars]
    return (
        max(float(bar.high) for bar in window),
        min(float(bar.low) for bar in window),
        sum(float(bar.volume) for bar in window),
    )


def compute_session_vwap(bars_5m: Iterable[MarketBar]) -> float:
    cum_pv = 0.0
    cum_vol = 0.0
    for bar in bars_5m:
        typical = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        cum_pv += typical * float(bar.volume)
        cum_vol += float(bar.volume)
    return cum_pv / cum_vol if cum_vol > 0 else 0.0


def compute_bar_rvol(bar_volume: float, expected_5m_volume: float) -> float:
    if expected_5m_volume <= 0:
        return 0.0
    return float(bar_volume) / float(expected_5m_volume)


def classify_breakout(
    bar: MarketBar,
    *,
    prior_day_high: float,
    or_high: float,
    rvol: float,
    cpr: float,
    config: KALCBConfig,
) -> EntryType | None:
    if rvol < config.rvol_threshold:
        return None
    if cpr < config.cpr_threshold:
        return None
    return classify_raw_breakout(bar, prior_day_high=prior_day_high, or_high=or_high)


def classify_raw_breakout(
    bar: MarketBar,
    *,
    prior_day_high: float,
    or_high: float,
) -> EntryType | None:
    above_or = float(bar.close) > float(or_high)
    above_pdh = float(bar.close) > float(prior_day_high) if prior_day_high > 0 else False
    if above_or and above_pdh:
        return EntryType.COMBINED_BREAKOUT
    if above_or:
        return EntryType.OR_BREAKOUT
    if above_pdh:
        return EntryType.PDH_BREAKOUT
    return None


def compute_momentum_score(
    bar: MarketBar,
    bars_today: list[MarketBar],
    *,
    prior_day_high: float,
    prior_day_close: float,
    or_high: float,
    avwap: float,
    adx_value: float,
    sector_flow: float,
    config: KALCBConfig,
) -> tuple[int, dict[str, int]]:
    """ALCB-style completed-bar quality score.

    Mirrors the stock ALCB T1 score factors while allowing KRX flow/regime
    features to be optional. Missing optional features simply contribute zero.
    """

    score = 0
    detail: dict[str, int] = {}
    if prior_day_high > 0 and bar.close > prior_day_high:
        score += 1
        detail["above_pdh"] = 1
    if or_high > 0 and bar.close > or_high:
        score += 1
        detail["above_or"] = 1
    if len(bars_today) >= 2:
        avg_vol = fmean(float(item.volume) for item in bars_today[:-1])
        if avg_vol > 0 and float(bar.volume) / avg_vol >= 1.3:
            score += 1
            detail["bar_vol_surge"] = 1
    if close_location_value(bar) >= config.cpr_threshold:
        score += 1
        detail["strong_cpr"] = 1
    if avwap > 0 and float(bar.close) > avwap:
        score += 1
        detail["above_avwap"] = 1
    if adx_value >= config.adx_threshold:
        score += 1
        detail["adx_trending"] = 1
    if sector_flow > 0:
        score += 1
        detail["sector_flow_pos"] = 1
    if bars_today and prior_day_close > 0 and float(bars_today[0].open) > prior_day_close:
        score += 1
        detail["gap_up"] = 1
    return score, detail


def true_range(high: float, low: float, previous_close: float) -> float:
    return max(float(high) - float(low), abs(float(high) - previous_close), abs(float(low) - previous_close))


def atr_from_daily_rows(rows: list[dict], period: int = 14) -> float:
    if len(rows) < 2:
        return 0.0
    ranges: list[float] = []
    prev_close = float(rows[0]["close"])
    for row in rows[1:]:
        ranges.append(true_range(float(row["high"]), float(row["low"]), prev_close))
        prev_close = float(row["close"])
    sample = ranges[-period:] if len(ranges) >= period else ranges
    return fmean(sample) if sample else 0.0
