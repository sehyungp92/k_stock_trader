"""Nulrimok Anchored VWAP (v1 Daily Approximation)."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .db import LRSDatabase
from ..config.constants import (
    ANCHOR_LOOKBACK_DAYS, ANCHOR_MIN_STREAK, IMPULSE_VOL_MULT,
    AVWAP_BAND_PCT,
)


@dataclass
class AVWAPResult:
    anchor_date: Optional[date]
    avwap_ref: float
    band_lower: float
    band_upper: float
    acceptance_pass: bool


def find_smart_money_streak_start(lrs: LRSDatabase, ticker: str) -> Optional[date]:
    flows = lrs.get_daily_flow(ticker, ANCHOR_LOOKBACK_DAYS)
    streak_start, streak_len = None, 0
    for flow in flows:
        if flow.smart_money > 0:
            if streak_len == 0:
                streak_start = flow.date
            streak_len += 1
            if streak_len >= ANCHOR_MIN_STREAK:
                return streak_start
        else:
            streak_len, streak_start = 0, None
    return None


def find_last_impulse_day(lrs: LRSDatabase, ticker: str) -> Optional[date]:
    end_date = date.today()
    bars = lrs.get_daily_bars(ticker, end_date - timedelta(days=ANCHOR_LOOKBACK_DAYS * 2), end_date)
    if len(bars) < 20:
        return None

    # Iterate backward (newest → oldest) to find most recent impulse day
    for i in range(len(bars) - 1, 18, -1):
        avg_vol = sum(b.volume for b in bars[i - 19:i + 1]) / 20
        bar = bars[i]
        if bar.volume > avg_vol * IMPULSE_VOL_MULT and bar.close > (bar.high + bar.low) / 2:
            return bar.date
    return None


def compute_anchored_vwap(lrs: LRSDatabase, ticker: str, anchor_date: date) -> float:
    bars = lrs.get_daily_bars(ticker, anchor_date, date.today() - timedelta(days=1))
    if not bars:
        return 0.0
    cum_vol, cum_pv = 0.0, 0.0
    for bar in bars:
        typical = (bar.high + bar.low + bar.close) / 3
        cum_vol += bar.volume
        cum_pv += typical * bar.volume
    return cum_pv / cum_vol if cum_vol > 0 else 0.0


def compute_avwap_reference(lrs: LRSDatabase, ticker: str) -> AVWAPResult:
    streak_start = find_smart_money_streak_start(lrs, ticker)
    impulse_day = find_last_impulse_day(lrs, ticker)

    candidates = sorted([d for d in [streak_start, impulse_day] if d], reverse=True)

    for anchor in candidates:
        avwap = compute_anchored_vwap(lrs, ticker, anchor)
        if avwap > 0:
            bars = lrs.get_daily_bars(ticker, anchor, date.today() - timedelta(days=1))
            acceptance = any(b.low <= avwap * 1.01 and b.high >= avwap * 0.99 for b in bars)
            if acceptance:
                return AVWAPResult(anchor, avwap, avwap * (1 - AVWAP_BAND_PCT),
                                   avwap * (1 + AVWAP_BAND_PCT), True)

    if candidates:
        avwap = compute_anchored_vwap(lrs, ticker, candidates[0])
        if avwap > 0:
            return AVWAPResult(candidates[0], avwap, avwap * (1 - AVWAP_BAND_PCT),
                               avwap * (1 + AVWAP_BAND_PCT), False)

    return AVWAPResult(None, 0.0, 0.0, 0.0, False)
