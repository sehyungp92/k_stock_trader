from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable

from strategy_common.clock import KST
from strategy_common.market import MarketBar


FIRST30_START = time(9, 0)
FIRST30_END = time(9, 30)
FIRST30_BAR_COUNT = 6


@dataclass(frozen=True, slots=True)
class KALCBFirst30Features:
    symbol: str
    trade_date: date
    bar_count: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    expected_30m_volume: float
    first30_ret: float
    vwap_ret: float
    gap: float
    rel_volume: float
    range_close_location: float
    signal_bar_cpr: float
    open_drawdown: float
    low_vs_prev_close: float
    range_atr: float
    signal_bar_timestamp: str
    rank_tiebreak_symbol: str

    def metadata(self) -> dict[str, float | int | str]:
        rel_volume_log = math.log1p(max(float(self.rel_volume), 0.0))
        gap_retention_ratio = float(self.low_vs_prev_close) / max(abs(float(self.gap)), 1e-6) if self.gap > 0.0 else 0.0
        return {
            "first30_bar_count": int(self.bar_count),
            "first30_open": float(self.open),
            "first30_high": float(self.high),
            "first30_low": float(self.low),
            "first30_close": float(self.close),
            "first30_volume": float(self.volume),
            "first30_vwap": float(self.vwap),
            "first30_expected_30m_volume": float(self.expected_30m_volume),
            "first30_ret": float(self.first30_ret),
            "first30_vwap_ret": float(self.vwap_ret),
            "first30_gap": float(self.gap),
            "first30_rel_volume": float(self.rel_volume),
            "first30_range_close_location": float(self.range_close_location),
            "first30_signal_bar_cpr": float(self.signal_bar_cpr),
            "first30_open_drawdown": float(self.open_drawdown),
            "first30_low_vs_prev_close": float(self.low_vs_prev_close),
            "first30_range_atr": float(self.range_atr),
            "first30_gap_retention_ratio": float(gap_retention_ratio),
            "first30_gap_relvol": float(self.gap * rel_volume_log),
            "first30_low_vs_prev_relvol": float(self.low_vs_prev_close * rel_volume_log),
            "first30_signal_bar": self.signal_bar_timestamp,
            "first30_rank_tiebreak_symbol": self.rank_tiebreak_symbol,
        }


def completed_first30_bars(bars: Iterable[MarketBar], *, required_count: int = FIRST30_BAR_COUNT) -> tuple[MarketBar, ...]:
    window_by_time: dict[datetime, MarketBar] = {}
    session_date: date | None = None
    for bar in sorted(bars, key=lambda item: item.timestamp):
        local_ts = bar.timestamp.astimezone(KST)
        t = local_ts.time()
        if FIRST30_START <= t < FIRST30_END:
            if not bar.is_completed:
                raise ValueError(f"KALCB first30 feature requires completed bars: {bar.symbol} {bar.timestamp.isoformat()}")
            if bar.timeframe.lower() != "5m":
                raise ValueError("KALCB first30 feature requires 5m bars")
            if local_ts.second != 0 or local_ts.microsecond != 0 or local_ts.minute % 5 != 0:
                raise ValueError(f"KALCB first30 feature requires aligned 5m bars: {bar.symbol} {local_ts.isoformat()}")
            normalized = local_ts.replace(second=0, microsecond=0)
            if session_date is None:
                session_date = normalized.date()
            if normalized in window_by_time:
                raise ValueError(f"KALCB first30 feature found duplicate 5m bar: {bar.symbol} {normalized.isoformat()}")
            window_by_time[normalized] = bar
    if session_date is None:
        return ()
    expected_count = max(1, int(required_count))
    start = datetime.combine(session_date, FIRST30_START, tzinfo=KST)
    expected_times = tuple(start + timedelta(minutes=5 * index) for index in range(expected_count))
    return tuple(window_by_time[item] for item in expected_times if item in window_by_time)


def build_first30_features(
    bars: Iterable[MarketBar],
    *,
    prior_close: float,
    daily_atr: float,
    expected_30m_volume: float,
    required_count: int = FIRST30_BAR_COUNT,
) -> KALCBFirst30Features | None:
    first = completed_first30_bars(bars, required_count=required_count)
    if len(first) < required_count:
        return None
    open_ = max(float(first[0].open), 1e-9)
    high = max(float(bar.high) for bar in first)
    low = min(float(bar.low) for bar in first)
    close = float(first[-1].close)
    volume = sum(max(float(bar.volume), 0.0) for bar in first)
    vwap_num = sum(((float(bar.high) + float(bar.low) + float(bar.close)) / 3.0) * max(float(bar.volume), 0.0) for bar in first)
    vwap = vwap_num / volume if volume > 0.0 else close
    width = max(high - low, 1e-9)
    prior = max(float(prior_close), 1e-9)
    atr = max(float(daily_atr), 1e-9)
    expected = max(float(expected_30m_volume), 1.0)
    signal_bar = first[-1]
    return KALCBFirst30Features(
        symbol=str(first[0].symbol),
        trade_date=signal_bar.timestamp.astimezone(KST).date(),
        bar_count=len(first),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=max(vwap, 1e-9),
        expected_30m_volume=expected,
        first30_ret=close / open_ - 1.0,
        vwap_ret=close / max(vwap, 1e-9) - 1.0,
        gap=open_ / prior - 1.0,
        rel_volume=volume / expected,
        range_close_location=(close - low) / width,
        signal_bar_cpr=_close_location(signal_bar),
        open_drawdown=low / open_ - 1.0,
        low_vs_prev_close=low / prior - 1.0,
        range_atr=(high - low) / atr,
        signal_bar_timestamp=signal_bar.timestamp.isoformat(),
        rank_tiebreak_symbol=str(first[0].symbol),
    )


def _close_location(bar: MarketBar) -> float:
    return (float(bar.close) - float(bar.low)) / max(float(bar.high) - float(bar.low), 1e-9)
