from __future__ import annotations

from datetime import date, datetime, timedelta
from hashlib import sha256
from typing import Iterable

from backtests.core.replay_bundle import EventReplayBundle
from backtests.core.replay_events import ReplayEvent
from strategy_common.clock import KST
from strategy_common.market import MarketBar


def make_intraday_bars(
    symbol: str,
    *,
    trade_date: date = date(2026, 1, 5),
    start_price: float = 10_000.0,
    minutes: int = 80,
    pattern: str = "breakout",
    timeframe: str = "1m",
) -> list[MarketBar]:
    bars: list[MarketBar] = []
    timestamp = datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=9)
    price = float(start_price)
    for index in range(minutes):
        if pattern == "breakout":
            drift = _breakout_drift(index)
        elif pattern == "pullback":
            drift = _pullback_drift(index)
        elif pattern == "dip_confirm":
            drift = _dip_confirm_drift(index)
        else:
            drift = 0.0005
        open_price = price
        close = max(100.0, price * (1.0 + drift))
        high = max(open_price, close) * (1.0 + 0.0015)
        low = min(open_price, close) * (1.0 - 0.0015)
        volume = 8_000 + index * 90
        if 15 <= index <= 22:
            volume *= 2.4
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=timestamp + timedelta(minutes=index),
                timeframe=timeframe,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                source="synthetic",
                source_fingerprint=synthetic_fingerprint(symbol, pattern),
            )
        )
        price = close
    return bars


def synthetic_fingerprint(symbol: str, pattern: str) -> str:
    return sha256(f"synthetic:{symbol}:{pattern}:v1".encode("utf-8")).hexdigest()


def bars_fingerprint(bars: Iterable[MarketBar]) -> str:
    raw = "|".join(f"{bar.symbol}:{bar.timestamp.isoformat()}:{bar.close:.4f}:{bar.volume:.2f}" for bar in bars)
    return sha256(raw.encode("utf-8")).hexdigest()


def make_strategy_synthetic_bars(strategy: str, config: dict | None = None) -> list[MarketBar]:
    config = dict(config or {})
    key = strategy.lower()
    if key == "kalcb":
        return make_kalcb_synthetic_bars(str(config.get("symbol", "005930")))
    raise ValueError(f"Unsupported synthetic strategy: {strategy}")


def make_synthetic_replay_bundle(strategy: str, config: dict | None = None) -> EventReplayBundle:
    bars = make_strategy_synthetic_bars(strategy, config)
    return EventReplayBundle(
        events=tuple(ReplayEvent.from_bar(bar) for bar in bars),
        source_fingerprint=bars_fingerprint(bars),
        metadata={"capability_level": "synthetic", "strategy": strategy.lower()},
    )


def _breakout_drift(index: int) -> float:
    if index < 15:
        if index < 5:
            return 0.003
        if index < 10:
            return -0.004
        return 0.0025
    if index == 16:
        return 0.035
    if 17 <= index <= 18:
        return -0.018
    if index == 19:
        return 0.010
    if 20 <= index <= 38:
        return 0.0045
    if 39 <= index <= 55:
        return -0.001
    return 0.0006


def make_kalcb_synthetic_bars(
    symbol: str,
    *,
    trade_date: date = date(2026, 1, 5),
    start_price: float = 70_300.0,
) -> list[MarketBar]:
    bars: list[MarketBar] = []
    timestamp = datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=9)
    rows = [
        (70_300.0, 70_410.0, 70_220.0, 70_330.0, 12_000.0),
        (70_330.0, 70_460.0, 70_210.0, 70_390.0, 11_600.0),
        (70_390.0, 70_440.0, 70_200.0, 70_260.0, 10_900.0),
        (70_260.0, 70_430.0, 70_230.0, 70_360.0, 11_800.0),
        (70_360.0, 70_470.0, 70_240.0, 70_410.0, 12_100.0),
        (70_410.0, 70_500.0, 70_200.0, 70_440.0, 12_300.0),
        (70_450.0, 70_700.0, 70_420.0, 70_660.0, 60_000.0),
        (70_680.0, 71_000.0, 70_600.0, 70_900.0, 42_000.0),
        (71_060.0, 71_350.0, 70_950.0, 71_250.0, 38_000.0),
        (71_250.0, 71_500.0, 71_100.0, 71_420.0, 35_000.0),
        (71_420.0, 71_700.0, 71_300.0, 71_600.0, 34_000.0),
        (71_600.0, 71_850.0, 71_480.0, 71_760.0, 33_000.0),
    ]
    for index, (open_price, high, low, close, volume) in enumerate(rows):
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=timestamp + timedelta(minutes=5 * index),
                timeframe="5m",
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                source="synthetic",
                source_fingerprint=synthetic_fingerprint(symbol, "kalcb_breakout"),
                metadata={"adx": 24.0} if index >= 6 else {},
            )
        )
    return bars


def _pullback_drift(index: int) -> float:
    if index < 8:
        return 0.006
    if index < 22:
        return -0.006
    if index < 27:
        return 0.004
    if index < 45:
        return 0.003
    return -0.001


def _dip_confirm_drift(index: int) -> float:
    if index < 10:
        return 0.004
    if index < 18:
        return -0.005
    if index < 26:
        return -0.001
    if index < 36:
        return 0.004
    return 0.001
