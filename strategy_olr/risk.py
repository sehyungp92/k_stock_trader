from __future__ import annotations

import math
from typing import Literal

TickIntent = Literal["buy_limit", "sell_limit", "buy_stop", "sell_stop", "protective_stop"]

_KRX_TICK_BANDS = (
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
)
_KRX_TOP_TICK = 1_000


def tick_size(price: float) -> float:
    for upper, size in _KRX_TICK_BANDS:
        if price < upper:
            return float(size)
    return float(_KRX_TOP_TICK)


def round_price_for_krx(price: float, intent: TickIntent) -> float:
    price_f = max(float(price), 0.0)
    if price_f <= 0.0:
        return 0.0
    if intent == "buy_limit":
        return _floor_tick(price_f)
    if intent in {"sell_limit", "buy_stop", "sell_stop", "protective_stop"}:
        return _ceil_tick(price_f)
    raise ValueError(f"Unsupported tick intent: {intent}")


def _floor_tick(price: float) -> float:
    current = float(price)
    for _ in range(4):
        size = tick_size(current)
        rounded = math.floor(current / size) * size
        if tick_size(rounded or current) == size:
            return float(rounded)
        current = rounded
    return float(current)


def _ceil_tick(price: float) -> float:
    current = float(price)
    for _ in range(4):
        size = tick_size(current)
        rounded = math.ceil(current / size) * size
        if tick_size(rounded) == size:
            return float(rounded)
        current = rounded
    return float(current)
