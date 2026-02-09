"""KRX tick size table for cash equities."""

from __future__ import annotations

# KRX price-band tick sizes (KRW, effective 2023-01-02).
_BANDS = (
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
)
_TOP_TICK = 1_000


def tick_size(price: float) -> float:
    """Return the KRX tick size for *price*."""
    for upper, ts in _BANDS:
        if price < upper:
            return float(ts)
    return float(_TOP_TICK)


def round_to_tick(price: float, ts: float | None = None) -> float:
    """Round *price* down to the nearest tick."""
    if ts is None:
        ts = tick_size(price)
    return int(price / ts) * ts
