"""Execution Vetoes: VI, Spread, Limit checks."""

from typing import Optional
from loguru import logger

from ..config.constants import VETOES
from ..config.switches import pcim_switches


def check_execution_veto(
    quote: dict,
    upper_limit_price: float,
    tick_size: float,
    is_in_vi: bool,
    symbol: str = "",
    switches=None,
) -> Optional[str]:
    """
    Check execution vetoes. Returns veto reason or None.

    Args:
        quote: Quote dict with bid/ask/last
        upper_limit_price: Upper limit price
        tick_size: Tick size for the stock
        is_in_vi: Whether stock is in volatility interruption
        symbol: Stock code for logging
        switches: Optional PCIMSwitches instance (defaults to global)

    Veto if: in VI, within 2 ticks of upper limit, or spread too wide.
    """
    if switches is None:
        switches = pcim_switches

    if is_in_vi:
        return "IN_VI"

    bid = float(quote.get('bid', 0))
    ask = float(quote.get('ask', 0))
    last = float(quote.get('last', 0))

    if upper_limit_price > 0 and tick_size > 0:
        distance_ticks = (upper_limit_price - last) / tick_size
        if distance_ticks <= VETOES["NEAR_UPPER_LIMIT_TICKS"]:
            return f"NEAR_UPPER_LIMIT_{distance_ticks:.1f}ticks"

    if last > 0 and bid > 0 and ask > 0:
        spread_pct = (ask - bid) / last
        spread_threshold = switches.spread_veto_pct

        if spread_pct > spread_threshold:
            return f"SPREAD_TOO_WIDE_{spread_pct:.2%}"

        # Log would-block: passed permissive but would fail strict (0.6%)
        if spread_pct > VETOES["MAX_SPREAD_PCT"]:
            switches.log_would_block(
                symbol or "UNKNOWN",
                "SPREAD_VETO",
                spread_pct,
                VETOES["MAX_SPREAD_PCT"],
            )

    return None
