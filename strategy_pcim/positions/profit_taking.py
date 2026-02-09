"""Profit-Taking Logic."""

from loguru import logger

from .manager import PCIMPosition
from ..config.constants import EXITS


def check_take_profit(pos: PCIMPosition, current_price: float) -> tuple:
    """
    Check if take-profit level reached.
    At +2.5 ATR from entry: sell 60%.
    Returns (should_take, qty_to_sell).
    """
    if pos.tp_done:
        return False, 0

    tp_price = pos.entry_price + (EXITS["TAKE_PROFIT_ATR"] * pos.atr_at_entry)

    if current_price >= tp_price:
        qty_to_sell = int(pos.remaining_qty * EXITS["TAKE_PROFIT_PCT"])
        logger.info(f"{pos.symbol}: Take profit @ {current_price:.0f} "
                     f"(target={tp_price:.0f}), sell {qty_to_sell}")
        return True, qty_to_sell

    return False, 0
