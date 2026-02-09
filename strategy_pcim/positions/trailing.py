"""EOD Trailing Stop Update."""

from loguru import logger

from .manager import PCIMPosition
from ..config.constants import EXITS


def update_trailing_stop_eod(pos: PCIMPosition, close_today: float, atr20_today: float) -> None:
    """
    Update trailing stop at end of day.
    trail_level = max(previous_trail, close - 1.5 x ATR20)
    stop = max(initial_stop, trail_level)
    """
    new_trail = close_today - (EXITS["TRAIL_ATR"] * atr20_today)

    pos.trailing_stop = max(pos.trailing_stop, new_trail)
    pos.current_stop = max(pos.initial_stop, pos.trailing_stop)
    pos.max_price = max(pos.max_price, close_today)

    logger.debug(f"{pos.symbol}: Trail update - close={close_today:.0f}, "
                 f"trail={pos.trailing_stop:.0f}, stop={pos.current_stop:.0f}")
