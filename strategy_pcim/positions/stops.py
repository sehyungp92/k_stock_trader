"""ATR Stop Logic."""

from loguru import logger

from .manager import PCIMPosition


def check_stop_hit(pos: PCIMPosition, current_price: float) -> bool:
    """Check if current price hit stop level."""
    if current_price <= pos.current_stop:
        logger.info(f"{pos.symbol}: Stop hit @ {current_price:.0f} (stop={pos.current_stop:.0f})")
        return True
    return False
