"""20DMA Trend Gate."""

from typing import List
from loguru import logger

from kis_core import sma


def check_trend_gate(closes: List[float]) -> bool:
    """
    Check 20DMA trend gate.
    Pass if: prior_close > 20DMA

    Args:
        closes: List of daily closes (oldest to newest)
    """
    if len(closes) < 20:
        logger.warning("Insufficient data for trend gate")
        return False

    sma20_values = sma(closes, 20)
    if not sma20_values:
        return False

    current_close = closes[-1]
    current_sma20 = sma20_values[-1]

    passes = current_close > current_sma20
    logger.debug(f"Trend gate: close={current_close:.0f}, SMA20={current_sma20:.0f}, pass={passes}")
    return passes
