"""KOSPI Regime Calculation."""

from dataclasses import dataclass
from typing import List
from loguru import logger

from kis_core import sma

from ..config.constants import REGIME


@dataclass
class RegimeResult:
    """Regime calculation result."""
    name: str
    value: float
    max_exposure: float
    disable_bucket_a: bool


def compute_regime(index_closes: List[float]) -> RegimeResult:
    """
    Compute KOSPI regime.

    regime_value = (KOSPI - SMA50) / ATR50

    Regimes:
    - Crisis (< -2): 20% max exposure, no Bucket A
    - Weak (-2 to 0): 50% max exposure, no Bucket A
    - Normal (0 to 2): 80% max exposure
    - Strong (> 2): 100% max exposure
    """
    if len(index_closes) < 50:
        logger.warning("Insufficient data for regime, defaulting to NORMAL")
        return RegimeResult("NORMAL", 0.0, 0.80, False)

    sma50_values = sma(index_closes, 50)

    # Approximate ATR50 with close-to-close changes (use most recent 50)
    n = len(index_closes)
    atr50 = sum(abs(index_closes[i] - index_closes[i-1])
                for i in range(max(1, n-50), n)) / min(50, n-1)

    if atr50 <= 0:
        atr50 = index_closes[-1] * 0.01

    last_close = index_closes[-1]
    sma50 = sma50_values[-1] if sma50_values else last_close

    regime_value = (last_close - sma50) / atr50

    for name, spec in REGIME.items():
        if regime_value < spec["lt"]:
            result = RegimeResult(
                name=name,
                value=regime_value,
                max_exposure=spec["max_exposure"],
                disable_bucket_a=spec["disable_bucket_a"],
            )
            logger.info(f"Regime: {name} (value={regime_value:.2f}, max_exp={spec['max_exposure']:.0%})")
            return result

    return RegimeResult("STRONG", regime_value, 1.0, False)
