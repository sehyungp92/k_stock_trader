"""Bucket A: Opening Range Bar Trigger."""

from dataclasses import dataclass
from typing import Optional
from loguru import logger

from ..config.constants import BUCKET_A


@dataclass
class BucketASignal:
    """Bucket A trigger signal."""
    triggered: bool
    reason: str
    bar: Optional[dict] = None
    vol_ratio: float = 0.0


def check_bucket_a_trigger(
    bar_3m: dict,
    baseline_volume: float,
    vol_threshold: float = None,
) -> BucketASignal:
    """
    Check Bucket A opening range bar trigger.

    Conditions:
    - Close in top 30% of range
    - Volume >= threshold of typical baseline (default 120%, adaptive via hit-rate)

    Args:
        bar_3m: 3-minute OHLCV bar
        baseline_volume: Typical opening 3-minute volume (20-day average)
        vol_threshold: Volume ratio threshold (default from BUCKET_A config, or adaptive)
    """
    high = float(bar_3m.get('high', 0))
    low = float(bar_3m.get('low', 0))
    close = float(bar_3m.get('close', 0))
    volume = float(bar_3m.get('volume', 0))

    bar_range = high - low
    if bar_range <= 0:
        return BucketASignal(False, "ZERO_RANGE")

    close_pos = (close - low) / bar_range
    top_range_threshold = 1.0 - BUCKET_A["ORB_TOP_RANGE_PCT"]

    if close_pos < top_range_threshold:
        return BucketASignal(False, f"CLOSE_NOT_STRONG_{close_pos:.2f}")

    # Use adaptive threshold if provided, otherwise fall back to config default
    threshold = vol_threshold if vol_threshold is not None else BUCKET_A["VOL_RATIO_THRESHOLD"]
    vol_ratio = volume / baseline_volume if baseline_volume > 0 else 0
    if vol_ratio < threshold:
        logger.debug(f"Bucket A vol check: baseline={baseline_volume:.0f}, actual_3m_vol={volume:.0f}, ratio={vol_ratio:.2f}")
        return BucketASignal(False, f"VOLUME_LOW_{vol_ratio:.2f}", vol_ratio=vol_ratio)

    logger.info(f"Bucket A triggered: close_pos={close_pos:.2f}, vol_ratio={vol_ratio:.2f}, "
                f"baseline={baseline_volume:.0f}, actual_3m_vol={volume:.0f}")
    return BucketASignal(
        triggered=True,
        reason="TRIGGERED",
        bar=bar_3m,
        vol_ratio=vol_ratio,
    )
