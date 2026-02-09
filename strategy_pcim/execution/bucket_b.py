"""Bucket B: VWAP Touch + Reclaim Trigger."""

from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

from .vwap import compute_vwap_series, check_vwap_touch, check_vwap_reclaim
from ..config.constants import BUCKET_B


@dataclass
class BucketBSignal:
    """Bucket B trigger signal."""
    triggered: bool
    reason: str
    touch_idx: Optional[int] = None
    reclaim_idx: Optional[int] = None
    vwap: float = 0.0
    bar: Optional[dict] = None


def check_bucket_b_trigger(bars_1m: List[dict]) -> BucketBSignal:
    """
    Check Bucket B VWAP touch + reclaim trigger.

    Trigger: touch at time k, reclaim at time t where t in [k, k+2].
    """
    if len(bars_1m) < 3:
        return BucketBSignal(False, "INSUFFICIENT_BARS")

    vwap_series = compute_vwap_series(bars_1m)
    window = BUCKET_B["TOUCH_RECLAIM_WINDOW_MINS"]

    touch_idx = None
    for i in range(len(bars_1m) - 1, max(-1, len(bars_1m) - 6), -1):
        if check_vwap_touch(bars_1m[i], vwap_series[i]):
            touch_idx = i
            break

    if touch_idx is None:
        return BucketBSignal(False, "NO_TOUCH")

    for j in range(touch_idx, min(touch_idx + window + 1, len(bars_1m))):
        if j == 0:
            continue
        if check_vwap_reclaim(
            bars_1m[j - 1], vwap_series[j - 1],
            bars_1m[j], vwap_series[j],
        ):
            logger.info(f"Bucket B triggered: touch@{touch_idx}, reclaim@{j}")
            return BucketBSignal(
                triggered=True,
                reason="TRIGGERED",
                touch_idx=touch_idx,
                reclaim_idx=j,
                vwap=vwap_series[j],
                bar=bars_1m[j],
            )

    return BucketBSignal(False, "NO_RECLAIM")
