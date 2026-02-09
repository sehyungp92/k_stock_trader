"""Tradability Tiers."""

from loguru import logger

from ..pipeline.candidate import Candidate
from ..config.constants import TIERS
from ..config.switches import pcim_switches


def classify_tier(adtv_20d: float) -> str:
    """Classify into tier T1/T2/T3 based on ADTV."""
    if adtv_20d >= TIERS["T1"]["adtv_min"]:
        return "T1"
    elif adtv_20d >= TIERS["T2"]["adtv_min"]:
        return "T2"
    else:
        return "T3"


def apply_tier(c: Candidate, switches=None) -> Candidate:
    """
    Apply tradability tier to candidate.

    Args:
        c: Candidate to classify
        switches: Optional PCIMSwitches instance (defaults to global)

    Returns:
        Candidate with tier applied
    """
    if switches is None:
        switches = pcim_switches

    if c.is_rejected():
        return c

    c.tier = classify_tier(c.adtv_20d)
    c.tier_mult = TIERS[c.tier]["size_mult"]

    logger.debug(f"{c.symbol}: ADTV={c.adtv_20d/1e9:.1f}B -> Tier {c.tier}")

    # T3 Bucket A handling with switch
    if c.tier == "T3" and c.bucket == "A":
        if switches.t3_bucket_a_allowed:
            # Permissive: allow T3 Bucket A, but log would-block
            switches.log_would_block(
                c.symbol,
                "T3_BUCKET_A",
                "allowed",
                "blocked",
                {"tier": c.tier, "bucket": c.bucket, "adtv": c.adtv_20d},
            )
        else:
            # Conservative: reject T3 Bucket A
            c.reject_reason = "T3_NO_BUCKET_A"
            return c

    return c
