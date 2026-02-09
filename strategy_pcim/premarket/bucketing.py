"""Gap Bucketing (08:40-09:00 KST)."""

from loguru import logger

from ..pipeline.candidate import Candidate
from ..config.constants import BUCKETS


def classify_bucket(gap_pct: float) -> str:
    """Classify gap into bucket A/B/D."""
    if BUCKETS["A"]["min"] <= gap_pct < BUCKETS["A"]["max"]:
        return "A"
    elif BUCKETS["B"]["min"] <= gap_pct < BUCKETS["B"]["max"]:
        return "B"
    else:
        return "D"


def apply_bucketing(c: Candidate, expected_open: float, regime) -> Candidate:
    """
    Apply gap bucketing to candidate.

    Args:
        c: Candidate
        expected_open: Expected open price from auction
        regime: RegimeResult
    """
    if c.is_rejected():
        return c

    prev_close = c.close_prev
    gap_pct = (expected_open - prev_close) / prev_close if prev_close > 0 else 0

    c.expected_open = expected_open
    c.gap_pct = gap_pct
    c.bucket = classify_bucket(gap_pct)

    logger.info(
        f"BUCKET_CLASSIFY: {c.symbol} expected_open={expected_open:.0f} "
        f"prev_close={prev_close:.0f} gap={gap_pct:.2%} -> Bucket {c.bucket}"
    )

    if c.bucket == "D":
        logger.info(
            f"{c.symbol}: REJECTED NO_TRADE_BUCKET_D "
            f"(gap={gap_pct:.2%} outside A={BUCKETS['A']['min']:.1%}-{BUCKETS['A']['max']:.1%}, "
            f"B={BUCKETS['B']['min']:.1%}-{BUCKETS['B']['max']:.1%})"
        )
        c.reject_reason = "NO_TRADE_BUCKET_D"
        return c

    if regime.disable_bucket_a and c.bucket == "A":
        logger.info(f"{c.symbol}: REJECTED REGIME_DISALLOWS_BUCKET_A (regime={regime.name})")
        c.reject_reason = "REGIME_DISALLOWS_BUCKET_A"
        return c

    return c
