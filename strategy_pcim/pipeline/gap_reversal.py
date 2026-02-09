"""Gap reversal rate calculation."""

from dataclasses import dataclass
from typing import List
from loguru import logger

from ..config.constants import GAP_REVERSAL


@dataclass
class GapReversalResult:
    """Gap reversal rate calculation result."""
    event_count: int
    reversal_count: int
    rate: float
    insufficient_sample: bool


def compute_gap_reversal_rate(daily_bars: List[dict]) -> GapReversalResult:
    """
    Compute gap reversal rate from daily bars.

    Gap-up event: open > prev_close by >= 1%
    Reversal: close < open on that day
    Rate: reversal_count / event_count

    Args:
        daily_bars: List of daily bar dicts with open, close keys
                   (sorted oldest to newest)
    """
    lookback = GAP_REVERSAL["LOOKBACK_DAYS"]
    min_gap = GAP_REVERSAL["GAP_EVENT_MIN_PCT"]
    min_events = GAP_REVERSAL["MIN_EVENTS"]

    bars = daily_bars[-lookback:] if len(daily_bars) > lookback else daily_bars

    event_count = 0
    reversal_count = 0

    for i in range(1, len(bars)):
        prev_close = bars[i - 1].get('close', 0)
        open_price = bars[i].get('open', 0)
        close_price = bars[i].get('close', 0)

        if prev_close <= 0:
            continue

        gap_pct = (open_price - prev_close) / prev_close

        if gap_pct >= min_gap:
            event_count += 1
            if close_price < open_price:
                reversal_count += 1

    insufficient = event_count < min_events
    rate = reversal_count / event_count if event_count > 0 else 0.0

    logger.debug(f"Gap reversal: events={event_count}, reversals={reversal_count}, "
                 f"rate={rate:.1%}, insufficient={insufficient}")

    return GapReversalResult(
        event_count=event_count,
        reversal_count=reversal_count,
        rate=rate,
        insufficient_sample=insufficient,
    )
