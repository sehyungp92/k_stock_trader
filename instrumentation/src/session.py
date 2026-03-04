"""KRX session type classification."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))


def classify_session_type(ts: datetime) -> str:
    """Classify KRX session type from timestamp.

    KRX schedule:
    - Pre-market: 08:30-09:00 (simultaneous matching)
    - Regular: 09:00-15:20 (continuous matching)
    - Closing auction: 15:20-15:30
    - After-hours: 15:40-16:00 (single price)
    """
    kst_time = ts.astimezone(KST).time()
    minutes = kst_time.hour * 60 + kst_time.minute

    if minutes < 9 * 60:
        return "pre_market"
    elif minutes < 15 * 60 + 20:
        return "regular"
    elif minutes < 15 * 60 + 30:
        return "closing_auction"
    else:
        return "after_hours"
