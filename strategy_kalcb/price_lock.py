"""Completed-print price locks; emit evidence, never assume executable depth."""
from __future__ import annotations

from datetime import timedelta
from math import isfinite
from typing import Any, Sequence

from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailyCandidate


def price_lock_evidence(candidate: KALCBDailyCandidate | None, bars: Sequence[MarketBar], config: KALCBConfig) -> dict[str, Any] | None:
    count = config.exit_price_lock_bars
    if count < 2 or candidate is None or len(bars) < count:
        return None
    recent = bars[-count:]
    day = recent[-1].timestamp.date()
    price = float(recent[-1].close)
    if candidate.trade_date != day or not isfinite(price) or price <= 0:
        return None
    if any(b.symbol != candidate.symbol or not b.is_completed or b.timeframe.lower() != "5m" or b.timestamp.date() != day
           or b.volume <= 0 or not (b.open == b.high == b.low == b.close == price) for b in recent):
        return None
    if any(right.timestamp - left.timestamp != timedelta(minutes=5) for left, right in zip(recent, recent[1:])):
        return None
    meta = candidate.metadata
    try:
        sourced_limit = float(meta.get("upper_limit_price") or 0.0)
    except (TypeError, ValueError):
        return None
    limit_day = str(meta.get("price_limit_trade_date") or "")
    source = str(meta.get("price_limit_source") or "")
    if "upper_limit_price" in meta:
        # A supplied but stale/mismatching limit must not silently fall back.
        if not isfinite(sourced_limit) or sourced_limit <= 0 or limit_day != day.isoformat() or not source.strip() or price != sourced_limit:
            return None
        kind, reference = "sourced_upper_limit", sourced_limit
    else:
        prior = float(candidate.prior_day_close)
        if not config.exit_price_lock_allow_prior_close_proxy or not isfinite(prior) or prior <= 0:
            return None
        if price < prior * (1 + config.exit_price_lock_min_prior_close_return):
            return None
        kind, reference = "prior_close_proxy", prior
    return {"reference_kind": kind, "reference_price": reference, "source": source,
            "session": day.isoformat(), "price": price, "completed_bars": count,
            "first_bar": recent[0].timestamp.isoformat(), "last_bar": recent[-1].timestamp.isoformat()}
