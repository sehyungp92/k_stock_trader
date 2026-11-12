"""Observed-volume liquidation urgency, not a promise of future fills.

The lower empirical volume quantile is a deliberately simple forecast. It is
only used to request an exit; the broker still decides every executable fill.
"""
from datetime import datetime, timedelta
from math import floor, isfinite

from kis_core.trading_calendar import get_trading_calendar
from strategy_common.execution_session import bar_execution_issue, bar_has_periodic_auction


def liquidation_evidence(symbol_state, position, config):
    if not config.exit_liquidity_guard_enabled:
        return None
    recent = symbol_state.bars_today[-config.exit_liquidity_guard_lookback:]
    if len(recent) < config.exit_liquidity_guard_lookback:
        return None
    last = recent[-1]
    available = last.timestamp + timedelta(minutes=5)
    if available.time() < config.exit_liquidity_guard_start:
        return None
    if any(b.symbol != position.symbol or b.timestamp.date() != last.timestamp.date()
           or b.timeframe != "5m" or not b.is_completed or not isfinite(b.volume)
           or b.volume < 0 or bar_execution_issue(b) not in (None, "zero_volume")
           or bar_has_periodic_auction(b) for b in recent):
        return None
    if any(b.timestamp - a.timestamp != timedelta(minutes=5) for a, b in zip(recent, recent[1:])):
        return None
    close = get_trading_calendar().session_hours(available.date())[1]
    deadline = datetime.combine(available.date(), close, tzinfo=available.tzinfo) - timedelta(minutes=10)
    slots = max(0, int((deadline - available).total_seconds() // 300))
    volumes = sorted(b.volume for b in recent)
    # An observed lower order statistic; do not interpolate optimistic volume.
    volume = volumes[floor((len(volumes) - 1) * config.exit_liquidity_guard_quantile)]
    per_slice = floor(volume * config.exit_liquidity_guard_participation)
    forecast = floor(slots * per_slice * (1 - config.exit_liquidity_guard_reserve))
    if position.qty_open <= forecast:
        return None
    return {"basis": "observed_volume_lower_quantile_extrapolation_v1",
            "available_at": available.isoformat(), "deadline": deadline.isoformat(),
            "observations": len(recent), "volume_order_statistic": volume,
            "participation_assumption": config.exit_liquidity_guard_participation,
            "reserve": config.exit_liquidity_guard_reserve, "remaining_open_slices": slots,
            "forecast_capacity": forecast, "inventory": position.qty_open,
            "future_liquidity_verified": False}
