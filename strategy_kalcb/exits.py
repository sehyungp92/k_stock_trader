from __future__ import annotations

from strategy_common.market import MarketBar

from .config import KALCBConfig
from .signals import close_location_value, compute_session_vwap


def should_quick_exit(hold_bars: int, unrealized_r: float, config: KALCBConfig) -> bool:
    if not config.quick_exit_enabled or config.quick_exit_bars <= 0:
        return False
    return hold_bars == config.quick_exit_bars and unrealized_r < config.quick_exit_min_r


def should_take_partial(
    *,
    qty_open: int,
    partial_taken: bool,
    partial_order_id: str,
    unrealized_r: float,
    config: KALCBConfig,
) -> bool:
    if not config.use_partial_takes:
        return False
    if partial_taken or partial_order_id:
        return False
    if qty_open <= 1:
        return False
    return unrealized_r >= config.partial_r_trigger


def partial_exit_qty(qty_open: int, config: KALCBConfig) -> int:
    if qty_open <= 1:
        return 0
    fraction = max(0.0, min(float(config.partial_fraction), 0.95))
    return max(1, min(int(qty_open) - 1, int(int(qty_open) * fraction)))


def should_mfe_conviction_exit(hold_bars: int, mfe_r: float, unrealized_r: float, config: KALCBConfig) -> bool:
    if not config.mfe_conviction_enabled or config.mfe_conviction_check_bars <= 0:
        return False
    if hold_bars != config.mfe_conviction_check_bars:
        return False
    if mfe_r >= config.mfe_conviction_min_r:
        return False
    return config.mfe_conviction_floor_r == 0.0 or unrealized_r < config.mfe_conviction_floor_r


def should_flow_reversal(
    bar: MarketBar,
    session_bars: list[MarketBar],
    *,
    entry_price: float,
    hold_bars: int,
    mfe_r: float,
    config: KALCBConfig,
) -> bool:
    if not config.flow_reversal_enabled:
        return False
    if hold_bars < config.flow_reversal_min_hold_bars:
        return False
    if config.flow_reversal_mfe_grace_r > 0 and mfe_r >= config.flow_reversal_mfe_grace_r:
        return False
    if close_location_value(bar) >= config.flow_reversal_cpr_threshold:
        return False
    avwap = compute_session_vwap(session_bars)
    if avwap <= 0 or bar.close >= avwap:
        return False
    if bar.close >= bar.open:
        return False
    return bar.close < entry_price or config.flow_reversal_mfe_grace_r > 0


def trailing_stop_from_mfe(
    *,
    current_stop: float,
    entry_price: float,
    risk_per_share: float,
    mfe_r: float,
    hold_bars: int,
    reference_price: float,
    config: KALCBConfig,
) -> tuple[float, str]:
    if risk_per_share <= 0:
        return current_stop, ""
    candidates: list[tuple[float, str]] = []
    if config.flow_reversal_trailing_activate_r > 0 and mfe_r >= config.flow_reversal_trailing_activate_r:
        trail_r = mfe_r - config.flow_reversal_trailing_distance_r
        if trail_r > 0:
            candidates.append((entry_price + trail_r * risk_per_share, "flow_reversal_trail"))
    if config.adaptive_trail_enabled and config.adaptive_trail_start_bars > 0 and hold_bars >= config.adaptive_trail_start_bars:
        if hold_bars >= config.adaptive_trail_tighten_bars:
            activate = config.adaptive_trail_late_activate_r
            distance = config.adaptive_trail_late_distance_r
            reason = "adaptive_late_trail"
        else:
            activate = config.adaptive_trail_mid_activate_r
            distance = config.adaptive_trail_mid_distance_r
            reason = "adaptive_mid_trail"
        if mfe_r >= activate:
            trail_r = mfe_r - distance
            if trail_r > 0:
                candidates.append((entry_price + trail_r * risk_per_share, reason))
    best = current_stop
    best_reason = ""
    for target, reason in candidates:
        if target > best and target < reference_price:
            best = target
            best_reason = reason
    return best, best_reason


def failure_stop_target(
    *,
    current_stop: float,
    entry_price: float,
    risk_per_share: float,
    close_price: float,
    hold_bars: int,
    mfe_r: float,
    unrealized_r: float,
    config: KALCBConfig,
) -> tuple[float, str]:
    if not config.failure_stop_enabled or config.failure_stop_bars <= 0:
        return current_stop, ""
    if hold_bars != config.failure_stop_bars:
        return current_stop, ""
    if mfe_r > config.failure_stop_mfe_max_r or unrealized_r > config.failure_stop_current_r_max:
        return current_stop, ""
    target = entry_price + config.failure_stop_to_r * risk_per_share
    cap = close_price * (1.0 - config.failure_stop_close_buffer_pct)
    target = min(target, cap)
    if target <= current_stop:
        return current_stop, ""
    return target, "failure_stop_tighten"
