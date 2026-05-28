from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time as dtime
from hashlib import sha256
from statistics import mean, median
from typing import Any, Iterable, Sequence

from strategy_common.clock import KST
from strategy_common.actions import (
    ReplaceProtectiveStop,
    StrategyAction,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
    SubmitProtectiveStop,
)
from strategy_common.market import MarketBar

from .config import OLRConfig
from .models import OLRDailyCandidate
from .risk import round_price_for_krx


EXECUTION_CORE_VERSION = "olr-execution-core-v5"
DECISION_CUTOFF = dtime(14, 30)
DEFAULT_LAST_CONTINUOUS_TIME = dtime(15, 15)
DEFAULT_CLOSE_AUCTION_TIME = dtime(15, 30)
DEFAULT_FLATTEN_TIME = DEFAULT_CLOSE_AUCTION_TIME
NEXT_CLOSE_EXIT_REASON = "next_close"


@dataclass(frozen=True, slots=True)
class OLREntryPlan:
    name: str
    mode: str
    max_signal_bars: int = 1
    after_bar: int = 0
    min_bar_ret: float = -9.99
    min_vwap_ret: float = -9.99
    min_breakout_pct: float = 0.0
    min_close_location: float = 0.0
    max_pullback_from_vwap_pct: float = 0.01
    min_reclaim_ret: float = -9.99
    require_above_decision_close: bool = False
    max_vwap_extension_pct: float = 9.99


@dataclass(frozen=True, slots=True)
class OLRExitPlan:
    name: str
    mode: str = "next_close"
    stop_mode: str = "atr"
    hard_stop_enabled: bool = False
    stop_atr_mult: float = 0.80
    stop_pct: float = 0.008
    target_r: float = 0.0
    partial_trigger_r: float = 0.0
    partial_fraction: float = 0.0
    partial_stop_r: float = 0.0
    breakeven_trigger_r: float = 0.0
    breakeven_stop_r: float = 0.0
    trail_start_r: float = 0.0
    trail_gap_r: float = 0.0
    mfe_fade_start_r: float = 0.0
    mfe_fade_gap_r: float = 0.0
    mfe_fade_floor_r: float = 0.0
    vwap_fail_bars: int = 0
    vwap_fail_pct: float = 0.0
    failed_followthrough_bars: int = 0
    failed_followthrough_mfe_r: float = 0.0
    failed_followthrough_close_r: float = 0.0
    no_mfe_bars: int = 0
    no_mfe_thresh_r: float = 0.0
    max_hold_bars: int = 0


@dataclass(frozen=True, slots=True)
class OLRTradePlan:
    name: str
    entry: OLREntryPlan
    exit: OLRExitPlan


@dataclass(frozen=True, slots=True)
class OLRAllocationPlan:
    name: str
    mode: str = "fixed_slots"
    target_gross_exposure: float = 1.0
    max_position_pct: float = 0.25
    min_selected: int = 1
    rank_decay: float = 1.0


@dataclass(frozen=True, slots=True)
class OLREntrySignal:
    fill_index: int
    signal_index: int
    reason: str
    fill_at_close: bool = False


@dataclass(frozen=True, slots=True)
class OLRExitLeg:
    fraction: float
    price: float
    reason: str
    bar_index: int


@dataclass(frozen=True, slots=True)
class OLRTradeOutcome:
    trade_date: date
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    stop_price: float
    risk_per_share: float
    gross_return_pct: float
    net_return_pct: float
    mfe_r: float
    mae_r: float
    mfe_capture: float
    bars_held: int
    entry_reason: str
    exit_reason: str
    ambiguous_bar_count: int
    stopped: bool
    target_hit: bool
    partial_hit: bool
    metadata: dict[str, Any]


def normalize_action_prices(action: StrategyAction) -> StrategyAction:
    """Normalize neutral OLR actions to conservative KRX tick prices."""

    if isinstance(action, SubmitEntry):
        limit = round_price_for_krx(action.limit_price, "buy_limit") if action.limit_price else None
        stop = round_price_for_krx(action.stop_price, "protective_stop") if action.stop_price else None
        return replace(action, limit_price=limit, stop_price=stop)
    if isinstance(action, (SubmitExit, SubmitPartialExit)):
        limit = round_price_for_krx(action.limit_price, "sell_limit") if action.limit_price else None
        return replace(action, limit_price=limit)
    if isinstance(action, SubmitProtectiveStop):
        return replace(action, stop_price=round_price_for_krx(action.stop_price, "protective_stop"))
    if isinstance(action, ReplaceProtectiveStop):
        return replace(action, stop_price=round_price_for_krx(action.stop_price, "protective_stop"))
    return action


def normalize_actions(actions: Iterable[StrategyAction]) -> list[StrategyAction]:
    return [normalize_action_prices(action) for action in actions]


def action_to_intent(action: StrategyAction):
    from strategy_common.oms_adapter import action_to_intent as shared_action_to_intent

    return shared_action_to_intent(normalize_action_prices(action))


def simulate_olr_trade(
    trade_date: date,
    symbol: str,
    entry_day_bars: Sequence[MarketBar],
    next_day_bars: Sequence[MarketBar],
    candidate: OLRDailyCandidate,
    entry_plan: OLREntryPlan,
    exit_plan: OLRExitPlan,
    config: OLRConfig | None = None,
) -> OLRTradeOutcome | None:
    """Replay one OLR candidate through the shared post-14:30 execution core.

    Entry signals that depend on a completed 5m bar always fill on a later bar.
    The close-auction entry is modeled as a resting close order submitted after
    the 14:30 decision, so it may fill at the entry-day close without observing
    that close for signal generation.
    """

    cfg = config or OLRConfig()
    entry_bars = _ordered_completed_bars(entry_day_bars, trade_date)
    next_bars = _ordered_completed_bars(next_day_bars)
    if not entry_bars or not next_bars:
        return None
    signal = find_olr_entry_signal(entry_bars, candidate, entry_plan)
    if signal is None or signal.fill_index >= len(entry_bars):
        return None
    fill_bar = entry_bars[signal.fill_index]
    submission_index = _entry_submission_index(entry_bars, signal)
    submission_bar = entry_bars[submission_index]
    signal_bar = entry_bars[max(0, min(signal.signal_index, len(entry_bars) - 1))]
    entry_price = max(float(fill_bar.close if signal.fill_at_close else fill_bar.open), 1e-9)
    entry_order_type = "CLOSE_AUCTION" if signal.reason == "close_auction" else "MARKET"
    entry_limit_price = 0.0
    if entry_order_type == "CLOSE_AUCTION":
        entry_limit_price = float(round_price_for_krx(float(submission_bar.close) * (1.0 + float(cfg.auction_limit_offset_bps) / 10_000.0), "buy_limit"))
        auction_fill_price = entry_price * (
            1.0 + (max(float(cfg.slippage_bps), 0.0) + max(float(cfg.auction_adverse_bps), 0.0)) / 10_000.0
        )
        if entry_limit_price > 0.0 and auction_fill_price > entry_limit_price:
            return None
        entry_sizing_price = max(entry_limit_price, 1e-9)
    else:
        entry_sizing_price = max(float(signal_bar.close) * (1.0 + max(float(cfg.market_entry_price_buffer_bps), 0.0) / 10_000.0), 1e-9)
    entry_time = fill_bar.timestamp
    stop = initial_olr_stop_price(entry_bars, candidate, signal, entry_price, exit_plan)
    if stop >= entry_price:
        stop = entry_price * (1.0 - max(float(exit_plan.stop_pct), 0.003))
    risk = max(entry_price - stop, entry_price * 0.001, 1.0)
    post_entry_bars = _post_entry_bars(entry_bars, next_bars, signal)
    if not post_entry_bars:
        return None
    legs, stats = simulate_olr_exits(post_entry_bars, entry_price, stop, risk, exit_plan)
    gross = sum(float(leg.fraction) * (float(leg.price) / entry_price - 1.0) for leg in legs)
    net = gross - round_trip_cost_pct(cfg)
    exit_leg = legs[-1]
    exit_bar = post_entry_bars[max(0, min(int(exit_leg.bar_index), len(post_entry_bars) - 1))]
    high = max(float(bar.high) for bar in post_entry_bars[: exit_leg.bar_index + 1])
    low = min(float(bar.low) for bar in post_entry_bars[: exit_leg.bar_index + 1])
    mfe_r = max(0.0, (high - entry_price) / risk)
    mae_r = (low - entry_price) / risk
    mfe_pct = max(high / entry_price - 1.0, 0.0)
    return OLRTradeOutcome(
        trade_date=trade_date,
        symbol=str(symbol).zfill(6),
        entry_time=entry_time,
        exit_time=exit_bar.timestamp,
        entry_price=entry_price,
        exit_price=float(exit_leg.price),
        stop_price=stop,
        risk_per_share=risk,
        gross_return_pct=gross,
        net_return_pct=net,
        mfe_r=mfe_r,
        mae_r=mae_r,
        mfe_capture=gross / max(mfe_pct, 1e-9) if mfe_pct > 0 else 0.0,
        bars_held=max(1, int(exit_leg.bar_index) + 1),
        entry_reason=signal.reason,
        exit_reason=exit_leg.reason,
        ambiguous_bar_count=int(stats["ambiguous_bar_count"]),
        stopped=bool(stats["stopped"]),
        target_hit=bool(stats["target_hit"]),
        partial_hit=bool(stats["partial_hit"]),
        metadata={
            "execution_core_version": EXECUTION_CORE_VERSION,
            "decision_cutoff": "14:30 KST",
            "entry_plan": asdict(entry_plan),
            "exit_plan": asdict(exit_plan),
            "candidate_rank": candidate.rank,
            "candidate_score": candidate.selection_score,
            "afternoon_score_band_rule": str(candidate.metadata.get("afternoon_score_band_rule") or ""),
            "entry_order_type": entry_order_type,
            "entry_submission_time": submission_bar.timestamp.isoformat(),
            "entry_submission_close": float(submission_bar.close),
            "entry_signal_time": signal_bar.timestamp.isoformat(),
            "entry_signal_close": float(signal_bar.close),
            "entry_sizing_price": entry_sizing_price,
            "entry_limit_price": entry_limit_price,
            "auction_limit_offset_bps": float(cfg.auction_limit_offset_bps) if entry_order_type == "CLOSE_AUCTION" else 0.0,
            "market_entry_price_buffer_bps": float(cfg.market_entry_price_buffer_bps) if entry_order_type == "MARKET" else 0.0,
        },
    )


def find_olr_entry_signal(
    bars: Sequence[MarketBar],
    candidate: OLRDailyCandidate,
    plan: OLREntryPlan,
    *,
    require_fill_bar: bool = True,
) -> OLREntrySignal | None:
    ordered = tuple(sorted(bars, key=lambda bar: bar.timestamp))
    first_index = _first_index_at_or_after(ordered, DECISION_CUTOFF)
    decision_index = _last_index_before(ordered, DECISION_CUTOFF)
    if first_index is None or decision_index is None:
        return None
    if plan.mode == "decision_next_open":
        return OLREntrySignal(fill_index=first_index, signal_index=decision_index, reason="decision_next_open")
    if plan.mode == "close_auction":
        close_index = _close_auction_index(ordered)
        if close_index is None or close_index < first_index:
            return None
        return OLREntrySignal(fill_index=close_index, signal_index=decision_index, reason="close_auction", fill_at_close=True)

    start = min(len(ordered) - 1, first_index + max(0, int(plan.after_bar)))
    last_signal_index = len(ordered) - 2 if require_fill_bar else len(ordered) - 1
    stop = min(last_signal_index, first_index + max(1, int(plan.max_signal_bars)) - 1)
    if stop < start:
        return None
    decision_high = max(float(bar.high) for bar in ordered[: first_index])
    decision_close = float(ordered[decision_index].close)
    touched_vwap = False
    for signal_index in range(start, stop + 1):
        bar = ordered[signal_index]
        vwap = _running_vwap(ordered[: signal_index + 1])
        close_location = _close_location(bar)
        if float(bar.low) <= vwap * (1.0 + plan.max_pullback_from_vwap_pct):
            touched_vwap = True
        if not _common_entry_bar_passes(bar, vwap, close_location, decision_close, plan):
            continue
        if plan.mode == "confirm_next_bar":
            return _next_fill(signal_index, "confirm_next_bar", ordered, require_fill_bar=require_fill_bar)
        if plan.mode == "momentum_breakout":
            level = max(decision_high, float(candidate.prior_day_high or 0.0))
            if float(bar.close) >= level * (1.0 + plan.min_breakout_pct):
                return _next_fill(signal_index, "momentum_breakout", ordered, require_fill_bar=require_fill_bar)
        elif plan.mode == "decision_high_breakout":
            if float(bar.close) >= decision_high * (1.0 + plan.min_breakout_pct):
                return _next_fill(signal_index, "decision_high_breakout", ordered, require_fill_bar=require_fill_bar)
        elif plan.mode == "pdh_breakout":
            if candidate.prior_day_high > 0 and float(bar.close) >= float(candidate.prior_day_high) * (1.0 + plan.min_breakout_pct):
                return _next_fill(signal_index, "pdh_breakout", ordered, require_fill_bar=require_fill_bar)
        elif plan.mode == "vwap_reclaim":
            if touched_vwap and float(bar.close) >= vwap * (1.0 + max(plan.min_reclaim_ret, -0.05)):
                return _next_fill(signal_index, "vwap_reclaim", ordered, require_fill_bar=require_fill_bar)
        elif plan.mode == "pullback_acceptance":
            reclaim = float(bar.close) / max(float(bar.open), 1e-9) - 1.0
            if touched_vwap and reclaim >= plan.min_reclaim_ret and float(bar.close) >= min(float(bar.open), vwap):
                return _next_fill(signal_index, "pullback_acceptance", ordered, require_fill_bar=require_fill_bar)
        elif plan.mode == "late_continuation":
            prior_high = max((float(item.high) for item in ordered[first_index:signal_index]), default=decision_high)
            if float(bar.close) >= prior_high * (1.0 + plan.min_breakout_pct):
                return _next_fill(signal_index, "late_continuation", ordered, require_fill_bar=require_fill_bar)
    return None


def initial_olr_stop_price(
    bars: Sequence[MarketBar],
    candidate: OLRDailyCandidate,
    signal: OLREntrySignal,
    entry_price: float,
    plan: OLRExitPlan,
) -> float:
    ordered = tuple(sorted(bars, key=lambda bar: bar.timestamp))
    signal_bar = ordered[max(0, min(signal.signal_index, len(ordered) - 1))]
    fill_bar = ordered[max(0, min(signal.fill_index, len(ordered) - 1))]
    atr = max(float(candidate.daily_atr), entry_price * 0.006, 1.0)
    if plan.stop_mode == "decision_low":
        decision_lows = [float(bar.low) for bar in ordered if bar.timestamp.astimezone(KST).time() < DECISION_CUTOFF]
        return min(min(decision_lows), entry_price - entry_price * 0.003) if decision_lows else entry_price - plan.stop_atr_mult * atr
    if plan.stop_mode == "signal_low":
        return min(float(signal_bar.low), entry_price - entry_price * 0.003)
    if plan.stop_mode == "fixed_pct":
        return entry_price * (1.0 - max(float(plan.stop_pct), 0.001))
    if plan.stop_mode == "vwap":
        vwap = _running_vwap(ordered[: signal.fill_index + 1])
        return min(vwap * 0.997, entry_price - entry_price * 0.003)
    if plan.stop_mode == "entry_open_gap":
        return min(float(fill_bar.open), entry_price) * (1.0 - max(float(plan.stop_pct), 0.003))
    return entry_price - float(plan.stop_atr_mult) * atr


def simulate_olr_exits(
    bars: Sequence[MarketBar],
    entry: float,
    initial_stop: float,
    risk: float,
    plan: OLRExitPlan,
) -> tuple[tuple[OLRExitLeg, ...], dict[str, float | bool]]:
    ordered = tuple(sorted(bars, key=lambda bar: bar.timestamp))
    if not ordered:
        return (OLRExitLeg(1.0, entry, "missing_exit_bars", 0),), {"ambiguous_bar_count": 0.0, "stopped": False, "target_hit": False, "partial_hit": False}
    if plan.mode == "next_open":
        return (OLRExitLeg(1.0, float(ordered[0].open), "next_open", 0),), {"ambiguous_bar_count": 0.0, "stopped": False, "target_hit": False, "partial_hit": False}
    if plan.mode == "next_close":
        return (OLRExitLeg(1.0, float(ordered[-1].close), NEXT_CLOSE_EXIT_REASON, len(ordered) - 1),), {
            "ambiguous_bar_count": 0.0,
            "stopped": False,
            "target_hit": False,
            "partial_hit": False,
        }

    remaining = 1.0
    active_stop = initial_stop
    pending_stop = initial_stop
    high_water = entry
    partial_done = False
    legs: list[OLRExitLeg] = []
    ambiguous = 0
    stopped = False
    target_hit = False
    partial_hit = False
    vwap_fail_streak = 0
    for index, bar in enumerate(ordered):
        active_targets = []
        if plan.partial_trigger_r > 0 and not partial_done and remaining > 0:
            active_targets.append(entry + plan.partial_trigger_r * risk)
        if plan.target_r > 0 and remaining > 0:
            active_targets.append(entry + plan.target_r * risk)
        if plan.hard_stop_enabled and active_targets and float(bar.low) <= active_stop and float(bar.high) >= min(active_targets):
            ambiguous += 1
        if plan.hard_stop_enabled and float(bar.low) <= active_stop:
            legs.append(OLRExitLeg(remaining, min(active_stop, float(bar.open)) if float(bar.open) < active_stop else active_stop, "hard_stop", index))
            stopped = True
            remaining = 0.0
            break
        if plan.partial_trigger_r > 0 and not partial_done and remaining > 0 and float(bar.high) >= entry + plan.partial_trigger_r * risk:
            fraction = min(max(plan.partial_fraction, 0.0), remaining)
            if fraction > 0:
                legs.append(OLRExitLeg(fraction, entry + plan.partial_trigger_r * risk, "partial_target", index))
                remaining -= fraction
                partial_done = True
                partial_hit = True
                pending_stop = max(pending_stop, entry + plan.partial_stop_r * risk)
        if plan.target_r > 0 and remaining > 0 and float(bar.high) >= entry + plan.target_r * risk:
            legs.append(OLRExitLeg(remaining, entry + plan.target_r * risk, "target", index))
            target_hit = True
            remaining = 0.0
            break
        high_water = max(high_water, float(bar.high))
        next_exit = _completed_bar_exit_reason(ordered, index, entry, risk, high_water, plan, vwap_fail_streak)
        vwap_fail_streak = int(next_exit.pop("vwap_fail_streak"))
        reason = str(next_exit.get("reason") or "")
        if reason and remaining > 0:
            fill_index = min(index + 1, len(ordered) - 1)
            fill_price = float(ordered[fill_index].open) if fill_index > index else float(bar.close)
            legs.append(OLRExitLeg(remaining, fill_price, reason, fill_index))
            remaining = 0.0
            break
        if plan.max_hold_bars > 0 and index + 1 >= plan.max_hold_bars and remaining > 0:
            fill_index = min(index + 1, len(ordered) - 1)
            fill_price = float(ordered[fill_index].open) if fill_index > index else float(bar.close)
            legs.append(OLRExitLeg(remaining, fill_price, "max_hold", fill_index))
            remaining = 0.0
            break
        pending_stop = max(pending_stop, _next_bar_stop_from_completed_bar(entry, risk, active_stop, high_water, plan))
        active_stop = pending_stop
    if remaining > 0:
        legs.append(OLRExitLeg(remaining, float(ordered[-1].close), NEXT_CLOSE_EXIT_REASON, len(ordered) - 1))
    return tuple(legs), {
        "ambiguous_bar_count": float(ambiguous),
        "stopped": stopped,
        "target_hit": target_hit,
        "partial_hit": partial_hit,
    }


def summarize_olr_outcomes(
    outcomes: Sequence[OLRTradeOutcome],
    *,
    session_dates: Sequence[date],
    selection_counts: dict[date, int],
    slot_count: int,
) -> dict[str, float]:
    date_set = set(session_dates)
    scoped = [outcome for outcome in outcomes if outcome.trade_date in date_set]
    by_day: dict[date, list[OLRTradeOutcome]] = {}
    for outcome in scoped:
        by_day.setdefault(outcome.trade_date, []).append(outcome)
    daily_slot_net: list[float] = []
    selected_day_net: list[float] = []
    active_day_net: list[float] = []
    for day in session_dates:
        day_outcomes = by_day.get(day, [])
        slot_net = sum(item.net_return_pct for item in day_outcomes) / max(1, int(slot_count))
        daily_slot_net.append(slot_net)
        selected = max(int(selection_counts.get(day, 0)), 1)
        if selection_counts.get(day, 0) > 0:
            selected_day_net.append(sum(item.net_return_pct for item in day_outcomes) / selected)
        if day_outcomes:
            active_day_net.append(sum(item.net_return_pct for item in day_outcomes) / len(day_outcomes))
    slot_cumulative_net = _compound(daily_slot_net)
    return {
        "selected_count": float(sum(selection_counts.get(day, 0) for day in session_dates)),
        "selected_days": float(sum(1 for day in session_dates if selection_counts.get(day, 0) > 0)),
        "trade_count": float(len(scoped)),
        "active_days": float(len(by_day)),
        "session_count": float(len(session_dates)),
        "signal_conversion": len(scoped) / max(float(sum(selection_counts.get(day, 0) for day in session_dates)), 1.0),
        "active_day_share": len(by_day) / max(float(len(session_dates)), 1.0),
        "selected_day_share": sum(1 for day in session_dates if selection_counts.get(day, 0) > 0) / max(float(len(session_dates)), 1.0),
        "avg_trade_net_pct": _avg(outcome.net_return_pct for outcome in scoped),
        "active_day_net_pct": _avg(active_day_net),
        "selected_day_net_pct": _avg(selected_day_net),
        "calendar_slot_net_pct": _avg(daily_slot_net),
        "slot_cumulative_net_return_pct": slot_cumulative_net,
        "equal_slot_net_return_pct": slot_cumulative_net,
        "max_drawdown_net_pct": _max_drawdown(daily_slot_net),
        "net_win_share": _share(outcome.net_return_pct > 0.0 for outcome in scoped),
        "avg_mfe_r": _avg(outcome.mfe_r for outcome in scoped),
        "median_mfe_r": _median(outcome.mfe_r for outcome in scoped),
        "avg_mae_r": _avg(outcome.mae_r for outcome in scoped),
        "mae_le_neg_1_share": _share(outcome.mae_r <= -1.0 for outcome in scoped),
        "mfe_ge_1_share": _share(outcome.mfe_r >= 1.0 for outcome in scoped),
        "mfe_ge_2_share": _share(outcome.mfe_r >= 2.0 for outcome in scoped),
        "stopout_share": _share(outcome.stopped for outcome in scoped),
        "target_hit_share": _share(outcome.target_hit for outcome in scoped),
        "partial_hit_share": _share(outcome.partial_hit for outcome in scoped),
        "avg_mfe_capture": _avg(outcome.mfe_capture for outcome in scoped),
        "avg_bars_held": _avg(outcome.bars_held for outcome in scoped),
        "ambiguous_bar_count": float(sum(outcome.ambiguous_bar_count for outcome in scoped)),
        **_exit_reason_metrics(scoped),
    }


def summarize_olr_outcomes_with_allocation(
    outcomes: Sequence[OLRTradeOutcome],
    *,
    session_dates: Sequence[date],
    selection_counts: dict[date, int],
    slot_count: int,
    allocation: OLRAllocationPlan | None = None,
) -> dict[str, float]:
    base = summarize_olr_outcomes(
        outcomes,
        session_dates=session_dates,
        selection_counts=selection_counts,
        slot_count=slot_count,
    )
    plan = allocation or OLRAllocationPlan(name="fixed_slots")
    date_set = set(session_dates)
    scoped = [outcome for outcome in outcomes if outcome.trade_date in date_set]
    by_day: dict[date, list[OLRTradeOutcome]] = {}
    for outcome in scoped:
        by_day.setdefault(outcome.trade_date, []).append(outcome)
    daily_net: list[float] = []
    exposures: list[float] = []
    active_weighted: list[float] = []
    for day in session_dates:
        selected = int(selection_counts.get(day, 0) or 0)
        day_outcomes = by_day.get(day, [])
        weights = _allocation_weights(day_outcomes, selected, max(1, int(slot_count)), plan)
        day_net = sum(float(outcome.net_return_pct) * weight for outcome, weight in zip(day_outcomes, weights))
        daily_net.append(day_net)
        exposure = sum(weights)
        exposures.append(exposure)
        if day_outcomes:
            active_weighted.append(day_net)
    base.update(
        {
            "allocation_daily_net_pct": _avg(daily_net),
            "allocation_cumulative_net_return_pct": _compound(daily_net),
            "allocation_max_drawdown_net_pct": _max_drawdown(daily_net),
            "allocation_avg_gross_exposure": _avg(exposures),
            "allocation_avg_active_gross_exposure": _avg(value for value in exposures if value > 0.0),
            "allocation_active_day_net_pct": _avg(active_weighted),
            "allocation_target_gross_exposure": float(plan.target_gross_exposure),
            "allocation_max_position_pct": float(plan.max_position_pct),
            "allocation_min_selected": float(plan.min_selected),
        }
    )
    return base


def summarize_olr_portfolio_proxy(
    outcomes: Sequence[OLRTradeOutcome],
    *,
    session_dates: Sequence[date],
    selection_counts: dict[date, int],
    slot_count: int,
    allocation: OLRAllocationPlan | None = None,
    initial_equity: float = 10_000_000.0,
    config: OLRConfig | None = None,
) -> dict[str, float]:
    """Fast stateful portfolio proxy using cash, integer quantities, and allocation weights."""
    plan = allocation or OLRAllocationPlan(name="fixed_slots")
    cfg = config or OLRConfig()
    by_day: dict[date, list[OLRTradeOutcome]] = {}
    date_set = set(session_dates)
    for outcome in outcomes:
        if outcome.trade_date in date_set:
            by_day.setdefault(outcome.trade_date, []).append(outcome)
    starting_equity = max(float(initial_equity), 1.0)
    cash = starting_equity
    open_positions: list[dict[str, Any]] = []
    daily_net: list[float] = []
    exposures: list[float] = []
    active_net: list[float] = []
    active_exposures: list[float] = []
    qty_zero = 0
    cash_rejected = 0
    symbol_blocked = 0
    deployed = 0
    buy_cost = max(0.0, float(cfg.slippage_bps) + float(cfg.commission_bps)) / 10_000.0
    sell_cost = max(0.0, float(cfg.slippage_bps) + float(cfg.commission_bps) + float(cfg.tax_bps_on_sell)) / 10_000.0
    auction_adverse = max(0.0, float(cfg.auction_adverse_bps)) / 10_000.0
    for day in session_dates:
        cash = _proxy_realize_exits(cash, open_positions, _proxy_day_start(day))
        start_equity = max(_proxy_equity(cash, open_positions), 1.0)
        max_open_notional = _proxy_open_notional(open_positions)
        day_outcomes = sorted(
            by_day.get(day, []),
            key=lambda row: (_proxy_submission_time(row), row.symbol, row.entry_time),
        )
        selected = int(selection_counts.get(day, 0) or 0)
        weights = _allocation_weights(day_outcomes, selected, max(1, int(slot_count)), plan)
        for outcome, weight in zip(day_outcomes, weights):
            cash = _proxy_realize_exits(cash, open_positions, outcome.entry_time)
            if _proxy_symbol_is_open(open_positions, outcome.symbol):
                symbol_blocked += 1
                continue
            equity = max(_proxy_equity(cash, open_positions), 1.0)
            target_notional = equity * max(0.0, float(weight))
            if target_notional <= 0.0:
                continue
            entry_price = max(float(outcome.entry_price), 1e-9)
            sizing_price = max(_metadata_float_value(outcome.metadata, "entry_sizing_price", entry_price), 1e-9)
            entry_cost = buy_cost + (auction_adverse if outcome.entry_reason in {"close_auction", "close_auction_entry"} else 0.0)
            spend = min(target_notional, cash)
            qty = int(spend // (sizing_price * (1.0 + entry_cost)))
            if qty <= 0:
                qty_zero += 1
                if cash <= 0.0:
                    cash_rejected += 1
                continue
            notional = float(qty) * entry_price
            entry_cash = notional * (1.0 + entry_cost)
            if entry_cash > cash + 1e-9:
                cash_rejected += 1
                continue
            exit_cost = sell_cost + (auction_adverse if outcome.exit_reason == NEXT_CLOSE_EXIT_REASON else 0.0)
            exit_cash = notional * (1.0 + float(outcome.gross_return_pct) - exit_cost)
            cash = max(0.0, cash - entry_cash)
            open_positions.append(
                {
                    "exit_time": outcome.exit_time,
                    "symbol": str(outcome.symbol).zfill(6),
                    "notional": notional,
                    "entry_cash": entry_cash,
                    "exit_cash": max(0.0, exit_cash),
                }
            )
            deployed += 1
            max_open_notional = max(max_open_notional, _proxy_open_notional(open_positions))
        cash = _proxy_realize_exits(cash, open_positions, _proxy_day_end(day))
        end_equity = max(_proxy_equity(cash, open_positions), 1.0)
        day_return = end_equity / start_equity - 1.0
        daily_net.append(day_return)
        exposure = max_open_notional / start_equity
        exposures.append(exposure)
        if day_outcomes:
            active_net.append(day_return)
        if exposure > 0.0:
            active_exposures.append(exposure)
    cash = _proxy_realize_exits(cash, open_positions, datetime.max.replace(tzinfo=KST))
    equity = max(_proxy_equity(cash, open_positions), 1.0)
    return {
        "portfolio_proxy_daily_net_pct": _avg(daily_net),
        "portfolio_proxy_net_return_pct": equity / starting_equity - 1.0,
        "portfolio_proxy_max_drawdown_pct": _max_drawdown(daily_net),
        "portfolio_proxy_avg_gross_exposure_pct": _avg(exposures),
        "portfolio_proxy_avg_active_gross_exposure_pct": _avg(active_exposures),
        "portfolio_proxy_active_day_net_pct": _avg(active_net),
        "portfolio_proxy_qty_zero_count": float(qty_zero),
        "portfolio_proxy_cash_rejected_count": float(cash_rejected),
        "portfolio_proxy_symbol_blocked_count": float(symbol_blocked),
        "portfolio_proxy_deployed_trade_count": float(deployed),
        "portfolio_proxy_stateful_cash": 1.0,
    }


def olr_outcome_hash(outcomes: Sequence[OLRTradeOutcome]) -> str:
    payload = [
        {
            "trade_date": item.trade_date.isoformat(),
            "symbol": item.symbol,
            "entry_time": item.entry_time.isoformat(),
            "exit_time": item.exit_time.isoformat(),
            "entry_price": round(item.entry_price, 8),
            "exit_price": round(item.exit_price, 8),
            "net_return_pct": round(item.net_return_pct, 12),
            "mfe_r": round(item.mfe_r, 8),
            "mae_r": round(item.mae_r, 8),
            "entry_reason": item.entry_reason,
            "exit_reason": item.exit_reason,
        }
        for item in sorted(outcomes, key=lambda row: (row.trade_date, row.symbol, row.entry_time, row.exit_time, row.entry_reason, row.exit_reason))
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(raw.encode("utf-8")).hexdigest()


def _allocation_weights(
    outcomes: Sequence[OLRTradeOutcome],
    selected_count: int,
    slot_count: int,
    plan: OLRAllocationPlan,
) -> list[float]:
    if not outcomes or selected_count < max(1, int(plan.min_selected)):
        return [0.0 for _ in outcomes]
    target = max(0.0, float(plan.target_gross_exposure))
    cap = max(0.0, float(plan.max_position_pct))
    selected = max(1, int(selected_count))
    mode = str(plan.mode or "fixed_slots")
    if mode == "fixed_slots":
        return [1.0 / max(1, int(slot_count)) for _ in outcomes]
    if mode == "selected_equal":
        return [target / selected for _ in outcomes]
    if mode in {"capped_equal", "selected_equal_capped"}:
        return [min(cap, target / selected) for _ in outcomes]
    if mode == "rank_weighted":
        selected_ranks = max(selected, max((int((outcome.metadata or {}).get("candidate_rank", 0) or 0) for outcome in outcomes), default=0))
        raw = [
            1.0 / (rank ** max(float(plan.rank_decay), 0.0))
            for rank in range(1, selected_ranks + 1)
        ]
        rank_weights = _cap_normalized_weights(raw, target=target, cap=cap)
        weights = []
        for outcome in outcomes:
            rank = int((outcome.metadata or {}).get("candidate_rank", 0) or 0)
            rank = max(rank, 1)
            weights.append(rank_weights[rank - 1] if rank <= len(rank_weights) else 0.0)
        return weights
    if mode in {"score_weighted", "rank_score_weighted"}:
        raw = []
        for index, outcome in enumerate(outcomes, start=1):
            score = max(_metadata_float_value(outcome.metadata, "candidate_score", 0.0), 0.0)
            rank_component = 1.0
            if mode == "rank_score_weighted":
                rank = int((outcome.metadata or {}).get("candidate_rank", index) or index)
                rank_component = 1.0 / (max(rank, 1) ** max(float(plan.rank_decay), 0.0))
            raw.append(score * rank_component)
        if sum(raw) <= 0.0:
            raw = [1.0 for _ in outcomes]
        return _cap_normalized_weights(raw, target=target, cap=cap)
    return [1.0 / max(1, int(slot_count)) for _ in outcomes]


def _proxy_day_start(day: date) -> datetime:
    return datetime.combine(day, dtime.min, tzinfo=KST)


def _proxy_day_end(day: date) -> datetime:
    return datetime.combine(day, DEFAULT_CLOSE_AUCTION_TIME, tzinfo=KST)


def _proxy_open_notional(open_positions: Sequence[dict[str, Any]]) -> float:
    return sum(float(position.get("notional", 0.0) or 0.0) for position in open_positions)


def _proxy_symbol_is_open(open_positions: Sequence[dict[str, Any]], symbol: str) -> bool:
    key = str(symbol).zfill(6)
    return any(str(position.get("symbol", "")).zfill(6) == key for position in open_positions)


def _proxy_equity(cash: float, open_positions: Sequence[dict[str, Any]]) -> float:
    return max(float(cash), 0.0) + _proxy_open_notional(open_positions)


def _proxy_realize_exits(cash: float, open_positions: list[dict[str, Any]], cutoff: datetime) -> float:
    remaining: list[dict[str, Any]] = []
    next_cash = float(cash)
    for position in sorted(open_positions, key=lambda item: str(item["exit_time"])):
        exit_time = _proxy_align_timestamp(position["exit_time"], cutoff)
        if exit_time <= cutoff:
            next_cash += float(position.get("exit_cash", 0.0) or 0.0)
        else:
            remaining.append(position)
    open_positions[:] = remaining
    return next_cash


def _proxy_align_timestamp(value: datetime, reference: datetime) -> datetime:
    timestamp = value
    if timestamp.tzinfo is None and reference.tzinfo is not None:
        return timestamp.replace(tzinfo=reference.tzinfo)
    if timestamp.tzinfo is not None and reference.tzinfo is None:
        return timestamp.replace(tzinfo=None)
    return timestamp


def _proxy_submission_time(outcome: OLRTradeOutcome) -> datetime:
    raw = (outcome.metadata or {}).get("entry_submission_time")
    if raw:
        try:
            parsed = datetime.fromisoformat(str(raw))
            return _proxy_align_timestamp(parsed, outcome.entry_time)
        except (TypeError, ValueError):
            pass
    return outcome.entry_time


def _metadata_float_value(metadata: dict[str, Any] | None, key: str, default: float) -> float:
    try:
        return float((metadata or {}).get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _entry_submission_index(bars: Sequence[MarketBar], signal: OLREntrySignal) -> int:
    if signal.reason == "close_auction":
        first = _first_index_at_or_after(bars, DECISION_CUTOFF)
        return signal.signal_index if first is None else first
    return max(0, min(signal.signal_index, len(bars) - 1))


def _cap_normalized_weights(raw: Sequence[float], *, target: float, cap: float) -> list[float]:
    if not raw:
        return []
    total_raw = sum(max(0.0, float(value)) for value in raw)
    if total_raw <= 0.0:
        return [0.0 for _ in raw]
    weights = [target * max(0.0, float(value)) / total_raw for value in raw]
    if cap <= 0.0:
        return [0.0 for _ in weights]
    capped = [False for _ in weights]
    while True:
        changed = False
        remaining_target = target
        remaining_raw = 0.0
        for index, weight in enumerate(weights):
            if capped[index]:
                remaining_target -= weight
            else:
                remaining_raw += max(0.0, float(raw[index]))
        if remaining_target <= 0.0 or remaining_raw <= 0.0:
            break
        for index, value in enumerate(raw):
            if capped[index]:
                continue
            proposed = remaining_target * max(0.0, float(value)) / remaining_raw
            if proposed > cap:
                weights[index] = cap
                capped[index] = True
                changed = True
        if not changed:
            for index, value in enumerate(raw):
                if not capped[index]:
                    weights[index] = remaining_target * max(0.0, float(value)) / remaining_raw
            break
    return weights


def round_trip_cost_pct(config: OLRConfig) -> float:
    bps = 2.0 * float(config.slippage_bps) + 2.0 * float(config.commission_bps) + float(config.tax_bps_on_sell)
    return max(0.0, bps) / 10_000.0


def _post_entry_bars(entry_bars: tuple[MarketBar, ...], next_bars: tuple[MarketBar, ...], signal: OLREntrySignal) -> tuple[MarketBar, ...]:
    if signal.fill_at_close:
        return tuple(next_bars)
    return tuple(entry_bars[signal.fill_index :]) + tuple(next_bars)


def _ordered_completed_bars(bars: Sequence[MarketBar], trade_date: date | None = None) -> tuple[MarketBar, ...]:
    out = []
    for bar in bars:
        ts = bar.timestamp.astimezone(KST)
        if trade_date is not None and ts.date() != trade_date:
            continue
        if not bar.is_completed:
            continue
        if ts.time() > DEFAULT_FLATTEN_TIME:
            continue
        out.append(bar)
    return tuple(sorted(out, key=lambda item: item.timestamp))


def _first_index_at_or_after(bars: Sequence[MarketBar], cutoff: dtime) -> int | None:
    for index, bar in enumerate(bars):
        if bar.timestamp.astimezone(KST).time() >= cutoff:
            return index
    return None


def _last_index_before(bars: Sequence[MarketBar], cutoff: dtime) -> int | None:
    out = None
    for index, bar in enumerate(bars):
        if bar.timestamp.astimezone(KST).time() < cutoff:
            out = index
    return out


def _last_continuous_index(bars: Sequence[MarketBar]) -> int | None:
    out = None
    for index, bar in enumerate(bars):
        t = bar.timestamp.astimezone(KST).time()
        if DECISION_CUTOFF <= t <= DEFAULT_LAST_CONTINUOUS_TIME:
            out = index
    return out


def _close_auction_index(bars: Sequence[MarketBar]) -> int | None:
    for index, bar in enumerate(bars):
        if bar.timestamp.astimezone(KST).time() == DEFAULT_CLOSE_AUCTION_TIME:
            return index
    return None


def _next_fill(signal_index: int, reason: str, bars: Sequence[MarketBar], *, require_fill_bar: bool = True) -> OLREntrySignal | None:
    fill_index = signal_index + 1
    if fill_index >= len(bars):
        if require_fill_bar:
            return None
        if bars[signal_index].timestamp.astimezone(KST).time() >= DEFAULT_LAST_CONTINUOUS_TIME:
            return None
        return OLREntrySignal(fill_index=fill_index, signal_index=signal_index, reason=reason)
    if bars[fill_index].timestamp.astimezone(KST).time() > DEFAULT_LAST_CONTINUOUS_TIME:
        return None
    return OLREntrySignal(fill_index=fill_index, signal_index=signal_index, reason=reason)


def _common_entry_bar_passes(bar: MarketBar, vwap: float, close_location: float, decision_close: float, plan: OLREntryPlan) -> bool:
    if float(bar.close) / max(float(bar.open), 1e-9) - 1.0 < plan.min_bar_ret:
        return False
    if float(bar.close) / max(vwap, 1e-9) - 1.0 < plan.min_vwap_ret:
        return False
    if close_location < plan.min_close_location:
        return False
    if plan.require_above_decision_close and float(bar.close) <= decision_close:
        return False
    if plan.max_vwap_extension_pct < 9.0 and float(bar.close) / max(vwap, 1e-9) - 1.0 > plan.max_vwap_extension_pct:
        return False
    return True


def _completed_bar_exit_reason(
    bars: Sequence[MarketBar],
    index: int,
    entry: float,
    risk: float,
    high_water: float,
    plan: OLRExitPlan,
    vwap_fail_streak: int,
) -> dict[str, Any]:
    bar = bars[index]
    held = index + 1
    close_r = (float(bar.close) - entry) / risk
    mfe_r = (high_water - entry) / risk
    if plan.vwap_fail_bars > 0:
        vwap = _running_vwap(bars[: index + 1])
        vwap_fail_streak = vwap_fail_streak + 1 if float(bar.close) < vwap * (1.0 - plan.vwap_fail_pct) else 0
        if vwap_fail_streak >= plan.vwap_fail_bars:
            return {"reason": "vwap_fail", "vwap_fail_streak": vwap_fail_streak}
    if plan.failed_followthrough_bars > 0 and held >= plan.failed_followthrough_bars and mfe_r < plan.failed_followthrough_mfe_r and close_r <= plan.failed_followthrough_close_r:
        return {"reason": "failed_followthrough", "vwap_fail_streak": vwap_fail_streak}
    if plan.no_mfe_bars > 0 and held >= plan.no_mfe_bars and mfe_r < plan.no_mfe_thresh_r:
        return {"reason": "no_mfe_time_stop", "vwap_fail_streak": vwap_fail_streak}
    if plan.mfe_fade_start_r > 0.0 and plan.mfe_fade_gap_r > 0.0 and mfe_r >= plan.mfe_fade_start_r:
        fade_trigger_r = max(float(plan.mfe_fade_floor_r), mfe_r - float(plan.mfe_fade_gap_r))
        if close_r <= fade_trigger_r:
            return {"reason": "mfe_fade", "vwap_fail_streak": vwap_fail_streak}
    return {"reason": "", "vwap_fail_streak": vwap_fail_streak}


def _next_bar_stop_from_completed_bar(entry: float, risk: float, current_stop: float, high_water: float, plan: OLRExitPlan) -> float:
    stop = current_stop
    mfe_r = (high_water - entry) / risk
    if plan.breakeven_trigger_r > 0 and mfe_r >= plan.breakeven_trigger_r:
        stop = max(stop, entry + plan.breakeven_stop_r * risk)
    if plan.trail_start_r > 0 and mfe_r >= plan.trail_start_r:
        stop = max(stop, high_water - plan.trail_gap_r * risk)
    return stop


def _running_vwap(bars: Sequence[MarketBar]) -> float:
    volume = sum(max(float(bar.volume), 0.0) for bar in bars)
    if volume <= 0:
        return float(bars[-1].close) if bars else 0.0
    return sum(((float(bar.high) + float(bar.low) + float(bar.close)) / 3.0) * max(float(bar.volume), 0.0) for bar in bars) / volume


def _close_location(bar: MarketBar) -> float:
    width = max(float(bar.high) - float(bar.low), 1e-9)
    return (float(bar.close) - float(bar.low)) / width


def _exit_reason_metrics(outcomes: Sequence[OLRTradeOutcome]) -> dict[str, float]:
    total = max(float(len(outcomes)), 1.0)
    reasons: dict[str, int] = {}
    for outcome in outcomes:
        reasons[outcome.exit_reason] = reasons.get(outcome.exit_reason, 0) + 1
    return {f"exit_reason_{key}_share": float(value) / total for key, value in sorted(reasons.items())}


def _avg(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return mean(items) if items else 0.0


def _median(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return median(items) if items else 0.0


def _share(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    return sum(1 for item in items if item) / max(float(len(items)), 1.0)


def _compound(values: Iterable[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + float(value)
    return equity - 1.0


def _max_drawdown(values: Iterable[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd
