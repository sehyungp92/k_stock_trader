from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from strategy_common.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    StrategyAction,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
    SubmitProtectiveStop,
)
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig, KALCB_CORE_VERSION, STRATEGY_ID, SUPPORTED_ENTRY_PLAN_MODES, CarryMode
from strategy_kalcb.exits import (
    failure_stop_target,
    partial_exit_qty,
    should_flow_reversal,
    should_mfe_conviction_exit,
    should_take_partial,
    should_quick_exit,
    trailing_stop_from_mfe,
)
from strategy_kalcb.first30 import build_first30_features
from strategy_kalcb.models import EntryType, KALCBDailyCandidate, KALCBDailySnapshot
from strategy_kalcb.risk import (
    compute_entry_qty,
    conditional_entry_blocked,
    effective_max_position_notional_pct,
    momentum_stop_price,
    regime_size_mult,
    round_price_for_krx,
)
from strategy_kalcb.signals import (
    classify_raw_breakout,
    close_location_value,
    compute_bar_rvol,
    compute_momentum_score,
    compute_opening_range,
    compute_session_vwap,
)

from .core_models import KALCBCoreResult, KALCBFillEvent, KALCBOrderUpdateEvent, KALCBPortfolioView
from .state import KALCBPositionState, KALCBState, KALCBSymbolState, SymbolStage


def step_kalcb_core(
    state: KALCBState,
    bar: MarketBar,
    config: KALCBConfig,
    candidate_snapshot: KALCBDailySnapshot | None,
    portfolio: KALCBPortfolioView,
) -> KALCBCoreResult:
    if not bar.is_completed:
        raise ValueError(f"KALCB core requires completed bars: {bar.symbol} {bar.timestamp}")
    if bar.timeframe.lower() != "5m":
        raise ValueError("KALCB core requires 5m bars")
    state.meta["fast_replay_suppress_rejections"] = bool(config.fast_replay_suppress_rejections)
    _ensure_session(state, bar, candidate_snapshot)
    _update_portfolio_context(state, portfolio)
    symbol_state = state.symbol_state(bar.symbol)
    candidate = _candidate_for_symbol(candidate_snapshot, bar.symbol, bar.timestamp.date())
    candidate_rank = _candidate_rank(candidate_snapshot, bar.symbol, bar.timestamp.date())
    if symbol_state.session_date != bar.timestamp.date():
        symbol_state.reset_for_session(bar.timestamp.date(), candidate, candidate_rank)
    elif candidate is not None:
        symbol_state.candidate = candidate
        symbol_state.candidate_rank = candidate_rank

    if _outside_regular_session(bar, config):
        return KALCBCoreResult(state)

    symbol_state.add_bar(bar)

    if symbol_state.position is not None:
        return _manage_position(state, symbol_state, bar, config)

    if symbol_state.stage == SymbolStage.ENTRY_QUEUED or symbol_state.pending_entry_order_id:
        return KALCBCoreResult(state)

    if _bar_blocks_entry(bar):
        symbol_state.stage = SymbolStage.BLOCKED
        symbol_state.rejected_reason = "untradable_or_vi"
        decision = _decision(bar, "entry_blocked", "untradable_or_vi", metadata={"bar_metadata": dict(bar.metadata)})
        return KALCBCoreResult(state, decisions=[decision])

    if candidate is None or not candidate.tradable:
        symbol_state.rejected_reason = "no_daily_candidate" if candidate is None else ",".join(candidate.reject_reasons) or "candidate_not_tradable"
        return KALCBCoreResult(state)

    if not symbol_state.opening_range_built:
        if len(symbol_state.bars) >= config.opening_range_bars:
            bars_today = symbol_state.bars_today
            if not _opening_range_window_complete(bars_today, config):
                symbol_state.stage = SymbolStage.BLOCKED
                symbol_state.rejected_reason = "opening_range_incomplete"
                decision = _decision(
                    bar,
                    "entry_blocked",
                    "opening_range_incomplete",
                    metadata={"n_bars": len(bars_today), "opening_range_bars": config.opening_range_bars},
                )
                return KALCBCoreResult(state, decisions=[decision])
            oh, ol, ov = compute_opening_range(bars_today, config.opening_range_bars)
            symbol_state.or_high = oh
            symbol_state.or_low = ol
            symbol_state.or_volume = ov
            symbol_state.opening_range_built = True
            decision = _decision(
                bar,
                "opening_range_built",
                "opening_range_complete",
                metadata={"or_high": oh, "or_low": ol, "or_volume": ov, "n_bars": config.opening_range_bars},
            )
            if _is_first30_entry_plan(config):
                entry_result = _evaluate_entry(state, symbol_state, bar, config, portfolio, first30_signal=True)
                return KALCBCoreResult(state, entry_result.actions, [decision, *entry_result.decisions])
            return KALCBCoreResult(state, decisions=[decision])
        symbol_state.rejected_reason = "building_opening_range"
        return KALCBCoreResult(state)

    if not (config.entry_window_start <= bar.timestamp.time() <= config.entry_window_end):
        return KALCBCoreResult(state)

    return _evaluate_entry(state, symbol_state, bar, config, portfolio)


def on_kalcb_fill(state: KALCBState, fill: KALCBFillEvent, config: KALCBConfig) -> KALCBCoreResult:
    symbol_state = state.symbol_state(fill.symbol)
    role_meta = dict(state.order_roles.pop(fill.order_id, {}))
    pending_metadata = dict(symbol_state.pending_entry_metadata or {}) if fill.side.upper() == "BUY" else {}
    metadata = {**role_meta, **pending_metadata, **dict(fill.metadata or {})}
    role = str(metadata.get("order_role") or "").upper()
    event_ts = fill.timestamp
    actions: list[StrategyAction] = []
    decisions: list[DecisionEvent] = []

    if fill.side.upper() == "BUY":
        stop_price = round_price_for_krx(float(metadata.get("stop_price") or fill.price * 0.985), "protective_stop")
        risk = max(float(fill.price) - stop_price, 1.0)
        position = KALCBPositionState(
            symbol=fill.symbol,
            qty_entry=int(fill.qty),
            qty_open=int(fill.qty),
            entry_price=float(fill.price),
            entry_time=event_ts,
            initial_stop=stop_price,
            current_stop=stop_price,
            risk_per_share=risk,
            entry_type=str(metadata.get("entry_type") or EntryType.OR_BREAKOUT.value),
            momentum_score=int(metadata.get("momentum_score") or 0),
            sector=str(metadata.get("sector") or "UNKNOWN"),
            regime_tier=str(metadata.get("regime_tier") or "A"),
            entry_order_id=fill.order_id,
            avwap_at_entry=float(metadata.get("avwap") or 0.0),
            or_high=float(metadata.get("or_high") or 0.0),
            or_low=float(metadata.get("or_low") or 0.0),
            metadata=dict(metadata),
        )
        symbol_state.position = position
        symbol_state.stage = SymbolStage.IN_POSITION
        symbol_state.pending_entry_order_id = ""
        symbol_state.pending_entry_metadata.clear()
        stop_action = None
        if config.exit_hard_stop_enabled:
            stop_action = SubmitProtectiveStop(
                strategy_id=STRATEGY_ID,
                symbol=fill.symbol,
                qty=fill.qty,
                stop_price=stop_price,
                reason="initial_stop",
                metadata=_exit_metadata(position, "initial_stop", {"order_role": "STOP", "stop_kind": "initial_stop"}),
            )
            actions.append(stop_action)
        decisions.append(
            DecisionEvent(
                timestamp=event_ts,
                strategy_id=STRATEGY_ID,
                symbol=fill.symbol,
                decision_code="entry_filled",
                reason="entry_fill",
                actions=tuple(actions),
                metadata={"entry_type": position.entry_type, "qty": fill.qty, "price": fill.price, "hard_stop_enabled": config.exit_hard_stop_enabled, "core_version": KALCB_CORE_VERSION},
            )
        )
        return KALCBCoreResult(state, actions, decisions)

    position = symbol_state.position
    if position is None:
        return KALCBCoreResult(state)
    exit_qty = min(int(fill.qty), int(position.qty_open))
    if exit_qty <= 0:
        return KALCBCoreResult(state)
    position.qty_open -= exit_qty
    position.update_mark(high=fill.price, low=fill.price)
    is_partial = role in {"TP", "PARTIAL"} or "partial" in fill.reason.lower()
    if is_partial and position.qty_open > 0:
        position.partial_taken = True
        position.partial_order_id = ""
        partial_metadata: dict[str, Any] = {"qty": exit_qty, "remaining_qty": position.qty_open, "core_version": KALCB_CORE_VERSION}
        if config.partial_stop_to_breakeven and position.risk_per_share > 0:
            target = position.entry_price + config.partial_breakeven_buffer_r * position.risk_per_share
            rounded = _valid_protective_stop(target, fill.price)
            if rounded > position.current_stop:
                position.current_stop = rounded
                position.stop_tightened = True
                action = ReplaceProtectiveStop(
                    strategy_id=STRATEGY_ID,
                    symbol=position.symbol,
                    qty=position.qty_open,
                    stop_price=rounded,
                    reason="partial_breakeven",
                    metadata=_exit_metadata(
                        position,
                        "protected_stop",
                        {
                            "order_role": "STOP",
                            "stop_kind": "partial_breakeven",
                            "partial_exit_order_id": fill.order_id,
                        },
                    ),
                )
                actions.append(action)
                partial_metadata.update({"breakeven_stop": rounded, "breakeven_buffer_r": config.partial_breakeven_buffer_r})
        decisions.append(
            DecisionEvent(
                timestamp=event_ts,
                strategy_id=STRATEGY_ID,
                symbol=fill.symbol,
                decision_code="partial_filled",
                reason="partial_profit",
                actions=tuple(actions),
                metadata=partial_metadata,
            )
        )
        return KALCBCoreResult(state, actions, decisions)

    exit_reason = fill.reason or str(metadata.get("exit_reason") or "exit")
    position.last_exit_reason = exit_reason
    if position.qty_open <= 0:
        symbol_state.position = None
        symbol_state.stage = SymbolStage.DONE
        symbol_state.pending_entry_order_id = ""
    else:
        position.exit_in_flight = False
    decisions.append(
        DecisionEvent(
            timestamp=event_ts,
            strategy_id=STRATEGY_ID,
            symbol=fill.symbol,
            decision_code="exit_filled",
            reason=exit_reason,
            metadata={"qty": exit_qty, "remaining_qty": max(position.qty_open, 0), "core_version": KALCB_CORE_VERSION},
        )
    )
    return KALCBCoreResult(state, actions, decisions)


def on_kalcb_order_update(state: KALCBState, update: KALCBOrderUpdateEvent) -> KALCBCoreResult:
    symbol_state = state.symbol_state(update.symbol)
    status = update.status.upper()
    terminal = status in {"BLOCKED", "REJECTED", "CANCELLED", "DEFERRED", "EXPIRED"}
    role_meta = state.order_roles.pop(str(update.order_id), {}) if terminal else dict(state.order_roles.get(str(update.order_id), {}))
    metadata = {**dict(role_meta), **dict(update.metadata)}
    role = str(update.role or metadata.get("order_role") or "").upper()
    if terminal and update.order_id == symbol_state.pending_entry_order_id:
        symbol_state.pending_entry_order_id = ""
        symbol_state.pending_entry_metadata.clear()
        symbol_state.stage = SymbolStage.WATCHING
        symbol_state.rejected_reason = status.lower()
    if terminal and _is_retryable_entry_update(update, status, metadata) and symbol_state.position is None:
        symbol_state.entry_attempted = False
        symbol_state.pending_entry_order_id = ""
        symbol_state.pending_entry_metadata.clear()
        symbol_state.stage = SymbolStage.WATCHING
        symbol_state.rejected_reason = status.lower()
        _decrement_entry_route_session_count(state, str(metadata.get("entry_route") or ""))
    position = symbol_state.position
    if terminal and position is not None:
        if role in {"TP", "PARTIAL"} or position.partial_order_id == str(update.order_id):
            position.partial_order_id = ""
        if role == "STOP" and position.stop_order_id == str(update.order_id):
            position.stop_order_id = ""
        if role == "EXIT":
            position.exit_in_flight = False
    return KALCBCoreResult(
        state,
        decisions=[
            DecisionEvent(
                timestamp=update.timestamp,
                strategy_id=STRATEGY_ID,
                symbol=update.symbol,
                decision_code="order_update",
                reason=status.lower(),
                metadata={"order_id": update.order_id, "role": role, **metadata},
            )
        ],
    )


def _is_retryable_entry_update(update: KALCBOrderUpdateEvent, status: str, metadata: dict[str, Any]) -> bool:
    role = str(update.role or metadata.get("order_role") or "").upper().strip()
    if role and role not in {"ENTRY", "BUY"}:
        return False
    if status == "DEFERRED":
        return True
    reason_text = " ".join(
        str(value or "")
        for value in (
            update.reason,
            metadata.get("message"),
            metadata.get("reason"),
            metadata.get("oms_status"),
            metadata.get("portfolio_reason_code"),
            metadata.get("resource_conflict_type"),
        )
    ).lower()
    retryable_markers = (
        "oms unreachable",
        "oms error 503",
        "timeout",
        "temporar",
        "missing_or_zero_account_state",
        "equity not yet loaded",
        "price unavailable",
        "reconciliation pending",
    )
    return status in {"REJECTED", "BLOCKED"} and any(marker in reason_text for marker in retryable_markers)


def on_kalcb_timer(state: KALCBState, timestamp: datetime, config: KALCBConfig) -> KALCBCoreResult:
    """Shared scheduled-event hook for live and replay adapters."""

    if timestamp.time() < config.flatten_time:
        return KALCBCoreResult(state)
    actions: list[StrategyAction] = []
    decisions: list[DecisionEvent] = []
    for symbol_state in state.symbols.values():
        position = symbol_state.position
        if position is None or position.exit_in_flight:
            continue
        latest_bar = symbol_state.bars_today[-1] if symbol_state.bars_today else None
        current_r = position.unrealized_r(latest_bar.close if latest_bar is not None else position.entry_price)
        latest_cpr = close_location_value(latest_bar) if latest_bar is not None else 0.0
        carry_ok = (
            config.carry_mode == CarryMode.STRICT_LIVE
            and current_r >= config.carry_min_r
            and latest_cpr >= config.carry_min_cpr
            and position.regime_tier in {"A", "B"}
        )
        if carry_ok:
            continue
        reason = "eod_flatten" if config.carry_mode == CarryMode.OFF else "eod_flatten_carry_shadow"
        action = FlattenPosition(
            strategy_id=STRATEGY_ID,
            symbol=position.symbol,
            reason=reason,
            metadata=_exit_metadata(position, reason, {"order_role": "EXIT", "timer_event": True}),
        )
        position.exit_in_flight = True
        actions.append(action)
        decisions.append(
            DecisionEvent(
                timestamp=timestamp,
                strategy_id=STRATEGY_ID,
                symbol=position.symbol,
                decision_code="timer_exit",
                reason=reason,
                actions=(action,),
                metadata=dict(action.metadata),
            )
        )
    return KALCBCoreResult(state, actions, decisions)


def remember_submitted_order(state: KALCBState, order_id: str | None, action: StrategyAction) -> None:
    if not order_id:
        return
    metadata = _action_role_metadata(action)
    state.order_roles[order_id] = metadata
    symbol_state = state.symbol_state(getattr(action, "symbol", ""))
    if isinstance(action, SubmitEntry):
        symbol_state.pending_entry_order_id = order_id
        symbol_state.pending_entry_metadata = dict(action.metadata)
    elif isinstance(action, SubmitProtectiveStop) and symbol_state.position is not None:
        symbol_state.position.stop_order_id = order_id
    elif isinstance(action, SubmitPartialExit) and symbol_state.position is not None:
        symbol_state.position.partial_order_id = order_id


def _ensure_session(state: KALCBState, bar: MarketBar, snapshot: KALCBDailySnapshot | None) -> None:
    session = bar.timestamp.date()
    if state.session_date == session:
        return
    state.session_date = session
    state.meta["entry_route_session_date"] = session.isoformat()
    state.meta["entry_route_session_counts"] = {}
    snapshot_is_fresh = snapshot is not None and snapshot.trade_date == session
    if snapshot_is_fresh:
        state.snapshot_hash = snapshot.artifact_hash
        state.source_fingerprint = snapshot.source_fingerprint
    state.meta["entry_context"] = _snapshot_entry_context(snapshot if snapshot_is_fresh else None)
    by_symbol = snapshot.by_symbol() if snapshot_is_fresh else {}
    for symbol_state in state.symbols.values():
        if symbol_state.position is None:
            symbol_state.reset_for_session(
                session,
                by_symbol.get(symbol_state.symbol),
                _candidate_rank(snapshot, symbol_state.symbol, session) if snapshot_is_fresh else 0,
            )


def _update_portfolio_context(state: KALCBState, portfolio: KALCBPortfolioView) -> None:
    equity = float(portfolio.equity or 0.0)
    if equity <= 0:
        equity = max(float(portfolio.cash or 0.0) + float(portfolio.open_notional or 0.0), 0.0)
    high_water = float(state.meta.get("portfolio_high_water_equity", 0.0) or 0.0)
    if high_water <= 0:
        high_water = equity
    high_water = max(high_water, equity)
    state.meta["portfolio_high_water_equity"] = high_water

    session_text = state.session_date.isoformat() if state.session_date else ""
    if state.meta.get("portfolio_session_date") != session_text:
        state.meta["portfolio_session_date"] = session_text
        state.meta["portfolio_session_start_equity"] = equity
    session_start = float(state.meta.get("portfolio_session_start_equity", equity) or equity)

    portfolio.equity = equity
    portfolio.high_water_equity = high_water
    portfolio.session_start_equity = session_start
    portfolio.drawdown_pct = max(0.0, high_water - equity) / high_water if high_water > 0 else 0.0
    portfolio.session_return_pct = equity / session_start - 1.0 if session_start > 0 else 0.0


def _snapshot_entry_context(snapshot: KALCBDailySnapshot | None) -> dict[str, float]:
    if snapshot is None:
        return {}
    metadata = dict(snapshot.metadata or {})
    candidates = tuple(snapshot.candidates or ())
    context: dict[str, float] = {}
    for source_key, target_key in (
        ("active_symbol_count", "session_active_symbol_count"),
        ("selection_count", "session_selection_count"),
        ("candidate_pool_count", "session_candidate_pool_count"),
        ("frontier_symbol_count", "session_frontier_symbol_count"),
    ):
        value = _optional_float(metadata.get(source_key))
        if value is not None:
            context[target_key] = value
    active_count = context.get("session_active_symbol_count", 0.0)
    frontier_count = context.get("session_frontier_symbol_count", float(len(candidates)))
    pool_count = context.get("session_candidate_pool_count", float(len(candidates)))
    selection_count = context.get("session_selection_count", float(len(candidates)))
    if active_count > 0:
        context["session_frontier_to_active_ratio"] = frontier_count / active_count
    if pool_count > 0:
        context["session_selection_to_pool_ratio"] = selection_count / pool_count

    first30_rel_volume = _candidate_numeric_values(candidates, "first30_rel_volume")
    first30_ret = _candidate_numeric_values(candidates, "first30_ret")
    first30_gap = _candidate_numeric_values(candidates, "first30_gap")
    first30_cpr = _candidate_numeric_values(candidates, "first30_signal_bar_cpr", "first30_close_location")
    first30_range_atr = _candidate_numeric_values(candidates, "first30_range_atr")
    sector_intraday_score = _candidate_numeric_values(candidates, "sector_intraday_score_pct")
    sector_intraday_ret = _candidate_numeric_values(candidates, "sector_intraday_ret")
    sector_intraday_effective_count = _candidate_numeric_values(candidates, "sector_intraday_effective_count")
    for prefix, values in (
        ("session_first30_rel_volume", first30_rel_volume),
        ("session_first30_ret", first30_ret),
        ("session_first30_gap", first30_gap),
        ("session_first30_signal_cpr", first30_cpr),
        ("session_first30_range_atr", first30_range_atr),
        ("session_sector_intraday_score_pct", sector_intraday_score),
        ("session_sector_intraday_ret", sector_intraday_ret),
        ("session_sector_intraday_effective_count", sector_intraday_effective_count),
    ):
        if values:
            context[f"{prefix}_mean"] = sum(values) / len(values)
            context[f"{prefix}_median"] = _median(values)
    if first30_ret:
        context["session_first30_positive_share"] = sum(1 for value in first30_ret if value > 0.0) / len(first30_ret)
    if first30_gap:
        context["session_first30_gap_dispersion"] = max(first30_gap) - min(first30_gap)
    if sector_intraday_score:
        context["session_sector_intraday_score_confirmed_share"] = sum(1 for value in sector_intraday_score if value > 50.0) / len(sector_intraday_score)
    if sector_intraday_ret:
        ret_positive_share = sum(1 for value in sector_intraday_ret if value > 0.0) / len(sector_intraday_ret)
        context["session_sector_intraday_positive_share"] = ret_positive_share
        context["session_sector_intraday_ret_positive_share"] = ret_positive_share
    return context


def _candidate_numeric_values(candidates: tuple[KALCBDailyCandidate, ...], *keys: str) -> list[float]:
    values: list[float] = []
    for candidate in candidates:
        metadata = dict(candidate.metadata or {})
        for key in keys:
            value = _optional_float(metadata.get(key))
            if value is None:
                continue
            values.append(value)
            break
    return values


_CANDIDATE_CONTEXT_KEYS = (
    "daily_return_5d",
    "daily_return_20d",
    "daily_return_60d",
    "daily_volume_ratio_20d",
    "daily_close20_loc",
    "daily_close60_loc",
    "daily_above_sma20",
    "daily_above_sma60",
    "daily_acceleration_5v20",
    "daily_momentum_pct",
    "daily_sector_alignment_pct",
    "stock_sector_daily_ret20_spread",
    "stock_sector_daily_ret5_spread",
    "first30_quality_pct",
    "first30_sector_ret_spread",
    "first30_sector_relvol_ratio",
    "first30_sector_leadership_pct",
    "first30_gap_relvol_sector_breadth",
    "first30_gap_retention_sector_breadth",
    "continuation_joint_quality_pct",
    "sector_participation",
    "sector_flow_participation",
    "sector_daily_score_pct",
    "sector_daily_ret_5d",
    "sector_daily_ret_20d",
    "sector_daily_ret_60d",
    "sector_daily_breadth_20d",
    "sector_daily_participation",
    "sector_daily_rel_volume",
    "sector_daily_flow_5d",
    "sector_daily_foreign_flow_5d",
    "sector_daily_institutional_flow_5d",
    "sector_daily_flow_agreement_5d",
    "sector_daily_effective_count",
    "sector_daily_shrinkage_weight",
    "sector_intraday_score_pct",
    "sector_intraday_ret",
    "sector_intraday_breadth",
    "sector_intraday_rel_volume",
    "sector_intraday_participation",
    "sector_intraday_effective_count",
    "sector_intraday_shrinkage_weight",
    "prev_session_sector_intraday_score_pct",
    "prev_session_sector_intraday_ret",
    "prev_session_sector_intraday_breadth",
    "prev_session_sector_intraday_participation",
    "prev_session_sector_intraday_effective_count",
    "structural_campaign_score",
    "first30_confirmation_score",
    "campaign_state_score",
    "campaign_box_high",
    "campaign_box_low",
    "campaign_box_mid",
    "campaign_box_range_pct",
    "campaign_box_containment",
    "campaign_box_atr_ratio",
    "campaign_box_squeeze_pct",
    "campaign_box_high_distance_pct",
    "campaign_avwap",
    "campaign_avwap_distance_pct",
    "campaign_breakout_level",
    "campaign_breakout_displacement",
    "first30_breakout_confirmation",
    "first30_breakout_acceptance",
    "first30_breakout_acceptance_closes",
)


def _candidate_context_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _CANDIDATE_CONTEXT_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float, str, bool)):
            out[key] = value
    return out


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _candidate_for_symbol(snapshot: KALCBDailySnapshot | None, symbol: str, trade_date) -> KALCBDailyCandidate | None:
    if snapshot is None or snapshot.trade_date != trade_date:
        return None
    return snapshot.by_symbol().get(symbol)


def _candidate_rank(snapshot: KALCBDailySnapshot | None, symbol: str, trade_date) -> int:
    if snapshot is None or snapshot.trade_date != trade_date:
        return 0
    wanted = str(symbol)
    for index, candidate in enumerate(snapshot.candidates, start=1):
        if candidate.symbol == wanted:
            return index
    return 0


def _outside_regular_session(bar: MarketBar, config: KALCBConfig) -> bool:
    bar_time = bar.timestamp.time()
    return bar_time < config.session_open or bar_time > config.session_close


def _in_time_window(value, start, end) -> bool:
    return start <= value <= end


def _two_close_confirmed(bars_today: list[MarketBar], breakout_level: float) -> bool:
    if len(bars_today) < 2 or breakout_level <= 0:
        return False
    return float(bars_today[-2].close) > breakout_level and float(bars_today[-1].close) > breakout_level


def _opening_range_window_complete(bars_today: list[MarketBar], config: KALCBConfig) -> bool:
    if len(bars_today) < config.opening_range_bars:
        return False
    window = bars_today[: config.opening_range_bars]
    first = window[0].timestamp
    duration = timedelta(minutes=5)
    session_start = datetime.combine(first.date(), config.session_open, tzinfo=first.tzinfo)
    allowed_starts = {session_start, session_start + duration}
    if first not in allowed_starts:
        return False
    for index, item in enumerate(window):
        if item.timestamp != first + duration * index:
            return False
    return True


def _is_first30_entry_plan(config: KALCBConfig) -> bool:
    if config.entry_plan_routes:
        return any(_is_first30_entry_mode(_route_mode(config, route)) for route in config.entry_plan_routes)
    return _is_first30_entry_mode(str(config.entry_plan_mode))


def _is_first30_entry_mode(mode: str) -> bool:
    return str(mode) in {"first30_open", "opening_drive"}


def _entry_plan_signal(
    symbol_state: KALCBSymbolState,
    bar: MarketBar,
    config: KALCBConfig,
    *,
    candidate: KALCBDailyCandidate,
    prior_day_high: float,
    prior_day_close: float,
    daily_atr: float,
    expected_30m_volume: float,
    avwap: float,
    first30_signal: bool,
    entry_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    routes = tuple(config.entry_plan_routes or ())
    if not routes:
        result = _entry_plan_signal_single(
            symbol_state,
            bar,
            config,
            candidate=candidate,
            prior_day_high=prior_day_high,
            prior_day_close=prior_day_close,
            daily_atr=daily_atr,
            expected_30m_volume=expected_30m_volume,
            avwap=avwap,
            first30_signal=first30_signal,
        )
        return _with_entry_route_metadata(
            result,
            effective_config=config,
            route={"name": str(config.entry_plan_mode), "mode": str(config.entry_plan_mode), "priority": 0},
            route_index=0,
        )

    attempts: list[dict[str, Any]] = []
    combined_gates: list[dict[str, Any]] = []
    selected_rejection: dict[str, Any] | None = None
    for route_index, route in _ordered_entry_routes(config):
        mode = _route_mode(config, route)
        route_meta = _entry_route_metadata(route, mode=mode, route_index=route_index)
        if first30_signal != _is_first30_entry_mode(mode):
            attempts.append({**route_meta, "reason": "route_not_applicable_for_bar", "gate_count": 0})
            continue
        effective_config = _entry_route_config(config, route)
        result = _entry_plan_signal_single(
            symbol_state,
            bar,
            effective_config,
            candidate=candidate,
            prior_day_high=prior_day_high,
            prior_day_close=prior_day_close,
            daily_atr=daily_atr,
            expected_30m_volume=expected_30m_volume,
            avwap=avwap,
            first30_signal=first30_signal,
        )
        decorated = _with_entry_route_metadata(
            result,
            effective_config=effective_config,
            route=route,
            route_index=route_index,
        )
        gates = list(decorated.get("gates") or [])
        combined_gates.extend(gates)
        attempts.append({**route_meta, "reason": str(decorated.get("reason") or ""), "gate_count": len(gates)})
        if decorated.get("entry_type") is not None:
            route_context = _entry_route_context_metadata(symbol_state, decorated, entry_context)
            context_reason, context_gates = _entry_context_gate_reject_reason(
                effective_config,
                route_context,
            )
            if context_gates:
                combined_gates.extend(context_gates)
                attempts[-1]["gate_count"] = int(attempts[-1]["gate_count"]) + len(context_gates)
            if context_reason:
                attempts[-1]["reason"] = context_reason
                rejected = dict(decorated)
                rejected["entry_type"] = None
                rejected["reason"] = context_reason
                rejected["gates"] = [*gates, *context_gates]
                rejected.update(route_context)
                rejected["entry_route_context"] = route_context
                selected_rejection = rejected
                continue
            decorated["entry_route_attempts"] = attempts
            return decorated
        selected_rejection = decorated

    if selected_rejection is not None:
        rejected = dict(selected_rejection)
        rejected["gates"] = combined_gates
        rejected["entry_route_attempts"] = attempts
        return rejected
    return {
        "entry_type": None,
        "reason": "no_applicable_entry_route",
        "gates": combined_gates,
        "entry_route_attempts": attempts,
    }


def _entry_route_context_metadata(
    symbol_state: KALCBSymbolState,
    plan_signal: dict[str, Any],
    entry_context: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate = symbol_state.candidate
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    candidate_rank = int(symbol_state.candidate_rank or metadata.get("candidate_rank") or 0)
    frontier_rank = int(metadata.get("frontier_rank") or candidate_rank or 0)
    return {
        "sector": str(getattr(candidate, "sector", metadata.get("sector", "UNKNOWN")) or "UNKNOWN"),
        "regime_tier": str(getattr(candidate, "regime_tier", metadata.get("regime_tier", "UNKNOWN")) or "UNKNOWN"),
        "candidate_rank": candidate_rank,
        "frontier_rank": frontier_rank,
        "frontier_selection_score": float(metadata.get("frontier_selection_score", getattr(candidate, "selection_score", 0.0)) or 0.0),
        "flow_score": float(getattr(candidate, "flow_score", 0.0) or 0.0),
        "accumulation_score": float(getattr(candidate, "accumulation_score", 0.0) or 0.0),
        **_candidate_context_metadata(metadata),
        **dict(plan_signal.get("first30_metadata") or {}),
        **dict(plan_signal.get("entry_route_metadata") or {}),
        **dict(entry_context or {}),
    }


def _ordered_entry_routes(config: KALCBConfig) -> list[tuple[int, dict[str, Any]]]:
    indexed = [(index, dict(route)) for index, route in enumerate(config.entry_plan_routes or ())]
    return sorted(indexed, key=lambda item: (int(item[1].get("priority", item[1].get("order", item[0]))), item[0]))


def _route_mode(config: KALCBConfig, route: dict[str, Any]) -> str:
    return str(route.get("mode") or route.get("plan_mode") or route.get("entry_plan_mode") or config.entry_plan_mode or "breakout")


def _entry_route_config(config: KALCBConfig, route: dict[str, Any]) -> KALCBConfig:
    mode = _route_mode(config, route)
    if mode not in SUPPORTED_ENTRY_PLAN_MODES:
        raise ValueError(f"Unsupported KALCB entry route mode: {mode}")
    mutations: dict[str, Any] = {"kalcb.entry.plan_mode": mode, "kalcb.entry.routes": ()}
    route_key_aliases = {
        "risk_mult": "kalcb.entry.route_risk_mult",
        "risk_per_trade_mult": "kalcb.entry.route_risk_mult",
        "route_risk_mult": "kalcb.entry.route_risk_mult",
        "notional_mult": "kalcb.entry.route_notional_mult",
        "route_notional_mult": "kalcb.entry.route_notional_mult",
        "participation_mult": "kalcb.entry.route_participation_mult",
        "route_participation_mult": "kalcb.entry.route_participation_mult",
        "max_session_trades": "kalcb.entry.route_max_session_trades",
        "route_max_session_trades": "kalcb.entry.route_max_session_trades",
        "context_min": "kalcb.entry.route_context_min",
        "route_context_min": "kalcb.entry.route_context_min",
        "regime_min": "kalcb.entry.route_context_min",
        "context_max": "kalcb.entry.route_context_max",
        "route_context_max": "kalcb.entry.route_context_max",
        "regime_max": "kalcb.entry.route_context_max",
        "context_exclude": "kalcb.entry.route_context_exclude",
        "route_context_exclude": "kalcb.entry.route_context_exclude",
        "context_not": "kalcb.entry.route_context_exclude",
        "level_source": "kalcb.entry.reclaim_level_source",
        "reclaim_level_source": "kalcb.entry.reclaim_level_source",
        "dynamic_notional_enabled": "kalcb.risk.dynamic_notional_enabled",
        "dynamic_max_position_notional_pct": "kalcb.risk.dynamic_max_position_notional_pct",
        "dynamic_notional_pct": "kalcb.risk.dynamic_max_position_notional_pct",
        "dynamic_max_drawdown_pct": "kalcb.risk.dynamic_max_drawdown_pct",
        "dynamic_min_session_return_pct": "kalcb.risk.dynamic_min_session_return_pct",
        "dynamic_max_open_positions": "kalcb.risk.dynamic_max_open_positions",
        "dynamic_max_open_notional_pct": "kalcb.risk.dynamic_max_open_notional_pct",
    }
    for key, value in route.items():
        key_s = str(key)
        if key_s in {"name", "route_name", "priority", "order", "mode", "plan_mode", "entry_plan_mode", "entry_plan_routes", "routes", "plan_routes"}:
            continue
        if key_s in route_key_aliases:
            mutations[route_key_aliases[key_s]] = value
        elif key_s.startswith("kalcb."):
            mutations[key_s] = value
        elif "." in key_s:
            mutations[f"kalcb.{key_s}"] = value
        else:
            mutations[f"kalcb.entry.{key_s}"] = value
    return config.with_mutations(mutations)


def _entry_route_metadata(route: dict[str, Any], *, mode: str, route_index: int) -> dict[str, Any]:
    priority = int(route.get("priority", route.get("order", route_index)))
    name = str(route.get("name") or route.get("route_name") or mode)
    return {
        "entry_route": name,
        "entry_route_mode": mode,
        "entry_route_priority": priority,
        "entry_route_index": int(route_index),
    }


def _with_entry_route_metadata(
    result: dict[str, Any],
    *,
    effective_config: KALCBConfig,
    route: dict[str, Any],
    route_index: int,
) -> dict[str, Any]:
    mode = _route_mode(effective_config, route)
    decorated = dict(result)
    route_meta = _entry_route_metadata(route, mode=mode, route_index=route_index)
    route_meta.update(
        {
            "entry_route_risk_mult": float(effective_config.entry_plan_route_risk_mult),
            "entry_route_notional_mult": float(effective_config.entry_plan_route_notional_mult),
            "entry_route_participation_mult": float(effective_config.entry_plan_route_participation_mult),
            "entry_route_max_session_trades": int(effective_config.entry_plan_route_max_session_trades),
            "entry_route_context_min_keys": sorted((effective_config.entry_plan_route_context_min or {}).keys()),
            "entry_route_context_max_keys": sorted((effective_config.entry_plan_route_context_max or {}).keys()),
            "entry_route_context_exclude_keys": sorted((effective_config.entry_plan_route_context_exclude or {}).keys()),
            "entry_route_reclaim_level_source": str(effective_config.entry_plan_reclaim_level_source or "legacy"),
        }
    )
    decorated["effective_config"] = effective_config
    decorated["entry_route_metadata"] = route_meta
    return decorated


def _legacy_reclaim_level_source(mode: str) -> str:
    return {
        "avwap_reclaim": "session_vwap",
        "pullback_acceptance": "session_vwap",
        "or_mid_reclaim": "or_mid",
        "or_high_reclaim": "or_high",
        "pdh_reclaim": "pdh",
    }.get(str(mode), "legacy")


def _candidate_campaign_level(candidate: KALCBDailyCandidate, key: str) -> float:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    value = _optional_float(metadata.get(key))
    if value is not None:
        return value
    campaign = metadata.get("structural_campaign")
    if isinstance(campaign, dict):
        nested = _optional_float(campaign.get(key))
        if nested is not None:
            return nested
    return 0.0


def _resolve_reclaim_level(
    *,
    mode: str,
    config: KALCBConfig,
    candidate: KALCBDailyCandidate,
    avwap: float,
    or_high: float,
    or_mid: float,
    prior_day_high: float,
) -> tuple[str, float]:
    source = str(config.entry_plan_reclaim_level_source or "legacy")
    if source == "legacy":
        source = _legacy_reclaim_level_source(mode)
    level_by_source = {
        "session_vwap": float(avwap or 0.0),
        "or_high": float(or_high or 0.0),
        "or_mid": float(or_mid or 0.0),
        "pdh": float(prior_day_high or 0.0),
        "campaign_avwap": _candidate_campaign_level(candidate, "campaign_avwap"),
        "campaign_box_high": _candidate_campaign_level(candidate, "campaign_box_high"),
        "campaign_box_mid": _candidate_campaign_level(candidate, "campaign_box_mid"),
        "campaign_breakout_level": _candidate_campaign_level(candidate, "campaign_breakout_level"),
    }
    return source, float(level_by_source.get(source) or 0.0)


def _mark_reclaim_touch(
    symbol_state: KALCBSymbolState,
    *,
    mode: str,
    level_source: str,
    touched_now: bool,
) -> tuple[str, bool]:
    if level_source == "session_vwap" and mode in {"avwap_reclaim", "pullback_acceptance"}:
        symbol_state.touched_vwap = symbol_state.touched_vwap or touched_now
        return "session_vwap", bool(symbol_state.touched_vwap)
    if level_source == "or_mid" and mode == "or_mid_reclaim":
        symbol_state.touched_or_mid = symbol_state.touched_or_mid or touched_now
        return "or_mid", bool(symbol_state.touched_or_mid)
    if level_source == "or_high" and mode == "or_high_reclaim":
        symbol_state.touched_or_high = symbol_state.touched_or_high or touched_now
        return "or_high", bool(symbol_state.touched_or_high)
    if level_source == "pdh" and mode == "pdh_reclaim":
        symbol_state.touched_pdh = symbol_state.touched_pdh or touched_now
        return "pdh", bool(symbol_state.touched_pdh)
    touch_key = f"{mode}:{level_source}"
    symbol_state.touched_reclaim_levels[touch_key] = bool(symbol_state.touched_reclaim_levels.get(touch_key, False) or touched_now)
    return touch_key, bool(symbol_state.touched_reclaim_levels.get(touch_key, False))


def _reclaim_close_count(bars: list[MarketBar], level: float, min_close_ret: float = 0.0) -> int:
    if level <= 0.0:
        return 0
    threshold = level * (1.0 + max(float(min_close_ret or 0.0), 0.0))
    count = 0
    for item in reversed(bars):
        if float(item.close) < threshold:
            break
        count += 1
    return count


def _entry_plan_signal_single(
    symbol_state: KALCBSymbolState,
    bar: MarketBar,
    config: KALCBConfig,
    *,
    candidate: KALCBDailyCandidate,
    prior_day_high: float,
    prior_day_close: float,
    daily_atr: float,
    expected_30m_volume: float,
    avwap: float,
    first30_signal: bool,
) -> dict[str, Any]:
    mode = str(config.entry_plan_mode or "breakout")
    bars_today = symbol_state.bars_today
    post_or_count = max(0, len(bars_today) - int(config.opening_range_bars))
    gates: list[dict[str, Any]] = []
    if first30_signal and not _is_first30_entry_mode(mode):
        return {"entry_type": None, "reason": "not_first30_plan", "gates": gates}
    if _is_first30_entry_mode(mode) and not first30_signal:
        return {"entry_type": None, "reason": "waiting_for_first30_signal", "gates": gates}
    if not first30_signal and post_or_count <= int(config.entry_plan_after_bar):
        gates.append(_gate("entry_after_bar", config.entry_plan_after_bar, post_or_count, False))
        return {"entry_type": None, "reason": "before_entry_after_bar", "gates": gates}
    if not first30_signal and config.entry_plan_max_signal_bars > 0 and post_or_count > int(config.entry_plan_max_signal_bars):
        gates.append(_gate("entry_max_signal_bars", config.entry_plan_max_signal_bars, post_or_count, False))
        return {"entry_type": None, "reason": "entry_signal_window_expired", "gates": gates}

    first30 = build_first30_features(
        bars_today,
        prior_close=prior_day_close,
        daily_atr=daily_atr,
        expected_30m_volume=expected_30m_volume,
    )
    first_bar = bars_today[0] if bars_today else bar
    first30_open = first30.open if first30 is not None else max(float(first_bar.open), 1e-9)
    first30_ret = first30.first30_ret if first30 is not None and first30_signal else float(bar.close) / first30_open - 1.0
    bar_ret = first30_ret if first30_signal else float(bar.close) / max(float(bar.open), 1e-9) - 1.0
    vwap_ret = first30.vwap_ret if first30 is not None and first30_signal else float(bar.close) / max(avwap, 1e-9) - 1.0 if avwap > 0 else 0.0
    gap = first30.gap if first30 is not None else float(first_bar.open) / max(prior_day_close, 1e-9) - 1.0 if prior_day_close > 0 else 0.0
    or_high = max(float(symbol_state.or_high), 1e-9)
    or_low = float(symbol_state.or_low)
    or_width = max(or_high - or_low, 1e-9)
    or_position = first30.range_close_location if first30 is not None and first30_signal else (float(bar.close) - or_low) / or_width if symbol_state.or_high > 0 else 0.0
    close_location = first30.range_close_location if first30 is not None and first30_signal else close_location_value(bar)
    avwap_extension = max(vwap_ret, 0.0)
    first30_metadata = first30.metadata() if first30 is not None else {}
    first30_rel_volume = first30.rel_volume if first30 is not None else 0.0
    first30_signal_cpr = first30.signal_bar_cpr if first30 is not None else close_location
    first30_open_drawdown = first30.open_drawdown if first30 is not None else 0.0
    first30_low_vs_prev_close = first30.low_vs_prev_close if first30 is not None else gap
    first30_range_atr = first30.range_atr if first30 is not None else 0.0
    entry_path_metadata = _entry_path_proof_metadata(symbol_state, candidate, config)

    def plan_result(
        entry_type: EntryType | None,
        reason: str,
        *,
        breakout_level: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"entry_type": entry_type, "reason": reason, "gates": gates}
        if breakout_level > 0:
            payload["breakout_level"] = breakout_level
        signal_metadata = {**first30_metadata, **entry_path_metadata, **dict(extra_metadata or {})}
        if signal_metadata:
            payload["first30_metadata"] = signal_metadata
        return payload

    common_checks = (
        ("entry_min_bar_ret", config.entry_plan_min_bar_ret, bar_ret, bar_ret >= config.entry_plan_min_bar_ret),
        ("entry_min_vwap_ret", config.entry_plan_min_vwap_ret, vwap_ret, vwap_ret >= config.entry_plan_min_vwap_ret),
        ("entry_min_close_location", config.entry_plan_min_close_location, close_location, close_location >= config.entry_plan_min_close_location),
        ("entry_min_or_position", config.entry_plan_min_or_position, or_position, or_position >= config.entry_plan_min_or_position),
        ("entry_gap_min", config.entry_plan_gap_min_pct, gap, gap >= config.entry_plan_gap_min_pct),
        ("entry_gap_max", config.entry_plan_gap_max_pct, gap, gap <= config.entry_plan_gap_max_pct),
        ("entry_avwap_extension_cap", config.entry_plan_max_avwap_extension_pct, avwap_extension, avwap_extension <= config.entry_plan_max_avwap_extension_pct),
        (
            "entry_first30_rel_volume",
            config.entry_plan_min_first30_rel_volume,
            first30_rel_volume,
            first30_rel_volume >= config.entry_plan_min_first30_rel_volume,
        ),
        (
            "entry_first30_signal_cpr",
            config.entry_plan_min_first30_signal_cpr,
            first30_signal_cpr,
            first30_signal_cpr >= config.entry_plan_min_first30_signal_cpr,
        ),
        (
            "entry_first30_open_drawdown",
            config.entry_plan_min_first30_open_drawdown,
            first30_open_drawdown,
            first30_open_drawdown >= config.entry_plan_min_first30_open_drawdown,
        ),
        (
            "entry_first30_low_vs_prev_close",
            config.entry_plan_min_first30_low_vs_prev_close,
            first30_low_vs_prev_close,
            first30_low_vs_prev_close >= config.entry_plan_min_first30_low_vs_prev_close,
        ),
        (
            "entry_first30_range_atr_min",
            config.entry_plan_min_first30_range_atr,
            first30_range_atr,
            first30_range_atr >= config.entry_plan_min_first30_range_atr,
        ),
        (
            "entry_first30_range_atr_max",
            config.entry_plan_max_first30_range_atr,
            first30_range_atr,
            first30_range_atr <= config.entry_plan_max_first30_range_atr,
        ),
    )
    for name, threshold, actual, passed in common_checks:
        gates.append(_gate(name, threshold, actual, bool(passed)))
        if not passed:
            return plan_result(None, name)
    if config.entry_plan_require_above_prev_close and prior_day_close > 0 and float(bar.close) < prior_day_close:
        gates.append(_gate("entry_above_prev_close", True, False, False))
        return plan_result(None, "entry_below_prev_close")
    if config.entry_plan_require_above_prev_close:
        gates.append(_gate("entry_above_prev_close", True, True, True))

    if mode == "first30_open":
        return plan_result(EntryType.FIRST30_OPEN, mode, breakout_level=or_high)
    if mode == "opening_drive":
        return plan_result(EntryType.OPENING_DRIVE, mode, breakout_level=or_high)

    raw = classify_raw_breakout(bar, prior_day_high=prior_day_high, or_high=symbol_state.or_high)
    breakout_level = _breakout_level(raw, symbol_state.or_high, prior_day_high) if raw is not None else 0.0
    if mode in {"breakout", "or_breakout", "pdh_breakout", "combined_breakout"}:
        allowed = {
            "breakout": {EntryType.OR_BREAKOUT, EntryType.PDH_BREAKOUT, EntryType.COMBINED_BREAKOUT},
            "or_breakout": {EntryType.OR_BREAKOUT, EntryType.COMBINED_BREAKOUT},
            "pdh_breakout": {EntryType.PDH_BREAKOUT, EntryType.COMBINED_BREAKOUT},
            "combined_breakout": {EntryType.COMBINED_BREAKOUT},
        }[mode]
        crossed = raw in allowed and breakout_level > 0 and float(bar.close) >= breakout_level * (1.0 + config.entry_plan_min_breakout_pct)
        gates.append(_gate("breakout_detection", sorted(item.value for item in allowed), getattr(raw, "value", None), bool(crossed)))
        if not crossed:
            return plan_result(None, "no_breakout")
        return plan_result(raw, mode, breakout_level=breakout_level)

    if mode == "post_or_momentum":
        passed = or_position >= max(config.entry_plan_min_or_position, 0.45)
        gates.append(_gate("post_or_position", max(config.entry_plan_min_or_position, 0.45), or_position, passed))
        return plan_result(EntryType.POST_OR_MOMENTUM if passed else None, mode if passed else "post_or_position", breakout_level=or_high)

    reclaim_modes = {"avwap_reclaim", "pullback_acceptance", "or_mid_reclaim", "or_high_reclaim", "pdh_reclaim"}
    if mode in reclaim_modes:
        level_source, level = _resolve_reclaim_level(
            mode=mode,
            config=config,
            candidate=candidate,
            avwap=avwap,
            or_high=or_high,
            or_mid=or_low + 0.5 * or_width,
            prior_day_high=prior_day_high,
        )
        touched_now = level > 0 and float(bar.low) <= level * (1.0 + max(config.entry_plan_max_pullback_from_vwap_pct, 0.0))
        touch_key, touched = _mark_reclaim_touch(
            symbol_state,
            mode=mode,
            level_source=level_source,
            touched_now=touched_now,
        )
        reclaimed = level > 0 and float(bar.close) >= level * (1.0 + max(config.entry_plan_min_reclaim_ret, -0.05))
        min_closes = max(1, int(config.entry_plan_min_reclaim_closes or 1))
        close_count = _reclaim_close_count(bars_today, level, config.entry_plan_min_reclaim_ret)
        close_confirmed = min_closes <= 1 or close_count >= min_closes
        passed = touched and reclaimed and close_confirmed
        gates.append(_gate(f"{mode}_touch", True, touched, touched))
        gates.append(_gate(f"{mode}_reclaim", config.entry_plan_min_reclaim_ret, float(bar.close) / max(level, 1e-9) - 1.0 if level > 0 else 0.0, reclaimed))
        gates.append(_gate(f"{mode}_close_count", min_closes, close_count, close_confirmed))
        level_metadata = {
            "entry_reclaim_level_source": level_source,
            "entry_reclaim_level": float(level),
            "entry_reclaim_mode": mode,
            "entry_reclaim_touch_key": touch_key,
            "entry_reclaim_min_closes": min_closes,
            "entry_reclaim_close_count": close_count,
            "entry_reclaim_close_confirmed": bool(close_confirmed),
            "entry_reclaim_touched": bool(touched),
            "entry_reclaim_reclaimed": bool(reclaimed),
            "entry_reclaim_distance_pct": float(bar.close) / max(level, 1e-9) - 1.0 if level > 0 else 0.0,
        }
        entry_type = {
            "avwap_reclaim": EntryType.AVWAP_RECLAIM,
            "pullback_acceptance": EntryType.PULLBACK_ACCEPTANCE,
            "or_mid_reclaim": EntryType.OR_MID_RECLAIM,
            "or_high_reclaim": EntryType.OR_HIGH_RECLAIM,
            "pdh_reclaim": EntryType.PDH_RECLAIM,
        }[mode]
        return plan_result(entry_type if passed else None, mode if passed else f"{mode}_missing", breakout_level=level, extra_metadata=level_metadata)

    if mode == "deferred_continuation":
        prior_high = max((float(item.high) for item in bars_today[config.opening_range_bars : -1]), default=or_high)
        passed = prior_high > 0 and float(bar.close) >= prior_high * (1.0 + config.entry_plan_min_breakout_pct)
        gates.append(_gate("deferred_continuation_high", config.entry_plan_min_breakout_pct, float(bar.close) / max(prior_high, 1e-9) - 1.0, passed))
        return plan_result(EntryType.DEFERRED_CONTINUATION if passed else None, mode if passed else "deferred_continuation_missing", breakout_level=prior_high)

    return plan_result(None, "unsupported_entry_plan")


def _entry_path_proof_metadata(
    symbol_state: KALCBSymbolState,
    candidate: KALCBDailyCandidate,
    config: KALCBConfig,
) -> dict[str, Any]:
    bars = list(symbol_state.bars_today)
    anchor_index = int(config.opening_range_bars)
    if len(bars) <= anchor_index:
        return {}
    post_first30 = bars[anchor_index:]
    anchor = post_first30[0]
    entry = max(float(anchor.open), 1e-9)
    risk = max(entry * max(float(config.exit_stop_pct or 0.0), 1e-6), 1.0)
    metadata: dict[str, Any] = {
        "entry_path_anchor_time": anchor.timestamp.isoformat(),
        "entry_path_anchor_price": float(entry),
        "entry_path_risk_per_share": float(risk),
        "entry_path_stop_pct": float(config.exit_stop_pct or 0.0),
        "entry_path_completed_bars": int(len(post_first30)),
        "entry_path_reference": "first_post_first30_open_fixed_stop_pct",
    }
    if candidate.daily_atr > 0:
        metadata["entry_path_risk_atr_fraction"] = float(risk) / float(candidate.daily_atr)
    for horizon in (1, 3, 6, 12):
        if len(post_first30) < horizon:
            continue
        metadata.update(_entry_path_horizon_metadata(post_first30[:horizon], entry, risk, horizon))
    return metadata


def _entry_path_horizon_metadata(
    bars: list[MarketBar],
    entry: float,
    risk: float,
    horizon: int,
) -> dict[str, float]:
    last = bars[-1]
    high = max(float(item.high) for item in bars)
    low = min(float(item.low) for item in bars)
    current_r = (float(last.close) - entry) / risk
    mfe_r = max(0.0, (high - entry) / risk)
    mae_r = (low - entry) / risk
    prefix = f"h{int(horizon)}"
    values = {
        f"{prefix}_current_r": float(current_r),
        f"{prefix}_mfe_r": float(mfe_r),
        f"{prefix}_mae_r": float(mae_r),
        f"{prefix}_giveback_r": float(max(0.0, mfe_r - current_r)),
        f"{prefix}_close_location": float(close_location_value(last)),
        f"{prefix}_recent_return": float(float(last.close) / max(entry, 1e-9) - 1.0),
        f"{prefix}_down_streak": float(_threshold_streak(bars, lambda item: float(item.close) < float(item.open))),
        f"{prefix}_below_entry_streak": float(_threshold_streak(bars, lambda item: float(item.close) < entry)),
    }
    return {**values, **{f"entry_path_{key}": value for key, value in values.items()}}


def _initial_stop_price(
    bar: MarketBar,
    symbol_state: KALCBSymbolState,
    candidate: KALCBDailyCandidate,
    config: KALCBConfig,
    *,
    avwap: float,
) -> float:
    mode = str(config.exit_stop_mode or "momentum")
    if mode == "fixed_pct":
        return round_price_for_krx(float(bar.close) * (1.0 - max(config.exit_stop_pct, 0.001)), "protective_stop")
    if mode == "vwap" and avwap > 0:
        return round_price_for_krx(min(avwap * 0.997, float(bar.close) * 0.997), "protective_stop")
    if mode == "first30_low" and symbol_state.or_low > 0:
        return round_price_for_krx(min(symbol_state.or_low, float(bar.close) * 0.997), "protective_stop")
    if mode in {"signal_low", "entry_low"}:
        return round_price_for_krx(min(float(bar.low), float(bar.close) * 0.997), "protective_stop")
    return momentum_stop_price(bar.close, symbol_state.or_low, bar.low, candidate.daily_atr, config)


def _evaluate_entry(
    state: KALCBState,
    symbol_state: KALCBSymbolState,
    bar: MarketBar,
    config: KALCBConfig,
    portfolio: KALCBPortfolioView,
    *,
    first30_signal: bool = False,
) -> KALCBCoreResult:
    candidate = symbol_state.candidate
    if candidate is None:
        return KALCBCoreResult(state)
    gates: list[dict[str, Any]] = []

    prior_day_high = float(candidate.prior_day_high)
    prior_day_close = float(candidate.prior_day_close)
    expected_vol = float(candidate.expected_5m_volume or candidate.average_30m_volume / 6.0 or 1.0)
    bar_rvol = compute_bar_rvol(bar.volume, expected_vol)
    cpr = close_location_value(bar)
    avwap = symbol_state.vwap
    adx_value = float(bar.metadata.get("adx", candidate.metadata.get("adx", 0.0)) or 0.0)
    sector_flow = float(bar.metadata.get("sector_flow", candidate.flow_score) or 0.0)
    bars_today = symbol_state.bars_today
    plan_signal = _entry_plan_signal(
        symbol_state,
        bar,
        config,
        prior_day_high=prior_day_high,
        prior_day_close=prior_day_close,
        daily_atr=candidate.daily_atr,
        expected_30m_volume=candidate.average_30m_volume or expected_vol * 6.0,
        avwap=avwap,
        first30_signal=first30_signal,
        candidate=candidate,
        entry_context=dict(state.meta.get("entry_context") or {}),
    )
    gates.extend(plan_signal["gates"])
    entry_type = plan_signal["entry_type"]
    effective_config = plan_signal.get("effective_config")
    if isinstance(effective_config, KALCBConfig):
        config = effective_config
    if entry_type is None:
        symbol_state.rejected_reason = str(plan_signal["reason"])
        if config.fast_replay_suppress_rejections:
            return KALCBCoreResult(state)
        return _rejected(
            state,
            bar,
            str(plan_signal["reason"]),
            gates,
            {
                "candidate_rank": int(symbol_state.candidate_rank or 0),
                **dict(plan_signal.get("first30_metadata") or {}),
                **dict(plan_signal.get("entry_route_metadata") or {}),
                **dict(plan_signal.get("entry_route_context") or {}),
                **dict(state.meta.get("entry_context") or {}),
                "entry_route_attempts": list(plan_signal.get("entry_route_attempts") or ()),
            },
        )
    stop_price = _initial_stop_price(bar, symbol_state, candidate, config, avwap=avwap)
    risk_per_share = max(float(bar.close) - stop_price, 0.0)
    momentum_score, score_detail = compute_momentum_score(
        bar,
        bars_today,
        prior_day_high=prior_day_high,
        prior_day_close=prior_day_close,
        or_high=symbol_state.or_high,
        avwap=avwap,
        adx_value=adx_value,
        sector_flow=sector_flow,
        config=config,
    )
    avwap_dist_pct = (bar.close - avwap) / avwap if avwap > 0 else 0.0
    breakout_level = float(plan_signal.get("breakout_level") or _breakout_level(entry_type, symbol_state.or_high, prior_day_high))
    candidate_rank = int(symbol_state.candidate_rank or 0)
    frontier_rank = int(candidate.metadata.get("frontier_rank") or candidate_rank or 0)
    frontier_initial_active = bool(candidate.metadata.get("frontier_initial_active", True))
    frontier_score_raw = candidate.metadata.get("frontier_selection_score")
    frontier_selection_score = float(frontier_score_raw if frontier_score_raw is not None else candidate.selection_score or 0.0)
    candidate_context_meta = _candidate_context_metadata(dict(candidate.metadata or {}))
    source_meta = {
        "candidate_rank": candidate_rank,
        "frontier_rank": frontier_rank,
        "frontier_selection_score": frontier_selection_score,
        "flow_score": float(candidate.flow_score or 0.0),
        "accumulation_score": float(candidate.accumulation_score or 0.0),
        **candidate_context_meta,
    }
    context_meta = dict(state.meta.get("entry_context") or {})
    entry_meta = {
        **_entry_metadata(entry_type, momentum_score, score_detail, bar_rvol, cpr, avwap),
        **dict(plan_signal.get("first30_metadata") or {}),
        **dict(plan_signal.get("entry_route_metadata") or {}),
        **context_meta,
    }
    context_reason, context_gates = _entry_context_gate_reject_reason(config, {**source_meta, **entry_meta})
    if context_gates:
        gates.extend(context_gates)
    if context_reason:
        return _rejected(state, bar, context_reason, gates, {**entry_meta, **source_meta})
    entry_route_session_count = 0
    route_limit = int(config.entry_plan_route_max_session_trades or 0)
    if route_limit > 0:
        route_name = str(entry_meta.get("entry_route") or "")
        entry_route_session_count = _entry_route_session_count(state, route_name)
        route_allowed = entry_route_session_count < route_limit
        gates.append(_gate("entry_route_session_limit", route_limit, entry_route_session_count, route_allowed))
        if not route_allowed:
            return _rejected(
                state,
                bar,
                "entry_route_session_limit",
                gates,
                {**entry_meta, **source_meta, "entry_route_session_count": entry_route_session_count},
            )

    if config.entry_plan_require_initial_active:
        gates.append(_gate("entry_initial_active", True, frontier_initial_active, frontier_initial_active))
        if not frontier_initial_active:
            return _rejected(
                state,
                bar,
                "entry_not_initial_active",
                gates,
                {**entry_meta, **source_meta},
            )
    if config.entry_plan_max_frontier_rank > 0:
        rank_passed = 0 < frontier_rank <= int(config.entry_plan_max_frontier_rank)
        gates.append(_gate("entry_frontier_rank", int(config.entry_plan_max_frontier_rank), frontier_rank, rank_passed))
        if not rank_passed:
            return _rejected(
                state,
                bar,
                "entry_frontier_rank",
                gates,
                {**entry_meta, **source_meta},
            )
    if config.entry_plan_min_frontier_score > -9.0:
        score_passed = frontier_selection_score >= float(config.entry_plan_min_frontier_score)
        gates.append(_gate("entry_frontier_score", config.entry_plan_min_frontier_score, frontier_selection_score, score_passed))
        if not score_passed:
            return _rejected(
                state,
                bar,
                "entry_frontier_score",
                gates,
                {**entry_meta, **source_meta},
            )
    if config.entry_plan_min_flow_score > -9.0:
        flow_passed = float(candidate.flow_score or 0.0) >= float(config.entry_plan_min_flow_score)
        gates.append(_gate("entry_flow_score", config.entry_plan_min_flow_score, float(candidate.flow_score or 0.0), flow_passed))
        if not flow_passed:
            return _rejected(state, bar, "entry_flow_score", gates, {**entry_meta, **source_meta})
    if config.entry_plan_min_accumulation_score > -9.0:
        accumulation_passed = float(candidate.accumulation_score or 0.0) >= float(config.entry_plan_min_accumulation_score)
        gates.append(_gate("entry_accumulation_score", config.entry_plan_min_accumulation_score, float(candidate.accumulation_score or 0.0), accumulation_passed))
        if not accumulation_passed:
            return _rejected(state, bar, "entry_accumulation_score", gates, {**entry_meta, **source_meta})

    quality_votes = _entry_quality_votes(entry_meta, source_meta, config)
    if quality_votes is not None:
        vote_count, vote_required, vote_gates = quality_votes
        gates.extend(vote_gates)
        entry_meta["entry_quality_votes"] = vote_count
        entry_meta["entry_quality_required_votes"] = vote_required
        entry_meta["entry_quality_vote_details"] = {
            str(gate["filter_name"]).replace("entry_quality_", ""): bool(gate["passed"])
            for gate in vote_gates
            if gate["filter_name"] != "entry_quality_votes"
        }
        if vote_count < vote_required:
            return _rejected(state, bar, "entry_quality_votes", gates, {**entry_meta, **source_meta})

    if bar_rvol < config.rvol_threshold:
        gates.append(_gate("rvol_min", config.rvol_threshold, bar_rvol, False))
        return _rejected(state, bar, "rvol_below_min", gates, {**entry_meta, "candidate_rank": candidate_rank})
    gates.append(_gate("rvol_min", config.rvol_threshold, bar_rvol, True))

    if cpr < config.cpr_threshold:
        relax_threshold = float(config.cpr_relax_threshold or 0.0)
        relax_min_score = int(config.cpr_relax_min_score or 0)
        if relax_threshold <= 0 or cpr < relax_threshold:
            gates.append(_gate("cpr_relax_floor", relax_threshold or config.cpr_threshold, cpr, False))
            return _rejected(state, bar, "cpr_below_min", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("cpr_relax_floor", relax_threshold, cpr, True))
        if relax_min_score > 0 and momentum_score < relax_min_score:
            gates.append(_gate("cpr_relax_score", relax_min_score, momentum_score, False))
            return _rejected(state, bar, "cpr_relax_score_too_low", gates, {**entry_meta, "candidate_rank": candidate_rank})
        if relax_min_score > 0:
            gates.append(_gate("cpr_relax_score", relax_min_score, momentum_score, True))
        gates.append(_gate("cpr_relax", {"threshold": relax_threshold, "min_score": relax_min_score}, {"cpr": cpr, "score": momentum_score}, True))
    else:
        gates.append(_gate("cpr_gate", config.cpr_threshold, cpr, True))

    if avwap > 0 and bar.close < avwap:
        gates.append(_gate("avwap_filter", avwap, bar.close, False))
        return _rejected(state, bar, "below_avwap", gates, entry_meta)
    gates.append(_gate("avwap_filter", avwap, bar.close, True))

    if config.rvol_max < 999 and bar_rvol > config.rvol_max:
        gates.append(_gate("rvol_cap", config.rvol_max, bar_rvol, False))
        return _rejected(state, bar, "rvol_exceeded", gates, {**entry_meta, "candidate_rank": candidate_rank})
    gates.append(_gate("rvol_cap", config.rvol_max, bar_rvol, True))

    if entry_type == EntryType.OR_BREAKOUT and config.or_breakout_min_rvol > 0 and bar_rvol < config.or_breakout_min_rvol:
        gates.append(_gate("or_breakout_min_rvol", config.or_breakout_min_rvol, bar_rvol, False, applicable=True))
        return _rejected(
            state,
            bar,
            "or_breakout_rvol_too_low",
            gates,
            {**entry_meta, "candidate_rank": candidate_rank},
        )
    if entry_type == EntryType.OR_BREAKOUT and config.or_breakout_min_rvol > 0:
        gates.append(_gate("or_breakout_min_rvol", config.or_breakout_min_rvol, bar_rvol, True, applicable=True))

    if entry_type == EntryType.PDH_BREAKOUT and config.pdh_breakout_min_rvol > 0 and bar_rvol < config.pdh_breakout_min_rvol:
        gates.append(_gate("pdh_breakout_min_rvol", config.pdh_breakout_min_rvol, bar_rvol, False, applicable=True))
        return _rejected(
            state,
            bar,
            "pdh_breakout_rvol_too_low",
            gates,
            {**entry_meta, "candidate_rank": candidate_rank},
        )
    if entry_type == EntryType.PDH_BREAKOUT and config.pdh_breakout_min_rvol > 0:
        gates.append(_gate("pdh_breakout_min_rvol", config.pdh_breakout_min_rvol, bar_rvol, True, applicable=True))

    if momentum_score < config.momentum_score_min:
        gates.append(_gate("momentum_score_gate", config.momentum_score_min, momentum_score, False))
        return _rejected(state, bar, "momentum_score_below_min", gates, {**entry_meta, "candidate_rank": candidate_rank})
    gates.append(_gate("momentum_score_gate", config.momentum_score_min, momentum_score, True))

    if entry_type == EntryType.COMBINED_BREAKOUT:
        if config.block_combined_regime_b and candidate.regime_tier.upper() == "B":
            gates.append(_gate("combined_regime_block", "not_B", candidate.regime_tier, False, applicable=True))
            return _rejected(state, bar, "combined_blocked_in_tier_b", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("combined_regime_block", "not_B", candidate.regime_tier, True, applicable=True))
        if momentum_score < config.combined_breakout_score_min:
            gates.append(_gate("combined_score", config.combined_breakout_score_min, momentum_score, False, applicable=True))
            return _rejected(state, bar, "combined_score_too_low", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("combined_score", config.combined_breakout_score_min, momentum_score, True, applicable=True))
        if bar_rvol < config.combined_breakout_min_rvol:
            gates.append(_gate("combined_rvol", config.combined_breakout_min_rvol, bar_rvol, False, applicable=True))
            return _rejected(state, bar, "combined_rvol_too_low", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("combined_rvol", config.combined_breakout_min_rvol, bar_rvol, True, applicable=True))
        if config.combined_avwap_cap_pct > 0 and avwap > 0 and avwap_dist_pct > config.combined_avwap_cap_pct:
            gates.append(_gate("combined_avwap_cap", config.combined_avwap_cap_pct, avwap_dist_pct, False, applicable=True))
            return _rejected(state, bar, "combined_avwap_distance_exceeded", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("combined_avwap_cap", config.combined_avwap_cap_pct, avwap_dist_pct, True, applicable=True))

    if entry_type == EntryType.PDH_BREAKOUT:
        if bar.timestamp.time() > config.pdh_entry_window_end:
            gates.append(_gate("pdh_entry_window", str(config.pdh_entry_window_end), str(bar.timestamp.time()), False, applicable=True))
            return _rejected(state, bar, "outside_pdh_entry_window", gates, {**entry_meta, "candidate_rank": candidate_rank})
        if config.pdh_avwap_cap_pct > 0 and avwap > 0 and avwap_dist_pct > config.pdh_avwap_cap_pct:
            gates.append(_gate("pdh_avwap_cap", config.pdh_avwap_cap_pct, avwap_dist_pct, False, applicable=True))
            return _rejected(state, bar, "pdh_avwap_distance_exceeded", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("pdh_avwap_cap", config.pdh_avwap_cap_pct, avwap_dist_pct, True, applicable=True))

    if entry_type == EntryType.OR_BREAKOUT and _in_time_window(bar.timestamp.time(), config.or_caution_window_start, config.or_caution_window_end):
        if config.or_caution_min_rvol > 0 and bar_rvol < config.or_caution_min_rvol:
            gates.append(_gate("or_caution_rvol", config.or_caution_min_rvol, bar_rvol, False, applicable=True))
            return _rejected(
                state,
                bar,
                "or_caution_rvol_too_low",
                gates,
                {**entry_meta, "candidate_rank": candidate_rank},
            )
        if config.or_caution_min_rvol > 0:
            gates.append(_gate("or_caution_rvol", config.or_caution_min_rvol, bar_rvol, True, applicable=True))
        if config.or_caution_max_avwap_dist_pct > 0 and avwap > 0 and avwap_dist_pct > config.or_caution_max_avwap_dist_pct:
            gates.append(_gate("or_caution_avwap_cap", config.or_caution_max_avwap_dist_pct, avwap_dist_pct, False, applicable=True))
            return _rejected(
                state,
                bar,
                "or_caution_avwap_distance_exceeded",
                gates,
                {**entry_meta, "candidate_rank": candidate_rank},
            )
        if config.or_caution_max_avwap_dist_pct > 0:
            gates.append(_gate("or_caution_avwap_cap", config.or_caution_max_avwap_dist_pct, avwap_dist_pct, True, applicable=True))
        if config.or_caution_require_two_close and not _two_close_confirmed(bars_today, breakout_level):
            gates.append(_gate("or_caution_two_close", True, False, False, applicable=True))
            return _rejected(
                state,
                bar,
                "or_caution_two_close_missing",
                gates,
                {**entry_meta, "candidate_rank": candidate_rank},
            )
        if config.or_caution_require_two_close:
            gates.append(_gate("or_caution_two_close", True, True, True, applicable=True))

    if config.secondary_rank_start > 0 and candidate_rank >= config.secondary_rank_start:
        gates.append(_gate("secondary_rank_gate", config.secondary_rank_start, candidate_rank, True, applicable=True))
        if config.secondary_min_score > 0 and momentum_score < config.secondary_min_score:
            gates.append(_gate("secondary_min_score", config.secondary_min_score, momentum_score, False, applicable=True))
            return _rejected(
                state,
                bar,
                "secondary_score_too_low",
                gates,
                {**entry_meta, "candidate_rank": candidate_rank},
            )
        if config.secondary_min_score > 0:
            gates.append(_gate("secondary_min_score", config.secondary_min_score, momentum_score, True, applicable=True))
        if config.secondary_min_rvol > 0 and bar_rvol < config.secondary_min_rvol:
            gates.append(_gate("secondary_min_rvol", config.secondary_min_rvol, bar_rvol, False, applicable=True))
            return _rejected(
                state,
                bar,
                "secondary_rvol_too_low",
                gates,
                {**entry_meta, "candidate_rank": candidate_rank},
            )
        if config.secondary_min_rvol > 0:
            gates.append(_gate("secondary_min_rvol", config.secondary_min_rvol, bar_rvol, True, applicable=True))
        route_time_ok = entry_type != EntryType.OR_BREAKOUT or bar.timestamp.time() >= config.secondary_late_time
        if config.secondary_require_pdh_or_late and not route_time_ok:
            gates.append(_gate("secondary_pdh_or_late", True, False, False, applicable=True))
            return _rejected(
                state,
                bar,
                "secondary_route_time_block",
                gates,
                {**entry_meta, "candidate_rank": candidate_rank},
            )
        if config.secondary_require_pdh_or_late:
            gates.append(_gate("secondary_pdh_or_late", True, True, True, applicable=True))

    if symbol_state.or_high > 0 and config.or_width_min_pct > 0:
        or_width_pct = (symbol_state.or_high - symbol_state.or_low) / symbol_state.or_high
        if or_width_pct < config.or_width_min_pct:
            gates.append(_gate("or_width_min", config.or_width_min_pct, or_width_pct, False))
            return _rejected(state, bar, "or_width_too_narrow", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("or_width_min", config.or_width_min_pct, or_width_pct, True))
        if config.or_width_max_pct > 0 and or_width_pct > config.or_width_max_pct:
            gates.append(_gate("or_width_max", config.or_width_max_pct, or_width_pct, False))
            return _rejected(state, bar, "or_width_too_wide", gates, {**entry_meta, "candidate_rank": candidate_rank})

    if config.breakout_distance_cap_r > 0 and risk_per_share > 0:
        breakout_dist_r = (bar.close - breakout_level) / risk_per_share
        if breakout_dist_r > config.breakout_distance_cap_r:
            gates.append(_gate("breakout_distance_cap", config.breakout_distance_cap_r, breakout_dist_r, False))
            return _rejected(state, bar, "breakout_distance_exceeded", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("breakout_distance_cap", config.breakout_distance_cap_r, breakout_dist_r, True))

    if config.orb_entry_range_cap_r > 0 and risk_per_share > 0 and symbol_state.or_low > 0:
        entry_range_r = (bar.close - symbol_state.or_low) / risk_per_share
        if entry_range_r > config.orb_entry_range_cap_r:
            gates.append(_gate("orb_entry_range_cap", config.orb_entry_range_cap_r, entry_range_r, False))
            return _rejected(state, bar, "orb_entry_range_exceeded", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("orb_entry_range_cap", config.orb_entry_range_cap_r, entry_range_r, True))

    if portfolio.open_positions + _pending_entry_count(state) >= config.max_positions:
        gates.append(_gate("max_positions", config.max_positions, portfolio.open_positions, False))
        return _rejected(state, bar, "max_positions", gates, {**entry_meta, "candidate_rank": candidate_rank})
    gates.append(_gate("max_positions", config.max_positions, portfolio.open_positions, True))

    sector = str(candidate.sector or "UNKNOWN")
    if _known_sector(sector):
        sector_count = portfolio.sector_counts.get(sector, 0) + _pending_sector_count(state, sector)
        if sector_count >= config.max_per_sector:
            gates.append(_gate("sector_limit", config.max_per_sector, sector_count, False))
            return _rejected(state, bar, "sector_limit", gates, {**entry_meta, "candidate_rank": candidate_rank})
        gates.append(_gate("sector_limit", config.max_per_sector, sector_count, True))

    risk_budget = max(portfolio.cash * config.risk_per_trade_pct, 1.0)
    heat_r = portfolio.open_risk / risk_budget
    if heat_r >= config.heat_cap_r:
        gates.append(_gate("heat_cap", config.heat_cap_r, heat_r, False))
        return _rejected(state, bar, "heat_cap_exceeded", gates, {**entry_meta, "candidate_rank": candidate_rank})
    gates.append(_gate("heat_cap", config.heat_cap_r, heat_r, True))

    reg_mult = regime_size_mult(candidate.regime_tier, config)
    if reg_mult <= 0:
        gates.append(_gate("regime_gate", 0.0, reg_mult, False))
        return _rejected(state, bar, "regime_blocked", gates, {**entry_meta, "candidate_rank": candidate_rank})
    gates.append(_gate("regime_gate", 0.0, reg_mult, True))

    if conditional_entry_blocked(candidate, entry_type, momentum_score, config, score_detail):
        gates.append(_gate("conditional_entry_block", False, True, False))
        return _rejected(state, bar, "conditional_entry_block", gates, {**entry_meta, "candidate_rank": candidate_rank})
    gates.append(_gate("conditional_entry_block", False, False, True))

    qty = compute_entry_qty(
        cash=portfolio.cash,
        open_notional=portfolio.open_notional,
        portfolio_equity=portfolio.equity,
        portfolio_drawdown_pct=portfolio.drawdown_pct,
        portfolio_session_return_pct=portfolio.session_return_pct,
        open_positions=portfolio.open_positions,
        entry_price=bar.close,
        stop_price=stop_price,
        config=config,
        candidate=candidate,
        entry_type=entry_type,
        momentum_score=momentum_score,
        score_detail=score_detail,
    )
    if qty <= 0:
        gates.append(_gate("qty_sizing", 1, qty, False))
        return _rejected(state, bar, "qty_zero", gates, {**entry_meta, "candidate_rank": candidate_rank})
    gates.append(_gate("qty_sizing", 1, qty, True))
    effective_notional_pct = effective_max_position_notional_pct(
        config=config,
        open_notional=portfolio.open_notional,
        portfolio_equity=portfolio.equity,
        portfolio_drawdown_pct=portfolio.drawdown_pct,
        portfolio_session_return_pct=portfolio.session_return_pct,
        open_positions=portfolio.open_positions,
    )

    metadata = {
        **entry_meta,
        **candidate_context_meta,
        "order_role": "ENTRY",
        "fill_timing": config.live_parity_fill_timing,
        "live_parity_fill_timing": config.live_parity_fill_timing,
        "signal_bar": bar.timestamp.isoformat(),
        "candidate_hash": candidate.source_fingerprint,
        "source_artifact_hash": state.snapshot_hash,
        "or_high": symbol_state.or_high,
        "or_low": symbol_state.or_low,
        "prior_day_high": prior_day_high,
        "candidate_rank": candidate_rank,
        "frontier_rank": frontier_rank,
        "frontier_initial_active": frontier_initial_active,
        "frontier_role": str(candidate.metadata.get("frontier_role") or "initial_active"),
        "frontier_selection_mode": str(candidate.metadata.get("frontier_selection_mode") or ""),
        "frontier_selection_score": frontier_selection_score,
        "flow_score": float(candidate.flow_score or 0.0),
        "accumulation_score": float(candidate.accumulation_score or 0.0),
        "entry_price_ref": bar.close,
        "stop_price": stop_price,
        "risk_per_share": max(bar.close - stop_price, 1.0),
        "effective_max_position_notional_pct": effective_notional_pct,
        "portfolio_equity": float(portfolio.equity or 0.0),
        "portfolio_drawdown_pct": float(portfolio.drawdown_pct or 0.0),
        "portfolio_session_return_pct": float(portfolio.session_return_pct or 0.0),
        "daily_atr": candidate.daily_atr,
        "sector": candidate.sector,
        "regime_tier": candidate.regime_tier,
        "filter_decisions": gates,
        "gate_decisions": {gate["filter_name"]: gate["passed"] for gate in gates},
        "entry_route_attempts": list(plan_signal.get("entry_route_attempts") or ()),
        "entry_route_session_count_before": entry_route_session_count,
        "core_version": KALCB_CORE_VERSION,
    }
    action = SubmitEntry(
        strategy_id=STRATEGY_ID,
        symbol=bar.symbol,
        qty=qty,
        order_type="MARKET",
        limit_price=None,
        stop_price=stop_price,
        reason="kalcb_next_5m_open",
        metadata=metadata,
    )
    symbol_state.stage = SymbolStage.ENTRY_QUEUED
    symbol_state.entry_attempted = True
    symbol_state.pending_entry_metadata = dict(metadata)
    _increment_entry_route_session_count(state, str(metadata.get("entry_route") or ""))
    decision = _decision(bar, "entry", "kalcb_breakout_next_5m_open", actions=[action], metadata=metadata)
    return KALCBCoreResult(state, [action], [decision])


def _manage_position(state: KALCBState, symbol_state: KALCBSymbolState, bar: MarketBar, config: KALCBConfig) -> KALCBCoreResult:
    position = symbol_state.position
    if position is None:
        return KALCBCoreResult(state)
    position.update_mark(high=bar.high, low=bar.low)
    position.hold_bars += 1
    if position.exit_in_flight:
        return KALCBCoreResult(state)
    actions: list[StrategyAction] = []
    decisions: list[DecisionEvent] = []

    current_r = position.unrealized_r(bar.close)
    mfe_r = position.mfe_r()
    stop_target, stop_reason = failure_stop_target(
        current_stop=position.current_stop,
        entry_price=position.entry_price,
        risk_per_share=position.risk_per_share,
        close_price=bar.close,
        hold_bars=position.hold_bars,
        mfe_r=mfe_r,
        unrealized_r=current_r,
        config=config,
    )
    trail_target, trail_reason = trailing_stop_from_mfe(
        current_stop=stop_target,
        entry_price=position.entry_price,
        risk_per_share=position.risk_per_share,
        mfe_r=mfe_r,
        hold_bars=position.hold_bars,
        reference_price=bar.close,
        config=config,
    )
    target_stop = max(stop_target, trail_target)
    reason = trail_reason if trail_target >= stop_target and trail_reason else stop_reason
    if config.exit_breakeven_trigger_r > 0 and mfe_r >= config.exit_breakeven_trigger_r:
        breakeven_target = position.entry_price + config.exit_breakeven_stop_r * position.risk_per_share
        if breakeven_target > target_stop:
            target_stop = breakeven_target
            reason = "breakeven_stop"
    if config.exit_trail_start_r > 0 and config.exit_trail_gap_r > 0 and mfe_r >= config.exit_trail_start_r:
        trail_r = mfe_r - config.exit_trail_gap_r
        if trail_r > 0:
            generic_trail = position.entry_price + trail_r * position.risk_per_share
            if generic_trail > target_stop and generic_trail < bar.close:
                target_stop = generic_trail
                reason = "trade_plan_trail"
    if config.exit_hard_stop_enabled and reason and target_stop > position.current_stop:
        rounded = _valid_protective_stop(target_stop, bar.close)
        if rounded > position.current_stop:
            position.current_stop = rounded
            position.stop_tightened = True
            action = ReplaceProtectiveStop(
                strategy_id=STRATEGY_ID,
                symbol=position.symbol,
                qty=position.qty_open,
                stop_price=rounded,
                reason=reason,
                metadata=_exit_metadata(position, "protected_stop", {"order_role": "STOP", "stop_kind": reason}),
            )
            actions.append(action)
            decisions.append(_decision(bar, "stop_replace", reason, actions=[action], metadata=dict(action.metadata)))

    if (
        not config.exit_hard_stop_enabled
        and config.exit_conditional_stop_activate_r > 0
        and config.exit_conditional_stop_gap_r > 0
        and position.hold_bars >= config.exit_conditional_stop_min_hold_bars
        and mfe_r >= config.exit_conditional_stop_activate_r
    ):
        stop_r = mfe_r - config.exit_conditional_stop_gap_r
        conditional_stop = position.entry_price + stop_r * position.risk_per_share
        if stop_r > 0 and conditional_stop > position.current_stop and conditional_stop < bar.close:
            rounded = _valid_protective_stop(conditional_stop, bar.close)
            if rounded > position.current_stop:
                position.current_stop = rounded
                position.stop_tightened = True
                stop_metadata = _exit_metadata(position, "conditional_stop", {"order_role": "STOP", "stop_kind": "conditional_mfe_stop"})
                if position.stop_order_id:
                    action = ReplaceProtectiveStop(
                        strategy_id=STRATEGY_ID,
                        symbol=position.symbol,
                        qty=position.qty_open,
                        stop_price=rounded,
                        reason="conditional_mfe_stop",
                        metadata=stop_metadata,
                    )
                    decisions.append(_decision(bar, "stop_replace", "conditional_mfe_stop", actions=[action], metadata=dict(action.metadata)))
                else:
                    action = SubmitProtectiveStop(
                        strategy_id=STRATEGY_ID,
                        symbol=position.symbol,
                        qty=position.qty_open,
                        stop_price=rounded,
                        reason="conditional_mfe_stop",
                        metadata=stop_metadata,
                    )
                    decisions.append(_decision(bar, "stop_submit", "conditional_mfe_stop", actions=[action], metadata=dict(action.metadata)))
                actions.append(action)
                return KALCBCoreResult(state, actions, decisions)

    if should_take_partial(
        qty_open=position.qty_open,
        partial_taken=position.partial_taken,
        partial_order_id=position.partial_order_id,
        unrealized_r=current_r,
        config=config,
    ):
        qty = partial_exit_qty(position.qty_open, config)
        action = SubmitPartialExit(
            strategy_id=STRATEGY_ID,
            symbol=position.symbol,
            qty=qty,
            order_type="MARKET",
            limit_price=None,
            reason="partial_profit",
            metadata=_exit_metadata(position, "partial", {"order_role": "TP"}),
        )
        position.partial_order_id = "__pending__"
        actions.append(action)
        decisions.append(_decision(bar, "partial_exit", "partial_profit", actions=[action], metadata=dict(action.metadata)))
        return KALCBCoreResult(state, actions, decisions)

    exit_reason = _exit_reason(symbol_state, position, bar, config, current_r=current_r, mfe_r=mfe_r)
    if exit_reason:
        action: StrategyAction
        if exit_reason.startswith("eod"):
            action = FlattenPosition(
                strategy_id=STRATEGY_ID,
                symbol=position.symbol,
                reason=exit_reason,
                metadata=_exit_metadata(position, exit_reason, {"order_role": "EXIT"}),
            )
        else:
            action = SubmitExit(
                strategy_id=STRATEGY_ID,
                symbol=position.symbol,
                qty=position.qty_open,
                order_type="MARKET",
                limit_price=None,
                reason=exit_reason,
                metadata=_exit_metadata(position, exit_reason, {"order_role": "EXIT"}),
            )
        position.exit_in_flight = True
        actions.append(action)
        decisions.append(_decision(bar, "exit", exit_reason, actions=[action], metadata=dict(action.metadata)))
    return KALCBCoreResult(state, actions, decisions)


def _exit_reason(
    symbol_state: KALCBSymbolState,
    position: KALCBPositionState,
    bar: MarketBar,
    config: KALCBConfig,
    *,
    current_r: float,
    mfe_r: float,
) -> str:
    if (
        config.exit_conditional_target_enabled
        and config.exit_conditional_target_r > 0
        and position.hold_bars >= config.exit_conditional_target_min_hold_bars
        and current_r >= config.exit_conditional_target_r
        and _conditional_target_cohort_matches(position, config)
    ):
        return "conditional_target_r"
    if config.exit_target_r > 0 and current_r >= config.exit_target_r:
        return "target_r"
    if config.exit_no_mfe_bars > 0 and position.hold_bars == config.exit_no_mfe_bars and mfe_r < config.exit_no_mfe_thresh_r:
        return "no_mfe_exit"
    failed_followthrough_due = (
        position.hold_bars >= config.exit_failed_followthrough_bars
        if config.exit_failed_followthrough_persistent
        else position.hold_bars == config.exit_failed_followthrough_bars
    )
    if (
        config.exit_failed_followthrough_bars > 0
        and failed_followthrough_due
        and mfe_r <= config.exit_failed_followthrough_mfe_r
        and current_r <= config.exit_failed_followthrough_close_r
    ):
        return "failed_followthrough"
    shadow_failed_followthrough_due = (
        position.hold_bars >= config.exit_shadow_failed_followthrough_bars
        if config.exit_shadow_failed_followthrough_persistent
        else position.hold_bars == config.exit_shadow_failed_followthrough_bars
    )
    if (
        config.exit_shadow_failed_followthrough_bars > 0
        and not bool(position.metadata.get("frontier_initial_active", True))
        and shadow_failed_followthrough_due
        and mfe_r <= config.exit_shadow_failed_followthrough_mfe_r
        and current_r <= config.exit_shadow_failed_followthrough_close_r
    ):
        return "shadow_failed_followthrough"
    if config.exit_vwap_fail_bars > 0:
        proof_ok = config.exit_vwap_fail_after_mfe_r <= 0 or mfe_r >= config.exit_vwap_fail_after_mfe_r
        avwap = compute_session_vwap(symbol_state.bars_today) if proof_ok else 0.0
        if proof_ok and avwap > 0 and bar.close < avwap * (1.0 - max(config.exit_vwap_fail_pct, 0.0)):
            position.vwap_fail_streak += 1
        else:
            position.vwap_fail_streak = 0
        if position.vwap_fail_streak >= config.exit_vwap_fail_bars:
            return "vwap_fail"
    path_quality_reason = _path_quality_exit_reason(symbol_state, position, bar, config, current_r=current_r, mfe_r=mfe_r)
    if path_quality_reason:
        return path_quality_reason
    if (
        config.exit_mfe_giveback_enabled
        and position.hold_bars >= config.exit_mfe_giveback_min_hold_bars
        and mfe_r >= config.exit_mfe_giveback_start_r
        and current_r <= mfe_r - config.exit_mfe_giveback_gap_r
    ):
        return "mfe_giveback"
    if (
        config.exit_mfe_floor_enabled
        and position.hold_bars >= config.exit_mfe_floor_min_hold_bars
        and mfe_r >= config.exit_mfe_floor_start_r
        and current_r <= config.exit_mfe_floor_floor_r
        and _mfe_floor_cohort_matches(position, config)
    ):
        return "mfe_floor"
    if (
        config.exit_late_giveback_start_bars > 0
        and position.hold_bars >= config.exit_late_giveback_start_bars
        and mfe_r >= config.exit_late_giveback_start_r
        and current_r <= mfe_r - config.exit_late_giveback_gap_r
    ):
        return "late_mfe_giveback"
    if (
        config.exit_time_decay_bars > 0
        and position.hold_bars >= config.exit_time_decay_bars
        and mfe_r < config.exit_time_decay_min_mfe_r
        and current_r <= config.exit_time_decay_max_current_r
    ):
        return "time_decay_no_progress"
    if config.exit_max_hold_bars > 0 and position.hold_bars >= config.exit_max_hold_bars:
        return "max_hold"
    if should_quick_exit(position.hold_bars, current_r, config):
        return "quick_exit"
    if should_mfe_conviction_exit(position.hold_bars, mfe_r, current_r, config):
        return "mfe_conviction"
    if should_flow_reversal(
        bar,
        symbol_state.bars_today,
        entry_price=position.entry_price,
        hold_bars=position.hold_bars,
        mfe_r=mfe_r,
        config=config,
    ):
        return "flow_reversal"
    if bar.timestamp.time() >= config.flatten_time:
        carry_ok = current_r >= config.carry_min_r and close_location_value(bar) >= config.carry_min_cpr and position.regime_tier in {"A", "B"}
        if config.carry_mode == CarryMode.STRICT_LIVE and carry_ok:
            return ""
        return "eod_flatten" if config.carry_mode == CarryMode.OFF else "eod_flatten_carry_shadow"
    return ""


def _mfe_floor_cohort_matches(position: KALCBPositionState, config: KALCBConfig) -> bool:
    metadata = dict(position.metadata or {})
    frontier_rank = _metadata_int(metadata, "frontier_rank")
    if config.exit_mfe_floor_min_frontier_rank > 0 and frontier_rank < int(config.exit_mfe_floor_min_frontier_rank):
        return False
    if config.exit_mfe_floor_max_frontier_rank > 0 and frontier_rank > int(config.exit_mfe_floor_max_frontier_rank):
        return False

    if config.exit_mfe_floor_max_first30_signal_cpr > 0:
        if _metadata_float(metadata, "first30_signal_bar_cpr") > float(config.exit_mfe_floor_max_first30_signal_cpr):
            return False
    if config.exit_mfe_floor_max_first30_rel_volume > 0:
        if _metadata_float(metadata, "first30_rel_volume") > float(config.exit_mfe_floor_max_first30_rel_volume):
            return False
    if config.exit_mfe_floor_max_first30_low_vs_prev_close > -9.0:
        if _metadata_float(metadata, "first30_low_vs_prev_close") > float(config.exit_mfe_floor_max_first30_low_vs_prev_close):
            return False
    if config.exit_mfe_floor_max_first30_ret > -9.0:
        if _metadata_float(metadata, "first30_ret") > float(config.exit_mfe_floor_max_first30_ret):
            return False
    if config.exit_mfe_floor_max_first30_range_close_location > 0:
        if _metadata_float(metadata, "first30_range_close_location") > float(config.exit_mfe_floor_max_first30_range_close_location):
            return False

    entry_routes = set(config.exit_mfe_floor_entry_routes or ())
    if entry_routes and str(metadata.get("entry_route") or "") not in entry_routes:
        return False
    route_modes = set(config.exit_mfe_floor_entry_route_modes or ())
    if route_modes and str(metadata.get("entry_route_mode") or "") not in route_modes:
        return False
    return True


def _conditional_target_cohort_matches(position: KALCBPositionState, config: KALCBConfig) -> bool:
    metadata = dict(position.metadata or {})
    frontier_rank = _metadata_int(metadata, "frontier_rank")
    if config.exit_conditional_target_min_frontier_rank > 0 and frontier_rank < int(config.exit_conditional_target_min_frontier_rank):
        return False
    if config.exit_conditional_target_max_frontier_rank > 0 and frontier_rank > int(config.exit_conditional_target_max_frontier_rank):
        return False

    relvol = _metadata_float(metadata, "first30_rel_volume")
    if config.exit_conditional_target_min_first30_rel_volume > 0 and relvol < float(config.exit_conditional_target_min_first30_rel_volume):
        return False
    if config.exit_conditional_target_max_first30_rel_volume > 0 and relvol > float(config.exit_conditional_target_max_first30_rel_volume):
        return False

    cpr = _metadata_float(metadata, "first30_signal_bar_cpr")
    if config.exit_conditional_target_min_first30_signal_cpr > 0 and cpr < float(config.exit_conditional_target_min_first30_signal_cpr):
        return False
    if config.exit_conditional_target_max_first30_signal_cpr > 0 and cpr > float(config.exit_conditional_target_max_first30_signal_cpr):
        return False

    entry_routes = set(config.exit_conditional_target_entry_routes or ())
    if entry_routes and str(metadata.get("entry_route") or "") not in entry_routes:
        return False
    route_modes = set(config.exit_conditional_target_entry_route_modes or ())
    if route_modes and str(metadata.get("entry_route_mode") or "") not in route_modes:
        return False
    return True


def _path_quality_exit_reason(
    symbol_state: KALCBSymbolState,
    position: KALCBPositionState,
    bar: MarketBar,
    config: KALCBConfig,
    *,
    current_r: float,
    mfe_r: float,
) -> str:
    if not config.exit_path_quality_enabled:
        return ""
    if position.hold_bars < int(config.exit_path_quality_min_hold_bars or 0):
        return ""
    if config.exit_path_quality_max_hold_bars > 0 and position.hold_bars > int(config.exit_path_quality_max_hold_bars):
        return ""
    if mfe_r < float(config.exit_path_quality_min_mfe_r or 0.0):
        return ""
    giveback_r = max(float(mfe_r) - float(current_r), 0.0)
    if giveback_r < float(config.exit_path_quality_min_giveback_r or 0.0):
        return ""
    if not _path_quality_cohort_matches(position, config):
        return ""
    context = _path_quality_context(symbol_state, position, bar, current_r=current_r, mfe_r=mfe_r)
    min_gates, max_gates = config.exit_path_quality_min or {}, config.exit_path_quality_max or {}
    if not min_gates and not max_gates:
        return ""
    for key, threshold in min_gates.items():
        value = _optional_float(context.get(str(key)))
        if value is None or value < float(threshold):
            return ""
    for key, threshold in max_gates.items():
        value = _optional_float(context.get(str(key)))
        if value is None or value > float(threshold):
            return ""
    position.metadata["exit_path_quality_context"] = {
        key: value
        for key, value in context.items()
        if isinstance(value, (int, float, str, bool))
    }
    return "path_quality_exit"


def _path_quality_cohort_matches(position: KALCBPositionState, config: KALCBConfig) -> bool:
    metadata = dict(position.metadata or {})
    entry_routes = set(config.exit_path_quality_entry_routes or ())
    if entry_routes and str(metadata.get("entry_route") or "") not in entry_routes:
        return False
    route_modes = set(config.exit_path_quality_entry_route_modes or ())
    if route_modes and str(metadata.get("entry_route_mode") or "") not in route_modes:
        return False
    return True


def _path_quality_context(
    symbol_state: KALCBSymbolState,
    position: KALCBPositionState,
    bar: MarketBar,
    *,
    current_r: float,
    mfe_r: float,
) -> dict[str, float]:
    metadata = dict(position.metadata or {})
    bars_today = list(symbol_state.bars_today)
    post_entry = [item for item in bars_today if item.timestamp >= position.entry_time]
    if not post_entry and bars_today:
        post_entry = [bars_today[-1]]
    close = float(bar.close)
    risk = max(float(position.risk_per_share), 1e-9)
    session_vwap = compute_session_vwap(bars_today)
    or_high = float(position.or_high or symbol_state.or_high or 0.0)
    or_low = float(position.or_low or symbol_state.or_low or 0.0)
    or_width = max(or_high - or_low, 1e-9)
    or_mid = or_low + 0.5 * or_width
    max_close = max((float(item.close) for item in post_entry), default=close)
    max_high = max((float(item.high) for item in post_entry), default=float(bar.high))
    max_high_index = 0
    if post_entry:
        max_high_index = max(range(len(post_entry)), key=lambda index: float(post_entry[index].high))
    bars_since_mfe = max(0, len(post_entry) - 1 - max_high_index)
    recent3 = post_entry[-3:]
    recent6 = post_entry[-6:]

    expected_30m = _metadata_float(metadata, "first30_expected_30m_volume", 0.0)
    expected_5m = expected_30m / 6.0 if expected_30m > 0 else 0.0
    bar_rvol = float(bar.volume) / max(expected_5m, 1.0) if expected_5m > 0 else 0.0

    context: dict[str, float] = {
        "hold_bars": float(position.hold_bars),
        "current_r": float(current_r),
        "mfe_r": float(mfe_r),
        "mae_r": float(position.mae_r()),
        "giveback_r": max(float(mfe_r) - float(current_r), 0.0),
        "bars_since_mfe": float(bars_since_mfe),
        "close_location": float(close_location_value(bar)),
        "bar_ret": float(bar.close) / max(float(bar.open), 1e-9) - 1.0,
        "bar_rvol": bar_rvol,
        "vwap_ret": close / max(session_vwap, 1e-9) - 1.0 if session_vwap > 0 else 0.0,
        "entry_vwap_ret": close / max(float(position.avwap_at_entry or 0.0), 1e-9) - 1.0 if position.avwap_at_entry > 0 else 0.0,
        "or_position": (close - or_low) / or_width if or_high > 0 else 0.0,
        "or_high_ret": close / max(or_high, 1e-9) - 1.0 if or_high > 0 else 0.0,
        "or_mid_ret": close / max(or_mid, 1e-9) - 1.0 if or_mid > 0 else 0.0,
        "entry_price_ret": close / max(position.entry_price, 1e-9) - 1.0,
        "high_close_giveback_r": max(max_close - close, 0.0) / risk,
        "high_low_giveback_r": max(max_high - close, 0.0) / risk,
        "recent3_ret": close / max(float(recent3[0].open), 1e-9) - 1.0 if recent3 else 0.0,
        "recent6_ret": close / max(float(recent6[0].open), 1e-9) - 1.0 if recent6 else 0.0,
        "recent3_down_count": float(sum(1 for item in recent3 if float(item.close) < float(item.open))),
        "recent6_down_count": float(sum(1 for item in recent6 if float(item.close) < float(item.open))),
        "below_vwap_streak": float(_threshold_streak(post_entry, lambda item: session_vwap > 0 and float(item.close) < session_vwap)),
        "below_or_high_streak": float(_threshold_streak(post_entry, lambda item: or_high > 0 and float(item.close) < or_high)),
        "below_or_mid_streak": float(_threshold_streak(post_entry, lambda item: or_mid > 0 and float(item.close) < or_mid)),
        "above_or_high_streak": float(_threshold_streak(post_entry, lambda item: or_high > 0 and float(item.close) >= or_high)),
    }
    for key in (
        "frontier_rank",
        "candidate_rank",
        "frontier_selection_score",
        "flow_score",
        "accumulation_score",
        "momentum_score",
        "bar_rvol",
        "cpr",
        "first30_ret",
        "first30_vwap_ret",
        "first30_gap",
        "first30_rel_volume",
        "first30_gap_retention_ratio",
        "first30_gap_relvol",
        "first30_low_vs_prev_relvol",
        "first30_signal_bar_cpr",
        "first30_range_close_location",
        "first30_low_vs_prev_close",
        "first30_range_atr",
        "portfolio_drawdown_pct",
        "portfolio_session_return_pct",
        "session_sector_intraday_score_pct_mean",
        "session_sector_intraday_score_pct_median",
        "session_sector_intraday_ret_mean",
        "session_sector_intraday_ret_median",
        "session_sector_intraday_effective_count_mean",
        "session_sector_intraday_effective_count_median",
        "session_sector_intraday_positive_share",
        "session_sector_intraday_ret_positive_share",
        "session_sector_intraday_score_confirmed_share",
        "session_first30_gap_dispersion",
        "session_first30_rel_volume_mean",
        "session_first30_ret_mean",
        "session_first30_positive_share",
        *_CANDIDATE_CONTEXT_KEYS,
    ):
        value = _optional_float(metadata.get(key))
        if value is not None:
            context[f"entry_{key}"] = value
            context.setdefault(key, value)
    return context


def _threshold_streak(bars: list[MarketBar], predicate) -> int:
    streak = 0
    for item in reversed(bars):
        if not predicate(item):
            break
        streak += 1
    return streak


def _metadata_float(metadata: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metadata.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _metadata_int(metadata: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(metadata.get(key, default) or default)
    except (TypeError, ValueError):
        return int(default)


def _valid_protective_stop(target_stop: float, reference_price: float) -> float:
    rounded = round_price_for_krx(target_stop, "protective_stop")
    if reference_price <= 0:
        return rounded
    marketable_cap = round_price_for_krx(float(reference_price) * 0.999, "buy_limit")
    if marketable_cap <= 0:
        return rounded
    return min(rounded, marketable_cap)


def _pending_entry_count(state: KALCBState) -> int:
    return sum(
        1
        for symbol_state in state.symbols.values()
        if symbol_state.position is None and (symbol_state.pending_entry_order_id or symbol_state.stage == SymbolStage.ENTRY_QUEUED)
    )


def _pending_sector_count(state: KALCBState, sector: str) -> int:
    if not _known_sector(sector):
        return 0
    return sum(
        1
        for symbol_state in state.symbols.values()
        if symbol_state.position is None
        and (symbol_state.pending_entry_order_id or symbol_state.stage == SymbolStage.ENTRY_QUEUED)
        and symbol_state.candidate is not None
        and symbol_state.candidate.sector == sector
    )


def _entry_route_session_count(state: KALCBState, route_name: str) -> int:
    counts = state.meta.get("entry_route_session_counts")
    if not isinstance(counts, dict):
        return 0
    try:
        return int(counts.get(str(route_name), 0) or 0)
    except (TypeError, ValueError):
        return 0


def _increment_entry_route_session_count(state: KALCBState, route_name: str) -> None:
    route = str(route_name or "")
    if not route:
        return
    counts = state.meta.get("entry_route_session_counts")
    if not isinstance(counts, dict):
        counts = {}
        state.meta["entry_route_session_counts"] = counts
    counts[route] = _entry_route_session_count(state, route) + 1


def _decrement_entry_route_session_count(state: KALCBState, route_name: str) -> None:
    route = str(route_name or "")
    if not route:
        return
    counts = state.meta.get("entry_route_session_counts")
    if not isinstance(counts, dict):
        return
    counts[route] = max(_entry_route_session_count(state, route) - 1, 0)


def _known_sector(sector: str) -> bool:
    normalized = str(sector or "").upper()
    return normalized not in {"", "UNKNOWN", "NONE", "N/A"}


def _entry_quality_votes(
    entry_meta: dict[str, Any],
    source_meta: dict[str, Any],
    config: KALCBConfig,
) -> tuple[int, int, list[dict[str, Any]]] | None:
    required = int(config.entry_plan_min_quality_votes or 0)
    if required <= 0:
        return None
    gates: list[dict[str, Any]] = []

    def add_min(name: str, threshold: float, actual: Any) -> None:
        if threshold <= -9.0:
            return
        value = float(actual or 0.0)
        gates.append(_gate(f"entry_quality_{name}", threshold, value, value >= float(threshold)))

    add_min("bar_ret", config.entry_plan_quality_min_bar_ret, entry_meta.get("first30_ret"))
    add_min(
        "first30_signal_cpr",
        config.entry_plan_quality_min_first30_signal_cpr,
        entry_meta.get("first30_signal_bar_cpr"),
    )
    add_min(
        "first30_rel_volume",
        config.entry_plan_quality_min_first30_rel_volume,
        entry_meta.get("first30_rel_volume"),
    )
    add_min(
        "first30_range_atr_min",
        config.entry_plan_quality_min_first30_range_atr,
        entry_meta.get("first30_range_atr"),
    )
    if config.entry_plan_quality_max_first30_range_atr > 0:
        actual_range = float(entry_meta.get("first30_range_atr") or 0.0)
        gates.append(
            _gate(
                "entry_quality_first30_range_atr_max",
                config.entry_plan_quality_max_first30_range_atr,
                actual_range,
                actual_range <= float(config.entry_plan_quality_max_first30_range_atr),
            )
        )
    add_min("flow_score", config.entry_plan_quality_min_flow_score, source_meta.get("flow_score"))
    add_min("accumulation_score", config.entry_plan_quality_min_accumulation_score, source_meta.get("accumulation_score"))
    if config.entry_plan_quality_max_frontier_rank > 0:
        frontier_rank = int(source_meta.get("frontier_rank") or 0)
        gates.append(
            _gate(
                "entry_quality_frontier_rank",
                int(config.entry_plan_quality_max_frontier_rank),
                frontier_rank,
                0 < frontier_rank <= int(config.entry_plan_quality_max_frontier_rank),
            )
        )
    vote_count = sum(1 for gate in gates if gate["passed"])
    gates.append(_gate("entry_quality_votes", required, vote_count, vote_count >= required))
    return vote_count, required, gates


def _entry_context_gate_reject_reason(config: KALCBConfig, metadata: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    gates: list[dict[str, Any]] = []
    for key, threshold in (config.entry_plan_route_context_min or {}).items():
        value = _optional_float(metadata.get(str(key)))
        passed = value is not None and value >= float(threshold)
        gates.append(_gate(f"entry_context_min:{key}", float(threshold), value, passed))
        if not passed:
            return f"entry_context_min:{key}", gates
    for key, threshold in (config.entry_plan_route_context_max or {}).items():
        value = _optional_float(metadata.get(str(key)))
        passed = value is not None and value <= float(threshold)
        gates.append(_gate(f"entry_context_max:{key}", float(threshold), value, passed))
        if not passed:
            return f"entry_context_max:{key}", gates
    for key, denied_values in (config.entry_plan_route_context_exclude or {}).items():
        value = metadata.get(str(key))
        denied = {str(item) for item in denied_values}
        passed = value is not None and str(value) not in denied
        gates.append(_gate(f"entry_context_exclude:{key}", sorted(denied), value, passed))
        if not passed:
            return f"entry_context_exclude:{key}", gates
    return "", gates


def _breakout_level(entry_type: EntryType, or_high: float, prior_day_high: float) -> float:
    if entry_type == EntryType.PDH_BREAKOUT:
        return prior_day_high
    if entry_type == EntryType.COMBINED_BREAKOUT:
        return max(or_high, prior_day_high)
    return or_high


def _gate(name: str, threshold: Any, actual: Any, passed: bool, *, applicable: bool = True) -> dict[str, Any]:
    return {
        "filter_name": name,
        "threshold": threshold,
        "actual_value": actual,
        "passed": bool(passed),
        "applicable": bool(applicable),
    }


def _rejected(
    state: KALCBState,
    bar: MarketBar,
    reason: str,
    gates: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> KALCBCoreResult:
    symbol_state = state.symbol_state(bar.symbol)
    symbol_state.rejected_reason = reason
    # Fast sweeps still mutate state/reason but skip bulky per-bar rejection events.
    if state.meta.get("fast_replay_suppress_rejections"):
        return KALCBCoreResult(state)
    payload = {**dict(metadata or {}), "filter_decisions": gates, "core_version": KALCB_CORE_VERSION}
    return KALCBCoreResult(state, decisions=[_decision(bar, "entry_rejected", reason, metadata=payload)])


def _entry_metadata(entry_type: EntryType, momentum_score: int, score_detail: dict[str, int], bar_rvol: float, cpr: float, avwap: float) -> dict[str, Any]:
    return {
        "entry_type": entry_type.value,
        "momentum_score": int(momentum_score),
        "score_detail": dict(score_detail),
        "bar_rvol": float(bar_rvol),
        "cpr": float(cpr),
        "avwap": float(avwap),
    }


def _bar_blocks_entry(bar: MarketBar) -> bool:
    metadata = dict(bar.metadata)
    return bool(
        metadata.get("halted")
        or metadata.get("is_halted")
        or metadata.get("managed_issue")
        or metadata.get("untradable")
        or metadata.get("vi_active")
    )


def _decision(
    bar: MarketBar,
    code: str,
    reason: str,
    *,
    actions: list[StrategyAction] | tuple[StrategyAction, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> DecisionEvent:
    return DecisionEvent(
        timestamp=bar.timestamp,
        strategy_id=STRATEGY_ID,
        symbol=bar.symbol,
        decision_code=code,
        reason=reason,
        actions=tuple(actions),
        metadata={"core_version": KALCB_CORE_VERSION, **dict(metadata or {})},
    )


def _exit_metadata(position: KALCBPositionState, cohort: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    entry_metadata = {
        key: position.metadata[key]
        for key in (
            "entry_route",
            "entry_route_mode",
            "entry_route_priority",
            "entry_route_attempts",
            "entry_route_risk_mult",
            "entry_route_notional_mult",
            "entry_route_participation_mult",
            "entry_route_max_session_trades",
            "entry_route_context_min_keys",
            "entry_route_context_max_keys",
            "entry_route_context_exclude_keys",
            "entry_route_session_count_before",
            "frontier_rank",
            "frontier_initial_active",
            "first30_gap",
            "first30_rel_volume",
            "first30_gap_retention_ratio",
            "first30_gap_relvol",
            "first30_low_vs_prev_relvol",
            "first30_signal_bar_cpr",
            "first30_low_vs_prev_close",
            "first30_ret",
            "first30_range_close_location",
            "entry_path_anchor_time",
            "entry_path_anchor_price",
            "entry_path_risk_per_share",
            "entry_path_stop_pct",
            "entry_path_completed_bars",
            "entry_path_reference",
            "h3_current_r",
            "h3_mfe_r",
            "h3_mae_r",
            "h3_giveback_r",
            "h6_current_r",
            "h6_mfe_r",
            "h6_mae_r",
            "h6_giveback_r",
            "effective_max_position_notional_pct",
            "portfolio_drawdown_pct",
            "portfolio_session_return_pct",
            "exit_path_quality_context",
        )
        if key in position.metadata
    }
    return {
        "entry_type": position.entry_type,
        "sector": position.sector,
        "regime_tier": position.regime_tier,
        "momentum_score": position.momentum_score,
        "risk_per_share": position.risk_per_share,
        "entry_price": position.entry_price,
        "initial_stop": position.initial_stop,
        "current_stop": position.current_stop,
        "mfe_r": position.mfe_r(),
        "mae_r": position.mae_r(),
        "hold_bars": position.hold_bars,
        "partial_taken": position.partial_taken,
        "exit_cohort": cohort,
        "core_version": KALCB_CORE_VERSION,
        **entry_metadata,
        **dict(extra or {}),
    }


def _action_role_metadata(action: StrategyAction) -> dict[str, Any]:
    if isinstance(action, SubmitEntry):
        return {"order_role": "ENTRY", **dict(action.metadata), "stop_price": action.stop_price}
    if isinstance(action, SubmitPartialExit):
        return {"order_role": "TP", **dict(action.metadata)}
    if isinstance(action, SubmitProtectiveStop):
        return {"order_role": "STOP", **dict(action.metadata), "stop_price": action.stop_price}
    if isinstance(action, ReplaceProtectiveStop):
        return {"order_role": "STOP", **dict(action.metadata), "stop_price": action.stop_price}
    if isinstance(action, (SubmitExit, FlattenPosition)):
        return {"order_role": "EXIT", **dict(action.metadata)}
    return dict(getattr(action, "metadata", {}) or {})
