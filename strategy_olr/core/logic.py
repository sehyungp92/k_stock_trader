from __future__ import annotations

from strategy_common.fill_lifecycle import record_fill_delta, order_fill_complete, is_duplicate_fill, entry_order_key
from copy import deepcopy

from dataclasses import asdict, replace
from strategy_common.execution_session import session_cutoff
from datetime import date, datetime, time, timedelta
from math import floor
from typing import Any, Iterable

from strategy_common.actions import (
    CancelOrders,
    ReplaceProtectiveStop,
    StrategyAction,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
    SubmitProtectiveStop,
)
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar

from strategy_olr.config import OLRConfig, OLR_CORE_VERSION, STRATEGY_ID
from strategy_olr.execution import (
    DECISION_CUTOFF,
    OLREntryPlan,
    OLRExitPlan,
    _completed_bar_exit_reason,
    _next_bar_stop_from_completed_bar,
    _running_vwap,
    find_olr_entry_signal,
)
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.risk import round_price_for_krx

from .core_models import OLRCoreResult, OLRExpiredOrderEvent, OLRFillEvent, OLROrderUpdateEvent, OLRPortfolioView
from .state import OLRPositionState, OLRState, OLRSymbolStage
from .capacity import observe_routed as observe_capacity, forecast as capacity_forecast, enabled as capacity_enabled

_SNAPSHOT_BY_SYMBOL_CACHE: dict[str, dict[str, OLRDailyCandidate]] = {}


def step_olr_core(
    state: OLRState,
    bar: MarketBar,
    config: OLRConfig,
    candidate_snapshot: OLRDailySnapshot | None,
    portfolio: OLRPortfolioView,
) -> OLRCoreResult:
    observe_capacity(state, bar, config, candidate_snapshot)
    config = replace(config, auction_fill_time=session_cutoff(bar.timestamp.date(), _parse_clock(config.auction_fill_time)).strftime("%H:%M"))
    _sync_snapshot_state(state, candidate_snapshot)
    symbol_state = state.symbol_state(bar.symbol)
    if symbol_state.position is not None:
        if bar.timestamp > symbol_state.position.entry_time or (
                bar.timestamp == symbol_state.position.entry_time
                and symbol_state.position.metadata.get("execution_phase") == "open"):
            symbol_state.position.update_mark(bar.high, bar.low)
        _remember_session_bar(symbol_state, bar)
    actions: list[StrategyAction] = []
    decisions: list[DecisionEvent] = []

    if candidate_snapshot is not None and bar.timestamp.date() == candidate_snapshot.trade_date:
        candidate = _snapshot_by_symbol(candidate_snapshot).get(str(bar.symbol).zfill(6))
        if candidate is not None and symbol_state.session_date != candidate_snapshot.trade_date:
            symbol_state.reset_for_session(candidate_snapshot.trade_date, candidate)
        _remember_session_bar(symbol_state, bar)
        entry_result = _maybe_submit_entry(state, bar, config, candidate_snapshot, candidate, portfolio)
        actions.extend(entry_result.actions)
        decisions.extend(entry_result.decisions)

    managed_result = _maybe_submit_managed_exit(state, bar, config)
    actions.extend(managed_result.actions)
    decisions.extend(managed_result.decisions)

    if config.capacity_stale_release and not managed_result.actions:
        from .stale_release import maybe_release_stale_inventory
        stale = maybe_release_stale_inventory(state, bar, config)
        if stale.actions:
            actions.extend(stale.actions)
            decisions.extend(stale.decisions)
            return OLRCoreResult(state=state, actions=actions, decisions=decisions)

    if not any(isinstance(action, (SubmitExit, SubmitPartialExit)) for action in managed_result.actions):
        exit_result = _maybe_submit_next_close_exit(state, bar, config)
        actions.extend(exit_result.actions)
        decisions.extend(exit_result.decisions)
    return OLRCoreResult(state=state, actions=actions, decisions=decisions)


def on_olr_fill(state: OLRState, fill: OLRFillEvent, config: OLRConfig) -> OLRCoreResult:
    if is_duplicate_fill(state.meta, fill):
        return OLRCoreResult(state=state)
    symbol_state = state.symbol_state(fill.symbol)
    existing = symbol_state.position
    if existing is None and fill.side.upper() == "BUY":
        existing = deepcopy(symbol_state.entry_history.get(entry_order_key(fill)))
    if existing is not None:
        if fill.side.upper() == "BUY" and existing.entry_order_id != fill.order_id:
            raise ValueError("Unrelated entry order cannot merge into an open trade")
        if fill.side.upper() == "SELL" and fill.qty > existing.qty_open:
            raise ValueError("Sell fill exceeds the managed position")
    if existing is None and fill.side.upper() == "SELL":
        raise ValueError("Sell execution requires a reconciled open lifecycle")
    if not record_fill_delta(state.meta, fill):
        return OLRCoreResult(state=state)
    symbol_state = state.symbol_state(fill.symbol)
    if existing is not None and fill.side.upper() == "BUY":
        symbol_state.position = existing
    role = state.order_roles.get(str(fill.order_id), state.order_history.get(str(fill.order_id), {}))
    fill_side = str(fill.side or "").upper().strip()
    matching_pending = symbol_state.pending_entry_order_id in {"", fill.order_id}
    pending_metadata = dict(symbol_state.pending_entry_metadata if fill_side == "BUY" and matching_pending
                            else symbol_state.pending_exit_metadata if fill_side != "BUY" else {})
    metadata = {**(existing.metadata if existing and fill_side == "BUY" else {}), **dict(role), **pending_metadata, **dict(fill.metadata or {})}
    complete = order_fill_complete(fill, metadata)
    if complete:
        state.order_roles.pop(str(fill.order_id), None)
    actions: list[StrategyAction] = []
    decisions: list[DecisionEvent] = []
    if fill_side == "BUY":
        candidate_rank = int(metadata.get("candidate_rank", 0) or 0)
        candidate_score = float(metadata.get("candidate_score", 0.0) or 0.0)
        position = symbol_state.position
        additional_fragment = position is not None
        if additional_fragment:
            if position.entry_order_id != fill.order_id:
                raise ValueError("Unrelated entry order cannot merge into an open OLR trade")
            position.entry_price = (position.entry_price * position.qty_open + fill.price * fill.qty) / (position.qty_open + fill.qty)
            position.qty_open += int(fill.qty)
            position.metadata["acquired_qty"] = int(position.metadata.get("acquired_qty", position.qty_open - fill.qty)) + int(fill.qty)
            position.metadata["entry_fees"] = float(position.metadata.get("entry_fees", 0)) + float(fill.metadata.get("fee") or 0)
            position.metadata["entry_fees_complete"] = bool(position.metadata.get("entry_fees_complete")) and fill.metadata.get("fee") is not None
        else:
            symbol_state.position = OLRPositionState(
                symbol=str(fill.symbol).zfill(6),
                qty_open=int(fill.qty),
                entry_price=float(fill.price),
                entry_time=fill.timestamp,
                candidate_rank=candidate_rank,
                candidate_score=candidate_score,
                source_artifact_hash=str(metadata.get("source_artifact_hash", "")),
                sector=str(metadata.get("sector", "UNKNOWN")),
                entry_order_id=fill.order_id,
                metadata={**metadata, "entry_fees": float(fill.metadata.get("fee") or 0),
                    "entry_fees_complete": fill.metadata.get("fee") is not None},
            )
        symbol_state.position.metadata.setdefault("entry_lifecycle_key", entry_order_key(fill))
        symbol_state.position.metadata.setdefault("acquired_qty", int(fill.qty))
        if config.capacity_session_budget_fraction:
            # Sum native cash debits. Average open-lot cost is insufficient
            # when another entry fragment arrives after a partial sale.
            fee=fill.metadata.get('fee')
            if fee is None:
                fee=fill.qty*fill.price*config.commission_bps/10000
            pmeta=symbol_state.position.metadata
            pmeta['capacity_acquired_cost']=float(pmeta.get('capacity_acquired_cost',0))+fill.qty*fill.price+float(fee)
        if config.capacity_value_mode!='off':
            from .allocation_value import record_cash
            record_cash(symbol_state.position,fill)
        if matching_pending and symbol_state.pending_entry_order_id:
            # Persist executed entry quantity independently of current inventory:
            # selling a partial position must not resurrect its original reserve.
            symbol_state.pending_entry_metadata["entry_filled_qty"] = int(symbol_state.position.metadata["acquired_qty"])
        symbol_state.stage = OLRSymbolStage.IN_POSITION
        if complete and matching_pending:
            symbol_state.pending_entry_order_id = ""
            symbol_state.pending_entry_metadata.clear()
        stop_price = float(metadata.get("protective_stop_price", 0.0) or 0.0)
        exit_plan = _exit_plan_from_payload(metadata.get("trade_exit_plan"))
        if stop_price >= float(fill.price):
            stop_price = float(fill.price) * (1.0 - max(float(exit_plan.stop_pct), 0.003))
            symbol_state.position.metadata["protective_stop_price"] = stop_price
            symbol_state.position.metadata["risk_per_share"] = max(float(fill.price) - stop_price, float(fill.price) * 0.001, 1.0)
        if additional_fragment:
            stop_price = float(symbol_state.position.metadata.get("current_stop_price", stop_price))
        symbol_state.position.metadata["current_stop_price"] = stop_price
        if exit_plan.mode == "managed" and exit_plan.hard_stop_enabled and stop_price > 0.0:
            actions.append(
                (ReplaceProtectiveStop if additional_fragment else SubmitProtectiveStop)(
                    strategy_id=STRATEGY_ID,
                    symbol=fill.symbol,
                    qty=int(symbol_state.position.qty_open),
                    stop_price=stop_price,
                    reason="managed_hard_stop",
                    metadata={
                        "strategy_core_version": OLR_CORE_VERSION,
                        "source_artifact_hash": metadata.get("source_artifact_hash", ""),
                        "candidate_rank": candidate_rank,
                        "candidate_score": candidate_score,
                        "trade_exit_plan": asdict(exit_plan),
                        "order_role": "STOP",
                    },
                )
            )
        if exit_plan.mode == "managed":
            actions.extend(_managed_resting_exit_actions(fill, symbol_state.position, exit_plan))
        decisions.append(
            _decision(fill.timestamp, fill.symbol, "ENTRY_FILLED", fill.reason or "entry_filled", metadata)
        )
    elif fill_side == "SELL":
        position = symbol_state.position
        if position is not None:
            if config.capacity_value_mode!='off':
                from .allocation_value import record_cash
                record_cash(position,fill)
            if int(fill.qty) > position.qty_open:
                raise ValueError("OLR sell fill exceeds managed quantity")
            position.qty_open -= int(fill.qty)
            position.exit_order_id = fill.order_id
        if complete and symbol_state.pending_exit_order_id == str(fill.order_id):
            symbol_state.pending_exit_order_id = ""
            symbol_state.pending_exit_metadata.clear()
        if position is not None:
            leg = str(metadata.get("order_role") or "").upper()
            executed = position.metadata.setdefault("target_executed", {})
            executed[leg] = int(executed.get(leg, 0)) + int(fill.qty)
            position.metadata["sold_qty"] = int(position.metadata.get("sold_qty", 0)) + int(fill.qty)
            per_order = position.metadata.setdefault("target_order_filled", {})
            per_order[str(fill.order_id)] = int(per_order.get(str(fill.order_id), 0)) + int(fill.qty)
        if position is None or position.qty_open <= 0:
            state.meta.get('priority_exit_pending', {}).pop(str(fill.symbol).zfill(6), None)
            if position is not None:
                symbol_state.entry_history[position.metadata.get("entry_lifecycle_key", position.entry_order_id)] = deepcopy(position)
            symbol_state.position = None
            symbol_state.stage = OLRSymbolStage.DONE
            symbol_state.pending_exit_order_id = ""
            symbol_state.pending_exit_metadata.clear()
            actions.append(CancelOrders(STRATEGY_ID, fill.symbol, "position_closed_cancel_resting_orders"))
        else:
            leg = str(metadata.get("order_role") or "").upper()
            if leg == "PARTIAL_TARGET":
                actions.extend(_managed_partial_fill_actions(fill, position))
            plan = _exit_plan_from_payload(position.metadata.get("trade_exit_plan"))
            if plan.hard_stop_enabled and not any(isinstance(action, ReplaceProtectiveStop) for action in actions):
                stop = float(position.metadata.get("current_stop_price", 0) or 0)
                if stop > 0:
                    actions.append(ReplaceProtectiveStop(STRATEGY_ID, fill.symbol, stop, qty=position.qty_open,
                        reason="fill_quantity_resize", metadata={"order_role": "STOP"}))
        decisions.append(
            _decision(fill.timestamp, fill.symbol, "EXIT_FILLED", fill.reason or "exit_filled", metadata)
        )
    return OLRCoreResult(state=state, actions=actions, decisions=decisions)


def on_olr_timer(state: OLRState, timestamp: datetime, config: OLRConfig) -> OLRCoreResult:
    decisions = []
    rotations=state.meta.get('capacity_rotation_pending',{})
    for symbol in list(rotations):
        if symbol not in state.symbols or state.symbols[symbol].position is None:
            rotations.pop(symbol)
    pending = state.meta.get('capacity_deferred_intents', {})
    for symbol, row in list(pending.items()):
        if timestamp > datetime.fromisoformat(row['end']):
            pending.pop(symbol)
            decisions.append(_decision(timestamp, symbol, 'ENTRY_SKIPPED', 'deferred_signal_expired',
                {'expires_at':row['end']}))
    return OLRCoreResult(state=state, decisions=decisions)


def on_olr_order_expired(state: OLRState, expired: OLRExpiredOrderEvent, config: OLRConfig) -> OLRCoreResult:
    symbol = str(expired.symbol).zfill(6)
    symbol_state = state.symbol_state(symbol)
    replacement = state.meta.get('priority_exit_pending', {}).get(str(expired.symbol).zfill(6))
    if replacement and replacement['cancel_order_id'] == str(expired.order_id):
        state.meta['priority_exit_pending'].pop(str(expired.symbol).zfill(6))
    role = state.order_roles.pop(str(expired.order_id), {})
    if role:
        state.order_history[str(expired.order_id)] = dict(role)
    metadata = dict(expired.metadata or {})
    actions: list[StrategyAction] = []
    decisions: list[DecisionEvent] = []
    if role.get("role") == "entry" and symbol_state.pending_entry_order_id == str(expired.order_id):
        symbol_state.pending_entry_order_id = ""
        symbol_state.pending_entry_metadata.clear()
        if symbol_state.position is None:
            symbol_state.stage = OLRSymbolStage.WATCHING
        decisions.append(_decision(expired.timestamp, symbol, "ENTRY_EXPIRED", expired.reason or "entry_expired", metadata))
    elif role.get("role") == "exit" and symbol_state.pending_exit_order_id == str(expired.order_id):
        pending_metadata = dict(symbol_state.pending_exit_metadata or {})
        symbol_state.pending_exit_order_id = ""
        symbol_state.pending_exit_metadata.clear()
        if symbol_state.position is not None and symbol_state.position.qty_open > 0:
            symbol_state.stage = OLRSymbolStage.IN_POSITION
            fallback_metadata = {
                **pending_metadata,
                **metadata,
                "strategy_core_version": OLR_CORE_VERSION,
                "auction_exit_nonfill_fallback": True,
                "expired_order_id": str(expired.order_id),
                "expired_order_type": expired.order_type,
                "expired_order_reason": expired.reason,
            }
            fallback_metadata.pop("expiry_ts", None)
            fallback_metadata.pop("expiry_timestamp", None)
            actions.append(
                SubmitExit(
                    strategy_id=STRATEGY_ID,
                    symbol=symbol,
                    qty=int(symbol_state.position.qty_open),
                    order_type="MARKET",
                    limit_price=None,
                    reason="auction_exit_nonfill_market_fallback",
                    metadata=fallback_metadata,
                )
            )
            decisions.append(
                _decision(
                    expired.timestamp,
                    symbol,
                    "EXIT_FALLBACK_SUBMITTED",
                    "auction_exit_nonfill_market_fallback",
                    {**fallback_metadata, "qty": int(symbol_state.position.qty_open)},
                )
            )
        else:
            symbol_state.stage = OLRSymbolStage.DONE
        decisions.append(_decision(expired.timestamp, symbol, "EXIT_EXPIRED", expired.reason or "exit_expired", metadata))
    elif role.get('role') == 'managed_partial_exit' and role.get('reason') == 'weak_followthrough_partial' and symbol_state.pending_exit_order_id == str(expired.order_id):
        symbol_state.pending_exit_order_id = ''
        symbol_state.pending_exit_metadata.clear()
        symbol_state.stage = OLRSymbolStage.IN_POSITION if symbol_state.position else OLRSymbolStage.DONE
    return OLRCoreResult(state=state, actions=actions, decisions=decisions)


def on_olr_order_update(state: OLRState, update: OLROrderUpdateEvent, config: OLRConfig) -> OLRCoreResult:
    symbol = str(update.symbol).zfill(6)
    symbol_state = state.symbol_state(symbol)
    status = str(update.status or "").upper().strip()
    terminal = status in {"BLOCKED", "REJECTED", "CANCELLED", "DEFERRED", "EXPIRED"}
    if terminal and str(update.order_id) in state.order_roles:
        state.order_history[str(update.order_id)] = state.order_roles.pop(str(update.order_id))
    role = dict(state.order_roles.get(str(update.order_id), state.order_history.get(str(update.order_id), {})))
    metadata = {**dict(role), **dict(update.metadata or {})}
    if terminal and metadata.get("order_role") in {"TARGET", "PARTIAL_TARGET"}:
        position = symbol_state.position
        lifecycle_key = metadata.get("entry_lifecycle_key")
        if lifecycle_key and (position is None or position.metadata.get("entry_lifecycle_key") != lifecycle_key):
            position = symbol_state.entry_history.get(lifecycle_key)
        if position is not None:
            released = position.metadata.setdefault("target_terminal_orders", {})
            if str(update.order_id) not in released:
                leg = metadata["order_role"]
                filled = int(position.metadata.get("target_order_filled", {}).get(str(update.order_id), 0))
                unfilled = max(int(metadata.get("submitted_qty", 0)) - filled, 0)
                submitted = position.metadata.setdefault("target_submitted", {})
                submitted[leg] = max(int(submitted.get(leg, 0)) - unfilled, 0)
                released[str(update.order_id)] = status
    if terminal and symbol_state.pending_entry_order_id == str(update.order_id):
        symbol_state.pending_entry_order_id = ""
        symbol_state.pending_entry_metadata.clear()
        if symbol_state.position is None:
            symbol_state.stage = OLRSymbolStage.WATCHING
    if terminal and _is_retryable_entry_update(update, status, metadata) and symbol_state.position is None:
        symbol_state.entry_attempted = False
        symbol_state.stage = OLRSymbolStage.WATCHING
    if terminal and symbol_state.pending_exit_order_id == str(update.order_id):
        symbol_state.pending_exit_order_id = ""
        symbol_state.pending_exit_metadata.clear()
        if symbol_state.position is not None:
            symbol_state.stage = OLRSymbolStage.IN_POSITION
        else:
            symbol_state.stage = OLRSymbolStage.DONE
    replacement = state.meta.get('priority_exit_pending', {}).get(symbol)
    if (replacement and replacement['cancel_order_id'] == str(update.order_id)
            and status == 'CANCELLED'):
        # A cancellation request does not authorize a second sell. Only the
        # matching terminal acknowledgement releases the remaining inventory.
        state.meta['priority_exit_pending'].pop(symbol)
        position = symbol_state.position
        if position is not None and position.qty_open > 0 and not symbol_state.pending_exit_order_id:
            return _managed_market_exit_at(state, symbol_state, update.timestamp, position,
                _exit_plan(config), replacement['reason'], _position_risk(position))
    return OLRCoreResult(
        state=state,
        decisions=[
            _decision(
                update.timestamp,
                symbol,
                "ORDER_UPDATE",
                (update.reason or status.lower() or "order_update"),
                {
                    "order_id": update.order_id,
                    "status": status,
                    "role": role.get("role", ""),
                    **metadata,
                },
            )
        ],
    )


def _is_retryable_entry_update(update: OLROrderUpdateEvent, status: str, metadata: dict[str, Any]) -> bool:
    side = str(update.side or metadata.get("side") or "").upper().strip()
    if side and side != "BUY":
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


def remember_submitted_order(state: OLRState, order_id: str | None, action: StrategyAction) -> None:
    if not order_id:
        return
    if isinstance(action, SubmitEntry):
        symbol_state = state.symbol_state(action.symbol)
        symbol_state.pending_entry_order_id = str(order_id)
        metadata = dict(action.metadata)
        reserved = _metadata_float(metadata, "reserved_notional")
        if reserved <= 0.0:
            estimated_price = max(_metadata_float(metadata, "estimated_entry_price"), 0.0)
            reserved = int(action.qty) * estimated_price * (1.0 + max(float(metadata.get("entry_cost_buffer_pct", 0.0) or 0.0), 0.0))
            metadata["reserved_notional"] = reserved
        metadata["submitted_qty"] = int(action.qty)
        symbol_state.pending_entry_metadata = metadata
        symbol_state.stage = OLRSymbolStage.ENTRY_QUEUED
        state.order_roles[str(order_id)] = {**metadata, "role": "entry", "symbol": action.symbol, "reason": action.reason}
    elif isinstance(action, SubmitExit):
        symbol_state = state.symbol_state(action.symbol)
        metadata = {**dict(action.metadata), "submitted_qty": action.qty, "order_type": action.order_type}
        order_role = str(metadata.get("order_role") or "").upper()
        state.order_roles[str(order_id)] = {
            **metadata,
            "role": "managed_exit_leg" if order_role == "TARGET" and action.order_type == "LIMIT" else "exit",
            "symbol": action.symbol,
            "reason": action.reason,
        }
        if order_role == "TARGET" and action.order_type == "LIMIT":
            return
        symbol_state.pending_exit_order_id = str(order_id)
        symbol_state.pending_exit_metadata = metadata
        symbol_state.stage = OLRSymbolStage.EXIT_QUEUED
    elif isinstance(action, SubmitPartialExit):
        metadata = {**dict(action.metadata), "submitted_qty": action.qty}
        state.order_roles[str(order_id)] = {**metadata, "role": "managed_partial_exit", "symbol": action.symbol, "reason": action.reason}
        if action.order_type in {"MARKET", "CLOSE_AUCTION"} and str(metadata.get("order_role") or "").upper() not in {"PARTIAL_TARGET"}:
            symbol_state = state.symbol_state(action.symbol)
            symbol_state.pending_exit_order_id = str(order_id)
            symbol_state.pending_exit_metadata = metadata
            symbol_state.stage = OLRSymbolStage.EXIT_QUEUED
    elif isinstance(action, (SubmitProtectiveStop, ReplaceProtectiveStop)):
        metadata = dict(action.metadata)
        state.order_roles[str(order_id)] = {
            **metadata,
            "role": "stop",
            "symbol": action.symbol,
            "reason": action.reason,
            "order_role": metadata.get("order_role", "STOP"),
        }


def _sync_snapshot_state(state: OLRState, snapshot: OLRDailySnapshot | None) -> None:
    if snapshot is None:
        return
    if state.snapshot_hash == snapshot.artifact_hash:
        return
    state.snapshot_hash = snapshot.artifact_hash
    state.source_fingerprint = snapshot.source_fingerprint
    state.session_date = snapshot.trade_date
    for candidate in snapshot.candidates:
        symbol_state = state.symbol_state(candidate.symbol)
        if symbol_state.position is None:
            symbol_state.reset_for_session(snapshot.trade_date, candidate)


def _snapshot_by_symbol(snapshot: OLRDailySnapshot) -> dict[str, OLRDailyCandidate]:
    key = snapshot.artifact_hash
    cached = _SNAPSHOT_BY_SYMBOL_CACHE.get(key)
    if cached is not None:
        return cached
    by_symbol = snapshot.by_symbol()
    if len(_SNAPSHOT_BY_SYMBOL_CACHE) > 2048:
        _SNAPSHOT_BY_SYMBOL_CACHE.clear()
    _SNAPSHOT_BY_SYMBOL_CACHE[key] = by_symbol
    return by_symbol


def _maybe_submit_entry(
    state: OLRState,
    bar: MarketBar,
    config: OLRConfig,
    snapshot: OLRDailySnapshot,
    candidate: OLRDailyCandidate | None,
    portfolio: OLRPortfolioView,
) -> OLRCoreResult:
    symbol = str(bar.symbol).zfill(6)
    symbol_state = state.symbol_state(symbol)
    if candidate is None or not candidate.tradable:
        return OLRCoreResult(state=state)
    if state.meta.get('observe_structural_signals_v1'):
        from .signal_episodes import observe
        observe(state,bar,candidate,snapshot,config,_entry_plan(config),portfolio)
    if symbol_state.entry_attempted or symbol_state.pending_entry_order_id or symbol_state.position is not None:
        return OLRCoreResult(state=state)
    if _bar_time(bar) < session_cutoff(bar.timestamp.date(), DECISION_CUTOFF):
        return OLRCoreResult(state=state)
    selected = tuple(c for c in snapshot.candidates[: int(config.overnight_slot_count)] if c.tradable)
    if len(selected) < int(config.min_selected) or candidate.symbol not in {item.symbol for item in selected}:
        return OLRCoreResult(state=state)
    weights = _candidate_weights(selected, config)
    weight = weights.get(candidate.symbol, 0.0)
    equity = max(float(portfolio.equity or 0.0), float(portfolio.cash or 0.0), 0.0)
    if config.allocation_equity_cap_krw > 0:
        equity = min(equity, config.allocation_equity_cap_krw)
    entry_plan = _entry_plan(config)
    exit_plan = _exit_plan(config)
    signal = None
    if entry_plan.mode == "close_auction":
        limit_price = round_price_for_krx(float(bar.close) * (1.0 + float(config.auction_limit_offset_bps) / 10_000.0), "buy_limit")
        order_type = "CLOSE_AUCTION"
        reason = "close_auction_entry"
        estimated_price = max(float(limit_price or bar.close), 1e-9)
    elif entry_plan.mode == "decision_next_open":
        limit_price = None
        order_type = "MARKET"
        reason = "decision_next_open_entry"
        estimated_price = _market_entry_sizing_price(bar, config)
    else:
        signal = find_olr_entry_signal(tuple(symbol_state.session_bars), candidate, entry_plan, require_fill_bar=False)
        if signal is None or signal.signal_index != len(symbol_state.session_bars) - 1:
            return OLRCoreResult(state=state)
        limit_price = None
        order_type = "MARKET"
        reason = signal.reason
        estimated_price = _market_entry_sizing_price(bar, config)
    pending_reserved = _pending_entry_reserved_notional(state)
    available_cash = max(float(portfolio.cash or 0.0) - pending_reserved, 0.0)
    if portfolio.external_entry_allocation:
        # Produce native desired size without consuming another intent's cash.
        # Actual shared funding is checked before any order is submitted.
        available_cash = equity
    entry_cost_buffer = (
        max(float(config.commission_bps), 0.0)
        + max(float(config.slippage_bps), 0.0)
        + (max(float(config.auction_adverse_bps), 0.0) if order_type == "CLOSE_AUCTION" else 0.0)
    ) / 10_000.0
    target_notional = equity * min(float(config.max_position_pct), weight)
    if portfolio.economic_size_envelope:
        target_notional = equity * float(config.max_position_pct)
    value_evidence = None
    original_target = target_notional
    if config.capacity_value_mode!='off':
        from .allocation_value import quote
        multiplier,value_evidence=quote(state,candidate,bar.timestamp+timedelta(minutes=5),config)
        target_notional=min(equity*config.max_position_pct,target_notional*multiplier)
    notional = min(target_notional, available_cash)
    qty = floor(notional / (estimated_price * (1.0 + entry_cost_buffer)))
    capacity_metadata = {}
    original_qty=floor(min(original_target,available_cash)/(estimated_price*(1+entry_cost_buffer)))
    if capacity_enabled(config):
        capacity_qty, _, evidence = capacity_forecast(state, bar, config)
        native_qty = qty
        if capacity_qty is not None:
            # The optional opportunity ledger must see a confirmed signal
            # even when its current executable size is zero. It applies the
            # capacity bound before any allocation/submission. Native callers
            # retain their existing sizing and attempt semantics.
            if not portfolio.entry_proposal_only:
                qty = min(qty, capacity_qty)
            original_qty=min(original_qty,capacity_qty)
        capacity_metadata = {"capacity_forecast": evidence, "capacity_native_qty": native_qty,
            "capacity_approved_qty": qty}
        if config.capacity_history_scope in ('conditional_shrink','conditional_exit') and evidence['exit'].get('selection_conditioning'):
            broad_qty=evidence['exit']['selection_conditioning']['broad_quantity']
            if config.capacity_entry_bars and 'entry' in evidence:
                broad_qty=min(broad_qty,evidence['entry']['quantity'])
            capacity_metadata['capacity_broad_only_qty']=min(native_qty,broad_qty)
        if config.capacity_volume_exponent and evidence['exit'].get('volume_conditioning'):
            unscaled=evidence['exit']['volume_conditioning']['base_quantity']
            if config.capacity_entry_bars and 'entry' in evidence:
                unscaled=min(unscaled,evidence['entry']['quantity'])
            capacity_metadata['capacity_unconditioned_qty']=min(native_qty,unscaled)
    if value_evidence is not None:
        capacity_metadata.update(capacity_value_evidence=value_evidence,
            capacity_without_value_qty=original_qty)
    if (not portfolio.entry_proposal_only and qty > 0
            and qty * estimated_price < equity * config.capacity_min_position_pct):
        symbol_state.entry_attempted = True
        return OLRCoreResult(state=state, decisions=[_decision(bar.timestamp, symbol,
            "ENTRY_SKIPPED", "below_meaningful_allocation", {**capacity_metadata, "qty": qty})])
    if qty <= 0:
        symbol_state.entry_attempted = True
        return OLRCoreResult(
            state=state,
            decisions=[_decision(bar.timestamp, symbol, "ENTRY_SKIPPED", "qty_zero", {"candidate_rank": candidate.rank, "allocation_weight": weight, **capacity_metadata})],
        )
    metadata = {
        **capacity_metadata,
        "strategy_core_version": OLR_CORE_VERSION,
        "source_artifact_hash": snapshot.artifact_hash,
        "source_fingerprint": snapshot.source_fingerprint,
        "candidate_rank": candidate.rank,
        "candidate_score": candidate.selection_score,
        "candidate_hash": f"{snapshot.artifact_hash}:{candidate.symbol}",
        "sector": candidate.sector,
        "afternoon_score_band_rule": str(candidate.metadata.get("afternoon_score_band_rule") or ""),
        "trade_entry_plan": asdict(entry_plan),
        "trade_exit_plan": asdict(exit_plan),
        "auction_fill_time": config.auction_fill_time,
        "auction_adverse_bps": config.auction_adverse_bps,
        "auction_nonfill_rate": config.auction_nonfill_rate,
        # Common deterministic liquidity scenario must use only information
        # known now. Whole-dataset provenance can change with future rows.
        "auction_nonfill_key": f"{snapshot.trade_date.isoformat()}:{candidate.symbol}:entry",
        "expiry_ts": _auction_expiry_ts(bar.timestamp, config.auction_fill_time),
        "expiry_basis": "closing_auction",
        "allocation_weight": weight,
        "daily_atr": candidate.daily_atr,
        "risk_per_share": max(float(candidate.daily_atr or 0.0), float(bar.close) * 0.005, 1.0),
        "estimated_entry_price": estimated_price,
        "entry_submission_time": bar.timestamp,
        "entry_submission_close": float(bar.close),
        "market_entry_price_buffer_bps": config.market_entry_price_buffer_bps if order_type == "MARKET" else 0.0,
        "target_notional": target_notional,
        "available_cash_before_entry": available_cash,
        "pending_entry_reserved_notional": pending_reserved,
        **({"external_entry_allocation": True} if portfolio.external_entry_allocation else {}),
        "entry_cost_buffer_pct": entry_cost_buffer,
    }
    protective_stop = _initial_protective_stop(tuple(symbol_state.session_bars), candidate, entry_plan, exit_plan, bar, estimated_price)
    metadata["hard_stop_enabled"] = exit_plan.mode == "managed" and exit_plan.hard_stop_enabled
    if protective_stop > 0.0:
        metadata["protective_stop_price"] = protective_stop
        metadata["risk_per_share"] = max(estimated_price - protective_stop, estimated_price * 0.001, 1.0)
    if config.capacity_opening_tail:
        from .opening_tail import evidence as opening_tail_evidence
        metadata['opening_tail_evidence'] = opening_tail_evidence(state, symbol, candidate.sector, bar.timestamp)
    metadata["reserved_notional"] = int(qty) * estimated_price * (1.0 + entry_cost_buffer)
    action = SubmitEntry(
        strategy_id=STRATEGY_ID,
        symbol=symbol,
        qty=int(qty),
        order_type=order_type,  # type: ignore[arg-type]
        limit_price=limit_price,
        stop_price=None,
        reason=reason,
        metadata=metadata,
    )
    if not portfolio.entry_proposal_only:
        symbol_state.entry_attempted = True
    decision = _decision(bar.timestamp, symbol, "ENTRY_SUBMITTED", reason, {**metadata, "qty": int(qty)})
    return OLRCoreResult(state=state, actions=[action], decisions=[decision])


def _managed_resting_exit_actions(fill: OLRFillEvent, position: OLRPositionState, exit_plan: OLRExitPlan) -> list[StrategyAction]:
    risk = _position_risk(position)
    entry = max(float(position.entry_price), 1e-9)
    qty = int(position.metadata.setdefault("acquired_qty", position.qty_open))
    fixed = position.metadata.setdefault("target_price_basis", {"entry": entry, "risk": risk})
    entry, risk = float(fixed["entry"]), float(fixed["risk"])
    submitted = position.metadata.setdefault("target_submitted", {})
    executed = position.metadata.get("target_executed", {})
    available = max(position.qty_open - sum(max(int(value) - int(executed.get(leg, 0)), 0)
                                           for leg, value in submitted.items()), 0)
    if qty <= 0 or risk <= 0.0:
        return []
    partial_qty = 0
    if exit_plan.partial_trigger_r > 0.0 and exit_plan.partial_fraction > 0.0 and qty > 1:
        partial_qty = max(1, floor(qty * min(max(float(exit_plan.partial_fraction), 0.0), 1.0)))
        if exit_plan.target_r > 0.0:
            partial_qty = min(partial_qty, qty - 1)
    actions: list[StrategyAction] = []
    base_metadata = {
        "strategy_core_version": OLR_CORE_VERSION,
        "source_artifact_hash": position.source_artifact_hash,
        "candidate_rank": position.candidate_rank,
        "candidate_score": position.candidate_score,
        "trade_exit_plan": asdict(exit_plan),
        "entry_price": entry,
        "risk_per_share": risk,
        "entry_lifecycle_key": position.metadata.get("entry_lifecycle_key", ""),
    }
    partial_delta = min(partial_qty - int(submitted.get("PARTIAL_TARGET", 0)), available)
    if partial_delta > 0:
        submitted["PARTIAL_TARGET"] = int(submitted.get("PARTIAL_TARGET", 0)) + partial_delta
        available -= partial_delta
        actions.append(
            SubmitPartialExit(
                strategy_id=STRATEGY_ID,
                symbol=fill.symbol,
                qty=partial_delta,
                order_type="LIMIT",
                limit_price=entry + float(exit_plan.partial_trigger_r) * risk,
                reason="partial_target",
                metadata={**base_metadata, "order_role": "PARTIAL_TARGET"},
            )
        )
    if exit_plan.target_r > 0.0:
        target_qty = qty - partial_qty if partial_qty > 0 else qty
        target_delta = min(target_qty - int(submitted.get("TARGET", 0)), available)
        if target_delta > 0:
            submitted["TARGET"] = int(submitted.get("TARGET", 0)) + target_delta
            actions.append(
                SubmitExit(
                    strategy_id=STRATEGY_ID,
                    symbol=fill.symbol,
                    qty=target_delta,
                    order_type="LIMIT",
                    limit_price=entry + float(exit_plan.target_r) * risk,
                    reason="target",
                    metadata={**base_metadata, "order_role": "TARGET"},
                )
            )
    return actions


def _managed_partial_fill_actions(fill: OLRFillEvent, position: OLRPositionState) -> list[StrategyAction]:
    exit_plan = _exit_plan_from_payload(position.metadata.get("trade_exit_plan") or fill.metadata.get("trade_exit_plan"))
    if exit_plan.mode != "managed" or not exit_plan.hard_stop_enabled or position.qty_open <= 0:
        return []
    risk = _position_risk(position)
    stop = max(float(position.metadata.get("current_stop_price", 0.0) or 0.0), 0.0)
    if risk > 0.0 and exit_plan.partial_stop_r != 0.0:
        stop = max(stop, float(position.entry_price) + float(exit_plan.partial_stop_r) * risk)
    if stop <= 0.0:
        return []
    position.metadata["current_stop_price"] = stop
    return [
        ReplaceProtectiveStop(
            strategy_id=STRATEGY_ID,
            symbol=fill.symbol,
            stop_price=stop,
            qty=int(position.qty_open),
            reason="partial_target_stop_update",
            metadata={
                "strategy_core_version": OLR_CORE_VERSION,
                "source_artifact_hash": position.source_artifact_hash,
                "candidate_rank": position.candidate_rank,
                "candidate_score": position.candidate_score,
                "trade_exit_plan": asdict(exit_plan),
                "order_role": "STOP",
                "risk_per_share": risk,
            },
        )
    ]


def _maybe_submit_managed_exit(state: OLRState, bar: MarketBar, config: OLRConfig) -> OLRCoreResult:
    symbol = str(bar.symbol).zfill(6)
    symbol_state = state.symbol_state(symbol)
    position = symbol_state.position
    if position is None or position.qty_open <= 0:
        return OLRCoreResult(state=state)
    pending_id = symbol_state.pending_exit_order_id
    if pending_id:
        role = state.order_roles.get(pending_id, {})
        replaceable = (role.get('order_type') == 'CLOSE_AUCTION' and role.get('reason') == 'next_close_exit') or (
            role.get('order_type') == 'MARKET' and role.get('reason') == 'weak_followthrough_partial')
        if (not config.capacity_exit_priority or not replaceable
                or symbol in state.meta.get('priority_exit_pending', {})):
            return OLRCoreResult(state=state)
    exit_plan = _exit_plan(config)
    if exit_plan.mode != "managed":
        return OLRCoreResult(state=state)
    if bar.timestamp <= position.entry_time:
        return OLRCoreResult(state=state)
    held_bars = tuple(item for item in symbol_state.session_bars if item.timestamp > position.entry_time)
    if not held_bars:
        return OLRCoreResult(state=state)
    risk = _position_risk(position)
    if risk <= 0.0:
        return OLRCoreResult(state=state)
    current_stop = max(float(position.metadata.get("current_stop_price", 0.0) or 0.0), 0.0)
    vwap_fail_streak = int(float(position.metadata.get("vwap_fail_streak", 0.0) or 0.0))
    exit_check = _completed_bar_exit_reason(
        held_bars,
        len(held_bars) - 1,
        float(position.entry_price),
        risk,
        float(position.max_favorable_price),
        exit_plan,
        vwap_fail_streak,
    )
    position.metadata["vwap_fail_streak"] = int(exit_check.pop("vwap_fail_streak", vwap_fail_streak))
    reason = str(exit_check.get("reason") or "")
    if not reason and exit_plan.max_hold_bars > 0 and len(held_bars) >= int(exit_plan.max_hold_bars):
        reason = "max_hold"
    if reason:
        if pending_id:
            state.meta.setdefault('priority_exit_pending', {})[symbol] = {
                'cancel_order_id': pending_id, 'reason': reason,
                'trigger_bar': bar.timestamp.isoformat()}
            return OLRCoreResult(state=state,
                actions=[CancelOrders(STRATEGY_ID, symbol, 'priority_risk_exit_cancel_auction')],
                decisions=[_decision(bar.timestamp, symbol, 'EXIT_CANCEL_REQUESTED', reason,
                    {'cancel_order_id': pending_id, 'remaining_qty': position.qty_open})])
        return _submit_managed_market_exit(state, symbol_state, bar, position, exit_plan, reason, risk)
    if config.capacity_weak_partial_fraction:
        from .weak_partial import maybe_reduce
        reduced = maybe_reduce(state, symbol_state, position, bar, config, exit_plan, risk)
        if reduced is not None:
            return reduced
    if pending_id:
        return OLRCoreResult(state=state)
    next_stop = _next_bar_stop_from_completed_bar(
        float(position.entry_price),
        risk,
        current_stop or max(float(position.entry_price) - risk, 0.0),
        float(position.max_favorable_price),
        exit_plan,
    )
    if current_stop > 0.0 and next_stop > current_stop * 1.000001:
        position.metadata["current_stop_price"] = next_stop
        metadata = {
            "strategy_core_version": OLR_CORE_VERSION,
            "source_artifact_hash": position.source_artifact_hash,
            "candidate_rank": position.candidate_rank,
            "candidate_score": position.candidate_score,
            "trade_exit_plan": asdict(exit_plan),
            "order_role": "STOP",
            "risk_per_share": risk,
        }
        action = ReplaceProtectiveStop(
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            stop_price=next_stop,
            qty=int(position.qty_open),
            reason="managed_stop_update",
            metadata=metadata,
        )
        decision = _decision(bar.timestamp, symbol, "STOP_REPLACED", "managed_stop_update", {**metadata, "stop_price": next_stop})
        return OLRCoreResult(state=state, actions=[action], decisions=[decision])
    return OLRCoreResult(state=state)


def _submit_managed_market_exit(
    state: OLRState,
    symbol_state,
    bar: MarketBar,
    position: OLRPositionState,
    exit_plan: OLRExitPlan,
    reason: str,
    risk: float,
) -> OLRCoreResult:
    return _managed_market_exit_at(state, symbol_state, bar.timestamp, position, exit_plan, reason, risk)


def _managed_market_exit_at(state, symbol_state, timestamp, position, exit_plan, reason, risk):
    symbol = str(position.symbol).zfill(6)
    metadata = {
        "strategy_core_version": OLR_CORE_VERSION,
        "source_artifact_hash": position.source_artifact_hash,
        "candidate_rank": position.candidate_rank,
        "candidate_score": position.candidate_score,
        "trade_exit_plan": asdict(exit_plan),
        "order_role": "DISCRETIONARY_EXIT",
        "risk_per_share": risk,
    }
    action = SubmitExit(
        strategy_id=STRATEGY_ID,
        symbol=symbol,
        qty=int(position.qty_open),
        order_type="MARKET",
        limit_price=None,
        reason=reason,
        metadata=metadata,
    )
    decision = _decision(timestamp, symbol, "EXIT_SUBMITTED", reason, {**metadata, "qty": int(position.qty_open)})
    symbol_state.exit_attempted_dates.add(timestamp.date())
    return OLRCoreResult(state=state, actions=[CancelOrders(STRATEGY_ID, symbol, f"replace_resting_orders_with_{reason}"), action], decisions=[decision])


def _maybe_submit_next_close_exit(state: OLRState, bar: MarketBar, config: OLRConfig) -> OLRCoreResult:
    symbol = str(bar.symbol).zfill(6)
    symbol_state = state.symbol_state(symbol)
    position = symbol_state.position
    if position is None or position.qty_open <= 0:
        return OLRCoreResult(state=state)
    if symbol_state.pending_exit_order_id:
        return OLRCoreResult(state=state)
    bar_date = bar.timestamp.date()
    if bar_date <= position.entry_time.date() or bar_date in symbol_state.exit_attempted_dates:
        return OLRCoreResult(state=state)
    exit_plan = _exit_plan(config)
    if exit_plan.mode not in {"next_close", "managed"}:
        return OLRCoreResult(state=state)
    if config.capacity_release_minutes:
        from kis_core.trading_calendar import get_trading_calendar
        closing = get_trading_calendar().session_hours(bar_date)[1]
        decision_at = bar.timestamp + timedelta(minutes=5)
        release_at = bar.timestamp.replace(hour=closing.hour, minute=closing.minute,
            second=0, microsecond=0) - timedelta(minutes=config.capacity_release_minutes)
        if decision_at < release_at:
            return OLRCoreResult(state=state)
        if (decision_at >= release_at
                and position.metadata.get('capital_release_date') != bar_date.isoformat()):
            qty = min(position.qty_open, max(1, floor(position.qty_open * config.capacity_release_fraction)))
            position.metadata['capital_release_date'] = bar_date.isoformat()
            if qty == position.qty_open:
                return _submit_managed_market_exit(state, symbol_state, bar, position,
                    exit_plan, 'scheduled_capital_release', _position_risk(position))
            metadata = {'strategy_core_version': OLR_CORE_VERSION,
                'source_artifact_hash': position.source_artifact_hash,
                'candidate_rank': position.candidate_rank, 'candidate_score': position.candidate_score,
                'trade_exit_plan': asdict(exit_plan), 'order_role': 'CAPACITY_SLICE',
                'risk_per_share': _position_risk(position), 'release_at': release_at.isoformat()}
            action = SubmitPartialExit(STRATEGY_ID, symbol, qty, 'MARKET', None,
                'scheduled_capital_release', metadata)
            return OLRCoreResult(state=state,
                actions=[CancelOrders(STRATEGY_ID, symbol, 'scheduled_capital_release'), action],
                decisions=[_decision(bar.timestamp, symbol, 'PARTIAL_EXIT_SUBMITTED',
                    'scheduled_capital_release', {**metadata, 'qty': qty})])
    if _bar_time(bar) < session_cutoff(bar.timestamp.date(), DECISION_CUTOFF):
        return OLRCoreResult(state=state)
    if config.capacity_exit_guard:
        _, exit_qty, evidence = capacity_forecast(state, bar, config)
        if (exit_qty is not None and position.qty_open > exit_qty
                and position.metadata.get('capacity_slice_date')!=bar_date.isoformat()):
            if config.capacity_exit_guard_mode=='excess_only' and exit_qty>0:
                qty=position.qty_open-exit_qty
                metadata={'strategy_core_version':OLR_CORE_VERSION,'source_artifact_hash':position.source_artifact_hash,
                    'candidate_rank':position.candidate_rank,'candidate_score':position.candidate_score,
                    'trade_exit_plan':asdict(exit_plan),'capacity_forecast':evidence,
                    'order_role':'CAPACITY_SLICE','risk_per_share':_position_risk(position)}
                position.metadata['capacity_slice_date']=bar_date.isoformat()
                action=SubmitPartialExit(STRATEGY_ID,symbol,qty,'MARKET',None,'capacity_exit_slice',metadata)
                # Wait for confirmed partial completion before placing the
                # remainder in the auction. No overlapping sell quantities.
                return OLRCoreResult(state=state,actions=[CancelOrders(STRATEGY_ID,symbol,'capacity_exit_slice'),action],
                    decisions=[_decision(bar.timestamp,symbol,'PARTIAL_EXIT_SUBMITTED','capacity_exit_slice',
                        {**metadata,'qty':qty,'retained_qty':exit_qty})])
            result = _submit_managed_market_exit(state, symbol_state, bar, position, exit_plan,
                "capacity_exit_before_auction", _position_risk(position))
            result.actions = [replace(action, metadata={**dict(action.metadata), "capacity_forecast": evidence})
                if isinstance(action, SubmitExit) else action for action in result.actions]
            return result
    limit_price = round_price_for_krx(float(bar.close) * (1.0 - float(config.auction_limit_offset_bps) / 10_000.0), "sell_limit")
    metadata = {
        "strategy_core_version": OLR_CORE_VERSION,
        "source_artifact_hash": position.source_artifact_hash,
        "candidate_rank": position.candidate_rank,
        "candidate_score": position.candidate_score,
        "trade_exit_plan": asdict(exit_plan),
        "auction_fill_time": config.auction_fill_time,
        "auction_adverse_bps": config.auction_adverse_bps,
        "auction_nonfill_rate": config.auction_nonfill_rate,
        "auction_nonfill_key": f"{bar_date.isoformat()}:{symbol}:exit",
        "expiry_ts": _auction_expiry_ts(bar.timestamp, config.auction_fill_time),
        "expiry_basis": "closing_auction",
    }
    action = SubmitExit(
        strategy_id=STRATEGY_ID,
        symbol=symbol,
        qty=int(position.qty_open),
        order_type="CLOSE_AUCTION",
        limit_price=limit_price,
        reason="next_close_exit",
        metadata=metadata,
    )
    symbol_state.exit_attempted_dates.add(bar_date)
    decision = _decision(bar.timestamp, symbol, "EXIT_SUBMITTED", "next_close_exit", {**metadata, "qty": int(position.qty_open)})
    return OLRCoreResult(state=state, actions=[CancelOrders(STRATEGY_ID, symbol, "replace_stop_with_next_close_exit"), action], decisions=[decision])


def _remember_session_bar(symbol_state, bar: MarketBar) -> None:
    if symbol_state.session_bars and symbol_state.session_bars[-1].timestamp == bar.timestamp:
        return
    symbol_state.session_bars.append(bar)


def _entry_plan(config: OLRConfig) -> OLREntryPlan:
    payload = dict(config.trade_entry_plan or {})
    if payload:
        allowed = set(OLREntryPlan.__dataclass_fields__)
        return OLREntryPlan(**{key: value for key, value in payload.items() if key in allowed})
    return OLREntryPlan("", config.entry_mode)


def _exit_plan(config: OLRConfig) -> OLRExitPlan:
    payload = dict(config.trade_exit_plan or {})
    return _exit_plan_from_payload(payload) if payload else OLRExitPlan("", mode=config.exit_mode)


def _exit_plan_from_payload(value) -> OLRExitPlan:
    payload = dict(value or {})
    if payload:
        allowed = set(OLRExitPlan.__dataclass_fields__)
        return OLRExitPlan(**{key: value for key, value in payload.items() if key in allowed})
    return OLRExitPlan("", mode="next_close")


def _initial_protective_stop(
    bars: tuple[MarketBar, ...],
    candidate: OLRDailyCandidate,
    entry_plan: OLREntryPlan,
    exit_plan: OLRExitPlan,
    signal_bar: MarketBar,
    estimated_entry: float,
) -> float:
    if exit_plan.mode != "managed" or not exit_plan.hard_stop_enabled:
        return 0.0
    entry = max(float(estimated_entry), 1e-9)
    atr = max(float(candidate.daily_atr or 0.0), entry * 0.006, 1.0)
    if exit_plan.stop_mode == "decision_low":
        decision_lows = [float(bar.low) for bar in bars if _bar_time(bar) < session_cutoff(bar.timestamp.date(), DECISION_CUTOFF)]
        if decision_lows:
            return min(min(decision_lows), entry * 0.997)
    if exit_plan.stop_mode == "signal_low":
        return min(float(signal_bar.low), entry * 0.997)
    if exit_plan.stop_mode == "fixed_pct":
        return entry * (1.0 - max(float(exit_plan.stop_pct), 0.001))
    if exit_plan.stop_mode == "vwap":
        vwap = _running_vwap(bars)
        if vwap > 0.0:
            return min(vwap * 0.997, entry * 0.997)
    if exit_plan.stop_mode == "entry_open_gap":
        return min(float(signal_bar.open), entry) * (1.0 - max(float(exit_plan.stop_pct), 0.003))
    return entry - float(exit_plan.stop_atr_mult) * atr


def _candidate_weights(selected: Iterable[OLRDailyCandidate], config: OLRConfig) -> dict[str, float]:
    items = tuple(selected)
    if not items:
        return {}
    mode = str(config.allocation_mode or "").lower()
    if mode in {"rank_decay", "rank_weighted"}:
        raw = [
            1.0 / max(float(item.rank or index + 1), 1.0) ** max(float(config.rank_decay), 0.0)
            for index, item in enumerate(items)
        ]
    elif mode in {"score_weighted", "rank_score_weighted"}:
        raw = []
        for index, item in enumerate(items):
            score = max(float(item.selection_score or 0.0), 0.0)
            rank_component = 1.0
            if mode == "rank_score_weighted":
                rank_component = 1.0 / max(float(item.rank or index + 1), 1.0) ** max(float(config.rank_decay), 0.0)
            raw.append(max(score, 1e-9) * rank_component)
    else:
        raw = [1.0 for _ in items]
    total = sum(raw) or 1.0
    gross = float(config.target_gross_exposure)
    cap = float(config.max_position_pct)
    return {
        item.symbol: max(0.0, min(cap, (value / total) * gross))
        for item, value in zip(items, raw)
    }


def _pending_entry_reserved_notional(state: OLRState) -> float:
    return sum(pending_entry_reservation(symbol_state)[1] for symbol_state in state.symbols.values())


def pending_entry_reservation(symbol_state) -> tuple[int, float]:
    """Remaining shares and cash obligation; filled shares already reduced cash."""
    if not symbol_state.pending_entry_order_id:
        return 0, 0.0
    metadata = symbol_state.pending_entry_metadata
    submitted = int(_metadata_float(metadata, "submitted_qty"))
    filled = int(_metadata_float(metadata, "entry_filled_qty"))
    # Accept checkpoints made before entry_filled_qty was introduced.
    positions = [symbol_state.position, *symbol_state.entry_history.values()]
    for position in positions:
        if position is not None and position.entry_order_id == symbol_state.pending_entry_order_id:
            filled = max(filled, int(position.metadata.get("acquired_qty", position.qty_open)))
    reserved = max(0.0, _metadata_float(metadata, "reserved_notional"))
    remaining = max(submitted - filled, 0)
    return remaining, reserved * remaining / submitted if submitted > 0 else reserved


def _metadata_float(metadata: dict, key: str) -> float:
    try:
        return float(metadata.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _position_risk(position: OLRPositionState) -> float:
    risk = _metadata_float(position.metadata, "risk_per_share")
    if risk <= 0.0:
        stop = _metadata_float(position.metadata, "current_stop_price")
        if stop > 0.0:
            risk = float(position.entry_price) - stop
    return max(float(risk), float(position.entry_price) * 0.001, 1.0)


def _market_entry_sizing_price(bar: MarketBar, config: OLRConfig) -> float:
    buffer = max(float(config.market_entry_price_buffer_bps), 0.0) / 10_000.0
    return max(float(bar.close) * (1.0 + buffer), 1e-9)


def _bar_time(bar: MarketBar) -> time:
    timestamp = bar.timestamp
    return timestamp.time().replace(second=0, microsecond=0)


def _auction_expiry_ts(timestamp: datetime, auction_fill_time: str) -> str:
    clock = _parse_clock(auction_fill_time)
    return datetime.combine(timestamp.date(), clock, tzinfo=timestamp.tzinfo).isoformat()


def _parse_clock(value: str) -> time:
    try:
        hour, minute, *_ = str(value or "15:30").split(":")
        return time(int(hour), int(minute))
    except (TypeError, ValueError):
        return time(15, 30)


def _candidate_label(candidate: OLRDailyCandidate, *keys: str) -> float:
    metadata = dict(candidate.metadata or {})
    for key in keys:
        if key in metadata:
            try:
                return float(metadata.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _decision(timestamp: datetime, symbol: str, code: str, reason: str, metadata: dict) -> DecisionEvent:
    return DecisionEvent(
        timestamp=timestamp,
        strategy_id=STRATEGY_ID,
        symbol=str(symbol).zfill(6),
        decision_code=code,
        reason=reason,
        state_snapshot_ref=str(metadata.get("source_artifact_hash", "")),
        metadata={key: _json_safe(value) for key, value in metadata.items()},
    )


def _json_safe(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    try:
        asdict(value)
    except Exception:
        return value
    return asdict(value)
