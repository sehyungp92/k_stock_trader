from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from strategy_common.actions import StrategyAction
from strategy_common.market import MarketBar

from .action_router import DecisionRefIndex, RuntimeActionRouter
from .coordinator import StrategyRuntimeDescriptor
from .hashing import canonical_json_hash
from .portfolio_context import PortfolioContextProvider
from .session_capture import PaperSessionRecorder, market_bar_hash


@dataclass(frozen=True, slots=True)
class CollectedAction:
    action: StrategyAction
    provisional_order_ref: str
    batch_index: int


@dataclass(slots=True)
class ActionCollector:
    strategy_id: str
    event_ref: str
    event_type: str
    event_timestamp: datetime
    source_artifact_hash: str = ""
    source_fingerprint: str = ""
    defer_order_memory: bool = True
    _actions: list[CollectedAction] = field(default_factory=list)

    @property
    def actions(self) -> tuple[CollectedAction, ...]:
        return tuple(self._actions)

    def submit(self, action: StrategyAction) -> str:
        index = len(self._actions)
        provisional = f"{self.strategy_id}:{self.event_ref}:action:{index}"
        metadata = dict(action.metadata or {})
        metadata.setdefault("event_ref", self.event_ref)
        metadata.setdefault("event_type", self.event_type)
        metadata.setdefault("timestamp", self.event_timestamp.isoformat())
        metadata.setdefault("provisional_order_ref", provisional)
        metadata.setdefault("source_artifact_hash", self.source_artifact_hash)
        metadata.setdefault("source_fingerprint", self.source_fingerprint)
        metadata.setdefault("batch_index", index)
        self._actions.append(CollectedAction(replace(action, metadata=metadata), provisional, index))
        return provisional


@dataclass(frozen=True, slots=True)
class RuntimeEventResult:
    strategy_id: str
    event_ref: str
    event_type: str
    timestamp: datetime
    decision_count: int
    action_count: int
    accepted_intent_count: int
    blocked_action_count: int
    state_hash: str


@dataclass(frozen=True, slots=True)
class PendingRuntimeEvent:
    strategy_id: str
    event_ref: str
    event_type: str
    timestamp: datetime
    decisions: tuple[Any, ...]
    actions: tuple[CollectedAction, ...]
    decision_refs: Any


@dataclass(slots=True)
class RuntimeSessionDriver:
    descriptor: StrategyRuntimeDescriptor
    action_router: RuntimeActionRouter
    recorder: PaperSessionRecorder
    portfolio_context: PortfolioContextProvider
    mode: str
    evidence_mode: str | None = None
    order_identity: dict[str, str] = field(default_factory=dict)
    order_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def handle_bar(self, bar: MarketBar) -> RuntimeEventResult:
        pending = await self.collect_bar(bar)
        route_results = await self.action_router.route_actions(
            pending.actions,
            portfolio_context=self.portfolio_context,
            event_ref=pending.event_ref,
            event_type=pending.event_type,
            event_timestamp=pending.timestamp,
            decision_refs=pending.decision_refs,
        )
        return self.finish_collected_event(pending, route_results)

    async def collect_bar(self, bar: MarketBar) -> PendingRuntimeEvent:
        if not bar.is_completed:
            raise ValueError(f"incomplete bar cannot be processed in {self.mode} mode: {bar.symbol} {bar.timestamp}")
        replay_mode = str(self.mode).lower() in {"replay", "offline_replay"}
        if not replay_mode:
            self.recorder.record_market_bar(bar)
        return await self._begin_event(
            event_type="bar",
            timestamp=bar.timestamp,
            payload=bar.to_json_dict(),
            callback=lambda collector: self.descriptor.engine.on_bar(
                bar,
                self.portfolio_context.portfolio_view(self.descriptor.strategy_id),
                collector.submit,
            ),
            no_action_reason=_no_action_reason_for_bar(self.descriptor, bar),
        )

    async def handle_fill(self, fill: object) -> RuntimeEventResult:
        fill = self._resolve_event_order_ref(fill, event_type="fill")
        pre_position = _position_view(getattr(self.descriptor.engine, "state", None), str(getattr(fill, "symbol", "")))
        self._release_router_reservation_for_event(fill, qty=int(getattr(fill, "qty", 0) or 0))
        timestamp = _event_timestamp(fill)
        strategy_id = self.descriptor.strategy_id
        self.portfolio_context.apply_fill(
            strategy_id,
            str(getattr(fill, "symbol", "")),
            str(getattr(fill, "side", "")),
            int(getattr(fill, "qty", 0) or 0),
            float(getattr(fill, "price", 0.0) or 0.0),
        )
        fill_row = _event_stream_row("runtime_fill_event", strategy_id, "fill", fill)
        fill_row["portfolio_context_after"] = _portfolio_context_payload(self.portfolio_context, reason="fill_applied")
        self.recorder.append_jsonl("fill_events.jsonl", fill_row)
        result = await self._handle(
            event_type="fill",
            timestamp=timestamp,
            payload=_json_value(fill),
            callback=lambda collector: self.descriptor.engine.on_fill(fill, collector.submit),
            no_action_reason="fill_no_followup_action",
            refresh_context=False,
        )
        if str(getattr(fill, "side", "") or "").upper().strip() == "SELL":
            self._record_trade_outcome(fill, pre_position)
        return result

    async def handle_timer(self, timestamp: datetime) -> RuntimeEventResult:
        return await self._handle(
            event_type="timer",
            timestamp=timestamp,
            payload={"timestamp": timestamp.isoformat()},
            callback=lambda collector: self.descriptor.engine.on_timer(timestamp, collector.submit),
            no_action_reason="timer_no_action",
        )

    async def handle_order_event(self, event: object) -> RuntimeEventResult:
        event = self._resolve_event_order_ref(event, event_type="order_event")
        event = self._canonical_order_event(event)
        if _is_terminal_order_event(event):
            self._release_router_reservation_for_event(event)
        timestamp = _event_timestamp(event)
        strategy_id = self.descriptor.strategy_id
        self.recorder.append_jsonl("order_events.jsonl", _event_stream_row("runtime_order_event", strategy_id, "order_event", event))
        expired_handler = getattr(self.descriptor.engine, "on_order_expired", None)
        update_handler = getattr(self.descriptor.engine, "on_order_update", None)
        if _is_expired_order_event(event) and callable(expired_handler):
            callback = lambda collector: expired_handler(event, collector.submit)
            reason = "order_event_no_action"
        elif callable(update_handler):
            callback = lambda collector: update_handler(event, collector.submit)
            reason = "order_update_no_action"
        elif callable(expired_handler):
            callback = lambda collector: expired_handler(event, collector.submit)
            reason = "order_event_no_action"
        else:
            callback = lambda collector: []
            reason = "unsupported_order_event"
        return await self._handle(
            event_type="order_event",
            timestamp=timestamp,
            payload=_json_value(event),
            callback=callback,
            no_action_reason=reason,
        )

    def _canonical_order_event(self, event: object) -> object:
        if self.descriptor.strategy_id != "OLR" or not _is_expired_order_event(event) or event.__class__.__name__.endswith("ExpiredOrderEvent"):
            return event
        from strategy_olr.core.core_models import OLRExpiredOrderEvent

        metadata = dict((event.get("metadata") if isinstance(event, Mapping) else getattr(event, "metadata", {})) or {})
        return OLRExpiredOrderEvent(
            order_id=str((event.get("order_id") if isinstance(event, Mapping) else getattr(event, "order_id", "")) or ""),
            symbol=str((event.get("symbol") if isinstance(event, Mapping) else getattr(event, "symbol", "")) or "").zfill(6),
            side=str((event.get("side") if isinstance(event, Mapping) else getattr(event, "side", "")) or "").upper().strip(),
            order_type=str((event.get("order_type") if isinstance(event, Mapping) else getattr(event, "order_type", "")) or ""),
            qty=_event_qty(event),
            timestamp=_event_timestamp(event),
            reason=str((event.get("reason") if isinstance(event, Mapping) else getattr(event, "reason", "")) or ""),
            metadata=metadata,
        )

    def _release_router_reservation_for_event(self, event: object, *, qty: int | None = None) -> None:
        metadata = dict((event.get("metadata") if isinstance(event, Mapping) else getattr(event, "metadata", {})) or {})
        refs = (
            event.get("order_id") if isinstance(event, Mapping) else getattr(event, "order_id", None),
            event.get("provisional_order_ref") if isinstance(event, Mapping) else getattr(event, "provisional_order_ref", None),
            event.get("broker_order_id") if isinstance(event, Mapping) else getattr(event, "broker_order_id", None),
            metadata.get("provisional_order_ref"),
            metadata.get("broker_order_id"),
            metadata.get("original_order_id"),
            metadata.get("intent_id"),
        )
        for ref in refs:
            if self.action_router.release_order_ref(ref, qty=qty):
                return

    def _record_trade_outcome(self, fill: object, pre_position: Mapping[str, Any] | None) -> None:
        if not pre_position:
            return
        qty = min(max(int(getattr(fill, "qty", 0) or 0), 0), max(int(pre_position.get("qty_open") or getattr(fill, "qty", 0) or 0), 0))
        entry_price = float(pre_position.get("entry_price") or 0.0)
        exit_price = float(getattr(fill, "price", 0.0) or 0.0)
        if qty <= 0 or entry_price <= 0.0 or exit_price <= 0.0:
            return
        symbol = str(getattr(fill, "symbol", "") or "").zfill(6)
        post_position = _position_view(getattr(self.descriptor.engine, "state", None), symbol)
        pre_metadata = dict(pre_position.get("metadata") or {})
        fill_metadata = dict(getattr(fill, "metadata", {}) or {})
        join_metadata = _trade_join_metadata(pre_metadata, fill_metadata)
        source_artifact_hash = str(
            pre_position.get("source_artifact_hash")
            or pre_metadata.get("source_artifact_hash")
            or fill_metadata.get("source_artifact_hash")
            or self.descriptor.artifact_hash
            or ""
        )
        exit_order_id = str(getattr(fill, "order_id", "") or "")
        broker_order_id = str(fill_metadata.get("broker_order_id") or fill_metadata.get("original_order_id") or "")
        trade_id = canonical_json_hash(
            {
                "strategy_id": self.descriptor.strategy_id,
                "symbol": symbol,
                "entry_order_id": str(pre_position.get("entry_order_id") or ""),
                "exit_order_id": exit_order_id,
                "exit_time": _event_timestamp(fill).isoformat(),
                "qty": qty,
            }
        )
        payload = {
            "record_type": "runtime_trade_outcome",
            "trade_id": trade_id,
            "strategy_id": self.descriptor.strategy_id,
            "symbol": symbol,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": _json_value(pre_position.get("entry_time")),
            "exit_time": _event_timestamp(fill).isoformat(),
            "gross_return_pct": (exit_price / entry_price) - 1.0,
            "realized_pnl": (exit_price - entry_price) * qty,
            "position_closed": post_position is None or int(post_position.get("qty_open") or 0) <= 0,
            "entry_order_id": str(pre_position.get("entry_order_id") or ""),
            "exit_order_id": exit_order_id,
            "order_id": exit_order_id,
            "broker_order_id": broker_order_id,
            "kis_order_id": broker_order_id,
            "exit_fill_id": str(fill_metadata.get("kis_exec_id") or fill_metadata.get("execution_id") or f"{exit_order_id}:{_event_timestamp(fill).isoformat()}:{qty}"),
            "source_artifact_hash": source_artifact_hash,
            "artifact_hash": source_artifact_hash,
            "candidate_rank": int(pre_position.get("candidate_rank") or pre_metadata.get("candidate_rank") or fill_metadata.get("candidate_rank") or 0),
            "sector": str(pre_position.get("sector") or pre_metadata.get("sector") or fill_metadata.get("sector") or "UNKNOWN"),
            "reason": str(getattr(fill, "reason", "") or ""),
            **join_metadata,
            "metadata": _stable_trade_metadata({**pre_metadata, **fill_metadata}),
        }
        self.recorder.append_jsonl("trade_outcomes.jsonl", payload)

    async def _handle(
        self,
        *,
        event_type: str,
        timestamp: datetime,
        payload: Mapping[str, Any],
        callback: Any,
        no_action_reason: str,
        refresh_context: bool = True,
    ) -> RuntimeEventResult:
        pending = await self._begin_event(
            event_type=event_type,
            timestamp=timestamp,
            payload=payload,
            callback=callback,
            no_action_reason=no_action_reason,
            refresh_context=refresh_context,
        )
        route_results = await self.action_router.route_actions(
            pending.actions,
            portfolio_context=self.portfolio_context,
            event_ref=pending.event_ref,
            event_type=pending.event_type,
            event_timestamp=pending.timestamp,
            decision_refs=pending.decision_refs,
        )
        return self.finish_collected_event(pending, route_results)

    async def _begin_event(
        self,
        *,
        event_type: str,
        timestamp: datetime,
        payload: Mapping[str, Any],
        callback: Any,
        no_action_reason: str,
        refresh_context: bool = True,
    ) -> PendingRuntimeEvent:
        event_timestamp = _ensure_aware(timestamp)
        event_ref = _event_ref(self.descriptor.strategy_id, event_type, payload)
        event_sequence = self.recorder.next_event_sequence()
        event_input = {
            "record_type": "runtime_event_input",
            "event_sequence": event_sequence,
            "strategy_id": self.descriptor.strategy_id,
            "event_ref": event_ref,
            "event_type": event_type,
            "timestamp": event_timestamp.isoformat(),
            "payload_hash": canonical_json_hash(payload),
            "event_input_hash": canonical_json_hash(
                {
                    "strategy_id": self.descriptor.strategy_id,
                    "event_type": event_type,
                    "timestamp": event_timestamp.isoformat(),
                    "payload": _json_value(payload),
                }
            ),
            "payload": _json_value(payload),
        }
        if event_type == "bar":
            event_input["bar_hash"] = market_bar_hash(payload)
            event_input["bar_row_key"] = event_input["bar_hash"]
        self.recorder.append_jsonl(
            "decision_stream.jsonl",
            event_input,
        )
        if refresh_context:
            await self.portfolio_context.refresh()
            self.action_router.rehydrate_pending_reservations(
                self.portfolio_context.iter_working_orders(),
                source="oms_positions",
                portfolio_context=self.portfolio_context,
            )
        collector = ActionCollector(
            strategy_id=self.descriptor.strategy_id,
            event_ref=event_ref,
            event_type=event_type,
            event_timestamp=event_timestamp,
            source_artifact_hash=self.descriptor.artifact_hash,
            source_fingerprint=getattr(self.descriptor.snapshot, "source_fingerprint", ""),
        )
        decisions = list(callback(collector) or [])
        decision_refs = self.action_router.record_decisions(decisions, event_ref=event_ref, event_type=event_type)
        if not decisions and not collector.actions:
            self._record_no_action(event_ref, event_type, event_timestamp, no_action_reason)
        return PendingRuntimeEvent(
            strategy_id=self.descriptor.strategy_id,
            event_ref=event_ref,
            event_type=event_type,
            timestamp=event_timestamp,
            decisions=tuple(decisions),
            actions=collector.actions,
            decision_refs=decision_refs,
        )

    def finish_collected_event(
        self,
        pending: PendingRuntimeEvent,
        route_results: tuple[Any, ...],
    ) -> RuntimeEventResult:
        own_results = tuple(result for result in route_results if _route_result_strategy_id(result) == self.descriptor.strategy_id)
        event_ref = pending.event_ref
        event_type = pending.event_type
        event_timestamp = pending.timestamp
        self._remember_order_identity(own_results)
        update_decisions = self._reconcile_route_results(own_results)
        if update_decisions:
            self.action_router.record_decisions(update_decisions, event_ref=event_ref, event_type=event_type)
        state_hash = self.action_router.record_state_snapshot(
            self.descriptor.strategy_id,
            getattr(self.descriptor.engine, "state", None),
            metadata={
                "record_reason": f"runtime_after_{event_type}",
                "mode": self.evidence_mode or self.mode,
                "event_ref": event_ref,
                "event_type": event_type,
                "event_timestamp": event_timestamp.isoformat(),
                "artifact_stage": self.descriptor.artifact_stage,
                "artifact_hash": self.descriptor.artifact_hash,
            },
        )
        return RuntimeEventResult(
            strategy_id=self.descriptor.strategy_id,
            event_ref=event_ref,
            event_type=event_type,
            timestamp=event_timestamp,
            decision_count=len(pending.decisions),
            action_count=len(pending.actions),
            accepted_intent_count=sum(1 for item in own_results if item.accepted),
            blocked_action_count=sum(1 for item in own_results if item.blocked),
            state_hash=state_hash,
        )

    def _record_no_action(self, event_ref: str, event_type: str, timestamp: datetime, reason: str) -> None:
        payload = {
            "record_type": "runtime_no_action",
            "strategy_id": self.descriptor.strategy_id,
            "event_ref": event_ref,
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "reason_code": reason,
        }
        payload["decision_ref"] = canonical_json_hash(payload)
        self.recorder.append_jsonl("decision_stream.jsonl", payload)

    def _remember_order_identity(self, route_results: tuple[Any, ...]) -> None:
        for result in route_results:
            provisional = str(getattr(result, "provisional_order_ref", "") or "")
            if not provisional:
                continue
            action_payload = dict(getattr(result, "routed_action_payload", {}) or {})
            action_metadata = dict(action_payload.get("metadata") or {})
            for raw in (
                provisional,
                getattr(result, "action_ref", None),
                getattr(result, "final_order_ref", None),
                getattr(result, "intent_id", None),
                getattr(result, "broker_order_id", None),
            ):
                if raw not in (None, ""):
                    self.order_identity[str(raw)] = provisional
            self.order_metadata[provisional] = {
                "action_ref": str(getattr(result, "action_ref", "") or ""),
                "provisional_order_ref": provisional,
                "final_order_ref": str(getattr(result, "final_order_ref", "") or ""),
                "intent_id": str(getattr(result, "intent_id", "") or ""),
                "broker_order_id": str(getattr(result, "broker_order_id", "") or ""),
                "portfolio_decision_ref": str(getattr(result, "portfolio_decision_ref", "") or ""),
                "oms_status": str(getattr(result, "oms_status", "") or ""),
                "event_ref": str(action_metadata.get("event_ref") or ""),
                "decision_ref": str(action_metadata.get("decision_ref") or ""),
                "source_artifact_hash": str(action_metadata.get("source_artifact_hash") or ""),
                "source_fingerprint": str(action_metadata.get("source_fingerprint") or ""),
                "candidate_hash": str(action_metadata.get("candidate_hash") or ""),
                "portfolio_policy_hash": str(action_metadata.get("portfolio_policy_hash") or ""),
            }

    def _reconcile_route_results(self, route_results: tuple[Any, ...]) -> list[Any]:
        decisions: list[Any] = []
        for result in route_results:
            provisional = str(getattr(result, "provisional_order_ref", "") or "")
            if not provisional:
                continue
            routed_action = getattr(result, "routed_action", None)
            if getattr(result, "accepted", False) and routed_action is not None:
                reconciler = getattr(self.descriptor.engine, "reconcile_submitted_order", None)
                if callable(reconciler):
                    reconciler(provisional, routed_action)
            if getattr(result, "accepted", False):
                continue
            update = self._route_rejection_event(result)
            handler = getattr(self.descriptor.engine, "on_order_update", None)
            if callable(handler):
                decisions.extend(list(handler(update, lambda action: None) or []))
        return decisions

    def _route_rejection_event(self, result: Any) -> object:
        action_payload = dict(getattr(result, "routed_action_payload", {}) or {})
        metadata = dict((action_payload.get("metadata") or {}))
        metadata.update(
            {
                "provisional_order_ref": getattr(result, "provisional_order_ref", ""),
                "action_ref": getattr(result, "action_ref", ""),
                "portfolio_decision": getattr(result, "portfolio_decision", ""),
                "portfolio_reason_code": getattr(result, "portfolio_reason_code", ""),
                "portfolio_decision_ref": getattr(result, "portfolio_decision_ref", ""),
                "oms_status": getattr(result, "oms_status", "") or "",
                "message": getattr(result, "oms_message", "") or "",
                "resource_conflict_type": getattr(result, "resource_conflict_type", "") or "",
                "intent_id": getattr(result, "intent_id", "") or "",
                "broker_order_id": getattr(result, "broker_order_id", "") or "",
            }
        )
        status = "BLOCKED" if getattr(result, "blocked", False) else (str(getattr(result, "oms_status", "") or "REJECTED"))
        reason = (
            str(getattr(result, "portfolio_reason_code", "") or status.lower())
            if getattr(result, "blocked", False)
            else status.lower()
        )
        timestamp = _route_event_timestamp(metadata)
        common = {
            "order_id": str(getattr(result, "provisional_order_ref", "") or ""),
            "symbol": str(action_payload.get("symbol") or "").zfill(6),
            "status": status,
            "timestamp": timestamp,
            "reason": reason,
            "metadata": metadata,
        }
        if self.descriptor.strategy_id == "KALCB":
            from strategy_kalcb.core.core_models import KALCBOrderUpdateEvent

            return KALCBOrderUpdateEvent(role=str(metadata.get("order_role") or ""), **common)
        if self.descriptor.strategy_id == "OLR":
            from strategy_olr.core.core_models import OLROrderUpdateEvent

            return OLROrderUpdateEvent(
                side=_action_side_from_payload(action_payload),
                order_type=str(action_payload.get("order_type") or ""),
                qty=int(action_payload["qty"]) if action_payload.get("qty") not in (None, "") else None,
                **common,
            )
        raise ValueError(f"unsupported strategy_id={self.descriptor.strategy_id!r}")

    def _resolve_event_order_ref(self, event: object, *, event_type: str) -> object:
        event_mapping = event if isinstance(event, Mapping) else None
        replay_mode = str(self.mode).lower() in {"replay", "offline_replay"}
        metadata = dict(
            (event_mapping.get("metadata") if event_mapping is not None else getattr(event, "metadata", {})) or {}
        )
        raw_order_id = str((event_mapping.get("order_id") if event_mapping is not None else getattr(event, "order_id", "")) or "")
        candidates = (
            metadata.get("provisional_order_ref"),
            event_mapping.get("provisional_order_ref") if event_mapping is not None else getattr(event, "provisional_order_ref", None),
            raw_order_id,
            metadata.get("broker_order_id"),
            event_mapping.get("broker_order_id") if event_mapping is not None else getattr(event, "broker_order_id", None),
            metadata.get("intent_id"),
            event_mapping.get("intent_id") if event_mapping is not None else getattr(event, "intent_id", None),
        )
        provisional = ""
        for raw in candidates:
            if raw in (None, ""):
                continue
            text = str(raw)
            if text in self.order_identity:
                provisional = self.order_identity[text]
                break
            if text.startswith(f"{self.descriptor.strategy_id}:"):
                if not replay_mode or self._known_strategy_order_ref(text):
                    provisional = text
                    break
        if not provisional:
            if replay_mode:
                raise ValueError(f"unmapped {event_type} order identity for {self.descriptor.strategy_id}: {raw_order_id}")
            return event
        self.order_identity[provisional] = provisional
        if raw_order_id and raw_order_id != provisional:
            metadata.setdefault("broker_order_id", raw_order_id)
            metadata.setdefault("original_order_id", raw_order_id)
        for key, value in dict(self.order_metadata.get(provisional) or {}).items():
            if value not in (None, ""):
                metadata.setdefault(key, value)
        metadata["provisional_order_ref"] = provisional
        if is_dataclass(event):
            return replace(event, order_id=provisional, metadata=metadata)
        if event_mapping is not None:
            data = dict(event_mapping)
        elif hasattr(event, "__dict__"):
            data = dict(vars(event))
        else:
            payload = _json_value(event)
            data = dict(payload) if isinstance(payload, Mapping) else {}
        data["order_id"] = provisional
        data["metadata"] = metadata
        return SimpleNamespace(**data)

    def _known_strategy_order_ref(self, order_ref: str) -> bool:
        if order_ref in self.order_identity:
            return True
        state = getattr(self.descriptor.engine, "state", None)
        if order_ref in dict(getattr(state, "order_roles", {}) or {}):
            return True
        for symbol_state in dict(getattr(state, "symbols", {}) or {}).values():
            for attr in ("pending_entry_order_id", "pending_exit_order_id"):
                if str(getattr(symbol_state, attr, "") or "") == order_ref:
                    return True
            position = getattr(symbol_state, "position", None)
            if position is None:
                continue
            for attr in ("entry_order_id", "exit_order_id", "stop_order_id", "partial_order_id"):
                if str(getattr(position, attr, "") or "") == order_ref:
                    return True
        return False


async def handle_combined_bar(
    drivers: Mapping[str, RuntimeSessionDriver],
    bar: MarketBar,
    *,
    target_strategy_ids: Sequence[str] | None = None,
) -> tuple[RuntimeEventResult, ...]:
    """Process one market bar through multiple drivers as a single portfolio batch."""

    if not bar.is_completed:
        raise ValueError(f"incomplete bar cannot be processed in combined runtime path: {bar.symbol} {bar.timestamp}")
    selected_ids = tuple(
        str(strategy_id).upper().strip()
        for strategy_id in (target_strategy_ids if target_strategy_ids is not None else tuple(drivers))
        if str(strategy_id).strip()
    )
    if not selected_ids:
        return ()
    missing = [strategy_id for strategy_id in selected_ids if strategy_id not in drivers]
    if missing:
        raise ValueError(f"combined bar references unavailable strategy drivers: {', '.join(missing)}")
    selected = [drivers[strategy_id] for strategy_id in selected_ids]
    first = selected[0]
    if any(driver.portfolio_context is not first.portfolio_context for driver in selected):
        raise RuntimeError("combined runtime bar handling requires a shared PortfolioContextProvider")
    if any(driver.action_router is not first.action_router for driver in selected):
        raise RuntimeError("combined runtime bar handling requires a shared RuntimeActionRouter")

    pending = [await driver.collect_bar(bar) for driver in selected]
    if not pending:
        return ()
    route_results = await first.action_router.route_actions(
        tuple(action for event in pending for action in event.actions),
        portfolio_context=first.portfolio_context,
        event_type="bar",
        event_timestamp=getattr(bar, "timestamp", None),
        decision_refs=merge_decision_refs(event.decision_refs for event in pending),
    )
    return tuple(drivers[event.strategy_id].finish_collected_event(event, route_results) for event in pending)


def merge_decision_refs(indices: Sequence[DecisionRefIndex]) -> DecisionRefIndex:
    refs: list[str] = []
    action_refs: dict[str, list[str]] = {}
    for index in indices:
        refs.extend(str(ref) for ref in getattr(index, "refs", ()) if ref)
        for key, values in dict(getattr(index, "action_refs", {}) or {}).items():
            action_refs.setdefault(str(key), []).extend(str(value) for value in values if value)
    return DecisionRefIndex(
        tuple(dict.fromkeys(refs)),
        {key: tuple(dict.fromkeys(values)) for key, values in action_refs.items()},
    )


_TERMINAL_ORDER_STATUSES = {"BLOCKED", "REJECTED", "CANCELLED", "DEFERRED", "EXPIRED", "FILLED"}
_VOLATILE_TRADE_METADATA_KEYS = {
    "broker_order_id",
    "original_order_id",
    "intent_id",
    "provisional_order_ref",
    "portfolio_decision_ref",
    "timestamp",
    "decision_time",
}


def _is_terminal_order_event(event: object) -> bool:
    if _is_expired_order_event(event):
        return True
    status = str((event.get("status") if isinstance(event, Mapping) else getattr(event, "status", "")) or "").upper().strip()
    return status in _TERMINAL_ORDER_STATUSES


def _is_expired_order_event(event: object) -> bool:
    if event.__class__.__name__.endswith("ExpiredOrderEvent"):
        return True
    status = str((event.get("status") if isinstance(event, Mapping) else getattr(event, "status", "")) or "").upper().strip()
    return status == "EXPIRED"


def _position_view(state: Any, symbol: str) -> dict[str, Any] | None:
    key = str(symbol or "").zfill(6)
    symbols = dict(getattr(state, "symbols", {}) or {})
    symbol_state = symbols.get(key)
    if symbol_state is None:
        return None
    position = getattr(symbol_state, "position", None)
    if position is None:
        return None
    if is_dataclass(position):
        return dict(asdict(position))
    if isinstance(position, Mapping):
        return dict(position)
    if hasattr(position, "__dict__"):
        return {str(name): value for name, value in vars(position).items() if not str(name).startswith("_")}
    return None


def _stable_trade_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_value(value)
        for key, value in sorted(dict(metadata or {}).items(), key=lambda item: str(item[0]))
        if str(key) not in _VOLATILE_TRADE_METADATA_KEYS
    }


def _trade_join_metadata(*sources: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "event_ref",
        "decision_ref",
        "action_ref",
        "portfolio_decision_ref",
        "provisional_order_ref",
        "intent_id",
        "idempotency_key",
        "broker_order_id",
        "original_order_id",
        "source_fingerprint",
        "candidate_hash",
        "portfolio_policy_hash",
        "kis_resource_plan_hash",
    )
    result: dict[str, Any] = {}
    for field_name in fields:
        for source in reversed(sources):
            value = dict(source or {}).get(field_name)
            if value not in (None, ""):
                result[field_name] = value
                break
    return result


def _event_ref(strategy_id: str, event_type: str, payload: Mapping[str, Any]) -> str:
    return canonical_json_hash(
        {
            "strategy_id": str(strategy_id).upper().strip(),
            "event_type": event_type,
            "payload": _json_value(payload),
        }
    )[:24]


def _event_timestamp(event: object) -> datetime:
    raw = event.get("timestamp") if isinstance(event, Mapping) else getattr(event, "timestamp", None)
    if isinstance(raw, datetime):
        return _ensure_aware(raw)
    if raw not in (None, ""):
        return _ensure_aware(datetime.fromisoformat(str(raw)))
    raise ValueError(f"runtime event is missing timestamp: {type(event).__name__}")


def _event_qty(event: object) -> int | None:
    raw = event.get("qty") if isinstance(event, Mapping) else getattr(event, "qty", None)
    return int(raw) if raw not in (None, "") else None


def _route_event_timestamp(metadata: Mapping[str, Any]) -> datetime:
    raw = metadata.get("timestamp") or metadata.get("decision_time")
    if raw in (None, ""):
        raise ValueError("routed action rejection is missing event timestamp")
    if isinstance(raw, datetime):
        return _ensure_aware(raw)
    return _ensure_aware(datetime.fromisoformat(str(raw)))


def _ensure_aware(timestamp: datetime) -> datetime:
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)


def _no_action_reason_for_bar(descriptor: StrategyRuntimeDescriptor, bar: MarketBar) -> str:
    if not bar.is_completed:
        return "stale_bar"
    snapshot = getattr(descriptor, "snapshot", None)
    candidates = getattr(snapshot, "candidates", ()) or ()
    candidate_symbols = {str(getattr(candidate, "symbol", "")).zfill(6) for candidate in candidates}
    if candidate_symbols and str(bar.symbol).zfill(6) not in candidate_symbols:
        return "symbol_not_in_snapshot"
    return "no_signal"


def _event_stream_row(record_type: str, strategy_id: str, event_type: str, event: object) -> dict[str, Any]:
    payload = _json_value(event)
    row = {
        "record_type": record_type,
        "strategy_id": strategy_id,
        "event_ref": _event_ref(strategy_id, event_type, payload if isinstance(payload, Mapping) else {"event": payload}),
        "timestamp": _event_timestamp(event).isoformat(),
        "event": payload,
    }
    metadata = dict((payload.get("metadata") if isinstance(payload, Mapping) else {}) or {})
    for key in (
        "decision_ref",
        "action_ref",
        "portfolio_decision_ref",
        "provisional_order_ref",
        "intent_id",
        "idempotency_key",
        "broker_order_id",
        "original_order_id",
        "source_artifact_hash",
        "source_fingerprint",
        "candidate_hash",
        "portfolio_policy_hash",
        "kis_resource_plan_hash",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            row[key] = value
    if isinstance(payload, Mapping):
        for key in ("order_id", "symbol", "side", "qty", "price"):
            value = payload.get(key)
            if value not in (None, ""):
                row[key] = value
    return row


def _portfolio_context_payload(context: PortfolioContextProvider, *, reason: str) -> dict[str, Any]:
    cash = context.cash_equity()
    positions = []
    allocations = []
    symbol_exposures: dict[str, dict[str, Any]] = {}
    strategy_exposures: dict[str, dict[str, Any]] = {}
    sector_exposures: dict[str, dict[str, Any]] = {}
    for symbol, position in sorted(dict(context.positions or {}).items(), key=lambda item: str(item[0])):
        key = str(symbol).zfill(6)
        pos_payload = _position_snapshot_payload(key, position)
        positions.append(pos_payload)
        real_qty = max(int(pos_payload.get("real_qty") or 0), 0)
        avg_price = max(float(pos_payload.get("avg_price") or 0.0), 0.0)
        notional = real_qty * avg_price
        sector = str(context.sector_map.get(key, "UNKNOWN") or "UNKNOWN").upper().strip() or "UNKNOWN"
        symbol_exposures[key] = {
            "qty": real_qty,
            "avg_price": avg_price,
            "notional_krw": notional,
            "sector": sector,
            "allocation_drift": int(pos_payload.get("allocation_drift") or 0),
            "frozen": bool(pos_payload.get("frozen", False)),
        }
        sector_row = sector_exposures.setdefault(sector, {"qty": 0, "notional_krw": 0.0, "symbols_count": 0})
        sector_row["qty"] += real_qty
        sector_row["notional_krw"] += notional
        sector_row["symbols_count"] += 1 if real_qty > 0 else 0
        for strategy_id, allocation in sorted(dict(getattr(position, "allocations", {}) or {}).items()):
            allocation_payload = {"symbol": key, "strategy_id": str(strategy_id).upper().strip(), **_json_value(allocation)}
            allocation_qty = max(int(allocation_payload.get("qty") or 0), 0)
            allocation_price = max(float(allocation_payload.get("cost_basis") or avg_price or 0.0), 0.0)
            allocation_payload["notional_krw"] = allocation_qty * allocation_price
            allocation_payload["allocation_drift"] = int(pos_payload.get("allocation_drift") or 0)
            allocations.append(allocation_payload)
            strategy_row = strategy_exposures.setdefault(
                allocation_payload["strategy_id"],
                {"qty": 0, "notional_krw": 0.0, "symbols_count": 0},
            )
            strategy_row["qty"] += allocation_qty
            strategy_row["notional_krw"] += allocation_qty * allocation_price
            strategy_row["symbols_count"] += 1 if allocation_qty > 0 else 0
    exposure = context.portfolio_exposure()
    return {
        "record_type": "runtime_portfolio_context",
        "reason": reason,
        "account": _json_value(context.account_state),
        "equity_krw": cash.equity,
        "buyable_cash_krw": cash.cash,
        "gross_exposure_krw": exposure.notional,
        "gross_exposure_pct": exposure.notional / cash.equity if cash.equity > 0 else 0.0,
        "daily_pnl_krw": float(getattr(context.account_state, "daily_pnl", 0.0) or 0.0),
        "daily_pnl_pct": float(getattr(context.account_state, "daily_pnl_pct", 0.0) or 0.0),
        "positions": positions,
        "allocations": allocations,
        "positions_count": len(positions),
        "working_orders_count": sum(int(row.get("working_orders_count") or row.get("working_order_count") or 0) for row in positions),
        "allocation_drift_count": sum(1 for row in positions if int(row.get("allocation_drift") or 0) != 0 or bool(row.get("frozen", False))),
        "allocation_count": len(allocations),
        "sector_exposures": sector_exposures,
        "strategy_exposures": strategy_exposures,
        "symbol_exposures": symbol_exposures,
    }


def _position_snapshot_payload(symbol: str, position: Any) -> dict[str, Any]:
    payload = _json_value(position)
    data = dict(payload or {}) if isinstance(payload, Mapping) else {"value": payload}
    data["symbol"] = str(data.get("symbol") or symbol).zfill(6)
    allocations = dict(data.get("allocations") or {})
    real_qty = int(data.get("real_qty", data.get("qty", 0)) or 0)
    total_allocated = sum(int(dict(row or {}).get("qty") or 0) for row in allocations.values() if isinstance(row, Mapping))
    data["total_allocated_qty"] = total_allocated
    data["allocation_drift"] = real_qty - total_allocated
    data["working_orders_count"] = int(data.get("working_order_count", data.get("working_orders_count", 0)) or 0)
    if "_UNKNOWN_" in allocations:
        data["unknown_allocation"] = allocations["_UNKNOWN_"]
    return data


def _action_side_from_payload(payload: Mapping[str, Any]) -> str:
    action_type = str(payload.get("action_type") or "")
    if action_type == "SubmitEntry":
        return "BUY"
    if action_type in {"SubmitExit", "SubmitPartialExit", "FlattenPosition"}:
        return "SELL"
    return ""


def _route_result_strategy_id(result: object) -> str:
    action = getattr(result, "routed_action", None)
    return str(getattr(action, "strategy_id", "") or "").upper().strip()


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _ensure_aware(value).isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _json_value(item) for key, item in vars(value).items() if not str(key).startswith("_")}
    return value
