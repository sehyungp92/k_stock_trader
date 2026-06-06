from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, timezone
from enum import Enum
import time
from typing import Any, Iterable, Mapping, Sequence

from oms.intent import IntentStatus
from strategy_common.actions import (
    CancelOrders,
    FlattenPosition,
    ReplaceProtectiveStop,
    StrategyAction,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
    SubmitProtectiveStop,
    action_to_json_dict,
)
from strategy_common.events import DecisionEvent
from strategy_common.oms_adapter import action_to_intent

from .dry_run_oms import intent_to_json_dict
from .hashing import canonical_json_hash
from .portfolio import PortfolioArbitrationDecision, PortfolioArbitrationInput, PortfolioArbitrationPolicy
from .portfolio_context import PortfolioContextProvider
from .session_capture import PaperSessionRecorder


@dataclass(slots=True)
class RoutedOMSAdapter:
    strategy_id: str
    router: RuntimeActionRouter
    portfolio_adapter: Any | None = None
    dry_run: bool = False
    direct_submit_enabled: bool = False

    async def refresh_portfolio(self) -> Any:
        if self.portfolio_adapter is None or not callable(getattr(self.portfolio_adapter, "refresh_portfolio", None)):
            return None
        return await self.portfolio_adapter.refresh_portfolio()

    async def submit(self, action: StrategyAction) -> str | None:
        if not self.direct_submit_enabled:
            raise RuntimeError("direct routed OMS submission is non-promotional; use RuntimeSessionDriver.handle_*")
        return await self.router.submit_action(action)


@dataclass(slots=True)
class RuntimeActionRouter:
    """Canonical pre-OMS evidence path for paper/live and dry-run actions."""

    recorder: PaperSessionRecorder
    oms_client: Any
    portfolio_policy: PortfolioArbitrationPolicy | None = None
    portfolio_enabled: bool = True
    dry_run: bool = False
    pending_reservations: dict[str, "_PendingReservation"] = field(default_factory=dict)
    _reservation_aliases: dict[str, str] = field(default_factory=dict)
    portfolio_context_degraded: bool = False
    portfolio_context_degraded_reason: str = ""
    rehydrated_reservations_count: int = 0
    rehydrated_pending_notional: float = 0.0
    rehydrated_source_hash: str = ""

    def record_decisions(
        self,
        decisions: Iterable[DecisionEvent],
        *,
        event_ref: str = "",
        event_type: str = "",
    ) -> "DecisionRefIndex":
        refs: list[str] = []
        action_refs: dict[str, list[str]] = {}
        for event in decisions:
            payload = event.to_json_dict()
            payload["record_type"] = "decision_event"
            if event_ref:
                payload["event_ref"] = event_ref
            if event_type:
                payload["event_type"] = event_type
            payload["decision_ref"] = canonical_json_hash(payload)
            refs.append(payload["decision_ref"])
            for action_payload in payload.get("actions") or ():
                key = _decision_action_key_from_payload(action_payload)
                action_refs.setdefault(key, []).append(payload["decision_ref"])
            self.recorder.append_jsonl("decision_stream.jsonl", payload)
        return DecisionRefIndex(tuple(refs), {key: tuple(dict.fromkeys(value)) for key, value in action_refs.items()})

    def record_state_snapshot(self, strategy_id: str, state: Any, *, metadata: Mapping[str, Any] | None = None) -> str:
        metadata_payload = dict(metadata or {})
        decodable_state_required = _decodable_state_required(metadata_payload)
        state_payload = _json_safe_full_state(state) if decodable_state_required else _json_safe_state(state)
        payload = {
            "record_type": "state_snapshot",
            "strategy_id": str(strategy_id).upper().strip(),
            "metadata": metadata_payload,
            "state_encoding": _state_encoding(state, decodable=decodable_state_required),
            "state": state_payload,
        }
        payload["state_hash"] = canonical_json_hash(state_payload)
        self.recorder.append_jsonl("state_snapshots.jsonl", payload)
        return str(payload["state_hash"])

    def rehydrate_pending_reservations(
        self,
        working_orders: Iterable[Any],
        *,
        source: str = "oms",
        portfolio_context: PortfolioContextProvider | None = None,
    ) -> dict[str, Any]:
        if (
            portfolio_context is not None
            and portfolio_context.last_refresh_ts > 0
            and not portfolio_context.last_refresh_ok
        ):
            degraded_reason = portfolio_context.last_refresh_error or "oms_context_unavailable"
            count = sum(
                1
                for reservation in self.pending_reservations.values()
                if reservation.provenance.startswith("rehydrated:") and reservation.side == "BUY"
            )
            notional = sum(
                reservation.notional
                for reservation in self.pending_reservations.values()
                if reservation.provenance.startswith("rehydrated:") and reservation.side == "BUY"
            )
            self.portfolio_context_degraded = True
            self.portfolio_context_degraded_reason = degraded_reason
            self.rehydrated_reservations_count = count
            self.rehydrated_pending_notional = notional
            evidence = {
                "record_type": "pending_reservations_rehydrated",
                "source": source,
                "source_hash": self.rehydrated_source_hash,
                "pending_reservations_count": len(self.pending_reservations),
                "rehydrated_reservations_count": count,
                "rehydrated_pending_notional": notional,
                "degraded": True,
                "degraded_reason": degraded_reason,
                "oms_working_order_count": 0,
                "preserved_existing_reservations": True,
            }
            self.recorder.append_jsonl("portfolio_arbitration.jsonl", evidence)
            return evidence
        rows = [_working_order_reservation_row(order, portfolio_context=portfolio_context, source=source) for order in working_orders]
        normalized = [row for row in rows if row is not None]
        source_hash = canonical_json_hash(_rehydration_hash_payload(normalized))
        removed_rehydrated = 0
        for reservation_id, reservation in list(self.pending_reservations.items()):
            if reservation.provenance.startswith("rehydrated:"):
                self.pending_reservations.pop(reservation_id, None)
                removed_rehydrated += 1
                for alias in reservation.order_refs:
                    self._reservation_aliases.pop(alias, None)

        degraded_reason = ""
        count = 0
        notional = 0.0
        for row in normalized:
            if row["missing_price"] and row["side"] == "BUY":
                degraded_reason = "working_order_price_missing"
                continue
            reservation = _PendingReservation(
                reservation_id=row["reservation_id"],
                strategy_id=row["strategy_id"],
                symbol=row["symbol"],
                side=row["side"],
                qty=row["remaining_qty"],
                notional=row["notional"],
                sector=row["sector"],
                order_refs=tuple(row["order_refs"]),
                provenance=f"rehydrated:{source}",
                source_hash=source_hash,
                rehydrated_at=time.time(),
            )
            self.pending_reservations[reservation.reservation_id] = reservation
            for ref in reservation.order_refs:
                self._reservation_aliases[ref] = reservation.reservation_id
            if reservation.side == "BUY":
                count += 1
                notional += reservation.notional

        self.portfolio_context_degraded = bool(degraded_reason)
        self.portfolio_context_degraded_reason = degraded_reason
        self.rehydrated_reservations_count = count
        self.rehydrated_pending_notional = notional
        self.rehydrated_source_hash = source_hash
        evidence = {
            "record_type": "pending_reservations_rehydrated",
            "source": source,
            "source_hash": source_hash,
            "working_orders": _rehydration_hash_payload(normalized),
            "pending_reservations_count": len(self.pending_reservations),
            "rehydrated_reservations_count": count,
            "rehydrated_pending_notional": notional,
            "degraded": self.portfolio_context_degraded,
            "degraded_reason": self.portfolio_context_degraded_reason,
            "oms_working_order_count": len(normalized),
        }
        if normalized or removed_rehydrated or self.portfolio_context_degraded:
            self.recorder.append_jsonl("portfolio_arbitration.jsonl", evidence)
        return evidence

    async def submit_action(self, action: StrategyAction) -> str | None:
        results = await self.route_actions((_CollectedForRouting(action=action, provisional_order_ref="", batch_index=0),))
        return results[0].final_order_ref if results else None

    async def route_actions(
        self,
        collected_actions: Iterable[Any],
        *,
        portfolio_context: PortfolioContextProvider | None = None,
        event_ref: str = "",
        event_type: str = "",
        event_timestamp: datetime | None = None,
        decision_refs: "DecisionRefIndex | Sequence[str]" = (),
    ) -> tuple["RoutedActionResult", ...]:
        ref_index = _coerce_decision_ref_index(decision_refs)
        prepared_rows: list[tuple[_PreparedAction, _CollectedForRouting]] = []
        for index, raw in enumerate(collected_actions):
            collected = _coerce_collected_action(raw, index)
            action = _enrich_action_metadata(
                collected.action,
                event_ref=event_ref,
                event_type=event_type,
                event_timestamp=event_timestamp,
                provisional_order_ref=collected.provisional_order_ref,
                decision_ref=_decision_ref_for_action(collected.action, ref_index),
                event_decision_refs=ref_index.refs,
                batch_index=collected.batch_index,
            )
            prepared_rows.append((_prepare_action(action), collected))

        results: list[RoutedActionResult] = []
        reservations = _BatchReservations()
        for prepared, collected in sorted(prepared_rows, key=lambda item: self._routing_sort_key(item[0], item[1])):
            result = await self._route_prepared_action(prepared, collected, portfolio_context=portfolio_context, reservations=reservations)
            results.append(result)
        return tuple(results)

    def _routing_sort_key(self, prepared: "_PreparedAction", collected: "_CollectedForRouting") -> tuple[Any, ...]:
        if self.portfolio_enabled and self.portfolio_policy is not None:
            try:
                priority = self.portfolio_policy.config.strategy_priority.index(prepared.action.strategy_id.upper().strip())
            except ValueError:
                priority = len(self.portfolio_policy.config.strategy_priority)
            return (priority, _action_timestamp(prepared.metadata), str(prepared.action.symbol).zfill(6), collected.batch_index, prepared.action_ref)
        return (collected.batch_index, prepared.action_ref)

    async def _route_prepared_action(
        self,
        prepared: "_PreparedAction",
        collected: "_CollectedForRouting",
        *,
        portfolio_context: PortfolioContextProvider | None,
        reservations: "_BatchReservations",
    ) -> "RoutedActionResult":
        action_payload = action_to_json_dict(prepared.action)
        self.recorder.append_jsonl(
            "strategy_actions.jsonl",
            {
                "record_type": "strategy_action",
                "action_ref": prepared.action_ref,
                "event_ref": prepared.metadata.get("event_ref", ""),
                "event_type": prepared.metadata.get("event_type", ""),
                "provisional_order_ref": prepared.metadata.get("provisional_order_ref", ""),
                "decision_ref": prepared.metadata.get("decision_ref", ""),
                "event_decision_refs": prepared.metadata.get("event_decision_refs", ()),
                "batch_index": prepared.metadata.get("batch_index", collected.batch_index),
                "strategy_action_hash": prepared.strategy_action_hash,
                "strategy_id": prepared.action.strategy_id,
                "symbol": str(prepared.action.symbol).zfill(6),
                "action_type": type(prepared.action).__name__,
                "source_artifact_hash": prepared.metadata.get("source_artifact_hash", ""),
                "source_fingerprint": prepared.metadata.get("source_fingerprint", ""),
                "candidate_hash": prepared.metadata.get("candidate_hash", ""),
                "action": action_payload,
            },
        )

        decision, portfolio_item = self._portfolio_decision(prepared.action, prepared, portfolio_context=portfolio_context, reservations=reservations)
        routed_action = _resize_action(prepared.action, decision.final_qty) if decision.decision == "resized" else prepared.action
        routed_payload = action_to_json_dict(routed_action)
        portfolio_record = _portfolio_record(decision)
        if portfolio_item is not None:
            portfolio_record["sector"] = portfolio_item.sector
        portfolio_record["pending_reservations_count"] = len(self.pending_reservations)
        portfolio_record["rehydrated_reservations_count"] = self.rehydrated_reservations_count
        portfolio_record["rehydrated_pending_notional"] = self.rehydrated_pending_notional
        portfolio_record["rehydrated_source_hash"] = self.rehydrated_source_hash
        portfolio_record["portfolio_context_degraded"] = self.portfolio_context_degraded
        portfolio_record["portfolio_context_degraded_reason"] = self.portfolio_context_degraded_reason
        portfolio_record["event_ref"] = prepared.metadata.get("event_ref", "")
        portfolio_record["provisional_order_ref"] = prepared.metadata.get("provisional_order_ref", "")
        portfolio_record["original_action"] = action_to_json_dict(prepared.action)
        portfolio_record["original_strategy_action_hash"] = prepared.strategy_action_hash
        portfolio_record["routed_action"] = routed_payload
        portfolio_record["routed_strategy_action_hash"] = canonical_json_hash(portfolio_record["routed_action"])
        portfolio_record["portfolio_decision_ref"] = canonical_json_hash(portfolio_record)
        self.recorder.append_jsonl("portfolio_arbitration.jsonl", portfolio_record)
        if decision.decision == "blocked":
            self._export_missed_action(prepared, routed_payload, portfolio_record, blocked_scope="portfolio_rule")
            return RoutedActionResult(
                action_ref=prepared.action_ref,
                provisional_order_ref=str(prepared.metadata.get("provisional_order_ref") or ""),
                final_order_ref=None,
                portfolio_decision=decision.decision,
                portfolio_reason_code=decision.reason_code,
                accepted=False,
                blocked=True,
                resized=False,
                routed_action=routed_action,
                routed_action_payload=routed_payload,
                oms_status=None,
                oms_message="",
                resource_conflict_type=None,
                intent_id=None,
                broker_order_id=None,
                portfolio_decision_ref=portfolio_record["portfolio_decision_ref"],
            )

        intent = action_to_intent(routed_action)
        intent.metadata.update(
            {
                "action_ref": prepared.action_ref,
                "event_ref": prepared.metadata.get("event_ref", ""),
                "event_type": prepared.metadata.get("event_type", ""),
                "provisional_order_ref": prepared.metadata.get("provisional_order_ref", ""),
                "strategy_action_hash": prepared.strategy_action_hash,
                "source_artifact_hash": prepared.metadata.get("source_artifact_hash", ""),
                "source_fingerprint": prepared.metadata.get("source_fingerprint", ""),
                "candidate_hash": prepared.metadata.get("candidate_hash", ""),
                "decision_ref": prepared.metadata.get("decision_ref", ""),
                "event_decision_refs": prepared.metadata.get("event_decision_refs", ()),
                "portfolio_decision_ref": portfolio_record["portfolio_decision_ref"],
                "portfolio_decision": decision.decision,
                "portfolio_reason_code": decision.reason_code,
                "portfolio_policy_hash": decision.policy_hash,
            }
        )
        intent_payload = intent_to_json_dict(
            intent,
            dry_run=self.dry_run,
            intended_broker_submit=not self.dry_run,
            submitted_to_broker=False,
            actually_submitted_to_broker=False,
            oms_status="PENDING_SUBMIT",
        )
        intent_payload["portfolio_decision_ref"] = portfolio_record["portfolio_decision_ref"]
        self.recorder.append_jsonl("oms_intents.jsonl", intent_payload)
        result = await self.oms_client.submit_intent(intent)
        status = getattr(result, "status", None)
        self.recorder.append_jsonl(
            "order_events.jsonl",
            _intent_result_record(result, intent_payload=intent_payload, dry_run=self.dry_run),
        )
        if status in {IntentStatus.REJECTED, IntentStatus.CANCELLED, IntentStatus.DEFERRED}:
            self._export_missed_action(
                prepared,
                routed_payload,
                {
                    **portfolio_record,
                    "oms_status": getattr(status, "name", str(status or "")),
                    "intent_id": getattr(result, "intent_id", None),
                    "blocking_positions": getattr(result, "blocking_positions", None),
                    "resource_conflict_type": getattr(result, "resource_conflict_type", None),
                    "message": getattr(result, "message", ""),
                },
                blocked_scope="oms_risk" if status == IntentStatus.REJECTED else "execution",
            )
        if status in {IntentStatus.REJECTED, IntentStatus.CANCELLED, IntentStatus.DEFERRED}:
            final_ref = None
        else:
            final_ref = getattr(result, "order_id", None) or getattr(result, "intent_id", None)
        result = RoutedActionResult(
            action_ref=prepared.action_ref,
            provisional_order_ref=str(prepared.metadata.get("provisional_order_ref") or ""),
            final_order_ref=final_ref,
            portfolio_decision=decision.decision,
            portfolio_reason_code=decision.reason_code,
            accepted=final_ref is not None,
            blocked=False,
            resized=decision.decision == "resized",
            routed_action=routed_action,
            routed_action_payload=routed_payload,
            oms_status=getattr(status, "name", str(status or "")),
            oms_message=getattr(result, "message", ""),
            resource_conflict_type=getattr(result, "resource_conflict_type", None),
            intent_id=getattr(result, "intent_id", None),
            broker_order_id=getattr(result, "order_id", None),
            portfolio_decision_ref=portfolio_record["portfolio_decision_ref"],
        )
        if result.accepted and portfolio_item is not None:
            self._remember_pending_reservation(result, portfolio_item, decision)
        return result

    def _export_missed_action(
        self,
        prepared: "_PreparedAction",
        action_payload: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        blocked_scope: str,
    ) -> None:
        exporter = getattr(self.recorder, "assistant_exporter", None)
        writer = getattr(exporter, "writer", None)
        if writer is None:
            return
        try:
            metadata = dict(prepared.metadata or {})
            payload = {
                "record_type": "missed_opportunity",
                "event_type": "missed_opportunity",
                "schema_version": "missed_opportunity_v2",
                "logical_event_id": f"{prepared.action.strategy_id}:{str(prepared.action.symbol).zfill(6)}:{metadata.get('event_ref', '')}:{prepared.action_ref}",
                "revision": 0,
                "strategy_id": prepared.action.strategy_id,
                "pair": str(prepared.action.symbol).zfill(6),
                "side": "LONG" if _action_side(prepared.action) == "BUY" else "EXIT",
                "signal": str(getattr(prepared.action, "reason", "") or ""),
                "signal_id": metadata.get("candidate_hash") or prepared.action_ref,
                "signal_strength": float(metadata.get("candidate_score") or metadata.get("signal_strength") or 0.0),
                "signal_time": _action_timestamp(metadata).isoformat(),
                "blocked_by": str(context.get("reason_code") or context.get("oms_status") or "blocked"),
                "block_reason": str(context.get("message") or context.get("reason_code") or ""),
                "blocked_scope": blocked_scope,
                "event_ref": metadata.get("event_ref", ""),
                "decision_ref": metadata.get("decision_ref", ""),
                "action_ref": prepared.action_ref,
                "provisional_order_ref": metadata.get("provisional_order_ref", ""),
                "portfolio_decision_ref": context.get("portfolio_decision_ref", ""),
                "intent_id": context.get("intent_id", ""),
                "blocking_positions": context.get("blocking_positions"),
                "resource_conflict_type": context.get("resource_conflict_type", ""),
                "action": dict(action_payload),
                "portfolio_context": dict(context),
            }
            writer.write(
                "missed_opportunity",
                payload,
                payload_key=f"{payload['logical_event_id']}:rev:0",
                exchange_timestamp=payload["signal_time"],
                lineage={
                    "strategy_id": prepared.action.strategy_id,
                    "artifact_hash": metadata.get("source_artifact_hash", ""),
                    "source_fingerprint": metadata.get("source_fingerprint", ""),
                    "candidate_hash": metadata.get("candidate_hash", ""),
                    "portfolio_policy_hash": context.get("policy_hash", ""),
                },
                logical_event_id=payload["logical_event_id"],
                revision=0,
                scope="strategy",
            )
        except Exception:
            return

    def pending_reservations_for(
        self,
        item: PortfolioArbitrationInput,
        *,
        portfolio_context: PortfolioContextProvider | None = None,
    ) -> "_ReservationTotals":
        totals = _ReservationTotals()
        for reservation in self.pending_reservations.values():
            if reservation.side == "BUY":
                if (
                    reservation.provenance.startswith("rehydrated:")
                    and _reservation_reflected_in_context(reservation, portfolio_context)
                ):
                    continue
                totals.admitted_gross += reservation.notional
                if reservation.symbol == item.symbol:
                    totals.admitted_symbol += reservation.notional
                if reservation.sector == item.sector:
                    totals.admitted_sector += reservation.notional
            elif reservation.side == "SELL" and reservation.strategy_id == item.strategy_id and reservation.symbol == item.symbol:
                totals.admitted_exit_notional += reservation.notional
                totals.admitted_exit_qty += reservation.qty
        return totals

    def release_order_ref(self, order_ref: str | None, *, qty: int | None = None) -> bool:
        raw = str(order_ref or "")
        if not raw:
            return False
        reservation_id = self._reservation_aliases.get(raw) or (raw if raw in self.pending_reservations else "")
        if not reservation_id:
            return False
        reservation = self.pending_reservations.get(reservation_id)
        if reservation is None:
            return False
        release_qty = int(qty) if qty is not None else reservation.qty
        if qty is not None and release_qty <= 0:
            return False
        if 0 < release_qty < reservation.qty:
            unit_notional = reservation.notional / max(reservation.qty, 1)
            self.pending_reservations[reservation_id] = replace(
                reservation,
                qty=reservation.qty - release_qty,
                notional=max(0.0, unit_notional * (reservation.qty - release_qty)),
            )
            return True
        self.pending_reservations.pop(reservation_id, None)
        for alias in reservation.order_refs:
            self._reservation_aliases.pop(alias, None)
        return True

    def _remember_pending_reservation(
        self,
        result: "RoutedActionResult",
        item: PortfolioArbitrationInput,
        decision: PortfolioArbitrationDecision,
    ) -> None:
        if item.side not in {"BUY", "SELL"} or result.final_order_ref is None or decision.final_qty <= 0:
            return
        reservation_id = str(result.provisional_order_ref or result.action_ref or result.final_order_ref)
        if not reservation_id:
            return
        refs = tuple(
            dict.fromkeys(
                str(raw)
                for raw in (
                    reservation_id,
                    result.action_ref,
                    result.provisional_order_ref,
                    result.final_order_ref,
                    result.intent_id,
                    result.broker_order_id,
                )
                if raw not in (None, "")
            )
        )
        qty = int(decision.final_qty)
        reservation = _PendingReservation(
            reservation_id=reservation_id,
            strategy_id=item.strategy_id,
            symbol=item.symbol,
            side=item.side,
            qty=qty,
            notional=float(decision.final_notional),
            sector=item.sector,
            order_refs=refs,
        )
        self.pending_reservations[reservation_id] = reservation
        for ref in refs:
            self._reservation_aliases[ref] = reservation_id

    def _portfolio_decision(
        self,
        action: StrategyAction,
        prepared: "_PreparedAction",
        *,
        portfolio_context: PortfolioContextProvider | None = None,
        reservations: "_BatchReservations",
    ) -> tuple[PortfolioArbitrationDecision, PortfolioArbitrationInput | None]:
        side = _action_side(action)
        if not self.portfolio_enabled or self.portfolio_policy is None:
            return _pass_through_decision(action, prepared, "portfolio_disabled", ""), None
        if side is None:
            return _pass_through_decision(action, prepared, "operational_pass_through", self.portfolio_policy.policy_hash), None
        if portfolio_context is None:
            raise ValueError("portfolio_context is required for routed BUY/SELL arbitration")
        item = _portfolio_input(action, prepared, side, portfolio_context=portfolio_context)
        if item.side == "BUY" and self.portfolio_context_degraded:
            return PortfolioArbitrationDecision(
                action_ref=item.action_ref,
                strategy_id=item.strategy_id,
                symbol=item.symbol,
                decision="blocked",
                final_qty=0,
                final_notional=0.0,
                reason_code=self.portfolio_context_degraded_reason or "portfolio_context_degraded",
                policy_hash=self.portfolio_policy.policy_hash,
                source_artifact_hashes=item.source_artifact_hashes,
                timestamp=item.timestamp,
            ), item
        pending = self.pending_reservations_for(item, portfolio_context=portfolio_context)
        decision = self.portfolio_policy.decide_one(
            item,
            admitted_gross=reservations.admitted_gross + pending.admitted_gross,
            admitted_symbol=reservations.admitted_buy_symbol.get(item.symbol, 0.0) + pending.admitted_symbol,
            admitted_sector=reservations.admitted_sector.get(item.sector, 0.0) + pending.admitted_sector,
            admitted_exit_notional=reservations.admitted_exit_strategy_symbol.get((item.strategy_id, item.symbol), 0.0) + pending.admitted_exit_notional,
            admitted_exit_qty=reservations.admitted_exit_qty.get((item.strategy_id, item.symbol), 0) + pending.admitted_exit_qty,
        )
        if decision.decision in {"accepted", "resized"} and decision.final_notional > 0:
            if item.side == "BUY":
                reservations.admitted_gross += decision.final_notional
                reservations.admitted_buy_symbol[item.symbol] = reservations.admitted_buy_symbol.get(item.symbol, 0.0) + decision.final_notional
                reservations.admitted_sector[item.sector] = reservations.admitted_sector.get(item.sector, 0.0) + decision.final_notional
            elif item.side == "SELL":
                key = (item.strategy_id, item.symbol)
                reservations.admitted_exit_strategy_symbol[key] = reservations.admitted_exit_strategy_symbol.get(key, 0.0) + decision.final_notional
                reservations.admitted_exit_qty[key] = reservations.admitted_exit_qty.get(key, 0) + int(decision.final_qty)
        return decision, item


@dataclass(frozen=True, slots=True)
class DecisionRefIndex:
    refs: tuple[str, ...] = ()
    action_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoutedActionResult:
    action_ref: str
    provisional_order_ref: str
    final_order_ref: str | None
    portfolio_decision: str
    portfolio_reason_code: str
    accepted: bool
    blocked: bool
    resized: bool
    routed_action: StrategyAction
    routed_action_payload: dict[str, Any]
    oms_status: str | None = None
    oms_message: str = ""
    resource_conflict_type: str | None = None
    intent_id: str | None = None
    broker_order_id: str | None = None
    portfolio_decision_ref: str = ""


@dataclass(frozen=True, slots=True)
class _PendingReservation:
    reservation_id: str
    strategy_id: str
    symbol: str
    side: str
    qty: int
    notional: float
    sector: str
    order_refs: tuple[str, ...]
    provenance: str = "local"
    source_hash: str = ""
    rehydrated_at: float = 0.0


@dataclass(slots=True)
class _ReservationTotals:
    admitted_gross: float = 0.0
    admitted_symbol: float = 0.0
    admitted_sector: float = 0.0
    admitted_exit_notional: float = 0.0
    admitted_exit_qty: int = 0


@dataclass(slots=True)
class _BatchReservations:
    admitted_gross: float = 0.0
    admitted_buy_symbol: dict[str, float] = field(default_factory=dict)
    admitted_sector: dict[str, float] = field(default_factory=dict)
    admitted_exit_strategy_symbol: dict[tuple[str, str], float] = field(default_factory=dict)
    admitted_exit_qty: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PreparedAction:
    action: StrategyAction
    action_ref: str
    strategy_action_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CollectedForRouting:
    action: StrategyAction
    provisional_order_ref: str
    batch_index: int


def _prepare_action(action: StrategyAction) -> _PreparedAction:
    base_payload = action_to_json_dict(action)
    strategy_action_hash = canonical_json_hash(base_payload)
    action_ref = str(action.metadata.get("action_ref") or strategy_action_hash[:16])
    metadata = {
        **dict(action.metadata or {}),
        "action_ref": action_ref,
        "strategy_action_hash": strategy_action_hash,
    }
    return _PreparedAction(replace(action, metadata=metadata), action_ref, strategy_action_hash, metadata)


def _coerce_collected_action(raw: Any, index: int) -> _CollectedForRouting:
    if isinstance(raw, _CollectedForRouting):
        return raw
    action = getattr(raw, "action", raw)
    provisional = str(getattr(raw, "provisional_order_ref", "") or "")
    batch_index = int(getattr(raw, "batch_index", index) or index)
    return _CollectedForRouting(action=action, provisional_order_ref=provisional, batch_index=batch_index)


def _enrich_action_metadata(
    action: StrategyAction,
    *,
    event_ref: str,
    event_type: str,
    event_timestamp: datetime | None,
    provisional_order_ref: str,
    decision_ref: str,
    event_decision_refs: Sequence[str],
    batch_index: int,
) -> StrategyAction:
    metadata = dict(action.metadata or {})
    if event_ref:
        metadata.setdefault("event_ref", event_ref)
    if event_type:
        metadata.setdefault("event_type", event_type)
    if event_timestamp is not None:
        if not metadata.get("timestamp"):
            metadata["timestamp"] = event_timestamp.isoformat()
        if not metadata.get("decision_time"):
            metadata["decision_time"] = event_timestamp.isoformat()
    if provisional_order_ref:
        metadata.setdefault("provisional_order_ref", provisional_order_ref)
    if decision_ref:
        metadata.setdefault("decision_ref", decision_ref)
    if event_decision_refs:
        metadata.setdefault("event_decision_refs", tuple(event_decision_refs))
    metadata.setdefault("batch_index", batch_index)
    return replace(action, metadata=metadata)


def _decision_ref_for_action(action: StrategyAction, decision_refs: DecisionRefIndex) -> str:
    refs = decision_refs.action_refs.get(_decision_action_key(action), ())
    if len(refs) == 1:
        return str(refs[0])
    existing = str((action.metadata or {}).get("decision_ref") or "")
    if existing:
        return existing
    if len(decision_refs.refs) == 1:
        return str(decision_refs.refs[0])
    return ""


def _coerce_decision_ref_index(value: DecisionRefIndex | Sequence[str]) -> DecisionRefIndex:
    if isinstance(value, DecisionRefIndex):
        return value
    return DecisionRefIndex(tuple(str(item) for item in value))


_WORKING_ORDER_TERMINAL_STATUSES = {"FILLED", "CANCELLED", "REJECTED", "EXPIRED", "FAILED"}


def _working_order_reservation_row(
    order: Any,
    *,
    portfolio_context: PortfolioContextProvider | None,
    source: str,
) -> dict[str, Any] | None:
    data = dict(order) if isinstance(order, Mapping) else {
        name: getattr(order, name)
        for name in (
            "order_id",
            "symbol",
            "side",
            "qty",
            "filled_qty",
            "remaining_qty",
            "price",
            "status",
            "strategy_id",
            "intent_id",
            "idempotency_key",
            "submit_ref",
            "risk_stop_px",
            "risk_hard_stop_px",
            "broker_order_id",
            "created_at",
        )
        if hasattr(order, name)
    }
    status = str(data.get("status") or "").upper().strip()
    if status in _WORKING_ORDER_TERMINAL_STATUSES:
        return None
    side = str(data.get("side") or "").upper().strip()
    if side not in {"BUY", "SELL"}:
        return None
    symbol = str(data.get("symbol") or "").zfill(6)
    qty = _int_or_zero(data.get("qty"))
    filled_qty = _int_or_zero(data.get("filled_qty"))
    remaining_qty = _int_or_zero(data.get("remaining_qty"))
    if remaining_qty <= 0:
        remaining_qty = max(qty - filled_qty, 0)
    if remaining_qty <= 0:
        return None
    price = _float_or_zero(data.get("price"))
    missing_price = price <= 0
    strategy_id = str(data.get("strategy_id") or "").upper().strip()
    refs = tuple(
        dict.fromkeys(
            str(raw)
            for raw in (
                data.get("order_id"),
                data.get("broker_order_id"),
                data.get("intent_id"),
                data.get("idempotency_key"),
                data.get("submit_ref"),
            )
            if raw not in (None, "")
        )
    )
    reservation_id = str(refs[0]) if refs else f"rehydrated:{source}:{canonical_json_hash(data)[:16]}"
    sector = "UNKNOWN"
    if portfolio_context is not None:
        sector = str(portfolio_context.sector_map.get(symbol, "UNKNOWN") or "UNKNOWN").upper().strip() or "UNKNOWN"
    return {
        "reservation_id": f"rehydrated:{source}:{reservation_id}",
        "strategy_id": strategy_id,
        "symbol": symbol,
        "side": side,
        "remaining_qty": remaining_qty,
        "notional": 0.0 if missing_price else remaining_qty * price,
        "price": price,
        "missing_price": missing_price,
        "status": status,
        "sector": sector,
        "order_refs": refs or (reservation_id,),
        "intent_id": data.get("intent_id"),
        "idempotency_key": data.get("idempotency_key"),
    }


def _reservation_reflected_in_context(
    reservation: _PendingReservation,
    portfolio_context: PortfolioContextProvider | None,
) -> bool:
    if portfolio_context is None or reservation.side != "BUY":
        return False
    reservation_refs = {str(ref) for ref in reservation.order_refs if ref not in (None, "")}
    if not reservation_refs:
        return False
    for order in portfolio_context.iter_working_orders():
        if str(getattr(order, "side", "") or "").upper().strip() != "BUY":
            continue
        if str(getattr(order, "status", "") or "").upper().strip() in _WORKING_ORDER_TERMINAL_STATUSES:
            continue
        remaining_qty = _int_or_zero(getattr(order, "remaining_qty", 0))
        if remaining_qty <= 0:
            remaining_qty = max(_int_or_zero(getattr(order, "qty", 0)) - _int_or_zero(getattr(order, "filled_qty", 0)), 0)
        if remaining_qty <= 0:
            continue
        refs = {
            str(raw)
            for raw in (
                getattr(order, "order_id", None),
                getattr(order, "intent_id", None),
                getattr(order, "idempotency_key", None),
                getattr(order, "submit_ref", None),
            )
            if raw not in (None, "")
        }
        if refs & reservation_refs:
            return True
    return False


def _rehydration_hash_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "strategy_id": row.get("strategy_id"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "remaining_qty": row.get("remaining_qty"),
                "price": row.get("price"),
                "status": row.get("status"),
                "sector": row.get("sector"),
                "intent_id": row.get("intent_id"),
                "idempotency_key": row.get("idempotency_key"),
                "missing_price": row.get("missing_price"),
            }
        )
    return sorted(payload, key=lambda item: (str(item.get("strategy_id")), str(item.get("symbol")), str(item.get("side"))))


def _decision_action_key(action: StrategyAction) -> str:
    return _decision_action_key_from_payload(action_to_json_dict(action))


def _decision_action_key_from_payload(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload or {})
    metadata = dict(normalized.get("metadata") or {})
    for key in (
        "action_ref",
        "batch_index",
        "decision_ref",
        "decision_time",
        "event_decision_refs",
        "event_ref",
        "event_type",
        "provisional_order_ref",
        "source_artifact_hash",
        "source_fingerprint",
        "strategy_action_hash",
        "timestamp",
    ):
        metadata.pop(key, None)
    normalized["metadata"] = metadata
    return canonical_json_hash(normalized)


def _action_side(action: StrategyAction) -> str | None:
    if isinstance(action, SubmitEntry):
        return "BUY"
    if isinstance(action, (SubmitExit, SubmitPartialExit, FlattenPosition)):
        return "SELL"
    if isinstance(action, (CancelOrders, SubmitProtectiveStop, ReplaceProtectiveStop)):
        return None
    raise TypeError(f"unsupported strategy action {type(action).__name__}")


def _portfolio_input(
    action: StrategyAction,
    prepared: _PreparedAction,
    side: str,
    *,
    portfolio_context: PortfolioContextProvider | None = None,
) -> PortfolioArbitrationInput:
    metadata = prepared.metadata
    symbol = str(action.symbol).zfill(6)
    sector = str(
        metadata.get("sector")
        or ((portfolio_context.sector_map.get(symbol) if portfolio_context is not None else "") or "")
        or "UNKNOWN"
    ).upper().strip() or "UNKNOWN"
    strategy_exposure = portfolio_context.strategy_exposure(action.strategy_id, action.symbol) if portfolio_context is not None else None
    symbol_exposure = portfolio_context.symbol_exposure(action.symbol) if portfolio_context is not None else None
    portfolio_exposure = portfolio_context.portfolio_exposure() if portfolio_context is not None else None
    sector_exposure = portfolio_context.sector_exposure(sector) if portfolio_context is not None else None
    cash_equity = portfolio_context.cash_equity() if portfolio_context is not None else None
    qty = _action_qty(action, metadata)
    if side == "SELL" and qty <= 0 and strategy_exposure is not None:
        qty = strategy_exposure.qty
    notional = _action_notional(action, metadata, qty)
    if side == "SELL" and notional <= 0 and strategy_exposure is not None and strategy_exposure.qty > 0:
        notional = strategy_exposure.notional * (qty / strategy_exposure.qty)
    timestamp = _action_timestamp(metadata)
    source_hash = str(metadata.get("source_artifact_hash") or "")
    return PortfolioArbitrationInput(
        action_ref=prepared.action_ref,
        strategy_id=action.strategy_id,
        symbol=action.symbol,
        side=side,
        intended_qty=qty,
        intended_notional=notional,
        timestamp=timestamp,
        sector=sector,
        candidate_rank=_int_or_zero(metadata.get("candidate_rank")),
        candidate_score_band=str(metadata.get("candidate_score_band") or ""),
        route_family=str(metadata.get("route_family") or ""),
        current_strategy_exposure=(
            strategy_exposure.notional if strategy_exposure is not None else _float_or_zero(metadata.get("current_strategy_exposure"))
        ),
        current_portfolio_exposure=(
            portfolio_exposure.notional if portfolio_exposure is not None else _float_or_zero(metadata.get("current_portfolio_exposure"))
        ),
        current_symbol_exposure=symbol_exposure.notional if symbol_exposure is not None else _float_or_zero(metadata.get("current_symbol_exposure")),
        current_sector_exposure=sector_exposure.notional if sector_exposure is not None else _float_or_zero(metadata.get("current_sector_exposure")),
        current_strategy_symbol_qty=strategy_exposure.qty if strategy_exposure is not None else _int_or_zero(metadata.get("current_strategy_symbol_qty")),
        current_strategy_symbol_notional=(
            strategy_exposure.notional if strategy_exposure is not None else _float_or_zero(metadata.get("current_strategy_symbol_notional"))
        ),
        current_symbol_qty=symbol_exposure.qty if symbol_exposure is not None else _int_or_zero(metadata.get("current_symbol_qty")),
        current_symbol_notional=symbol_exposure.notional if symbol_exposure is not None else _float_or_zero(metadata.get("current_symbol_notional")),
        cash=cash_equity.cash if cash_equity is not None else _float_or_zero(metadata.get("cash")),
        equity=cash_equity.equity if cash_equity is not None else _float_or_zero(metadata.get("equity")),
        source_artifact_hashes=(source_hash,) if source_hash else (),
        metadata=metadata,
    )


def _pass_through_decision(
    action: StrategyAction,
    prepared: _PreparedAction,
    reason: str,
    policy_hash: str,
) -> PortfolioArbitrationDecision:
    return PortfolioArbitrationDecision(
        action_ref=prepared.action_ref,
        strategy_id=action.strategy_id,
        symbol=str(action.symbol).zfill(6),
        decision="accepted",
        final_qty=_action_qty(action, prepared.metadata, allow_zero=True),
        final_notional=_action_notional(action, prepared.metadata, _action_qty(action, prepared.metadata, allow_zero=True)),
        reason_code=reason,
        policy_hash=policy_hash,
        source_artifact_hashes=tuple(filter(None, (str(prepared.metadata.get("source_artifact_hash") or ""),))),
        timestamp=_action_timestamp(prepared.metadata),
    )


def _portfolio_record(decision: PortfolioArbitrationDecision) -> dict[str, Any]:
    payload = decision.to_json_dict()
    payload["record_type"] = "portfolio_arbitration"
    return payload


def _resize_action(action: StrategyAction, final_qty: int) -> StrategyAction:
    if final_qty <= 0:
        return action
    if isinstance(action, (SubmitEntry, SubmitPartialExit)):
        return replace(action, qty=final_qty)
    if isinstance(action, SubmitExit):
        return replace(action, qty=final_qty)
    return action


def _intent_result_record(result: Any, *, intent_payload: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    status = getattr(result, "status", None)
    status_name = getattr(status, "name", str(status or ""))
    broker_order_id = getattr(result, "order_id", None)
    actually_submitted = bool((not dry_run) and broker_order_id)
    return {
        "record_type": "dry_run_order_result" if dry_run else "oms_order_result",
        "dry_run": bool(dry_run),
        "intended_broker_submit": bool(intent_payload.get("intended_broker_submit", not bool(dry_run))),
        "actually_submitted_to_broker": actually_submitted,
        "submitted_to_broker": actually_submitted,
        "event_ref": intent_payload.get("event_ref") or (intent_payload.get("metadata") or {}).get("event_ref", ""),
        "action_ref": intent_payload.get("action_ref", ""),
        "provisional_order_ref": (intent_payload.get("metadata") or {}).get("provisional_order_ref", ""),
        "portfolio_decision_ref": intent_payload.get("portfolio_decision_ref", ""),
        "intent_id": getattr(result, "intent_id", None),
        "order_id": broker_order_id,
        "status": status_name,
        "message": getattr(result, "message", ""),
        "modified_qty": getattr(result, "modified_qty", None),
        "cooldown_until": getattr(result, "cooldown_until", None),
        "blocking_positions": getattr(result, "blocking_positions", None),
        "resource_conflict_type": getattr(result, "resource_conflict_type", None),
        "oms_received_at": getattr(result, "oms_received_at", None),
        "order_submitted_at": getattr(result, "order_submitted_at", None),
    }


def _action_qty(action: StrategyAction, metadata: Mapping[str, Any], *, allow_zero: bool = False) -> int:
    raw = getattr(action, "qty", None)
    if raw is None:
        raw = metadata.get("current_strategy_qty") or metadata.get("position_qty") or metadata.get("qty")
    qty = _int_or_zero(raw)
    if qty <= 0 and not allow_zero:
        return 0
    return max(qty, 0)


def _action_notional(action: StrategyAction, metadata: Mapping[str, Any], qty: int) -> float:
    for key in ("intended_notional", "target_notional", "estimated_notional", "notional"):
        value = _float_or_zero(metadata.get(key))
        if value > 0:
            return value
    price = 0.0
    for key in ("limit_price", "stop_price"):
        value = _float_or_zero(getattr(action, key, None))
        if value > 0:
            price = value
            break
    if price <= 0:
        for key in ("estimated_price", "entry_price", "entry_submission_close", "last_price", "close"):
            value = _float_or_zero(metadata.get(key))
            if value > 0:
                price = value
                break
    if qty > 0 and price > 0:
        return float(qty) * price
    return _float_or_zero(metadata.get("current_strategy_exposure"))


def _action_timestamp(metadata: Mapping[str, Any]) -> datetime:
    for key in ("timestamp", "decision_time", "bar_timestamp", "entry_submission_timestamp"):
        raw = metadata.get(key)
        if isinstance(raw, datetime):
            return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        if raw not in (None, ""):
            try:
                parsed = datetime.fromisoformat(str(raw))
            except ValueError:
                continue
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _json_safe_state(state: Any) -> Any:
    if state is None:
        return {}
    if state.__class__.__name__ == "KALCBState":
        return _compact_kalcb_state(state)
    if state.__class__.__name__ == "OLRState":
        return _compact_olr_state(state)
    if hasattr(state, "to_json_dict"):
        return state.to_json_dict()
    if hasattr(state, "__dataclass_fields__"):
        return asdict(state)
    if isinstance(state, Mapping):
        return dict(state)
    return repr(state)


def _json_safe_full_state(state: Any) -> Any:
    if state is None:
        return {}
    if state.__class__.__name__ == "KALCBState":
        from strategy_kalcb.core.serializers import snapshot_state

        return snapshot_state(state)
    if state.__class__.__name__ == "OLRState":
        from strategy_olr.core.serializers import snapshot_state

        return snapshot_state(state)
    return _json_safe_state(state)


def _decodable_state_required(metadata: Mapping[str, Any]) -> bool:
    return str(metadata.get("record_reason") or "").endswith("pre_start")


def _state_encoding(state: Any, *, decodable: bool = False) -> str:
    if state is None:
        return "empty"
    name = state.__class__.__name__
    if decodable and name == "KALCBState":
        return "kalcb-state-decodable-v1"
    if decodable and name == "OLRState":
        return "olr-state-decodable-v1"
    if name == "KALCBState":
        return "kalcb-state-compact-v1"
    if name == "OLRState":
        return "olr-state-compact-v1"
    return "json-safe-full"


def _compact_kalcb_state(state: Any) -> dict[str, Any]:
    symbols = {
        str(symbol).zfill(6): _compact_kalcb_symbol(symbol_state)
        for symbol, symbol_state in sorted(dict(getattr(state, "symbols", {}) or {}).items())
    }
    return {
        "schema": "kalcb-state-compact-v1",
        "snapshot_hash": str(getattr(state, "snapshot_hash", "") or ""),
        "source_fingerprint": str(getattr(state, "source_fingerprint", "") or ""),
        "session_date": _json_value(getattr(state, "session_date", None)),
        "order_roles": _digest_payload(getattr(state, "order_roles", {}) or {}),
        "meta": _digest_payload(getattr(state, "meta", {}) or {}),
        "symbol_count": len(symbols),
        "symbols_hash": canonical_json_hash(symbols),
        "stage_counts": _stage_counts(symbols),
        "position_symbols": _symbols_matching(symbols, lambda item: item.get("position") is not None),
        "pending_symbols": _symbols_matching(symbols, lambda item: bool(item.get("pending_entry_order_id"))),
    }


def _compact_kalcb_symbol(symbol_state: Any) -> dict[str, Any]:
    bars = list(getattr(symbol_state, "bars", []) or [])
    candidate = getattr(symbol_state, "candidate", None)
    position = getattr(symbol_state, "position", None)
    return {
        "symbol": str(getattr(symbol_state, "symbol", "") or "").zfill(6),
        "stage": _enum_value(getattr(symbol_state, "stage", "")),
        "candidate": _candidate_ref(candidate),
        "candidate_rank": int(getattr(symbol_state, "candidate_rank", 0) or 0),
        "session_date": _json_value(getattr(symbol_state, "session_date", None)),
        "bars_count": len(bars),
        "last_bar_hash": _bar_hash(bars[-1]) if bars else "",
        "opening_range_built": bool(getattr(symbol_state, "opening_range_built", False)),
        "or_high": _float_or_zero(getattr(symbol_state, "or_high", 0.0)),
        "or_low": _float_or_zero(getattr(symbol_state, "or_low", 0.0)),
        "or_volume": _float_or_zero(getattr(symbol_state, "or_volume", 0.0)),
        "vwap_value": _float_or_zero(getattr(symbol_state, "vwap_value", 0.0)),
        "vwap_volume": _float_or_zero(getattr(symbol_state, "vwap_volume", 0.0)),
        "pending_entry_order_id": str(getattr(symbol_state, "pending_entry_order_id", "") or ""),
        "pending_entry_metadata": _digest_payload(getattr(symbol_state, "pending_entry_metadata", {}) or {}),
        "touched_vwap": bool(getattr(symbol_state, "touched_vwap", False)),
        "touched_or_mid": bool(getattr(symbol_state, "touched_or_mid", False)),
        "touched_or_high": bool(getattr(symbol_state, "touched_or_high", False)),
        "touched_pdh": bool(getattr(symbol_state, "touched_pdh", False)),
        "touched_reclaim_levels": _digest_payload(getattr(symbol_state, "touched_reclaim_levels", {}) or {}),
        "position": _compact_position(position),
        "rejected_reason": str(getattr(symbol_state, "rejected_reason", "") or ""),
        "entry_attempted": bool(getattr(symbol_state, "entry_attempted", False)),
        "last_decision_code": str(getattr(symbol_state, "last_decision_code", "") or ""),
        "last_decision_details": _digest_payload(getattr(symbol_state, "last_decision_details", {}) or {}),
    }


def _compact_olr_state(state: Any) -> dict[str, Any]:
    symbols = {
        str(symbol).zfill(6): _compact_olr_symbol(symbol_state)
        for symbol, symbol_state in sorted(dict(getattr(state, "symbols", {}) or {}).items())
    }
    return {
        "schema": "olr-state-compact-v1",
        "snapshot_hash": str(getattr(state, "snapshot_hash", "") or ""),
        "source_fingerprint": str(getattr(state, "source_fingerprint", "") or ""),
        "session_date": _json_value(getattr(state, "session_date", None)),
        "order_roles": _digest_payload(getattr(state, "order_roles", {}) or {}),
        "meta": _digest_payload(getattr(state, "meta", {}) or {}),
        "symbol_count": len(symbols),
        "symbols_hash": canonical_json_hash(symbols),
        "stage_counts": _stage_counts(symbols),
        "position_symbols": _symbols_matching(symbols, lambda item: item.get("position") is not None),
        "pending_symbols": _symbols_matching(
            symbols,
            lambda item: bool(item.get("pending_entry_order_id") or item.get("pending_exit_order_id")),
        ),
    }


def _compact_olr_symbol(symbol_state: Any) -> dict[str, Any]:
    session_bars = list(getattr(symbol_state, "session_bars", []) or [])
    candidate = getattr(symbol_state, "candidate", None)
    position = getattr(symbol_state, "position", None)
    return {
        "symbol": str(getattr(symbol_state, "symbol", "") or "").zfill(6),
        "stage": _enum_value(getattr(symbol_state, "stage", "")),
        "session_date": _json_value(getattr(symbol_state, "session_date", None)),
        "candidate": _candidate_ref(candidate),
        "pending_entry_order_id": str(getattr(symbol_state, "pending_entry_order_id", "") or ""),
        "pending_exit_order_id": str(getattr(symbol_state, "pending_exit_order_id", "") or ""),
        "pending_entry_metadata": _digest_payload(getattr(symbol_state, "pending_entry_metadata", {}) or {}),
        "pending_exit_metadata": _digest_payload(getattr(symbol_state, "pending_exit_metadata", {}) or {}),
        "session_bars_count": len(session_bars),
        "last_session_bar_hash": _bar_hash(session_bars[-1]) if session_bars else "",
        "position": _compact_position(position),
        "entry_attempted": bool(getattr(symbol_state, "entry_attempted", False)),
        "exit_attempted_dates": _json_value(getattr(symbol_state, "exit_attempted_dates", set()) or set()),
        "last_decision_code": str(getattr(symbol_state, "last_decision_code", "") or ""),
        "last_decision_details": _digest_payload(getattr(symbol_state, "last_decision_details", {}) or {}),
    }


def _candidate_ref(candidate: Any) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "symbol": str(getattr(candidate, "symbol", "") or "").zfill(6),
        "trade_date": _json_value(getattr(candidate, "trade_date", None)),
        "source_fingerprint": str(getattr(candidate, "source_fingerprint", "") or ""),
        "tradable": bool(getattr(candidate, "tradable", True)),
    }


def _compact_position(position: Any) -> dict[str, Any] | None:
    if position is None:
        return None
    payload = _json_value(position)
    if isinstance(payload, dict) and "metadata" in payload:
        payload["metadata"] = _digest_payload(payload.get("metadata") or {})
    return payload if isinstance(payload, dict) else {"value": payload}


def _digest_payload(payload: Any) -> dict[str, Any]:
    normalized = _json_value(payload)
    if isinstance(normalized, dict):
        keys = sorted(str(key) for key in normalized)
    else:
        keys = []
    return {
        "hash": canonical_json_hash(normalized),
        "keys": keys,
    }


def _stage_counts(symbols: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in symbols.values():
        stage = str(item.get("stage") or "")
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _symbols_matching(symbols: Mapping[str, Mapping[str, Any]], predicate: Any) -> list[str]:
    return [symbol for symbol, item in sorted(symbols.items()) if predicate(item)]


def _bar_hash(bar: Any) -> str:
    return canonical_json_hash(_json_value(bar))


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_json_dict") and callable(getattr(value, "to_json_dict")):
        return _json_value(value.to_json_dict())
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _enum_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
