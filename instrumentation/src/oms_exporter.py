"""Fail-open OMS assistant telemetry emitter."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .event_writer import JSONLEventWriter
from .allocation_snapshot import build_allocation_snapshot
from .lineage import LineageContext, context_from_oms, stable_hash
from .portfolio_snapshot import build_portfolio_snapshot
from .position_snapshot import build_position_snapshot, position_payloads


class OMSEventEmitter:
    def __init__(self, data_dir: str | Path = "instrumentation/data", *, lineage: LineageContext | None = None) -> None:
        self.lineage = lineage or context_from_oms({}, data_source_id="postgres_oms")
        self.writer = JSONLEventWriter(data_dir, lineage=self.lineage)

    def update_lineage(self, **overrides: Any) -> None:
        self.lineage = self.lineage.with_overrides(**overrides)
        self.writer.lineage = self.lineage

    def emit_deployment(self, status: str, payload: Mapping[str, Any] | None = None) -> None:
        row = {"record_type": "oms_deployment", "status": status, "timestamp": _now(), **dict(payload or {})}
        self._write("deployment", row, payload_key=f"oms:{status}:{row['timestamp']}", scope="oms")

    def emit_intent(self, intent: Any, result: Any | None = None, *, phase: str = "finalized", extra: Mapping[str, Any] | None = None) -> None:
        row = {
            "record_type": "oms_intent",
            "phase": phase,
            "timestamp": _now(),
            "intent": _intent_payload(intent),
            "result": _result_payload(result) if result is not None else None,
            **dict(extra or {}),
        }
        row.update(_intent_join_keys(intent))
        if result is not None:
            row.update(_result_join_keys(result))
        self._write("oms_intent", row, payload_key=f"{row.get('intent_id', '')}:{phase}:{row.get('status', '')}", intent=intent, scope="oms")

    def emit_risk_decision(
        self,
        intent: Any,
        risk_result: Any,
        *,
        trace: list[dict[str, Any]] | None = None,
        oms: Any | None = None,
        state_summary: Mapping[str, Any] | None = None,
    ) -> None:
        row = {
            "record_type": "risk_decision",
            "timestamp": _now(),
            "intent": _intent_payload(intent),
            "decision": _enum_name(getattr(risk_result, "decision", "")),
            "reason": str(getattr(risk_result, "reason", "") or ""),
            "modified_qty": getattr(risk_result, "modified_qty", None),
            "cooldown_sec": getattr(risk_result, "cooldown_sec", None),
            "blocking_positions": getattr(risk_result, "blocking_positions", None),
            "resource_conflict_type": getattr(risk_result, "resource_conflict_type", None),
            "trace": trace if trace is not None else list(getattr(risk_result, "trace", []) or []),
            "current_state_summary": _risk_state_summary(oms, intent=intent, state_summary=state_summary),
            **_intent_join_keys(intent),
        }
        self._write("risk_decision", row, payload_key=f"{row.get('intent_id', '')}:{row['decision']}:{stable_hash(row.get('trace') or [])}", intent=intent, scope="oms")

    def emit_order_event(self, order: Any, event_type: str, *, payload: Mapping[str, Any] | None = None, intent: Any | None = None) -> None:
        row = {
            "record_type": "order_event",
            "order_event_type": event_type,
            "timestamp": _now(),
            "order": _json_value(order),
            **dict(payload or {}),
        }
        row.update(_working_order_join_keys(order))
        if intent is not None:
            row.update(_intent_join_keys(intent))
        self._write("order", row, payload_key=f"{row.get('order_id', '')}:{event_type}:{row.get('status_after', '')}", intent=intent, scope="oms")

    def emit_fill(
        self,
        order: Any,
        fill_qty: int,
        *,
        intent: Any | None = None,
        inferred: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        row = {
            "record_type": "fill_event",
            "timestamp": _now(),
            "fill_qty": int(fill_qty),
            "qty": int(fill_qty),
            "price": float(getattr(order, "price", 0.0) or 0.0),
            "side": str(getattr(order, "side", "") or ""),
            "inferred": bool(inferred),
            "order": _json_value(order),
            **_working_order_join_keys(order),
            **dict(extra or {}),
        }
        if intent is not None:
            row.update(_intent_join_keys(intent))
        self._write("fill", row, payload_key=f"{row.get('order_id', '')}:{row.get('filled_qty', 0)}:{fill_qty}", intent=intent, scope="oms")

    def emit_position_snapshot(self, state: Any, *, reason: str = "") -> None:
        row = build_position_snapshot(state, reason=reason)
        self._write("position_snapshot", row, payload_key=f"{reason}:{stable_hash(row['positions'])}", scope="oms")

    def emit_allocation_snapshot(self, state: Any, *, reason: str = "") -> None:
        row = build_allocation_snapshot(state, reason=reason)
        self._write("allocation_snapshot", row, payload_key=f"{reason}:{stable_hash(row['allocations'])}", scope="oms")

    def emit_portfolio_snapshot(self, oms: Any, *, reason: str = "") -> None:
        row = build_portfolio_snapshot(oms, portfolio_id=self.lineage.portfolio_id, account_alias=self.lineage.account_alias, reason=reason)
        self._write("portfolio_snapshot", row, payload_key=f"{reason}:{stable_hash(row)}", scope="portfolio")

    def emit_reconciliation(self, event_type: str, *, symbol: str = "", payload: Mapping[str, Any] | None = None) -> None:
        symbol_value = str(symbol).zfill(6) if str(symbol or "").strip() else ""
        row = {"record_type": "reconciliation_event", "reconciliation_type": event_type, "timestamp": _now(), "symbol": symbol_value, **dict(payload or {})}
        self._write("reconciliation_event", row, payload_key=f"{event_type}:{symbol}:{stable_hash(row)}", scope="oms")

    def emit_heartbeat(self, oms: Any, *, reason: str = "reconcile") -> None:
        row = {
            "record_type": "heartbeat",
            "timestamp": _now(),
            "reason": reason,
            "equity_krw": float(getattr(oms.state, "equity", 0.0) or 0.0),
            "buyable_cash_krw": float(getattr(oms.state, "buyable_cash", 0.0) or 0.0),
            "safe_mode": bool(getattr(oms.risk, "safe_mode", False)),
            "halt_new_entries": bool(getattr(oms.risk, "halt_new_entries", False)),
        }
        self._write("heartbeat", row, payload_key=f"oms:{reason}:{row['timestamp']}", scope="oms")

    def _write(self, event_type: str, payload: Mapping[str, Any], *, payload_key: str, intent: Any | None = None, scope: str = "oms") -> None:
        lineage = context_from_oms(payload, strategy_id=str(getattr(intent, "strategy_id", "") or payload.get("strategy_id") or ""), data_source_id="postgres_oms").with_overrides(
            family_id=self.lineage.family_id,
            portfolio_id=self.lineage.portfolio_id,
            account_alias=self.lineage.account_alias,
            deployment_id=self.lineage.deployment_id,
            strategy_version=self.lineage.strategy_version,
            config_version=self.lineage.config_version,
            code_sha=self.lineage.code_sha,
            portfolio_config_version=self.lineage.portfolio_config_version,
            risk_config_version=self.lineage.risk_config_version,
            allocation_version=self.lineage.allocation_version,
            strategy_registry_version=self.lineage.strategy_registry_version,
            kis_resource_plan_hash=self.lineage.kis_resource_plan_hash,
            portfolio_policy_hash=self.lineage.portfolio_policy_hash,
        )
        self.writer.write(event_type, payload, payload_key=payload_key, lineage=lineage, scope=scope)


def _risk_state_summary(oms: Any | None, *, intent: Any | None = None, state_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(state_summary, Mapping) and state_summary:
        return dict(state_summary)
    if oms is None:
        return {}
    try:
        summary = build_portfolio_snapshot(oms, portfolio_id=getattr(getattr(oms, "event_emitter", None), "lineage", LineageContext()).portfolio_id, reason="risk_decision")
    except Exception:
        summary = {}
    try:
        symbol = str(getattr(intent, "symbol", "") or "").zfill(6)
        if symbol.strip("0"):
            for position in position_payloads(getattr(oms, "state", None)):
                if str(position.get("symbol") or "").zfill(6) == symbol:
                    summary["intent_symbol_position"] = position
                    break
    except Exception:
        pass
    return summary


def _intent_payload(intent: Any) -> dict[str, Any]:
    return {
        "intent_id": str(getattr(intent, "intent_id", "") or ""),
        "idempotency_key": str(getattr(intent, "idempotency_key", "") or ""),
        "intent_type": _enum_name(getattr(intent, "intent_type", "")),
        "strategy_id": str(getattr(intent, "strategy_id", "") or "").upper().strip(),
        "symbol": str(getattr(intent, "symbol", "") or "").zfill(6),
        "desired_qty": getattr(intent, "desired_qty", None),
        "target_qty": getattr(intent, "target_qty", None),
        "urgency": _enum_name(getattr(intent, "urgency", "")),
        "time_horizon": _enum_name(getattr(intent, "time_horizon", "")),
        "constraints": _json_value(getattr(intent, "constraints", None)),
        "risk_payload": _json_value(getattr(intent, "risk_payload", None)),
        "signal_hash": getattr(intent, "signal_hash", None),
        "metadata": _json_value(getattr(intent, "metadata", {}) or {}),
        "timestamp": getattr(intent, "timestamp", None),
    }


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "intent_id": str(getattr(result, "intent_id", "") or ""),
        "status": _enum_name(getattr(result, "status", "")),
        "message": str(getattr(result, "message", "") or ""),
        "modified_qty": getattr(result, "modified_qty", None),
        "order_id": getattr(result, "order_id", None),
        "cooldown_until": getattr(result, "cooldown_until", None),
        "blocking_positions": getattr(result, "blocking_positions", None),
        "resource_conflict_type": getattr(result, "resource_conflict_type", None),
        "oms_received_at": getattr(result, "oms_received_at", None),
        "order_submitted_at": getattr(result, "order_submitted_at", None),
    }


def _intent_join_keys(intent: Any) -> dict[str, Any]:
    metadata = dict(getattr(intent, "metadata", {}) or {})
    return {
        "strategy_id": str(getattr(intent, "strategy_id", "") or "").upper().strip(),
        "symbol": str(getattr(intent, "symbol", "") or "").zfill(6),
        "intent_id": str(getattr(intent, "intent_id", "") or ""),
        "idempotency_key": str(getattr(intent, "idempotency_key", "") or ""),
        "event_ref": metadata.get("event_ref", ""),
        "decision_ref": metadata.get("decision_ref", ""),
        "action_ref": metadata.get("action_ref", ""),
        "provisional_order_ref": metadata.get("provisional_order_ref", ""),
        "portfolio_decision_ref": metadata.get("portfolio_decision_ref", ""),
        "artifact_hash": metadata.get("source_artifact_hash", ""),
        "source_fingerprint": metadata.get("source_fingerprint", ""),
        "candidate_hash": metadata.get("candidate_hash", ""),
        "portfolio_policy_hash": metadata.get("portfolio_policy_hash", ""),
    }


def _result_join_keys(result: Any) -> dict[str, Any]:
    return {
        "status": _enum_name(getattr(result, "status", "")),
        "order_id": getattr(result, "order_id", None),
        "kis_order_id": getattr(result, "order_id", None),
    }


def _working_order_join_keys(order: Any) -> dict[str, Any]:
    return {
        "strategy_id": str(getattr(order, "strategy_id", "") or "").upper().strip(),
        "symbol": str(getattr(order, "symbol", "") or "").zfill(6),
        "order_id": str(getattr(order, "order_id", "") or ""),
        "kis_order_id": str(getattr(order, "order_id", "") or ""),
        "oms_order_id": str(getattr(order, "oms_order_id", "") or ""),
        "intent_id": str(getattr(order, "intent_id", "") or ""),
        "idempotency_key": str(getattr(order, "idempotency_key", "") or ""),
        "side": str(getattr(order, "side", "") or ""),
        "qty": getattr(order, "qty", None),
        "filled_qty": getattr(order, "filled_qty", None),
        "price": getattr(order, "price", None),
    }


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    if hasattr(value, "__dict__"):
        return {str(key): _json_value(item) for key, item in vars(value).items() if not str(key).startswith("_")}
    return value


def _enum_name(value: Any) -> str:
    return str(value.name if isinstance(value, Enum) else value or "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
