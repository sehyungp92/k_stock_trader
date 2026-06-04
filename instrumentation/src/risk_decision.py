"""Risk decision payload helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def build_risk_decision_payload(
    intent: Any,
    risk_result: Any,
    *,
    current_state_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "record_type": "risk_decision",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy_id": str(getattr(intent, "strategy_id", "") or "").upper().strip(),
        "symbol": str(getattr(intent, "symbol", "") or "").zfill(6),
        "intent_id": str(getattr(intent, "intent_id", "") or ""),
        "idempotency_key": str(getattr(intent, "idempotency_key", "") or ""),
        "intent_type": _enum_name(getattr(intent, "intent_type", "")),
        "decision": _enum_name(getattr(risk_result, "decision", "")),
        "reason": str(getattr(risk_result, "reason", "") or ""),
        "modified_qty": getattr(risk_result, "modified_qty", None),
        "cooldown_sec": getattr(risk_result, "cooldown_sec", None),
        "blocking_positions": getattr(risk_result, "blocking_positions", None),
        "resource_conflict_type": getattr(risk_result, "resource_conflict_type", None),
        "current_state_summary": _json_value(current_state_summary or {}),
        "trace": _json_value(getattr(risk_result, "trace", []) or []),
    }


def _enum_name(value: Any) -> str:
    return str(value.name if isinstance(value, Enum) else value or "")


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.name
    return value
