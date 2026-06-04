"""Normalize local instrumentation rows into relay envelopes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from .event_contract import event_priority, event_scope


LINEAGE_ENVELOPE_FIELDS = (
    "schema_version",
    "scope",
    "strategy_id",
    "family_id",
    "portfolio_id",
    "account_alias",
    "strategy_version",
    "config_version",
    "portfolio_config_version",
    "risk_config_version",
    "allocation_version",
    "strategy_registry_version",
    "deployment_id",
    "parameter_set_id",
    "experiment_id",
    "variant_id",
    "code_sha",
    "trace_id",
    "decision_id",
    "logical_event_id",
    "revision",
    "exchange",
    "asset_class",
    "currency",
    "timezone",
    "bar_id",
    "event_ref",
    "decision_ref",
    "action_ref",
    "portfolio_decision_ref",
    "intent_id",
    "idempotency_key",
    "oms_order_id",
    "kis_order_id",
    "kis_order_date",
    "kis_exec_id",
    "trade_id",
    "artifact_hash",
    "source_artifact_hash",
    "source_fingerprint",
    "candidate_hash",
    "kis_resource_plan_hash",
    "portfolio_policy_hash",
)


def wrap_for_relay(
    raw_event: Mapping[str, Any],
    event_type: str,
    *,
    bot_id: str,
    serialize_payload: bool = True,
) -> dict[str, Any]:
    """Wrap a local JSON row in the relay envelope.

    Local canonical rows may already have top-level identity and an object
    payload. Legacy rows usually carry ``event_metadata`` and are forwarded
    unchanged inside the payload string.
    """

    raw = dict(raw_event or {})
    metadata = dict(raw.get("event_metadata") or {})
    payload_object = raw.get("payload") if isinstance(raw.get("payload"), Mapping) else {}
    event_type = str(raw.get("event_type") or metadata.get("event_type") or event_type)
    event_id = str(raw.get("event_id") or metadata.get("event_id") or "")
    exchange_ts = str(
        raw.get("exchange_timestamp")
        or metadata.get("exchange_timestamp")
        or raw.get("entry_time")
        or raw.get("signal_time")
        or raw.get("timestamp")
        or raw.get("event_time")
        or datetime.now(timezone.utc).isoformat()
    )
    if not event_id:
        key = _payload_key(raw)
        event_id = hashlib.sha256(f"{bot_id}|{exchange_ts}|{event_type}|{key}".encode("utf-8")).hexdigest()[:16]
    priority = raw.get("priority", event_priority(event_type))
    if event_type == "bot_error":
        priority = _bot_error_priority(raw, priority)
    payload_value: Any = raw.get("payload") if _is_canonical(raw) and "payload" in raw else raw
    if serialize_payload:
        payload_value = json.dumps(payload_value, default=str, sort_keys=True)
    envelope = {
        "event_id": event_id,
        "bot_id": str(raw.get("bot_id") or metadata.get("bot_id") or bot_id),
        "event_type": event_type,
        "priority": priority,
        "scope": str(raw.get("scope") or metadata.get("scope") or event_scope(event_type, raw)),
        "payload": payload_value,
        "exchange_timestamp": exchange_ts,
    }
    for field in LINEAGE_ENVELOPE_FIELDS:
        value = raw.get(field)
        if value in (None, ""):
            value = metadata.get(field)
        if value in (None, ""):
            value = payload_object.get(field)
        if value not in (None, ""):
            envelope[field] = value
    return envelope


def _is_canonical(raw: Mapping[str, Any]) -> bool:
    return bool(raw.get("event_id") and raw.get("event_type") and "payload" in raw)


def _payload_key(raw: Mapping[str, Any]) -> str:
    metadata = raw.get("event_metadata") if isinstance(raw.get("event_metadata"), Mapping) else {}
    payload_object = raw.get("payload") if isinstance(raw.get("payload"), Mapping) else {}
    for key in (
        "payload_key",
        "trade_id",
        "event_id",
        "logical_event_id",
        "decision_ref",
        "action_ref",
        "portfolio_decision_ref",
        "intent_id",
        "idempotency_key",
        "order_id",
        "snapshot_id",
        "date",
    ):
        value = raw.get(key)
        if value in (None, ""):
            value = metadata.get(key)
        if value in (None, ""):
            value = payload_object.get(key)
        if value not in (None, ""):
            return str(value)
    return json.dumps(raw, sort_keys=True, default=str)[:256]


def _bot_error_priority(raw: Mapping[str, Any], fallback: Any) -> Any:
    severity = str(raw.get("severity") or "").lower()
    if severity == "critical":
        return 0
    if severity == "error":
        return 0
    if severity == "warning":
        return 1
    return fallback
