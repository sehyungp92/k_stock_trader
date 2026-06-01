"""Bounded, fail-open JSONL writer for canonical assistant events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .event_contract import event_dir, event_priority, event_schema_version, event_scope
from .event_metadata import compute_clock_skew, compute_event_id
from .lineage import LineageContext, context_from_env, stable_hash


class JSONLEventWriter:
    """Append canonical events to ``instrumentation/data`` style directories.

    The writer never raises to the trading hot path. Errors are bounded in
    memory and mirrored to ``errors/`` when possible.
    """

    def __init__(
        self,
        data_dir: str | Path = "instrumentation/data",
        *,
        lineage: LineageContext | None = None,
        bot_id: str = "k_stock_trader",
        max_errors: int = 20,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.lineage = lineage or context_from_env()
        self.bot_id = bot_id or self.lineage.bot_id
        self.max_errors = max(int(max_errors), 1)
        self.errors: list[dict[str, Any]] = []

    def write(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        payload_key: str | None = None,
        exchange_timestamp: datetime | str | None = None,
        lineage: LineageContext | Mapping[str, Any] | None = None,
        schema_version: str | None = None,
        scope: str | None = None,
        subdir: str | None = None,
        revision: int | None = None,
        logical_event_id: str = "",
    ) -> dict[str, Any] | None:
        try:
            payload_dict = dict(payload or {})
            ctx = _coerce_lineage(lineage, self.lineage)
            exchange_dt = _coerce_datetime(
                exchange_timestamp
                or payload_dict.get("exchange_timestamp")
                or payload_dict.get("timestamp")
                or payload_dict.get("event_time")
                or payload_dict.get("recorded_at")
                or payload_dict.get("fill_ts")
            )
            local_now = datetime.now(timezone.utc)
            exchange_ts = exchange_dt.isoformat()
            local_ts = local_now.isoformat()
            event_type = str(event_type or payload_dict.get("event_type") or "event")
            key = str(payload_key or _payload_key(payload_dict))
            event_id = compute_event_id(ctx.bot_id or self.bot_id, exchange_ts, event_type, key)
            scope_value = scope or event_scope(event_type, payload_dict)
            lineage_payload = ctx.to_dict()
            gaps = ctx.monthly_lineage_gaps(scope=scope_value)
            event = {
                "event_id": event_id,
                "bot_id": ctx.bot_id or self.bot_id,
                "event_type": event_type,
                "schema_version": schema_version or payload_dict.get("schema_version") or event_schema_version(event_type),
                "scope": scope_value,
                "priority": event_priority(event_type),
                "exchange_timestamp": exchange_ts,
                "local_timestamp": local_ts,
                "clock_skew_ms": compute_clock_skew(exchange_dt, local_now),
                "data_source_id": ctx.data_source_id,
                "payload_key": key,
                "logical_event_id": logical_event_id or str(payload_dict.get("logical_event_id") or ""),
                "revision": int(revision if revision is not None else payload_dict.get("revision") or 0),
                **lineage_payload,
                "lineage_gap": bool(gaps),
                "lineage_gaps": list(gaps),
                "payload": payload_dict,
            }
            for join_key in _JOIN_KEY_FIELDS:
                value = payload_dict.get(join_key)
                if value not in (None, "") and join_key not in event:
                    event[join_key] = value
            self._append(event, subdir or event_dir(event_type), exchange_dt)
            return event
        except Exception as exc:
            self._record_error("write", event_type, exc)
            return None

    def _append(self, event: Mapping[str, Any], subdir: str, exchange_dt: datetime) -> None:
        directory = self.data_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        date_str = exchange_dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        filename = f"{subdir}_{date_str}.jsonl"
        with (directory / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True, default=str) + "\n")

    def _record_error(self, method: str, context: str, error: Exception) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": "event_writer",
            "method": method,
            "context": str(context),
            "error": str(error),
            "error_type": type(error).__name__,
        }
        self.errors.append(entry)
        del self.errors[:-self.max_errors]
        try:
            directory = self.data_dir / "errors"
            directory.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with (directory / f"instrumentation_errors_{date_str}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except Exception:
            pass


def _payload_key(payload: Mapping[str, Any]) -> str:
    for key in (
        "event_id",
        "logical_event_id",
        "decision_ref",
        "action_ref",
        "portfolio_decision_ref",
        "intent_id",
        "idempotency_key",
        "order_id",
        "oms_order_id",
        "kis_order_id",
        "kis_exec_id",
        "trade_id",
        "state_hash",
        "plan_hash",
        "snapshot_id",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return stable_hash(payload)


_JOIN_KEY_FIELDS = (
    "bar_id",
    "event_ref",
    "decision_ref",
    "action_ref",
    "provisional_order_ref",
    "portfolio_decision_ref",
    "intent_id",
    "idempotency_key",
    "order_id",
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


def _coerce_lineage(value: LineageContext | Mapping[str, Any] | None, fallback: LineageContext) -> LineageContext:
    if isinstance(value, LineageContext):
        return value
    if isinstance(value, Mapping):
        allowed = set(LineageContext.__dataclass_fields__)
        overrides = {key: item for key, item in value.items() if key in allowed}
        return fallback.with_overrides(**overrides)
    return fallback


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            pass
    if value not in (None, ""):
        try:
            raw = str(value)
            if raw.replace(".", "", 1).isdigit():
                return datetime.fromtimestamp(float(raw), tz=timezone.utc)
            parsed = datetime.fromisoformat(raw)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)
