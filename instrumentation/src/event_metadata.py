"""
Event Metadata — attached to every event emitted by the instrumentation layer.

Provides:
- EventMetadata dataclass with deterministic event_id
- compute_event_id: SHA-256 based idempotent ID generation
- compute_clock_skew: exchange vs local time drift in ms
- create_event_metadata: factory function for stamping events
"""

import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from .lineage import LineageContext, context_from_oms, context_from_runtime


@dataclass
class EventMetadata:
    """Attached to every event emitted by this bot."""
    event_id: str                    # deterministic hash (see compute_event_id)
    bot_id: str                      # from instrumentation_config.yaml
    exchange_timestamp: str          # ISO 8601, from exchange/broker
    local_timestamp: str             # ISO 8601, from this machine's clock
    clock_skew_ms: int               # exchange_ts - local_ts in milliseconds
    data_source_id: str              # e.g. "kis_rest", "kis_ws"
    event_type: str = ""             # "trade" | "missed_opportunity" | "snapshot" | "daily"
    payload_key: str = ""            # unique key within event type (e.g. trade_id)
    bar_id: Optional[str] = None     # candle open time, e.g. "2026-03-01T14:00+09:00_1d"
    schema_version: Optional[str] = None
    strategy_id: Optional[str] = None
    family_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    account_alias: Optional[str] = None
    strategy_version: Optional[str] = None
    config_version: Optional[str] = None
    portfolio_config_version: Optional[str] = None
    risk_config_version: Optional[str] = None
    allocation_version: Optional[str] = None
    strategy_registry_version: Optional[str] = None
    deployment_id: Optional[str] = None
    parameter_set_id: Optional[str] = None
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None
    code_sha: Optional[str] = None
    trace_id: Optional[str] = None
    decision_id: Optional[str] = None
    logical_event_id: Optional[str] = None
    revision: Optional[int] = None
    scope: Optional[str] = None
    exchange: str = "KRX"
    asset_class: str = "kr_equity"
    currency: str = "KRW"
    timezone: str = "Asia/Seoul"

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def compute_event_id(bot_id: str, timestamp: str, event_type: str, payload_key: str) -> str:
    """
    Deterministic event ID. Guarantees idempotency at every layer.

    Args:
        bot_id: this bot's unique identifier
        timestamp: exchange timestamp as ISO string
        event_type: "trade" | "missed_opportunity" | "error" | "snapshot" | "daily"
        payload_key: unique key within event type (e.g. trade_id, signal hash)

    Returns:
        16-character hex hash
    """
    raw = f"{bot_id}|{timestamp}|{event_type}|{payload_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_clock_skew(exchange_ts: datetime, local_ts: datetime) -> int:
    """Returns estimated clock skew in milliseconds."""
    exchange_ts = exchange_ts if exchange_ts.tzinfo is not None else exchange_ts.replace(tzinfo=timezone.utc)
    local_ts = local_ts if local_ts.tzinfo is not None else local_ts.replace(tzinfo=timezone.utc)
    delta = exchange_ts - local_ts
    return int(delta.total_seconds() * 1000)


def create_event_metadata(
    bot_id: str,
    event_type: str,
    payload_key: str,
    exchange_timestamp: datetime,
    data_source_id: str,
    bar_id: Optional[str] = None,
    *,
    schema_version: Optional[str] = None,
    lineage: LineageContext | Mapping[str, Any] | None = None,
    strategy_id: Optional[str] = None,
    family_id: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    account_alias: Optional[str] = None,
    strategy_version: Optional[str] = None,
    config_version: Optional[str] = None,
    portfolio_config_version: Optional[str] = None,
    risk_config_version: Optional[str] = None,
    allocation_version: Optional[str] = None,
    strategy_registry_version: Optional[str] = None,
    deployment_id: Optional[str] = None,
    parameter_set_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    variant_id: Optional[str] = None,
    code_sha: Optional[str] = None,
    trace_id: Optional[str] = None,
    decision_id: Optional[str] = None,
    logical_event_id: Optional[str] = None,
    revision: Optional[int] = None,
    scope: Optional[str] = None,
    exchange: str = "KRX",
    asset_class: str = "kr_equity",
    currency: str = "KRW",
    timezone_name: str = "Asia/Seoul",
) -> EventMetadata:
    """Factory function. Call this for every event you emit."""
    local_now = datetime.now(timezone.utc)
    exchange_timestamp = exchange_timestamp if exchange_timestamp.tzinfo is not None else exchange_timestamp.replace(tzinfo=timezone.utc)
    exchange_ts_str = exchange_timestamp.isoformat()
    local_ts_str = local_now.isoformat()

    lineage_payload = _coerce_lineage(lineage)
    overrides = {
        "strategy_id": strategy_id,
        "family_id": family_id,
        "portfolio_id": portfolio_id,
        "account_alias": account_alias,
        "strategy_version": strategy_version,
        "config_version": config_version,
        "portfolio_config_version": portfolio_config_version,
        "risk_config_version": risk_config_version,
        "allocation_version": allocation_version,
        "strategy_registry_version": strategy_registry_version,
        "deployment_id": deployment_id,
        "parameter_set_id": parameter_set_id,
        "experiment_id": experiment_id,
        "variant_id": variant_id,
        "code_sha": code_sha,
    }
    lineage_payload.update({key: value for key, value in overrides.items() if value is not None})

    return EventMetadata(
        event_id=compute_event_id(bot_id, exchange_ts_str, event_type, payload_key),
        bot_id=bot_id,
        exchange_timestamp=exchange_ts_str,
        local_timestamp=local_ts_str,
        clock_skew_ms=compute_clock_skew(exchange_timestamp, local_now),
        data_source_id=data_source_id,
        event_type=event_type,
        payload_key=payload_key,
        bar_id=bar_id,
        schema_version=schema_version,
        trace_id=trace_id,
        decision_id=decision_id,
        logical_event_id=logical_event_id,
        revision=revision,
        scope=scope,
        exchange=exchange,
        asset_class=asset_class,
        currency=currency,
        timezone=timezone_name,
        **lineage_payload,
    )


def create_revision_metadata(
    bot_id: str,
    event_type: str,
    logical_event_id: str,
    revision: int,
    exchange_timestamp: datetime,
    data_source_id: str,
    bar_id: Optional[str] = None,
    **kwargs: Any,
) -> EventMetadata:
    payload_key = f"{logical_event_id}:rev:{int(revision)}"
    return create_event_metadata(
        bot_id=bot_id,
        event_type=event_type,
        payload_key=payload_key,
        exchange_timestamp=exchange_timestamp,
        data_source_id=data_source_id,
        bar_id=bar_id,
        logical_event_id=logical_event_id,
        revision=int(revision),
        **kwargs,
    )


def metadata_from_runtime(
    payload: Mapping[str, Any],
    event_type: str,
    payload_key: str,
    exchange_timestamp: datetime,
    *,
    bot_id: str = "k_stock_trader",
    data_source_id: str = "runtime_session",
    bar_id: Optional[str] = None,
    **kwargs: Any,
) -> EventMetadata:
    lineage = context_from_runtime(payload, data_source_id=data_source_id)
    return create_event_metadata(
        bot_id=bot_id,
        event_type=event_type,
        payload_key=payload_key,
        exchange_timestamp=exchange_timestamp,
        data_source_id=data_source_id,
        bar_id=bar_id,
        lineage=lineage,
        **kwargs,
    )


def metadata_from_oms(
    payload: Mapping[str, Any],
    event_type: str,
    payload_key: str,
    exchange_timestamp: datetime,
    *,
    bot_id: str = "k_stock_trader",
    data_source_id: str = "postgres_oms",
    bar_id: Optional[str] = None,
    **kwargs: Any,
) -> EventMetadata:
    lineage = context_from_oms(payload, data_source_id=data_source_id)
    return create_event_metadata(
        bot_id=bot_id,
        event_type=event_type,
        payload_key=payload_key,
        exchange_timestamp=exchange_timestamp,
        data_source_id=data_source_id,
        bar_id=bar_id,
        lineage=lineage,
        **kwargs,
    )


def _coerce_lineage(lineage: LineageContext | Mapping[str, Any] | None) -> dict[str, Any]:
    base = {
        "event_id",
        "bot_id",
        "exchange_timestamp",
        "local_timestamp",
        "clock_skew_ms",
        "data_source_id",
        "event_type",
        "payload_key",
        "bar_id",
        "exchange",
        "asset_class",
        "currency",
        "timezone",
    }
    if isinstance(lineage, LineageContext):
        allowed = set(EventMetadata.__dataclass_fields__) - base
        return {str(key): value for key, value in lineage.to_dict().items() if key in allowed}
    if isinstance(lineage, Mapping):
        allowed = set(EventMetadata.__dataclass_fields__) - base
        return {str(key): value for key, value in lineage.items() if key in allowed}
    return {}
