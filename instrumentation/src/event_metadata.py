"""
Event Metadata — attached to every event emitted by the instrumentation layer.

Provides:
- EventMetadata dataclass with deterministic event_id
- compute_event_id: SHA-256 based idempotent ID generation
- compute_clock_skew: exchange vs local time drift in ms
- create_event_metadata: factory function for stamping events
"""

import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class EventMetadata:
    """Attached to every event emitted by this bot."""
    event_id: str                    # deterministic hash (see compute_event_id)
    bot_id: str                      # from instrumentation_config.yaml
    exchange_timestamp: str          # ISO 8601, from exchange/broker
    local_timestamp: str             # ISO 8601, from this machine's clock
    clock_skew_ms: int               # exchange_ts - local_ts in milliseconds
    data_source_id: str              # e.g. "kis_rest", "kis_ws"
    bar_id: Optional[str] = None     # candle open time, e.g. "2026-03-01T14:00+09:00_1d"

    def to_dict(self) -> dict:
        return asdict(self)


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
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_clock_skew(exchange_ts: datetime, local_ts: datetime) -> int:
    """Returns estimated clock skew in milliseconds."""
    delta = exchange_ts - local_ts
    return int(delta.total_seconds() * 1000)


def create_event_metadata(
    bot_id: str,
    event_type: str,
    payload_key: str,
    exchange_timestamp: datetime,
    data_source_id: str,
    bar_id: Optional[str] = None,
) -> EventMetadata:
    """Factory function. Call this for every event you emit."""
    local_now = datetime.now(timezone.utc)
    exchange_ts_str = exchange_timestamp.isoformat()
    local_ts_str = local_now.isoformat()

    return EventMetadata(
        event_id=compute_event_id(bot_id, exchange_ts_str, event_type, payload_key),
        bot_id=bot_id,
        exchange_timestamp=exchange_ts_str,
        local_timestamp=local_ts_str,
        clock_skew_ms=compute_clock_skew(exchange_timestamp, local_now),
        data_source_id=data_source_id,
        bar_id=bar_id,
    )
