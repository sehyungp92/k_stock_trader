"""Durable protective stop contracts and trigger semantics."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional


class StopSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TriggerPriceSource(str, Enum):
    LAST = "LAST"
    BID = "BID"
    MID = "MID"
    BAR_LOW = "BAR_LOW"


class StopProtectionMode(str, Enum):
    BROKER_NATIVE = "BROKER_NATIVE"
    OMS_WATCHER = "OMS_WATCHER"
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"


class StopStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    TRIGGERED_PENDING_EXECUTION = "TRIGGERED_PENDING_EXECUTION"
    EXIT_SUBMITTED = "EXIT_SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


ACTIVE_STOP_STATUSES = {
    StopStatus.PENDING.value,
    StopStatus.ACTIVE.value,
    StopStatus.TRIGGERED_PENDING_EXECUTION.value,
}

LIVE_BACKTEST_STOP_PARITY_VERSION = "oms-watcher-bar-low-v1"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def deterministic_stop_id(
    oms_id: str,
    strategy_id: str,
    symbol: str,
    generation_key: str | None = None,
) -> str:
    key = f"k-stock-trader:protective-stop:{oms_id}:{strategy_id.upper()}:{str(symbol).zfill(6)}"
    if generation_key:
        key = f"{key}:{str(generation_key).strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@dataclass(slots=True)
class ProtectiveStop:
    stop_id: str
    oms_id: str
    strategy_id: str
    symbol: str
    side: str = StopSide.LONG.value
    qty: int = 0
    stop_price: float = 0.0
    trigger_price_source: str = TriggerPriceSource.LAST.value
    protection_mode: str = StopProtectionMode.OMS_WATCHER.value
    status: str = StopStatus.PENDING.value
    broker_order_id: Optional[str] = None
    broker_order_date: Optional[str] = None
    entry_intent_id: Optional[str] = None
    entry_order_id: Optional[str] = None
    exit_intent_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    activated_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None
    last_checked_at: Optional[datetime] = None
    last_price: Optional[float] = None
    last_error: Optional[str] = None
    failure_count: int = 0
    config_hash: Optional[str] = None
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_allocation(
        cls,
        *,
        oms_id: str,
        strategy_id: str,
        symbol: str,
        qty: int,
        stop_price: float,
        trigger_price_source: str = TriggerPriceSource.LAST.value,
        protection_mode: str = StopProtectionMode.OMS_WATCHER.value,
        status: str = StopStatus.ACTIVE.value,
        entry_intent_id: Optional[str] = None,
        entry_order_id: Optional[str] = None,
        config_hash: Optional[str] = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> "ProtectiveStop":
        metadata = dict(source_metadata or {})
        generation_key = entry_intent_id or entry_order_id or metadata.get("allocation_id")
        return cls(
            stop_id=deterministic_stop_id(oms_id, strategy_id, symbol, generation_key),
            oms_id=oms_id,
            strategy_id=strategy_id.upper().strip(),
            symbol=str(symbol).zfill(6),
            qty=max(int(qty or 0), 0),
            stop_price=float(stop_price),
            trigger_price_source=str(trigger_price_source or TriggerPriceSource.LAST.value).upper(),
            protection_mode=str(protection_mode or StopProtectionMode.OMS_WATCHER.value).upper(),
            status=str(status or StopStatus.ACTIVE.value).upper(),
            entry_intent_id=entry_intent_id,
            entry_order_id=entry_order_id,
            activated_at=utcnow() if str(status or "").upper() == StopStatus.ACTIVE.value else None,
            config_hash=config_hash,
            source_metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class PriceObservation:
    symbol: str
    price: float
    timestamp: float
    source: str = TriggerPriceSource.LAST.value
    market_open: bool = True
    executable: bool = True


@dataclass(frozen=True, slots=True)
class StopTriggerDecision:
    triggered: bool
    reason: str
    stale: bool = False
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class StopTriggerResult:
    stop: ProtectiveStop
    observation: PriceObservation
    decision: StopTriggerDecision
    exit_intent_id: Optional[str] = None
    order_id: Optional[str] = None


def evaluate_stop_trigger(
    *,
    stop_price: float,
    side: str,
    observation: PriceObservation,
    stale_after_sec: float,
    now: float | None = None,
) -> StopTriggerDecision:
    """Pure stop trigger rule shared by watcher tests and backtest mirrors."""
    current_time = time.time() if now is None else float(now)
    if stop_price <= 0:
        return StopTriggerDecision(False, "invalid_stop_price", degraded=True)
    if observation.price <= 0:
        return StopTriggerDecision(False, "invalid_price", degraded=True)
    age = max(current_time - float(observation.timestamp or 0.0), 0.0)
    if stale_after_sec > 0 and age > stale_after_sec:
        return StopTriggerDecision(False, "stale_price", stale=True, degraded=True)
    not_executable = not bool(observation.executable)

    normalized_side = str(side or StopSide.LONG.value).upper().strip()
    if normalized_side == StopSide.LONG.value:
        if float(observation.price) <= float(stop_price):
            if not_executable:
                return StopTriggerDecision(True, "long_stop_breached_not_executable", degraded=True)
            return StopTriggerDecision(True, "long_stop_breached")
        if not_executable:
            return StopTriggerDecision(False, "not_executable", degraded=True)
        return StopTriggerDecision(False, "above_stop")
    if normalized_side == StopSide.SHORT.value:
        if float(observation.price) >= float(stop_price):
            if not_executable:
                return StopTriggerDecision(True, "short_stop_breached_not_executable", degraded=True)
            return StopTriggerDecision(True, "short_stop_breached")
        if not_executable:
            return StopTriggerDecision(False, "not_executable", degraded=True)
        return StopTriggerDecision(False, "below_stop")
    return StopTriggerDecision(False, "unsupported_side", degraded=True)


def _row_get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def stop_from_row(row: Mapping[str, Any]) -> ProtectiveStop:
    return ProtectiveStop(
        stop_id=str(row["stop_id"]),
        oms_id=str(row["oms_id"]),
        strategy_id=str(row["strategy_id"]).upper().strip(),
        symbol=str(row["symbol"]).zfill(6),
        side=str(_row_get(row, "side") or StopSide.LONG.value).upper(),
        qty=int(_row_get(row, "qty") or 0),
        stop_price=float(_row_get(row, "stop_price") or 0.0),
        trigger_price_source=str(_row_get(row, "trigger_price_source") or TriggerPriceSource.LAST.value).upper(),
        protection_mode=str(_row_get(row, "protection_mode") or StopProtectionMode.OMS_WATCHER.value).upper(),
        status=str(_row_get(row, "status") or StopStatus.PENDING.value).upper(),
        broker_order_id=_row_get(row, "broker_order_id"),
        broker_order_date=_row_get(row, "broker_order_date"),
        entry_intent_id=str(_row_get(row, "entry_intent_id")) if _row_get(row, "entry_intent_id") else None,
        entry_order_id=_row_get(row, "entry_order_id"),
        exit_intent_id=str(_row_get(row, "exit_intent_id")) if _row_get(row, "exit_intent_id") else None,
        idempotency_key=_row_get(row, "idempotency_key"),
        created_at=_row_get(row, "created_at") or utcnow(),
        updated_at=_row_get(row, "updated_at") or utcnow(),
        activated_at=_row_get(row, "activated_at"),
        triggered_at=_row_get(row, "triggered_at"),
        last_checked_at=_row_get(row, "last_checked_at"),
        last_price=float(_row_get(row, "last_price")) if _row_get(row, "last_price") is not None else None,
        last_error=_row_get(row, "last_error"),
        failure_count=int(_row_get(row, "failure_count") or 0),
        config_hash=_row_get(row, "config_hash"),
        source_metadata=dict(_row_get(row, "source_metadata") or {}),
    )
