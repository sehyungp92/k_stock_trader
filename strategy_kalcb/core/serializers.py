from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from strategy_kalcb.models import KALCBDailyCandidate

from .state import KALCBBarState, KALCBPositionState, KALCBState, KALCBSymbolState, SymbolStage


def snapshot_state(state: KALCBState) -> dict[str, Any]:
    return _json_value(state)


def restore_state(payload: dict[str, Any]) -> KALCBState:
    symbols: dict[str, KALCBSymbolState] = {}
    for symbol, raw_state in dict(payload.get("symbols", {}) or {}).items():
        data = dict(raw_state)
        data.pop("_market_bars", None)
        candidate = data.get("candidate")
        if candidate:
            candidate["trade_date"] = _coerce_date(candidate["trade_date"])
            candidate["reject_reasons"] = tuple(candidate.get("reject_reasons", ()) or ())
            data["candidate"] = KALCBDailyCandidate(**candidate)
        data["bars"] = [
            KALCBBarState(
                timestamp=_coerce_datetime(item["timestamp"]),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
            )
            for item in data.get("bars", []) or []
        ]
        position = data.get("position")
        if position:
            position["entry_time"] = _coerce_datetime(position["entry_time"])
            data["position"] = KALCBPositionState(**position)
        data["stage"] = SymbolStage(data.get("stage", SymbolStage.WATCHING))
        data["session_date"] = _coerce_date(data["session_date"]) if data.get("session_date") else None
        symbols[symbol] = KALCBSymbolState(**data)
    return KALCBState(
        symbols=symbols,
        snapshot_hash=str(payload.get("snapshot_hash", "")),
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        session_date=_coerce_date(payload["session_date"]) if payload.get("session_date") else None,
        order_roles=dict(payload.get("order_roles", {}) or {}),
        meta=dict(payload.get("meta", {}) or {}),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value))
