"""Position snapshot payload helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def build_position_snapshot(state: Any, *, reason: str = "") -> dict[str, Any]:
    positions = getattr(state, "get_all_positions", lambda: {})()
    return {
        "record_type": "position_snapshot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "positions": [
            _position_payload(symbol, pos)
            for symbol, pos in sorted(dict(positions or {}).items(), key=lambda item: str(item[0]))
        ],
    }


def position_payloads(state: Any) -> list[dict[str, Any]]:
    positions = getattr(state, "get_all_positions", lambda: {})()
    return [
        _position_payload(symbol, pos)
        for symbol, pos in sorted(dict(positions or {}).items(), key=lambda item: str(item[0]))
    ]


def _position_payload(symbol: str, pos: Any) -> dict[str, Any]:
    payload = _json_value(pos)
    if not isinstance(payload, Mapping):
        payload = {"value": payload}
    data = dict(payload)
    data["symbol"] = str(data.get("symbol") or symbol).zfill(6)
    allocations = dict(data.get("allocations") or {})
    real_qty = _int(data.get("real_qty", data.get("qty", 0)))
    total_allocated = sum(_int(row.get("qty") if isinstance(row, Mapping) else 0) for row in allocations.values())
    working_orders = data.get("working_orders")
    working_order_count = (
        len(working_orders)
        if isinstance(working_orders, list)
        else _int(data.get("working_order_count", data.get("working_orders_count", 0)))
    )
    data["total_allocated_qty"] = total_allocated
    data["allocation_drift"] = real_qty - total_allocated
    data["working_orders_count"] = working_order_count
    unknown = allocations.get("_UNKNOWN_")
    if unknown is not None:
        data["unknown_allocation"] = _json_value(unknown)
    return data


def _json_value(value: Any) -> Any:
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


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
