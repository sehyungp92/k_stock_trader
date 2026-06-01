"""Allocation snapshot payload helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .position_snapshot import _json_value


def build_allocation_snapshot(state: Any, *, reason: str = "") -> dict[str, Any]:
    return {
        "record_type": "allocation_snapshot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "allocations": allocation_payloads(state),
    }


def allocation_payloads(state: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = getattr(state, "get_all_positions", lambda: {})()
    realized = dict(getattr(state, "strategy_realized_pnl", {}) or {})
    for symbol, pos in sorted(dict(positions or {}).items(), key=lambda item: str(item[0])):
        real_qty = _int(getattr(pos, "real_qty", 0))
        avg_price = _float(getattr(pos, "avg_price", 0.0))
        drift = _allocation_drift(pos, real_qty=real_qty)
        frozen = bool(getattr(pos, "frozen", False))
        for strategy_id, alloc in sorted(dict(getattr(pos, "allocations", {}) or {}).items()):
            row = {"symbol": str(symbol).zfill(6), "strategy_id": str(strategy_id), **_json_value(alloc)}
            qty = _int(row.get("qty"))
            cost_basis = _float(row.get("cost_basis"))
            row.update(
                {
                    "notional_krw": qty * cost_basis,
                    "realized_pnl_krw": _float(realized.get(str(strategy_id))),
                    "position_real_qty": real_qty,
                    "position_avg_price": avg_price,
                    "allocation_drift": drift,
                    "frozen": frozen,
                }
            )
            rows.append(row)
    return rows


def _allocation_drift(pos: Any, *, real_qty: int) -> int:
    method = getattr(pos, "allocation_drift", None)
    if callable(method):
        return _int(method())
    allocated = sum(_allocation_qty(alloc) for alloc in dict(getattr(pos, "allocations", {}) or {}).values())
    return real_qty - allocated


def _allocation_qty(allocation: Any) -> int:
    if isinstance(allocation, dict):
        return _int(allocation.get("qty"))
    return _int(getattr(allocation, "qty", 0))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
