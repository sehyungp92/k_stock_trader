"""Portfolio snapshot payload helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from .position_snapshot import _json_value


def build_portfolio_snapshot(oms: Any, *, portfolio_id: str = "olr_kalcb", account_alias: str = "kis_primary", reason: str = "") -> dict[str, Any]:
    positions = getattr(getattr(oms, "state", None), "get_all_positions", lambda: {})()
    working_count = sum(_working_order_count(pos) for pos in dict(positions or {}).values())
    equity = float(getattr(oms.state, "equity", 0.0) or 0.0)
    exposures = _portfolio_exposures(oms, dict(positions or {}), equity=equity)
    return {
        "record_type": "portfolio_snapshot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "portfolio_id": portfolio_id,
        "account_alias": account_alias,
        "equity_krw": equity,
        "buyable_cash_krw": float(getattr(oms.state, "buyable_cash", 0.0) or 0.0),
        "daily_pnl_krw": float(getattr(oms.state, "daily_pnl", 0.0) or 0.0),
        "daily_pnl_pct": float(getattr(oms.state, "daily_pnl_pct", 0.0) or 0.0),
        "daily_realized_pnl_krw": float(getattr(oms.state, "daily_realized_pnl", 0.0) or 0.0),
        "gross_exposure_krw": exposures["gross_exposure_krw"],
        "gross_exposure_pct": exposures["gross_exposure_pct"],
        "positions_count": len(dict(positions or {})),
        "working_orders_count": working_count,
        "safe_mode": bool(getattr(oms.risk, "safe_mode", False)),
        "halt_new_entries": bool(getattr(oms.risk, "halt_new_entries", False)),
        "flatten_in_progress": bool(getattr(oms.risk, "flatten_in_progress", False)),
        "regime": str(getattr(oms.risk.config, "current_regime", "") or ""),
        "regime_exposure_cap": getattr(oms.risk.config, "regime_exposure_caps", {}).get(getattr(oms.risk.config, "current_regime", ""), 1.0),
        "allocation_drift_count": sum(1 for pos in dict(positions or {}).values() if _allocation_drift(pos) != 0 or bool(getattr(pos, "frozen", False))),
        "sector_exposures": exposures["sector_exposures"],
        "strategy_exposures": exposures["strategy_exposures"],
        "symbol_exposures": exposures["symbol_exposures"],
        "pending_reservations": exposures["pending_reservations"],
    }


def _portfolio_exposures(oms: Any, positions: Mapping[str, Any], *, equity: float) -> dict[str, Any]:
    sector_map = _sector_map(oms)
    realized = dict(getattr(getattr(oms, "state", None), "strategy_realized_pnl", {}) or {})
    symbol_exposures: dict[str, dict[str, Any]] = {}
    strategy_exposures: dict[str, dict[str, Any]] = {}
    sector_exposures: dict[str, dict[str, Any]] = {}
    gross = 0.0
    for symbol, pos in sorted(dict(positions or {}).items(), key=lambda item: str(item[0])):
        key = str(symbol).zfill(6)
        real_qty = max(_int(getattr(pos, "real_qty", 0)), 0)
        avg_price = max(_float(getattr(pos, "avg_price", 0.0)), 0.0)
        broker_notional = real_qty * avg_price
        allocated_qty = 0
        allocation_notional = 0.0
        allocations = dict(getattr(pos, "allocations", {}) or {})
        for allocation in allocations.values():
            qty = max(_allocation_field_int(allocation, "qty"), 0)
            price = max(_allocation_field_float(allocation, "cost_basis", avg_price), 0.0)
            allocated_qty += qty
            allocation_notional += qty * (price or avg_price)
        effective_qty = allocated_qty if allocations else real_qty
        effective_notional = allocation_notional if allocations else broker_notional
        gross += abs(effective_notional)
        sector = str(sector_map.get(key, "UNKNOWN") or "UNKNOWN").upper().strip() or "UNKNOWN"
        symbol_exposures[key] = {
            "qty": effective_qty,
            "real_qty": real_qty,
            "allocated_qty": allocated_qty,
            "avg_price": avg_price,
            "notional_krw": effective_notional,
            "broker_notional_krw": broker_notional,
            "sector": sector,
            "allocation_drift": _allocation_drift(pos),
            "frozen": bool(getattr(pos, "frozen", False)),
        }
        sector_row = sector_exposures.setdefault(sector, {"qty": 0, "notional_krw": 0.0, "symbols_count": 0})
        sector_row["qty"] += effective_qty
        sector_row["notional_krw"] += effective_notional
        sector_row["symbols_count"] += 1 if effective_qty > 0 else 0
        for strategy_id, allocation in sorted(allocations.items()):
            sid = str(strategy_id).upper().strip()
            qty = max(_allocation_field_int(allocation, "qty"), 0)
            cost_basis = max(_allocation_field_float(allocation, "cost_basis", avg_price), 0.0)
            allocation_notional = qty * (cost_basis or avg_price)
            row = strategy_exposures.setdefault(
                sid,
                {"qty": 0, "notional_krw": 0.0, "symbols_count": 0, "realized_pnl_krw": _float(realized.get(sid))},
            )
            row["qty"] += qty
            row["notional_krw"] += allocation_notional
            row["symbols_count"] += 1 if qty > 0 else 0
    return {
        "gross_exposure_krw": gross,
        "gross_exposure_pct": gross / equity if equity > 0 else 0.0,
        "sector_exposures": sector_exposures,
        "strategy_exposures": strategy_exposures,
        "symbol_exposures": symbol_exposures,
        "pending_reservations": _pending_reservations(oms),
    }


def _sector_map(oms: Any) -> dict[str, str]:
    risk = getattr(oms, "risk", None)
    sector_exposure = getattr(risk, "_sector_exposure", None)
    mapping = dict(getattr(sector_exposure, "sym_to_sector", {}) or {})
    return {str(symbol).zfill(6): str(sector).upper().strip() for symbol, sector in mapping.items()}


def _pending_reservations(oms: Any) -> dict[str, Any]:
    risk = getattr(oms, "risk", None)
    sector_exposure = getattr(risk, "_sector_exposure", None)
    if sector_exposure is None:
        return {}
    return {
        "sector_working_count": _json_value(getattr(sector_exposure, "sector_working_count", {}) or {}),
        "sector_working_notional": _json_value(getattr(sector_exposure, "sector_working_notional", {}) or {}),
    }


def _allocation_drift(pos: Any) -> int:
    method = getattr(pos, "allocation_drift", None)
    if callable(method):
        return _int(method())
    real_qty = _int(getattr(pos, "real_qty", 0))
    allocated = sum(_allocation_field_int(alloc, "qty") for alloc in dict(getattr(pos, "allocations", {}) or {}).values())
    return real_qty - allocated


def _working_order_count(pos: Any) -> int:
    working_orders = getattr(pos, "working_orders", None)
    if working_orders is not None:
        return len(working_orders or ())
    return _int(getattr(pos, "working_order_count", 0))


def _allocation_field_int(allocation: Any, field: str) -> int:
    if isinstance(allocation, Mapping):
        return _int(allocation.get(field))
    return _int(getattr(allocation, field, 0))


def _allocation_field_float(allocation: Any, field: str, default: float = 0.0) -> float:
    if isinstance(allocation, Mapping):
        return _float(allocation.get(field, default))
    return _float(getattr(allocation, field, default))


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
