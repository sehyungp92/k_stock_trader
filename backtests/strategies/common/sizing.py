from __future__ import annotations

from typing import Any


def resolve_entry_qty(
    mutations: dict[str, Any],
    *,
    entry_price: float,
    stop_price: float,
    available_capital: float,
    sizing_capital: float | None = None,
    default_fixed_qty: int = 10,
) -> int:
    """Resolve replay entry size without making absolute share count the alpha knob."""
    if "risk_per_trade_pct" not in mutations and "max_position_notional_pct" not in mutations:
        return max(1, int(float(mutations.get("fixed_qty", default_fixed_qty))))

    entry = max(float(entry_price), 1.0)
    risk_per_share = max(abs(float(entry_price) - float(stop_price)), 1.0)
    capital = max(float(sizing_capital if sizing_capital is not None else available_capital), 0.0)
    liquidity_capital = max(float(available_capital), 0.0)
    risk_budget = capital * float(mutations.get("risk_per_trade_pct", 0.001))
    notional_budget = capital * float(mutations.get("max_position_notional_pct", 0.05))

    risk_qty = int(risk_budget // risk_per_share) if risk_budget > 0 else 0
    notional_qty = int(notional_budget // entry) if notional_budget > 0 else 0
    qty = min(value for value in (risk_qty, notional_qty) if value >= 0)

    max_qty = mutations.get("max_qty")
    if max_qty is not None:
        qty = min(qty, int(float(max_qty)))
    if qty <= 0 and bool(mutations.get("allow_minimum_lot_oversize", False)) and liquidity_capital >= entry:
        qty = 1
    return max(0, qty)
