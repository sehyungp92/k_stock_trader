from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from oms_client.client import AccountState, AllocationInfo, PositionInfo


@dataclass(frozen=True, slots=True)
class PortfolioExposure:
    qty: int = 0
    notional: float = 0.0


@dataclass(frozen=True, slots=True)
class CashEquity:
    cash: float = 0.0
    equity: float = 0.0


@dataclass(slots=True)
class PortfolioContextProvider:
    """Runtime portfolio/account context used by the evidence driver.

    Action metadata is still useful provenance, but arbitration must read
    current cash and strategy-owned exposure from account state. This provider
    is intentionally small so dry-run, paper/live, and offline replay can share
    the same lookup surface.
    """

    oms_client: Any | None = None
    account_state: AccountState = field(default_factory=AccountState)
    positions: dict[str, PositionInfo] = field(default_factory=dict)
    sector_map: dict[str, str] = field(default_factory=dict)

    async def refresh(self) -> None:
        if self.oms_client is None:
            return
        get_account = getattr(self.oms_client, "get_account_state", None)
        if callable(get_account):
            account = await _maybe_await(get_account())
            if account is not None:
                self.account_state = _coerce_account(account)
        get_positions = getattr(self.oms_client, "get_all_positions", None)
        if callable(get_positions):
            positions = await _maybe_await(get_positions())
            if positions is not None:
                self.positions = _coerce_positions(positions)

    def strategy_exposure(self, strategy_id: str, symbol: str) -> PortfolioExposure:
        sid = str(strategy_id or "").upper().strip()
        position = self.positions.get(_symbol(symbol))
        if position is None:
            return PortfolioExposure()
        qty = int(position.get_allocation(sid) or 0)
        allocation = position.allocations.get(sid)
        price = float(getattr(allocation, "cost_basis", 0.0) or position.avg_price or 0.0)
        return PortfolioExposure(qty=max(qty, 0), notional=max(qty, 0) * max(price, 0.0))

    def symbol_exposure(self, symbol: str) -> PortfolioExposure:
        position = self.positions.get(_symbol(symbol))
        if position is None:
            return PortfolioExposure()
        qty = int(position.real_qty or 0)
        price = float(position.avg_price or 0.0)
        return PortfolioExposure(qty=max(qty, 0), notional=max(qty, 0) * max(price, 0.0))

    def portfolio_exposure(self) -> PortfolioExposure:
        qty = 0
        notional = 0.0
        for position in self.positions.values():
            position_qty = max(int(position.real_qty or 0), 0)
            qty += position_qty
            notional += position_qty * max(float(position.avg_price or 0.0), 0.0)
        return PortfolioExposure(qty=qty, notional=notional)

    def sector_exposure(self, sector: str) -> PortfolioExposure:
        wanted = str(sector or "UNKNOWN").upper().strip() or "UNKNOWN"
        qty = 0
        notional = 0.0
        for symbol, position in self.positions.items():
            mapped = str(self.sector_map.get(_symbol(symbol), "UNKNOWN")).upper().strip() or "UNKNOWN"
            if mapped != wanted:
                continue
            position_qty = max(int(position.real_qty or 0), 0)
            qty += position_qty
            notional += position_qty * max(float(position.avg_price or 0.0), 0.0)
        return PortfolioExposure(qty=qty, notional=notional)

    def cash_equity(self) -> CashEquity:
        return CashEquity(
            cash=float(self.account_state.buyable_cash or 0.0),
            equity=float(self.account_state.equity or self.account_state.buyable_cash or 0.0),
        )

    def portfolio_view(self, strategy_id: str) -> Any:
        sid = str(strategy_id or "").upper().strip()
        cash_equity = self.cash_equity()
        positions: dict[str, int] = {}
        open_notional = 0.0
        for symbol, position in self.positions.items():
            exposure = self.strategy_exposure(sid, symbol)
            if exposure.qty <= 0:
                continue
            positions[symbol] = exposure.qty
            open_notional += exposure.notional
        if sid == "KALCB":
            from strategy_kalcb.core.core_models import KALCBPortfolioView

            return KALCBPortfolioView(
                cash=cash_equity.cash,
                positions=positions,
                open_positions=len(positions),
                open_notional=open_notional,
                equity=cash_equity.equity,
                session_start_equity=cash_equity.equity,
                session_return_pct=float(self.account_state.daily_pnl_pct or 0.0),
            )
        if sid == "OLR":
            from strategy_olr.core.core_models import OLRPortfolioView

            gross = open_notional / cash_equity.equity if cash_equity.equity > 0 else 0.0
            return OLRPortfolioView(
                cash=cash_equity.cash,
                equity=cash_equity.equity,
                positions=positions,
                open_positions=len(positions),
                open_notional=open_notional,
                gross_exposure_pct=gross,
            )
        raise ValueError(f"unsupported strategy_id={strategy_id!r}")

    def apply_fill(self, strategy_id: str, symbol: str, side: str, qty: int, price: float) -> None:
        sid = str(strategy_id or "").upper().strip()
        key = _symbol(symbol)
        fill_qty = max(int(qty or 0), 0)
        fill_price = max(float(price or 0.0), 0.0)
        if fill_qty <= 0:
            return
        position = self.positions.get(key)
        if position is None:
            position = PositionInfo(symbol=key, real_qty=0, avg_price=fill_price, allocations={})
            self.positions[key] = position
        allocation = position.allocations.get(sid)
        if allocation is None:
            allocation = AllocationInfo(strategy_id=sid, qty=0, cost_basis=fill_price)
            position.allocations[sid] = allocation
        if str(side or "").upper().strip() == "SELL":
            reducible_qty = min(fill_qty, max(int(allocation.qty or 0), 0))
            delta = -reducible_qty
            cash_delta = reducible_qty * fill_price
        else:
            delta = fill_qty
            cash_delta = -(fill_qty * fill_price)
        self.account_state.buyable_cash = float(self.account_state.buyable_cash or 0.0) + cash_delta
        if self.oms_client is not None and hasattr(self.oms_client, "account_state"):
            self.oms_client.account_state = self.account_state
        if delta > 0:
            old_qty = max(int(allocation.qty or 0), 0)
            old_notional = old_qty * max(float(allocation.cost_basis or 0.0), 0.0)
            allocation.qty = old_qty + delta
            allocation.cost_basis = (old_notional + delta * fill_price) / max(allocation.qty, 1)
            position.real_qty = max(int(position.real_qty or 0), 0) + delta
            position.avg_price = (
                (max(int(position.real_qty or 0), 0) - delta) * max(float(position.avg_price or 0.0), 0.0)
                + delta * fill_price
            ) / max(position.real_qty, 1)
        elif delta < 0:
            reduce_qty = abs(delta)
            allocation.qty = max(int(allocation.qty or 0) - reduce_qty, 0)
            position.real_qty = max(int(position.real_qty or 0) - reduce_qty, 0)
            if allocation.qty <= 0:
                position.allocations.pop(sid, None)
            if position.real_qty <= 0:
                self.positions.pop(key, None)
        if self.oms_client is not None and hasattr(self.oms_client, "positions"):
            self.oms_client.positions = dict(self.positions)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _coerce_account(value: Any) -> AccountState:
    if isinstance(value, AccountState):
        return value
    data = dict(value or {})
    return AccountState(
        equity=float(data.get("equity", 0.0) or 0.0),
        buyable_cash=float(data.get("buyable_cash", data.get("cash", 0.0)) or 0.0),
        daily_pnl=float(data.get("daily_pnl", 0.0) or 0.0),
        daily_pnl_pct=float(data.get("daily_pnl_pct", 0.0) or 0.0),
        safe_mode=bool(data.get("safe_mode", False)),
        halt_new_entries=bool(data.get("halt_new_entries", False)),
        flatten_in_progress=bool(data.get("flatten_in_progress", False)),
        gross_exposure_pct=float(data.get("gross_exposure_pct", 0.0) or 0.0),
        regime_exposure_cap=float(data.get("regime_exposure_cap", 1.0) or 1.0),
    )


def _coerce_positions(value: Any) -> dict[str, PositionInfo]:
    positions: dict[str, PositionInfo] = {}
    items = _position_items(value)
    for symbol, raw in items:
        key = _symbol(symbol)
        if isinstance(raw, PositionInfo):
            positions[key] = raw
            continue
        data = dict(raw or {})
        allocations: dict[str, AllocationInfo] = {}
        for sid, allocation in dict(data.get("allocations", {}) or {}).items():
            if isinstance(allocation, AllocationInfo):
                allocations[str(sid).upper().strip()] = allocation
            else:
                alloc_data = dict(allocation or {})
                alloc_sid = str(alloc_data.get("strategy_id") or sid).upper().strip()
                allocations[alloc_sid] = AllocationInfo(
                    strategy_id=alloc_sid,
                    qty=int(alloc_data.get("qty", 0) or 0),
                    cost_basis=float(alloc_data.get("cost_basis", 0.0) or 0.0),
                    entry_ts=alloc_data.get("entry_ts"),
                    soft_stop_px=alloc_data.get("soft_stop_px"),
                    time_stop_ts=alloc_data.get("time_stop_ts"),
                )
        positions[key] = PositionInfo(
            symbol=key,
            real_qty=int(data.get("real_qty", data.get("qty", 0)) or 0),
            avg_price=float(data.get("avg_price", data.get("price", 0.0)) or 0.0),
            allocations=allocations,
            hard_stop_px=data.get("hard_stop_px"),
            entry_lock_owner=data.get("entry_lock_owner"),
            entry_lock_until=data.get("entry_lock_until"),
            frozen=bool(data.get("frozen", False)),
            working_order_count=int(data.get("working_order_count", 0) or 0),
        )
    return positions


def _position_items(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, list):
        return [(str(row.get("symbol") or ""), row) for row in value if isinstance(row, dict)]
    return list(dict(value or {}).items())


def _symbol(symbol: str) -> str:
    return str(symbol or "").zfill(6)
