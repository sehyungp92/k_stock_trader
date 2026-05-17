from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from strategy_common.actions import (
    CancelOrders,
    FlattenPosition,
    ReplaceProtectiveStop,
    StrategyAction,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
    SubmitProtectiveStop,
)
from strategy_common.events import TradeOutcome
from strategy_common.market import MarketBar


@dataclass(frozen=True, slots=True)
class BrokerCosts:
    commission_bps: float = 1.5
    tax_bps_on_sell: float = 18.0
    slippage_bps: float = 5.0


@dataclass(slots=True)
class SimOrder:
    order_id: str
    strategy_id: str
    symbol: str
    side: str
    qty: int | None
    order_type: str
    submitted_at: datetime
    reason: str
    limit_price: float | None = None
    stop_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimPosition:
    strategy_id: str
    symbol: str
    qty: int
    avg_price: float
    entry_decision_time: datetime
    entry_fill_time: datetime
    stop_price: float | None = None
    route_metadata: dict[str, Any] = field(default_factory=dict)
    max_price: float = 0.0
    min_price: float = float("inf")

    def mark(self, bar: MarketBar) -> None:
        self.max_price = max(self.max_price, bar.high)
        self.min_price = min(self.min_price, bar.low)


@dataclass(slots=True)
class FillEvent:
    order_id: str
    symbol: str
    side: str
    qty: int
    price: float
    timestamp: datetime
    reason: str


class SimBroker:
    """Single long-only KRX fill model used by all replay runners."""

    def __init__(self, initial_equity: float, costs: BrokerCosts | None = None):
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.costs = costs or BrokerCosts()
        self.orders: list[SimOrder] = []
        self.positions: dict[tuple[str, str], SimPosition] = {}
        self.trades: list[TradeOutcome] = []
        self.fills: list[FillEvent] = []
        self.rejected_orders: list[SimOrder] = []
        self.last_prices: dict[str, float] = {}
        self.equity_curve: list[float] = [float(initial_equity)]
        self.timestamps: list[datetime] = []
        self.same_bar_fill_violations = 0

    def submit(self, action: StrategyAction, submitted_at: datetime) -> str | None:
        if isinstance(action, CancelOrders):
            self.orders = [
                order for order in self.orders
                if not (order.strategy_id == action.strategy_id and order.symbol == action.symbol)
            ]
            return None
        if isinstance(action, ReplaceProtectiveStop):
            key = (action.strategy_id, action.symbol)
            if key in self.positions:
                self.positions[key].stop_price = action.stop_price
            return None
        if isinstance(action, SubmitProtectiveStop):
            order = SimOrder(
                order_id=str(uuid4()),
                strategy_id=action.strategy_id,
                symbol=action.symbol,
                side="SELL",
                qty=action.qty,
                order_type="STOP",
                submitted_at=submitted_at,
                reason=action.reason,
                stop_price=action.stop_price,
                metadata=dict(action.metadata),
            )
            self.orders.append(order)
            return order.order_id
        if isinstance(action, FlattenPosition):
            qty = self.position_qty(action.strategy_id, action.symbol)
            if qty <= 0:
                return None
            order = SimOrder(
                order_id=str(uuid4()),
                strategy_id=action.strategy_id,
                symbol=action.symbol,
                side="SELL",
                qty=qty,
                order_type="MARKET",
                submitted_at=submitted_at,
                reason=action.reason,
                metadata=dict(action.metadata),
            )
            self.orders.append(order)
            return order.order_id
        if isinstance(action, SubmitEntry):
            order = SimOrder(
                order_id=str(uuid4()),
                strategy_id=action.strategy_id,
                symbol=action.symbol,
                side="BUY",
                qty=action.qty,
                order_type=action.order_type,
                submitted_at=submitted_at,
                reason=action.reason,
                limit_price=action.limit_price,
                stop_price=action.stop_price,
                metadata=dict(action.metadata),
            )
            self.orders.append(order)
            return order.order_id
        if isinstance(action, (SubmitExit, SubmitPartialExit)):
            qty = action.qty or self.position_qty(action.strategy_id, action.symbol)
            if qty <= 0:
                return None
            order = SimOrder(
                order_id=str(uuid4()),
                strategy_id=action.strategy_id,
                symbol=action.symbol,
                side="SELL",
                qty=qty,
                order_type=action.order_type,
                submitted_at=submitted_at,
                reason=action.reason,
                limit_price=action.limit_price,
                metadata=dict(action.metadata),
            )
            self.orders.append(order)
            return order.order_id
        raise TypeError(f"Unsupported action: {type(action).__name__}")

    def position_qty(self, strategy_id: str, symbol: str) -> int:
        position = self.positions.get(_position_key(strategy_id, symbol))
        return int(position.qty) if position else 0

    def process_bar(self, bar: MarketBar) -> list[FillEvent]:
        self.last_prices[bar.symbol] = float(bar.close)
        fills: list[FillEvent] = []
        for position in self.positions.values():
            if position.symbol == bar.symbol:
                position.mark(bar)

        remaining_orders: list[SimOrder] = []
        for order in self.orders:
            if order.symbol != bar.symbol:
                remaining_orders.append(order)
                continue
            if order.submitted_at >= bar.timestamp:
                remaining_orders.append(order)
                continue
            fill_price = self._fill_price(order, bar)
            if fill_price is None:
                remaining_orders.append(order)
                continue
            if order.side == "BUY" and not self._can_afford(order, fill_price):
                self.rejected_orders.append(order)
                continue
            fill = self._apply_fill(order, fill_price, bar.timestamp)
            if fill is not None:
                fills.append(fill)
        self.orders = remaining_orders
        self.fills.extend(fills)
        self.mark_to_market(bar)
        return fills

    def mark_to_market(self, bar: MarketBar) -> float:
        self.last_prices[bar.symbol] = float(bar.close)
        equity = self.cash
        for position in self.positions.values():
            if position.symbol == bar.symbol:
                position.mark(bar)
            mark_price = self.last_prices.get(position.symbol, position.avg_price)
            equity += position.qty * mark_price
        self.equity_curve.append(float(equity))
        self.timestamps.append(bar.timestamp)
        return float(equity)

    def close_all_at_end(self, bar: MarketBar, reason: str = "end_of_replay") -> None:
        for key, position in list(self.positions.items()):
            order = SimOrder(
                order_id=str(uuid4()),
                strategy_id=position.strategy_id,
                symbol=position.symbol,
                side="SELL",
                qty=position.qty,
                order_type="MARKET",
                submitted_at=position.entry_fill_time,
                reason=reason,
            )
            mark_price = self.last_prices.get(position.symbol, bar.close)
            self._apply_fill(order, self._sell_slippage(mark_price), bar.timestamp)
            self.positions.pop(key, None)
        self.mark_to_market(bar)

    def _fill_price(self, order: SimOrder, bar: MarketBar) -> float | None:
        if order.side == "BUY":
            if order.order_type == "MARKET":
                return self._buy_slippage(bar.open)
            if order.order_type == "LIMIT" and order.limit_price is not None:
                if bar.low <= order.limit_price:
                    return self._buy_slippage(min(order.limit_price, bar.open))
            if order.order_type in {"STOP", "STOP_LIMIT"} and order.stop_price is not None:
                if bar.high >= order.stop_price:
                    limit = order.limit_price or bar.open
                    return self._buy_slippage(max(order.stop_price, min(limit, bar.high)))
        else:
            if order.order_type == "MARKET":
                return self._sell_slippage(bar.open)
            if order.order_type == "LIMIT" and order.limit_price is not None:
                if bar.high >= order.limit_price:
                    return self._sell_slippage(max(order.limit_price, bar.open))
            if order.order_type == "STOP" and order.stop_price is not None:
                if bar.low <= order.stop_price:
                    return self._sell_slippage(min(order.stop_price, bar.open))
        return None

    def _apply_fill(self, order: SimOrder, price: float, timestamp: datetime) -> FillEvent | None:
        qty = int(order.qty or 0)
        if qty <= 0:
            return None
        if order.side == "BUY":
            commission = self._commission(qty, price, sell=False)
            if self.cash < qty * price + commission:
                self.rejected_orders.append(order)
                return None
            self.cash -= qty * price + commission
            key = _position_key(order.strategy_id, order.symbol)
            current = self.positions.get(key)
            if current is None:
                self.positions[key] = SimPosition(
                    strategy_id=order.strategy_id,
                    symbol=order.symbol,
                    qty=qty,
                    avg_price=price,
                    entry_decision_time=order.submitted_at,
                    entry_fill_time=timestamp,
                    stop_price=order.stop_price,
                    route_metadata={
                        **dict(order.metadata),
                        "entry_commission": commission,
                        "risk_per_share": dict(order.metadata).get("risk_per_share", 0.0),
                    },
                    max_price=price,
                    min_price=price,
                )
            else:
                total_qty = current.qty + qty
                current.avg_price = ((current.avg_price * current.qty) + (price * qty)) / total_qty
                current.qty = total_qty
            return FillEvent(order.order_id, order.symbol, order.side, qty, price, timestamp, order.reason)
        else:
            executed_qty = self._exit_position(order, price, timestamp)
            if executed_qty <= 0:
                return None
            return FillEvent(order.order_id, order.symbol, order.side, executed_qty, price, timestamp, order.reason)

    def _exit_position(self, order: SimOrder, price: float, timestamp: datetime) -> int:
        key = _position_key(order.strategy_id, order.symbol)
        position = self.positions.get(key)
        if position is None:
            return 0
        exit_qty = min(int(order.qty or position.qty), position.qty)
        if exit_qty <= 0:
            return 0
        sell_commission = self._commission(exit_qty, price, sell=True)
        entry_commission = float(position.route_metadata.get("entry_commission", 0.0)) * (exit_qty / position.qty)
        gross = (price - position.avg_price) * exit_qty
        net = gross - entry_commission - sell_commission
        self.cash += exit_qty * price - sell_commission
        mfe = max(0.0, position.max_price - position.avg_price)
        mae = min(0.0, position.min_price - position.avg_price)
        self.trades.append(
            TradeOutcome(
                strategy_id=position.strategy_id,
                symbol=position.symbol,
                qty=exit_qty,
                entry_decision_time=position.entry_decision_time,
                entry_fill_time=position.entry_fill_time,
                entry_price=position.avg_price,
                exit_fill_time=timestamp,
                exit_price=price,
                gross_pnl=gross,
                commission=entry_commission + sell_commission,
                net_pnl=net,
                realized=True,
                exit_reason=order.reason,
                route_metadata=dict(position.route_metadata),
                cohort_metadata=dict(order.metadata),
                mfe=mfe,
                mae=mae,
            )
        )
        position.qty -= exit_qty
        if position.qty <= 0:
            self.positions.pop(key, None)
        else:
            remaining_entry_commission = float(position.route_metadata.get("entry_commission", 0.0)) - entry_commission
            position.route_metadata = {**position.route_metadata, "entry_commission": remaining_entry_commission}
        return exit_qty

    def _can_afford(self, order: SimOrder, price: float) -> bool:
        qty = int(order.qty or 0)
        if qty <= 0:
            return False
        return self.cash >= qty * price + self._commission(qty, price, sell=False)

    def _buy_slippage(self, price: float) -> float:
        return float(price) * (1.0 + self.costs.slippage_bps / 10_000.0)

    def _sell_slippage(self, price: float) -> float:
        return float(price) * (1.0 - self.costs.slippage_bps / 10_000.0)

    def _commission(self, qty: int, price: float, *, sell: bool) -> float:
        bps = self.costs.commission_bps + (self.costs.tax_bps_on_sell if sell else 0.0)
        return qty * price * bps / 10_000.0


def _position_key(strategy_id: str, symbol: str) -> tuple[str, str]:
    return (str(strategy_id).upper(), str(symbol))
