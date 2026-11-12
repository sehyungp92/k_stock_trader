from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, time
from datetime import timedelta
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from oms.stop_protection import (
    LIVE_BACKTEST_STOP_PARITY_VERSION,
    PriceObservation,
    StopProtectionMode,
    StopSide,
    TriggerPriceSource,
    evaluate_stop_trigger,
)
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
from strategy_common.execution_session import bar_execution_issue, bar_reopens_after_label, bar_has_periodic_auction, bar_is_delayed_closing_auction
from .contracts import EXECUTION_VERSION, LEGACY_EXECUTION_VERSION


@dataclass(frozen=True, slots=True)
class BrokerCosts:
    commission_bps: float = 1.5
    tax_bps_on_sell: float = 18.0
    slippage_bps: float = 5.0
    auction_slippage_bps: float = 0.0


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
    strategy_id: str
    symbol: str
    side: str
    qty: int
    price: float
    timestamp: datetime
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SimBroker:
    """Single long-only KRX fill model used by all replay runners."""

    def __init__(self, initial_equity: float, costs: BrokerCosts | None = None, *, buying_power_leverage: float = 1.0, execution_clock: str = EXECUTION_VERSION, capacity_policy=None):
        if execution_clock not in {EXECUTION_VERSION, LEGACY_EXECUTION_VERSION}:
            raise ValueError(f"Unknown execution clock: {execution_clock}")
        self.execution_clock = execution_clock
        self.journal: list[dict[str, Any]] = []
        self.bar_issues: dict[str, int] = {}
        self.last_bar_times: dict[str, datetime] = {}
        self.mark_coverage: list[dict[str, Any]] = []
        self.source_events = hashlib.sha256()
        self.source_event_count = 0
        self.capacity_policy = capacity_policy
        self.capacity_used = {}
        self.previous_capacity_bars = {}
        self.order_filled = {}
        self._phase = "decision"
        self._decision_available_at: datetime | None = None
        self._activation_phase = "decision"
        self._order_sequence = 0
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.costs = costs or BrokerCosts()
        self.buying_power_leverage = max(float(buying_power_leverage), 1.0)
        self.orders: list[SimOrder] = []
        self.positions: dict[tuple[str, str], SimPosition] = {}
        self.trades: list[TradeOutcome] = []
        self.fills: list[FillEvent] = []
        self.rejected_orders: list[SimOrder] = []
        self.expired_orders: list[SimOrder] = []
        self.last_prices: dict[str, float] = {}
        self.last_closing_mark_days: dict[str, Any] = {}
        self.equity_curve: list[float] = [float(initial_equity)]
        self.timestamps: list[datetime] = []
        self.same_bar_fill_violations = 0
        self.same_bar_stop_ambiguities = 0
        self.auction_nonfill_count = 0
        self.same_day_forced_exit_count = 0

    def _next_order_id(self) -> str:
        self._order_sequence += 1
        return f"sim-{self._order_sequence:09d}"

    def submit(self, action: StrategyAction, submitted_at: datetime) -> str | None:
        before = {order.order_id: asdict(order) for order in self.orders}
        order_id = self._submit(action, submitted_at)
        for order in self.orders:
            if order.side == "SELL" and order.order_type == "STOP":
                order.metadata = _with_stop_parity_metadata(order.metadata)
            old = before.pop(order.order_id, None)
            if old is None:
                order.metadata["order_date"] = submitted_at.date().isoformat()
            else:
                order.metadata["order_date"] = old["metadata"].get("order_date", old["submitted_at"].date().isoformat())
            if old is None or old != asdict(order):
                order.metadata = {**order.metadata,
                    "activation_phase": self._activation_phase,
                    "activation_at": submitted_at.isoformat(),
                    "decision_available_at": (self._decision_available_at or submitted_at).isoformat()}
                self.journal.append({"event": "submit" if old is None else "amend", "order": asdict(order)})
        for old in before.values():
            self.journal.append({"event": "cancel", "order": old, "timestamp": submitted_at.isoformat()})
        return order_id

    def _submit(self, action: StrategyAction, submitted_at: datetime) -> str | None:
        if isinstance(action, CancelOrders):
            self.orders = [
                order for order in self.orders
                if not (order.strategy_id == action.strategy_id and order.symbol == action.symbol)
            ]
            return None
        if isinstance(action, ReplaceProtectiveStop):
            key = (action.strategy_id, action.symbol)
            position = self.positions.get(key)
            if key in self.positions:
                position = self.positions[key]
                position.stop_price = action.stop_price
                position.route_metadata = {
                    **position.route_metadata,
                    "current_stop": action.stop_price,
                    "stop_protected": True,
                }
            matched = False
            for order in self.orders:
                if order.strategy_id == action.strategy_id and order.symbol == action.symbol and order.side == "SELL" and order.order_type == "STOP":
                    matched = True
                    order.submitted_at = submitted_at
                    order.stop_price = action.stop_price
                    order.qty = action.qty or order.qty
                    order.reason = action.reason
                    order.metadata = {**order.metadata, **dict(action.metadata)}
            if matched or position is None:
                return None
            qty = int(action.qty or position.qty)
            if qty <= 0:
                return None
            metadata = {"order_role": "STOP", "stop_kind": "protected_stop", **dict(action.metadata)}
            order = SimOrder(
                order_id=self._next_order_id(),
                strategy_id=action.strategy_id,
                symbol=action.symbol,
                side="SELL",
                qty=qty,
                order_type="STOP",
                submitted_at=submitted_at,
                reason=action.reason,
                stop_price=action.stop_price,
                metadata=metadata,
            )
            self.orders.append(order)
            return order.order_id
        if isinstance(action, SubmitProtectiveStop):
            order = SimOrder(
                order_id=self._next_order_id(),
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
            metadata = dict(action.metadata)
            order_type = str(metadata.get("order_type") or metadata.get("flatten_order_type") or "MARKET").upper()
            if order_type not in {"MARKET", "CLOSE_AUCTION"}:
                order_type = "MARKET"
            order = SimOrder(
                order_id=self._next_order_id(),
                strategy_id=action.strategy_id,
                symbol=action.symbol,
                side="SELL",
                qty=qty,
                order_type=order_type,
                submitted_at=submitted_at,
                reason=action.reason,
                metadata=metadata,
            )
            self.orders.append(order)
            return order.order_id
        if isinstance(action, SubmitEntry):
            order = SimOrder(
                order_id=self._next_order_id(),
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
                order_id=self._next_order_id(),
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

    def _phase_order_key(self, order):
        return (0 if order.side == "SELL" and order.order_type == "STOP" else 1 if order.side == "SELL" else 2,
                order.submitted_at, order.strategy_id, order.symbol, order.order_id)

    def _policy_execution_quantity(self, order, price, quantity, timestamp):
        return quantity

    def _legacy_process_bar(self, bar: MarketBar) -> list[FillEvent]:
        if bar_execution_issue(bar) or bar_reopens_after_label(bar) or bar_has_periodic_auction(bar):
            return []
        self.last_prices[bar.symbol] = float(bar.close)
        fills: list[FillEvent] = []
        for position in self.positions.values():
            if position.symbol == bar.symbol:
                position.mark(bar)

        remaining_orders: list[SimOrder] = []
        eligible_orders: list[SimOrder] = []
        for order in self.orders:
            if order.symbol != bar.symbol:
                remaining_orders.append(order)
                continue
            if _order_expired(order, bar.timestamp):
                self.expired_orders.append(order)
                if order.order_type == "CLOSE_AUCTION":
                    self.auction_nonfill_count += 1
                continue
            if order.submitted_at >= bar.timestamp:
                remaining_orders.append(order)
                continue
            eligible_orders.append(order)
        eligible_orders.sort(key=lambda order: _execution_priority(order, bar))
        for order in eligible_orders:
            if bar_is_delayed_closing_auction(bar) and order.order_type != "CLOSE_AUCTION":
                remaining_orders.append(order)
                continue
            fill_price = self._fill_price(order, bar)
            if fill_price is None:
                if order.order_type == "CLOSE_AUCTION" and _is_close_auction_bar(order, bar):
                    self.expired_orders.append(order)
                    self.auction_nonfill_count += 1
                else:
                    remaining_orders.append(order)
                continue
            if order.side == "BUY" and not self._can_afford(order, fill_price):
                self.rejected_orders.append(order)
                continue
            fill = self._apply_fill(order, fill_price, bar.timestamp)
            if fill is not None:
                fill.metadata.update({"execution_id": f"legacy-fill-{len(self.fills) + len(fills) + 1}",
                    "order_date": order.metadata.get("order_date", order.submitted_at.date().isoformat()),
                    "order_qty": order.qty, "filled_qty_before": 0, "filled_qty_after": fill.qty})
                fills.append(fill)
                ambiguity_fill = self._maybe_apply_same_bar_entry_stop(order, fill, bar)
                if ambiguity_fill is not None:
                    fills.append(ambiguity_fill)
        self.orders = remaining_orders
        self.fills.extend(fills)
        self._legacy_mark_to_market(bar)
        return fills

    def _legacy_mark_to_market(self, bar: MarketBar) -> float:
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

    def process_bar(self, bar: MarketBar) -> list[FillEvent]:
        if self.execution_clock == LEGACY_EXECUTION_VERSION:
            return self._legacy_process_bar(bar)
        return self.process_batch((bar,))

    def process_batch(self, bars, *, on_fills=None) -> list[FillEvent]:
        """Opening flows, intrabar stop-first, auction, then synchronized marks.

        Priority is sell before buy, then submission time, strategy, symbol and
        submission sequence. Intrabar ambiguity is resolved adversely (stop first).
        """
        bars = tuple(bars)
        if not bars:
            return []
        timestamp = bars[0].timestamp
        if any(bar.timestamp != timestamp for bar in bars) or len({bar.symbol for bar in bars}) != len(bars):
            raise ValueError("A batch requires one bar per symbol at one timestamp")
        if self.timestamps and timestamp <= self.timestamps[-1]:
            raise ValueError("Replay batches must have strictly increasing timestamps")
        start = len(self.fills)
        executable = {}
        closing_bars = {}
        for bar in bars:
            self.source_events.update(json.dumps(bar.to_json_dict(), sort_keys=True, default=str).encode())
            self.source_event_count += 1
            issue = bar_execution_issue(bar)
            if issue:
                self.bar_issues[issue] = self.bar_issues.get(issue, 0) + 1
            else:
                closing_bars[bar.symbol] = bar
                self.last_bar_times[bar.symbol] = timestamp
                if bar_has_periodic_auction(bar):
                    if any(order.symbol == bar.symbol and not _order_expired(order, timestamp) for order in self.orders):
                        self.bar_issues["unsupported_periodic_auction_execution"] = self.bar_issues.get("unsupported_periodic_auction_execution", 0) + 1
                elif not bar_reopens_after_label(bar):
                    executable[bar.symbol] = bar
                    if not bar_is_delayed_closing_auction(bar):
                        self.last_prices[bar.symbol] = float(bar.open)
                elif any(order.symbol == bar.symbol and not _order_expired(order, timestamp) for order in self.orders):
                    # Five-minute OHLC cannot order this resumption auction against
                    # the other symbols' earlier opening flows. Do not invent a fill.
                    self.bar_issues["unsupported_halt_boundary_chronology"] = self.bar_issues.get("unsupported_halt_boundary_chronology", 0) + 1
        for order in list(self.orders):
            if _order_expired(order, timestamp):
                self.orders.remove(order)
                self.expired_orders.append(order)
                self.journal.append({"event": "expire", "order": asdict(order), "timestamp": timestamp.isoformat()})
                self.auction_nonfill_count += int(order.order_type == "CLOSE_AUCTION")
        self.journal.append({"event": "opening_marks", "timestamp": timestamp.isoformat(),
            "prices": {symbol: float(bar.open) for symbol, bar in executable.items() if not bar_is_delayed_closing_auction(bar)}})
        for phase in ("open", "intrabar", "auction"):
            self._phase = phase
            if self.execution_clock == LEGACY_EXECUTION_VERSION:
                for bar in executable.values():
                    self.last_prices[bar.symbol] = bar.close
            candidates = sorted(list(self.orders), key=self._phase_order_key)
            for order in candidates:
                if order not in self.orders:
                    continue
                bar = executable.get(order.symbol)
                if bar is None:
                    continue
                if bar_is_delayed_closing_auction(bar) and order.order_type != "CLOSE_AUCTION":
                    continue
                available = datetime.fromisoformat(order.metadata.get("decision_available_at", order.submitted_at.isoformat()))
                same_open_activation = (order.submitted_at == timestamp
                    and order.metadata.get("activation_phase") == "open_fill"
                    and order.side == "SELL")
                if available > timestamp or (order.submitted_at >= timestamp and not same_open_activation):
                    continue
                if order.order_type == "STOP_LIMIT":
                    self.orders.remove(order)
                    self.rejected_orders.append(order)
                    self.journal.append({"event": "reject", "order": asdict(order), "reason": "unsupported_stop_limit"})
                    continue
                opening = order.order_type == "MARKET" or (
                    order.order_type == "STOP" and order.stop_price is not None and
                    (bar.open <= order.stop_price if order.side == "SELL" else bar.open >= order.stop_price)) or (
                    order.order_type == "LIMIT" and order.limit_price is not None and
                    (bar.open >= order.limit_price if order.side == "SELL" else bar.open <= order.limit_price))
                intended_phase = "auction" if order.order_type == "CLOSE_AUCTION" else "open" if opening and not same_open_activation else "intrabar"
                if intended_phase != phase:
                    continue
                # Newly activated protection cannot claim an earlier opening gap.
                pricing_bar = replace(bar, open=order.stop_price) if same_open_activation and order.order_type == "STOP" else bar
                price = self._fill_price(order, pricing_bar)
                if price is None:
                    if phase == "auction" and _is_close_auction_bar(order, bar):
                        self.orders.remove(order)
                        self.expired_orders.append(order)
                        self.auction_nonfill_count += 1
                        self.journal.append({"event": "expire", "order": asdict(order), "timestamp": timestamp.isoformat()})
                    continue
                capacity_key = (bar.symbol, timestamp.isoformat(), phase)
                allowed_qty = int(order.qty or self.position_qty(order.strategy_id, order.symbol))
                budget = None
                if self.capacity_policy is not None:
                    quote = self.capacity_policy.quote(bar, phase, self.previous_capacity_bars.get(bar.symbol))
                    budget = quote.budget
                    allowed_qty = min(allowed_qty, max(0, budget - self.capacity_used.get(capacity_key, 0)))
                    # Later interval volume cannot determine an opening callback.
                    # The independent auditor checks realized feasibility afterward.
                    self.journal.append({"event": "capacity", "order": asdict(order),
                        "timestamp": timestamp.isoformat(), "phase": phase,
                        "requested_qty": int(order.qty or self.position_qty(order.strategy_id, order.symbol)),
                        "allowed_qty": allowed_qty, "used_qty": self.capacity_used.get(capacity_key, 0),
                        "quote": asdict(quote)})
                    if not allowed_qty:
                        if phase == "auction":
                            self.orders.remove(order)
                            self.expired_orders.append(order)
                            self.auction_nonfill_count += 1
                            self.journal.append({"event": "expire", "order": asdict(order), "reason": "auction_capacity", "timestamp": timestamp.isoformat()})
                        continue
                self.orders.remove(order)
                allowed_qty = self._policy_execution_quantity(order, price, allowed_qty, timestamp)
                if allowed_qty <= 0:
                    self.rejected_orders.append(order)
                    self.journal.append({"event": "reject", "order": asdict(order), "reason": "portfolio_policy", "timestamp": timestamp.isoformat()})
                    continue
                if order.side == "BUY" and not self._can_afford(replace(order, qty=allowed_qty), price):
                    self.rejected_orders.append(order)
                    self.journal.append({"event": "reject", "order": asdict(order), "reason": "buying_power", "timestamp": timestamp.isoformat(), "cash": self.cash, "marks": dict(self.last_prices),
                        "attempted_qty": allowed_qty, "execution_price": price, "execution_bar": bar.to_json_dict()})
                    continue
                if order.side == "SELL" and not self.position_qty(order.strategy_id, order.symbol):
                    self.journal.append({"event": "cancel", "order": asdict(order), "reason": "position_closed"})
                    continue
                if order.side == "SELL" and int(order.qty or 0) > self.position_qty(order.strategy_id, order.symbol):
                    # Alternative exits reserve the same shares. Resize explicitly.
                    order.qty = self.position_qty(order.strategy_id, order.symbol)
                    self.journal.append({"event": "amend", "order": asdict(order), "reason": "alternative_exit_resize"})
                costs = getattr(self, "strategy_costs", {}).get(order.strategy_id, self.costs)
                bps = costs.slippage_bps
                if phase == "auction":
                    bps = max(0.0, bps + float(order.metadata.get("auction_adverse_bps", costs.auction_slippage_bps) or 0))
                if order.order_type == "MARKET":
                    base = bar.open
                elif order.order_type == "CLOSE_AUCTION":
                    base = bar.close
                elif order.order_type == "LIMIT":
                    base = min(order.limit_price, bar.open) if order.side == "BUY" else max(order.limit_price, bar.open)
                else:
                    base = float(order.stop_price) if same_open_activation else (max(order.stop_price, bar.open) if order.side == "BUY" else min(order.stop_price, bar.open))
                cash_before = self.cash
                order_evidence = asdict(order)
                allowed_qty = min(allowed_qty, int(order.qty or allowed_qty))
                fill = self._apply_fill(replace(order, qty=allowed_qty), price, timestamp)
                if fill is None:
                    continue
                if fill.side == "BUY":
                    self.positions[(fill.strategy_id, fill.symbol)].route_metadata["entry_execution_phase"] = phase
                fee = fill.qty * price * (costs.commission_bps + (costs.tax_bps_on_sell if fill.side == "SELL" else 0)) / 10000
                before = self.order_filled.get(order.order_id, 0)
                fill.metadata.update({"execution_id": f"fill-{len(self.fills) + 1}", "order_qty": before + int(order.qty or fill.qty),
                    "order_date": order.metadata.get("order_date", order.submitted_at.date().isoformat()),
                    "filled_qty_before": before, "filled_qty_after": before + fill.qty,
                    "execution_phase": phase, "base_price": base, "fee": fee,
                    "execution_bar": bar.to_json_dict(), "submitted_at": order.submitted_at.isoformat()})
                self.fills.append(fill)
                self.journal.append({"event": "fill", "order": order_evidence, "fill": asdict(fill),
                    "costs": asdict(costs), "cash_before": cash_before, "cash_after": self.cash,
                    "marks": dict(self.last_prices)})
                self.order_filled[order.order_id] = before + fill.qty
                self.capacity_used[capacity_key] = self.capacity_used.get(capacity_key, 0) + fill.qty
                if order.qty and fill.qty < order.qty:
                    order.qty -= fill.qty
                    if phase == "auction":
                        self.expired_orders.append(order)
                        self.journal.append({"event": "expire", "order": asdict(order), "reason": "auction_remainder", "timestamp": timestamp.isoformat()})
                    else:
                        self.orders.append(order)
                        self.journal.append({"event": "amend", "order": asdict(order), "reason": "partial_remainder"})
                if on_fills:
                    self._activation_phase = "open_fill" if phase == "open" else "post_interval_fill"
                    self._decision_available_at = None
                    on_fills()
                    self._activation_phase = "decision"
                if phase == "intrabar" and fill.side == "BUY":
                    # General intrabar entry protection needs finer event ordering.
                    self.bar_issues["unsupported_intrabar_entry_chronology"] = self.bar_issues.get("unsupported_intrabar_entry_chronology", 0) + 1
        for bar in closing_bars.values():
            self.previous_capacity_bars[bar.symbol] = bar
            self.last_prices[bar.symbol] = float(bar.close)
            for position in self.positions.values():
                if position.symbol == bar.symbol and (position.entry_fill_time != timestamp
                        or position.route_metadata.get("entry_execution_phase") == "open"):
                    position.mark(bar)
        self.journal.append({"event": "closing_marks", "timestamp": timestamp.isoformat(),
            "prices": {symbol: float(bar.close) for symbol, bar in closing_bars.items()}})
        self.equity_curve.append(self._portfolio_equity())
        self.timestamps.append(timestamp)
        from kis_core.trading_calendar import get_trading_calendar
        calendar = get_trading_calendar()
        for bar in closing_bars.values():
            if bar.timestamp.time().replace(tzinfo=None) == calendar.closing_match_time(bar.symbol, timestamp.date()):
                self.last_closing_mark_days[bar.symbol] = timestamp.date()
        closed_symbols = {symbol for symbol, day in self.last_closing_mark_days.items() if day == timestamp.date()}
        stale = sorted({p.symbol for p in self.positions.values()
                        if self.last_bar_times.get(p.symbol) != timestamp and p.symbol not in closed_symbols})
        closing_complete = all(p.symbol in closed_symbols for p in self.positions.values())
        self.mark_coverage.append({"timestamp": timestamp.isoformat(), "stale_symbols": stale,
            "cash_only": not self.positions, "closing_marks_complete": closing_complete})
        self._phase = "decision"
        return self.fills[start:]

    def mark_to_market(self, bar: MarketBar) -> float:
        if bar_execution_issue(bar) is None:
            self.last_prices[bar.symbol] = float(bar.close)
        return self._portfolio_equity()

    def close_all_at_end(self, bar: MarketBar, reason: str = "end_of_replay") -> None:
        # A valuation mark is not a liquidation execution. Retain unresolved holdings.
        if self.positions:
            self.journal.append({"event": "unresolved_terminal", "timestamp": bar.timestamp.isoformat(),
                "reason": reason, "positions": {str(k): v.qty for k, v in self.positions.items()}})

    def force_same_day_exits(self, bar: MarketBar) -> list[FillEvent]:
        # Already eligible orders executed in process_batch against their own symbol.
        # Missing execution data must remain an unresolved order/position.
        return []

    def _fill_price(self, order: SimOrder, bar: MarketBar) -> float | None:
        if order.order_type == "CLOSE_AUCTION":
            return self._close_auction_fill_price(order, bar)
        if order.side == "BUY":
            if order.order_type == "MARKET":
                return self._buy_slippage(bar.open)
            if order.order_type == "LIMIT" and order.limit_price is not None:
                if bar.low <= order.limit_price:
                    return min(float(order.limit_price), self._buy_slippage(min(order.limit_price, bar.open)))
            if order.order_type == "STOP" and order.stop_price is not None:
                if bar.high >= order.stop_price:
                    return self._buy_slippage(max(order.stop_price, bar.open))
        else:
            if order.order_type == "MARKET":
                return self._sell_slippage(bar.open)
            if order.order_type == "LIMIT" and order.limit_price is not None:
                if bar.high >= order.limit_price:
                    return max(float(order.limit_price), self._sell_slippage(max(order.limit_price, bar.open)))
            if order.order_type == "STOP" and order.stop_price is not None:
                decision = _sell_stop_trigger_decision(order, bar)
                if decision.triggered:
                    return self._sell_slippage(min(order.stop_price, bar.open))
        return None

    def _close_auction_fill_price(self, order: SimOrder, bar: MarketBar) -> float | None:
        if not _is_close_auction_bar(order, bar):
            return None
        if _auction_forced_nonfill(order, bar):
            return None
        close = float(bar.close)
        adverse_bps = float(order.metadata.get("auction_adverse_bps", self.costs.auction_slippage_bps) or 0.0)
        total_bps = max(0.0, float(self.costs.slippage_bps) + adverse_bps)
        if order.side == "BUY":
            fill_price = close * (1.0 + total_bps / 10_000.0)
            if order.limit_price is not None and fill_price > float(order.limit_price):
                return None
            return fill_price
        fill_price = close * (1.0 - total_bps / 10_000.0)
        if order.limit_price is not None and fill_price < float(order.limit_price):
            return None
        return fill_price

    def _apply_fill(self, order: SimOrder, price: float, timestamp: datetime) -> FillEvent | None:
        qty = int(order.qty or 0)
        if qty <= 0:
            return None
        if order.side == "BUY":
            commission = self._commission(qty, price, sell=False)
            if not self._can_afford(order, price):
                self.rejected_orders.append(order)
                return None
            self.cash -= qty * price + commission
            key = _position_key(order.strategy_id, order.symbol)
            current = self.positions.get(key)
            if current is None:
                protective_stop = _metadata_float(order.metadata, "protective_stop_price")
                if protective_stop is None:
                    protective_stop = _metadata_float(order.metadata, "stop_price")
                self.positions[key] = SimPosition(
                    strategy_id=order.strategy_id,
                    symbol=order.symbol,
                    qty=qty,
                    avg_price=price,
                    entry_decision_time=order.submitted_at,
                    entry_fill_time=timestamp,
                    stop_price=protective_stop if protective_stop is not None else order.stop_price,
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
                current.route_metadata["entry_commission"] = float(current.route_metadata.get("entry_commission", 0)) + commission
            return FillEvent(order.order_id, order.strategy_id, order.symbol, order.side, qty, price, timestamp, order.reason, dict(order.metadata))
        else:
            executed_qty = self._exit_position(order, price, timestamp)
            if executed_qty <= 0:
                return None
            return FillEvent(order.order_id, order.strategy_id, order.symbol, order.side, executed_qty, price, timestamp, order.reason, dict(order.metadata))

    def _maybe_apply_same_bar_entry_stop(self, order: SimOrder, fill: FillEvent, bar: MarketBar) -> FillEvent | None:
        if order.side != "BUY" or order.order_type not in {"STOP", "STOP_LIMIT"}:
            return None
        metadata = dict(order.metadata)
        if not bool(metadata.get("same_bar_stop_first", True)):
            return None
        protective = float(metadata.get("protective_stop_price") or metadata.get("stop_price") or 0.0)
        if protective <= 0.0 or protective >= float(fill.price):
            return None
        decision = _long_bar_low_stop_decision(protective, order, bar)
        if not decision.triggered:
            return None
        self.same_bar_stop_ambiguities += 1
        fill.metadata["same_bar_stop_fired"] = True
        stop_order = SimOrder(
            order_id=self._next_order_id(),
            strategy_id=order.strategy_id,
            symbol=order.symbol,
            side="SELL",
            qty=fill.qty,
            order_type="STOP",
            submitted_at=order.submitted_at,
            reason="same_bar_stop_first",
            stop_price=protective,
            metadata={
                "order_role": "STOP",
                "stop_kind": "same_bar_stop_first",
                "same_bar_ambiguity": True,
                **_stop_parity_metadata(),
                **metadata,
            },
        )
        exit_price = self._sell_slippage(min(protective, float(bar.open)))
        return self._apply_fill(stop_order, exit_price, bar.timestamp)

    def _exit_position(self, order: SimOrder, price: float, timestamp: datetime) -> int:
        key = _position_key(order.strategy_id, order.symbol)
        position = self.positions.get(key)
        if position is None:
            return 0
        exit_qty = int(order.qty or position.qty)
        if exit_qty > position.qty:
            raise ValueError("Sell execution exceeds the open position")
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
                cohort_metadata={**order.metadata, "position_qty_after": position.qty - exit_qty},
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
        required = qty * price + self._commission(qty, price, sell=False)
        if self.buying_power_leverage <= 1.0:
            return self.cash >= required
        return self._buying_power_available() >= required

    def _buy_slippage(self, price: float) -> float:
        return float(price) * (1.0 + self.costs.slippage_bps / 10_000.0)

    def _sell_slippage(self, price: float) -> float:
        return float(price) * (1.0 - self.costs.slippage_bps / 10_000.0)

    def _commission(self, qty: int, price: float, *, sell: bool) -> float:
        bps = self.costs.commission_bps + (self.costs.tax_bps_on_sell if sell else 0.0)
        return qty * price * bps / 10_000.0

    def _portfolio_equity(self) -> float:
        equity = float(self.cash)
        for position in self.positions.values():
            mark_price = self.last_prices.get(position.symbol, position.avg_price)
            equity += float(position.qty) * float(mark_price)
        return float(equity)

    def _open_notional(self) -> float:
        total = 0.0
        for position in self.positions.values():
            mark_price = self.last_prices.get(position.symbol, position.avg_price)
            total += float(position.qty) * float(mark_price)
        return float(total)

    def _buying_power_available(self) -> float:
        if self.buying_power_leverage <= 1.0:
            return max(float(self.cash), 0.0)
        return max(self._portfolio_equity() * self.buying_power_leverage - self._open_notional(), 0.0)


def _position_key(strategy_id: str, symbol: str) -> tuple[str, str]:
    return (str(strategy_id).upper(), str(symbol))


def _order_expired(order: SimOrder, timestamp: datetime) -> bool:
    if timestamp.date() > order.submitted_at.date() and _same_day_forced_exit_order(order, order.submitted_at):
        return True
    raw = order.metadata.get("expiry_timestamp") or order.metadata.get("expiry_ts")
    if not raw:
        if order.order_type != "CLOSE_AUCTION":
            return False
        target = _auction_target(order)
        submitted = order.submitted_at
        current = timestamp
        if submitted.tzinfo is not None and current.tzinfo is not None:
            submitted = submitted.astimezone(current.tzinfo)
        return current.date() > submitted.date() or (current.date() == submitted.date() and current.time().replace(second=0, microsecond=0) > target)
    try:
        expiry = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return False
    if expiry.tzinfo is None and timestamp.tzinfo is not None:
        expiry = expiry.replace(tzinfo=timestamp.tzinfo)
    if order.order_type == "CLOSE_AUCTION" and order.metadata.get("expiry_basis") == "closing_auction":
        scheduled = _parse_time(order.metadata.get("auction_fill_time", "15:30"))
        if expiry.date() == order.submitted_at.date() and expiry.time().replace(tzinfo=None) == scheduled:
            expiry = datetime.combine(expiry.date(), _auction_target(order), tzinfo=expiry.tzinfo)
    return timestamp > expiry


def _same_day_forced_exit_order(order: SimOrder, timestamp: datetime) -> bool:
    if order.side != "SELL":
        return False
    if order.submitted_at.date() != timestamp.date():
        return False
    metadata = dict(order.metadata)
    return bool(
        metadata.get("same_day_force_exit")
        or metadata.get("same_day_only")
        or metadata.get("eod_same_day_only")
    )


def _metadata_float(metadata: dict[str, Any], key: str) -> float | None:
    try:
        value = metadata.get(key)
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _long_bar_low_stop_decision(stop_price: float, order: SimOrder, bar: MarketBar):
    return evaluate_stop_trigger(
        stop_price=float(stop_price),
        side=StopSide.LONG.value,
        observation=PriceObservation(
            symbol=order.symbol,
            price=float(bar.low),
            timestamp=bar.timestamp.timestamp(),
            source=TriggerPriceSource.BAR_LOW.value,
            market_open=True,
            executable=True,
        ),
        stale_after_sec=0.0,
        now=bar.timestamp.timestamp(),
    )


def _sell_stop_trigger_decision(order: SimOrder, bar: MarketBar):
    return _long_bar_low_stop_decision(float(order.stop_price or 0.0), order, bar)


def _stop_parity_metadata() -> dict[str, Any]:
    return {
        "stop_protection_mode": StopProtectionMode.OMS_WATCHER.value,
        "stop_trigger_price_source": TriggerPriceSource.BAR_LOW.value,
        "stop_fill_model": "sell_stop_fills_at_stop_or_bar_open_gap_through_with_slippage",
        "live_backtest_stop_parity_version": LIVE_BACKTEST_STOP_PARITY_VERSION,
    }


def _with_stop_parity_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {**dict(metadata), **_stop_parity_metadata()}


def _execution_priority(order: SimOrder, bar: MarketBar) -> tuple[int, datetime, str]:
    if (
        order.side == "SELL"
        and order.order_type == "STOP"
        and order.stop_price is not None
        and _sell_stop_trigger_decision(order, bar).triggered
    ):
        return (0, order.submitted_at, order.order_id)
    if order.side == "SELL" and order.order_type == "MARKET":
        return (1, order.submitted_at, order.order_id)
    if order.side == "SELL" and order.order_type == "CLOSE_AUCTION" and _is_close_auction_bar(order, bar):
        return (2, order.submitted_at, order.order_id)
    if order.side == "SELL" and order.order_type == "LIMIT" and order.limit_price is not None and bar.high >= order.limit_price:
        return (3, order.submitted_at, order.order_id)
    return (3, order.submitted_at, order.order_id)


_KST = ZoneInfo("Asia/Seoul")


def _is_close_auction_bar(order: SimOrder, bar: MarketBar) -> bool:
    target = _auction_target(order)
    timestamp = bar.timestamp
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(_KST)
    return timestamp.time().replace(second=0, microsecond=0) == target


def _auction_target(order: SimOrder) -> time:
    target = _parse_time(order.metadata.get("auction_fill_time", "15:30"))
    if order.metadata.get("expiry_basis") == "closing_auction":
        from kis_core.trading_calendar import get_trading_calendar
        calendar = get_trading_calendar()
        day = order.submitted_at.astimezone(_KST).date() if order.submitted_at.tzinfo else order.submitted_at.date()
        if target == calendar.session_hours(day)[1]:
            return calendar.closing_match_time(order.symbol, day)
    return target


def _parse_time(value: Any) -> time:
    if isinstance(value, time):
        return value.replace(second=0, microsecond=0)
    text = str(value or "15:30")
    try:
        hour, minute, *_ = text.split(":")
        return time(int(hour), int(minute))
    except (TypeError, ValueError):
        return time(15, 30)


def _auction_forced_nonfill(order: SimOrder, bar: MarketBar) -> bool:
    try:
        rate = float(order.metadata.get("auction_nonfill_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        rate = 0.0
    if rate <= 0.0:
        return False
    threshold = min(max(rate, 0.0), 1.0)
    key = str(
        order.metadata.get("auction_nonfill_key")
        or f"{bar.timestamp.date()}:{order.strategy_id}:{order.symbol}:{order.reason}:{order.order_id}"
    )
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return bucket < threshold
