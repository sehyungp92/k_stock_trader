from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from itertools import groupby
from datetime import timedelta

from .contracts import EXECUTION_VERSION, LEGACY_EXECUTION_VERSION

from strategy_common.events import DecisionEvent, TradeOutcome
from strategy_common.market import MarketBar, require_completed_bar

from .sim_broker import BrokerCosts, SimBroker


class StrategyReplayAdapter(Protocol):
    strategy_id: str

    def on_bar(self, bar: MarketBar, broker: SimBroker) -> list[DecisionEvent]: ...


@dataclass(slots=True)
class ReplayResult:
    trades: list[TradeOutcome]
    decisions: list[DecisionEvent]
    equity_curve: list[float]
    timestamps: list
    broker: SimBroker


def candidate_management_bars(bars_by_key, snapshots, dates):
    """Retain observations that may be needed after a capacity-limited exit."""
    first = {}
    for day, snapshot in snapshots.items():
        for candidate in snapshot.candidates:
            first[candidate.symbol] = min(first.get(candidate.symbol, day), day)
    days = set(dates)
    return tuple(sorted((bar for (day, symbol), rows in bars_by_key.items()
        if day in days and symbol in first and day >= first[symbol] for bar in rows),
        key=lambda bar: (bar.timestamp, bar.symbol)))


def run_replay(
    bars: list[MarketBar],
    adapter: StrategyReplayAdapter,
    *,
    initial_equity: float = 10_000_000.0,
    costs: BrokerCosts | None = None,
    close_open_positions: bool = True,
    bars_are_ordered: bool = False,
    buying_power_leverage: float = 1.0,
    execution_clock: str = EXECUTION_VERSION,
    capacity_policy=None,
    candidate_scoped: bool = False,
) -> ReplayResult:
    broker = SimBroker(initial_equity=initial_equity, costs=costs, buying_power_leverage=buying_power_leverage, execution_clock=execution_clock, capacity_policy=capacity_policy)
    decisions: list[DecisionEvent] = []
    validated = [require_completed_bar(bar) for bar in bars]
    ordered = validated if bars_are_ordered else sorted(validated, key=lambda bar: (bar.timestamp, bar.symbol))
    if any(a.timestamp > b.timestamp for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Ordered replay input is not time ordered")
    candidate_symbols = {day: {candidate.symbol for candidate in snapshot.candidates}
        for day, snapshot in getattr(adapter, "snapshots", {}).items()} if candidate_scoped else {}
    for timestamp, grouped in groupby(ordered, key=lambda bar: bar.timestamp):
        batch = tuple(grouped)
        if candidate_scoped:
            active = candidate_symbols.get(timestamp.date(), set()) | {
                symbol for _, symbol in broker.positions} | {order.symbol for order in broker.orders}
            batch = tuple(bar for bar in batch if bar.symbol in active)
            if not batch:
                continue
        decisions.extend(drive_timestamp_batch(broker, batch, (adapter,)))
    if close_open_positions and ordered:
        broker.close_all_at_end(ordered[-1])
    return ReplayResult(
        trades=list(broker.trades),
        decisions=decisions,
        equity_curve=list(broker.equity_curve),
        timestamps=list(broker.timestamps),
        broker=broker,
    )


def drive_timestamp_batch(broker: SimBroker, bars, adapters, *, accepts=None) -> list[DecisionEvent]:
    """The only full-account interval driver, shared by standalone and native replay."""
    bars = tuple(bars)
    if not hasattr(broker, "expected_batches"):
        broker.expected_batches = []
        broker.capacity_shadow_brokers = []
        for adapter in adapters:
            shadow = getattr(adapter, "frontier_shadow", None)
            if shadow is not None:
                shadow.broker.capacity_policy = broker.capacity_policy
                shadow.broker.execution_clock = broker.execution_clock
                broker.capacity_shadow_brokers.append(shadow.broker)
        from backtests.auto.shared.cache_keys import stable_signature
        def audit_snapshots(adapter):
            provider=getattr(adapter,'qualification_snapshot_sequence',None)
            return tuple(provider()) if provider is not None else tuple(
                snapshot for _,snapshot in sorted(getattr(adapter,'snapshots',{}).items()))
        snapshot_rows=[[snapshot.to_json_dict() for snapshot in audit_snapshots(adapter)] for adapter in adapters]
        snapshot_scopes={getattr(adapter,'strategy_id',f'unbound_adapter:{i}'):rows
            for i,(adapter,rows) in enumerate(zip(adapters,snapshot_rows))}
        snapshots = [snapshot for rows in snapshot_rows for snapshot in rows]
        broker.qualification_snapshot_hash = stable_signature(snapshots)
        broker.qualification_snapshots = snapshots
        broker.qualification_snapshot_scopes = snapshot_scopes
    # Bound before the broker can mutate marks, orders or execution evidence.
    broker.expected_batches.append(tuple(bar.to_json_dict() for bar in bars))
    if not hasattr(broker, "expected_strategy_state_timestamps"):
        broker.expected_strategy_state_timestamps = {}
    for adapter in adapters:
        if getattr(adapter, "records_strategy_state", False):
            expected = broker.expected_strategy_state_timestamps.setdefault(adapter.strategy_id, [])
            if any(accepts is None or accepts(adapter, bar) for bar in bars):
                expected.append(bars[0].timestamp.isoformat())
    decisions = []
    def deliver_fills():
        for adapter in adapters:
            hook = getattr(adapter, "on_fills", None)
            if hook is not None:
                decisions.extend(hook(broker))
    if broker.execution_clock == LEGACY_EXECUTION_VERSION:
        for bar in bars:
            broker.process_bar(bar)
            for adapter in adapters:
                if accepts is None or accepts(adapter, bar):
                    decisions.extend(adapter.on_bar(bar, broker))
        for adapter in adapters:
            selected = tuple(bar for bar in bars if accepts is None or accepts(adapter, bar))
            hook = getattr(adapter, "on_timestamp_end", None)
            if selected and callable(hook):
                decisions.extend(hook(selected[-1].timestamp, selected, broker))
        return decisions
    broker.process_batch(bars, on_fills=deliver_fills)
    finalize = getattr(broker, "finalize_entry_intents", None)
    deferred_end_hooks = []
    for adapter in adapters:
        selected = tuple(bar for bar in bars if accepts is None or accepts(adapter, bar))
        ordering = getattr(broker, "order_strategy_decisions", None)
        if selected and callable(ordering):
            ordered = tuple(ordering(adapter, selected))
            if sorted(map(id, ordered)) != sorted(map(id, selected)):
                raise ValueError("Decision priority must preserve every market observation")
            selected = ordered
        native_batch = bool(getattr(getattr(adapter, 'config', None), 'capacity_joint_allocation', False))
        if selected and native_batch:
            if callable(finalize):
                raise ValueError('Cannot combine native OLR joint allocation with an external intent allocator')
            broker._decision_available_at = selected[0].timestamp + timedelta(minutes=5)
            decisions.extend(adapter.on_bars(selected, broker))
        for bar in (() if native_batch else selected):
            minutes = int(bar.timeframe[:-1]) if bar.timeframe.endswith("m") else 0
            broker._decision_available_at = bar.timestamp + timedelta(minutes=minutes)
            decisions.extend(adapter.on_bar(bar, broker))
        if selected:
            hook = getattr(adapter, "on_timestamp_end", None)
            if callable(hook):
                if callable(finalize):
                    deferred_end_hooks.append((hook, selected))
                else:
                    decisions.extend(hook(selected[-1].timestamp, selected, broker))
    if callable(finalize):
        decisions.extend(finalize(bars, adapters))
        for hook, selected in deferred_end_hooks:
            decisions.extend(hook(selected[-1].timestamp, selected, broker))
    broker._decision_available_at = None
    return decisions
