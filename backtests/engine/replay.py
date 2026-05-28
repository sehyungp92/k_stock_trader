from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


def run_replay(
    bars: list[MarketBar],
    adapter: StrategyReplayAdapter,
    *,
    initial_equity: float = 10_000_000.0,
    costs: BrokerCosts | None = None,
    close_open_positions: bool = True,
    bars_are_ordered: bool = False,
    buying_power_leverage: float = 1.0,
) -> ReplayResult:
    broker = SimBroker(initial_equity=initial_equity, costs=costs, buying_power_leverage=buying_power_leverage)
    decisions: list[DecisionEvent] = []
    ordered = list(bars) if bars_are_ordered else sorted((require_completed_bar(bar) for bar in bars), key=lambda bar: (bar.timestamp, bar.symbol))
    timestamp_bars: list[MarketBar] = []
    for index, bar in enumerate(ordered):
        broker.process_bar(bar)
        decisions.extend(adapter.on_bar(bar, broker))
        timestamp_bars.append(bar)
        next_ts = ordered[index + 1].timestamp if index + 1 < len(ordered) else None
        if next_ts != bar.timestamp:
            hook = getattr(adapter, "on_timestamp_end", None)
            if callable(hook):
                decisions.extend(hook(bar.timestamp, tuple(timestamp_bars), broker))
            if next_ts is None or next_ts.date() != bar.timestamp.date():
                broker.force_same_day_exits(bar)
            timestamp_bars = []
    if close_open_positions and ordered:
        broker.close_all_at_end(ordered[-1])
    return ReplayResult(
        trades=list(broker.trades),
        decisions=decisions,
        equity_curve=list(broker.equity_curve),
        timestamps=list(broker.timestamps),
        broker=broker,
    )
