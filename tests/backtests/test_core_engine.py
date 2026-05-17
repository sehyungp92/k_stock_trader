from __future__ import annotations

from datetime import datetime

import pytest

from backtests.core.completed_bar_policy import bar_availability_time, visible_bars_at
from backtests.engine.replay import run_replay
from backtests.engine.sim_broker import SimBroker
from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_common.events import DecisionEvent
from strategy_common.clock import KST
from strategy_common.market import MarketBar


def test_completed_bar_policy_delays_higher_timeframe_visibility():
    bar = MarketBar("000001", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "30m", 1, 2, 1, 2, 10)
    assert bar_availability_time(bar).hour == 9
    assert bar_availability_time(bar).minute == 30
    assert visible_bars_at([bar], datetime(2026, 1, 5, 9, 29, tzinfo=KST)) == []
    assert visible_bars_at([bar], datetime(2026, 1, 5, 9, 30, tzinfo=KST)) == [bar]


def test_sim_broker_blocks_same_bar_fill_after_completed_signal():
    broker = SimBroker(1_000_000)
    signal_bar = MarketBar("000001", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "1m", 100, 101, 99, 100, 1000)
    broker.submit(SubmitEntry("KMP", "000001", 1, "MARKET", None, 98, "test"), signal_bar.timestamp)
    assert broker.process_bar(signal_bar) == []
    next_bar = MarketBar("000001", datetime(2026, 1, 5, 9, 1, tzinfo=KST), "1m", 101, 102, 100, 101, 1000)
    fills = broker.process_bar(next_bar)
    assert len(fills) == 1
    assert fills[0].timestamp == next_bar.timestamp


def test_sim_broker_marks_each_symbol_with_its_own_last_price():
    broker = SimBroker(1_000_000)
    first = MarketBar("000001", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "1m", 100, 100, 100, 100, 1000)
    second = MarketBar("000002", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "1m", 200, 200, 200, 200, 1000)
    broker.submit(SubmitEntry("kmp", "000001", 10, "MARKET", None, 95, "first"), first.timestamp)
    broker.submit(SubmitEntry("kmp", "000002", 10, "MARKET", None, 190, "second"), second.timestamp)
    broker.process_bar(first)
    broker.process_bar(second)

    first_next = MarketBar("000001", datetime(2026, 1, 5, 9, 1, tzinfo=KST), "1m", 110, 110, 110, 110, 1000)
    second_next = MarketBar("000002", datetime(2026, 1, 5, 9, 1, tzinfo=KST), "1m", 220, 220, 220, 220, 1000)
    broker.process_bar(first_next)
    broker.process_bar(second_next)

    first_later = MarketBar("000001", datetime(2026, 1, 5, 9, 2, tzinfo=KST), "1m", 120, 120, 120, 120, 1000)
    equity = broker.mark_to_market(first_later)
    assert equity == pytest.approx(broker.cash + 10 * 120 + 10 * 220)


def test_sim_broker_rejects_cash_overallocation():
    broker = SimBroker(1_000)
    signal_bar = MarketBar("000001", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "1m", 100, 100, 100, 100, 1000)
    fill_bar = MarketBar("000001", datetime(2026, 1, 5, 9, 1, tzinfo=KST), "1m", 100, 100, 100, 100, 1000)
    broker.submit(SubmitEntry("KMP", "000001", 20, "MARKET", None, 95, "too_large"), signal_bar.timestamp)

    broker.process_bar(signal_bar)
    assert broker.process_bar(fill_bar) == []
    assert broker.position_qty("KMP", "000001") == 0
    assert broker.rejected_orders


def test_exit_fill_event_qty_matches_position_qty():
    broker = SimBroker(1_000_000)
    signal_bar = MarketBar("000001", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "1m", 100, 100, 100, 100, 1000)
    fill_bar = MarketBar("000001", datetime(2026, 1, 5, 9, 1, tzinfo=KST), "1m", 100, 100, 100, 100, 1000)
    exit_bar = MarketBar("000001", datetime(2026, 1, 5, 9, 2, tzinfo=KST), "1m", 101, 101, 101, 101, 1000)
    broker.submit(SubmitEntry("KMP", "000001", 5, "MARKET", None, 95, "entry"), signal_bar.timestamp)
    broker.process_bar(signal_bar)
    broker.process_bar(fill_bar)
    broker.submit(SubmitExit("KMP", "000001", 100, "MARKET", None, "oversized_exit"), fill_bar.timestamp)

    fills = broker.process_bar(exit_bar)
    assert len(fills) == 1
    assert fills[0].qty == 5


def test_replay_rejects_incomplete_bars_before_strategy_logic():
    class Adapter:
        strategy_id = "TEST"

        def on_bar(self, bar: MarketBar, broker: SimBroker) -> list[DecisionEvent]:
            del bar, broker
            return []

    incomplete = MarketBar("000001", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "1m", 1, 1, 1, 1, 1, is_completed=False)
    with pytest.raises(ValueError, match="Incomplete bar"):
        run_replay([incomplete], Adapter())
