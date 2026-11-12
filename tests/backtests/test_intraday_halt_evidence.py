from datetime import date, datetime, timedelta

import pytest

from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.engine.sim_broker import BrokerCosts, SimBroker
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from strategy_common.execution_session import bar_execution_issue, bar_reopens_after_label
from strategy_common.market import MarketBar
from strategy_common.session_coverage import session_bar_coverage


def bar(symbol, day, hour, minute):
    return MarketBar(symbol, datetime.combine(day, datetime.min.time(), tzinfo=KST).replace(hour=hour, minute=minute),
                     "5m", 100, 110, 90, 105, 1000)


@pytest.mark.parametrize("day,symbol,missing,reopening", [
    (date(2026, 3, 4), "005930", ["11:20", "11:25", "11:30", "11:35", "11:40"], (11, 45)),
    (date(2026, 3, 4), "058470", ["11:20", "11:25", "11:30", "11:35", "11:40"], (11, 45)),
    (date(2026, 3, 9), "005930", ["10:35", "10:40", "10:45", "10:50", "10:55"], (11, 0)),
])
def test_halt_excuses_only_full_nonmatching_intervals(day, symbol, missing, reopening):
    opening = bar(symbol, day, 9, 0).timestamp
    grid = [opening + timedelta(minutes=i) for i in range(0, 380, 5)] + [opening.replace(hour=15, minute=30)]
    observed = [stamp for stamp in grid if stamp.strftime("%H:%M") not in missing]
    result = session_bar_coverage(observed, day, 5, symbol=symbol)
    assert result["status"] == "pass"
    assert len(result["sourced_halt_intervals"]) == 5
    boundary = bar(symbol, day, *reopening)
    assert bar_execution_issue(boundary) is None
    assert bar_reopens_after_label(boundary)
    assert session_bar_coverage([s for s in observed if s != boundary.timestamp], day, 5, symbol=symbol)["status"] == "unverified"
    assert session_bar_coverage(grid, day, 5, symbol=symbol)["observations_during_halt"]
    first = datetime.fromisoformat(result["sourced_halt_intervals"][0])
    assert bar_execution_issue(bar(symbol, day, first.hour, first.minute)) == "halted_interval"


def test_reopening_does_not_use_future_open_and_retains_pending_order():
    day = date(2026, 3, 9)
    broker = SimBroker(10000, BrokerCosts(0, 0, 0))
    for symbol in ("005930", "058470"):
        broker.submit(SubmitEntry("KALCB", symbol, 10, "MARKET", None, None, "entry"), bar(symbol, day, 10, 30).timestamp)
    drive_timestamp_batch(broker, [bar("005930", day, 11, 0), bar("058470", day, 11, 0)], [])
    assert [fill.symbol for fill in broker.fills] == ["058470"]
    opening = next(row for row in broker.journal if row["event"] == "opening_marks")
    assert "005930" not in opening["prices"]
    assert broker.last_prices["005930"] == 105
    assert broker.bar_issues["unsupported_halt_boundary_chronology"] == 1
    audit = audit_broker(broker)
    assert audit["execution_validity_pass"]
    assert audit["audit_receipt"]["coverage"]["halt_boundary_chronology"] == "unverified"
    drive_timestamp_batch(broker, [bar("005930", day, 11, 5), bar("058470", day, 11, 5)], [])
    assert [fill.symbol for fill in broker.fills] == ["058470", "005930"]
    assert audit_broker(broker)["execution_validity_pass"]
    broker.bar_issues["unsupported_halt_boundary_chronology"] = 0
    assert "halt_boundary_counter_mismatch" in audit_broker(broker)["audit_receipt"]["errors"]


def test_halt_does_not_create_unrelated_session_obligations():
    assert not bar_reopens_after_label(bar("058470", date(2026, 3, 9), 11, 0))
    broker = SimBroker(10000, BrokerCosts(0, 0, 0))
    drive_timestamp_batch(broker, [bar("005930", date(2026, 3, 9), 11, 0)], [])
    assert audit_broker(broker)["audit_receipt"]["coverage"]["halt_boundary_chronology"] == "not_applicable"
