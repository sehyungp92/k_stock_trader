from datetime import date, datetime

import pytest

from backtests.engine.sim_broker import BrokerCosts, SimBroker
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from strategy_common.execution_session import bar_execution_issue
from strategy_common.market import MarketBar
from strategy_common.session_coverage import session_bar_coverage


@pytest.mark.parametrize("symbol,day", [
    ("207940", date(2025, 10, 30)), ("207940", date(2025, 11, 21)),
    ("010120", date(2026, 4, 8)), ("010120", date(2026, 4, 10)),
    ("377300", date(2025, 6, 24)), ("377300", date(2025, 6, 26)),
])
def test_sourced_suspension_accounts_for_empty_session_and_prevents_fill(symbol, day):
    result = session_bar_coverage([], day, 5, symbol=symbol)
    assert result["status"] == "pass"
    assert result["expected_intervals"] == 0
    assert result["source"]
    stamp = datetime(day.year, day.month, day.day, 10, tzinfo=KST)
    bar = MarketBar(symbol, stamp, "5m", 100, 100, 100, 100, 1000)
    assert bar_execution_issue(bar) == "suspended_symbol"
    # Contradictory source rows must not get cleared just because an exception exists.
    assert session_bar_coverage([stamp], day, 5, symbol=symbol)["status"] == "unverified"
    broker = SimBroker(10000, BrokerCosts(0, 0, 0))
    broker.submit(SubmitEntry("KALCB", symbol, 10, "MARKET", None, None, "entry"), stamp.replace(hour=9))
    assert broker.process_bar(bar) == []
    assert broker.cash == 10000
    assert len(broker.orders) == 1


@pytest.mark.parametrize("symbol,day", [
    ("207940", date(2025, 10, 29)), ("207940", date(2025, 11, 24)),
    ("010120", date(2026, 4, 7)), ("010120", date(2026, 4, 13)),
    ("377300", date(2025, 6, 25)), ("005930", date(2025, 11, 13)),
])
def test_suspension_does_not_excuse_other_symbols_or_dates(symbol, day):
    assert session_bar_coverage([], day, 5, symbol=symbol)["status"] == "unverified"


def test_replay_requires_held_and_pending_sessions_but_not_inactive_future_days():
    from datetime import timedelta
    from backtests.auto.shared.cache_keys import stable_signature
    from backtests.engine.qualification_evidence import _replay_scope
    from strategy_common.actions import SubmitExit
    days = [date(2026, 1, 5) + timedelta(days=i) for i in range(3)]
    def stamp(day, hour=9, minute=0):
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=KST)
    def bar(at):
        return MarketBar("005930", at, "5m", 100, 100, 100, 100, 1000)
    broker = SimBroker(10000, BrokerCosts(0, 0, 0))
    broker.submit(SubmitEntry("KALCB", "005930", 10, "MARKET", None, None, "entry"), stamp(days[0], 8))
    broker.process_bar(bar(stamp(days[0])))
    broker.submit(SubmitExit("KALCB", "005930", 10, "MARKET", None, "exit"), stamp(days[0], 15, 30))
    broker.process_bar(bar(stamp(days[1])))
    broker.qualification_snapshots = [{"trade_date": d.isoformat(), "candidates":
        [{"symbol": "005930"}] if d == days[0] else []} for d in days]
    identity = {"qualification_snapshot_hash": stable_signature(broker.qualification_snapshots)}
    required = {("005930", d, "5m") for d in days}
    events = [bar(t).to_json_dict() for d in days[:2]
              for t in [*(stamp(d) + timedelta(minutes=i) for i in range(0, 380, 5)), stamp(d, 15, 30)]]
    _replay_scope(broker, identity, events, required)
    with pytest.raises(ValueError, match="management_feed_incomplete:005930:2026-01-06"):
        _replay_scope(broker, identity, events[:77], required)
    # A new pending order extends the obligation even without a new candidate.
    broker.submit(SubmitEntry("KALCB", "005930", 10, "MARKET", None, None, "entry"), stamp(days[1], 15, 30))
    with pytest.raises(ValueError, match="management_feed_incomplete:005930:2026-01-07"):
        _replay_scope(broker, identity, events, required)


def test_daily_reconciliation_does_not_hide_a_missing_trade_or_price_change():
    import pandas as pd
    from scripts.reconcile_integrity_sessions import compare_session
    bars = pd.DataFrame({"open": [100, 102], "high": [104, 103], "low": [99, 101],
                         "close": [102, 101], "volume": [50, 70]})
    daily = {"open": 100, "high": 104, "low": 99, "close": 101, "volume": 120}
    assert compare_session(bars, daily)["status"] == "exact_daily_ohlcv_match"
    assert "volume" in compare_session(bars.iloc[:1], daily)["mismatched_fields"]
    assert "high" in compare_session(bars, {**daily, "high": 105})["mismatched_fields"]
    assert compare_session(bars.iloc[:0], daily)["status"] == "missing_intraday"
