from datetime import date, timedelta

import pytest

from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.engine.sim_broker import BrokerCosts, SimBroker
from strategy_common.actions import SubmitEntry
from strategy_common.session_coverage import session_bar_coverage
from tests.backtests.test_intraday_halt_evidence import bar


@pytest.mark.parametrize("symbol,day", [
    ("323410", date(2025, 6, 25)), ("323410", date(2025, 6, 27)),
    ("086520", date(2025, 10, 23)), ("086520", date(2025, 10, 27)),
])
def test_notice_accounts_for_periodic_grid_without_waiving_missing_auctions(symbol, day):
    first = bar(symbol, day, 9, 0).timestamp
    stamps = [first + timedelta(minutes=30 * i) for i in range(14)]
    result = session_bar_coverage(stamps, day, 5, symbol=symbol)
    assert result["status"] == "pass"
    assert result["expected_intervals"] == 14
    assert result["periodic_auction_sources"]
    assert session_bar_coverage(stamps[:-1], day, 5, symbol=symbol)["status"] == "unverified"
    assert session_bar_coverage(stamps + [first + timedelta(minutes=5)], day, 5, symbol=symbol)["status"] == "unverified"
    assert session_bar_coverage(stamps, day, 5, symbol="005930")["status"] == "unverified"


def test_periodic_auction_does_not_silently_execute_continuous_order():
    symbol, day = "086520", date(2025, 10, 23)
    broker = SimBroker(10000, BrokerCosts(0, 0, 0))
    broker.submit(SubmitEntry("KALCB", symbol, 10, "MARKET", None, None, "entry"),
                  bar(symbol, day, 9, 5).timestamp)
    drive_timestamp_batch(broker, [bar(symbol, day, 9, 30)], [])
    assert not broker.fills
    assert broker.orders
    assert broker.last_prices[symbol] == 105
    audit = audit_broker(broker)
    assert audit["execution_validity_pass"]
    assert audit["audit_receipt"]["coverage"]["periodic_auction_execution"] == "unverified"
    broker.bar_issues["unsupported_periodic_auction_execution"] = 0
    assert "periodic_auction_counter_mismatch" in audit_broker(broker)["audit_receipt"]["errors"]


def test_marks_alone_do_not_require_periodic_execution_model():
    broker = SimBroker(10000, BrokerCosts(0, 0, 0))
    drive_timestamp_batch(broker, [bar("086520", date(2025, 10, 23), 9, 30)], [])
    assert audit_broker(broker)["audit_receipt"]["coverage"]["periodic_auction_execution"] == "not_applicable"
