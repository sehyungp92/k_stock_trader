from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.engine.sim_broker import BrokerCosts, SimBroker
from kis_core.trading_calendar import get_trading_calendar
from scripts.update_kis_intraday import aggregate_timeframe, _collection_end, _previous_day_close
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_common.session_coverage import session_bar_coverage


@pytest.fixture
def auction(monkeypatch):
    day, symbol = date(2026, 4, 9), "000001"
    calendar = get_trading_calendar()
    monkeypatch.setattr(calendar, "closing_auctions", {(symbol, day): time(15, 32)})
    stamp = datetime.combine(day, time(15, 32), KST)
    return MarketBar(symbol, stamp, "5m", 100, 100, 100, 100, 1000)


def test_late_print_keeps_matching_time_and_replaces_expected_close(auction):
    from backtests.core.completed_bar_policy import bar_availability_time
    assert bar_availability_time(auction) == auction.timestamp
    frame = pd.DataFrame([dict(timestamp=auction.timestamp, open=100, high=100, low=100, close=100, volume=1000)])
    assert aggregate_timeframe(frame, "5m").timestamp.iloc[0] == auction.timestamp
    start = auction.timestamp.replace(hour=9, minute=0)
    stamps = [start + timedelta(minutes=i) for i in range(0, 380, 5)] + [auction.timestamp]
    assert session_bar_coverage(stamps, start.date(), 5, symbol=auction.symbol)["status"] == "pass"
    stamps[-1] -= timedelta(minutes=2)
    assert session_bar_coverage(stamps, start.date(), 5, symbol=auction.symbol)["status"] == "unverified"
    assert _collection_end(start.date()).time() == time(15, 40)
    assert _previous_day_close(start + timedelta(days=1)).time() == time(15, 40)


@pytest.mark.parametrize("basis,filled", [("closing_auction", True), ("fixed_deadline", False)])
def test_only_auction_linked_expiry_follows_delayed_match(auction, basis, filled):
    broker = SimBroker(10000, BrokerCosts(0, 0, 0))
    metadata = {"auction_fill_time": "15:30", "expiry_basis": basis,
                "expiry_ts": auction.timestamp.replace(minute=30).isoformat()}
    broker.submit(SubmitEntry("OLR", auction.symbol, 10, "CLOSE_AUCTION", None, None, "entry", metadata),
                  auction.timestamp.replace(hour=14, minute=30))
    drive_timestamp_batch(broker, [auction], [])
    assert bool(broker.fills) is filled
    if filled:
        assert broker.fills[0].timestamp == auction.timestamp
        assert broker.fills[0].metadata["execution_phase"] == "auction"
        assert broker.journal[1]["prices"] == {}
    assert audit_broker(broker)["execution_validity_pass"]


def test_late_auction_does_not_execute_pending_continuous_market_order(auction):
    broker = SimBroker(10000, BrokerCosts(0, 0, 0))
    broker.submit(SubmitEntry("KALCB", auction.symbol, 10, "MARKET", None, None, "entry"),
                  auction.timestamp.replace(minute=15))
    drive_timestamp_batch(broker, [auction], [])
    assert not broker.fills
    assert broker.last_prices[auction.symbol] == 100
    assert audit_broker(broker)["execution_validity_pass"]
