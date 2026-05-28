"""Tests for filter decision context logging."""
from instrumentation.src.trade_logger import TradeEvent
from instrumentation.src.missed_opportunity import MissedOpportunityEvent


def test_trade_event_filter_decisions():
    decisions = [
        {"filter": "volume_gate", "threshold": 1.5, "actual": 1.3,
         "passed": False, "margin_pct": -13.3},
        {"filter": "spread_gate", "threshold": 0.004, "actual": 0.002,
         "passed": True, "margin_pct": 50.0},
    ]
    event = TradeEvent(
        trade_id="test_f1",
        event_metadata={"event_id": "ef1"},
        entry_snapshot={},
        filter_decisions=decisions,
    )
    assert len(event.filter_decisions) == 2
    assert event.filter_decisions[0]["margin_pct"] == -13.3


def test_filter_decisions_defaults_empty():
    event = TradeEvent(
        trade_id="test_f2",
        event_metadata={"event_id": "ef2"},
        entry_snapshot={},
    )
    assert event.filter_decisions == []


def test_missed_opportunity_filter_decisions():
    fd = [{"filter": "risk_budget", "threshold": 0.04, "actual": 0.05,
           "passed": False, "margin_pct": 25.0}]
    event = MissedOpportunityEvent(
        event_metadata={},
        market_snapshot={},
        filter_decisions=fd,
    )
    assert len(event.filter_decisions) == 1
    assert event.filter_decisions[0]["filter"] == "risk_budget"


def test_missed_opportunity_filter_decisions_defaults_empty():
    event = MissedOpportunityEvent(event_metadata={}, market_snapshot={})
    assert event.filter_decisions == []


def test_filter_decisions_serializable():
    """Verify filter_decisions survive to_dict() round-trip."""
    fd = [{"filter": "test", "threshold": 1.0, "actual": 0.5,
           "passed": True, "margin_pct": 50.0}]
    event = TradeEvent(
        trade_id="test_serial",
        event_metadata={"event_id": "es1"},
        entry_snapshot={},
        filter_decisions=fd,
    )
    d = event.to_dict()
    assert d["filter_decisions"] == fd
