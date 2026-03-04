"""Tests for filter decision context logging."""
import pytest
from instrumentation.src.trade_logger import TradeEvent
from instrumentation.src.missed_opportunity import MissedOpportunityEvent
from strategy_kmp.core.gates import build_filter_decisions


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


def test_build_filter_decisions_passed():
    checks = {
        "spread_gate": (True, 0.004, 0.002),
        "rvol_gate": (True, 2.0, 3.5),
    }
    result = build_filter_decisions(checks)
    assert len(result) == 2

    spread = next(d for d in result if d["filter"] == "spread_gate")
    assert spread["passed"] is True
    assert spread["threshold"] == 0.004
    assert spread["actual"] == 0.002
    assert spread["margin_pct"] == 50.0  # (0.004 - 0.002) / 0.004 * 100

    rvol = next(d for d in result if d["filter"] == "rvol_gate")
    assert rvol["passed"] is True
    assert rvol["margin_pct"] == -75.0  # (2.0 - 3.5) / 2.0 * 100


def test_build_filter_decisions_failed():
    checks = {
        "spread_gate": (False, 0.004, 0.006),
    }
    result = build_filter_decisions(checks)
    assert len(result) == 1
    assert result[0]["passed"] is False
    assert result[0]["margin_pct"] == 50.0  # (0.006 - 0.004) / 0.004 * 100


def test_build_filter_decisions_zero_threshold():
    checks = {
        "dummy": (True, 0, 5.0),
    }
    result = build_filter_decisions(checks)
    assert result[0]["margin_pct"] == 0


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
