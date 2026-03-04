"""Tests for position sizing decision logging."""
from instrumentation.src.trade_logger import TradeEvent


def test_trade_event_sizing_context():
    ctx = {
        "sizing_model": "risk_based_quality_adj",
        "target_risk_pct": 0.005,
        "account_equity": 50_000_000,
        "volatility_basis": 1250.0,
        "quality_mult": 1.0,
        "time_mult": 0.85,
        "raw_qty": 40,
        "final_qty": 35,
        "cap_reason": "liquidity_5m",
    }
    event = TradeEvent(
        trade_id="test_s1",
        event_metadata={"event_id": "es1"},
        entry_snapshot={},
        sizing_context=ctx,
    )
    assert event.sizing_context["target_risk_pct"] == 0.005
    assert event.sizing_context["sizing_model"] == "risk_based_quality_adj"


def test_sizing_context_defaults_none():
    event = TradeEvent(
        trade_id="test_s2",
        event_metadata={"event_id": "es2"},
        entry_snapshot={},
    )
    assert event.sizing_context is None
