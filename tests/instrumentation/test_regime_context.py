"""Tests for multi-timeframe regime context."""
from instrumentation.src.trade_logger import TradeEvent


def test_trade_event_has_regime_context():
    ctx = {
        "primary_regime": "trending_up",
        "higher_tf_regime": "ranging",
        "sector_regime": "unknown",
    }
    event = TradeEvent(
        trade_id="test_r1",
        event_metadata={"event_id": "er1"},
        entry_snapshot={},
        regime_context=ctx,
    )
    assert event.regime_context["primary_regime"] == "trending_up"
    assert event.regime_context["higher_tf_regime"] == "ranging"


def test_regime_context_defaults_none():
    event = TradeEvent(
        trade_id="test_r2",
        event_metadata={"event_id": "er2"},
        entry_snapshot={},
    )
    assert event.regime_context is None
