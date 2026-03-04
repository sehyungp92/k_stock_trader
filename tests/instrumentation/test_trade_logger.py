"""Tests for TradeEvent signal confluence logging."""
import pytest
from instrumentation.src.trade_logger import TradeEvent


def test_trade_event_has_signal_factors_field():
    """TradeEvent should accept signal_factors as list of dicts."""
    factors = [
        {"factor": "RSI_oversold", "value": 28.5, "threshold": 30.0, "contribution": 0.25},
        {"factor": "volume_spike", "value": 2.3, "threshold": 2.0, "contribution": 0.20},
    ]
    event = TradeEvent(
        trade_id="test_001",
        event_metadata={"event_id": "e1"},
        entry_snapshot={},
        signal_factors=factors,
    )
    assert event.signal_factors == factors
    d = event.to_dict()
    assert d["signal_factors"] == factors


def test_trade_event_signal_factors_defaults_empty():
    """signal_factors should default to empty list for backward compat."""
    event = TradeEvent(
        trade_id="test_002",
        event_metadata={"event_id": "e2"},
        entry_snapshot={},
    )
    assert event.signal_factors == []
