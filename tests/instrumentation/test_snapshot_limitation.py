"""Tests for KIS REST bid/ask limitation handling."""
from instrumentation.src.market_snapshot import MarketSnapshot


def test_snapshot_bid_ask_defaults_none():
    """bid/ask should default to None (not 0) for REST-only data."""
    snap = MarketSnapshot(
        snapshot_id="test_snap",
        symbol="005930",
        timestamp="2026-03-03T10:00:00+09:00",
    )
    assert snap.bid is None
    assert snap.ask is None
    assert snap.spread_bps is None
    assert snap.data_source == "kis_rest"
    assert snap.bid_ask_available is False


def test_snapshot_data_source_field():
    """data_source should be explicitly set."""
    snap = MarketSnapshot(
        snapshot_id="test_snap2",
        symbol="005930",
        timestamp="2026-03-03T10:00:00+09:00",
        data_source="kis_ws",
        bid_ask_available=True,
        bid=50000.0,
        ask=50100.0,
    )
    assert snap.data_source == "kis_ws"
    assert snap.bid_ask_available is True
    assert snap.bid == 50000.0
