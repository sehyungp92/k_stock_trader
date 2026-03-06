"""Tests for market_snapshot module — adapted for KIS API."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService


def _make_daily_bars(n=20, base_price=50000):
    """Create fake daily bars DataFrame matching KIS API format."""
    rows = []
    for i in range(n):
        rows.append({
            "date": f"2026-02-{10 + i:02d}",
            "open": base_price,
            "high": base_price + 500,
            "low": base_price - 500,
            "close": base_price + 100,
            "volume": 1_000_000,
        })
    return pd.DataFrame(rows)


def _make_minute_bars(n=5, base_price=50000):
    """Create fake minute bars DataFrame matching KIS API format."""
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": f"2026-03-01T10:{i:02d}:00+09:00",
            "open": base_price,
            "high": base_price + 50,
            "low": base_price - 50,
            "close": base_price + 10,
            "volume": 5000,
        })
    return pd.DataFrame(rows)


class MockKISDataProvider:
    """Mock KIS API client for testing."""

    def get_last_price(self, symbol):
        return 50000.0

    def get_daily_bars(self, symbol, days=20):
        return _make_daily_bars(days)

    def get_minute_bars(self, symbol, minutes=5):
        return _make_minute_bars(minutes)


class TestMarketSnapshot:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "market_snapshots": {"interval_seconds": 60, "symbols": ["005930"]},
        }
        self.service = MarketSnapshotService(self.config, MockKISDataProvider())

    def test_capture_now_returns_snapshot(self):
        snap = self.service.capture_now("005930")
        assert isinstance(snap, MarketSnapshot)
        assert snap.symbol == "005930"
        assert snap.last_trade_price == 50000.0
        assert snap.mid == 50000.0

    def test_bid_ask_none_for_krx(self):
        """KRX equity: bid/ask not available on-demand via REST."""
        snap = self.service.capture_now("005930")
        assert snap.bid is None
        assert snap.ask is None
        assert snap.spread_bps is None
        assert snap.bid_ask_available is False

    def test_equity_fields_none(self):
        """Equity market: funding rate, open interest, mark price always None."""
        snap = self.service.capture_now("005930")
        assert snap.funding_rate is None
        assert snap.open_interest is None
        assert snap.mark_price is None

    def test_atr_computed(self):
        snap = self.service.capture_now("005930")
        assert snap.atr_14 is not None
        assert snap.atr_14 > 0

    def test_volume_computed(self):
        snap = self.service.capture_now("005930")
        assert snap.volume_24h is not None
        assert snap.volume_24h > 0
        assert snap.volume_1m is not None

    def test_snapshot_id_deterministic(self):
        """Same symbol + timestamp produces same ID."""
        sid = self.service._compute_snapshot_id("005930", "2026-03-01T10:00:00+09:00")
        sid2 = self.service._compute_snapshot_id("005930", "2026-03-01T10:00:00+09:00")
        assert sid == sid2

    def test_snapshot_id_unique_across_symbols(self):
        sid1 = self.service._compute_snapshot_id("005930", "2026-03-01T10:00:00+09:00")
        sid2 = self.service._compute_snapshot_id("000660", "2026-03-01T10:00:00+09:00")
        assert sid1 != sid2

    def test_capture_writes_to_jsonl(self):
        self.service.capture_now("005930")
        files = list(Path(self.tmpdir).joinpath("snapshots").glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text().strip()
        data = json.loads(content)
        assert data["symbol"] == "005930"
        assert "snapshot_id" in data

    def test_degraded_snapshot_on_provider_failure(self):
        """Snapshot service must never crash, even with a broken data provider."""
        bad_provider = MagicMock()
        bad_provider.get_last_price.side_effect = Exception("connection lost")
        bad_provider.get_daily_bars.side_effect = Exception("connection lost")
        bad_provider.get_minute_bars.side_effect = Exception("connection lost")
        service = MarketSnapshotService(self.config, bad_provider)
        snap = service.capture_now("005930")
        assert snap.symbol == "005930"
        assert snap.last_trade_price == 0.0  # degraded

    def test_no_data_provider_returns_degraded(self):
        """Service with no data provider returns degraded but valid snapshot."""
        service = MarketSnapshotService(self.config, data_provider=None)
        snap = service.capture_now("005930")
        assert snap.symbol == "005930"
        assert snap.last_trade_price == 0.0

    def test_to_dict(self):
        snap = self.service.capture_now("005930")
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert d["symbol"] == "005930"
        assert "atr_14" in d

    def test_cache_stores_latest(self):
        self.service.capture_now("005930")
        cached = self.service.get_latest("005930")
        assert cached is not None
        assert cached.symbol == "005930"

    def test_run_periodic(self):
        self.service.run_periodic()
        assert self.service.get_latest("005930") is not None
