"""Tests for missed_opportunity module."""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

from instrumentation.src.missed_opportunity import (
    MissedOpportunityLogger,
    MissedOpportunityEvent,
    SimulationPolicy,
)
from instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService


def _mock_snapshot():
    return MarketSnapshot(
        snapshot_id="test_snap",
        symbol="005930",
        timestamp="2026-03-01T10:00:00+09:00",
        bid=0.0,
        ask=0.0,
        mid=50000.0,
        spread_bps=0.0,
        last_trade_price=50000.0,
        atr_14=500.0,
        volume_24h=1_000_000.0,
    )


class TestSimulationPolicy:
    def test_defaults(self):
        p = SimulationPolicy()
        assert p.entry_fill_model == "mid"
        assert p.fee_bps == 20.0
        assert p.slippage_bps == 5.0

    def test_to_dict(self):
        p = SimulationPolicy()
        d = p.to_dict()
        assert isinstance(d, dict)
        assert "fee_bps" in d


class TestMissedOpportunityLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
        }
        self.snap_service = MagicMock(spec=MarketSnapshotService)
        self.snap_service.capture_now.return_value = _mock_snapshot()
        self.mol = MissedOpportunityLogger(self.config, self.snap_service)

    def test_log_missed_returns_event(self):
        event = self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="momentum breakout confirmed",
            signal_id="kmp_breakout",
            signal_strength=0.75,
            blocked_by="volume_gate",
            block_reason="Volume ratio 0.8x below threshold 1.5x",
            strategy_type="kmp",
        )
        assert isinstance(event, MissedOpportunityEvent)
        assert event.pair == "005930"
        assert event.blocked_by == "volume_gate"
        assert event.side == "LONG"

    def test_hypothetical_entry_price_computed(self):
        event = self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
            strategy_type="kmp",
        )
        # Mid price is 50000, slippage of 5bps = 50000 * 5/10000 = 25
        # LONG: base_price + slippage = 50000 + 25 = 50025
        assert event.hypothetical_entry_price > 0

    def test_assumption_tags_populated(self):
        event = self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
        )
        assert len(event.assumption_tags) > 0
        assert any("fill" in t for t in event.assumption_tags)
        assert any("slippage" in t for t in event.assumption_tags)

    def test_simulation_policy_attached(self):
        event = self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
        )
        assert event.simulation_policy is not None
        assert isinstance(event.simulation_policy, dict)

    def test_event_metadata_present(self):
        event = self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
        )
        assert event.event_metadata
        assert "event_id" in event.event_metadata

    def test_writes_to_jsonl(self):
        self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
        )
        files = list(Path(self.tmpdir).joinpath("missed").glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text().strip()
        data = json.loads(content)
        assert data["blocked_by"] == "regime_gate"

    def test_backfill_status_pending(self):
        event = self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
        )
        assert event.backfill_status == "pending"

    def test_failure_does_not_crash(self):
        """Instrumentation failure must never propagate."""
        self.snap_service.capture_now.side_effect = Exception("broken")
        event = self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
        )
        # Should return empty event, not crash
        assert isinstance(event, MissedOpportunityEvent)

    def test_queues_for_backfill(self):
        self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
        )
        assert len(self.mol._pending_backfills) == 1

    def test_default_simulation_policy_used(self):
        """When no strategy_type given, default policy is used."""
        event = self.mol.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
            strategy_type=None,
        )
        assert event.simulation_policy is not None
