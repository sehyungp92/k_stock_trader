"""Integration test — simulates a full trade lifecycle day."""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

from instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService
from instrumentation.src.trade_logger import TradeLogger
from instrumentation.src.missed_opportunity import MissedOpportunityLogger
from instrumentation.src.process_scorer import ProcessScorer
from instrumentation.src.daily_snapshot import DailySnapshotBuilder


def _mock_snapshot(symbol="005930"):
    return MarketSnapshot(
        snapshot_id="test_snap",
        symbol=symbol,
        timestamp=datetime.now(timezone.utc).isoformat(),
        bid=0.0,
        ask=0.0,
        mid=50000.0,
        spread_bps=0.0,
        last_trade_price=50000.0,
        atr_14=500.0,
        volume_24h=1_000_000.0,
    )


class TestFullLifecycle:
    """Simulate a complete day: signals, blocked trades, executed trades, daily rollup."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "integration_test",
            "strategy_type": "kmp",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
        }

        self.snap_service = MagicMock(spec=MarketSnapshotService)
        self.snap_service.capture_now.return_value = _mock_snapshot()

        self.trade_logger = TradeLogger(self.config, self.snap_service)
        self.missed_logger = MissedOpportunityLogger(self.config, self.snap_service)

    def test_full_day_lifecycle(self):
        """Simulate: 2 executed trades + 1 missed opportunity + daily snapshot."""

        # --- Trade 1: winning trade ---
        self.trade_logger.log_entry(
            trade_id="t1",
            pair="005930",
            side="LONG",
            entry_price=50000,
            position_size=10,
            position_size_quote=500000,
            entry_signal="momentum breakout",
            entry_signal_id="kmp_breakout",
            entry_signal_strength=0.8,
            active_filters=["regime_gate", "volume_gate"],
            passed_filters=["regime_gate", "volume_gate"],
            strategy_params={"quality_threshold": 30},
            market_regime="trending_up",
        )
        trade1 = self.trade_logger.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fees_paid=100,
        )
        assert trade1 is not None
        assert trade1.pnl > 0

        # --- Trade 2: losing trade ---
        self.trade_logger.log_entry(
            trade_id="t2",
            pair="000660",
            side="LONG",
            entry_price=51000,
            position_size=5,
            position_size_quote=255000,
            entry_signal="momentum breakout",
            entry_signal_id="kmp_breakout",
            entry_signal_strength=0.6,
            active_filters=["regime_gate"],
            passed_filters=["regime_gate"],
            strategy_params={"quality_threshold": 30},
            market_regime="trending_up",
        )
        trade2 = self.trade_logger.log_exit(
            trade_id="t2",
            exit_price=50000,
            exit_reason="STOP_LOSS",
            fees_paid=50,
        )
        assert trade2 is not None
        assert trade2.pnl < 0

        # --- Missed opportunity ---
        missed = self.missed_logger.log_missed(
            pair="035420",
            side="LONG",
            signal="momentum breakout",
            signal_id="kmp_breakout",
            signal_strength=0.75,
            blocked_by="volume_gate",
            block_reason="Volume ratio 0.8x below threshold 1.5x",
            strategy_type="kmp",
            market_regime="trending_up",
        )
        assert missed.blocked_by == "volume_gate"

        # --- Build daily snapshot ---
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build()
        builder.save(snapshot)

        # Verify trade counts
        assert snapshot.total_trades == 2
        assert snapshot.win_count == 1
        assert snapshot.loss_count == 1
        assert snapshot.net_pnl != 0

        # Verify missed count
        assert snapshot.missed_count == 1

        # --- Verify files on disk ---
        trade_files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        missed_files = list(Path(self.tmpdir).joinpath("missed").glob("*.jsonl"))
        daily_files = list(Path(self.tmpdir).joinpath("daily").glob("*.json"))

        assert len(trade_files) == 1
        assert len(missed_files) == 1
        assert len(daily_files) == 1

        # Verify trade JSONL has both entries and exits (4 lines: 2 entries + 2 exits)
        trade_lines = trade_files[0].read_text().strip().split("\n")
        assert len(trade_lines) == 4

        # Verify each line is valid JSON
        for line in trade_lines:
            data = json.loads(line)
            assert "trade_id" in data
            assert "event_metadata" in data

        # Verify daily snapshot JSON is valid
        daily_data = json.loads(daily_files[0].read_text())
        assert daily_data["total_trades"] == 2
        assert daily_data["bot_id"] == "integration_test"

    def test_process_scoring_integration(self):
        """Score a trade and verify the scorer integrates with trade events."""
        # Create a trade
        self.trade_logger.log_entry(
            trade_id="t1",
            pair="005930",
            side="LONG",
            entry_price=50000,
            position_size=10,
            position_size_quote=500000,
            entry_signal="test",
            entry_signal_id="test",
            entry_signal_strength=0.8,
            active_filters=[],
            passed_filters=[],
            strategy_params={},
        )
        trade = self.trade_logger.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fees_paid=50,
        )

        # Score it
        scorer = ProcessScorer()  # uses default rules
        trade_dict = trade.to_dict()
        # Map fields to what scorer expects
        trade_dict["regime"] = trade_dict.get("market_regime", "")
        trade_dict["signal_strength"] = trade_dict.get("entry_signal_strength", 0)
        score = scorer.score_trade(trade_dict, "kmp")

        assert 0 <= score.process_quality_score <= 100
        assert score.classification in ["good_process", "neutral", "bad_process"]

    def test_fault_tolerance_full_chain(self):
        """If snapshot service breaks, entire trade chain still works."""
        self.snap_service.capture_now.side_effect = Exception("total failure")

        # Entry should still succeed (degraded)
        trade = self.trade_logger.log_entry(
            trade_id="t1",
            pair="005930",
            side="LONG",
            entry_price=50000,
            position_size=10,
            position_size_quote=500000,
            entry_signal="test",
            entry_signal_id="test",
            entry_signal_strength=0.5,
            active_filters=[],
            passed_filters=[],
            strategy_params={},
        )
        assert trade.trade_id == "t1"

        # Missed opportunity should still succeed (degraded)
        missed = self.missed_logger.log_missed(
            pair="005930",
            side="LONG",
            signal="test",
            signal_id="test",
            signal_strength=0.5,
            blocked_by="regime_gate",
        )
        assert missed is not None

    def test_event_ids_are_unique(self):
        """All events from a day should have unique event_ids."""
        # Create multiple events
        for i in range(5):
            self.trade_logger.log_entry(
                trade_id=f"t{i}",
                pair="005930",
                side="LONG",
                entry_price=50000 + i * 100,
                position_size=10,
                position_size_quote=500000,
                entry_signal="test",
                entry_signal_id=f"test_{i}",
                entry_signal_strength=0.5,
                active_filters=[],
                passed_filters=[],
                strategy_params={},
            )

        trade_files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        assert len(trade_files) == 1

        event_ids = []
        for line in trade_files[0].read_text().strip().split("\n"):
            data = json.loads(line)
            eid = data.get("event_metadata", {}).get("event_id")
            if eid:
                event_ids.append(eid)

        assert len(event_ids) == len(set(event_ids)), "Duplicate event_ids found!"
