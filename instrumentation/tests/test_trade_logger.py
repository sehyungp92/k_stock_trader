"""Tests for trade_logger module."""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

from instrumentation.src.trade_logger import TradeLogger, TradeEvent
from instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService


def _mock_snapshot():
    return MarketSnapshot(
        snapshot_id="test_snap_001",
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


class TestTradeLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
        }
        self.snap_service = MagicMock(spec=MarketSnapshotService)
        self.snap_service.capture_now.return_value = _mock_snapshot()
        self.logger = TradeLogger(self.config, self.snap_service)

    def test_log_entry_creates_event(self):
        trade = self.logger.log_entry(
            trade_id="t1",
            pair="005930",
            side="LONG",
            entry_price=50000,
            position_size=10,
            position_size_quote=500000,
            entry_signal="momentum breakout",
            entry_signal_id="kmp_breakout",
            entry_signal_strength=0.8,
            active_filters=["volume_gate", "regime_gate"],
            passed_filters=["volume_gate", "regime_gate"],
            strategy_params={"quality_threshold": 30},
        )
        assert isinstance(trade, TradeEvent)
        assert trade.trade_id == "t1"
        assert trade.pair == "005930"
        assert trade.side == "LONG"
        assert trade.stage == "entry"
        assert trade.entry_price == 50000

    def test_log_entry_captures_snapshot(self):
        trade = self.logger.log_entry(
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
        assert trade.entry_snapshot != {}
        assert trade.atr_at_entry == 500.0

    def test_log_exit_computes_pnl(self):
        self.logger.log_entry(
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
        trade = self.logger.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fees_paid=100,
        )
        assert trade is not None
        # PnL = (51000 - 50000) * 10 - 100 = 9900
        assert trade.pnl == 9900.0
        assert trade.pnl_pct is not None
        assert trade.pnl_pct > 0
        assert trade.stage == "exit"
        assert trade.exit_reason == "TAKE_PROFIT"

    def test_log_exit_losing_trade(self):
        self.logger.log_entry(
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
        trade = self.logger.log_exit(
            trade_id="t1",
            exit_price=49000,
            exit_reason="STOP_LOSS",
            fees_paid=100,
        )
        assert trade is not None
        # PnL = (49000 - 50000) * 10 - 100 = -10100
        assert trade.pnl == -10100.0
        assert trade.pnl_pct < 0

    def test_log_exit_missing_trade_returns_none(self):
        result = self.logger.log_exit(
            trade_id="nonexistent",
            exit_price=51000,
            exit_reason="SIGNAL",
        )
        assert result is None

    def test_entry_failure_does_not_crash(self):
        """Instrumentation failure must never block trading."""
        self.snap_service.capture_now.side_effect = Exception("broken")
        trade = self.logger.log_entry(
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
        # Should return a minimal trade, not crash
        assert trade.trade_id == "t1"

    def test_events_written_to_jsonl(self):
        self.logger.log_entry(
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
        files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text().strip()
        data = json.loads(content)
        assert data["trade_id"] == "t1"
        assert data["stage"] == "entry"

    def test_entry_slippage_computed(self):
        trade = self.logger.log_entry(
            trade_id="t1",
            pair="005930",
            side="LONG",
            entry_price=50050,
            position_size=10,
            position_size_quote=500500,
            entry_signal="test",
            entry_signal_id="test",
            entry_signal_strength=0.5,
            active_filters=[],
            passed_filters=[],
            strategy_params={},
            expected_entry_price=50000,
        )
        assert trade.entry_slippage_bps is not None
        assert trade.entry_slippage_bps > 0

    def test_open_trades_tracking(self):
        self.logger.log_entry(
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
        assert "t1" in self.logger.get_open_trades()

        self.logger.log_exit(trade_id="t1", exit_price=51000, exit_reason="TP")
        assert "t1" not in self.logger.get_open_trades()

    def test_event_metadata_present(self):
        trade = self.logger.log_entry(
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
        assert trade.event_metadata
        assert "event_id" in trade.event_metadata
        assert "bot_id" in trade.event_metadata
