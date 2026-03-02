"""Tests for daily_snapshot module."""

import json
import tempfile
from pathlib import Path

from instrumentation.src.daily_snapshot import DailySnapshot, DailySnapshotBuilder


def _write_jsonl(filepath: Path, events: list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


class TestDailySnapshot:
    def test_to_dict(self):
        snap = DailySnapshot(date="2026-03-01", bot_id="test", strategy_type="kmp")
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert d["date"] == "2026-03-01"
        assert d["bot_id"] == "test"

    def test_defaults(self):
        snap = DailySnapshot(date="2026-03-01", bot_id="test", strategy_type="kmp")
        assert snap.total_trades == 0
        assert snap.win_count == 0
        assert snap.net_pnl == 0.0
        assert snap.error_count == 0


class TestDailySnapshotBuilder:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "strategy_type": "kmp",
            "data_dir": self.tmpdir,
        }
        self.builder = DailySnapshotBuilder(self.config)
        self.date = "2026-03-01"

    def test_empty_day(self):
        snap = self.builder.build(self.date)
        assert snap.total_trades == 0
        assert snap.missed_count == 0
        assert snap.error_count == 0

    def test_trades_aggregated(self):
        trades = [
            # Entry event (ignored in aggregation — only exit events with pnl count)
            {"trade_id": "t1", "stage": "entry", "entry_price": 50000},
            # Exit: winning trade
            {
                "trade_id": "t1", "stage": "exit", "pnl": 1000.0,
                "fees_paid": 50.0, "entry_slippage_bps": 3.0,
                "exit_slippage_bps": 4.0, "entry_latency_ms": 200,
                "market_regime": "trending_up",
            },
            # Entry event
            {"trade_id": "t2", "stage": "entry", "entry_price": 51000},
            # Exit: losing trade
            {
                "trade_id": "t2", "stage": "exit", "pnl": -500.0,
                "fees_paid": 50.0, "entry_slippage_bps": 5.0,
                "exit_slippage_bps": 6.0, "entry_latency_ms": 300,
                "market_regime": "trending_up",
            },
        ]
        _write_jsonl(
            Path(self.tmpdir) / "trades" / f"trades_{self.date}.jsonl", trades
        )

        snap = self.builder.build(self.date)
        assert snap.total_trades == 2
        assert snap.win_count == 1
        assert snap.loss_count == 1
        assert snap.net_pnl == 500.0  # 1000 - 500
        assert snap.best_trade_pnl == 1000.0
        assert snap.worst_trade_pnl == -500.0
        assert snap.win_rate == 0.5
        assert snap.avg_entry_slippage_bps is not None
        assert snap.avg_exit_slippage_bps is not None
        assert snap.avg_entry_latency_ms is not None

    def test_missed_opportunities_counted(self):
        missed = [
            {
                "pair": "005930", "blocked_by": "volume_gate",
                "first_hit": "TP", "hypothetical_pnl": 200,
            },
            {
                "pair": "000660", "blocked_by": "regime_gate",
                "first_hit": "SL", "hypothetical_pnl": -100,
            },
            {
                "pair": "035420", "blocked_by": "volume_gate",
                "first_hit": "TP", "hypothetical_pnl": 150,
            },
        ]
        _write_jsonl(
            Path(self.tmpdir) / "missed" / f"missed_{self.date}.jsonl", missed
        )

        snap = self.builder.build(self.date)
        assert snap.missed_count == 3
        assert snap.missed_would_have_won == 2
        assert snap.top_missed_filter == "volume_gate"

    def test_process_scores_aggregated(self):
        scores = [
            {
                "trade_id": "t1",
                "process_quality_score": 90,
                "classification": "good_process",
                "root_causes": [],
            },
            {
                "trade_id": "t2",
                "process_quality_score": 30,
                "classification": "bad_process",
                "root_causes": ["regime_mismatch", "weak_signal"],
            },
        ]
        _write_jsonl(
            Path(self.tmpdir) / "scores" / f"scores_{self.date}.jsonl", scores
        )

        snap = self.builder.build(self.date)
        assert snap.avg_process_quality == 60.0
        assert snap.process_scores_distribution["good_process"] == 1
        assert snap.process_scores_distribution["bad_process"] == 1
        assert snap.root_cause_distribution.get("regime_mismatch") == 1

    def test_errors_counted(self):
        errors = [
            {"timestamp": "2026-03-01T10:00:00Z", "component": "trade_logger", "error": "test"},
            {"timestamp": "2026-03-01T11:00:00Z", "component": "snapshot", "error": "test2"},
        ]
        _write_jsonl(
            Path(self.tmpdir) / "errors" / f"instrumentation_errors_{self.date}.jsonl",
            errors,
        )

        snap = self.builder.build(self.date)
        assert snap.error_count == 2

    def test_regime_breakdown(self):
        trades = [
            {"trade_id": "t1", "stage": "exit", "pnl": 500, "market_regime": "trending_up", "fees_paid": 0},
            {"trade_id": "t2", "stage": "exit", "pnl": -200, "market_regime": "trending_up", "fees_paid": 0},
            {"trade_id": "t3", "stage": "exit", "pnl": 300, "market_regime": "ranging", "fees_paid": 0},
        ]
        _write_jsonl(
            Path(self.tmpdir) / "trades" / f"trades_{self.date}.jsonl", trades
        )

        snap = self.builder.build(self.date)
        assert "trending_up" in snap.regime_breakdown
        assert snap.regime_breakdown["trending_up"]["trades"] == 2
        assert "ranging" in snap.regime_breakdown
        assert snap.regime_breakdown["ranging"]["trades"] == 1

    def test_save_creates_json(self):
        snap = self.builder.build(self.date)
        self.builder.save(snap)

        filepath = Path(self.tmpdir) / "daily" / f"daily_{self.date}.json"
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["date"] == self.date
        assert data["bot_id"] == "test_bot"

    def test_profit_factor(self):
        trades = [
            {"trade_id": "t1", "stage": "exit", "pnl": 1000, "fees_paid": 0},
            {"trade_id": "t2", "stage": "exit", "pnl": -500, "fees_paid": 0},
        ]
        _write_jsonl(
            Path(self.tmpdir) / "trades" / f"trades_{self.date}.jsonl", trades
        )
        snap = self.builder.build(self.date)
        assert snap.profit_factor == 2.0  # 1000 / 500
