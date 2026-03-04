"""Tests for post-exit price movement backfill."""
import pytest
from datetime import datetime, timedelta, timezone
from instrumentation.src.exit_backfill import ExitBackfiller


class TestComputeMovement:
    def test_price_went_up_after_exit(self):
        """Exit was premature — price continued favorably."""
        backfiller = ExitBackfiller(data_dir="/tmp/test_instr")
        exit_time = "2026-03-03T11:00:00+09:00"
        result = backfiller._compute_movement(
            exit_price=50000.0,
            exit_time=exit_time,
            symbol="005930",
            candles=[
                {"time": "2026-03-03T12:00:00+09:00", "close": 51000},
                {"time": "2026-03-03T15:00:00+09:00", "close": 49000},
            ],
        )
        assert result["price_1h"] == 51000
        assert result["move_pct_1h"] == pytest.approx(2.0, abs=0.1)
        assert result["exit_was_premature_1h"] is True
        assert result["price_4h"] == 49000
        assert result["move_pct_4h"] == pytest.approx(-2.0, abs=0.1)
        assert result["exit_was_premature_4h"] is False

    def test_no_candles_returns_none(self):
        backfiller = ExitBackfiller(data_dir="/tmp/test_instr")
        result = backfiller._compute_movement(
            exit_price=50000.0,
            exit_time="2026-03-03T11:00:00+09:00",
            symbol="005930",
            candles=[],
        )
        assert result is None

    def test_zero_exit_price_returns_none(self):
        backfiller = ExitBackfiller(data_dir="/tmp/test_instr")
        result = backfiller._compute_movement(
            exit_price=0.0,
            exit_time="2026-03-03T11:00:00+09:00",
            symbol="005930",
            candles=[{"time": "2026-03-03T12:00:00+09:00", "close": 51000}],
        )
        assert result is None


class TestQueueExit:
    def test_queue_stores_pending(self):
        backfiller = ExitBackfiller(data_dir="/tmp/test_instr")
        backfiller.queue_exit(
            trade_id="KMP:005930:20260303",
            symbol="005930", side="LONG",
            exit_price=50000.0,
            exit_time="2026-03-03T11:00:00+09:00",
        )
        assert len(backfiller._pending) == 1
        assert backfiller._pending[0]["trade_id"] == "KMP:005930:20260303"
