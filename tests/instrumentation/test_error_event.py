"""Tests for explicit error event emission."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from instrumentation.facade import InstrumentationKit


def test_emit_error_method_exists():
    """InstrumentationKit should have an emit_error method."""
    assert hasattr(InstrumentationKit, "emit_error")


def test_emit_error_writes_jsonl():
    """emit_error should write error events to bot_errors directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal InstrumentationKit with mocked dependencies
        kit = InstrumentationKit(
            trade_logger=MagicMock(),
            missed_logger=MagicMock(),
            snapshot_service=MagicMock(),
            process_scorer=MagicMock(),
            regime_classifier=MagicMock(),
            daily_builder=MagicMock(),
            heartbeat=MagicMock(),
            exit_backfiller=MagicMock(),
            data_provider=MagicMock(),
            strategy_type="kmp",
            data_dir=tmpdir,
        )
        kit.emit_error(
            severity="error",
            error_type="oms_timeout",
            message="OMS did not respond within 5s",
            stack_trace="Traceback...",
            context={"symbol": "005930", "intent_id": "abc123"},
        )

        err_dir = Path(tmpdir) / "bot_errors"
        files = list(err_dir.glob("*.jsonl"))
        assert len(files) == 1

        with open(files[0]) as f:
            record = json.loads(f.readline())

        assert record["severity"] == "error"
        assert record["error_type"] == "oms_timeout"
        assert record["message"] == "OMS did not respond within 5s"
        assert record["strategy_type"] == "kmp"
        assert record["context"]["symbol"] == "005930"


def test_emit_error_never_raises():
    """emit_error must never crash the strategy."""
    kit = InstrumentationKit(
        trade_logger=MagicMock(),
        missed_logger=MagicMock(),
        snapshot_service=MagicMock(),
        process_scorer=MagicMock(),
        regime_classifier=MagicMock(),
        daily_builder=MagicMock(),
        heartbeat=MagicMock(),
        exit_backfiller=MagicMock(),
        data_provider=MagicMock(),
        strategy_type="kmp",
        data_dir="/nonexistent/path/that/will/fail",
    )
    # Should not raise even with invalid path
    kit.emit_error(severity="critical", error_type="test", message="test")
