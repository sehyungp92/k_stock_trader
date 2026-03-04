"""Tests for heartbeat emission."""
import json
import tempfile
from pathlib import Path
from instrumentation.src.heartbeat import HeartbeatEmitter


def test_emit_heartbeat_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        emitter = HeartbeatEmitter(
            bot_id="k_stock_trader_kmp",
            strategy_type="kmp",
            data_dir=tmpdir,
        )
        emitter.emit(active_positions=3, open_orders=1, uptime_s=3600)

        hb_dir = Path(tmpdir) / "heartbeats"
        files = list(hb_dir.glob("*.jsonl"))
        assert len(files) == 1

        with open(files[0]) as f:
            record = json.loads(f.readline())

        assert record["bot_id"] == "k_stock_trader_kmp"
        assert record["strategy_type"] == "kmp"
        assert record["active_positions"] == 3
        assert record["open_orders"] == 1
        assert record["status"] == "alive"
        assert record["uptime_s"] == 3600.0


def test_emit_heartbeat_with_extra():
    with tempfile.TemporaryDirectory() as tmpdir:
        emitter = HeartbeatEmitter(
            bot_id="k_stock_trader_kpr",
            strategy_type="kpr",
            data_dir=tmpdir,
        )
        emitter.emit(active_positions=0, extra={"ws_connected": True})

        hb_dir = Path(tmpdir) / "heartbeats"
        files = list(hb_dir.glob("*.jsonl"))
        with open(files[0]) as f:
            record = json.loads(f.readline())

        assert record["extra"]["ws_connected"] is True


def test_heartbeat_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        emitter = HeartbeatEmitter(
            bot_id="test",
            strategy_type="test",
            data_dir=tmpdir,
        )
        emitter.emit()

        hb_dir = Path(tmpdir) / "heartbeats"
        files = list(hb_dir.glob("*.jsonl"))
        with open(files[0]) as f:
            record = json.loads(f.readline())

        assert record["active_positions"] == 0
        assert record["error_count_1h"] == 0
        assert "extra" not in record
