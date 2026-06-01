"""Tests for the instrumentation sidecar relay path."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from instrumentation.src.market_snapshot import MarketSnapshot
from instrumentation.src.trade_logger import TradeLogger
from instrumentation.src.sidecar import Sidecar

from schemas.events import TradeEvent as AssistantTradeEvent


class _SnapshotService:
    def capture_now(self, symbol: str) -> MarketSnapshot:
        timestamp = datetime.now(timezone.utc).isoformat()
        return MarketSnapshot(
            snapshot_id="snap-1",
            symbol=symbol,
            timestamp=timestamp,
            mid=70000.0,
            last_trade_price=70000.0,
            volume_1m=1000.0,
            volume_5m=5000.0,
            volume_24h=100000.0,
            atr_14=1200.0,
        )


def _make_config(tmp_path, relay_url: str = "") -> dict:
    return {
        "bot_id": "k_stock_trader",
        "data_dir": str(tmp_path),
        "sidecar": {
            "relay_url": relay_url,
            "buffer_dir": str(tmp_path / ".sidecar_buffer"),
        },
    }


def test_relay_url_normalizes_events_endpoint(tmp_path, monkeypatch):
    """A base relay URL should be normalized to the /events ingest endpoint."""
    monkeypatch.delenv("RELAY_URL", raising=False)

    sidecar = Sidecar(_make_config(tmp_path, "https://relay.example.com"))
    already_normalized = Sidecar(_make_config(tmp_path, "https://relay.example.com/events"))

    assert sidecar.relay_url == "https://relay.example.com/events"
    assert already_normalized.relay_url == "https://relay.example.com/events"


def test_read_unsent_events_streams_jsonl_with_watermark(tmp_path, monkeypatch):
    """JSONL backlogs should stream from disk and resume after the saved watermark."""
    monkeypatch.delenv("RELAY_URL", raising=False)

    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    filepath = trades_dir / "trades_2026-03-10.jsonl"
    filepath.write_text(
        "\n".join(
            [
                json.dumps({"trade_id": "t1", "timestamp": "2026-03-10T09:00:00Z"}),
                json.dumps({"trade_id": "t2", "timestamp": "2026-03-10T09:01:00Z"}),
                json.dumps({"trade_id": "t3", "timestamp": "2026-03-10T09:02:00Z"}),
            ]
        ),
        encoding="utf-8",
    )

    sidecar = Sidecar(_make_config(tmp_path))
    sidecar.watermarks[str(filepath)] = 1

    events = sidecar._read_unsent_events(filepath, "trade")

    assert [event["_line_number"] for event in events] == [1, 2]
    assert [json.loads(event["payload"])["trade_id"] for event in events] == ["t2", "t3"]


def test_daily_directory_includes_canonical_jsonl(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAY_URL", raising=False)

    daily_dir = tmp_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    filepath = daily_dir / "daily_2026-03-10.jsonl"
    filepath.write_text(
        json.dumps(
            {
                "event_id": "evt-1",
                "bot_id": "k_stock_trader",
                "event_type": "session_closeout",
                "exchange_timestamp": "2026-03-10T15:30:00+09:00",
                "payload": {"record_type": "session_closeout"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    files = Sidecar(_make_config(tmp_path))._get_event_files()

    assert (filepath, "daily_snapshot") in files


def test_trade_forwarding_skips_entry_stage_and_validates_completed_trade(tmp_path, monkeypatch):
    """Only completed trade records should leave the bot as assistant trade events."""
    monkeypatch.delenv("RELAY_URL", raising=False)

    config = _make_config(tmp_path)
    logger = TradeLogger(config, _SnapshotService())
    logger.log_entry(
        trade_id="trade-1",
        pair="005930",
        side="LONG",
        entry_price=70000.0,
        position_size=10,
        position_size_quote=700000.0,
        entry_signal="gap_breakout",
        entry_signal_id="sig-1",
        entry_signal_strength=0.9,
        active_filters=[],
        passed_filters=[],
        strategy_params={},
        bot_id="k_stock_trader",
        strategy_id="ALPHA",
    )
    logger.log_exit(
        trade_id="trade-1",
        exit_price=71000.0,
        exit_reason="SIGNAL",
    )

    trade_files = sorted((tmp_path / "trades").glob("trades_*.jsonl"))
    sidecar = Sidecar(config)

    events = sidecar._read_unsent_events(trade_files[0], "trade")

    assert len(events) == 1
    payload = json.loads(events[0]["payload"])
    assert payload["stage"] == "exit"
    AssistantTradeEvent.model_validate(payload)
