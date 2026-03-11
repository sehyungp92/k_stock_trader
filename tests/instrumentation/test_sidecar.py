"""Tests for the instrumentation sidecar relay path."""

from __future__ import annotations

import json

from instrumentation.src.sidecar import Sidecar


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
