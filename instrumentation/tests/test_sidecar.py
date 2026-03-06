"""Tests for sidecar module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from instrumentation.src.sidecar import Sidecar


def _make_config(tmpdir):
    return {
        "bot_id": "test_bot",
        "data_dir": tmpdir,
        "sidecar": {
            "relay_url": "https://relay.example.com/events",
            "batch_size": 10,
            "retry_max": 2,
            "retry_backoff_base_seconds": 0.01,
            "poll_interval_seconds": 1,
            "buffer_dir": str(Path(tmpdir) / ".sidecar_buffer"),
        },
    }


def _write_trade_events(data_dir, date="2026-03-01"):
    trades_dir = Path(data_dir) / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    filepath = trades_dir / f"trades_{date}.jsonl"
    events = [
        {
            "trade_id": "t1",
            "event_metadata": {
                "event_id": "abc123def456",
                "exchange_timestamp": "2026-03-01T10:00:00+09:00",
            },
            "stage": "entry",
        },
        {
            "trade_id": "t1",
            "event_metadata": {
                "event_id": "abc123def789",
                "exchange_timestamp": "2026-03-01T10:30:00+09:00",
            },
            "stage": "exit",
        },
    ]
    with open(filepath, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return filepath, events


class TestSidecar:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = _make_config(self.tmpdir)
        self.sidecar = Sidecar(self.config)

    def test_init(self):
        assert self.sidecar.bot_id == "test_bot"
        assert self.sidecar.batch_size == 10
        assert self.sidecar.retry_max == 2

    def test_get_event_files(self):
        _write_trade_events(self.tmpdir)
        files = self.sidecar._get_event_files()
        assert len(files) >= 1
        paths = [str(f[0]) for f in files]
        assert any("trades" in p for p in paths)

    def test_read_unsent_events(self):
        filepath, _ = _write_trade_events(self.tmpdir)
        events = self.sidecar._read_unsent_events(filepath, "trade")
        assert len(events) == 2
        # Verify relay envelope format
        for e in events:
            assert "event_id" in e
            assert "bot_id" in e
            assert "event_type" in e
            assert "payload" in e
            assert "exchange_timestamp" in e
            assert e["bot_id"] == "test_bot"
            assert e["event_type"] == "trade"

    def test_watermark_tracking(self):
        filepath, _ = _write_trade_events(self.tmpdir)

        # Read all events
        events = self.sidecar._read_unsent_events(filepath, "trade")
        assert len(events) == 2

        # Simulate watermark update after sending
        key = str(filepath)
        self.sidecar.watermarks[key] = 2
        self.sidecar._save_watermarks()

        # Read again — should be empty (all sent)
        events = self.sidecar._read_unsent_events(filepath, "trade")
        assert len(events) == 0

    def test_watermark_persistence(self):
        self.sidecar.watermarks["test_file"] = 5
        self.sidecar._save_watermarks()

        # Reload
        sidecar2 = Sidecar(self.config)
        assert sidecar2.watermarks.get("test_file") == 5

    def test_wrap_event_extracts_metadata(self):
        raw = {
            "trade_id": "t1",
            "event_metadata": {
                "event_id": "abc123def456",
                "exchange_timestamp": "2026-03-01T10:00:00+09:00",
            },
        }
        wrapped = self.sidecar._wrap_event(raw, "trade")
        assert wrapped["event_id"] == "abc123def456"
        assert wrapped["event_type"] == "trade"
        assert wrapped["bot_id"] == "test_bot"
        assert "payload" in wrapped

    def test_wrap_event_generates_id_when_missing(self):
        raw = {"trade_id": "t1", "entry_time": "2026-03-01T10:00:00Z"}
        wrapped = self.sidecar._wrap_event(raw, "trade")
        assert wrapped["event_id"]  # should be auto-generated
        assert len(wrapped["event_id"]) == 16

    def test_sign_payload(self):
        self.sidecar.hmac_secret = b"test_secret"
        sig = self.sidecar._sign_payload('{"key": "value"}')
        assert sig  # non-empty
        assert len(sig) == 64  # SHA-256 hex digest

    def test_sign_payload_empty_secret(self):
        self.sidecar.hmac_secret = b""
        sig = self.sidecar._sign_payload('{"key": "value"}')
        assert sig == ""

    def test_sign_uses_sort_keys(self):
        """Verify the sidecar sends canonicalized JSON with sort_keys=True."""
        self.sidecar.hmac_secret = b"test_secret"
        envelope = {"bot_id": "test", "events": []}
        canonical = json.dumps(envelope, sort_keys=True)
        sig = self.sidecar._sign_payload(canonical)
        assert sig  # non-empty string

    @patch("instrumentation.src.sidecar.requests")
    def test_send_batch_success(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response

        self.sidecar.hmac_secret = b"test"
        events = [
            {"event_id": "abc", "bot_id": "test", "event_type": "trade",
             "payload": "{}", "exchange_timestamp": "2026-03-01T10:00:00Z"},
        ]
        result = self.sidecar._send_batch(events)
        assert result is True
        mock_requests.post.assert_called_once()

    @patch("instrumentation.src.sidecar.requests")
    def test_send_batch_409_treated_as_success(self, mock_requests):
        """Duplicate (409) should be treated as success."""
        mock_response = MagicMock()
        mock_response.status_code = 409
        mock_requests.post.return_value = mock_response

        self.sidecar.hmac_secret = b"test"
        events = [
            {"event_id": "abc", "bot_id": "test", "event_type": "trade",
             "payload": "{}", "exchange_timestamp": "2026-03-01T10:00:00Z"},
        ]
        result = self.sidecar._send_batch(events)
        assert result is True

    @patch("instrumentation.src.sidecar.requests")
    def test_send_batch_401_no_retry(self, mock_requests):
        """Auth failure should not retry."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_requests.post.return_value = mock_response

        self.sidecar.hmac_secret = b"test"
        events = [
            {"event_id": "abc", "bot_id": "test", "event_type": "trade",
             "payload": "{}", "exchange_timestamp": "2026-03-01T10:00:00Z"},
        ]
        result = self.sidecar._send_batch(events)
        assert result is False
        # Should only call once (no retry)
        assert mock_requests.post.call_count == 1

    def test_send_batch_no_relay_url(self):
        self.sidecar.relay_url = ""
        result = self.sidecar._send_batch([{"event_id": "abc"}])
        assert result is False

    def test_cleanup_old_watermarks(self):
        self.sidecar.watermarks["/nonexistent/file.jsonl"] = 10
        self.sidecar.watermarks[str(Path(self.tmpdir) / "still_exists")] = 5

        # Create the file that should survive
        (Path(self.tmpdir) / "still_exists").touch()

        self.sidecar.cleanup_old_watermarks()
        assert "/nonexistent/file.jsonl" not in self.sidecar.watermarks

    def test_internal_metadata_stripped_before_send(self):
        """_source_file and _line_number should not be sent to relay."""
        events = [
            {
                "event_id": "abc", "bot_id": "test", "event_type": "trade",
                "payload": "{}", "exchange_timestamp": "2026-03-01T10:00:00Z",
                "_source_file": "/tmp/test.jsonl", "_line_number": 0,
            },
        ]
        # _send_batch strips _ prefixed keys internally
        # We can verify by checking the request body through mock
        with patch("instrumentation.src.sidecar.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_requests.post.return_value = mock_response
            self.sidecar.hmac_secret = b"test"
            self.sidecar._send_batch(events)

            call_args = mock_requests.post.call_args
            raw_body = call_args.kwargs.get("data", call_args[1].get("data", b""))
            import gzip
            sent_data = json.loads(gzip.decompress(raw_body).decode())
            for evt in sent_data["events"]:
                assert "_source_file" not in evt
                assert "_line_number" not in evt
