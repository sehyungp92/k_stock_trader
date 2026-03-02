"""Tests for event_metadata module."""

from datetime import datetime, timezone
from instrumentation.src.event_metadata import (
    compute_event_id,
    create_event_metadata,
    compute_clock_skew,
    EventMetadata,
)


class TestComputeEventId:
    def test_deterministic(self):
        """Same inputs must always produce the same event_id."""
        id1 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc123")
        id2 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc123")
        assert id1 == id2

    def test_unique_on_different_input(self):
        """Different inputs must produce different event_ids."""
        id1 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc123")
        id2 = compute_event_id("bot1", "2026-03-01T10:00:01Z", "trade", "abc123")
        assert id1 != id2

    def test_unique_on_different_event_type(self):
        id1 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc123")
        id2 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "missed_opportunity", "abc123")
        assert id1 != id2

    def test_unique_on_different_bot(self):
        id1 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc123")
        id2 = compute_event_id("bot2", "2026-03-01T10:00:00Z", "trade", "abc123")
        assert id1 != id2

    def test_length_is_16(self):
        eid = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc")
        assert len(eid) == 16

    def test_hex_characters_only(self):
        eid = compute_event_id("bot1", "ts", "trade", "key")
        assert all(c in "0123456789abcdef" for c in eid)


class TestComputeClockSkew:
    def test_positive_skew(self):
        """Exchange ahead of local."""
        exch = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        local = datetime(2026, 3, 1, 9, 59, 59, tzinfo=timezone.utc)
        assert compute_clock_skew(exch, local) == 1000

    def test_negative_skew(self):
        """Exchange behind local."""
        exch = datetime(2026, 3, 1, 9, 59, 59, tzinfo=timezone.utc)
        local = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert compute_clock_skew(exch, local) == -1000

    def test_zero_skew(self):
        ts = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert compute_clock_skew(ts, ts) == 0


class TestCreateEventMetadata:
    def test_returns_all_fields(self):
        now = datetime.now(timezone.utc)
        meta = create_event_metadata(
            bot_id="bot1",
            event_type="trade",
            payload_key="test123",
            exchange_timestamp=now,
            data_source_id="test_source",
        )
        assert isinstance(meta, EventMetadata)
        assert meta.event_id
        assert meta.bot_id == "bot1"
        assert meta.exchange_timestamp
        assert meta.local_timestamp
        assert meta.data_source_id == "test_source"
        assert meta.bar_id is None

    def test_bar_id_passed_through(self):
        now = datetime.now(timezone.utc)
        meta = create_event_metadata(
            bot_id="bot1",
            event_type="trade",
            payload_key="test123",
            exchange_timestamp=now,
            data_source_id="test_source",
            bar_id="2026-03-01T14:00+09:00_1d",
        )
        assert meta.bar_id == "2026-03-01T14:00+09:00_1d"

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        meta = create_event_metadata(
            bot_id="bot1",
            event_type="trade",
            payload_key="test123",
            exchange_timestamp=now,
            data_source_id="test_source",
        )
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert "event_id" in d
        assert "bot_id" in d
        assert "clock_skew_ms" in d
