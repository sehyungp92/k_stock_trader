"""Tests for experiment_id and session_type fields."""
from datetime import datetime, timezone, timedelta
from instrumentation.src.trade_logger import TradeEvent
from instrumentation.src.session import classify_session_type

KST = timezone(timedelta(hours=9))


class TestExperimentId:
    def test_field_exists(self):
        event = TradeEvent(
            trade_id="exp_001",
            event_metadata={"event_id": "e1"},
            entry_snapshot={},
            experiment_id="kmp_v2_relaxed",
        )
        assert event.experiment_id == "kmp_v2_relaxed"
        assert event.to_dict()["experiment_id"] == "kmp_v2_relaxed"

    def test_defaults_none(self):
        event = TradeEvent(
            trade_id="exp_002",
            event_metadata={"event_id": "e2"},
            entry_snapshot={},
        )
        assert event.experiment_id is None

    def test_facade_accepts_experiment_id(self):
        import inspect
        from instrumentation.facade import InstrumentationKit
        sig = inspect.signature(InstrumentationKit.on_entry_fill)
        assert "experiment_id" in sig.parameters


class TestSessionType:
    def test_pre_market(self):
        t = datetime(2026, 3, 4, 8, 45, tzinfo=KST)
        assert classify_session_type(t) == "pre_market"

    def test_regular(self):
        t = datetime(2026, 3, 4, 10, 30, tzinfo=KST)
        assert classify_session_type(t) == "regular"

    def test_opening(self):
        t = datetime(2026, 3, 4, 9, 0, tzinfo=KST)
        assert classify_session_type(t) == "regular"

    def test_closing_auction(self):
        t = datetime(2026, 3, 4, 15, 25, tzinfo=KST)
        assert classify_session_type(t) == "closing_auction"

    def test_after_hours(self):
        t = datetime(2026, 3, 4, 16, 30, tzinfo=KST)
        assert classify_session_type(t) == "after_hours"

    def test_field_on_trade_event(self):
        event = TradeEvent(
            trade_id="st_001",
            event_metadata={"event_id": "e1"},
            entry_snapshot={},
            session_type="regular",
        )
        assert event.session_type == "regular"

    def test_session_type_defaults_none(self):
        event = TradeEvent(
            trade_id="st_002",
            event_metadata={"event_id": "e2"},
            entry_snapshot={},
        )
        assert event.session_type is None
