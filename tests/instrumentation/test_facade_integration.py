"""Tests that InstrumentationKit facade API is complete and consistent."""
import inspect
from instrumentation.facade import InstrumentationKit


REQUIRED_METHODS = [
    "on_entry_fill",
    "on_exit_fill",
    "on_signal_blocked",
    "on_order_event",
    "periodic_tick",
    "build_daily_snapshot",
    "classify_regime",
    "emit_heartbeat",
    "emit_error",
    "shutdown",
]


def test_facade_has_all_required_methods():
    for method_name in REQUIRED_METHODS:
        assert hasattr(InstrumentationKit, method_name), \
            f"InstrumentationKit missing method: {method_name}"
        assert callable(getattr(InstrumentationKit, method_name))


def test_on_entry_fill_accepts_new_params():
    sig = inspect.signature(InstrumentationKit.on_entry_fill)
    params = list(sig.parameters.keys())
    assert "signal_factors" in params
    assert "filter_decisions" in params
    assert "sizing_context" in params


def test_on_signal_blocked_accepts_filter_decisions():
    sig = inspect.signature(InstrumentationKit.on_signal_blocked)
    params = list(sig.parameters.keys())
    assert "filter_decisions" in params


def test_on_entry_fill_accepts_param_set_id():
    sig = inspect.signature(InstrumentationKit.on_entry_fill)
    params = list(sig.parameters.keys())
    assert "param_set_id" in params


def test_trade_event_has_param_set_id():
    from instrumentation.src.trade_logger import TradeEvent
    event = TradeEvent(
        trade_id="ps_test",
        event_metadata={"event_id": "ps1"},
        entry_snapshot={},
    )
    d = event.to_dict()
    assert "param_set_id" in d
    assert d["param_set_id"] is None  # backward compat default


def test_daily_snapshot_has_new_fields():
    from instrumentation.src.daily_snapshot import DailySnapshot
    snap = DailySnapshot(date="2026-03-05", bot_id="test", strategy_type="kmp")
    d = snap.to_dict()
    assert d["max_concurrent_positions"] == 0
    assert d["avg_exit_efficiency"] is None
    assert d["total_risk_deployed_pct"] == 0.0


def test_jsonl_backward_compat():
    """Old consumers that don't know about new fields should still work."""
    from instrumentation.src.trade_logger import TradeEvent
    event = TradeEvent(
        trade_id="compat_1",
        event_metadata={"event_id": "ec1"},
        entry_snapshot={},
    )
    d = event.to_dict()
    # New fields present with safe defaults
    assert d["signal_factors"] == []
    assert d["filter_decisions"] == []
    assert d["sizing_context"] is None
    assert d["regime_context"] is None
    assert d["param_set_id"] is None


def test_on_order_event_signature():
    sig = inspect.signature(InstrumentationKit.on_order_event)
    params = list(sig.parameters.keys())
    assert "order_id" in params
    assert "pair" in params
    assert "order_type" in params
    assert "status" in params
    assert "requested_qty" in params
    assert "reject_reason" in params
    assert "latency_ms" in params


def test_emit_heartbeat_accepts_positions():
    sig = inspect.signature(InstrumentationKit.emit_heartbeat)
    params = list(sig.parameters.keys())
    assert "positions" in params
    assert "portfolio_exposure" in params


def test_daily_snapshot_has_experiment_breakdown():
    from instrumentation.src.daily_snapshot import DailySnapshot
    snap = DailySnapshot(date="2026-03-06", bot_id="test", strategy_type="kmp")
    d = snap.to_dict()
    assert "experiment_breakdown" in d
    assert d["experiment_breakdown"] == {}


def test_sidecar_orders_mapping():
    from instrumentation.src.sidecar import _DIR_TO_EVENT_TYPE, _EVENT_TYPE_PRIORITY
    assert _DIR_TO_EVENT_TYPE["orders"] == "order"
    assert _EVENT_TYPE_PRIORITY["order"] == "normal"


def test_on_order_event_fire_and_forget():
    """on_order_event should never raise even with bad inputs."""
    import tempfile
    from unittest.mock import MagicMock
    from instrumentation.src.order_logger import OrderLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal kit with mocked dependencies
        kit = InstrumentationKit.__new__(InstrumentationKit)
        kit._order_logger = OrderLogger({"bot_id": "test", "data_dir": tmpdir})
        # Call with valid params — should not raise
        kit.on_order_event(
            order_id="test_001", pair="005930", order_type="MARKET",
            status="SUBMITTED", requested_qty=10,
        )
        # Call with broken logger — should swallow exception
        kit._order_logger = MagicMock()
        kit._order_logger.log_order.side_effect = RuntimeError("boom")
        kit.on_order_event(
            order_id="test_002", pair="005930", order_type="MARKET",
            status="SUBMITTED", requested_qty=10,
        )  # should not raise
