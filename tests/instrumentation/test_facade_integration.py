"""Tests that InstrumentationKit facade API is complete and consistent."""
import inspect
from instrumentation.facade import InstrumentationKit


REQUIRED_METHODS = [
    "on_entry_fill",
    "on_exit_fill",
    "on_signal_blocked",
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
