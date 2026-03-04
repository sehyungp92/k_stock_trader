"""Tests for drawdown state computation."""
from instrumentation.src.drawdown import compute_drawdown_context
from instrumentation.src.trade_logger import TradeEvent


class TestComputeDrawdownContext:
    def test_positive_pnl(self):
        ctx = compute_drawdown_context(daily_pnl_pct=1.5)
        assert ctx["drawdown_pct"] == 0.0
        assert ctx["drawdown_tier"] == "full"
        assert ctx["drawdown_size_mult"] == 1.0

    def test_small_drawdown(self):
        ctx = compute_drawdown_context(daily_pnl_pct=-0.5)
        assert ctx["drawdown_pct"] == -0.5
        assert ctx["drawdown_tier"] == "full"
        assert ctx["drawdown_size_mult"] == 1.0

    def test_half_tier(self):
        ctx = compute_drawdown_context(daily_pnl_pct=-1.5)
        assert ctx["drawdown_tier"] == "half"
        assert ctx["drawdown_size_mult"] == 0.5

    def test_quarter_tier(self):
        ctx = compute_drawdown_context(daily_pnl_pct=-2.5)
        assert ctx["drawdown_tier"] == "quarter"
        assert ctx["drawdown_size_mult"] == 0.25

    def test_halt_tier(self):
        ctx = compute_drawdown_context(daily_pnl_pct=-3.5)
        assert ctx["drawdown_tier"] == "halt"
        assert ctx["drawdown_size_mult"] == 0.0

    def test_exactly_at_boundary(self):
        ctx = compute_drawdown_context(daily_pnl_pct=-1.0)
        assert ctx["drawdown_tier"] == "full"

    def test_zero_pnl(self):
        ctx = compute_drawdown_context(daily_pnl_pct=0.0)
        assert ctx["drawdown_pct"] == 0.0
        assert ctx["drawdown_tier"] == "full"


class TestTradeEventDrawdownFields:
    def test_fields_exist(self):
        event = TradeEvent(
            trade_id="dd_001",
            event_metadata={"event_id": "e1"},
            entry_snapshot={},
            drawdown_pct=-1.5,
            drawdown_tier="half",
            drawdown_size_mult=0.5,
        )
        assert event.drawdown_pct == -1.5
        assert event.drawdown_tier == "half"
        assert event.drawdown_size_mult == 0.5

    def test_defaults_none(self):
        event = TradeEvent(
            trade_id="dd_002",
            event_metadata={"event_id": "e2"},
            entry_snapshot={},
        )
        assert event.drawdown_pct is None
        assert event.drawdown_tier is None
        assert event.drawdown_size_mult is None


class TestFacadeDrawdownParam:
    def test_on_entry_fill_accepts_drawdown_context(self):
        import inspect
        from instrumentation.facade import InstrumentationKit
        sig = inspect.signature(InstrumentationKit.on_entry_fill)
        assert "drawdown_context" in sig.parameters
