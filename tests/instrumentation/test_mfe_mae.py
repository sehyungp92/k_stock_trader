"""Tests for MFE/MAE tracking on TradeEvent and shared helper."""
import math
import pytest
from instrumentation.src.trade_logger import TradeEvent
from instrumentation.src.mfe_mae import build_mfe_mae_context


class TestBuildMfeMaeContext:
    def test_normal_long_trade(self):
        ctx = build_mfe_mae_context(
            entry_price=70000, stop_price=69000,
            max_fav_price=73000, min_adverse_price=69500,
        )
        assert ctx["mfe_price"] == 73000
        assert ctx["mae_price"] == 69500
        assert abs(ctx["mfe_pct"] - 4.2857) < 0.01
        assert abs(ctx["mae_pct"] - 0.7143) < 0.01
        assert abs(ctx["mfe_r"] - 3.0) < 0.01
        assert abs(ctx["mae_r"] - 0.5) < 0.01

    def test_no_mfe_data(self):
        ctx = build_mfe_mae_context(
            entry_price=70000, stop_price=69000,
            max_fav_price=0, min_adverse_price=float("inf"),
        )
        assert ctx["mfe_price"] is None
        assert ctx["mae_price"] is None
        assert ctx["mfe_pct"] is None
        assert ctx["mae_pct"] is None

    def test_entry_equals_stop(self):
        """Edge case: stop at entry (risk ~0)."""
        ctx = build_mfe_mae_context(
            entry_price=70000, stop_price=70000,
            max_fav_price=71000, min_adverse_price=69000,
        )
        assert ctx["mfe_price"] == 71000
        assert ctx["mae_price"] == 69000
        # R-multiples will be very large but should not crash
        assert ctx["mfe_r"] is not None

    def test_breakeven_trade(self):
        ctx = build_mfe_mae_context(
            entry_price=70000, stop_price=69000,
            max_fav_price=70000, min_adverse_price=70000,
        )
        assert ctx["mfe_pct"] == 0.0
        assert ctx["mae_pct"] == 0.0


class TestTradeEventMfeMae:
    def test_fields_exist(self):
        event = TradeEvent(
            trade_id="test_001",
            event_metadata={"event_id": "e1"},
            entry_snapshot={},
            mfe_price=73000.0,
            mae_price=69500.0,
            mfe_pct=4.29,
            mae_pct=0.71,
            mfe_r=3.0,
            mae_r=0.5,
        )
        assert event.mfe_price == 73000.0
        assert event.mae_price == 69500.0

    def test_defaults_to_none(self):
        event = TradeEvent(
            trade_id="test_002",
            event_metadata={"event_id": "e2"},
            entry_snapshot={},
        )
        assert event.mfe_price is None
        assert event.mae_price is None
        assert event.mfe_pct is None
        assert event.exit_efficiency is None

    def test_serializes(self):
        event = TradeEvent(
            trade_id="test_003",
            event_metadata={"event_id": "e3"},
            entry_snapshot={},
            mfe_price=73000.0,
        )
        d = event.to_dict()
        assert "mfe_price" in d
        assert d["mfe_price"] == 73000.0


class TestFacadeSignature:
    def test_on_exit_fill_accepts_mfe_mae_context(self):
        import inspect
        from instrumentation.facade import InstrumentationKit
        sig = inspect.signature(InstrumentationKit.on_exit_fill)
        assert "mfe_mae_context" in sig.parameters
