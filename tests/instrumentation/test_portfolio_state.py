"""Tests for portfolio state at entry."""
from instrumentation.src.trade_logger import TradeEvent


class TestTradeEventPortfolioState:
    def test_field_exists(self):
        ps = {
            "total_exposure_pct": 45.2,
            "num_positions": 3,
            "concurrent_positions_same_strategy": 2,
        }
        event = TradeEvent(
            trade_id="ps_001",
            event_metadata={"event_id": "e1"},
            entry_snapshot={},
            portfolio_state_at_entry=ps,
        )
        assert event.portfolio_state_at_entry["num_positions"] == 3
        assert event.portfolio_state_at_entry["total_exposure_pct"] == 45.2

    def test_defaults_none(self):
        event = TradeEvent(
            trade_id="ps_002",
            event_metadata={"event_id": "e2"},
            entry_snapshot={},
        )
        assert event.portfolio_state_at_entry is None

    def test_serializes(self):
        event = TradeEvent(
            trade_id="ps_003",
            event_metadata={"event_id": "e3"},
            entry_snapshot={},
            portfolio_state_at_entry={"num_positions": 2},
        )
        d = event.to_dict()
        assert d["portfolio_state_at_entry"]["num_positions"] == 2


class TestFacadePortfolioParam:
    def test_on_entry_fill_accepts_portfolio_state(self):
        import inspect
        from instrumentation.facade import InstrumentationKit
        sig = inspect.signature(InstrumentationKit.on_entry_fill)
        assert "portfolio_state" in sig.parameters
