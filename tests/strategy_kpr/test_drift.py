import pytest
from unittest.mock import AsyncMock, MagicMock

from strategy_kpr.core.drift import DriftMonitor, DriftEvent
from strategy_kpr.core.state import FSMState, SymbolState
from strategy_kpr.main import _run_drift_check


class _FakeOMSPosition:
    def __init__(self, qty: int):
        self.qty = qty

    def get_allocation(self, strategy_id: str) -> int:
        return self.qty


class TestDriftMonitor:
    def test_initial_state(self):
        dm = DriftMonitor()
        assert dm.global_trade_block is False
        assert dm.reconcile_needed is False
        assert dm.last_drift_events == []

    def test_no_drift(self):
        dm = DriftMonitor()
        events = dm.compute_drift({"005930": 100}, {"005930": 100})
        assert events == []

    def test_position_mismatch(self):
        dm = DriftMonitor()
        events = dm.compute_drift({"005930": 100}, {"005930": 150})
        assert len(events) == 1
        assert events[0].drift_type == "POSITION_MISMATCH"
        assert events[0].local_qty == 100
        assert events[0].broker_qty == 150

    def test_missing_broker(self):
        dm = DriftMonitor()
        events = dm.compute_drift({"005930": 100}, {})
        assert len(events) == 1
        assert events[0].drift_type == "MISSING_BROKER"

    def test_local_zero_qty_no_missing_broker(self):
        dm = DriftMonitor()
        events = dm.compute_drift({"005930": 0}, {})
        assert len(events) == 0  # 0 qty is not considered a position

    def test_order_orphan_local(self):
        dm = DriftMonitor()
        events = dm.compute_drift({}, {}, local_orders={"ORD1"}, broker_orders=set())
        assert len(events) == 1
        assert events[0].drift_type == "ORDER_ORPHAN_LOCAL"

    def test_order_orphan_broker(self):
        dm = DriftMonitor()
        events = dm.compute_drift({}, {}, local_orders=set(), broker_orders={"ORD2"})
        assert len(events) == 1
        assert events[0].drift_type == "ORDER_ORPHAN_BROKER"

    def test_handle_drift_activates_block(self):
        dm = DriftMonitor()
        events = [DriftEvent("POSITION_MISMATCH", "005930", 100, 150)]
        result = dm.handle_drift(events)
        assert result is True
        assert dm.global_trade_block is True
        assert dm.reconcile_needed is True

    def test_handle_empty_drift_no_block(self):
        dm = DriftMonitor()
        result = dm.handle_drift([])
        assert result is False
        assert dm.global_trade_block is False

    def test_clear_after_reconcile(self):
        dm = DriftMonitor()
        dm.global_trade_block = True
        dm.reconcile_needed = True
        dm.last_drift_events = [DriftEvent("TEST", "X")]
        dm.clear_after_reconcile()
        assert dm.global_trade_block is False
        assert dm.reconcile_needed is False
        assert dm.last_drift_events == []

    def test_block_on_oms_unavailable(self):
        dm = DriftMonitor()
        dm.block_on_oms_unavailable()
        assert dm.global_trade_block is True
        assert dm.reconcile_needed is False  # OMS block does NOT set reconcile

    def test_clear_oms_block_after_recovery(self):
        dm = DriftMonitor()
        dm.block_on_oms_unavailable()
        dm.clear_oms_block()
        assert dm.global_trade_block is False

    def test_clear_oms_block_preserves_drift_block(self):
        dm = DriftMonitor()
        events = [DriftEvent("POSITION_MISMATCH", "005930", 100, 150)]
        dm.handle_drift(events)  # sets both global_trade_block AND reconcile_needed
        dm.clear_oms_block()
        assert dm.global_trade_block is True  # NOT cleared — real drift

    def test_clear_oms_block_noop_when_not_blocked(self):
        dm = DriftMonitor()
        dm.clear_oms_block()  # should not raise
        assert dm.global_trade_block is False


class TestDriftOrchestration:
    @pytest.mark.asyncio
    async def test_oms_recovery_clears_unavailable_block(self):
        oms = MagicMock()
        oms.get_all_positions = AsyncMock(side_effect=[None, {}])
        drift_monitor = DriftMonitor()
        states = {"005930": SymbolState(code="005930")}
        positions = set()
        sector_exposure = MagicMock()

        should_skip = await _run_drift_check(
            oms, drift_monitor, states, positions, sector_exposure
        )
        assert should_skip is True
        assert drift_monitor.global_trade_block is True
        assert drift_monitor.reconcile_needed is False

        should_skip = await _run_drift_check(
            oms, drift_monitor, states, positions, sector_exposure
        )
        assert should_skip is False
        assert drift_monitor.global_trade_block is False
        assert drift_monitor.reconcile_needed is False

    @pytest.mark.asyncio
    async def test_real_drift_block_is_not_cleared_by_recovery_path(self):
        oms = MagicMock()
        oms.get_all_positions = AsyncMock(return_value={
            "005930": _FakeOMSPosition(100),
        })
        drift_monitor = DriftMonitor()
        drift_monitor.handle_drift([
            DriftEvent("POSITION_MISMATCH", "005930", 80, 100)
        ])
        state = SymbolState(code="005930", fsm=FSMState.IN_POSITION, qty=100)
        states = {"005930": state}
        positions = {"005930"}
        sector_exposure = MagicMock()

        should_skip = await _run_drift_check(
            oms, drift_monitor, states, positions, sector_exposure
        )

        assert should_skip is False
        assert drift_monitor.global_trade_block is True
        assert drift_monitor.reconcile_needed is True
