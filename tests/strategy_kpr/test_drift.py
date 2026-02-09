import pytest
from strategy_kpr.core.drift import DriftMonitor, DriftEvent

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
