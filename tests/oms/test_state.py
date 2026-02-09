"""Tests for OMS state module."""

import pytest
from datetime import datetime
import time
import threading

from oms.state import (
    StateStore,
    SymbolPosition,
    StrategyAllocation,
    WorkingOrder,
    OrderStatus,
)


class TestWorkingOrder:
    """Tests for WorkingOrder dataclass."""

    def test_default_values(self):
        """Test default values for WorkingOrder."""
        wo = WorkingOrder(
            order_id="ORD001",
            symbol="005930",
            side="BUY",
            qty=100,
        )
        assert wo.order_id == "ORD001"
        assert wo.symbol == "005930"
        assert wo.side == "BUY"
        assert wo.qty == 100
        assert wo.filled_qty == 0
        assert wo.price == 0.0
        assert wo.order_type == "LIMIT"
        assert wo.status == OrderStatus.PENDING
        assert wo.strategy_id == ""

    def test_with_all_fields(self):
        """Test WorkingOrder with all fields."""
        wo = WorkingOrder(
            order_id="ORD001",
            symbol="005930",
            side="SELL",
            qty=100,
            filled_qty=50,
            price=72000,
            order_type="MARKET",
            status=OrderStatus.PARTIAL,
            strategy_id="KMP",
            cancel_after_sec=30.0,
        )
        assert wo.filled_qty == 50
        assert wo.price == 72000
        assert wo.order_type == "MARKET"
        assert wo.status == OrderStatus.PARTIAL
        assert wo.strategy_id == "KMP"
        assert wo.cancel_after_sec == 30.0


class TestStrategyAllocation:
    """Tests for StrategyAllocation dataclass."""

    def test_default_values(self):
        """Test default values."""
        alloc = StrategyAllocation(strategy_id="KMP")
        assert alloc.strategy_id == "KMP"
        assert alloc.qty == 0
        assert alloc.cost_basis == 0.0
        assert alloc.entry_ts is None
        assert alloc.soft_stop_px is None
        assert alloc.time_stop_ts is None

    def test_with_values(self):
        """Test with all values."""
        now = datetime.now()
        alloc = StrategyAllocation(
            strategy_id="KPR",
            qty=100,
            cost_basis=70000,
            entry_ts=now,
            soft_stop_px=69000,
            time_stop_ts=time.time() + 3600,
        )
        assert alloc.qty == 100
        assert alloc.cost_basis == 70000
        assert alloc.entry_ts == now
        assert alloc.soft_stop_px == 69000


class TestSymbolPosition:
    """Tests for SymbolPosition dataclass."""

    def test_default_values(self):
        """Test default values."""
        pos = SymbolPosition(symbol="005930")
        assert pos.symbol == "005930"
        assert pos.real_qty == 0
        assert pos.avg_price == 0.0
        assert pos.allocations == {}
        assert pos.hard_stop_px is None
        assert pos.entry_lock_owner is None
        assert pos.entry_lock_until is None
        assert pos.cooldown_until is None
        assert pos.vi_cooldown_until is None
        assert pos.working_orders == []
        assert pos.frozen is False

    def test_has_working_orders(self):
        """Test has_working_orders method."""
        pos = SymbolPosition(symbol="005930")
        assert pos.has_working_orders() is False

        wo = WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100)
        pos.working_orders.append(wo)
        assert pos.has_working_orders() is True

    def test_working_qty_all(self):
        """Test working_qty without filters."""
        pos = SymbolPosition(symbol="005930")
        pos.working_orders = [
            WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100, filled_qty=20),
            WorkingOrder(order_id="ORD002", symbol="005930", side="SELL", qty=50, filled_qty=0),
        ]
        # (100-20) + (50-0) = 80 + 50 = 130
        assert pos.working_qty() == 130

    def test_working_qty_by_side(self):
        """Test working_qty filtered by side."""
        pos = SymbolPosition(symbol="005930")
        pos.working_orders = [
            WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100, filled_qty=20),
            WorkingOrder(order_id="ORD002", symbol="005930", side="SELL", qty=50, filled_qty=0),
        ]
        assert pos.working_qty(side="BUY") == 80
        assert pos.working_qty(side="SELL") == 50

    def test_working_qty_by_strategy(self):
        """Test working_qty filtered by strategy."""
        pos = SymbolPosition(symbol="005930")
        pos.working_orders = [
            WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100, strategy_id="KMP"),
            WorkingOrder(order_id="ORD002", symbol="005930", side="BUY", qty=50, strategy_id="KPR"),
        ]
        assert pos.working_qty(strategy_id="KMP") == 100
        assert pos.working_qty(strategy_id="KPR") == 50
        assert pos.working_qty(strategy_id="PCIM") == 0

    def test_total_allocated(self):
        """Test total_allocated calculation."""
        pos = SymbolPosition(symbol="005930")
        pos.allocations = {
            "KMP": StrategyAllocation(strategy_id="KMP", qty=100),
            "KPR": StrategyAllocation(strategy_id="KPR", qty=50),
        }
        assert pos.total_allocated() == 150

    def test_allocation_drift(self):
        """Test allocation_drift calculation."""
        pos = SymbolPosition(symbol="005930", real_qty=150)
        pos.allocations = {
            "KMP": StrategyAllocation(strategy_id="KMP", qty=100),
            "KPR": StrategyAllocation(strategy_id="KPR", qty=40),
        }
        # real=150, allocated=140, drift=10
        assert pos.allocation_drift() == 10

    def test_allocation_drift_negative(self):
        """Test negative allocation drift."""
        pos = SymbolPosition(symbol="005930", real_qty=100)
        pos.allocations = {
            "KMP": StrategyAllocation(strategy_id="KMP", qty=120),
        }
        # real=100, allocated=120, drift=-20
        assert pos.allocation_drift() == -20

    def test_get_allocation(self):
        """Test get_allocation method."""
        pos = SymbolPosition(symbol="005930")
        pos.allocations["KMP"] = StrategyAllocation(strategy_id="KMP", qty=100)

        assert pos.get_allocation("KMP") == 100
        assert pos.get_allocation("KPR") == 0

    def test_is_entry_locked(self):
        """Test is_entry_locked method."""
        pos = SymbolPosition(symbol="005930")
        now = time.time()

        # No lock
        assert pos.is_entry_locked(now) is False

        # Lock expired
        pos.entry_lock_until = now - 10
        assert pos.is_entry_locked(now) is False

        # Lock active
        pos.entry_lock_until = now + 60
        assert pos.is_entry_locked(now) is True

    def test_can_strategy_enter(self):
        """Test can_strategy_enter method."""
        pos = SymbolPosition(symbol="005930")
        now = time.time()

        # No lock - anyone can enter
        assert pos.can_strategy_enter("KMP", now) is True
        assert pos.can_strategy_enter("KPR", now) is True

        # Lock held by KMP
        pos.entry_lock_owner = "KMP"
        pos.entry_lock_until = now + 60
        assert pos.can_strategy_enter("KMP", now) is True
        assert pos.can_strategy_enter("KPR", now) is False


class TestStateStore:
    """Tests for StateStore."""

    def test_get_position_creates_new(self):
        """Test get_position creates new position if not exists."""
        store = StateStore()
        pos = store.get_position("005930")

        assert pos.symbol == "005930"
        assert pos.real_qty == 0

    def test_get_position_returns_same(self):
        """Test get_position returns same instance."""
        store = StateStore()
        pos1 = store.get_position("005930")
        pos1.real_qty = 100

        pos2 = store.get_position("005930")
        assert pos2.real_qty == 100
        assert pos1 is pos2

    def test_get_all_positions(self):
        """Test get_all_positions returns copy."""
        store = StateStore()
        store.get_position("005930")
        store.get_position("000660")

        all_pos = store.get_all_positions()
        assert len(all_pos) == 2
        assert "005930" in all_pos
        assert "000660" in all_pos

    def test_update_position(self):
        """Test update_position method."""
        store = StateStore()
        store.update_position("005930", real_qty=100, avg_price=70000)

        pos = store.get_position("005930")
        assert pos.real_qty == 100
        assert pos.avg_price == 70000

    def test_update_allocation_new(self):
        """Test update_allocation creates new allocation."""
        store = StateStore()
        store.update_allocation("005930", "KMP", qty_delta=100, cost_basis=70000)

        pos = store.get_position("005930")
        alloc = pos.allocations.get("KMP")
        assert alloc is not None
        assert alloc.qty == 100
        assert alloc.cost_basis == 70000
        assert alloc.entry_ts is not None

    def test_update_allocation_add(self):
        """Test update_allocation adds to existing."""
        store = StateStore()
        store.update_allocation("005930", "KMP", qty_delta=100, cost_basis=70000)
        store.update_allocation("005930", "KMP", qty_delta=50)

        pos = store.get_position("005930")
        assert pos.allocations["KMP"].qty == 150

    def test_update_allocation_reduce_to_zero(self):
        """Test update_allocation clears entry_ts when qty becomes zero."""
        store = StateStore()
        store.update_allocation("005930", "KMP", qty_delta=100)
        store.update_allocation("005930", "KMP", qty_delta=-100)

        pos = store.get_position("005930")
        alloc = pos.allocations["KMP"]
        assert alloc.qty == 0
        assert alloc.entry_ts is None

    def test_set_entry_lock_success(self):
        """Test acquiring entry lock."""
        store = StateStore()
        now = time.time()

        result = store.set_entry_lock("005930", "KMP", now + 60)
        assert result is True

        pos = store.get_position("005930")
        assert pos.entry_lock_owner == "KMP"
        assert pos.entry_lock_until == now + 60

    def test_set_entry_lock_already_held(self):
        """Test acquiring lock when held by another strategy."""
        store = StateStore()
        now = time.time()

        store.set_entry_lock("005930", "KMP", now + 60)
        result = store.set_entry_lock("005930", "KPR", now + 60)

        assert result is False

    def test_set_entry_lock_same_owner(self):
        """Test re-acquiring lock by same owner."""
        store = StateStore()
        now = time.time()

        store.set_entry_lock("005930", "KMP", now + 60)
        result = store.set_entry_lock("005930", "KMP", now + 120)

        assert result is True
        pos = store.get_position("005930")
        assert pos.entry_lock_until == now + 120

    def test_release_entry_lock(self):
        """Test releasing entry lock."""
        store = StateStore()
        now = time.time()

        store.set_entry_lock("005930", "KMP", now + 60)
        store.release_entry_lock("005930", "KMP")

        pos = store.get_position("005930")
        assert pos.entry_lock_owner is None
        assert pos.entry_lock_until is None

    def test_release_entry_lock_wrong_owner(self):
        """Test releasing lock by wrong owner does nothing."""
        store = StateStore()
        now = time.time()

        store.set_entry_lock("005930", "KMP", now + 60)
        store.release_entry_lock("005930", "KPR")

        pos = store.get_position("005930")
        assert pos.entry_lock_owner == "KMP"

    def test_add_working_order(self):
        """Test adding working order."""
        store = StateStore()
        wo = WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100)

        store.add_working_order("005930", wo)

        pos = store.get_position("005930")
        assert len(pos.working_orders) == 1
        assert pos.working_orders[0].order_id == "ORD001"

    def test_remove_working_order(self):
        """Test removing working order."""
        store = StateStore()
        wo = WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100)
        store.add_working_order("005930", wo)

        store.remove_working_order("005930", "ORD001")

        pos = store.get_position("005930")
        assert len(pos.working_orders) == 0

    def test_get_working_orders_all(self):
        """Test getting all working orders."""
        store = StateStore()
        wo1 = WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100)
        wo2 = WorkingOrder(order_id="ORD002", symbol="000660", side="BUY", qty=50)

        store.add_working_order("005930", wo1)
        store.add_working_order("000660", wo2)

        orders = store.get_working_orders()
        assert len(orders) == 2

    def test_get_working_orders_by_symbol(self):
        """Test getting working orders by symbol."""
        store = StateStore()
        wo1 = WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100)
        wo2 = WorkingOrder(order_id="ORD002", symbol="000660", side="BUY", qty=50)

        store.add_working_order("005930", wo1)
        store.add_working_order("000660", wo2)

        orders = store.get_working_orders("005930")
        assert len(orders) == 1
        assert orders[0].order_id == "ORD001"

    def test_get_allocations_for_strategy(self):
        """Test getting all allocations for a strategy."""
        store = StateStore()
        store.update_allocation("005930", "KMP", 100)
        store.update_allocation("000660", "KMP", 50)
        store.update_allocation("035420", "KPR", 75)

        kmp_allocs = store.get_allocations_for_strategy("KMP")
        assert len(kmp_allocs) == 2
        assert kmp_allocs["005930"].qty == 100
        assert kmp_allocs["000660"].qty == 50

    def test_update_daily_pnl(self):
        """Test updating daily P&L."""
        store = StateStore()
        store.equity = 100_000_000
        store.update_position("005930", real_qty=100, avg_price=70000)

        prices = {"005930": 72000}
        store.update_daily_pnl(prices)

        # PnL = (72000 - 70000) * 100 = 200,000
        assert store.daily_pnl == 200_000
        assert store.daily_pnl_pct == pytest.approx(0.002, abs=0.0001)

    def test_thread_safety(self):
        """Test basic thread safety of StateStore."""
        store = StateStore()
        results = []

        def update_position(symbol, qty):
            for _ in range(100):
                store.update_position(symbol, real_qty=qty)
            results.append(True)

        threads = [
            threading.Thread(target=update_position, args=("005930", 100)),
            threading.Thread(target=update_position, args=("005930", 200)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 2
        pos = store.get_position("005930")
        assert pos.real_qty in (100, 200)


class TestStateStoreRealizedPnl:
    """Tests for record_realized_pnl and its integration with daily_pnl."""

    def test_record_realized_pnl(self):
        """Test that record_realized_pnl accumulates realized P&L."""
        store = StateStore()
        store.record_realized_pnl(100000)
        store.record_realized_pnl(50000)
        assert store.daily_realized_pnl == 150000

    def test_realized_pnl_in_daily_pnl(self):
        """Test that realized P&L is included in total daily P&L calculation."""
        store = StateStore()
        store.equity = 100_000_000
        store.record_realized_pnl(200_000)
        store.update_position("005930", real_qty=100, avg_price=70000)
        store.update_daily_pnl({"005930": 72000})
        # unrealized = (72000-70000)*100 = 200000, realized = 200000, total = 400000
        assert store.daily_pnl == 400_000


class TestStateStoreAllocationWeightedAvg:
    """Tests for update_allocation weighted-average cost basis."""

    def test_update_allocation_weighted_avg_cost(self):
        """Test weighted average cost basis on multiple buys."""
        store = StateStore()
        store.update_allocation("005930", "KMP", qty_delta=100, cost_basis=70000)
        store.update_allocation("005930", "KMP", qty_delta=50, cost_basis=73000)
        alloc = store.get_position("005930").allocations["KMP"]
        assert alloc.qty == 150
        # Weighted avg: (70000*100 + 73000*50) / 150 = 71000
        assert alloc.cost_basis == pytest.approx(71000, abs=1)


class TestStateStoreDailyPnlMissingPrice:
    """Tests for update_daily_pnl when price is missing for a symbol."""

    def test_daily_pnl_missing_price_uses_avg(self):
        """Test that missing symbol price falls back to avg_price, giving 0 unrealized."""
        store = StateStore()
        store.equity = 100_000_000
        store.update_position("005930", real_qty=100, avg_price=70000)
        store.update_daily_pnl({})  # No prices provided
        assert store.daily_pnl == 0  # (70000 - 70000)*100 = 0


class TestSymbolPositionWorkingQtyCombined:
    """Tests for working_qty with combined strategy_id AND side filters."""

    def test_working_qty_combined_filters(self):
        """Test working_qty filtered by both strategy_id and side."""
        pos = SymbolPosition(symbol="005930")
        pos.working_orders = [
            WorkingOrder(order_id="ORD001", symbol="005930", side="BUY", qty=100, strategy_id="KMP"),
            WorkingOrder(order_id="ORD002", symbol="005930", side="SELL", qty=50, strategy_id="KMP"),
            WorkingOrder(order_id="ORD003", symbol="005930", side="BUY", qty=75, strategy_id="KPR"),
        ]
        assert pos.working_qty(strategy_id="KMP", side="BUY") == 100
        assert pos.working_qty(strategy_id="KMP", side="SELL") == 50
        assert pos.working_qty(strategy_id="KPR", side="BUY") == 75
        assert pos.working_qty(strategy_id="KPR", side="SELL") == 0
