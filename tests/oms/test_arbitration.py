"""Tests for OMS arbitration module."""

import pytest
import time

from oms.arbitration import ArbitrationEngine, ArbitrationResult, ArbitrationDecision
from oms.state import StateStore, StrategyAllocation
from oms.intent import Intent, IntentType, Urgency


class TestArbitrationResult:
    """Tests for ArbitrationResult enum."""

    def test_all_results_defined(self):
        """Test all arbitration results are defined."""
        assert ArbitrationResult.PROCEED
        assert ArbitrationResult.DEFER
        assert ArbitrationResult.MERGE
        assert ArbitrationResult.CANCEL


class TestArbitrationDecision:
    """Tests for ArbitrationDecision dataclass."""

    def test_default_values(self):
        """Test default decision values."""
        decision = ArbitrationDecision(result=ArbitrationResult.PROCEED)

        assert decision.result == ArbitrationResult.PROCEED
        assert decision.reason == ""
        assert decision.merged_qty is None
        assert decision.defer_until is None

    def test_with_all_fields(self):
        """Test decision with all fields."""
        defer_until = time.time() + 60
        decision = ArbitrationDecision(
            result=ArbitrationResult.DEFER,
            reason="Entry locked by KMP",
            defer_until=defer_until,
        )

        assert decision.result == ArbitrationResult.DEFER
        assert decision.reason == "Entry locked by KMP"
        assert decision.defer_until == defer_until


class TestArbitrationEngineLockDurations:
    """Tests for lock duration configuration."""

    def test_lock_durations_defined(self):
        """Test lock durations are defined for strategies."""
        assert ArbitrationEngine.LOCK_DURATIONS["KMP"] == 90
        assert ArbitrationEngine.LOCK_DURATIONS["KPR"] == 180
        assert ArbitrationEngine.LOCK_DURATIONS["PCIM"] == 300
        assert ArbitrationEngine.LOCK_DURATIONS["NULRIMOK"] == 60


class TestArbitrationEngineExits:
    """Tests for exit intent arbitration."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    def test_exit_always_proceeds(self, engine):
        """Test EXIT intent always proceeds."""
        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        decision = engine.arbitrate(intent)

        assert decision.result == ArbitrationResult.PROCEED

    def test_flatten_always_proceeds(self, engine):
        """Test FLATTEN intent always proceeds."""
        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id="KMP",
            symbol="005930",
        )

        decision = engine.arbitrate(intent)

        assert decision.result == ArbitrationResult.PROCEED

    def test_exit_proceeds_despite_lock(self, engine, state_store):
        """Test EXIT proceeds even when entry is locked."""
        # Set up entry lock by another strategy
        now = time.time()
        state_store.set_entry_lock("005930", "KPR", now + 60)

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        decision = engine.arbitrate(intent)

        assert decision.result == ArbitrationResult.PROCEED


class TestArbitrationEngineReductions:
    """Tests for reduction intent arbitration."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    def test_reduce_proceeds(self, engine):
        """Test REDUCE intent proceeds."""
        intent = Intent(
            intent_type=IntentType.REDUCE,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=50,
        )

        decision = engine.arbitrate(intent)

        assert decision.result == ArbitrationResult.PROCEED


class TestArbitrationEngineEntry:
    """Tests for entry intent arbitration."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    @pytest.fixture
    def enter_intent(self):
        """Create sample ENTER intent."""
        return Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

    def test_entry_acquires_lock(self, engine, state_store, enter_intent):
        """Test entry intent acquires lock."""
        decision = engine.arbitrate(enter_intent)

        assert decision.result == ArbitrationResult.PROCEED

        pos = state_store.get_position("005930")
        assert pos.entry_lock_owner == "KMP"
        assert pos.entry_lock_until is not None

    def test_entry_lock_duration(self, engine, state_store, enter_intent):
        """Test entry lock uses strategy-specific duration."""
        now = time.time()
        decision = engine.arbitrate(enter_intent)

        assert decision.result == ArbitrationResult.PROCEED

        pos = state_store.get_position("005930")
        # KMP lock duration is 90 seconds
        expected_until = now + 90
        assert abs(pos.entry_lock_until - expected_until) < 1

    def test_entry_blocked_by_another_strategy(self, engine, state_store):
        """Test entry is deferred when locked by another strategy."""
        now = time.time()
        state_store.set_entry_lock("005930", "KPR", now + 60)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        decision = engine.arbitrate(intent)

        assert decision.result == ArbitrationResult.DEFER
        assert "locked" in decision.reason.lower()
        assert "KPR" in decision.reason

    def test_entry_allowed_by_same_strategy(self, engine, state_store):
        """Test entry is allowed when already locked by same strategy."""
        now = time.time()
        state_store.set_entry_lock("005930", "KMP", now + 60)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        decision = engine.arbitrate(intent)

        assert decision.result == ArbitrationResult.PROCEED

    def test_entry_cancelled_if_already_holds(self, engine, state_store):
        """Test entry is cancelled if strategy already holds position."""
        state_store.update_allocation("005930", "KMP", 100)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        decision = engine.arbitrate(intent)

        assert decision.result == ArbitrationResult.CANCEL
        assert "already holds" in decision.reason.lower()

    def test_entry_deferred_if_exit_pending(self, engine, state_store):
        """Test entry is deferred if exit intent is pending."""
        # Add pending exit intent
        exit_intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KPR",
            symbol="005930",
        )
        engine.add_pending(exit_intent)

        enter_intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        decision = engine.arbitrate(enter_intent)

        assert decision.result == ArbitrationResult.DEFER
        assert "exit" in decision.reason.lower()

    def test_unknown_strategy_gets_default_lock(self, engine, state_store):
        """Test unknown strategy gets default lock duration."""
        now = time.time()

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="UNKNOWN",
            symbol="005930",
            desired_qty=100,
        )

        decision = engine.arbitrate(intent)

        assert decision.result == ArbitrationResult.PROCEED

        pos = state_store.get_position("005930")
        # Default lock duration is 60 seconds
        expected_until = now + 60
        assert abs(pos.entry_lock_until - expected_until) < 1


class TestArbitrationEnginePending:
    """Tests for pending intent management."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    def test_add_pending(self, engine):
        """Test adding pending intent."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        engine.add_pending(intent)

        assert "005930" in engine._pending_intents
        assert len(engine._pending_intents["005930"]) == 1

    def test_remove_pending(self, engine):
        """Test removing pending intent."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        engine.add_pending(intent)
        engine.remove_pending(intent)

        assert engine._pending_intents["005930"] == []

    def test_remove_pending_by_intent_id(self, engine):
        """Test removing pending intent by ID."""
        intent1 = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )
        intent2 = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KPR",
            symbol="005930",
            desired_qty=50,
        )

        engine.add_pending(intent1)
        engine.add_pending(intent2)
        engine.remove_pending(intent1)

        assert len(engine._pending_intents["005930"]) == 1
        assert engine._pending_intents["005930"][0].strategy_id == "KPR"


class TestArbitrationEngineNetTarget:
    """Tests for net target computation."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    def test_compute_net_target_empty(self, engine):
        """Test net target with no allocations."""
        net, allocs = engine.compute_net_target("005930")

        assert net == 0
        assert allocs == {}

    def test_compute_net_target_single(self, engine, state_store):
        """Test net target with single allocation."""
        state_store.update_allocation("005930", "KMP", 100)

        net, allocs = engine.compute_net_target("005930")

        assert net == 100
        assert allocs == {"KMP": 100}

    def test_compute_net_target_multiple(self, engine, state_store):
        """Test net target with multiple allocations."""
        state_store.update_allocation("005930", "KMP", 100)
        state_store.update_allocation("005930", "KPR", 50)

        net, allocs = engine.compute_net_target("005930")

        assert net == 150
        assert allocs == {"KMP": 100, "KPR": 50}


class TestArbitrationEngineTradeQty:
    """Tests for trade quantity computation."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    def test_compute_trade_qty_from_zero(self, engine):
        """Test trade qty from zero position."""
        qty = engine.compute_trade_qty("005930", desired_total=100)
        assert qty == 100

    def test_compute_trade_qty_add(self, engine, state_store):
        """Test trade qty to add to existing."""
        state_store.update_position("005930", real_qty=50)

        qty = engine.compute_trade_qty("005930", desired_total=100)
        assert qty == 50

    def test_compute_trade_qty_reduce(self, engine, state_store):
        """Test trade qty to reduce existing."""
        state_store.update_position("005930", real_qty=100)

        qty = engine.compute_trade_qty("005930", desired_total=50)
        assert qty == -50

    def test_compute_trade_qty_no_change(self, engine, state_store):
        """Test trade qty when already at target."""
        state_store.update_position("005930", real_qty=100)

        qty = engine.compute_trade_qty("005930", desired_total=100)
        assert qty == 0


class TestArbitrationEngineOperational:
    """Tests for operational intent types falling through to PROCEED."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    def test_set_target_proceeds(self, engine):
        """Test SET_TARGET intent proceeds through arbitration."""
        intent = Intent(
            intent_type=IntentType.SET_TARGET,
            strategy_id="KMP",
            symbol="005930",
            target_qty=100,
        )
        decision = engine.arbitrate(intent)
        assert decision.result == ArbitrationResult.PROCEED

    def test_cancel_orders_proceeds(self, engine):
        """Test CANCEL_ORDERS intent proceeds through arbitration."""
        intent = Intent(
            intent_type=IntentType.CANCEL_ORDERS,
            strategy_id="KMP",
            symbol="005930",
        )
        decision = engine.arbitrate(intent)
        assert decision.result == ArbitrationResult.PROCEED

    def test_modify_risk_proceeds(self, engine):
        """Test MODIFY_RISK intent proceeds through arbitration."""
        intent = Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id="KMP",
            symbol="005930",
        )
        decision = engine.arbitrate(intent)
        assert decision.result == ArbitrationResult.PROCEED


class TestArbitrationEngineRemovePendingNonexistent:
    """Tests for remove_pending with non-existent symbols."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    def test_remove_pending_nonexistent(self, engine):
        """Test remove_pending for a symbol not in pending dict does not raise."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="999999",
            desired_qty=100,
        )
        engine.remove_pending(intent)  # Should not raise


class TestArbitrationEngineNetTargetZeroQty:
    """Tests for compute_net_target with zero-qty allocations."""

    @pytest.fixture
    def engine(self, state_store):
        """Create ArbitrationEngine for testing."""
        return ArbitrationEngine(state_store)

    def test_compute_net_target_zero_qty_alloc(self, engine, state_store):
        """Test net target when allocation is added then fully removed."""
        state_store.update_allocation("005930", "KMP", 100)
        state_store.update_allocation("005930", "KMP", -100)
        net, allocs = engine.compute_net_target("005930")
        assert net == 0
        assert allocs["KMP"] == 0
