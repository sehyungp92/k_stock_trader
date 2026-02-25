"""Tests for OMS core module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import time

from oms.oms_core import OMSCore, InMemoryIdempotencyStore, IdempotencyStore
from oms.state import StateStore, StrategyAllocation
from oms.risk import RiskConfig
from oms.intent import Intent, IntentType, IntentStatus, IntentResult, Urgency, RiskPayload
from oms.adapter import BrokerQueryResult


class TestInMemoryIdempotencyStore:
    """Tests for InMemoryIdempotencyStore."""

    def test_get_missing_key(self):
        """Test getting missing key returns None."""
        store = InMemoryIdempotencyStore()
        result = store.get("missing_key")
        assert result is None

    def test_put_and_get(self):
        """Test putting and getting a result."""
        store = InMemoryIdempotencyStore()
        result = IntentResult(
            intent_id="test-id",
            status=IntentStatus.EXECUTED,
        )

        store.put("key1", result)
        retrieved = store.get("key1")

        assert retrieved is not None
        assert retrieved.intent_id == "test-id"
        assert retrieved.status == IntentStatus.EXECUTED

    def test_overwrite_key(self):
        """Test overwriting an existing key."""
        store = InMemoryIdempotencyStore()
        result1 = IntentResult(intent_id="id1", status=IntentStatus.PENDING)
        result2 = IntentResult(intent_id="id2", status=IntentStatus.EXECUTED)

        store.put("key1", result1)
        store.put("key1", result2)

        retrieved = store.get("key1")
        assert retrieved.intent_id == "id2"


class TestOMSCoreInit:
    """Tests for OMSCore initialization."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    def test_init_with_defaults(self, mock_api):
        """Test initialization with defaults."""
        oms = OMSCore(mock_api)

        assert oms.state is not None
        assert oms.risk is not None
        assert oms.arbitration is not None
        assert oms.planner is not None
        assert oms.adapter is not None
        assert oms._idem is not None

    def test_init_with_custom_config(self, mock_api):
        """Test initialization with custom config."""
        config = RiskConfig(max_positions_count=5)
        oms = OMSCore(mock_api, risk_config=config)

        assert oms.risk.config.max_positions_count == 5

    def test_init_with_custom_idempotency_store(self, mock_api):
        """Test initialization with custom idempotency store."""
        custom_store = InMemoryIdempotencyStore()
        oms = OMSCore(mock_api, idempotency_store=custom_store)

        assert oms._idem is custom_store


class TestOMSCoreSubmitIntent:
    """Tests for OMSCore.submit_intent method."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        return oms

    @pytest.mark.asyncio
    async def test_idempotency_returns_cached(self, oms):
        """Test idempotent intent returns cached result."""
        cached_result = IntentResult(
            intent_id="cached-id",
            status=IntentStatus.EXECUTED,
            order_id="ORD001",
        )
        oms._idem.put("KMP:005930:ENTER:20240115:test", cached_result)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            idempotency_key="KMP:005930:ENTER:20240115:test",
        )

        result = await oms.submit_intent(intent)

        assert result.intent_id == "cached-id"
        assert result.status == IntentStatus.EXECUTED

    @pytest.mark.asyncio
    async def test_validation_failure(self, oms):
        """Test validation failure rejects intent."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="",  # Invalid: empty symbol
            desired_qty=100,
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.REJECTED
        assert "validation" in result.message.lower()

    @pytest.mark.asyncio
    async def test_expired_intent_rejected(self, oms):
        """Test expired intent is rejected."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )
        intent.constraints.expiry_ts = time.time() - 10

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.REJECTED
        assert "expired" in result.message.lower()


class TestOMSCoreProcessIntent:
    """Tests for OMSCore intent processing."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        return oms

    @pytest.mark.asyncio
    async def test_risk_rejection(self, oms):
        """Test risk check rejection."""
        oms.risk.safe_mode = True

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.DEFERRED
        assert "safe mode" in result.message.lower()

    @pytest.mark.asyncio
    async def test_arbitration_defer(self, oms):
        """Test arbitration deferral."""
        # Set up entry lock by another strategy
        now = time.time()
        oms.state.set_entry_lock("005930", "KPR", now + 60)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.DEFERRED
        assert "locked" in result.message.lower()

    @pytest.mark.asyncio
    async def test_cancel_orders_intent(self, oms):
        """Test CANCEL_ORDERS intent processing."""
        intent = Intent(
            intent_type=IntentType.CANCEL_ORDERS,
            strategy_id="KMP",
            symbol="005930",
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        assert "cancelled" in result.message.lower()


class TestOMSCorePlanAndExecute:
    """Tests for OMSCore plan and execute flow."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        return oms

    @pytest.mark.asyncio
    async def test_enter_creates_working_order(self, oms):
        """Test ENTER intent creates working order."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        assert result.order_id is not None

        # Check working order was created
        pos = oms.state.get_position("005930")
        assert pos.has_working_orders()

    @pytest.mark.asyncio
    async def test_exit_without_allocation_rejected(self, oms):
        """Test EXIT without allocation is rejected."""
        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.REJECTED
        assert "no allocation" in result.message.lower()

    @pytest.mark.asyncio
    async def test_exit_with_allocation_succeeds(self, oms):
        """Test EXIT with allocation succeeds."""
        # Set up allocation
        oms.state.update_allocation("005930", "KMP", 100)

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
            risk_payload=RiskPayload(rationale_code="stop_hit"),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED

    @pytest.mark.asyncio
    async def test_reduce_creates_sell_order(self, oms):
        """Test REDUCE creates sell order."""
        # Set up allocation
        oms.state.update_allocation("005930", "KMP", 100)

        intent = Intent(
            intent_type=IntentType.REDUCE,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=50,
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED


class TestOMSCoreApplyFill:
    """Tests for fill handling."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    @pytest.mark.asyncio
    async def test_buy_fill_updates_allocation(self, oms):
        """Test buy fill updates allocation."""
        from oms.state import WorkingOrder, OrderStatus

        wo = WorkingOrder(
            order_id="ORD001",
            symbol="005930",
            side="BUY",
            qty=100,
            price=72000,
            strategy_id="KMP",
        )

        await oms._apply_fill(wo, 100)

        pos = oms.state.get_position("005930")
        alloc = pos.allocations.get("KMP")
        assert alloc is not None
        assert alloc.qty == 100

    @pytest.mark.asyncio
    async def test_sell_fill_reduces_allocation(self, oms):
        """Test sell fill reduces allocation."""
        from oms.state import WorkingOrder

        # Set up initial allocation
        oms.state.update_allocation("005930", "KMP", 100)

        wo = WorkingOrder(
            order_id="ORD001",
            symbol="005930",
            side="SELL",
            qty=50,
            price=72000,
            strategy_id="KMP",
        )

        await oms._apply_fill(wo, 50)

        pos = oms.state.get_position("005930")
        assert pos.allocations["KMP"].qty == 50


class TestOMSCoreReconciliation:
    """Tests for reconciliation."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    @pytest.mark.asyncio
    async def test_allocation_drift_detection(self, oms):
        """Test allocation drift is detected and assigned to UNKNOWN."""
        # Set up position with drift
        oms.state.update_position("005930", real_qty=150)
        oms.state.update_allocation("005930", "KMP", 100)
        # Drift = 150 - 100 = 50

        await oms._check_allocation_drift()

        pos = oms.state.get_position("005930")
        assert pos.frozen is True
        assert "_UNKNOWN_" in pos.allocations
        assert pos.allocations["_UNKNOWN_"].qty == 50

    @pytest.mark.asyncio
    async def test_no_drift_when_orders_in_flight(self, oms):
        """Test drift is not flagged when orders in flight."""
        from oms.state import WorkingOrder

        # Set up position with apparent drift
        oms.state.update_position("005930", real_qty=150)
        oms.state.update_allocation("005930", "KMP", 100)

        # But order is in flight
        wo = WorkingOrder(
            order_id="ORD001",
            symbol="005930",
            side="BUY",
            qty=50,
            strategy_id="KMP",
        )
        oms.state.add_working_order("005930", wo)

        await oms._check_allocation_drift()

        pos = oms.state.get_position("005930")
        assert pos.frozen is False


class TestOMSCoreHelpers:
    """Tests for helper methods."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    def test_get_position(self, oms):
        """Test get_position returns position."""
        oms.state.update_position("005930", real_qty=100)

        pos = oms.get_position("005930")

        assert pos.real_qty == 100

    def test_get_allocation(self, oms):
        """Test get_allocation returns allocation."""
        oms.state.update_allocation("005930", "KMP", 100)

        alloc = oms.get_allocation("005930", "KMP")

        assert alloc == 100

    def test_get_allocation_missing(self, oms):
        """Test get_allocation returns 0 for missing."""
        alloc = oms.get_allocation("005930", "KMP")
        assert alloc == 0


class TestOMSCoreLifecycle:
    """Tests for OMS lifecycle methods."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        return OMSCore(mock_api)

    @pytest.mark.asyncio
    async def test_flatten_all(self, oms):
        """Test flatten_all submits market sells."""
        oms.state.update_position("005930", real_qty=100)
        oms.state.update_position("000660", real_qty=50)

        await oms.flatten_all()

        assert oms.risk.flatten_in_progress is True
        assert oms.risk.halt_new_entries is True

    @pytest.mark.asyncio
    async def test_eod_cleanup(self, oms):
        """Test EOD cleanup resets state."""
        from oms.state import WorkingOrder

        oms.state.daily_pnl = 1000000
        oms.state.daily_pnl_pct = 0.01
        oms.risk.halt_new_entries = True

        await oms.eod_cleanup()

        assert oms.state.daily_pnl == 0.0
        assert oms.state.daily_pnl_pct == 0.0
        assert oms.risk.halt_new_entries is False

    @pytest.mark.asyncio
    async def test_shutdown(self, oms):
        """Test shutdown cancels reconciliation task."""
        # Start reconciliation
        oms._reconcile_task = asyncio.create_task(asyncio.sleep(100))

        await oms.shutdown()

        # After shutdown, task should be done (cancelled or finished)
        # The task may not report cancelled() immediately, but it should be done
        await asyncio.sleep(0.01)  # Allow cancellation to propagate
        assert oms._reconcile_task.done()


class TestSyncWorkingOrders:
    """Tests for _sync_working_orders method."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    @pytest.mark.asyncio
    async def test_broker_filled_qty_triggers_fill(self, oms):
        """Test broker returns updated filled_qty and fill is applied."""
        from oms.state import WorkingOrder, OrderStatus
        from oms.adapter import BrokerOrder

        wo = WorkingOrder(
            order_id="ORD001",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            strategy_id="KMP",
            status=OrderStatus.WORKING,
        )
        oms.state.add_working_order("005930", wo)

        # Mock adapter.get_orders to return broker order with fills
        broker_order = BrokerOrder(
            order_id="ORD001",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=100,
            price=72000,
            status="FILLED",
            created_at="09:30:00",
            branch="",
        )
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[broker_order]))

        await oms._sync_working_orders()

        # Fill should have been applied, creating an allocation
        pos = oms.state.get_position("005930")
        alloc = pos.allocations.get("KMP")
        assert alloc is not None
        assert alloc.qty == 100
        assert wo.filled_qty == 100
        assert wo.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_broker_order_disappeared_removes_working_order(self, oms):
        """Test broker order disappeared results in working order removal."""
        from oms.state import WorkingOrder, OrderStatus

        wo = WorkingOrder(
            order_id="ORD002",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            strategy_id="KMP",
            status=OrderStatus.WORKING,
        )
        oms.state.add_working_order("005930", wo)

        # Broker returns empty list (order disappeared)
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))

        await oms._sync_working_orders()

        # Working order should have been removed
        pos = oms.state.get_position("005930")
        assert not pos.has_working_orders()
        assert wo.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_branch_code_captured_from_broker(self, oms):
        """Test branch code is captured from broker order."""
        from oms.state import WorkingOrder, OrderStatus
        from oms.adapter import BrokerOrder

        wo = WorkingOrder(
            order_id="ORD003",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            strategy_id="KMP",
            status=OrderStatus.WORKING,
            branch="",  # No branch initially
        )
        oms.state.add_working_order("005930", wo)

        broker_order = BrokerOrder(
            order_id="ORD003",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            status="WORKING",
            created_at="09:30:00",
            branch="06010",  # Branch code from broker
        )
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[broker_order]))

        await oms._sync_working_orders()

        assert wo.branch == "06010"


class TestEnforceOrderTimeouts:
    """Tests for _enforce_order_timeouts method."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    @pytest.mark.asyncio
    async def test_order_past_timeout_is_cancelled(self, oms):
        """Test order past its timeout is cancelled."""
        from oms.state import WorkingOrder, OrderStatus
        from oms.adapter import BrokerOrder, AdapterResult

        wo = WorkingOrder(
            order_id="ORD010",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            strategy_id="KMP",
            status=OrderStatus.WORKING,
            cancel_after_sec=10,
            submit_ts=time.time() - 20,  # Submitted 20s ago, timeout 10s
        )
        oms.state.add_working_order("005930", wo)

        # Broker shows no additional fills
        broker_order = BrokerOrder(
            order_id="ORD010",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            status="WORKING",
            created_at="09:30:00",
        )
        oms.adapter.cancel_order = AsyncMock(return_value=AdapterResult(success=True))

        broker_by_id = {broker_order.order_id: broker_order}
        await oms._enforce_order_timeouts(broker_by_id)

        # Order should be removed from working orders
        pos = oms.state.get_position("005930")
        assert not pos.has_working_orders()
        oms.adapter.cancel_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_within_timeout_is_kept(self, oms):
        """Test order within its timeout is not cancelled."""
        from oms.state import WorkingOrder, OrderStatus

        wo = WorkingOrder(
            order_id="ORD011",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            strategy_id="KMP",
            status=OrderStatus.WORKING,
            cancel_after_sec=60,
            submit_ts=time.time() - 5,  # Submitted 5s ago, timeout 60s
        )
        oms.state.add_working_order("005930", wo)

        oms.adapter.cancel_order = AsyncMock()

        await oms._enforce_order_timeouts({})

        # Order should still be in working orders
        pos = oms.state.get_position("005930")
        assert pos.has_working_orders()
        oms.adapter.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_pre_cancel_fill_query_detects_fills(self, oms):
        """Test pre-cancel fill query detects fills before cancelling."""
        from oms.state import WorkingOrder, OrderStatus
        from oms.adapter import BrokerOrder, AdapterResult

        wo = WorkingOrder(
            order_id="ORD012",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            strategy_id="KMP",
            status=OrderStatus.WORKING,
            cancel_after_sec=10,
            submit_ts=time.time() - 20,  # Past timeout
        )
        oms.state.add_working_order("005930", wo)

        # Broker shows 50 shares filled just before cancel
        broker_order = BrokerOrder(
            order_id="ORD012",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=50,
            price=72000,
            status="WORKING",
            created_at="09:30:00",
        )
        oms.adapter.cancel_order = AsyncMock(return_value=AdapterResult(success=True))

        broker_by_id = {broker_order.order_id: broker_order}
        await oms._enforce_order_timeouts(broker_by_id)

        # Fill should have been applied before cancel
        pos = oms.state.get_position("005930")
        alloc = pos.allocations.get("KMP")
        assert alloc is not None
        assert alloc.qty == 50
        # Cancel was called for remaining 50
        oms.adapter.cancel_order.assert_called_once_with(
            "ORD012", "005930", 50, branch=""
        )


class TestReconcile:
    """Tests for _reconcile full reconciliation cycle."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        return oms

    @pytest.mark.asyncio
    async def test_positions_synced_from_broker(self, oms):
        """Test positions are synced from broker during reconciliation."""
        from oms.adapter import BrokerPosition

        # Set up initial state with no positions
        # Broker has a position
        broker_pos = BrokerPosition(
            symbol="005930",
            qty=100,
            avg_price=70000,
            current_price=72000,
            pnl=2.86,
        )
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))
        oms.adapter.get_balance_snapshot = AsyncMock(return_value=(
            BrokerQueryResult(ok=True, data=[broker_pos]),
            100_000_000,
        ))
        oms.adapter.get_buyable_cash = AsyncMock(return_value=50_000_000)

        await oms._reconcile(cycle_count=0)

        pos = oms.state.get_position("005930")
        assert pos.real_qty == 100
        assert pos.avg_price == 70000

    @pytest.mark.asyncio
    async def test_account_info_updated(self, oms):
        """Test account info is updated during reconciliation."""
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))
        oms.adapter.get_balance_snapshot = AsyncMock(return_value=(
            BrokerQueryResult(ok=True, data=[]),
            120_000_000,
        ))
        oms.adapter.get_buyable_cash = AsyncMock(return_value=60_000_000)

        await oms._reconcile(cycle_count=0)

        assert oms.state.equity == 120_000_000
        assert oms.state.buyable_cash == 60_000_000

    @pytest.mark.asyncio
    async def test_buyable_cash_skipped_on_non_zero_cycle(self, oms):
        """Test buyable_cash is NOT fetched on non-zero cycle (every 6th only)."""
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))
        oms.adapter.get_balance_snapshot = AsyncMock(return_value=(
            BrokerQueryResult(ok=True, data=[]),
            100_000_000,
        ))
        oms.adapter.get_buyable_cash = AsyncMock(return_value=99_000_000)

        # Set initial buyable_cash
        oms.state.buyable_cash = 50_000_000

        # Cycle 1 (not multiple of 6) — should NOT call get_buyable_cash
        await oms._reconcile(cycle_count=1)

        oms.adapter.get_buyable_cash.assert_not_called()
        assert oms.state.buyable_cash == 50_000_000  # unchanged

    @pytest.mark.asyncio
    async def test_buyable_cash_fetched_on_sixth_cycle(self, oms):
        """Test buyable_cash IS fetched on 6th cycle."""
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))
        oms.adapter.get_balance_snapshot = AsyncMock(return_value=(
            BrokerQueryResult(ok=True, data=[]),
            100_000_000,
        ))
        oms.adapter.get_buyable_cash = AsyncMock(return_value=99_000_000)

        await oms._reconcile(cycle_count=6)

        oms.adapter.get_buyable_cash.assert_called_once()
        assert oms.state.buyable_cash == 99_000_000


class TestHandleModifyRisk:
    """Tests for _handle_modify_risk method."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        return oms

    @pytest.mark.asyncio
    async def test_updates_soft_stop_px(self, oms):
        """Test MODIFY_RISK updates soft_stop_px on allocation."""
        # Set up existing allocation
        oms.state.update_allocation("005930", "KMP", 100, cost_basis=72000)

        intent = Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id="KMP",
            symbol="005930",
            risk_payload=RiskPayload(stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        pos = oms.state.get_position("005930")
        alloc = pos.allocations["KMP"]
        assert alloc.soft_stop_px == 71000

    @pytest.mark.asyncio
    async def test_updates_hard_stop_px(self, oms):
        """Test MODIFY_RISK updates hard_stop_px on position."""
        # Set up existing allocation
        oms.state.update_allocation("005930", "KMP", 100, cost_basis=72000)

        intent = Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id="KMP",
            symbol="005930",
            risk_payload=RiskPayload(hard_stop_px=70000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        pos = oms.state.get_position("005930")
        assert pos.hard_stop_px == 70000

    @pytest.mark.asyncio
    async def test_no_allocation_rejected(self, oms):
        """Test MODIFY_RISK with no allocation is rejected."""
        intent = Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id="KMP",
            symbol="005930",
            risk_payload=RiskPayload(stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.REJECTED
        assert "no allocation" in result.message.lower()


class TestApplyFillSellPath:
    """Tests for _apply_fill SELL path with realized P&L."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    @pytest.mark.asyncio
    async def test_sell_fill_records_realized_pnl(self, oms):
        """Test sell fill records realized P&L based on cost basis."""
        from oms.state import WorkingOrder, OrderStatus

        # Set up allocation with known cost basis
        oms.state.update_allocation("005930", "KMP", 100, cost_basis=70000)

        wo = WorkingOrder(
            order_id="ORD020",
            symbol="005930",
            side="SELL",
            qty=50,
            price=72000,  # Sell at 72000, cost basis 70000
            strategy_id="KMP",
        )

        await oms._apply_fill(wo, 50)

        # Realized PnL = (72000 - 70000) * 50 = 100,000
        assert oms.state.daily_realized_pnl == 100_000

        # Allocation should be reduced
        pos = oms.state.get_position("005930")
        assert pos.allocations["KMP"].qty == 50


class TestPlanAndExecuteSetTarget:
    """Tests for _plan_and_execute SET_TARGET path."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        """Create OMSCore for testing."""
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        return oms

    @pytest.mark.asyncio
    async def test_positive_delta_creates_buy(self, oms):
        """Test SET_TARGET with delta > 0 creates BUY order."""
        # No existing allocation, target 100 -> delta = +100
        intent = Intent(
            intent_type=IntentType.SET_TARGET,
            strategy_id="KMP",
            symbol="005930",
            target_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        assert result.order_id is not None

        # Working order should be a BUY
        pos = oms.state.get_position("005930")
        assert pos.has_working_orders()
        wo = pos.working_orders[0]
        assert wo.side == "BUY"
        assert wo.qty == 100

    @pytest.mark.asyncio
    async def test_negative_delta_creates_sell(self, oms):
        """Test SET_TARGET with delta < 0 creates SELL order."""
        # Existing allocation of 100, target 50 -> delta = -50
        oms.state.update_allocation("005930", "KMP", 100, cost_basis=72000)

        intent = Intent(
            intent_type=IntentType.SET_TARGET,
            strategy_id="KMP",
            symbol="005930",
            target_qty=50,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        assert result.order_id is not None

        # Working order should be a SELL for 50
        pos = oms.state.get_position("005930")
        assert pos.has_working_orders()
        wo = pos.working_orders[0]
        assert wo.side == "SELL"
        assert wo.qty == 50

    @pytest.mark.asyncio
    async def test_zero_delta_returns_already_at_target(self, oms):
        """Test SET_TARGET with delta == 0 returns already at target."""
        # Existing allocation of 100, target 100 -> delta = 0
        oms.state.update_allocation("005930", "KMP", 100, cost_basis=72000)

        intent = Intent(
            intent_type=IntentType.SET_TARGET,
            strategy_id="KMP",
            symbol="005930",
            target_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        assert "already at target" in result.message.lower()


class TestOMSCoreStart:
    """Tests for start() and _load_persisted_state() methods."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        """Create mock KIS API."""
        return mock_kis_api

    @pytest.mark.asyncio
    async def test_start_calls_persistence_connect_and_load(self, mock_api):
        """Test start() connects persistence and loads state."""
        mock_persistence = MagicMock()
        mock_persistence.connect = AsyncMock()
        mock_persistence.load_positions = AsyncMock(return_value={})
        mock_persistence.load_allocations = AsyncMock(return_value={})
        mock_persistence.load_working_orders = AsyncMock(return_value=[])
        mock_persistence.load_oms_state = AsyncMock(return_value=None)
        mock_persistence.close = AsyncMock()

        oms = OMSCore(mock_api, persistence=mock_persistence)

        # Patch start_reconciliation_loop to avoid background task
        oms.start_reconciliation_loop = AsyncMock()

        await oms.start()

        mock_persistence.connect.assert_awaited_once()
        mock_persistence.load_positions.assert_awaited_once()
        mock_persistence.load_allocations.assert_awaited_once()
        mock_persistence.load_working_orders.assert_awaited_once()
        mock_persistence.load_oms_state.assert_awaited_once()
        oms.start_reconciliation_loop.assert_awaited_once()

        # Cleanup
        await oms.shutdown()

    @pytest.mark.asyncio
    async def test_start_without_persistence(self, mock_api):
        """Test start() works without persistence configured."""
        oms = OMSCore(mock_api, persistence=None)

        # Patch start_reconciliation_loop to avoid background task
        oms.start_reconciliation_loop = AsyncMock()

        await oms.start()

        # Should still start reconciliation loop
        oms.start_reconciliation_loop.assert_awaited_once()

        # Cleanup
        await oms.shutdown()
