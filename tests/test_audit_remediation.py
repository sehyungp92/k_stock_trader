"""Tests for audit remediation changes (2026-03-19).

Covers:
- Change 1: OMS client None semantics + strategy guards
- Change 2: Exit idempotency eviction
- Change 3: Adapter retry identity tracking
- Change 4: Gamma reconciliation + mutation deferral
- Change 5: PCIM partial/dust exit handling
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from datetime import datetime, timedelta


# =====================================================================
# Change 1: OMS Client None Semantics
# =====================================================================

class TestOMSClientNoneSemantics:
    """Verify OMS client returns None on failure, not empty defaults."""

    @pytest.mark.asyncio
    async def test_get_all_positions_returns_none_on_failure(self):
        """get_all_positions returns None when OMS is unreachable."""
        from oms_client.client import OMSClient

        client = OMSClient("http://localhost:8000")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))
        mock_session.closed = False

        async def fake_get_session():
            return mock_session

        client._get_session = fake_get_session
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.get_all_positions()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_positions_returns_empty_dict_when_genuinely_empty(self):
        """get_all_positions returns {} when OMS says no positions."""
        from oms_client.client import OMSClient

        client = OMSClient("http://localhost:8000")
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = await client.get_all_positions()
        assert result == {}
        await client.close()

    @pytest.mark.asyncio
    async def test_get_allocation_returns_none_on_failure(self):
        """get_allocation returns None when get_position returns None."""
        from oms_client.client import OMSClient

        client = OMSClient("http://localhost:8000")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))
        mock_session.closed = False

        async def fake_get_session():
            return mock_session

        client._get_session = fake_get_session
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.get_allocation("005930", "ALPHA")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_account_state_returns_none_on_failure(self):
        """get_account_state returns None when OMS is unreachable."""
        from oms_client.client import OMSClient

        client = OMSClient("http://localhost:8000")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))
        mock_session.closed = False

        async def fake_get_session():
            return mock_session

        client._get_session = fake_get_session
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.get_account_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_strategy_allocations_returns_none_on_failure(self):
        """get_strategy_allocations returns None when OMS is unreachable."""
        from oms_client.client import OMSClient

        client = OMSClient("http://localhost:8000")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))
        mock_session.closed = False

        async def fake_get_session():
            return mock_session

        client._get_session = fake_get_session
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.get_strategy_allocations("ALPHA")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_strategy_allocations_returns_empty_when_genuinely_empty(self):
        """get_strategy_allocations returns {} when OMS says no allocations."""
        from oms_client.client import OMSClient

        client = OMSClient("http://localhost:8000")
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = await client.get_strategy_allocations("ALPHA")
        assert result == {}
        await client.close()

    @pytest.mark.asyncio
    async def test_state_proxy_refresh_preserves_cache_on_none(self):
        """_OMSStateProxy.refresh() keeps old cache when OMS returns None."""
        from oms_client.client import OMSClient, AccountState

        client = OMSClient("http://localhost:8000")
        proxy = client.state

        # Seed with valid state
        proxy._cached_account = AccountState(equity=50_000_000)
        proxy._cached_positions = {"005930": MagicMock()}

        # Make OMS return None
        client.get_account_state = AsyncMock(return_value=None)
        client.get_all_positions = AsyncMock(return_value=None)

        await proxy.refresh()

        # Cache should be preserved
        assert proxy._cached_account.equity == 50_000_000
        assert "005930" in proxy._cached_positions


# =====================================================================
# Change 2: Exit Idempotency Eviction
# =====================================================================

class TestIdempotencyEviction:
    """Test idempotency cache eviction for unfilled SELL orders."""

    def test_remove_method_exists_and_works(self):
        """InMemoryIdempotencyStore.remove() deletes cached entry."""
        from oms.oms_core import InMemoryIdempotencyStore
        from oms.intent import IntentResult, IntentStatus

        store = InMemoryIdempotencyStore()
        result = IntentResult(intent_id="i1", status=IntentStatus.EXECUTED)
        store.put("key1", result)

        assert store.get("key1") is not None
        assert store.remove("key1") is True
        assert store.get("key1") is None
        # Removing non-existent key returns False
        assert store.remove("key1") is False

    def test_clear_method_empties_store(self):
        """InMemoryIdempotencyStore.clear() removes all entries."""
        from oms.oms_core import InMemoryIdempotencyStore
        from oms.intent import IntentResult, IntentStatus

        store = InMemoryIdempotencyStore()
        store.put("k1", IntentResult(intent_id="i1", status=IntentStatus.EXECUTED))
        store.put("k2", IntentResult(intent_id="i2", status=IntentStatus.EXECUTED))

        store.clear()
        assert store.get("k1") is None
        assert store.get("k2") is None

    @pytest.mark.asyncio
    async def test_sell_cancelled_evicts_idem_key(self):
        """Unfilled SELL CANCELLED working order evicts its idempotency cache entry."""
        from oms.oms_core import OMSCore, InMemoryIdempotencyStore
        from oms.state import WorkingOrder, OrderStatus
        from oms.intent import IntentResult, IntentStatus

        idem = InMemoryIdempotencyStore()
        idem.put("ALPHA:005930:EXIT:20260319:stop:100", IntentResult(
            intent_id="i1", status=IntentStatus.EXECUTED, order_id="ORD1",
        ))

        wo = WorkingOrder(
            order_id="ORD1", symbol="005930", side="SELL", qty=100,
            filled_qty=0, status=OrderStatus.WORKING, strategy_id="ALPHA",
            idempotency_key="ALPHA:005930:EXIT:20260319:stop:100",
        )

        core = MagicMock()
        core._idem = idem
        core.state = MagicMock()
        core.risk = MagicMock()
        core.persistence = None
        core._release_sector_reservation = MagicMock()
        core._rejection_counts = {}

        # Call the real method
        await OMSCore._finalize_working_order(
            core, wo, OrderStatus.CANCELLED, OrderStatus.WORKING, "ORDER_CANCELLED",
        )

        # Key should be evicted
        assert idem.get("ALPHA:005930:EXIT:20260319:stop:100") is None

    @pytest.mark.asyncio
    async def test_sell_filled_retains_idem_key(self):
        """Filled SELL working order does NOT evict idempotency cache."""
        from oms.oms_core import OMSCore, InMemoryIdempotencyStore
        from oms.state import WorkingOrder, OrderStatus
        from oms.intent import IntentResult, IntentStatus

        idem = InMemoryIdempotencyStore()
        key = "ALPHA:005930:EXIT:20260319:stop:100"
        idem.put(key, IntentResult(intent_id="i1", status=IntentStatus.EXECUTED, order_id="ORD1"))

        wo = WorkingOrder(
            order_id="ORD1", symbol="005930", side="SELL", qty=100,
            filled_qty=100, status=OrderStatus.FILLED, strategy_id="ALPHA",
            idempotency_key=key,
        )

        core = MagicMock()
        core._idem = idem
        core.state = MagicMock()
        core.risk = MagicMock()
        core.persistence = None
        core._release_sector_reservation = MagicMock()

        await OMSCore._finalize_working_order(
            core, wo, OrderStatus.FILLED, OrderStatus.WORKING, "ORDER_FILLED",
        )

        # Key should be retained (FILLED is not in eviction set)
        assert idem.get(key) is not None

    def test_working_order_has_idempotency_key_field(self):
        """WorkingOrder dataclass includes idempotency_key field."""
        from oms.state import WorkingOrder

        wo = WorkingOrder(
            order_id="ORD1", symbol="005930", side="SELL", qty=100,
            idempotency_key="test:key",
        )
        assert wo.idempotency_key == "test:key"

    def test_working_order_idempotency_key_defaults_to_none(self):
        """WorkingOrder.idempotency_key defaults to None for backward compat."""
        from oms.state import WorkingOrder

        wo = WorkingOrder(order_id="ORD1", symbol="005930", side="SELL", qty=100)
        assert wo.idempotency_key is None


# =====================================================================
# Change 3: Adapter Retry Safety
# =====================================================================

class TestAdapterRetryIdentity:
    """Test adapter tracks known IDs and fails closed on timeout ambiguity."""

    def test_adapter_has_known_order_ids(self):
        """KISExecutionAdapter initializes _known_order_ids set."""
        from oms.adapter import KISExecutionAdapter

        mock_api = MagicMock()
        adapter = KISExecutionAdapter(mock_api)
        assert hasattr(adapter, '_known_order_ids')
        assert isinstance(adapter._known_order_ids, set)
        assert len(adapter._known_order_ids) == 0

    def test_reset_clears_known_order_ids(self):
        """adapter.reset() clears the known order ID set."""
        from oms.adapter import KISExecutionAdapter

        mock_api = MagicMock()
        adapter = KISExecutionAdapter(mock_api)
        adapter._known_order_ids.add("ORD1")
        adapter._known_order_ids.add("ORD2")

        adapter.reset()
        assert len(adapter._known_order_ids) == 0

    @pytest.mark.asyncio
    async def test_successful_submit_tracks_order_id(self):
        """Successful order submit adds order_id to _known_order_ids."""
        from oms.adapter import KISExecutionAdapter

        mock_api = MagicMock()

        @dataclass
        class FakeOrderResult:
            success: bool = True
            order_id: str = "ORD123"
            error_code: str = ""
            error_message: str = ""

        mock_api.place_limit_buy = MagicMock(return_value=FakeOrderResult())
        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order("005930", "BUY", 100, "LIMIT", limit_price=50000)
        assert result.success
        assert "ORD123" in adapter._known_order_ids

    @pytest.mark.asyncio
    async def test_timeout_does_not_bind_unknown_open_order(self):
        """Timeouts do not bind a retry to an unknown open order."""
        from oms.adapter import KISExecutionAdapter, BrokerOrder, BrokerQueryResult

        mock_api = MagicMock()
        adapter = KISExecutionAdapter(mock_api)

        # Pre-populate known IDs
        adapter._known_order_ids.add("ORD_EXISTING")

        call_count = 0
        def fake_place(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timeout")

        mock_api.place_limit_buy = fake_place

        # Mock get_orders to return both known and unknown matches
        async def fake_get_orders():
            return BrokerQueryResult(ok=True, data=[
                BrokerOrder("ORD_EXISTING", "005930", "BUY", 100, 0, 50000, "WORKING", "09:01"),
                BrokerOrder("ORD_NEW", "005930", "BUY", 100, 0, 50000, "WORKING", "09:02"),
            ])

        adapter.get_orders = fake_get_orders

        result = await adapter.submit_order("005930", "BUY", 100, "LIMIT", limit_price=50000, max_retries=2)

        assert result.success is False
        assert "ambiguous after timeout" in result.message.lower()
        assert "ORD_NEW" not in adapter._known_order_ids
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_with_multiple_matches_fails_closed(self):
        """Timeouts with multiple possible matches also fail closed."""
        from oms.adapter import KISExecutionAdapter, BrokerOrder, BrokerQueryResult

        mock_api = MagicMock()
        adapter = KISExecutionAdapter(mock_api)

        call_count = 0

        def fake_place(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timeout")

        mock_api.place_limit_buy = fake_place

        # Two unknown matching orders ??ambiguous
        async def fake_get_orders():
            return BrokerQueryResult(ok=True, data=[
                BrokerOrder("ORD_A", "005930", "BUY", 100, 0, 50000, "WORKING", "09:01"),
                BrokerOrder("ORD_B", "005930", "BUY", 100, 0, 50000, "WORKING", "09:02"),
            ])

        adapter.get_orders = fake_get_orders

        result = await adapter.submit_order("005930", "BUY", 100, "LIMIT", limit_price=50000, max_retries=2)

        assert result.success is False
        assert "ambiguous after timeout" in result.message.lower()
        assert call_count == 1


# =====================================================================
# Change 4: Gamma Reconciliation + Mutation Deferral
# =====================================================================

class TestSoftStopPxPersistence:
    """Test soft_stop_px is persisted on BUY fill."""

    @pytest.mark.asyncio
    async def test_apply_fill_sets_soft_stop_px_on_buy(self):
        """_apply_fill persists soft_stop_px from intent risk payload."""
        from oms.oms_core import OMSCore
        from oms.state import StateStore, WorkingOrder, OrderStatus, StrategyAllocation
        from oms.intent import Intent, IntentType, Urgency, TimeHorizon, RiskPayload

        state = StateStore()
        # Pre-create allocation (simulating update_allocation was just called)
        state.update_allocation("005930", "GAMMA", 100, cost_basis=50000)

        intent = Intent(
            intent_type=IntentType.ENTER, strategy_id="GAMMA",
            symbol="005930", desired_qty=100,
            urgency=Urgency.LOW, time_horizon=TimeHorizon.SWING,
            risk_payload=RiskPayload(entry_px=50000, stop_px=48500),
        )

        wo = WorkingOrder(
            order_id="ORD1", symbol="005930", side="BUY", qty=100,
            price=50000, strategy_id="GAMMA",
        )

        core = MagicMock()
        core.state = state
        core.risk = MagicMock()
        core.persistence = None

        await OMSCore._apply_fill(core, wo, 100, intent=intent)

        alloc = state.get_position("005930").allocations["GAMMA"]
        assert alloc.soft_stop_px == 48500

    @pytest.mark.asyncio
    async def test_apply_fill_no_stop_px_leaves_none(self):
        """_apply_fill with no stop_px in intent leaves soft_stop_px as None."""
        from oms.oms_core import OMSCore
        from oms.state import StateStore, WorkingOrder
        from oms.intent import Intent, IntentType, Urgency, TimeHorizon, RiskPayload

        state = StateStore()
        state.update_allocation("005930", "ALPHA", 50, cost_basis=60000)

        intent = Intent(
            intent_type=IntentType.ENTER, strategy_id="ALPHA",
            symbol="005930", desired_qty=50,
            urgency=Urgency.HIGH, time_horizon=TimeHorizon.INTRADAY,
            risk_payload=RiskPayload(entry_px=60000, stop_px=None),
        )

        wo = WorkingOrder(
            order_id="ORD2", symbol="005930", side="BUY", qty=50,
            price=60000, strategy_id="ALPHA",
        )

        core = MagicMock()
        core.state = state
        core.risk = MagicMock()
        core.persistence = None

        await OMSCore._apply_fill(core, wo, 50, intent=intent)

        alloc = state.get_position("005930").allocations["ALPHA"]
        assert alloc.soft_stop_px is None


# =====================================================================
# Change 5: PCIM Partial/Dust Exit Handling
# =====================================================================

class TestPCIMPartialFillBranch:
    """Test PCIM handles partial STOP/DAY15_EXIT fills."""

    # These are integration-level tests that would require full PCIM setup.
    # We test the component behavior instead.

    def test_reduce_position_exists(self):
        """PositionManager.reduce_position exists and works."""
        from strategy_pcim.positions.manager import PositionManager, PCIMPosition
        from datetime import date

        pm = PositionManager()
        pm.add_position(PCIMPosition(
            symbol="005930", entry_date=date.today(),
            entry_price=50000, qty=100, atr_at_entry=1000,
        ))

        pm.reduce_position("005930", 30)
        pos = pm.get_position("005930")
        assert pos.remaining_qty == 70

    def test_submit_exit_exists(self):
        """PositionManager.submit_exit sets pending exit state."""
        from strategy_pcim.positions.manager import PositionManager, PCIMPosition
        from datetime import date

        pm = PositionManager()
        pm.add_position(PCIMPosition(
            symbol="005930", entry_date=date.today(),
            entry_price=50000, qty=100, atr_at_entry=1000,
        ))

        pm.submit_exit("005930", "PARTIAL_FILL_EXIT", 100, "intent-1", 50000)
        pos = pm.get_position("005930")
        assert pos.pending_exit_type == "PARTIAL_FILL_EXIT"


# =====================================================================
# EOD cleanup clears idem store + adapter
# =====================================================================

class TestEODCleanup:
    """Test eod_cleanup clears idempotency store and adapter state."""

    @pytest.mark.asyncio
    async def test_eod_clears_idem_store(self):
        """eod_cleanup calls _idem.clear()."""
        from oms.oms_core import OMSCore, InMemoryIdempotencyStore
        from oms.intent import IntentResult, IntentStatus

        idem = InMemoryIdempotencyStore()
        idem.put("k1", IntentResult(intent_id="i1", status=IntentStatus.EXECUTED))

        mock_adapter = AsyncMock()
        mock_adapter.get_orders = AsyncMock(return_value=MagicMock(ok=True, data=[]))
        mock_adapter.reset = MagicMock()

        core = MagicMock()
        core._idem = idem
        core._rejection_counts = {"k1": 2}
        core.state = MagicMock()
        core.state.get_all_positions = MagicMock(return_value={})
        core.risk = MagicMock()
        core.adapter = mock_adapter
        core.persistence = None

        await OMSCore.eod_cleanup(core)

        assert idem.get("k1") is None
        mock_adapter.reset.assert_called_once()
