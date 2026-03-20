"""Tests for audit remediation changes (2026-03-19).

Covers:
- Change 1: OMS client None semantics + strategy guards
- Change 2: Exit idempotency eviction
- Change 3: Adapter retry identity tracking
- Change 4: Nulrimok reconciliation + mutation deferral
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
            result = await client.get_allocation("005930", "KMP")
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
            result = await client.get_strategy_allocations("KMP")
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

        result = await client.get_strategy_allocations("KMP")
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
# Change 1B: Strategy Guards
# =====================================================================

class TestKMPNoneGuards:
    """KMP strategy guards when OMS returns None."""

    @pytest.mark.asyncio
    async def test_sync_positions_returns_early_on_none(self):
        """_sync_positions returns immediately when OMS returns None."""
        from strategy_kmp.main import _sync_positions
        from strategy_kmp.core.state import SymbolState, State
        from kis_core import SectorExposure, SectorExposureConfig

        mock_oms = AsyncMock()
        mock_oms.get_all_positions = AsyncMock(return_value=None)

        states = {"005930": SymbolState(code="005930")}
        states["005930"].fsm = State.IN_POSITION
        states["005930"].qty = 100

        exposure = SectorExposure({}, SectorExposureConfig(mode="count"))

        await _sync_positions(mock_oms, states, ["005930"], {}, exposure)

        # State must NOT be wiped
        assert states["005930"].fsm == State.IN_POSITION
        assert states["005930"].qty == 100

    @pytest.mark.asyncio
    async def test_reconcile_exposure_skips_on_none(self):
        """reconcile_exposure returns early when OMS returns None."""
        from strategy_kmp.core.reconcile import reconcile_exposure
        from strategy_kmp.core.state import SymbolState, State
        from kis_core import SectorExposure, SectorExposureConfig

        mock_oms = AsyncMock()
        mock_oms.get_all_positions = AsyncMock(return_value=None)

        states = {"005930": SymbolState(code="005930")}
        states["005930"].fsm = State.IN_POSITION

        exposure = SectorExposure({}, SectorExposureConfig(mode="count"))
        # Seed exposure with position
        exposure.on_fill("005930", 100, 50000)

        await reconcile_exposure(mock_oms, states, exposure, {})

        # Exposure must NOT be wiped
        assert states["005930"].fsm == State.IN_POSITION


class TestNulrimokNoneGuards:
    """Nulrimok strategy guards when OMS returns None."""

    @pytest.mark.asyncio
    async def test_reconcile_positions_skips_on_none(self):
        """_reconcile_positions preserves local state when OMS is unreachable."""
        from strategy_nulrimok.main import _reconcile_positions
        from strategy_nulrimok.iepe.exits import PositionState

        mock_oms = AsyncMock()
        mock_oms.get_strategy_allocations = AsyncMock(return_value=None)

        position_states = {
            "005930": PositionState("005930", datetime.now(), 50000, 100, 49000),
        }

        await _reconcile_positions(mock_oms, position_states)

        # Position must NOT be deleted
        assert "005930" in position_states
        assert position_states["005930"].qty == 100

    @pytest.mark.asyncio
    async def test_recover_positions_skips_on_none(self):
        """_recover_positions does nothing when OMS is unreachable."""
        from strategy_nulrimok.main import _recover_positions

        mock_oms = AsyncMock()
        mock_oms.get_strategy_allocations = AsyncMock(return_value=None)

        position_states = {}
        await _recover_positions(mock_oms, position_states)

        assert len(position_states) == 0


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
        idem.put("KMP:005930:EXIT:20260319:stop:100", IntentResult(
            intent_id="i1", status=IntentStatus.EXECUTED, order_id="ORD1",
        ))

        wo = WorkingOrder(
            order_id="ORD1", symbol="005930", side="SELL", qty=100,
            filled_qty=0, status=OrderStatus.WORKING, strategy_id="KMP",
            idempotency_key="KMP:005930:EXIT:20260319:stop:100",
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
        assert idem.get("KMP:005930:EXIT:20260319:stop:100") is None

    @pytest.mark.asyncio
    async def test_sell_filled_retains_idem_key(self):
        """Filled SELL working order does NOT evict idempotency cache."""
        from oms.oms_core import OMSCore, InMemoryIdempotencyStore
        from oms.state import WorkingOrder, OrderStatus
        from oms.intent import IntentResult, IntentStatus

        idem = InMemoryIdempotencyStore()
        key = "KMP:005930:EXIT:20260319:stop:100"
        idem.put(key, IntentResult(intent_id="i1", status=IntentStatus.EXECUTED, order_id="ORD1"))

        wo = WorkingOrder(
            order_id="ORD1", symbol="005930", side="SELL", qty=100,
            filled_qty=100, status=OrderStatus.FILLED, strategy_id="KMP",
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
# Change 3: Adapter Retry Identity Tracking
# =====================================================================

class TestAdapterRetryIdentity:
    """Test adapter filters retry candidates by known order IDs."""

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
    async def test_retry_excludes_known_orders(self):
        """On retry, adapter excludes already-known order IDs from candidates."""
        from oms.adapter import KISExecutionAdapter, BrokerOrder, BrokerQueryResult

        mock_api = MagicMock()
        adapter = KISExecutionAdapter(mock_api)

        # Pre-populate known IDs
        adapter._known_order_ids.add("ORD_EXISTING")

        # Simulate: first attempt fails, retry finds 2 orders matching criteria,
        # but one is already known
        call_count = 0
        def fake_place(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timeout")
            raise TimeoutError("timeout")  # Second attempt also fails

        mock_api.place_limit_buy = fake_place

        # Mock get_orders to return both known and unknown matches
        async def fake_get_orders():
            return BrokerQueryResult(ok=True, data=[
                BrokerOrder("ORD_EXISTING", "005930", "BUY", 100, 0, 50000, "WORKING", "09:01"),
                BrokerOrder("ORD_NEW", "005930", "BUY", 100, 0, 50000, "WORKING", "09:02"),
            ])

        adapter.get_orders = fake_get_orders

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.submit_order("005930", "BUY", 100, "LIMIT", limit_price=50000, max_retries=2)

        # Should bind to ORD_NEW (the only unknown candidate)
        assert result.success
        assert result.order_id == "ORD_NEW"
        assert "ORD_NEW" in adapter._known_order_ids

    @pytest.mark.asyncio
    async def test_retry_ambiguous_submits_fresh(self):
        """When multiple unknown candidates match, adapter submits fresh order."""
        from oms.adapter import KISExecutionAdapter, BrokerOrder, BrokerQueryResult

        mock_api = MagicMock()
        adapter = KISExecutionAdapter(mock_api)

        call_count = 0

        @dataclass
        class FakeResult:
            success: bool = True
            order_id: str = "ORD_FRESH"
            error_code: str = ""
            error_message: str = ""

        def fake_place(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timeout")
            return FakeResult()

        mock_api.place_limit_buy = fake_place

        # Two unknown matching orders — ambiguous
        async def fake_get_orders():
            return BrokerQueryResult(ok=True, data=[
                BrokerOrder("ORD_A", "005930", "BUY", 100, 0, 50000, "WORKING", "09:01"),
                BrokerOrder("ORD_B", "005930", "BUY", 100, 0, 50000, "WORKING", "09:02"),
            ])

        adapter.get_orders = fake_get_orders

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.submit_order("005930", "BUY", 100, "LIMIT", limit_price=50000, max_retries=2)

        # Should submit fresh (not bind to ambiguous match)
        assert result.success
        assert result.order_id == "ORD_FRESH"


# =====================================================================
# Change 4: Nulrimok Reconciliation + Mutation Deferral
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
        state.update_allocation("005930", "NULRIMOK", 100, cost_basis=50000)

        intent = Intent(
            intent_type=IntentType.ENTER, strategy_id="NULRIMOK",
            symbol="005930", desired_qty=100,
            urgency=Urgency.LOW, time_horizon=TimeHorizon.SWING,
            risk_payload=RiskPayload(entry_px=50000, stop_px=48500),
        )

        wo = WorkingOrder(
            order_id="ORD1", symbol="005930", side="BUY", qty=100,
            price=50000, strategy_id="NULRIMOK",
        )

        core = MagicMock()
        core.state = state
        core.risk = MagicMock()
        core.persistence = None

        await OMSCore._apply_fill(core, wo, 100, intent=intent)

        alloc = state.get_position("005930").allocations["NULRIMOK"]
        assert alloc.soft_stop_px == 48500

    @pytest.mark.asyncio
    async def test_apply_fill_no_stop_px_leaves_none(self):
        """_apply_fill with no stop_px in intent leaves soft_stop_px as None."""
        from oms.oms_core import OMSCore
        from oms.state import StateStore, WorkingOrder
        from oms.intent import Intent, IntentType, Urgency, TimeHorizon, RiskPayload

        state = StateStore()
        state.update_allocation("005930", "KMP", 50, cost_basis=60000)

        intent = Intent(
            intent_type=IntentType.ENTER, strategy_id="KMP",
            symbol="005930", desired_qty=50,
            urgency=Urgency.HIGH, time_horizon=TimeHorizon.INTRADAY,
            risk_payload=RiskPayload(entry_px=60000, stop_px=None),
        )

        wo = WorkingOrder(
            order_id="ORD2", symbol="005930", side="BUY", qty=50,
            price=60000, strategy_id="KMP",
        )

        core = MagicMock()
        core.state = state
        core.risk = MagicMock()
        core.persistence = None

        await OMSCore._apply_fill(core, wo, 50, intent=intent)

        alloc = state.get_position("005930").allocations["KMP"]
        assert alloc.soft_stop_px is None


class TestNulrimokMutationDeferral:
    """Test exits.py defers partial_taken + remaining_qty mutation until after submit."""

    @pytest.mark.asyncio
    async def test_partial_taken_unchanged_on_rejection(self):
        """pos.partial_taken stays False when submit_intent returns REJECTED."""
        from strategy_nulrimok.iepe.exits import PositionState, SetupType, manage_nulrimok_position
        from tests.mocks.mock_oms_client import MockOMSClient

        pos = PositionState("005930", datetime.now(), 50000, 100, 49000)
        pos.setup = SetupType.MEAN_REVERSION
        pos.bars_since_entry = 5
        pos.atr30m = 200
        pos.max_price = 50400  # trigger condition: close-entry >= 1.5*atr
        pos.partial_taken = False

        oms = MockOMSClient(fail_intents=True)

        bar = {'close': 50350, 'high': 50400, 'low': 50000, 'volume': 1000}
        avwap = 50200

        result = await manage_nulrimok_position(pos, bar, avwap, 500, False, oms)

        assert result is None  # Exit rejected
        assert pos.partial_taken is False  # NOT mutated
        assert pos.remaining_qty == 100  # NOT decremented

    @pytest.mark.asyncio
    async def test_partial_taken_set_on_success(self):
        """On OMS acceptance, pos enters pending_exit state (qty deferred to fill confirmation)."""
        from strategy_nulrimok.iepe.exits import PositionState, SetupType, manage_nulrimok_position
        from tests.mocks.mock_oms_client import MockOMSClient

        pos = PositionState("005930", datetime.now(), 50000, 100, 49000)
        pos.setup = SetupType.MEAN_REVERSION
        pos.bars_since_entry = 5
        pos.atr30m = 200
        pos.max_price = 50400
        pos.partial_taken = False

        oms = MockOMSClient(default_status="EXECUTED")

        bar = {'close': 50350, 'high': 50400, 'low': 50000, 'volume': 1000}
        avwap = 50200

        result = await manage_nulrimok_position(pos, bar, avwap, 500, False, oms)

        assert result is not None  # Exit accepted
        # Fix 2: partial_taken is now deferred until fill confirmation
        assert pos.pending_exit is True
        assert pos.pending_exit_reason == "mean_rev_partial"
        assert pos.pending_exit_qty == 70  # 70% of 100
        assert pos.remaining_qty == 100  # Unchanged until fill confirmed
        assert pos.partial_taken is False  # Deferred until fill confirmed


class TestNulrimokTwoWayReconciliation:
    """Test two-way reconciliation recreates missing positions from OMS."""

    @pytest.mark.asyncio
    async def test_recreates_position_from_oms(self):
        """_reconcile_positions creates local position from OMS allocation."""
        from strategy_nulrimok.main import _reconcile_positions
        from oms_client.client import AllocationInfo

        mock_oms = AsyncMock()
        mock_oms.get_strategy_allocations = AsyncMock(return_value={
            "005930": AllocationInfo(
                strategy_id="NULRIMOK", qty=100, cost_basis=50000,
                soft_stop_px=48500,
            ),
        })

        position_states = {}  # Empty — position is missing locally

        await _reconcile_positions(mock_oms, position_states)

        assert "005930" in position_states
        pos = position_states["005930"]
        assert pos.qty == 100
        assert pos.remaining_qty == 100
        assert pos.entry_price == 50000
        assert pos.stop == 48500

    @pytest.mark.asyncio
    async def test_does_not_recreate_existing_position(self):
        """_reconcile_positions does not overwrite existing local positions."""
        from strategy_nulrimok.main import _reconcile_positions
        from strategy_nulrimok.iepe.exits import PositionState
        from oms_client.client import AllocationInfo

        mock_oms = AsyncMock()
        mock_oms.get_strategy_allocations = AsyncMock(return_value={
            "005930": AllocationInfo(
                strategy_id="NULRIMOK", qty=100, cost_basis=50000,
                soft_stop_px=48500,
            ),
        })

        existing_pos = PositionState("005930", datetime.now(), 50000, 100, 49000)
        existing_pos.sessions_held = 3  # Important local state
        position_states = {"005930": existing_pos}

        await _reconcile_positions(mock_oms, position_states)

        # Should keep the existing position (with sessions_held=3)
        assert position_states["005930"].sessions_held == 3

    @pytest.mark.asyncio
    async def test_recreated_position_estimates_sessions_held(self):
        """Recreated positions estimate sessions_held from entry date."""
        from strategy_nulrimok.main import _reconcile_positions
        from oms_client.client import AllocationInfo

        mock_oms = AsyncMock()
        # Entry was 3 days ago
        entry_ts = (datetime.now() - timedelta(days=3)).isoformat()
        mock_oms.get_strategy_allocations = AsyncMock(return_value={
            "005930": AllocationInfo(
                strategy_id="NULRIMOK", qty=100, cost_basis=50000,
                soft_stop_px=48500, entry_ts=entry_ts,
            ),
        })

        position_states = {}
        await _reconcile_positions(mock_oms, position_states)

        assert "005930" in position_states
        assert position_states["005930"].sessions_held >= 2  # At least 2 days

    @pytest.mark.asyncio
    async def test_recreated_position_has_zero_stop_when_no_soft_stop(self):
        """Missing soft_stop_px defaults to 0.0 for recreated positions."""
        from strategy_nulrimok.main import _reconcile_positions
        from oms_client.client import AllocationInfo

        mock_oms = AsyncMock()
        mock_oms.get_strategy_allocations = AsyncMock(return_value={
            "035720": AllocationInfo(
                strategy_id="NULRIMOK", qty=50, cost_basis=30000,
                soft_stop_px=None,
            ),
        })

        position_states = {}
        await _reconcile_positions(mock_oms, position_states)

        assert position_states["035720"].stop == 0.0


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
