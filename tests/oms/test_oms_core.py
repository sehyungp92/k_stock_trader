"""Tests for OMS core module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import time
from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo

from oms.oms_core import OMSCore, InMemoryIdempotencyStore, IdempotencyStore
from oms.state import StateStore, StrategyAllocation
from oms.risk import RiskConfig
from oms.intent import Intent, IntentConstraints, IntentType, IntentStatus, IntentResult, Urgency, RiskPayload
from oms.adapter import BrokerOrder, BrokerQueryResult
from oms.stop_protection import PriceObservation, ProtectiveStop, StopStatus
from oms.stop_watcher import StopWatcher, StopWatcherHealth
from oms_client.client import _health_payload_ready


class _CapturingOMSEmitter:
    def __init__(self):
        self.intents = []
        self.risks = []
        self.order_events = []

    def emit_intent(self, intent, result, *, phase):
        self.intents.append((intent, result, phase))

    def emit_risk_decision(self, intent, risk_result, *, trace):
        self.risks.append((intent, risk_result, list(trace)))

    def emit_order_event(self, order, event_type, *, payload, intent):
        self.order_events.append((order, event_type, dict(payload), intent))


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
        oms._idem.put("ALPHA:005930:ENTER:20240115:test", cached_result)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="ALPHA",
            symbol="005930",
            desired_qty=100,
            idempotency_key="ALPHA:005930:ENTER:20240115:test",
        )

        result = await oms.submit_intent(intent)

        assert result.intent_id == "cached-id"
        assert result.status == IntentStatus.EXECUTED

    @pytest.mark.asyncio
    async def test_validation_failure(self, oms):
        """Test validation failure rejects intent."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="ALPHA",
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
            strategy_id="ALPHA",
            symbol="005930",
            desired_qty=100,
        )
        intent.constraints.expiry_ts = time.time() - 10

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.REJECTED
        assert "expired" in result.message.lower()

    @pytest.mark.asyncio
    async def test_durable_pending_reservation_defers_without_submit(self, mock_api):
        """Existing durable PENDING reservation must fail closed before broker submit."""
        persistence = MagicMock()
        persistence._is_connected = MagicMock(return_value=True)
        persistence.reserve_intent = AsyncMock(
            return_value=IntentResult(
                intent_id="reserved-intent",
                status=IntentStatus.DEFERRED,
                message="Idempotency key is already pending; reconcile before retry",
            )
        )
        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        oms.adapter.submit_order = AsyncMock()

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="ALPHA",
            symbol="005930",
            desired_qty=100,
            signal_hash="abc",
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.DEFERRED
        assert "already pending" in result.message
        persistence.reserve_intent.assert_awaited_once_with(intent)
        oms.adapter.submit_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_broker_success_order_persistence_failure_remains_durable_ambiguous(self, mock_api):
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence._is_connected = MagicMock(return_value=True)
        persistence.reserve_intent = AsyncMock(return_value=None)
        persistence.update_intent_submission_plan = AsyncMock()
        persistence.record_order = AsyncMock(return_value=None)
        persistence.record_order_event = AsyncMock()
        persistence.mark_intent_ambiguous = AsyncMock()
        persistence.record_intent = AsyncMock()

        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        oms.adapter.submit_order = AsyncMock(return_value=SimpleNamespace(success=True, order_id="ORD-ACCEPTED", message="ok"))

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="ALPHA",
            symbol="005930",
            desired_qty=10,
            signal_hash="abc",
            risk_payload=RiskPayload(entry_px=100.0, stop_px=95.0),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.DEFERRED
        assert result.order_id == "ORD-ACCEPTED"
        assert "reconciliation required" in result.message
        assert oms.risk.halt_new_entries is True
        assert oms._idem.get(intent.idempotency_key) is None
        persistence.mark_intent_ambiguous.assert_awaited_once()
        persistence.record_order_event.assert_not_awaited()
        assert persistence.record_intent.await_args.args[1].status == IntentStatus.DEFERRED

    @pytest.mark.asyncio
    async def test_stale_ambiguous_idempotency_reconciliation_records_missing_order_row(self, mock_api):
        row = {
            "intent_id": "intent-ambiguous",
            "idempotency_key": "idem-ambiguous",
            "strategy_id": "ALPHA",
            "symbol": "005930",
            "intent_type": "ENTER",
            "desired_qty": 10,
            "target_qty": None,
            "status": "DEFERRED",
            "order_id": "ORD-ACCEPTED",
            "planned_side": "BUY",
            "planned_qty": 10,
            "planned_order_type": "LIMIT",
            "planned_limit_price": 100.0,
            "submit_ref": "OMS-submit-ref",
            "created_ts": 1_780_000_000.0,
            "stop_px": 95.0,
            "hard_stop_px": 94.0,
            "reservation_reconcile_status": "AMBIGUOUS",
        }
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence.list_pending_idempotency = AsyncMock(return_value=[row])
        persistence.record_order = AsyncMock(return_value="oms-order-1")
        persistence.resolve_idempotency = AsyncMock(
            return_value=IntentResult("intent-ambiguous", IntentStatus.EXECUTED, order_id="ORD-ACCEPTED")
        )
        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)
        oms.adapter.get_orders = AsyncMock(
            return_value=BrokerQueryResult(
                ok=True,
                data=[
                    BrokerOrder(
                        order_id="ORD-ACCEPTED",
                        symbol="005930",
                        side="BUY",
                        qty=10,
                        filled_qty=0,
                        price=100.0,
                        status="WORKING",
                        created_at="2026-06-05T09:00:00+09:00",
                        order_type="LIMIT",
                        submit_ref="OMS-submit-ref",
                    )
                ],
            )
        )
        oms.adapter.get_orders.return_value.data[0].created_ts = 1_780_000_010.0

        reconciled = await oms.reconcile_stale_pending_idempotency(stale_after_sec=0)

        assert reconciled == 1
        persisted_order = persistence.record_order.await_args.args[0]
        assert persisted_order.order_id == "ORD-ACCEPTED"
        assert persisted_order.idempotency_key == "idem-ambiguous"
        assert persisted_order.risk_stop_px == 95.0
        assert persisted_order.risk_hard_stop_px == 94.0
        assert oms.state.get_position("005930").working_orders[0].order_id == "ORD-ACCEPTED"
        persistence.resolve_idempotency.assert_awaited_once()
        assert oms._idem.get("idem-ambiguous").status == IntentStatus.EXECUTED

    @pytest.mark.asyncio
    async def test_stale_idempotency_reconciliation_uses_adapter_normalized_known_order_id(self, mock_api):
        """Durable order_id recovery must work with real adapter-normalized KIS orders."""
        import pandas as pd

        row = {
            "intent_id": "intent-known-order",
            "idempotency_key": "idem-known-order",
            "strategy_id": "ALPHA",
            "symbol": "005930",
            "intent_type": "ENTER",
            "desired_qty": 10,
            "target_qty": None,
            "status": "DEFERRED",
            "order_id": "ORD-KIS",
            "planned_side": "BUY",
            "planned_qty": 10,
            "planned_order_type": "LIMIT",
            "planned_limit_price": 72000.0,
            "submit_ref": "OMS-submit-ref-not-echoed",
            "created_ts": 1_780_000_000.0,
            "reservation_reconcile_status": "AMBIGUOUS",
        }
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence.list_pending_idempotency = AsyncMock(return_value=[row])
        persistence.record_order = AsyncMock(return_value="oms-order-1")
        persistence.resolve_idempotency = AsyncMock(
            return_value=IntentResult("intent-known-order", IntentStatus.EXECUTED, order_id="ORD-KIS")
        )
        persistence.mark_idempotency_ambiguous = AsyncMock()
        mock_api.get_orders = MagicMock(return_value=pd.DataFrame(
            {
                "pdno": ["005930"],
                "ord_qty": [10],
                "psbl_qty": [10],
                "ord_unpr": [72000],
                "sll_buy_dvsn_cd": ["02"],
                "ord_tmd": ["093001"],
                "ord_dt": ["20260605"],
                "ord_gno_brno": ["001"],
            },
            index=["ORD-KIS"],
        ))
        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)

        reconciled = await oms.reconcile_stale_pending_idempotency(stale_after_sec=0)

        assert reconciled == 1
        persisted_order = persistence.record_order.await_args.args[0]
        assert persisted_order.order_id == "ORD-KIS"
        assert persisted_order.idempotency_key == "idem-known-order"
        persistence.resolve_idempotency.assert_awaited_once()
        assert persistence.resolve_idempotency.await_args.kwargs["order_id"] == "ORD-KIS"
        persistence.mark_idempotency_ambiguous.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_idempotency_reconciliation_rejects_adapter_order_with_unknown_side(self, mock_api):
        import pandas as pd

        row = {
            "intent_id": "intent-unknown-side",
            "idempotency_key": "idem-unknown-side",
            "strategy_id": "ALPHA",
            "symbol": "005930",
            "intent_type": "ENTER",
            "desired_qty": 10,
            "status": "DEFERRED",
            "order_id": "ORD-UNKNOWN-SIDE",
            "planned_side": "BUY",
            "planned_qty": 10,
            "planned_order_type": "LIMIT",
            "planned_limit_price": 72000.0,
            "reservation_reconcile_status": "AMBIGUOUS",
        }
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence.list_pending_idempotency = AsyncMock(return_value=[row])
        persistence.record_order = AsyncMock()
        persistence.resolve_idempotency = AsyncMock()
        persistence.mark_idempotency_ambiguous = AsyncMock()
        mock_api.get_orders = MagicMock(return_value=pd.DataFrame(
            {
                "pdno": ["005930"],
                "ord_qty": [10],
                "psbl_qty": [10],
                "ord_unpr": [72000],
                "sll_buy_dvsn_cd": ["99"],
                "ord_tmd": ["093001"],
                "ord_dt": ["20260605"],
            },
            index=["ORD-UNKNOWN-SIDE"],
        ))
        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)

        reconciled = await oms.reconcile_stale_pending_idempotency(stale_after_sec=0)

        assert reconciled == 0
        persistence.record_order.assert_not_awaited()
        persistence.resolve_idempotency.assert_not_awaited()
        persistence.mark_idempotency_ambiguous.assert_awaited_once()
        assert "order_id_candidate_mismatch:side" in persistence.mark_idempotency_ambiguous.await_args.kwargs["reason"]

    @pytest.mark.asyncio
    async def test_stale_idempotency_reconciliation_uses_price_and_time_to_choose_single_candidate(self, mock_api):
        row = {
            "intent_id": "intent-ambiguous",
            "idempotency_key": "idem-ambiguous",
            "strategy_id": "ALPHA",
            "symbol": "005930",
            "intent_type": "ENTER",
            "desired_qty": 10,
            "status": "PENDING",
            "planned_side": "BUY",
            "planned_qty": 10,
            "planned_order_type": "LIMIT",
            "planned_limit_price": 100.0,
            "submit_ref": "OMS-submit-ref",
            "created_ts": 1_780_000_000.0,
        }
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence.list_pending_idempotency = AsyncMock(return_value=[row])
        persistence.record_order = AsyncMock(return_value="oms-order-1")
        persistence.resolve_idempotency = AsyncMock(
            return_value=IntentResult("intent-ambiguous", IntentStatus.EXECUTED, order_id="ORD-MATCH")
        )
        persistence.mark_idempotency_ambiguous = AsyncMock()
        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)
        wrong_price = BrokerOrder(
            order_id="ORD-WRONG-PRICE",
            symbol="005930",
            side="BUY",
            qty=10,
            filled_qty=0,
            price=104.0,
            status="WORKING",
            created_at="",
            order_type="LIMIT",
            submit_ref="OMS-submit-ref",
        )
        wrong_price.created_ts = 1_780_000_005.0
        match = BrokerOrder(
            order_id="ORD-MATCH",
            symbol="005930",
            side="BUY",
            qty=10,
            filled_qty=0,
            price=100.0,
            status="WORKING",
            created_at="",
            order_type="LIMIT",
            submit_ref="OMS-submit-ref",
        )
        match.created_ts = 1_780_000_010.0
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[wrong_price, match]))

        reconciled = await oms.reconcile_stale_pending_idempotency(stale_after_sec=0)

        assert reconciled == 1
        assert persistence.record_order.await_args.args[0].order_id == "ORD-MATCH"
        persistence.resolve_idempotency.assert_awaited_once()
        persistence.mark_idempotency_ambiguous.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_with_broker_order_exists_reconciles_exact_candidate(self, mock_api):
        row = {
            "intent_id": "intent-timeout",
            "idempotency_key": "idem-timeout",
            "strategy_id": "ALPHA",
            "symbol": "005930",
            "intent_type": "ENTER",
            "desired_qty": 10,
            "status": "PENDING",
            "result_message": "Order status ambiguous after timeout",
            "planned_side": "BUY",
            "planned_qty": 10,
            "planned_order_type": "LIMIT",
            "planned_limit_price": 100.0,
            "submit_ref": "OMS-timeout-ref",
            "created_ts": 1_780_000_000.0,
            "reservation_reconcile_status": "SUBMITTING",
        }
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence.list_pending_idempotency = AsyncMock(return_value=[row])
        persistence.record_order = AsyncMock(return_value="oms-order-timeout")
        persistence.resolve_idempotency = AsyncMock(
            return_value=IntentResult("intent-timeout", IntentStatus.EXECUTED, order_id="ORD-TIMEOUT")
        )
        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)
        broker_order = BrokerOrder(
            order_id="ORD-TIMEOUT",
            symbol="005930",
            side="BUY",
            qty=10,
            filled_qty=0,
            price=100.0,
            status="WORKING",
            created_at="",
            order_type="LIMIT",
            submit_ref="OMS-timeout-ref",
        )
        broker_order.created_ts = 1_780_000_003.0
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[broker_order]))

        reconciled = await oms.reconcile_stale_pending_idempotency(stale_after_sec=0)

        assert reconciled == 1
        assert persistence.record_order.await_args.args[0].order_id == "ORD-TIMEOUT"
        persistence.resolve_idempotency.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stale_idempotency_reconciliation_rejects_sparse_single_candidate(self, mock_api):
        row = {
            "intent_id": "intent-sparse",
            "idempotency_key": "idem-sparse",
            "strategy_id": "ALPHA",
            "symbol": "005930",
            "intent_type": "ENTER",
            "desired_qty": 10,
            "status": "PENDING",
            "planned_side": "BUY",
            "planned_qty": 10,
            "planned_order_type": "LIMIT",
            "planned_limit_price": 100.0,
            "submit_ref": "OMS-sparse-ref",
            "created_ts": 1_780_000_000.0,
        }
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence.list_pending_idempotency = AsyncMock(return_value=[row])
        persistence.record_order = AsyncMock()
        persistence.resolve_idempotency = AsyncMock()
        persistence.mark_idempotency_ambiguous = AsyncMock()
        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)
        sparse = BrokerOrder(
            order_id="ORD-SPARSE",
            symbol="005930",
            side="BUY",
            qty=10,
            filled_qty=0,
            price=0.0,
            status="WORKING",
            created_at="",
        )
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[sparse]))

        reconciled = await oms.reconcile_stale_pending_idempotency(stale_after_sec=0)

        assert reconciled == 0
        persistence.record_order.assert_not_awaited()
        persistence.resolve_idempotency.assert_not_awaited()
        persistence.mark_idempotency_ambiguous.assert_awaited_once()
        assert persistence.mark_idempotency_ambiguous.await_args.args[0] == "idem-sparse"
        assert "broker_candidates_do_not_match_plan" in persistence.mark_idempotency_ambiguous.await_args.kwargs["reason"]

    @pytest.mark.asyncio
    async def test_stale_idempotency_reconciliation_leaves_multiple_exact_candidates_ambiguous(self, mock_api):
        row = {
            "intent_id": "intent-ambiguous",
            "idempotency_key": "idem-ambiguous",
            "strategy_id": "ALPHA",
            "symbol": "005930",
            "intent_type": "ENTER",
            "desired_qty": 10,
            "status": "PENDING",
            "planned_side": "BUY",
            "planned_qty": 10,
            "planned_order_type": "LIMIT",
            "planned_limit_price": 100.0,
            "submit_ref": "OMS-submit-ref",
            "created_ts": 1_780_000_000.0,
        }
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence.list_pending_idempotency = AsyncMock(return_value=[row])
        persistence.record_order = AsyncMock()
        persistence.resolve_idempotency = AsyncMock()
        persistence.mark_idempotency_ambiguous = AsyncMock()
        oms = OMSCore(mock_api, persistence=persistence, require_persistence=True)
        first = BrokerOrder(
            order_id="ORD-1",
            symbol="005930",
            side="BUY",
            qty=10,
            filled_qty=0,
            price=100.0,
            status="WORKING",
            created_at="",
            order_type="LIMIT",
            submit_ref="OMS-submit-ref",
        )
        first.created_ts = 1_780_000_005.0
        second = BrokerOrder(
            order_id="ORD-2",
            symbol="005930",
            side="BUY",
            qty=10,
            filled_qty=0,
            price=100.0,
            status="WORKING",
            created_at="",
            order_type="LIMIT",
            submit_ref="OMS-submit-ref",
        )
        second.created_ts = 1_780_000_006.0
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[first, second]))

        reconciled = await oms.reconcile_stale_pending_idempotency(stale_after_sec=0)

        assert reconciled == 0
        persistence.record_order.assert_not_awaited()
        persistence.resolve_idempotency.assert_not_awaited()
        persistence.mark_idempotency_ambiguous.assert_awaited_once()
        assert persistence.mark_idempotency_ambiguous.await_args.args[0] == "idem-ambiguous"
        assert "multiple_exact_candidates" in persistence.mark_idempotency_ambiguous.await_args.kwargs["reason"]


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
            strategy_id="ALPHA",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.DEFERRED
        assert "safe mode" in result.message.lower()

    @pytest.mark.asyncio
    async def test_risk_rejection_emits_pre_order_lifecycle_event(self, oms):
        """Risk blocks before working-order creation still emit order evidence."""
        emitter = _CapturingOMSEmitter()
        oms.event_emitter = emitter
        oms.risk.safe_mode = True
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="ALPHA",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
            metadata={"provisional_order_ref": "ALPHA:event:action:0"},
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.DEFERRED
        assert emitter.order_events
        order, event_type, payload, event_intent = emitter.order_events[-1]
        assert event_type == "ORDER_DEFERRED"
        assert payload["status_after"] == "DEFERRED"
        assert payload["pre_working_order"] is True
        assert order.order_id == "ALPHA:event:action:0"
        assert order.intent_id == intent.intent_id
        assert event_intent is intent

    @pytest.mark.asyncio
    async def test_arbitration_defer(self, oms):
        """Test arbitration deferral."""
        # Set up entry lock by another strategy
        now = time.time()
        oms.state.set_entry_lock("005930", "BETA", now + 60)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="ALPHA",
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
            strategy_id="ALPHA",
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
            strategy_id="ALPHA",
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
    async def test_synthetic_stop_entry_is_accepted_without_immediate_broker_order(self, oms):
        """Synthetic stops are trigger intents, not immediate KIS stop-limit submissions."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="DELTA",
            symbol="005930",
            desired_qty=100,
            constraints=IntentConstraints(stop_price=72100, execution_style="SYNTHETIC_STOP"),
            risk_payload=RiskPayload(entry_px=72100, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.ACCEPTED
        assert result.order_id is None
        assert not oms.state.get_position("005930").has_working_orders()

    @pytest.mark.asyncio
    async def test_exit_without_allocation_rejected(self, oms):
        """Test EXIT without allocation is rejected."""
        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="ALPHA",
            symbol="005930",
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.REJECTED
        assert "no allocation" in result.message.lower()

    @pytest.mark.asyncio
    async def test_exit_with_allocation_succeeds(self, oms):
        """Test EXIT with allocation succeeds."""
        # Set up allocation and real broker position
        oms.state.update_allocation("005930", "ALPHA", 100)
        oms.state.update_position("005930", real_qty=100)

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="ALPHA",
            symbol="005930",
            risk_payload=RiskPayload(rationale_code="stop_hit"),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED

    @pytest.mark.asyncio
    async def test_reduce_creates_sell_order(self, oms):
        """Test REDUCE creates sell order."""
        # Set up allocation and real broker position
        oms.state.update_allocation("005930", "ALPHA", 100)
        oms.state.update_position("005930", real_qty=100)

        intent = Intent(
            intent_type=IntentType.REDUCE,
            strategy_id="ALPHA",
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
            strategy_id="ALPHA",
        )

        await oms._apply_fill(wo, 100)

        pos = oms.state.get_position("005930")
        alloc = pos.allocations.get("ALPHA")
        assert alloc is not None
        assert alloc.qty == 100

    @pytest.mark.asyncio
    async def test_sell_fill_reduces_allocation(self, oms):
        """Test sell fill reduces allocation."""
        from oms.state import WorkingOrder

        # Set up initial allocation and real broker position
        oms.state.update_allocation("005930", "ALPHA", 100)
        oms.state.update_position("005930", real_qty=100)

        wo = WorkingOrder(
            order_id="ORD001",
            symbol="005930",
            side="SELL",
            qty=50,
            price=72000,
            strategy_id="ALPHA",
        )

        await oms._apply_fill(wo, 50)

        pos = oms.state.get_position("005930")
        assert pos.allocations["ALPHA"].qty == 50


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
        oms.state.update_allocation("005930", "ALPHA", 100)
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
        oms.state.update_allocation("005930", "ALPHA", 100)

        # But order is in flight
        wo = WorkingOrder(
            order_id="ORD001",
            symbol="005930",
            side="BUY",
            qty=50,
            strategy_id="ALPHA",
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
        oms.state.update_allocation("005930", "ALPHA", 100)

        alloc = oms.get_allocation("005930", "ALPHA")

        assert alloc == 100

    def test_get_allocation_missing(self, oms):
        """Test get_allocation returns 0 for missing."""
        alloc = oms.get_allocation("005930", "ALPHA")
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
            strategy_id="ALPHA",
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
        alloc = pos.allocations.get("ALPHA")
        assert alloc is not None
        assert alloc.qty == 100
        assert wo.filled_qty == 100
        assert wo.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_broker_fill_price_overrides_planned_working_price(self, oms):
        """Broker-reported execution price should drive cost basis on fill."""
        from oms.state import WorkingOrder, OrderStatus
        from oms.adapter import BrokerOrder

        wo = WorkingOrder(
            order_id="ORDMKT",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=0.0,
            order_type="MARKET",
            strategy_id="ALPHA",
            status=OrderStatus.WORKING,
        )
        oms.state.add_working_order("005930", wo)
        oms.adapter.get_orders = AsyncMock(
            return_value=BrokerQueryResult(
                ok=True,
                data=[
                    BrokerOrder(
                        order_id="ORDMKT",
                        symbol="005930",
                        side="BUY",
                        qty=100,
                        filled_qty=100,
                        price=72550,
                        status="FILLED",
                        created_at="09:30:00",
                    )
                ],
            )
        )

        await oms._sync_working_orders()

        alloc = oms.state.get_position("005930").allocations["ALPHA"]
        assert alloc.cost_basis == 72550
        assert wo.price == 72550

    @pytest.mark.asyncio
    async def test_broker_order_disappeared_waits_for_position_reconcile(self, oms):
        """Missing broker orders stay pending until position sync clarifies the outcome."""
        from oms.state import WorkingOrder, OrderStatus

        wo = WorkingOrder(
            order_id="ORD002",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=0,
            price=72000,
            strategy_id="ALPHA",
            status=OrderStatus.WORKING,
        )
        oms.state.add_working_order("005930", wo)

        # Broker returns empty list (order disappeared)
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))

        await oms._sync_working_orders()

        # Order should remain until broker positions confirm fill vs cancel
        pos = oms.state.get_position("005930")
        assert pos.has_working_orders()
        assert wo.status == OrderStatus.WORKING
        assert wo.missing_from_broker_count == 1

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
            strategy_id="ALPHA",
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
            strategy_id="ALPHA",
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
            strategy_id="ALPHA",
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
            strategy_id="ALPHA",
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
        alloc = pos.allocations.get("ALPHA")
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

        # Cycle 1 (not multiple of 6) ??should NOT call get_buyable_cash
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

    @pytest.mark.asyncio
    async def test_missing_broker_order_infers_fill_from_position_delta(self, oms):
        """A pending order disappearing from KIS should credit fills before drift handling."""
        from oms.state import WorkingOrder, OrderStatus
        from oms.adapter import BrokerPosition

        wo = WorkingOrder(
            order_id="ORD013",
            symbol="005930",
            side="BUY",
            qty=100,
            price=72000,
            strategy_id="ALPHA",
            status=OrderStatus.WORKING,
        )
        oms.state.add_working_order("005930", wo)

        broker_pos = BrokerPosition(
            symbol="005930",
            qty=100,
            avg_price=72000,
            current_price=72000,
            pnl=0.0,
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
        assert pos.allocations["ALPHA"].qty == 100
        assert not pos.has_working_orders()
        assert pos.frozen is False
        assert wo.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_missing_broker_symbol_is_zeroed_out(self, oms):
        """Symbols absent from the broker snapshot should be reset to flat."""
        oms.state.update_position("005930", real_qty=100, avg_price=70000)

        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))
        oms.adapter.get_balance_snapshot = AsyncMock(return_value=(
            BrokerQueryResult(ok=True, data=[]),
            100_000_000,
        ))
        oms.adapter.get_buyable_cash = AsyncMock(return_value=50_000_000)

        await oms._reconcile(cycle_count=0)

        pos = oms.state.get_position("005930")
        assert pos.real_qty == 0
        assert pos.avg_price == 0.0

    @pytest.mark.asyncio
    async def test_missing_broker_order_cancels_after_grace_cycles(self, oms):
        """Orders missing from KIS with no position delta should eventually cancel."""
        from oms.state import WorkingOrder, OrderStatus

        wo = WorkingOrder(
            order_id="ORD014",
            symbol="005930",
            side="BUY",
            qty=100,
            price=72000,
            strategy_id="ALPHA",
            status=OrderStatus.WORKING,
        )
        oms.state.add_working_order("005930", wo)

        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))
        oms.adapter.get_balance_snapshot = AsyncMock(return_value=(
            BrokerQueryResult(ok=True, data=[]),
            100_000_000,
        ))
        oms.adapter.get_buyable_cash = AsyncMock(return_value=50_000_000)

        await oms._reconcile(cycle_count=0)
        assert oms.state.get_position("005930").has_working_orders()

        await oms._reconcile(cycle_count=1)

        pos = oms.state.get_position("005930")
        assert not pos.has_working_orders()
        assert wo.status == OrderStatus.CANCELLED


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
        oms.state.update_allocation("005930", "ALPHA", 100, cost_basis=72000)

        intent = Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id="ALPHA",
            symbol="005930",
            risk_payload=RiskPayload(stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        pos = oms.state.get_position("005930")
        alloc = pos.allocations["ALPHA"]
        assert alloc.soft_stop_px == 71000

    @pytest.mark.asyncio
    async def test_updates_hard_stop_px(self, oms):
        """Test MODIFY_RISK updates hard_stop_px on position."""
        # Set up existing allocation
        oms.state.update_allocation("005930", "ALPHA", 100, cost_basis=72000)

        intent = Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id="ALPHA",
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
            strategy_id="ALPHA",
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
        oms.state.update_allocation("005930", "ALPHA", 100, cost_basis=70000)

        wo = WorkingOrder(
            order_id="ORD020",
            symbol="005930",
            side="SELL",
            qty=50,
            price=72000,  # Sell at 72000, cost basis 70000
            strategy_id="ALPHA",
        )

        await oms._apply_fill(wo, 50)

        # Realized PnL = (72000 - 70000) * 50 = 100,000
        assert oms.state.daily_realized_pnl == 100_000

        # Allocation should be reduced
        pos = oms.state.get_position("005930")
        assert pos.allocations["ALPHA"].qty == 50


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
            strategy_id="ALPHA",
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
        oms.state.update_allocation("005930", "ALPHA", 100, cost_basis=72000)
        oms.state.update_position("005930", real_qty=100)

        intent = Intent(
            intent_type=IntentType.SET_TARGET,
            strategy_id="ALPHA",
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
        oms.state.update_allocation("005930", "ALPHA", 100, cost_basis=72000)

        intent = Intent(
            intent_type=IntentType.SET_TARGET,
            strategy_id="ALPHA",
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
        mock_persistence.load_daily_realized_pnl = AsyncMock(return_value={})
        mock_persistence.close = AsyncMock()

        oms = OMSCore(mock_api, persistence=mock_persistence)

        # Patch start_reconciliation_loop and _reconcile to avoid background task/API
        oms.start_reconciliation_loop = AsyncMock()
        oms._reconcile = AsyncMock()

        await oms.start()

        mock_persistence.connect.assert_awaited_once()
        mock_persistence.load_positions.assert_awaited_once()
        mock_persistence.load_allocations.assert_awaited_once()
        mock_persistence.load_working_orders.assert_awaited_once()
        mock_persistence.load_oms_state.assert_awaited_once()
        oms._reconcile.assert_awaited_once_with(cycle_count=0)
        oms.start_reconciliation_loop.assert_awaited_once()

        # Cleanup
        await oms.shutdown()

    @pytest.mark.asyncio
    async def test_start_hydrates_idempotency_results_from_persistence(self, mock_api):
        """Accepted/executed intents should dedupe after OMS restart."""
        mock_persistence = MagicMock()
        mock_persistence.connect = AsyncMock()
        mock_persistence.load_positions = AsyncMock(return_value={})
        mock_persistence.load_allocations = AsyncMock(return_value={})
        mock_persistence.load_working_orders = AsyncMock(return_value=[])
        mock_persistence.load_idempotency_results = AsyncMock(
            return_value={
                "KALCB:005930:ENTER:20260604:abc:10": IntentResult(
                    intent_id="intent-1",
                    status=IntentStatus.EXECUTED,
                    order_id="ORD1",
                )
            }
        )
        mock_persistence.load_oms_state = AsyncMock(return_value=None)
        mock_persistence.load_daily_realized_pnl = AsyncMock(return_value={})
        mock_persistence.close = AsyncMock()

        oms = OMSCore(mock_api, persistence=mock_persistence)
        oms.start_reconciliation_loop = AsyncMock()
        oms._reconcile = AsyncMock()

        await oms.start()

        assert oms._idem.get("KALCB:005930:ENTER:20260604:abc:10").order_id == "ORD1"
        mock_persistence.load_idempotency_results.assert_awaited_once()
        await oms.shutdown()

    @pytest.mark.asyncio
    async def test_start_requires_persistence_when_configured(self, mock_api):
        """Paper/live mode can block startup when durable persistence is unavailable."""
        mock_persistence = MagicMock()
        mock_persistence.connect = AsyncMock()
        mock_persistence._is_connected = MagicMock(return_value=False)

        oms = OMSCore(mock_api, persistence=mock_persistence, require_persistence=True)

        with pytest.raises(RuntimeError, match="persistence is required"):
            await oms.start()

    @pytest.mark.asyncio
    async def test_start_without_persistence(self, mock_api):
        """Test start() works without persistence configured."""
        oms = OMSCore(mock_api, persistence=None)

        # Patch start_reconciliation_loop to avoid background task
        oms.start_reconciliation_loop = AsyncMock()
        # Patch _reconcile to avoid real API calls
        oms._reconcile = AsyncMock()

        await oms.start()

        oms._reconcile.assert_awaited_once_with(cycle_count=0)
        oms.start_reconciliation_loop.assert_awaited_once()

        # Cleanup
        await oms.shutdown()

    @pytest.mark.asyncio
    async def test_start_runs_initial_reconcile_before_loop(self, mock_api):
        """Test start() runs _reconcile(0) before starting the loop, loading equity."""
        from oms.adapter import BrokerPosition, BrokerQueryResult

        oms = OMSCore(mock_api, persistence=None)
        assert oms.state.equity == 0.0  # starts at zero

        # Mock adapter to return equity=53M
        broker_pos = BrokerPosition(
            symbol="005930", qty=100, avg_price=70000,
            current_price=72000, pnl=2.86,
        )
        oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))
        oms.adapter.get_balance_snapshot = AsyncMock(return_value=(
            BrokerQueryResult(ok=True, data=[broker_pos]),
            53_000_000,
        ))
        oms.adapter.get_buyable_cash = AsyncMock(return_value=30_000_000)

        # Patch loop to avoid background task
        oms.start_reconciliation_loop = AsyncMock()

        await oms.start()

        # Equity should be loaded from the initial reconcile
        assert oms.state.equity == 53_000_000

        await oms.shutdown()

    @pytest.mark.asyncio
    async def test_start_continues_if_initial_reconcile_fails(self, mock_api):
        """Test start() still starts the loop even if initial reconcile fails."""
        oms = OMSCore(mock_api, persistence=None)

        # Make _reconcile raise an exception
        oms._reconcile = AsyncMock(side_effect=Exception("KIS unavailable"))
        oms.start_reconciliation_loop = AsyncMock()

        await oms.start()

        # Loop should still start despite initial reconcile failure
        oms.start_reconciliation_loop.assert_awaited_once()
        assert oms.state.equity == 0.0  # not loaded

        await oms.shutdown()


class TestExitQtyCappedAtRealQty:
    """Tests for Fix 1: sell qty capped at real_qty."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        return oms

    @pytest.mark.asyncio
    async def test_exit_capped_at_real_qty(self, oms):
        """EXIT with alloc=19, real_qty=9 should sell 9 (not reject)."""
        oms.state.update_allocation("005930", "PCIM", 19)
        oms.state.update_position("005930", real_qty=9)

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="PCIM",
            symbol="005930",
            risk_payload=RiskPayload(rationale_code="signal_exit"),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        # Working order should be for 9 shares, not 19
        pos = oms.state.get_position("005930")
        assert pos.has_working_orders()
        wo = pos.working_orders[0]
        assert wo.qty == 9
        assert wo.side == "SELL"

    @pytest.mark.asyncio
    async def test_exit_rejected_when_real_qty_zero(self, oms):
        """EXIT with alloc=19, real_qty=0 should be REJECTED."""
        oms.state.update_allocation("005930", "PCIM", 19)
        oms.state.update_position("005930", real_qty=0)

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="PCIM",
            symbol="005930",
            risk_payload=RiskPayload(rationale_code="signal_exit"),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.REJECTED
        assert "no sellable shares" in result.message.lower()

    @pytest.mark.asyncio
    async def test_reduce_capped_at_real_qty(self, oms):
        """REDUCE with reduce_qty=50, real_qty=30 should sell 30."""
        oms.state.update_allocation("005930", "ALPHA", 100)
        oms.state.update_position("005930", real_qty=30)

        intent = Intent(
            intent_type=IntentType.REDUCE,
            strategy_id="ALPHA",
            symbol="005930",
            desired_qty=50,
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        pos = oms.state.get_position("005930")
        assert pos.has_working_orders()
        wo = pos.working_orders[0]
        assert wo.qty == 30
        assert wo.side == "SELL"

    @pytest.mark.asyncio
    async def test_set_target_sell_capped_at_real_qty(self, oms):
        """SET_TARGET sell with delta=50, real_qty=20 should sell 20."""
        oms.state.update_allocation("005930", "ALPHA", 100, cost_basis=72000)
        oms.state.update_position("005930", real_qty=20)

        intent = Intent(
            intent_type=IntentType.SET_TARGET,
            strategy_id="ALPHA",
            symbol="005930",
            target_qty=50,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await oms.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        pos = oms.state.get_position("005930")
        assert pos.has_working_orders()
        wo = pos.working_orders[0]
        assert wo.qty == 20
        assert wo.side == "SELL"


class TestNegativeDriftHandling:
    """Tests for Fix 2: drift auto-correction and spam prevention."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    @pytest.mark.asyncio
    async def test_negative_drift_single_strategy_auto_corrects(self, oms):
        """Single-strategy negative drift should auto-correct allocation."""
        oms.state.update_position("005930", real_qty=9)
        oms.state.update_allocation("005930", "PCIM", 19)
        # Drift = 9 - 19 = -10

        await oms._check_allocation_drift()

        pos = oms.state.get_position("005930")
        assert pos.allocations["PCIM"].qty == 9  # auto-corrected
        assert pos.frozen is False  # NOT frozen ??drift resolved

    @pytest.mark.asyncio
    async def test_negative_drift_multi_strategy_freezes(self, oms):
        """Multi-strategy negative drift should freeze but not correct."""
        oms.state.update_position("005930", real_qty=20)
        oms.state.update_allocation("005930", "ALPHA", 15)
        oms.state.update_allocation("005930", "PCIM", 10)
        # Drift = 20 - 25 = -5

        await oms._check_allocation_drift()

        pos = oms.state.get_position("005930")
        assert pos.frozen is True
        # Neither allocation was modified
        assert pos.allocations["ALPHA"].qty == 15
        assert pos.allocations["PCIM"].qty == 10

    @pytest.mark.asyncio
    async def test_frozen_symbol_not_re_logged(self, oms):
        """Frozen symbol should not be re-processed on subsequent cycles."""
        oms.state.update_position("005930", real_qty=20)
        oms.state.update_allocation("005930", "ALPHA", 15)
        oms.state.update_allocation("005930", "PCIM", 10)

        # First cycle: freezes
        await oms._check_allocation_drift()
        pos = oms.state.get_position("005930")
        assert pos.frozen is True

        # Mock persistence to verify no re-logging
        mock_persistence = MagicMock()
        mock_persistence.log_recon = AsyncMock()
        oms.persistence = mock_persistence

        # Second cycle: should skip because frozen
        await oms._check_allocation_drift()

        mock_persistence.log_recon.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_corrected_symbol_not_frozen(self, oms):
        """After single-strategy auto-correction, drift=0 means not frozen."""
        oms.state.update_position("005930", real_qty=9)
        oms.state.update_allocation("005930", "PCIM", 19)

        await oms._check_allocation_drift()

        pos = oms.state.get_position("005930")
        # Drift should be 0 now (9 - 9 = 0)
        assert pos.allocation_drift() == 0
        assert pos.frozen is False

    @pytest.mark.asyncio
    async def test_auto_correct_freezes_if_unknown_remains(self, oms):
        """Auto-correction with leftover _UNKNOWN_ allocation freezes to prevent spam."""
        oms.state.update_position("005930", real_qty=5)
        oms.state.update_allocation("005930", "PCIM", 10)
        # Simulate leftover _UNKNOWN_ from a previous positive drift
        pos = oms.state.get_position("005930")
        pos.allocations["_UNKNOWN_"] = StrategyAllocation(strategy_id="_UNKNOWN_", qty=5)
        # Drift = 5 - 15 = -10; non_unknown = {PCIM: 10} (single)
        # Auto-corrects PCIM to 5, but _UNKNOWN_=5 remains ??drift = 5 - 10 = -5

        await oms._check_allocation_drift()

        assert pos.allocations["PCIM"].qty == 5  # auto-corrected
        assert pos.frozen is True  # frozen because drift persists

        # Second cycle should NOT re-process (frozen guard)
        mock_persistence = MagicMock()
        mock_persistence.log_recon = AsyncMock()
        oms.persistence = mock_persistence

        await oms._check_allocation_drift()
        mock_persistence.log_recon.assert_not_called()

    @pytest.mark.asyncio
    async def test_frozen_single_strategy_negative_drift_self_heals_when_unknown_cleared(self, oms):
        """Frozen stale drift should self-heal once only one strategy remains and _UNKNOWN_ is cleared."""
        oms.state.update_position("068270", real_qty=7)
        oms.state.update_allocation("068270", "GAMMA", 31)
        pos = oms.state.get_position("068270")
        pos.allocations["_UNKNOWN_"] = StrategyAllocation(strategy_id="_UNKNOWN_", qty=0)
        pos.frozen = True

        await oms._check_allocation_drift()

        assert pos.allocations["GAMMA"].qty == 7
        assert pos.frozen is False
        assert "_UNKNOWN_" not in pos.allocations


class TestZeroPositionCleanup:
    """Tests for zero-position auto-cleanup of stale allocations."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    @pytest.mark.asyncio
    async def test_zero_position_unknown_only_auto_clears(self, oms):
        """real_qty=0 with only _UNKNOWN_ allocation should clear and unfreeze."""
        oms.state.update_position("259960", real_qty=0)
        pos = oms.state.get_position("259960")
        pos.allocations["_UNKNOWN_"] = StrategyAllocation(strategy_id="_UNKNOWN_", qty=2)
        pos.frozen = True

        await oms._check_allocation_drift()

        assert len(pos.allocations) == 0
        assert pos.frozen is False

    @pytest.mark.asyncio
    async def test_zero_position_strategy_alloc_auto_clears(self, oms):
        """real_qty=0 with a strategy allocation should clear it."""
        oms.state.update_position("005930", real_qty=0)
        oms.state.update_allocation("005930", "PCIM", 10)

        await oms._check_allocation_drift()

        pos = oms.state.get_position("005930")
        assert len(pos.allocations) == 0
        assert pos.frozen is False

    @pytest.mark.asyncio
    async def test_zero_position_mixed_allocs_auto_clears(self, oms):
        """real_qty=0 with mixed _UNKNOWN_ + strategy allocations should clear all."""
        oms.state.update_position("068270", real_qty=0)
        oms.state.update_allocation("068270", "BETA", 10)
        pos = oms.state.get_position("068270")
        pos.allocations["_UNKNOWN_"] = StrategyAllocation(strategy_id="_UNKNOWN_", qty=5)
        pos.frozen = True

        await oms._check_allocation_drift()

        assert len(pos.allocations) == 0
        assert pos.frozen is False

    @pytest.mark.asyncio
    async def test_zero_position_with_working_orders_skipped(self, oms):
        """real_qty=0 with working orders should NOT be cleaned up."""
        from oms.state import WorkingOrder

        oms.state.update_position("009150", real_qty=0)
        pos = oms.state.get_position("009150")
        pos.allocations["_UNKNOWN_"] = StrategyAllocation(strategy_id="_UNKNOWN_", qty=2)
        pos.frozen = True
        pos.working_orders.append(
            WorkingOrder(
                order_id="ORD999",
                symbol="009150",
                side="BUY",
                qty=2,
                price=5000,
                strategy_id="_UNKNOWN_",
            )
        )

        await oms._check_allocation_drift()

        # Should be skipped ??working orders guard runs first
        assert pos.allocations["_UNKNOWN_"].qty == 2
        assert pos.frozen is True


class TestAdminCorrectAllocation:
    """Tests for Fix 3: admin allocation correction."""

    @pytest.fixture
    def mock_api(self, mock_kis_api):
        return mock_kis_api

    @pytest.fixture
    def oms(self, mock_api):
        oms = OMSCore(mock_api)
        oms.state.equity = 100_000_000
        return oms

    @pytest.mark.asyncio
    async def test_admin_correct_allocation(self, oms):
        """Admin correction sets allocation and unfreezes if drift resolved."""
        oms.state.update_position("005930", real_qty=9)
        oms.state.update_allocation("005930", "PCIM", 19)
        pos = oms.state.get_position("005930")
        pos.frozen = True

        result = await oms.correct_allocation("005930", "PCIM", 9)

        assert result["old_qty"] == 19
        assert result["new_qty"] == 9
        assert result["drift"] == 0
        assert result["frozen"] is False
        assert pos.allocations["PCIM"].qty == 9

    @pytest.mark.asyncio
    async def test_admin_correct_keeps_frozen_if_drift_remains(self, oms):
        """Admin correction that doesn't fully resolve drift keeps symbol frozen."""
        oms.state.update_position("005930", real_qty=9)
        oms.state.update_allocation("005930", "PCIM", 19)
        oms.state.update_allocation("005930", "ALPHA", 5)
        pos = oms.state.get_position("005930")
        pos.frozen = True

        # Correct PCIM to 9, but ALPHA still has 5 ??total=14, real=9 ??drift=-5
        result = await oms.correct_allocation("005930", "PCIM", 9)

        assert result["frozen"] is True  # still frozen ??drift remains


class _StopPersistence:
    oms_id = "primary"

    def __init__(self):
        self.stops = []
        self.events = []

    async def upsert_stop(self, stop):
        self.stops.append(stop)
        return stop

    async def update_stop_quantity(self, strategy_id, symbol, qty):
        if not self.stops:
            return None
        stop = self.stops[-1]
        previous_status = stop.status
        stop.qty = max(int(qty or 0), 0)
        if previous_status in {"TRIGGERED", "TRIGGERED_PENDING_EXECUTION", "EXIT_SUBMITTED"}:
            stop.status = previous_status
        else:
            stop.status = "CANCELLED" if stop.qty <= 0 else "ACTIVE"
        return stop

    async def load_active_stops(self):
        return []

    async def load_stop_for_allocation(self, strategy_id, symbol):
        return self.stops[-1] if self.stops else None

    async def mark_filled(self, stop_id):
        if self.stops:
            self.stops[-1].status = "FILLED"

    async def mark_cancelled(self, stop_id, reason=""):
        if self.stops:
            self.stops[-1].status = "CANCELLED"

    async def record_order_event(self, *args, **kwargs):
        self.events.append((args, kwargs))

    async def record_intent(self, *args, **kwargs):
        return None

    async def sync_allocation(self, *args, **kwargs):
        return None

    async def record_fill(self, *args, **kwargs):
        return None

    async def open_trade(self, *args, **kwargs):
        return "trade-1"

    async def find_open_trade(self, *args, **kwargs):
        return "trade-1"

    async def close_trade(self, *args, **kwargs):
        return None


class _DefaultWatcherPersistence:
    oms_id = "primary"

    def __init__(self, stop):
        self.stop = stop
        self.exit_submitted = None
        self.events = []

    def _is_connected(self):
        return True

    async def load_active_stops(self):
        return [self.stop] if self.stop.status in {"ACTIVE", "TRIGGERED_PENDING_EXECUTION"} else []

    async def touch_stop_check(self, stop_id, *, checked_at, last_price, last_error=None):
        self.stop.last_checked_at = checked_at
        self.stop.last_price = last_price
        self.stop.last_error = last_error

    async def mark_triggered(self, stop_id, trigger_price, triggered_at):
        self.stop.status = StopStatus.TRIGGERED_PENDING_EXECUTION.value
        self.stop.triggered_at = triggered_at
        self.stop.last_price = trigger_price
        return True

    async def mark_exit_submitted(self, stop_id, exit_intent_id, order_id, idempotency_key=None):
        self.stop.status = StopStatus.EXIT_SUBMITTED.value
        self.stop.exit_intent_id = exit_intent_id
        self.stop.broker_order_id = order_id
        self.stop.idempotency_key = idempotency_key
        self.exit_submitted = (exit_intent_id, order_id, idempotency_key)

    async def reserve_intent(self, intent):
        return None

    async def update_intent_submission_plan(self, *args, **kwargs):
        return None

    async def record_order(self, *args, **kwargs):
        return "oms-order-stop"

    async def record_order_event(self, *args, **kwargs):
        self.events.append((args, kwargs))

    async def record_intent(self, *args, **kwargs):
        return None


class _TerminalCollisionStopPersistence(_StopPersistence):
    def __init__(self, terminal_stop):
        super().__init__()
        self.by_id = {terminal_stop.stop_id: terminal_stop}
        self.stops = [terminal_stop]

    async def upsert_stop(self, stop):
        existing = self.by_id.get(stop.stop_id)
        if existing is not None:
            return existing
        self.by_id[stop.stop_id] = stop
        self.stops.append(stop)
        return stop

    async def load_stop_for_allocation(self, strategy_id, symbol):
        for stop in reversed(self.stops):
            if stop.strategy_id == strategy_id and stop.symbol == str(symbol).zfill(6) and stop.status in {
                "PENDING",
                "ACTIVE",
                "TRIGGERED_PENDING_EXECUTION",
                "EXIT_SUBMITTED",
            }:
                return stop
        return None


class TestDurableProtectiveStopIntegration:
    @pytest.fixture
    def oms(self, mock_kis_api):
        persistence = _StopPersistence()
        oms = OMSCore(mock_kis_api, persistence=persistence)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        return oms

    @pytest.mark.asyncio
    async def test_buy_fill_with_explicit_stop_creates_durable_stop(self, oms):
        from oms.state import WorkingOrder

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KALCB",
            symbol="005930",
            desired_qty=10,
            risk_payload=RiskPayload(entry_px=100.0, stop_px=95.0),
        )
        order = WorkingOrder(
            order_id="ORD-ENTRY",
            symbol="005930",
            side="BUY",
            qty=10,
            price=100.0,
            strategy_id="KALCB",
            intent_id=intent.intent_id,
            idempotency_key=intent.idempotency_key,
        )

        await oms._apply_fill(order, 10, intent=intent)

        stop = oms.persistence.stops[-1]
        assert stop.strategy_id == "KALCB"
        assert stop.symbol == "005930"
        assert stop.qty == 10
        assert stop.stop_price == 95.0
        assert stop.status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_reconciliation_buy_fill_uses_working_order_stop_metadata_without_intent(self, oms):
        from oms.state import WorkingOrder

        order = WorkingOrder(
            order_id="ORD-ENTRY",
            symbol="005930",
            side="BUY",
            qty=10,
            price=100.0,
            strategy_id="KALCB",
            intent_id="entry-intent",
            idempotency_key="entry-key",
            risk_stop_px=96.0,
            risk_hard_stop_px=95.0,
        )

        await oms._apply_fill(order, 10)

        pos = oms.state.get_position("005930")
        alloc = pos.allocations["KALCB"]
        stop = oms.persistence.stops[-1]
        assert alloc.soft_stop_px == 96.0
        assert pos.hard_stop_px == 95.0
        assert stop.entry_intent_id == "entry-intent"
        assert stop.entry_order_id == "ORD-ENTRY"
        assert stop.stop_price == 95.0
        assert stop.source_metadata["risk_stop_px"] == 96.0
        assert stop.source_metadata["risk_hard_stop_px"] == 95.0

    @pytest.mark.asyncio
    async def test_modify_risk_updates_existing_durable_stop(self, oms):
        oms.state.update_position("005930", real_qty=10)
        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)

        intent = Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id="KALCB",
            symbol="005930",
            risk_payload=RiskPayload(stop_px=93.0),
        )

        result = await oms._handle_modify_risk(intent)

        assert result.status == IntentStatus.EXECUTED
        assert oms.persistence.stops[-1].stop_price == 93.0
        assert oms.persistence.stops[-1].qty == 10

    @pytest.mark.asyncio
    async def test_exit_fill_adjusts_durable_stop_quantity(self, oms):
        from oms.state import WorkingOrder

        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)
        await oms._upsert_durable_stop(
            symbol="005930",
            strategy_id="KALCB",
            qty=10,
            stop_price=95.0,
            entry_intent_id=None,
            entry_order_id="ORD-ENTRY",
            source_metadata={"source": "unit"},
            event_type="STOP_CREATED",
        )
        sell = WorkingOrder(order_id="ORD-SELL", symbol="005930", side="SELL", qty=4, price=101.0, strategy_id="KALCB")

        await oms._apply_fill(sell, 4)

        assert oms.persistence.stops[-1].qty == 6
        assert oms.persistence.stops[-1].status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_full_non_stop_exit_fill_cancels_durable_stop(self, oms):
        from oms.state import WorkingOrder

        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)
        await oms._upsert_durable_stop(
            symbol="005930",
            strategy_id="KALCB",
            qty=10,
            stop_price=95.0,
            entry_intent_id=None,
            entry_order_id="ORD-ENTRY",
            source_metadata={"source": "unit"},
            event_type="STOP_CREATED",
        )
        sell = WorkingOrder(order_id="ORD-SELL", symbol="005930", side="SELL", qty=10, price=101.0, strategy_id="KALCB")

        await oms._apply_fill(sell, 10)

        assert oms.persistence.stops[-1].qty == 0
        assert oms.persistence.stops[-1].status == StopStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_partial_stop_exit_fill_preserves_exit_submitted_stop_status(self, oms):
        from oms.state import WorkingOrder

        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)
        stop = await oms._upsert_durable_stop(
            symbol="005930",
            strategy_id="KALCB",
            qty=10,
            stop_price=95.0,
            entry_intent_id=None,
            entry_order_id="ORD-ENTRY",
            source_metadata={"source": "unit"},
            event_type="STOP_CREATED",
        )
        stop.status = StopStatus.EXIT_SUBMITTED.value
        sell = WorkingOrder(
            order_id="ORD-STOP-SELL",
            symbol="005930",
            side="SELL",
            qty=10,
            filled_qty=0,
            price=94.0,
            strategy_id="KALCB",
            idempotency_key=f"STOP:primary:{stop.stop_id}:1780000000:10",
        )

        await oms._apply_fill(sell, 3)

        assert oms.persistence.stops[-1].qty == 7
        assert oms.persistence.stops[-1].status == StopStatus.EXIT_SUBMITTED.value

    @pytest.mark.asyncio
    async def test_startup_gate_halts_unprotected_allocation_with_stop_metadata(self, oms):
        oms.state.update_position("005930", real_qty=10)
        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)
        pos = oms.state.get_position("005930")
        pos.allocations["KALCB"].soft_stop_px = 95.0

        await oms._reconcile_protective_stops_on_startup()

        assert oms.risk.halt_new_entries is True
        assert oms.unprotected_positions_count == 1
        assert oms.stop_protection_status == "error"

    def test_stop_health_payload_uses_current_watcher_counts(self, oms):
        oms.active_stop_count = 1
        oms.triggered_stop_count = 0
        oms.stop_watcher_price_stale_count = 0
        oms._stop_watcher = SimpleNamespace(
            health=StopWatcherHealth(
                status="ok",
                active_stop_count=3,
                triggered_stop_count=2,
                last_check_ts=time.time(),
                stale_price_count=1,
            )
        )

        payload = oms.stop_health_payload()

        assert payload["active_stop_count"] == 3
        assert payload["triggered_stop_count"] == 2
        assert payload["stop_watcher_price_stale_count"] == 1

    def test_stop_health_payload_clears_degraded_after_watcher_recovery(self, oms):
        oms.unprotected_positions_count = 0
        oms._stop_watcher = SimpleNamespace(
            health=StopWatcherHealth(
                status="degraded",
                active_stop_count=1,
                triggered_stop_count=0,
                last_check_ts=time.time(),
                stale_price_count=1,
                last_error="previous stale price",
            )
        )

        degraded = oms.stop_health_payload()
        assert degraded["stop_protection_status"] == "degraded"

        oms._stop_watcher.health = StopWatcherHealth(
            status="ok",
            active_stop_count=1,
            triggered_stop_count=0,
            last_check_ts=time.time(),
            stale_price_count=0,
            last_error="",
        )

        payload = oms.stop_health_payload()

        assert payload["stop_protection_status"] == "ok"
        assert payload["stop_watcher_price_stale_count"] == 0
        assert payload["stop_protection_last_error"] == ""

    def test_stop_health_payload_does_not_clear_durable_stop_error_on_watcher_recovery(self, oms):
        oms.risk.halt_new_entries = True
        oms.stop_protection_status = "error"
        oms.stop_protection_last_error = "durable stop upsert failed after BUY fill"
        oms.unprotected_positions_count = 0
        oms._stop_watcher = SimpleNamespace(
            health=StopWatcherHealth(
                status="ok",
                active_stop_count=1,
                triggered_stop_count=0,
                last_check_ts=time.time(),
                stale_price_count=0,
                last_error="",
            )
        )

        payload = oms.stop_health_payload()

        assert payload["stop_protection_status"] == "error"
        assert payload["stop_protection_last_error"] == "durable stop upsert failed after BUY fill"
        assert oms.risk.halt_new_entries is True
        assert _health_payload_ready({"status": "ok", "idempotency_status": "ok", **payload}) is False

    def test_stop_health_payload_preserves_durable_stop_error_through_watcher_degraded_recovery(self, oms):
        oms.risk.halt_new_entries = True
        oms._set_stop_protection_status(
            "error",
            last_error="durable stop upsert failed after BUY fill",
            source="durable_stop",
        )
        oms.unprotected_positions_count = 0
        oms._stop_watcher = SimpleNamespace(
            health=StopWatcherHealth(
                status="degraded",
                active_stop_count=1,
                triggered_stop_count=0,
                last_check_ts=time.time(),
                stale_price_count=1,
                last_error="watcher price stale",
            )
        )

        degraded = oms.stop_health_payload()

        assert degraded["stop_protection_status"] == "error"
        assert degraded["stop_protection_last_error"] == "durable stop upsert failed after BUY fill"
        assert degraded["stop_watcher_price_stale_count"] == 1
        assert _health_payload_ready({"status": "ok", "idempotency_status": "ok", **degraded}) is False

        oms._stop_watcher.health = StopWatcherHealth(
            status="ok",
            active_stop_count=1,
            triggered_stop_count=0,
            last_check_ts=time.time(),
            stale_price_count=0,
            last_error="",
        )

        recovered = oms.stop_health_payload()

        assert recovered["stop_protection_status"] == "error"
        assert recovered["stop_protection_last_error"] == "durable stop upsert failed after BUY fill"
        assert recovered["stop_watcher_price_stale_count"] == 0
        assert oms.risk.halt_new_entries is True
        assert _health_payload_ready({"status": "ok", "idempotency_status": "ok", **recovered}) is False

    @pytest.mark.asyncio
    async def test_default_stop_price_observation_degrades_raw_price_only_mapping(self, mock_kis_api):
        oms = OMSCore(mock_kis_api)

        observation = await oms._price_observation("005930")

        assert observation.symbol == "005930"
        assert observation.price == 72000.0
        assert observation.timestamp == 0.0
        assert observation.source == "UNVERIFIED_LAST"
        assert observation.market_open is False
        assert observation.executable is False

    @pytest.mark.asyncio
    async def test_default_stop_price_observation_uses_provider_quote_timestamp_when_available(self, mock_kis_api):
        mock_kis_api.get_current_price = MagicMock(
            return_value={
                "stck_prpr": 72000.0,
                "stck_bsop_date": "20260605",
                "stck_cntg_hour": "100001",
            }
        )
        oms = OMSCore(mock_kis_api)
        oms.adapter._is_order_session_open = MagicMock(return_value=True)

        observation = await oms._price_observation("005930")

        assert observation.symbol == "005930"
        assert observation.price == 72000.0
        assert observation.timestamp == pytest.approx(
            datetime(2026, 6, 5, 10, 0, 1, tzinfo=ZoneInfo("Asia/Seoul")).timestamp()
        )
        assert observation.source == "LAST"
        assert observation.market_open is True
        assert observation.executable is True

    @pytest.mark.asyncio
    async def test_default_oms_watcher_path_degrades_raw_price_only_mapping(self, mock_kis_api, monkeypatch):
        mock_kis_api.prices["005930"] = 94.0
        stop = ProtectiveStop.for_allocation(
            oms_id="primary",
            strategy_id="KALCB",
            symbol="005930",
            qty=10,
            stop_price=95.0,
            status=StopStatus.ACTIVE.value,
        )
        persistence = _DefaultWatcherPersistence(stop)
        oms = OMSCore(mock_kis_api, persistence=persistence, require_persistence=True)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        oms.state.update_position("005930", real_qty=10, avg_price=100.0)
        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)
        oms.adapter.submit_order = AsyncMock()

        async def no_background_start(self):
            return None

        monkeypatch.setattr(StopWatcher, "start", no_background_start)

        await oms._start_stop_watcher()
        results = await oms._stop_watcher.check_once(now=time.time())

        assert len(results) == 1
        assert results[0].observation.source == "UNVERIFIED_LAST"
        assert results[0].observation.executable is False
        assert results[0].decision.stale is True
        assert results[0].decision.triggered is False
        assert persistence.exit_submitted is None
        oms.adapter.submit_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_default_oms_watcher_path_submits_exit_on_verified_live_breach(self, mock_kis_api, monkeypatch):
        mock_kis_api.prices["005930"] = 94.0
        mock_kis_api.get_current_price = MagicMock(
            return_value={
                "stck_prpr": 94.0,
                "stck_bsop_date": "20260605",
                "stck_cntg_hour": "100001",
            }
        )
        stop = ProtectiveStop.for_allocation(
            oms_id="primary",
            strategy_id="KALCB",
            symbol="005930",
            qty=10,
            stop_price=95.0,
            status=StopStatus.ACTIVE.value,
        )
        persistence = _DefaultWatcherPersistence(stop)
        oms = OMSCore(mock_kis_api, persistence=persistence, require_persistence=True)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        oms.state.update_position("005930", real_qty=10, avg_price=100.0)
        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)
        oms.adapter._is_order_session_open = MagicMock(return_value=True)
        oms.adapter.submit_order = AsyncMock(return_value=SimpleNamespace(success=True, order_id="ORD-STOP", message="ok"))

        async def no_background_start(self):
            return None

        monkeypatch.setattr(StopWatcher, "start", no_background_start)

        provider_ts = datetime(2026, 6, 5, 10, 0, 1, tzinfo=ZoneInfo("Asia/Seoul")).timestamp()
        await oms._start_stop_watcher()
        results = await oms._stop_watcher.check_once(now=provider_ts + 1.0)

        assert len(results) == 1
        assert results[0].observation.source == "LAST"
        assert results[0].observation.executable is True
        assert results[0].decision.triggered is True
        assert persistence.exit_submitted is not None
        assert persistence.exit_submitted[1] == "ORD-STOP"
        oms.adapter.submit_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reentry_after_terminal_stop_creates_distinct_active_stop_generation(self, mock_kis_api):
        terminal = ProtectiveStop.for_allocation(
            oms_id="primary",
            strategy_id="KALCB",
            symbol="005930",
            qty=0,
            stop_price=95.0,
            status=StopStatus.CANCELLED.value,
        )
        persistence = _TerminalCollisionStopPersistence(terminal)
        oms = OMSCore(mock_kis_api, persistence=persistence, require_persistence=True)

        created = await oms._upsert_durable_stop(
            symbol="005930",
            strategy_id="KALCB",
            qty=8,
            stop_price=96.0,
            entry_intent_id="11111111-1111-1111-1111-111111111111",
            entry_order_id="ORD-REENTRY",
            source_metadata={"source": "reentry"},
            event_type="STOP_CREATED",
        )

        assert created is not None
        assert created.status == StopStatus.ACTIVE.value
        assert created.stop_id != terminal.stop_id
        assert len(persistence.stops) == 2

    @pytest.mark.asyncio
    async def test_stop_exit_idempotency_key_is_stable_across_retry_observations(self, mock_kis_api):
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence._is_connected = MagicMock(return_value=True)
        captured_keys: list[str] = []

        async def reserve_intent(intent):
            captured_keys.append(intent.idempotency_key)
            return IntentResult(intent.intent_id, IntentStatus.DEFERRED, "pending broker reconcile")

        persistence.reserve_intent = reserve_intent
        oms = OMSCore(mock_kis_api, persistence=persistence, require_persistence=True)
        oms.state.update_position("005930", real_qty=10)
        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)
        stop = ProtectiveStop.for_allocation(
            oms_id="primary",
            strategy_id="KALCB",
            symbol="005930",
            qty=10,
            stop_price=95.0,
            status=StopStatus.TRIGGERED_PENDING_EXECUTION.value,
        )
        stop.triggered_at = datetime(2026, 6, 5, 9, 0, tzinfo=None)

        await oms._submit_stop_exit(stop, PriceObservation("005930", price=94.0, timestamp=1_780_000_000.0))
        await oms._submit_stop_exit(stop, PriceObservation("005930", price=93.5, timestamp=1_780_000_010.0))

        assert len(captured_keys) == 2
        assert captured_keys[0] == captured_keys[1]
        assert captured_keys[0].startswith(f"STOP:primary:{stop.stop_id}:")

    @pytest.mark.asyncio
    async def test_persisted_stop_breach_exits_without_strategy_runtime(self, mock_kis_api):
        persistence = MagicMock()
        persistence.oms_id = "primary"
        persistence._is_connected = MagicMock(return_value=True)
        persistence.reserve_intent = AsyncMock(return_value=None)
        persistence.update_intent_submission_plan = AsyncMock()
        persistence.record_order = AsyncMock(return_value="oms-order-stop")
        persistence.record_order_event = AsyncMock()
        persistence.record_intent = AsyncMock()
        oms = OMSCore(mock_kis_api, persistence=persistence, require_persistence=True)
        oms.state.equity = 100_000_000
        oms.state.buyable_cash = 50_000_000
        oms.state.update_position("005930", real_qty=10, avg_price=100.0)
        oms.state.update_allocation("005930", "KALCB", 10, cost_basis=100.0)
        oms.adapter.submit_order = AsyncMock(return_value=SimpleNamespace(success=True, order_id="ORD-STOP", message="ok"))
        stop = ProtectiveStop.for_allocation(
            oms_id="primary",
            strategy_id="KALCB",
            symbol="005930",
            qty=10,
            stop_price=95.0,
            status=StopStatus.ACTIVE.value,
        )

        class Store:
            def __init__(self, stop):
                self.stop = stop
                self.exit_submitted = None

            async def load_active_stops(self):
                return [self.stop] if self.stop.status in {"ACTIVE", "TRIGGERED_PENDING_EXECUTION"} else []

            async def touch_stop_check(self, stop_id, *, checked_at, last_price, last_error=None):
                self.stop.last_checked_at = checked_at
                self.stop.last_price = last_price
                self.stop.last_error = last_error

            async def mark_triggered(self, stop_id, trigger_price, triggered_at):
                self.stop.status = StopStatus.TRIGGERED_PENDING_EXECUTION.value
                self.stop.triggered_at = triggered_at
                self.stop.last_price = trigger_price
                return True

            async def mark_exit_submitted(self, stop_id, exit_intent_id, order_id, idempotency_key=None):
                self.stop.status = StopStatus.EXIT_SUBMITTED.value
                self.stop.exit_intent_id = exit_intent_id
                self.stop.broker_order_id = order_id
                self.stop.idempotency_key = idempotency_key
                self.exit_submitted = (exit_intent_id, order_id, idempotency_key)

        store = Store(stop)
        watcher = StopWatcher(
            store=store,
            price_source=lambda symbol: PriceObservation(symbol, price=94.0, timestamp=1_780_000_000.0),
            exit_submitter=oms._submit_stop_exit,
            stale_after_sec=30.0,
        )

        results = await watcher.check_once(now=1_780_000_000.0)

        assert len(results) == 1
        assert results[0].decision.triggered is True
        assert store.exit_submitted is not None
        assert store.exit_submitted[1] == "ORD-STOP"
        exit_intent = persistence.reserve_intent.await_args.args[0]
        assert exit_intent.intent_type == IntentType.EXIT
        assert exit_intent.idempotency_key.startswith(f"STOP:primary:{stop.stop_id}:")
        oms.adapter.submit_order.assert_awaited_once()
        assert oms.adapter.submit_order.await_args.kwargs["side"] == "SELL"
        assert oms.adapter.submit_order.await_args.kwargs["order_type"] == "MARKET"
