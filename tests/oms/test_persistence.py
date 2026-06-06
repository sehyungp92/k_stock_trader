"""Tests for OMS persistence order keying."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from oms.intent import Intent, IntentStatus, IntentType
from oms.persistence import OMSPersistence
from oms.state import OrderStatus, WorkingOrder


def _schema_fetchval_responder(
    column_map: dict[tuple[str, str], bool],
    pk_map: dict[str, list[str]],
    index_map: dict[str, bool] | None = None,
):
    async def _fetchval(query: str, *args):
        if "information_schema.columns" in query:
            return column_map[(args[0], args[1])]
        if "information_schema.table_constraints" in query:
            return pk_map[args[0]]
        if "pg_indexes" in query:
            return dict(index_map or {})[args[0]]
        raise AssertionError(f"Unexpected schema query: {query}")

    return _fetchval


def _full_schema_column_map(overrides: dict[tuple[str, str], bool] | None = None) -> dict[tuple[str, str], bool]:
    values = {
        ("positions", "oms_id"): True,
        ("allocations", "oms_id"): True,
        ("risk_daily_strategy", "oms_id"): True,
        ("strategy_state", "oms_id"): True,
        ("v_live_positions", "oms_id"): True,
        ("v_today_risk", "oms_id"): True,
        ("v_service_health", "oms_id"): True,
        ("v_live_allocations", "oms_id"): True,
        ("protective_stops", "oms_id"): True,
        ("protective_stops", "stop_id"): True,
        ("protective_stops", "status"): True,
        ("protective_stops", "idempotency_key"): True,
        ("protective_stops", "triggered_at"): True,
        ("protective_stops", "source_metadata"): True,
        ("intents", "reservation_started_at"): True,
        ("intents", "reservation_owner"): True,
        ("intents", "reservation_reconcile_status"): True,
        ("intents", "reservation_reconcile_message"): True,
        ("intents", "submit_ref"): True,
        ("intents", "planned_side"): True,
        ("intents", "planned_qty"): True,
        ("intents", "planned_order_type"): True,
    }
    values.update(overrides or {})
    return values


def _full_schema_index_map(**overrides: bool) -> dict[str, bool]:
    values = {
        "idx_protective_stops_oms_status_updated": True,
        "idx_intents_oms_status_created": True,
        "idx_intents_oms_idempotency": True,
        "idx_intents_oms_order_id": True,
        "idx_orders_oms_status_created": True,
        "idx_orders_oms_kis_order": True,
    }
    values.update(overrides)
    return values


def _scoped_pk_map() -> dict[str, list[str]]:
    return {
        "risk_daily_strategy": ["oms_id", "trade_date", "strategy_id"],
        "strategy_state": ["oms_id", "strategy_id"],
    }


class TestOMSPersistenceOrderKeying:
    """Tests for broker-order-ID to OMS-order-ID persistence mapping."""

    @pytest.mark.asyncio
    async def test_record_order_uses_broker_id_as_kis_order_id(self):
        """Broker order IDs should be stored in kis_order_id, not UUID columns."""
        persistence = OMSPersistence(dsn="postgres://test")
        persistence.pool = MagicMock()

        inserted_uuid = uuid.uuid4()
        persistence.pool.fetchval = AsyncMock(side_effect=[None, inserted_uuid])

        intent_id = str(uuid.uuid4())
        order = WorkingOrder(
            order_id="1234567890",
            symbol="005930",
            side="BUY",
            qty=100,
            price=72000,
            strategy_id="ALPHA",
            status=OrderStatus.WORKING,
        )

        oms_order_id = await persistence.record_order(order, intent_id=intent_id)

        assert oms_order_id == str(inserted_uuid)
        assert order.oms_order_id == str(inserted_uuid)

        insert_call = persistence.pool.fetchval.await_args_list[1]
        assert insert_call.args[10] == "1234567890"
        assert insert_call.args[12] == intent_id

    @pytest.mark.asyncio
    async def test_record_order_event_resolves_oms_order_id_from_broker_id(self):
        """Order events should resolve broker IDs back to the OMS UUID row."""
        persistence = OMSPersistence(dsn="postgres://test")
        persistence.pool = MagicMock()

        resolved_uuid = str(uuid.uuid4())
        persistence.pool.fetchval = AsyncMock(return_value=resolved_uuid)
        persistence.pool.execute = AsyncMock()

        await persistence.record_order_event(
            "ORDER_SUBMITTED",
            order_id="1234567890",
            strategy_id="ALPHA",
            symbol="005930",
        )

        execute_call = persistence.pool.execute.await_args
        assert execute_call.args[1] == resolved_uuid


class TestOMSPersistenceSchemaCompat:
    """Tests for scoped schema compatibility checks."""

    @pytest.mark.asyncio
    async def test_schema_compat_passes_when_all_scoped_requirements_exist(self):
        persistence = OMSPersistence(dsn="postgres://test")
        pool = MagicMock()
        pool.close = AsyncMock()
        pool.fetchval = AsyncMock(
            side_effect=_schema_fetchval_responder(
                column_map={
                    ("positions", "oms_id"): True,
                    ("allocations", "oms_id"): True,
                    ("risk_daily_strategy", "oms_id"): True,
                    ("strategy_state", "oms_id"): True,
                    ("v_live_positions", "oms_id"): True,
                    ("v_today_risk", "oms_id"): True,
                    ("v_service_health", "oms_id"): True,
                    ("v_live_allocations", "oms_id"): True,
                    ("protective_stops", "oms_id"): True,
                    ("protective_stops", "stop_id"): True,
                    ("protective_stops", "status"): True,
                    ("protective_stops", "idempotency_key"): True,
                    ("protective_stops", "triggered_at"): True,
                    ("protective_stops", "source_metadata"): True,
                    ("intents", "reservation_started_at"): True,
                    ("intents", "reservation_owner"): True,
                    ("intents", "reservation_reconcile_status"): True,
                    ("intents", "reservation_reconcile_message"): True,
                    ("intents", "submit_ref"): True,
                    ("intents", "planned_side"): True,
                    ("intents", "planned_qty"): True,
                    ("intents", "planned_order_type"): True,
                },
                pk_map={
                    "risk_daily_strategy": ["oms_id", "trade_date", "strategy_id"],
                    "strategy_state": ["oms_id", "strategy_id"],
                },
                index_map={
                    "idx_protective_stops_oms_status_updated": True,
                    "idx_intents_oms_status_created": True,
                    "idx_intents_oms_idempotency": True,
                    "idx_intents_oms_order_id": True,
                    "idx_orders_oms_status_created": True,
                    "idx_orders_oms_kis_order": True,
                },
            )
        )
        persistence.pool = pool

        await persistence._check_schema_compat()

        assert persistence.pool is pool
        pool.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_schema_compat_disables_persistence_when_scoped_view_missing(self):
        persistence = OMSPersistence(dsn="postgres://test")
        pool = MagicMock()
        pool.close = AsyncMock()
        pool.fetchval = AsyncMock(
            side_effect=_schema_fetchval_responder(
                column_map={
                    ("positions", "oms_id"): True,
                    ("allocations", "oms_id"): True,
                    ("risk_daily_strategy", "oms_id"): True,
                    ("strategy_state", "oms_id"): True,
                    ("v_live_positions", "oms_id"): True,
                    ("v_today_risk", "oms_id"): True,
                    ("v_service_health", "oms_id"): True,
                    ("v_live_allocations", "oms_id"): False,
                    ("protective_stops", "oms_id"): True,
                    ("protective_stops", "stop_id"): True,
                    ("protective_stops", "status"): True,
                    ("protective_stops", "idempotency_key"): True,
                    ("protective_stops", "triggered_at"): True,
                    ("protective_stops", "source_metadata"): True,
                    ("intents", "reservation_started_at"): True,
                    ("intents", "reservation_owner"): True,
                    ("intents", "reservation_reconcile_status"): True,
                    ("intents", "reservation_reconcile_message"): True,
                    ("intents", "submit_ref"): True,
                    ("intents", "planned_side"): True,
                    ("intents", "planned_qty"): True,
                    ("intents", "planned_order_type"): True,
                },
                pk_map={
                    "risk_daily_strategy": ["oms_id", "trade_date", "strategy_id"],
                    "strategy_state": ["oms_id", "strategy_id"],
                },
                index_map={
                    "idx_protective_stops_oms_status_updated": True,
                    "idx_intents_oms_status_created": True,
                    "idx_intents_oms_idempotency": True,
                    "idx_intents_oms_order_id": True,
                    "idx_orders_oms_status_created": True,
                    "idx_orders_oms_kis_order": True,
                },
            )
        )
        persistence.pool = pool

        await persistence._check_schema_compat()

        pool.close.assert_awaited_once()
        assert persistence.pool is None

    @pytest.mark.asyncio
    async def test_schema_compat_disables_persistence_when_hardening_migration_missing(self):
        persistence = OMSPersistence(dsn="postgres://test")
        pool = MagicMock()
        pool.close = AsyncMock()
        pool.fetchval = AsyncMock(
            side_effect=_schema_fetchval_responder(
                column_map=_full_schema_column_map({("protective_stops", "oms_id"): False}),
                pk_map=_scoped_pk_map(),
                index_map=_full_schema_index_map(),
            )
        )
        persistence.pool = pool

        await persistence._check_schema_compat()

        pool.close.assert_awaited_once()
        assert persistence.pool is None


class TestOMSPersistenceScopedUpserts:
    """Tests for conflict targets that must include oms_id."""

    @pytest.mark.asyncio
    async def test_update_daily_risk_strategy_uses_scoped_conflict_target(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="vps1")
        persistence.pool = MagicMock()
        persistence.pool.execute = AsyncMock()

        await persistence.update_daily_risk_strategy(
            trade_date=date(2026, 4, 20),
            strategy_id="ALPHA",
            realized_pnl_krw=1000,
            unrealized_pnl_krw=200,
            trades_count=3,
            wins=2,
            losses=1,
            halted=False,
        )

        sql = persistence.pool.execute.await_args.args[0]
        assert "ON CONFLICT (oms_id, trade_date, strategy_id)" in sql

    @pytest.mark.asyncio
    async def test_update_strategy_state_uses_scoped_conflict_target(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="vps2")
        persistence.pool = MagicMock()
        persistence.pool.execute = AsyncMock()

        await persistence.update_strategy_state(
            strategy_id="BETA",
            mode="RUNNING",
            symbols_hot=1,
            symbols_warm=2,
            symbols_cold=3,
            positions_count=1,
        )

        sql = persistence.pool.execute.await_args.args[0]
        assert "ON CONFLICT (oms_id, strategy_id)" in sql


class TestOMSPersistenceRestartMetadata:
    """Tests for restart-safe OMS metadata hydration."""

    @pytest.mark.asyncio
    async def test_record_order_persists_idempotency_metadata(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        persistence.pool.fetchval = AsyncMock(side_effect=[None, uuid.uuid4()])

        order = WorkingOrder(
            order_id="ORD123",
            symbol="005930",
            side="BUY",
            qty=10,
            price=72000,
            order_type="MARKET",
            strategy_id="KALCB",
            idempotency_key="KALCB:005930:ENTER:20260604:abc:10",
            submit_ref="OMS-submit-ref",
            branch="001",
            risk_stop_px=71000.0,
            risk_hard_stop_px=70500.0,
        )

        await persistence.record_order(order, intent_id=str(uuid.uuid4()))

        insert_call = persistence.pool.fetchval.await_args_list[1]
        meta = OMSPersistence._jsonb_dict(insert_call.args[-1])
        assert meta["idempotency_key"] == "KALCB:005930:ENTER:20260604:abc:10"
        assert meta["submit_ref"] == "OMS-submit-ref"
        assert meta["branch"] == "001"
        assert meta["working_price"] == 72000
        assert meta["risk_stop_px"] == 71000.0
        assert meta["risk_hard_stop_px"] == 70500.0

    @pytest.mark.asyncio
    async def test_load_working_orders_restores_idempotency_and_price_meta(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        order_uuid = uuid.uuid4()
        persistence.pool.fetch = AsyncMock(
            return_value=[
                {
                    "oms_order_id": order_uuid,
                    "kis_order_id": "ORD123",
                    "symbol": "005930",
                    "side": "BUY",
                    "qty": 10,
                    "filled_qty": 0,
                    "limit_price": None,
                    "avg_fill_price": None,
                    "order_type": "MARKET",
                    "status": "WORKING",
                    "strategy_id": "KALCB",
                    "created_at": datetime(2026, 6, 4, 9, 0),
                    "last_update_at": datetime(2026, 6, 4, 9, 1),
                    "cancel_after_sec": 60,
                    "intent_id": uuid.uuid4(),
                    "meta": {
                        "idempotency_key": "KALCB:005930:ENTER:20260604:abc:10",
                        "branch": "001",
                        "working_price": 72000,
                        "submit_ts": 1_780_000_000.0,
                        "submit_ref": "OMS-submit-ref",
                        "risk_stop_px": 71000.0,
                        "risk_hard_stop_px": 70500.0,
                    },
                }
            ]
        )

        orders = await persistence.load_working_orders()

        assert len(orders) == 1
        assert orders[0].idempotency_key == "KALCB:005930:ENTER:20260604:abc:10"
        assert orders[0].branch == "001"
        assert orders[0].price == 72000
        assert orders[0].submit_ts == 1_780_000_000.0
        assert orders[0].submit_ref == "OMS-submit-ref"
        assert orders[0].risk_stop_px == 71000.0
        assert orders[0].risk_hard_stop_px == 70500.0

    @pytest.mark.asyncio
    async def test_load_idempotency_results_hydrates_executed_outcomes(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        intent_id = uuid.uuid4()
        persistence.pool.fetch = AsyncMock(
            return_value=[
                {
                    "idempotency_key": "OLR:005930:ENTER:20260604:abc:10",
                    "intent_id": intent_id,
                    "status": "EXECUTED",
                    "result_message": "submitted",
                    "modified_qty": None,
                    "order_id": "ORD789",
                    "cooldown_until": None,
                }
            ]
        )

        results = await persistence.load_idempotency_results()

        assert results["OLR:005930:ENTER:20260604:abc:10"].status == IntentStatus.EXECUTED
        assert results["OLR:005930:ENTER:20260604:abc:10"].order_id == "ORD789"

    @pytest.mark.asyncio
    async def test_reserve_intent_inserts_pending_row_before_submit(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        persistence._intents_execution_style_column = False
        persistence.pool.fetchval = AsyncMock(return_value=uuid.uuid4())

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KALCB",
            symbol="005930",
            desired_qty=10,
            signal_hash="abc",
        )

        result = await persistence.reserve_intent(intent)

        assert result is None
        insert_sql = persistence.pool.fetchval.await_args_list[0].args[0]
        assert "ON CONFLICT (idempotency_key) DO NOTHING" in insert_sql
        assert "Reserved before broker submission" in persistence.pool.fetchval.await_args_list[0].args

    @pytest.mark.asyncio
    async def test_reserve_intent_defers_existing_pending_key(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        persistence._intents_execution_style_column = False
        persistence.pool.fetchval = AsyncMock(return_value=None)
        persistence.pool.fetchrow = AsyncMock(
            return_value={
                "idempotency_key": "KALCB:005930:ENTER:20260604:abc:10",
                "intent_id": uuid.uuid4(),
                "status": "PENDING",
                "result_message": "Reserved before broker submission",
                "modified_qty": None,
                "order_id": None,
                "cooldown_until": None,
            }
        )
        persistence.pool.execute = AsyncMock()

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KALCB",
            symbol="005930",
            desired_qty=10,
            idempotency_key="KALCB:005930:ENTER:20260604:abc:10",
        )

        result = await persistence.reserve_intent(intent)

        assert result is not None
        assert result.status == IntentStatus.DEFERRED
        assert "already pending" in result.message
        persistence.pool.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reserve_intent_reuses_pre_broker_terminal_key(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        persistence._intents_execution_style_column = False
        persistence.pool.fetchval = AsyncMock(return_value=None)
        persistence.pool.fetchrow = AsyncMock(
            return_value={
                "idempotency_key": "KALCB:005930:ENTER:20260604:abc:10",
                "intent_id": uuid.uuid4(),
                "status": "DEFERRED",
                "result_message": "OMS unreachable",
                "modified_qty": None,
                "order_id": None,
                "cooldown_until": None,
            }
        )
        persistence.pool.execute = AsyncMock()

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KALCB",
            symbol="005930",
            desired_qty=10,
            idempotency_key="KALCB:005930:ENTER:20260604:abc:10",
        )

        result = await persistence.reserve_intent(intent)

        assert result is None
        update_args = persistence.pool.execute.await_args.args
        assert "UPDATE intents" in update_args[0]
        assert "strategy_id = $3" in update_args[0]
        assert update_args[2] == intent.intent_id
        assert update_args[21] == IntentStatus.PENDING.name

    @pytest.mark.asyncio
    async def test_timeout_no_order_manual_resolution_can_be_rereserved(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        persistence._intents_execution_style_column = False
        persistence.pool.fetchval = AsyncMock(return_value=None)
        persistence.pool.fetchrow = AsyncMock(
            return_value={
                "idempotency_key": "KALCB:005930:ENTER:20260604:timeout:10",
                "intent_id": uuid.uuid4(),
                "status": "DEFERRED",
                "result_message": "timeout reconciled by operator: no broker order exists",
                "modified_qty": None,
                "order_id": None,
                "cooldown_until": None,
            }
        )
        persistence.pool.execute = AsyncMock()

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KALCB",
            symbol="005930",
            desired_qty=10,
            idempotency_key="KALCB:005930:ENTER:20260604:timeout:10",
        )

        result = await persistence.reserve_intent(intent)

        assert result is None
        update_args = persistence.pool.execute.await_args.args
        assert "UPDATE intents" in update_args[0]
        assert update_args[21] == IntentStatus.PENDING.name

    @pytest.mark.asyncio
    async def test_update_intent_submission_plan_persists_submit_ref_before_submit(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        persistence._intents_submit_ref_column = True
        persistence.pool.execute = AsyncMock()
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KALCB",
            symbol="005930",
            desired_qty=10,
            idempotency_key="idem-submit-ref",
        )

        await persistence.update_intent_submission_plan(
            intent,
            side="BUY",
            planned_qty=10,
            order_type="LIMIT",
            limit_price=100.0,
            stop_price=95.0,
            submit_ref="OMS-submit-ref",
        )

        args = persistence.pool.execute.await_args.args
        assert "planned_side" in args[0]
        assert "submit_ref" in args[0]
        assert args[7] == "OMS-submit-ref"

    @pytest.mark.asyncio
    async def test_mark_idempotency_ambiguous_preserves_pending_row_for_operator_reconcile(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        persistence._intents_submit_ref_column = True
        persistence.pool.execute = AsyncMock()

        await persistence.mark_idempotency_ambiguous(
            "idem-ambiguous",
            reason="multiple_exact_candidates:2",
            submit_ref="OMS-submit-ref",
        )

        args = persistence.pool.execute.await_args.args
        assert "reservation_reconcile_status = 'AMBIGUOUS'" in args[0]
        assert "SET status" not in args[0]
        assert args[1] == "primary"
        assert args[2] == "idem-ambiguous"
        assert args[3] == "multiple_exact_candidates:2"
        assert args[5] == "OMS-submit-ref"

    @pytest.mark.asyncio
    async def test_idempotency_health_degrades_for_unresolved_pending_or_ambiguous_reservations(self):
        persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
        persistence.pool = MagicMock()
        persistence._intents_submit_ref_column = True
        persistence.pool.fetchrow = AsyncMock(return_value={"pending_count": 1, "ambiguous_count": 2})

        health = await persistence.idempotency_health()

        assert health == {
            "status": "degraded",
            "pending_count": 1,
            "ambiguous_count": 2,
        }
        query = persistence.pool.fetchrow.await_args.args[0]
        assert "status = 'PENDING'" in query
        assert "reservation_reconcile_status" in query
        assert persistence.pool.fetchrow.await_args.args[1] == "primary"


def test_finalize_migration_scopes_remaining_primary_keys():
    migration = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "postgres"
        / "init"
        / "007_oms_scope_finalize.sql"
    ).read_text(encoding="utf-8")

    assert "ADD PRIMARY KEY (oms_id, trade_date, strategy_id)" in migration
    assert "ADD PRIMARY KEY (oms_id, strategy_id)" in migration
    assert "DROP CONSTRAINT %I" in migration


def test_idempotency_hardening_migration_documents_global_key_scope():
    migration = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "postgres"
        / "init"
        / "009_idempotency_hardening.sql"
    ).read_text(encoding="utf-8")

    assert "idempotency_key remains account-global" in migration
    assert "COMMENT ON COLUMN intents.idempotency_key" in migration
    assert "idx_intents_oms_status_created" in migration
    assert "idx_intents_oms_idempotency" in migration
    assert "idx_intents_oms_order_id" in migration
