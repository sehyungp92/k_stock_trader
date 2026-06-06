"""
OMS Postgres persistence layer.

Provides async methods to persist OMS state to Postgres.
Uses asyncpg for non-blocking database access.
"""

from __future__ import annotations
from datetime import datetime, date
from typing import Any, Dict, List, Optional
import asyncpg
import json
import os
import time
import uuid
from loguru import logger

from .intent import Intent, IntentResult, IntentStatus
from .state import WorkingOrder, SymbolPosition, StrategyAllocation, OrderStatus
from .stop_protection import ProtectiveStop, StopStatus, stop_from_row


class OMSPersistence:
    """Async persistence layer for OMS state."""

    def __init__(self, dsn: Optional[str] = None, oms_id: Optional[str] = None):
        self.dsn = dsn or os.environ.get("DATABASE_URL")
        self.oms_id = oms_id or os.environ.get("OMS_ID", "primary")
        if not self.dsn:
            logger.critical(
                "DATABASE_URL not set and no dsn provided — "
                "Postgres persistence will be unavailable"
            )
        self.pool: Optional[asyncpg.Pool] = None
        self.consecutive_failures: int = 0
        self.total_failures: int = 0
        self._intents_execution_style_column: Optional[bool] = None
        self._intents_submit_ref_column: Optional[bool] = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        if not self.dsn:
            logger.error("No DATABASE_URL configured — skipping Postgres connection")
            return
        try:
            self.pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
            logger.info("Postgres connection pool established")
            await self._check_schema_compat()
        except Exception as e:
            logger.warning(f"Postgres connection failed (will retry): {e}")
            self.pool = None

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    def _is_connected(self) -> bool:
        return self.pool is not None

    def _record_success(self) -> None:
        """Reset consecutive failure counter on success."""
        self.consecutive_failures = 0

    def _record_failure(self) -> None:
        """Track persistence failures."""
        self.consecutive_failures += 1
        self.total_failures += 1

    async def _relation_has_column(self, relation: str, column: str) -> bool:
        """Return True when a table/view exposes the requested column."""
        if not self.pool:
            return False
        return bool(
            await self.pool.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = $1
                      AND column_name = $2
                )
                """,
                relation,
                column,
            )
        )

    async def _primary_key_columns(self, table_name: str) -> List[str]:
        """Return the ordered primary-key column list for a table."""
        if not self.pool:
            return []
        columns = await self.pool.fetchval(
            """
            SELECT COALESCE(array_agg(kcu.column_name ORDER BY kcu.ordinal_position), ARRAY[]::text[])
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = 'public'
              AND tc.table_name = $1
              AND tc.constraint_type = 'PRIMARY KEY'
            """,
            table_name,
        )
        return list(columns or [])

    async def _index_exists(self, index_name: str) -> bool:
        """Return True when the named public index exists."""
        if not self.pool:
            return False
        return bool(
            await self.pool.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND indexname = $1
                )
                """,
                index_name,
            )
        )

    async def _intent_execution_style_supported(self) -> bool:
        """Return whether this schema can persist close-auction execution style."""
        if self._intents_execution_style_column is None:
            self._intents_execution_style_column = await self._relation_has_column("intents", "execution_style")
        return bool(self._intents_execution_style_column)

    async def _intent_submission_plan_supported(self) -> bool:
        """Return whether the idempotency hardening columns are available."""
        if self._intents_submit_ref_column is None:
            self._intents_submit_ref_column = await self._relation_has_column("intents", "submit_ref")
        return bool(self._intents_submit_ref_column)

    async def _check_schema_compat(self) -> None:
        """Verify DB has the scoped schema and required dashboard views."""
        if not self.pool:
            return
        try:
            checks = [
                ("positions.oms_id", await self._relation_has_column("positions", "oms_id")),
                ("allocations.oms_id", await self._relation_has_column("allocations", "oms_id")),
                (
                    "risk_daily_strategy.oms_id",
                    await self._relation_has_column("risk_daily_strategy", "oms_id"),
                ),
                ("strategy_state.oms_id", await self._relation_has_column("strategy_state", "oms_id")),
                ("v_live_positions.oms_id", await self._relation_has_column("v_live_positions", "oms_id")),
                ("v_today_risk.oms_id", await self._relation_has_column("v_today_risk", "oms_id")),
                (
                    "v_service_health.oms_id",
                    await self._relation_has_column("v_service_health", "oms_id"),
                ),
                (
                    "v_live_allocations.oms_id",
                    await self._relation_has_column("v_live_allocations", "oms_id"),
                ),
                (
                    "risk_daily_strategy primary key",
                    await self._primary_key_columns("risk_daily_strategy")
                    == ["oms_id", "trade_date", "strategy_id"],
                ),
                (
                    "strategy_state primary key",
                    await self._primary_key_columns("strategy_state") == ["oms_id", "strategy_id"],
                ),
                ("protective_stops.oms_id", await self._relation_has_column("protective_stops", "oms_id")),
                ("protective_stops.stop_id", await self._relation_has_column("protective_stops", "stop_id")),
                ("protective_stops.status", await self._relation_has_column("protective_stops", "status")),
                ("protective_stops.idempotency_key", await self._relation_has_column("protective_stops", "idempotency_key")),
                ("protective_stops.triggered_at", await self._relation_has_column("protective_stops", "triggered_at")),
                ("protective_stops.source_metadata", await self._relation_has_column("protective_stops", "source_metadata")),
                ("intents.reservation_started_at", await self._relation_has_column("intents", "reservation_started_at")),
                ("intents.reservation_owner", await self._relation_has_column("intents", "reservation_owner")),
                ("intents.reservation_reconcile_status", await self._relation_has_column("intents", "reservation_reconcile_status")),
                ("intents.reservation_reconcile_message", await self._relation_has_column("intents", "reservation_reconcile_message")),
                ("intents.submit_ref", await self._relation_has_column("intents", "submit_ref")),
                ("intents.planned_side", await self._relation_has_column("intents", "planned_side")),
                ("intents.planned_qty", await self._relation_has_column("intents", "planned_qty")),
                ("intents.planned_order_type", await self._relation_has_column("intents", "planned_order_type")),
                ("idx_protective_stops_oms_status_updated", await self._index_exists("idx_protective_stops_oms_status_updated")),
                ("idx_intents_oms_status_created", await self._index_exists("idx_intents_oms_status_created")),
                ("idx_intents_oms_idempotency", await self._index_exists("idx_intents_oms_idempotency")),
                ("idx_intents_oms_order_id", await self._index_exists("idx_intents_oms_order_id")),
                ("idx_orders_oms_status_created", await self._index_exists("idx_orders_oms_status_created")),
                ("idx_orders_oms_kis_order", await self._index_exists("idx_orders_oms_kis_order")),
            ]
            missing = [name for name, ok in checks if not ok]
            if missing:
                logger.critical(
                    "SCHEMA MISMATCH: missing or outdated OMS hardening schema objects: {}. "
                    "Apply migrations: psql $DATABASE_URL -f infra/postgres/init/005_oms_scoping.sql && "
                    "psql $DATABASE_URL -f infra/postgres/init/006_views_oms_scoped.sql && "
                    "psql $DATABASE_URL -f infra/postgres/init/007_oms_scope_finalize.sql && "
                    "psql $DATABASE_URL -f infra/postgres/init/008_protective_stops.sql && "
                    "psql $DATABASE_URL -f infra/postgres/init/009_idempotency_hardening.sql",
                    ", ".join(missing),
                )
                await self.pool.close()
                self.pool = None
            else:
                self._record_success()
                logger.info("Schema compatibility check passed")
        except Exception as e:
            logger.warning(f"Schema compatibility check failed (non-fatal): {e}")

    @staticmethod
    def _normalize_uuid(value: Optional[str]) -> Optional[str]:
        """Return canonical UUID string, or None when value is not a UUID."""
        if not value:
            return None
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _jsonb_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _order_meta(order: WorkingOrder) -> str:
        payload = {
            "idempotency_key": order.idempotency_key,
            "submit_ref": order.submit_ref,
            "branch": order.branch,
            "submit_ts": order.submit_ts,
            "working_price": order.price,
            "risk_stop_px": order.risk_stop_px,
            "risk_hard_stop_px": order.risk_hard_stop_px,
        }
        return json.dumps({k: v for k, v in payload.items() if v not in (None, "")}, sort_keys=True, default=str)

    async def _resolve_oms_order_id(self, order_id: Optional[str]) -> Optional[str]:
        """Resolve broker/KIS order IDs back to the OMS UUID primary key."""
        if not self._is_connected() or not order_id:
            return None

        normalized = self._normalize_uuid(order_id)
        if normalized:
            return normalized

        try:
            resolved = await self.pool.fetchval(
                """
                SELECT oms_order_id
                FROM orders
                WHERE kis_order_id = $1 AND oms_id = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                order_id, self.oms_id,
            )
            return str(resolved) if resolved else None
        except Exception as e:
            logger.error(f"Failed to resolve oms_order_id for {order_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Intent Recording
    # ------------------------------------------------------------------

    async def record_intent(self, intent: Intent, result: IntentResult) -> None:
        """Record intent and its result."""
        if not self._is_connected():
            return
        try:
            has_execution_style = await self._intent_execution_style_supported()
            execution_style_column = ", execution_style" if has_execution_style else ""
            execution_style_value = ", $15" if has_execution_style else ""
            execution_style_update = ",\n                    execution_style = EXCLUDED.execution_style" if has_execution_style else ""
            entry_base = 16 if has_execution_style else 15
            expiry_index = 14
            cooldown_index = entry_base + 10
            oms_index = entry_base + 11
            params = [
                intent.intent_id,
                intent.idempotency_key,
                intent.strategy_id,
                intent.symbol,
                intent.intent_type.name,
                intent.desired_qty,
                intent.target_qty,
                intent.urgency.name,
                intent.time_horizon.name,
                intent.constraints.max_slippage_bps,
                intent.constraints.max_spread_bps,
                intent.constraints.limit_price,
                intent.constraints.stop_price,
                intent.constraints.expiry_ts,
            ]
            if has_execution_style:
                params.append(intent.constraints.execution_style)
            params.extend(
                [
                    intent.risk_payload.entry_px,
                    intent.risk_payload.stop_px,
                    intent.risk_payload.hard_stop_px,
                    intent.risk_payload.rationale_code,
                    intent.risk_payload.confidence,
                    intent.signal_hash,
                    result.status.name,
                    result.message,
                    result.modified_qty,
                    result.order_id,
                    result.cooldown_until,
                    self.oms_id,
                ]
            )
            await self.pool.execute(
                f"""
                INSERT INTO intents (
                    intent_id, idempotency_key, strategy_id, symbol,
                    intent_type, desired_qty, target_qty, urgency, time_horizon,
                    max_slippage_bps, max_spread_bps, limit_price, stop_price, expiry_ts{execution_style_column},
                    entry_px, stop_px, hard_stop_px, rationale_code, confidence, signal_hash,
                    status, result_message, modified_qty, order_id, cooldown_until, processed_at,
                    oms_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    to_timestamp(${expiry_index}){execution_style_value}, ${entry_base}, ${entry_base + 1}, ${entry_base + 2}, ${entry_base + 3}, ${entry_base + 4}, ${entry_base + 5}, ${entry_base + 6}, ${entry_base + 7}, ${entry_base + 8}, ${entry_base + 9},
                    to_timestamp(${cooldown_index}), NOW(), ${oms_index}
                )
                ON CONFLICT (idempotency_key) DO UPDATE SET
                    status = EXCLUDED.status,
                    result_message = EXCLUDED.result_message,
                    modified_qty = EXCLUDED.modified_qty,
                    order_id = EXCLUDED.order_id,
                    cooldown_until = EXCLUDED.cooldown_until{execution_style_update},
                    processed_at = NOW()
                """,
                *params,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to record intent: {e}")

    @staticmethod
    def _intent_result_from_row(row: Any) -> IntentResult:
        try:
            status = IntentStatus[str(row["status"])]
        except KeyError:
            status = IntentStatus.DEFERRED
        return IntentResult(
            intent_id=str(row["intent_id"]),
            status=status,
            message=str(row["result_message"] or ""),
            modified_qty=row["modified_qty"],
            order_id=row["order_id"],
            cooldown_until=float(row["cooldown_until"]) if row["cooldown_until"] is not None else None,
        )

    async def reserve_intent(self, intent: Intent) -> Optional[IntentResult]:
        """Reserve an idempotency key before broker submission.

        Returns None when this process owns the reservation. Returns an
        existing/fail-closed result when the key has already been reserved.
        """
        if not self._is_connected():
            return IntentResult(
                intent_id=intent.intent_id,
                status=IntentStatus.DEFERRED,
                message="Durable idempotency reservation unavailable; persistence is disconnected",
            )
        try:
            has_execution_style = await self._intent_execution_style_supported()
            execution_style_column = ", execution_style" if has_execution_style else ""
            execution_style_value = ", $15" if has_execution_style else ""
            entry_base = 16 if has_execution_style else 15
            params = [
                intent.intent_id,
                intent.idempotency_key,
                intent.strategy_id,
                intent.symbol,
                intent.intent_type.name,
                intent.desired_qty,
                intent.target_qty,
                intent.urgency.name,
                intent.time_horizon.name,
                intent.constraints.max_slippage_bps,
                intent.constraints.max_spread_bps,
                intent.constraints.limit_price,
                intent.constraints.stop_price,
                intent.constraints.expiry_ts,
            ]
            if has_execution_style:
                params.append(intent.constraints.execution_style)
            params.extend(
                [
                    intent.risk_payload.entry_px,
                    intent.risk_payload.stop_px,
                    intent.risk_payload.hard_stop_px,
                    intent.risk_payload.rationale_code,
                    intent.risk_payload.confidence,
                    intent.signal_hash,
                    IntentStatus.PENDING.name,
                    "Reserved before broker submission",
                    self.oms_id,
                ]
            )
            inserted = await self.pool.fetchval(
                f"""
                INSERT INTO intents (
                    intent_id, idempotency_key, strategy_id, symbol,
                    intent_type, desired_qty, target_qty, urgency, time_horizon,
                    max_slippage_bps, max_spread_bps, limit_price, stop_price, expiry_ts{execution_style_column},
                    entry_px, stop_px, hard_stop_px, rationale_code, confidence, signal_hash,
                    status, result_message, oms_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    to_timestamp($14){execution_style_value}, ${entry_base}, ${entry_base + 1}, ${entry_base + 2}, ${entry_base + 3}, ${entry_base + 4}, ${entry_base + 5},
                    ${entry_base + 6}, ${entry_base + 7}, ${entry_base + 8}
                )
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING intent_id
                """,
                *params,
            )
            if inserted:
                await self._mark_intent_reserved(intent)
                self._record_success()
                return None

            existing = await self.pool.fetchrow(
                """
                SELECT idempotency_key, intent_id, status, result_message,
                       modified_qty, order_id, EXTRACT(EPOCH FROM cooldown_until) AS cooldown_until
                FROM intents
                WHERE idempotency_key = $1
                LIMIT 1
                """,
                intent.idempotency_key,
            )
            if existing is None:
                self._record_failure()
                return IntentResult(
                    intent_id=intent.intent_id,
                    status=IntentStatus.DEFERRED,
                    message="Durable idempotency reservation conflict could not be reconciled",
                )

            result = self._intent_result_from_row(existing)
            if result.status == IntentStatus.PENDING:
                return IntentResult(
                    intent_id=result.intent_id,
                    status=IntentStatus.DEFERRED,
                    message="Idempotency key is already pending; reconcile before retry",
                    order_id=result.order_id,
                )
            if result.status in {IntentStatus.REJECTED, IntentStatus.DEFERRED, IntentStatus.CANCELLED} and not result.order_id:
                execution_style_update = ", execution_style = $15" if has_execution_style else ""
                rereserve_params = [
                    intent.idempotency_key,
                    intent.intent_id,
                    intent.strategy_id,
                    intent.symbol,
                    intent.intent_type.name,
                    intent.desired_qty,
                    intent.target_qty,
                    intent.urgency.name,
                    intent.time_horizon.name,
                    intent.constraints.max_slippage_bps,
                    intent.constraints.max_spread_bps,
                    intent.constraints.limit_price,
                    intent.constraints.stop_price,
                    intent.constraints.expiry_ts,
                ]
                if has_execution_style:
                    rereserve_params.append(intent.constraints.execution_style)
                rereserve_params.extend(
                    [
                        intent.risk_payload.entry_px,
                        intent.risk_payload.stop_px,
                        intent.risk_payload.hard_stop_px,
                        intent.risk_payload.rationale_code,
                        intent.risk_payload.confidence,
                        intent.signal_hash,
                        IntentStatus.PENDING.name,
                        "Reserved before broker submission",
                        self.oms_id,
                    ]
                )
                await self.pool.execute(
                    f"""
                    UPDATE intents
                    SET intent_id = $2,
                        strategy_id = $3,
                        symbol = $4,
                        intent_type = $5,
                        desired_qty = $6,
                        target_qty = $7,
                        urgency = $8,
                        time_horizon = $9,
                        max_slippage_bps = $10,
                        max_spread_bps = $11,
                        limit_price = $12,
                        stop_price = $13,
                        expiry_ts = to_timestamp($14){execution_style_update},
                        entry_px = ${entry_base},
                        stop_px = ${entry_base + 1},
                        hard_stop_px = ${entry_base + 2},
                        rationale_code = ${entry_base + 3},
                        confidence = ${entry_base + 4},
                        signal_hash = ${entry_base + 5},
                        status = ${entry_base + 6},
                        result_message = ${entry_base + 7},
                        modified_qty = NULL,
                        order_id = NULL,
                        cooldown_until = NULL,
                        processed_at = NULL,
                        oms_id = ${entry_base + 8}
                    WHERE idempotency_key = $1
                    """,
                    *rereserve_params,
                )
                await self._mark_intent_reserved(intent)
                self._record_success()
                return None
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to reserve intent idempotency key: {e}")
            return IntentResult(
                intent_id=intent.intent_id,
                status=IntentStatus.DEFERRED,
                message="Durable idempotency reservation failed; retry after persistence recovers",
            )

    async def load_idempotency_results(self) -> Dict[str, IntentResult]:
        """Load durable accepted/executed intent outcomes for restart dedupe."""
        if not self._is_connected():
            return {}
        try:
            rows = await self.pool.fetch(
                """
                SELECT DISTINCT ON (idempotency_key)
                    idempotency_key, intent_id, status, result_message,
                    modified_qty, order_id, EXTRACT(EPOCH FROM cooldown_until) AS cooldown_until
                FROM intents
                WHERE oms_id = $1
                  AND status IN ('EXECUTED', 'ACCEPTED')
                ORDER BY idempotency_key, processed_at DESC NULLS LAST, created_at DESC
                """,
                self.oms_id,
            )
            results: Dict[str, IntentResult] = {}
            for row in rows:
                key = str(row["idempotency_key"] or "")
                if not key:
                    continue
                results[key] = self._intent_result_from_row(row)
            self._record_success()
            return results
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load idempotency results: {e}")
            return {}

    async def _mark_intent_reserved(self, intent: Intent) -> None:
        """Attach reservation lifecycle metadata when hardening columns exist."""
        if not self._is_connected():
            return
        if not await self._intent_submission_plan_supported():
            return
        try:
            await self.pool.execute(
                """
                UPDATE intents
                SET reservation_started_at = COALESCE(reservation_started_at, NOW()),
                    reservation_owner = COALESCE(reservation_owner, $2),
                    reservation_expires_at = COALESCE(reservation_expires_at, NOW() + INTERVAL '10 minutes'),
                    reservation_reconcile_status = COALESCE(reservation_reconcile_status, 'PENDING'),
                    reservation_reconcile_message = COALESCE(reservation_reconcile_message, 'Reserved before broker submission')
                WHERE idempotency_key = $1
                """,
                intent.idempotency_key,
                f"oms:{self.oms_id}",
            )
        except Exception as e:
            logger.debug(f"Intent reservation metadata unavailable: {e}")

    async def update_intent_submission_plan(
        self,
        intent: Intent,
        *,
        side: str,
        planned_qty: int,
        order_type: str,
        limit_price: Optional[float],
        stop_price: Optional[float],
        submit_ref: str,
    ) -> None:
        """Persist final broker-submit metadata before any adapter call."""
        if not self._is_connected():
            return
        if not await self._intent_submission_plan_supported():
            return
        try:
            await self.pool.execute(
                """
                UPDATE intents
                SET planned_side = $2,
                    planned_qty = $3,
                    planned_order_type = $4,
                    planned_limit_price = $5,
                    planned_stop_price = $6,
                    submit_ref = $7,
                    reservation_owner = COALESCE(reservation_owner, $8),
                    reservation_started_at = COALESCE(reservation_started_at, NOW()),
                    reservation_reconcile_status = 'SUBMITTING',
                    reservation_reconcile_message = 'Broker submission planned'
                WHERE idempotency_key = $1
                  AND oms_id = $9
                """,
                intent.idempotency_key,
                side,
                int(planned_qty or 0),
                order_type,
                limit_price,
                stop_price,
                submit_ref,
                f"oms:{self.oms_id}",
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to persist intent submission plan: {e}")

    async def list_pending_idempotency(self, stale_after_sec: float = 60.0) -> List[Dict[str, Any]]:
        """List durable PENDING reservations requiring reconciliation."""
        if not self._is_connected():
            return []
        try:
            if not await self._intent_submission_plan_supported():
                rows = await self.pool.fetch(
                    """
                    SELECT intent_id, idempotency_key, strategy_id, symbol, intent_type,
                           desired_qty, target_qty, status, result_message, order_id,
                           created_at, processed_at,
                           stop_px, hard_stop_px,
                           EXTRACT(EPOCH FROM created_at) AS created_ts
                    FROM intents
                    WHERE oms_id = $1
                      AND status = 'PENDING'
                      AND created_at <= NOW() - ($2::text || ' seconds')::interval
                    ORDER BY created_at ASC
                    """,
                    self.oms_id,
                    int(stale_after_sec),
                )
                self._record_success()
                return [dict(row) for row in rows]
            rows = await self.pool.fetch(
                """
                SELECT intent_id, idempotency_key, strategy_id, symbol, intent_type,
                       desired_qty, target_qty, status, result_message, order_id,
                       created_at, processed_at,
                       stop_px, hard_stop_px,
                       EXTRACT(EPOCH FROM created_at) AS created_ts,
                       submit_ref, planned_side, planned_qty, planned_order_type,
                       planned_limit_price, planned_stop_price,
                       reservation_owner, reservation_reconcile_status,
                       reservation_reconcile_message
                FROM intents
                WHERE oms_id = $1
                  AND (
                      status = 'PENDING'
                      OR (
                          status = 'DEFERRED'
                          AND order_id IS NOT NULL
                          AND reservation_reconcile_status = 'AMBIGUOUS'
                      )
                  )
                  AND created_at <= NOW() - ($2::text || ' seconds')::interval
                ORDER BY created_at ASC
                """,
                self.oms_id,
                int(stale_after_sec),
            )
            self._record_success()
            return [dict(row) for row in rows]
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to list pending idempotency reservations: {e}")
            return []

    async def idempotency_health(self) -> Dict[str, Any]:
        """Return readiness-impacting unresolved idempotency reservation counts."""
        if not self._is_connected():
            return {"status": "error", "pending_count": 0, "ambiguous_count": 0}
        try:
            if await self._intent_submission_plan_supported():
                row = await self.pool.fetchrow(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'PENDING') AS pending_count,
                        COUNT(*) FILTER (WHERE COALESCE(reservation_reconcile_status, '') = 'AMBIGUOUS') AS ambiguous_count
                    FROM intents
                    WHERE oms_id = $1
                      AND (
                          status = 'PENDING'
                          OR COALESCE(reservation_reconcile_status, '') = 'AMBIGUOUS'
                      )
                    """,
                    self.oms_id,
                )
            else:
                row = await self.pool.fetchrow(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'PENDING') AS pending_count,
                        0 AS ambiguous_count
                    FROM intents
                    WHERE oms_id = $1
                      AND status = 'PENDING'
                    """,
                    self.oms_id,
                )
            pending_count = int(row["pending_count"] or 0) if row else 0
            ambiguous_count = int(row["ambiguous_count"] or 0) if row else 0
            self._record_success()
            return {
                "status": "degraded" if pending_count or ambiguous_count else "ok",
                "pending_count": pending_count,
                "ambiguous_count": ambiguous_count,
            }
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to compute idempotency health: {e}")
            return {"status": "error", "pending_count": 0, "ambiguous_count": 0}

    async def mark_intent_ambiguous(
        self,
        intent: Intent,
        *,
        order_id: Optional[str],
        submit_ref: Optional[str],
        reason: str,
    ) -> None:
        """Persist a broker-success/persistence-failure ambiguity without allowing re-submit."""
        if not self._is_connected():
            return
        try:
            if await self._intent_submission_plan_supported():
                await self.pool.execute(
                    """
                    UPDATE intents
                    SET status = 'DEFERRED',
                        result_message = $3,
                        order_id = COALESCE($4, order_id),
                        submit_ref = COALESCE($5, submit_ref),
                        reservation_reconcile_status = 'AMBIGUOUS',
                        reservation_reconcile_message = $3,
                        processed_at = NOW()
                    WHERE oms_id = $1
                      AND idempotency_key = $2
                    """,
                    self.oms_id,
                    intent.idempotency_key,
                    reason,
                    order_id,
                    submit_ref,
                )
            else:
                await self.pool.execute(
                    """
                    UPDATE intents
                    SET status = 'DEFERRED',
                        result_message = $3,
                        order_id = COALESCE($4, order_id),
                        processed_at = NOW()
                    WHERE oms_id = $1
                      AND idempotency_key = $2
                    """,
                    self.oms_id,
                    intent.idempotency_key,
                    reason,
                    order_id,
                )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to mark idempotency ambiguity for {intent.idempotency_key}: {e}")

    async def resolve_idempotency(
        self,
        idempotency_key: str,
        *,
        status: IntentStatus,
        reason: str,
        order_id: Optional[str] = None,
    ) -> Optional[IntentResult]:
        """Operator/manual reconciliation endpoint for stale pending keys."""
        if not self._is_connected() or not reason:
            return None
        try:
            if await self._intent_submission_plan_supported():
                row = await self.pool.fetchrow(
                    """
                    UPDATE intents
                    SET status = $3,
                        result_message = $4,
                        order_id = COALESCE($5, order_id),
                        reservation_reconcile_status = $3,
                        reservation_reconcile_message = $4,
                        processed_at = NOW()
                    WHERE oms_id = $1
                      AND idempotency_key = $2
                    RETURNING idempotency_key, intent_id, status, result_message,
                              modified_qty, order_id,
                              EXTRACT(EPOCH FROM cooldown_until) AS cooldown_until
                    """,
                    self.oms_id,
                    idempotency_key,
                    status.name,
                    reason,
                    order_id,
                )
            else:
                row = await self.pool.fetchrow(
                    """
                    UPDATE intents
                    SET status = $3,
                        result_message = $4,
                        order_id = COALESCE($5, order_id),
                        processed_at = NOW()
                    WHERE oms_id = $1
                      AND idempotency_key = $2
                    RETURNING idempotency_key, intent_id, status, result_message,
                              modified_qty, order_id,
                              EXTRACT(EPOCH FROM cooldown_until) AS cooldown_until
                    """,
                    self.oms_id,
                    idempotency_key,
                    status.name,
                    reason,
                    order_id,
                )
            if not row:
                return None
            self._record_success()
            return self._intent_result_from_row(row)
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to resolve idempotency key {idempotency_key}: {e}")
            return None

    # ------------------------------------------------------------------
    # Durable Protective Stops
    # ------------------------------------------------------------------

    async def upsert_stop(self, stop: ProtectiveStop) -> Optional[ProtectiveStop]:
        """Create or update the durable protective stop for an allocation."""
        if not self._is_connected():
            return None
        try:
            row = await self.pool.fetchrow(
                """
                INSERT INTO protective_stops (
                    stop_id, oms_id, strategy_id, symbol, side, qty, stop_price,
                    trigger_price_source, protection_mode, status,
                    broker_order_id, broker_order_date, entry_intent_id, entry_order_id,
                    exit_intent_id, idempotency_key, activated_at, triggered_at,
                    last_checked_at, last_price, last_error, failure_count,
                    config_hash, source_metadata
                ) VALUES (
                    $1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13::uuid, $14, $15::uuid, $16, $17, $18,
                    $19, $20, $21, $22, $23, $24::jsonb
                )
                ON CONFLICT (stop_id) DO UPDATE SET
                    qty = CASE
                        WHEN protective_stops.status IN ('TRIGGERED', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED', 'FILLED', 'CANCELLED', 'FAILED')
                        THEN protective_stops.qty ELSE EXCLUDED.qty END,
                    stop_price = CASE
                        WHEN protective_stops.status IN ('TRIGGERED', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED', 'FILLED', 'CANCELLED', 'FAILED')
                        THEN protective_stops.stop_price ELSE EXCLUDED.stop_price END,
                    trigger_price_source = EXCLUDED.trigger_price_source,
                    protection_mode = EXCLUDED.protection_mode,
                    status = CASE
                        WHEN protective_stops.status IN ('TRIGGERED', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED', 'FILLED', 'CANCELLED', 'FAILED')
                        THEN protective_stops.status ELSE EXCLUDED.status END,
                    broker_order_id = COALESCE(EXCLUDED.broker_order_id, protective_stops.broker_order_id),
                    broker_order_date = COALESCE(EXCLUDED.broker_order_date, protective_stops.broker_order_date),
                    entry_intent_id = COALESCE(EXCLUDED.entry_intent_id, protective_stops.entry_intent_id),
                    entry_order_id = COALESCE(EXCLUDED.entry_order_id, protective_stops.entry_order_id),
                    activated_at = COALESCE(protective_stops.activated_at, EXCLUDED.activated_at, NOW()),
                    last_error = CASE
                        WHEN protective_stops.status IN ('TRIGGERED', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED', 'FILLED', 'CANCELLED', 'FAILED')
                        THEN protective_stops.last_error ELSE NULL END,
                    config_hash = EXCLUDED.config_hash,
                    source_metadata = protective_stops.source_metadata || EXCLUDED.source_metadata,
                    updated_at = NOW()
                RETURNING *
                """,
                stop.stop_id,
                stop.oms_id,
                stop.strategy_id,
                stop.symbol,
                stop.side,
                stop.qty,
                stop.stop_price,
                stop.trigger_price_source,
                stop.protection_mode,
                stop.status,
                stop.broker_order_id,
                stop.broker_order_date,
                self._normalize_uuid(stop.entry_intent_id),
                stop.entry_order_id,
                self._normalize_uuid(stop.exit_intent_id),
                stop.idempotency_key,
                stop.activated_at,
                stop.triggered_at,
                stop.last_checked_at,
                stop.last_price,
                stop.last_error,
                stop.failure_count,
                stop.config_hash,
                json.dumps(stop.source_metadata or {}, sort_keys=True, default=str),
            )
            self._record_success()
            return stop_from_row(row) if row else None
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to upsert protective stop {stop.stop_id}: {e}")
            return None

    async def load_active_stops(self) -> List[ProtectiveStop]:
        if not self._is_connected():
            return []
        try:
            rows = await self.pool.fetch(
                """
                SELECT *
                FROM protective_stops
                WHERE oms_id = $1
                  AND status IN ('PENDING', 'ACTIVE', 'TRIGGERED_PENDING_EXECUTION')
                ORDER BY updated_at ASC
                """,
                self.oms_id,
            )
            self._record_success()
            return [stop_from_row(row) for row in rows]
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load active protective stops: {e}")
            return []

    async def load_stop_for_allocation(self, strategy_id: str, symbol: str) -> Optional[ProtectiveStop]:
        if not self._is_connected():
            return None
        try:
            row = await self.pool.fetchrow(
                """
                SELECT *
                FROM protective_stops
                WHERE oms_id = $1
                  AND strategy_id = $2
                  AND symbol = $3
                  AND status IN ('PENDING', 'ACTIVE', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                self.oms_id,
                strategy_id.upper().strip(),
                str(symbol).zfill(6),
            )
            self._record_success()
            return stop_from_row(row) if row else None
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load protective stop for {strategy_id}/{symbol}: {e}")
            return None

    async def mark_active(self, stop_id: str, broker_order_id: Optional[str] = None) -> None:
        await self._update_stop_status(stop_id, StopStatus.ACTIVE.value, broker_order_id=broker_order_id, activated=True)

    async def mark_triggered(self, stop_id: str, trigger_price: float, triggered_at: datetime) -> bool:
        if not self._is_connected():
            return False
        try:
            row = await self.pool.fetchrow(
                """
                UPDATE protective_stops
                SET status = 'TRIGGERED_PENDING_EXECUTION',
                    triggered_at = $2,
                    last_checked_at = $2,
                    last_price = $3,
                    updated_at = NOW()
                WHERE stop_id = $1::uuid
                  AND oms_id = $4
                  AND status IN ('PENDING', 'ACTIVE')
                RETURNING stop_id
                """,
                stop_id,
                triggered_at,
                trigger_price,
                self.oms_id,
            )
            self._record_success()
            return row is not None
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to mark protective stop triggered {stop_id}: {e}")
            return False

    async def mark_exit_submitted(
        self,
        stop_id: str,
        exit_intent_id: Optional[str],
        order_id: Optional[str],
        idempotency_key: Optional[str] = None,
    ) -> None:
        if not self._is_connected():
            return
        if not order_id:
            await self.touch_stop_check(
                stop_id,
                checked_at=datetime.now(),
                last_price=None,
                last_error="exit_not_submitted:no_broker_order_id",
            )
            return
        try:
            await self.pool.execute(
                """
                UPDATE protective_stops
                SET status = 'EXIT_SUBMITTED',
                    exit_intent_id = COALESCE($2::uuid, exit_intent_id),
                    idempotency_key = COALESCE(idempotency_key, $4),
                    broker_order_id = COALESCE($3, broker_order_id),
                    updated_at = NOW()
                WHERE stop_id = $1::uuid
                  AND oms_id = $5
                """,
                stop_id,
                self._normalize_uuid(exit_intent_id),
                order_id,
                idempotency_key,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to mark protective stop exit submitted {stop_id}: {e}")

    async def mark_filled(self, stop_id: str) -> None:
        await self._update_stop_status(stop_id, StopStatus.FILLED.value)

    async def mark_cancelled(self, stop_id: str, reason: str = "") -> None:
        await self._update_stop_status(stop_id, StopStatus.CANCELLED.value, last_error=reason)

    async def mark_failed(self, stop_id: str, reason: str) -> None:
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                UPDATE protective_stops
                SET status = 'FAILED',
                    last_error = $2,
                    failure_count = failure_count + 1,
                    updated_at = NOW()
                WHERE stop_id = $1::uuid
                  AND oms_id = $3
                """,
                stop_id,
                reason,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to mark protective stop failed {stop_id}: {e}")

    async def touch_stop_check(
        self,
        stop_id: str,
        *,
        checked_at: datetime,
        last_price: float,
        last_error: Optional[str] = None,
    ) -> None:
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                UPDATE protective_stops
                SET last_checked_at = $2,
                    last_price = $3,
                    last_error = $4,
                    failure_count = CASE WHEN $4 IS NULL THEN failure_count ELSE failure_count + 1 END,
                    updated_at = NOW()
                WHERE stop_id = $1::uuid
                  AND oms_id = $5
                """,
                stop_id,
                checked_at,
                last_price,
                last_error,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to touch protective stop {stop_id}: {e}")

    async def update_stop_quantity(self, strategy_id: str, symbol: str, qty: int) -> Optional[ProtectiveStop]:
        if not self._is_connected():
            return None
        try:
            row = await self.pool.fetchrow(
                """
                UPDATE protective_stops
                SET qty = GREATEST($4, 0),
                    status = CASE
                        WHEN status IN ('TRIGGERED', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED')
                        THEN status
                        WHEN $4 <= 0 THEN 'CANCELLED'
                        ELSE 'ACTIVE'
                    END,
                    updated_at = NOW()
                WHERE oms_id = $1
                  AND strategy_id = $2
                  AND symbol = $3
                  AND status IN ('PENDING', 'ACTIVE', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED')
                RETURNING *
                """,
                self.oms_id,
                strategy_id.upper().strip(),
                str(symbol).zfill(6),
                int(qty or 0),
            )
            self._record_success()
            return stop_from_row(row) if row else None
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to update protective stop quantity {strategy_id}/{symbol}: {e}")
            return None

    async def mark_idempotency_ambiguous(
        self,
        idempotency_key: str,
        *,
        reason: str,
        order_id: Optional[str] = None,
        submit_ref: Optional[str] = None,
    ) -> None:
        """Persist operator-visible reconciliation ambiguity without resolving the reservation."""
        if not self._is_connected():
            return
        try:
            if await self._intent_submission_plan_supported():
                await self.pool.execute(
                    """
                    UPDATE intents
                    SET reservation_reconcile_status = 'AMBIGUOUS',
                        reservation_reconcile_message = $3,
                        order_id = COALESCE($4, order_id),
                        submit_ref = COALESCE($5, submit_ref),
                        processed_at = COALESCE(processed_at, NOW())
                    WHERE oms_id = $1
                      AND idempotency_key = $2
                    """,
                    self.oms_id,
                    idempotency_key,
                    reason,
                    order_id,
                    submit_ref,
                )
            else:
                await self.pool.execute(
                    """
                    UPDATE intents
                    SET result_message = $3,
                        order_id = COALESCE($4, order_id),
                        processed_at = COALESCE(processed_at, NOW())
                    WHERE oms_id = $1
                      AND idempotency_key = $2
                    """,
                    self.oms_id,
                    idempotency_key,
                    reason,
                    order_id,
                )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to mark idempotency ambiguity {idempotency_key}: {e}")

    async def _update_stop_status(
        self,
        stop_id: str,
        status: str,
        *,
        broker_order_id: Optional[str] = None,
        last_error: Optional[str] = None,
        activated: bool = False,
    ) -> None:
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                UPDATE protective_stops
                SET status = $2,
                    broker_order_id = COALESCE($3, broker_order_id),
                    last_error = $4,
                    activated_at = CASE WHEN $5 THEN COALESCE(activated_at, NOW()) ELSE activated_at END,
                    updated_at = NOW()
                WHERE stop_id = $1::uuid
                  AND oms_id = $6
                """,
                stop_id,
                status,
                broker_order_id,
                last_error,
                activated,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to update protective stop status {stop_id}: {e}")

    # ------------------------------------------------------------------
    # Order Recording
    # ------------------------------------------------------------------

    async def record_order(
        self,
        order: WorkingOrder,
        intent_id: Optional[str] = None,
        kis_order_id: Optional[str] = None,
        kis_order_date: Optional[str] = None,
    ) -> Optional[str]:
        """Record order creation or update. Returns OMS order UUID when available."""
        if not self._is_connected():
            return None

        broker_order_id = kis_order_id or order.order_id
        intent_uuid = self._normalize_uuid(intent_id)
        existing_oms_order_id = self._normalize_uuid(order.oms_order_id)
        if existing_oms_order_id is None:
            existing_oms_order_id = await self._resolve_oms_order_id(broker_order_id)

        try:
            if existing_oms_order_id:
                await self.pool.execute(
                    """
                    UPDATE orders SET
                        strategy_id = $2,
                        symbol = $3,
                        side = $4,
                        order_type = $5,
                        qty = $6,
                        filled_qty = $7,
                        limit_price = COALESCE($8, limit_price),
                        status = $9,
                        kis_order_id = COALESCE($10, kis_order_id),
                        kis_order_date = COALESCE($11, kis_order_date),
                        intent_id = COALESCE($12::uuid, intent_id),
                        cancel_after_sec = COALESCE($13, cancel_after_sec),
                        meta = COALESCE(meta, '{}'::jsonb) || $14::jsonb,
                        last_update_at = NOW()
                    WHERE oms_order_id = $1::uuid
                    """,
                    existing_oms_order_id,
                    order.strategy_id,
                    order.symbol,
                    order.side,
                    order.order_type,
                    order.qty,
                    order.filled_qty,
                    order.price if order.order_type == "LIMIT" else None,
                    order.status.name,
                    broker_order_id,
                    kis_order_date,
                    intent_uuid,
                    int(order.cancel_after_sec) if order.cancel_after_sec else None,
                    self._order_meta(order),
                )
                order.oms_order_id = existing_oms_order_id
                self._record_success()
                return existing_oms_order_id

            oms_order_id = await self.pool.fetchval(
                """
                INSERT INTO orders (
                    strategy_id, symbol, side, order_type,
                    qty, filled_qty, limit_price, stop_price, status,
                    kis_order_id, kis_order_date, intent_id, cancel_after_sec,
                    submitted_at, oms_id, meta
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12::uuid, $13, NOW(), $14, $15::jsonb
                )
                RETURNING oms_order_id
                """,
                order.strategy_id,
                order.symbol,
                order.side,
                order.order_type,
                order.qty,
                order.filled_qty,
                order.price if order.order_type == "LIMIT" else None,
                None,  # stop_price
                order.status.name,
                broker_order_id,
                kis_order_date,
                intent_uuid,
                int(order.cancel_after_sec) if order.cancel_after_sec else None,
                self.oms_id,
                self._order_meta(order),
            )
            order.oms_order_id = str(oms_order_id) if oms_order_id else None
            self._record_success()
            return order.oms_order_id
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to record order: {e}")
            return None

    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_qty: int,
        avg_fill_price: Optional[float] = None,
    ) -> None:
        """Update order status and fill info."""
        if not self._is_connected():
            return
        oms_order_id = await self._resolve_oms_order_id(order_id)
        if oms_order_id is None:
            logger.warning(f"Skipping order status update: unresolved order_id={order_id}")
            return
        try:
            await self.pool.execute(
                """
                UPDATE orders SET
                    status = $2,
                    filled_qty = $3,
                    avg_fill_price = COALESCE($4, avg_fill_price),
                    last_update_at = NOW()
                WHERE oms_order_id = $1
                """,
                oms_order_id, status.name, filled_qty, avg_fill_price,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to update order status: {e}")

    # ------------------------------------------------------------------
    # Order Events
    # ------------------------------------------------------------------

    async def record_order_event(
        self,
        event_type: str,
        order_id: Optional[str] = None,
        intent_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        status_before: Optional[str] = None,
        status_after: Optional[str] = None,
    ) -> None:
        """Record an order event."""
        if not self._is_connected():
            return
        oms_order_id = await self._resolve_oms_order_id(order_id)
        intent_uuid = self._normalize_uuid(intent_id)
        try:
            await self.pool.execute(
                """
                INSERT INTO order_events (
                    oms_order_id, intent_id, strategy_id, symbol,
                    event_type, payload, status_before, status_after,
                    oms_id
                ) VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9)
                """,
                oms_order_id,
                intent_uuid,
                strategy_id,
                symbol,
                event_type,
                json.dumps(payload) if payload else None,
                status_before,
                status_after,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to record order event: {e}")

    # ------------------------------------------------------------------
    # Fill Recording
    # ------------------------------------------------------------------

    async def record_fill(
        self,
        kis_exec_id: str,
        order_id: str,
        strategy_id: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        fill_ts: datetime,
        commission: Optional[float] = None,
        tax: Optional[float] = None,
    ) -> None:
        """Record a fill. Idempotent by kis_exec_id."""
        if not self._is_connected():
            return
        oms_order_id = await self._resolve_oms_order_id(order_id)
        try:
            await self.pool.execute(
                """
                INSERT INTO fills (
                    kis_exec_id, oms_order_id, strategy_id, symbol,
                    side, qty, price, commission, tax, fill_ts,
                    oms_id
                ) VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (kis_exec_id) DO NOTHING
                """,
                kis_exec_id, oms_order_id, strategy_id, symbol,
                side, qty, price, commission, tax, fill_ts,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to record fill: {e}")

    # ------------------------------------------------------------------
    # Position & Allocation Sync
    # ------------------------------------------------------------------

    async def sync_position(self, pos: SymbolPosition) -> None:
        """Sync position state to database."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO positions (
                    symbol, real_qty, avg_price, hard_stop_px,
                    entry_lock_owner, entry_lock_until,
                    cooldown_until, vi_cooldown_until, frozen,
                    oms_id
                ) VALUES ($1, $2, $3, $4, $5, to_timestamp($6), to_timestamp($7), to_timestamp($8), $9, $10)
                ON CONFLICT (oms_id, symbol) DO UPDATE SET
                    real_qty = EXCLUDED.real_qty,
                    avg_price = EXCLUDED.avg_price,
                    hard_stop_px = EXCLUDED.hard_stop_px,
                    entry_lock_owner = EXCLUDED.entry_lock_owner,
                    entry_lock_until = EXCLUDED.entry_lock_until,
                    cooldown_until = EXCLUDED.cooldown_until,
                    vi_cooldown_until = EXCLUDED.vi_cooldown_until,
                    frozen = EXCLUDED.frozen,
                    last_update_at = NOW()
                """,
                pos.symbol,
                pos.real_qty,
                pos.avg_price,
                pos.hard_stop_px,
                pos.entry_lock_owner,
                pos.entry_lock_until,
                pos.cooldown_until,
                pos.vi_cooldown_until,
                pos.frozen,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to sync position: {e}")

    async def sync_allocation(self, symbol: str, alloc: StrategyAllocation) -> None:
        """Sync allocation state to database."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO allocations (
                    symbol, strategy_id, qty, cost_basis, entry_ts,
                    soft_stop_px, time_stop_ts,
                    oms_id
                ) VALUES ($1, $2, $3, $4, $5, $6, to_timestamp($7), $8)
                ON CONFLICT (oms_id, symbol, strategy_id) DO UPDATE SET
                    qty = EXCLUDED.qty,
                    cost_basis = EXCLUDED.cost_basis,
                    entry_ts = EXCLUDED.entry_ts,
                    soft_stop_px = EXCLUDED.soft_stop_px,
                    time_stop_ts = EXCLUDED.time_stop_ts,
                    last_update_at = NOW()
                """,
                symbol,
                alloc.strategy_id,
                alloc.qty,
                alloc.cost_basis,
                alloc.entry_ts,
                alloc.soft_stop_px,
                alloc.time_stop_ts,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to sync allocation: {e}")

    # ------------------------------------------------------------------
    # Risk Updates
    # ------------------------------------------------------------------

    async def update_daily_risk_portfolio(
        self,
        trade_date: date,
        equity_krw: float,
        buyable_cash_krw: float,
        realized_pnl_krw: float,
        unrealized_pnl_krw: float,
        gross_exposure_krw: float,
        positions_count: int,
        halted: bool = False,
        safe_mode: bool = False,
        regime: Optional[str] = None,
    ) -> None:
        """Update portfolio daily risk."""
        if not self._is_connected():
            return
        try:
            daily_pnl_pct = (realized_pnl_krw + unrealized_pnl_krw) / max(equity_krw, 1)
            gross_pct = gross_exposure_krw / max(equity_krw, 1) * 100
            await self.pool.execute(
                """
                INSERT INTO risk_daily_portfolio (
                    trade_date, equity_krw, buyable_cash_krw,
                    realized_pnl_krw, unrealized_pnl_krw, daily_pnl_pct,
                    gross_exposure_krw, gross_exposure_pct, positions_count,
                    halted, safe_mode, regime,
                    oms_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (oms_id, trade_date) DO UPDATE SET
                    equity_krw = EXCLUDED.equity_krw,
                    buyable_cash_krw = EXCLUDED.buyable_cash_krw,
                    realized_pnl_krw = EXCLUDED.realized_pnl_krw,
                    unrealized_pnl_krw = EXCLUDED.unrealized_pnl_krw,
                    daily_pnl_pct = EXCLUDED.daily_pnl_pct,
                    gross_exposure_krw = EXCLUDED.gross_exposure_krw,
                    gross_exposure_pct = EXCLUDED.gross_exposure_pct,
                    positions_count = EXCLUDED.positions_count,
                    halted = EXCLUDED.halted,
                    safe_mode = EXCLUDED.safe_mode,
                    regime = COALESCE(EXCLUDED.regime, risk_daily_portfolio.regime),
                    last_update_at = NOW()
                """,
                trade_date, int(equity_krw), int(buyable_cash_krw),
                int(realized_pnl_krw), int(unrealized_pnl_krw), daily_pnl_pct,
                int(gross_exposure_krw), gross_pct, positions_count,
                halted, safe_mode, regime,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to update portfolio risk: {e}")

    async def update_daily_risk_strategy(
        self,
        trade_date: date,
        strategy_id: str,
        realized_pnl_krw: float,
        unrealized_pnl_krw: float,
        trades_count: int,
        wins: int,
        losses: int,
        halted: bool = False,
    ) -> None:
        """Update strategy daily risk."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO risk_daily_strategy (
                    trade_date, strategy_id, realized_pnl_krw, unrealized_pnl_krw,
                    trades_count, wins, losses, halted,
                    oms_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (oms_id, trade_date, strategy_id) DO UPDATE SET
                    realized_pnl_krw = EXCLUDED.realized_pnl_krw,
                    unrealized_pnl_krw = EXCLUDED.unrealized_pnl_krw,
                    trades_count = EXCLUDED.trades_count,
                    wins = EXCLUDED.wins,
                    losses = EXCLUDED.losses,
                    halted = EXCLUDED.halted,
                    last_update_at = NOW()
                """,
                trade_date, strategy_id, int(realized_pnl_krw), int(unrealized_pnl_krw),
                trades_count, wins, losses, halted,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to update strategy risk: {e}")

    # ------------------------------------------------------------------
    # Strategy State
    # ------------------------------------------------------------------

    async def update_strategy_state(
        self,
        strategy_id: str,
        mode: str,
        symbols_hot: int = 0,
        symbols_warm: int = 0,
        symbols_cold: int = 0,
        positions_count: int = 0,
        last_error: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """Update strategy state (heartbeat from strategy)."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO strategy_state (
                    strategy_id, mode, symbols_hot, symbols_warm, symbols_cold,
                    positions_count, last_error, version, last_heartbeat_ts,
                    oms_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), $9)
                ON CONFLICT (oms_id, strategy_id) DO UPDATE SET
                    mode = EXCLUDED.mode,
                    symbols_hot = EXCLUDED.symbols_hot,
                    symbols_warm = EXCLUDED.symbols_warm,
                    symbols_cold = EXCLUDED.symbols_cold,
                    positions_count = EXCLUDED.positions_count,
                    last_error = EXCLUDED.last_error,
                    version = COALESCE(EXCLUDED.version, strategy_state.version),
                    last_heartbeat_ts = NOW(),
                    last_update_at = NOW()
                """,
                strategy_id, mode, symbols_hot, symbols_warm, symbols_cold,
                positions_count, last_error, version,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to update strategy state: {e}")

    # ------------------------------------------------------------------
    # OMS Heartbeat
    # ------------------------------------------------------------------

    async def heartbeat(
        self,
        equity_krw: float,
        buyable_cash_krw: float,
        daily_pnl_krw: float,
        daily_pnl_pct: float,
        safe_mode: bool,
        halt_new_entries: bool,
        kis_connected: bool,
        recon_status: str,
        drift_count: int,
        version: str = "2.0.0",
    ) -> None:
        """Update OMS heartbeat."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO oms_state (
                    oms_id, last_heartbeat_ts, equity_krw, buyable_cash_krw,
                    daily_pnl_krw, daily_pnl_pct, safe_mode, halt_new_entries,
                    kis_connected, last_recon_ts, recon_status,
                    allocation_drift_count, version, last_update_at
                ) VALUES ($1, NOW(), $2, $3, $4, $5, $6, $7, $8, NOW(), $9, $10, $11, NOW())
                ON CONFLICT (oms_id) DO UPDATE SET
                    last_heartbeat_ts = NOW(),
                    equity_krw = EXCLUDED.equity_krw,
                    buyable_cash_krw = EXCLUDED.buyable_cash_krw,
                    daily_pnl_krw = EXCLUDED.daily_pnl_krw,
                    daily_pnl_pct = EXCLUDED.daily_pnl_pct,
                    safe_mode = EXCLUDED.safe_mode,
                    halt_new_entries = EXCLUDED.halt_new_entries,
                    kis_connected = EXCLUDED.kis_connected,
                    last_recon_ts = NOW(),
                    recon_status = EXCLUDED.recon_status,
                    allocation_drift_count = EXCLUDED.allocation_drift_count,
                    version = EXCLUDED.version,
                    last_update_at = NOW()
                """,
                self.oms_id, int(equity_krw), int(buyable_cash_krw), int(daily_pnl_krw),
                daily_pnl_pct, safe_mode, halt_new_entries, kis_connected,
                recon_status, drift_count, version,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to update heartbeat: {e}")

    # ------------------------------------------------------------------
    # Trade Lifecycle
    # ------------------------------------------------------------------

    async def open_trade(
        self,
        strategy_id: str,
        symbol: str,
        direction: str,
        entry_qty: int,
        entry_price: float,
        entry_ts: datetime,
        entry_intent_id: str,
        setup_type: str = "",
        confidence: str = "",
    ) -> Optional[str]:
        """Open or accumulate into an existing trade. Returns trade_id."""
        if not self._is_connected():
            return None

        # Check for existing open trade (partial fill of same entry)
        existing_id = await self.find_open_trade(strategy_id, symbol)
        if existing_id:
            try:
                await self.pool.execute(
                    """
                    UPDATE trades SET
                        entry_price = (entry_price * entry_qty + $2 * $3)
                                      / (entry_qty + $3),
                        entry_qty = entry_qty + $3
                    WHERE trade_id = $1::uuid
                    """,
                    existing_id, entry_price, entry_qty,
                )
                self._record_success()
                logger.debug(f"Accumulated fill into trade {existing_id}: +{entry_qty}@{entry_price}")
                return existing_id
            except Exception as e:
                self._record_failure()
                logger.error(f"Failed to accumulate trade fill: {e}")
                return None

        # No existing trade — create new row
        trade_id = str(uuid.uuid4())
        try:
            await self.pool.execute(
                """
                INSERT INTO trades (
                    trade_id, strategy_id, symbol, direction,
                    entry_qty, entry_price, entry_ts, entry_intent_id,
                    setup_type, confidence, status,
                    oms_id
                ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::uuid, $9, $10, 'OPEN', $11)
                """,
                trade_id, strategy_id, symbol, direction,
                entry_qty, entry_price, entry_ts, entry_intent_id,
                setup_type, confidence,
                self.oms_id,
            )
            self._record_success()
            logger.debug(f"Opened trade {trade_id}: {symbol} {direction} {entry_qty}@{entry_price}")
            return trade_id
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to open trade: {e}")
            return None

    async def close_trade(
        self,
        trade_id: str,
        exit_qty: int,
        exit_price: float,
        exit_ts: datetime,
        exit_intent_id: str,
        exit_reason: str = "",
        realized_pnl: float = 0.0,
    ) -> None:
        """Record a partial or full exit on a trade.

        Accumulates exit_qty and VWAP exit_price. Sets status=CLOSED
        when cumulative exit_qty >= entry_qty.
        """
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                UPDATE trades SET
                    exit_price = CASE
                        WHEN COALESCE(exit_qty, 0) = 0 THEN $3
                        ELSE (exit_price * exit_qty + $3 * $2)
                             / (exit_qty + $2)
                    END,
                    exit_qty = COALESCE(exit_qty, 0) + $2,
                    exit_ts = $4,
                    exit_intent_id = $5::uuid,
                    exit_reason = $6,
                    realized_pnl_krw = COALESCE(realized_pnl_krw, 0) + $7,
                    status = CASE
                        WHEN COALESCE(exit_qty, 0) + $2 >= entry_qty THEN 'CLOSED'
                        ELSE status
                    END,
                    closed_at = CASE
                        WHEN COALESCE(exit_qty, 0) + $2 >= entry_qty THEN NOW()
                        ELSE closed_at
                    END
                WHERE trade_id = $1::uuid
                """,
                trade_id, exit_qty, exit_price, exit_ts,
                exit_intent_id, exit_reason, realized_pnl,
            )
            self._record_success()
            logger.debug(f"Exit fill on trade {trade_id}: {exit_qty}@{exit_price} reason={exit_reason}")
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to record exit on trade: {e}")

    async def record_trade_marks(
        self,
        trade_id: str,
        duration_seconds: int,
        mae_pct: float,
        mfe_pct: float,
        capture_ratio: float,
    ) -> None:
        """Record MAE/MFE metrics for a trade."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO trade_marks (trade_id, duration_seconds, mae_pct, mfe_pct, capture_ratio)
                VALUES ($1::uuid, $2, $3, $4, $5)
                ON CONFLICT (trade_id) DO UPDATE SET
                    duration_seconds = EXCLUDED.duration_seconds,
                    mae_pct = EXCLUDED.mae_pct,
                    mfe_pct = EXCLUDED.mfe_pct,
                    capture_ratio = EXCLUDED.capture_ratio,
                    computed_at = NOW()
                """,
                trade_id, duration_seconds, mae_pct, mfe_pct, capture_ratio,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to record trade marks: {e}")

    async def find_open_trade(
        self,
        strategy_id: str,
        symbol: str,
    ) -> Optional[str]:
        """Find an open trade for strategy+symbol. Returns trade_id if found."""
        if not self._is_connected():
            return None
        try:
            row = await self.pool.fetchrow(
                """
                SELECT trade_id FROM trades
                WHERE strategy_id = $1 AND symbol = $2 AND oms_id = $3 AND status = 'OPEN'
                ORDER BY entry_ts DESC LIMIT 1
                """,
                strategy_id, symbol, self.oms_id,
            )
            self._record_success()
            return str(row['trade_id']) if row else None
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to find open trade: {e}")
            return None

    async def get_strategy_trade_stats(
        self,
        trade_date: date,
    ) -> Dict[str, Dict[str, int]]:
        """Get completed trade counts, wins, and losses by strategy for a date."""
        if not self._is_connected():
            return {}
        try:
            rows = await self.pool.fetch(
                """
                SELECT strategy_id,
                       COUNT(*) AS trades,
                       COUNT(*) FILTER (WHERE realized_pnl_krw > 0) AS wins,
                       COUNT(*) FILTER (WHERE realized_pnl_krw <= 0) AS losses
                FROM trades
                WHERE oms_id = $1
                  AND entry_ts::date = $2
                  AND status = 'CLOSED'
                GROUP BY strategy_id
                """,
                self.oms_id, trade_date,
            )
            self._record_success()
            return {
                row['strategy_id']: {
                    'trades': row['trades'],
                    'wins': row['wins'],
                    'losses': row['losses'],
                }
                for row in rows
            }
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to get strategy trade stats: {e}")
            return {}

    async def load_daily_realized_pnl(
        self,
        trade_date: date,
    ) -> Dict[str, float]:
        """Load per-strategy realized PnL from today's closed trades.

        Used on startup to restore in-memory state after mid-day restart.
        """
        if not self._is_connected():
            return {}
        try:
            rows = await self.pool.fetch(
                """
                SELECT strategy_id, SUM(realized_pnl_krw) AS total_pnl
                FROM trades
                WHERE oms_id = $1
                  AND entry_ts::date = $2
                  AND status = 'CLOSED'
                  AND realized_pnl_krw IS NOT NULL
                GROUP BY strategy_id
                """,
                self.oms_id, trade_date,
            )
            self._record_success()
            return {row['strategy_id']: float(row['total_pnl']) for row in rows}
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load daily realized PnL: {e}")
            return {}

    # ------------------------------------------------------------------
    # Recon Log
    # ------------------------------------------------------------------

    async def log_recon(
        self,
        recon_type: str,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        before_value: Optional[Dict] = None,
        after_value: Optional[Dict] = None,
        action: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log a reconciliation event."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO recon_log (
                    recon_type, symbol, strategy_id,
                    before_value, after_value, action, details,
                    oms_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                recon_type, symbol, strategy_id,
                json.dumps(before_value) if before_value else None,
                json.dumps(after_value) if after_value else None,
                action, details,
                self.oms_id,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to log recon: {e}")

    # ------------------------------------------------------------------
    # State Loading (startup)
    # ------------------------------------------------------------------

    async def load_positions(self) -> Dict[str, SymbolPosition]:
        """Load positions from database on startup."""
        if not self._is_connected():
            return {}
        try:
            rows = await self.pool.fetch(
                "SELECT * FROM positions WHERE oms_id = $1 AND (real_qty != 0 OR frozen = TRUE)",
                self.oms_id,
            )
            positions = {}
            for row in rows:
                pos = SymbolPosition(
                    symbol=row["symbol"],
                    real_qty=row["real_qty"],
                    avg_price=float(row["avg_price"]) if row["avg_price"] else 0.0,
                    hard_stop_px=float(row["hard_stop_px"]) if row["hard_stop_px"] else None,
                    entry_lock_owner=row["entry_lock_owner"],
                    entry_lock_until=row["entry_lock_until"].timestamp() if row["entry_lock_until"] else None,
                    cooldown_until=row["cooldown_until"].timestamp() if row["cooldown_until"] else None,
                    vi_cooldown_until=row["vi_cooldown_until"].timestamp() if row["vi_cooldown_until"] else None,
                    frozen=row["frozen"],
                )
                positions[row["symbol"]] = pos
            self._record_success()
            logger.info(f"Loaded {len(positions)} positions from database")
            return positions
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load positions: {e}")
            return {}

    async def load_allocations(self) -> Dict[str, Dict[str, StrategyAllocation]]:
        """Load allocations from database on startup."""
        if not self._is_connected():
            return {}
        try:
            rows = await self.pool.fetch(
                "SELECT * FROM allocations WHERE oms_id = $1 AND qty > 0",
                self.oms_id,
            )
            allocs: Dict[str, Dict[str, StrategyAllocation]] = {}
            for row in rows:
                symbol = row["symbol"]
                if symbol not in allocs:
                    allocs[symbol] = {}
                allocs[symbol][row["strategy_id"]] = StrategyAllocation(
                    strategy_id=row["strategy_id"],
                    qty=row["qty"],
                    cost_basis=float(row["cost_basis"]) if row["cost_basis"] else 0.0,
                    entry_ts=row["entry_ts"],
                    soft_stop_px=float(row["soft_stop_px"]) if row["soft_stop_px"] else None,
                    time_stop_ts=row["time_stop_ts"].timestamp() if row["time_stop_ts"] else None,
                )
            self._record_success()
            logger.info(f"Loaded allocations for {len(allocs)} symbols from database")
            return allocs
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load allocations: {e}")
            return {}

    async def load_working_orders(self) -> List[WorkingOrder]:
        """Load working orders from database on startup."""
        if not self._is_connected():
            return []
        try:
            rows = await self.pool.fetch(
                """
                SELECT * FROM orders
                WHERE oms_id = $1 AND status IN ('WORKING', 'PARTIAL', 'SUBMITTING')
                """,
                self.oms_id,
            )
            orders = []
            for row in rows:
                meta = self._jsonb_dict(row["meta"])
                working_price = (
                    float(row["limit_price"])
                    if row["limit_price"]
                    else float(row["avg_fill_price"])
                    if row["avg_fill_price"]
                    else float(meta.get("working_price") or 0.0)
                )
                orders.append(WorkingOrder(
                    order_id=row["kis_order_id"] or str(row["oms_order_id"]),
                    symbol=row["symbol"],
                    side=row["side"],
                    qty=row["qty"],
                    filled_qty=row["filled_qty"],
                    price=working_price,
                    order_type=row["order_type"],
                    status=OrderStatus[row["status"]],
                    strategy_id=row["strategy_id"],
                    created_at=row["created_at"],
                    updated_at=row["last_update_at"],
                    cancel_after_sec=row["cancel_after_sec"],
                    intent_id=str(row["intent_id"]) if row["intent_id"] else None,
                    idempotency_key=str(meta.get("idempotency_key") or "") or None,
                    submit_ref=str(meta.get("submit_ref") or "") or None,
                    branch=str(meta.get("branch") or ""),
                    submit_ts=float(meta.get("submit_ts") or 0.0) or time.time(),
                    oms_order_id=str(row["oms_order_id"]),
                    risk_stop_px=float(meta["risk_stop_px"]) if meta.get("risk_stop_px") is not None else None,
                    risk_hard_stop_px=float(meta["risk_hard_stop_px"]) if meta.get("risk_hard_stop_px") is not None else None,
                ))
            self._record_success()
            logger.info(f"Loaded {len(orders)} working orders from database")
            return orders
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load working orders: {e}")
            return []

    async def load_oms_state(self) -> Optional[Dict[str, Any]]:
        """Load OMS state from database on startup."""
        if not self._is_connected():
            return None
        try:
            row = await self.pool.fetchrow(
                "SELECT * FROM oms_state WHERE oms_id = $1",
                self.oms_id,
            )
            self._record_success()
            if row:
                return dict(row)
            return None
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load OMS state: {e}")
            return None
