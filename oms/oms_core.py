"""
OMS Core: Main orchestrator that ties everything together.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Mapping
import asyncio
import inspect
import time
import uuid
from zoneinfo import ZoneInfo
from loguru import logger

from collections import defaultdict

from .intent import Intent, IntentResult, IntentStatus, IntentType, Urgency, RiskPayload
from .state import StateStore, WorkingOrder, OrderStatus, StrategyAllocation
from .risk import RiskGateway, RiskConfig, RiskDecision
from .arbitration import ArbitrationEngine, ArbitrationResult
from .planner import OrderPlanner
from .adapter import KISExecutionAdapter
from .persistence import OMSPersistence
from .stop_protection import (
    PriceObservation,
    ProtectiveStop,
    StopProtectionMode,
    StopStatus,
    TriggerPriceSource,
    utcnow,
)
from .stop_watcher import StopWatcher


# ---------------------------------------------------------------------------
# Idempotency store abstraction (swap InMemory for Redis/Postgres in prod)
# ---------------------------------------------------------------------------

class IdempotencyStore(ABC):
    """Abstract store for intent deduplication. Back with Redis/Postgres for persistence."""

    @abstractmethod
    def get(self, key: str) -> Optional[IntentResult]:
        ...

    @abstractmethod
    def put(self, key: str, result: IntentResult) -> None:
        ...

    @abstractmethod
    def remove(self, key: str) -> bool:
        """Remove a cached result. Returns True if key existed."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all cached results."""
        ...


class InMemoryIdempotencyStore(IdempotencyStore):
    def __init__(self):
        self._store: Dict[str, IntentResult] = {}

    def get(self, key: str) -> Optional[IntentResult]:
        return self._store.get(key)

    def put(self, key: str, result: IntentResult) -> None:
        self._store[key] = result

    def remove(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# OMS Core
# ---------------------------------------------------------------------------

UNKNOWN_STRATEGY = "_UNKNOWN_"
DRIFT_TOLERANCE = 0  # shares
BROKER_MISSING_GRACE_CYCLES = 2


def _allocation_payload(symbol: str, allocation: StrategyAllocation | None) -> Dict:
    if allocation is None:
        return {"symbol": str(symbol).zfill(6), "strategy_id": "", "qty": 0, "cost_basis": 0.0}
    return {
        "symbol": str(symbol).zfill(6),
        "strategy_id": allocation.strategy_id,
        "qty": allocation.qty,
        "cost_basis": allocation.cost_basis,
        "entry_ts": allocation.entry_ts.isoformat() if getattr(allocation.entry_ts, "isoformat", None) else allocation.entry_ts,
        "soft_stop_px": allocation.soft_stop_px,
        "time_stop_ts": allocation.time_stop_ts,
    }


def _intent_order_side(intent: Intent) -> str:
    if intent.intent_type == IntentType.ENTER:
        return "BUY"
    if intent.intent_type in {IntentType.EXIT, IntentType.REDUCE}:
        return "SELL"
    return ""


def _incremental_fill_price(
    wo: WorkingOrder,
    new_filled_qty: int,
    fill_delta: int,
    broker_avg_price: Optional[float],
) -> float:
    """Derive the newly filled slice price from broker VWAP when available."""
    if not broker_avg_price or broker_avg_price <= 0 or fill_delta <= 0:
        return float(wo.price or 0.0)
    previous_qty = max(int(wo.filled_qty or 0), 0)
    if previous_qty <= 0:
        return float(broker_avg_price)
    previous_avg = float(wo.price or 0.0)
    if previous_avg <= 0:
        return float(broker_avg_price)
    new_total = max(int(new_filled_qty or 0), previous_qty + fill_delta)
    incremental_notional = float(broker_avg_price) * new_total - previous_avg * previous_qty
    return max(incremental_notional / max(fill_delta, 1), 0.0)


def _working_order_stop_metadata(wo: WorkingOrder, intent: Optional[Intent] = None) -> tuple[Optional[float], Optional[float]]:
    """Return explicit entry stop metadata from the live intent or persisted order metadata."""
    if intent is not None and intent.risk_payload is not None:
        if intent.risk_payload.stop_px is not None:
            wo.risk_stop_px = intent.risk_payload.stop_px
        if intent.risk_payload.hard_stop_px is not None:
            wo.risk_hard_stop_px = intent.risk_payload.hard_stop_px
    return wo.risk_stop_px, wo.risk_hard_stop_px


def _stop_exit_epoch(stop: ProtectiveStop, observation: PriceObservation) -> int:
    triggered_at = getattr(stop, "triggered_at", None)
    if isinstance(triggered_at, datetime):
        return int(triggered_at.timestamp())
    if triggered_at:
        try:
            return int(float(triggered_at))
        except (TypeError, ValueError):
            pass
    return int(observation.timestamp or time.time())


_IDEMPOTENCY_MATCH_WINDOW_SEC = 30 * 60
_KST = ZoneInfo("Asia/Seoul")
_TERMINAL_STOP_STATUSES = {
    StopStatus.FILLED.value,
    StopStatus.CANCELLED.value,
    StopStatus.FAILED.value,
}
_ACTIVE_STOP_STATUSES = {
    StopStatus.PENDING.value,
    StopStatus.ACTIVE.value,
    StopStatus.TRIGGERED_PENDING_EXECUTION.value,
    StopStatus.EXIT_SUBMITTED.value,
}


def _coerce_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return float(value.timestamp())
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return float(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _broker_attr(order: Any, *names: str) -> str:
    for name in names:
        value = getattr(order, name, None)
        if value:
            return str(value).strip()
    metadata = getattr(order, "metadata", None) or getattr(order, "meta", None)
    if isinstance(metadata, Mapping):
        for name in names:
            value = metadata.get(name)
            if value:
                return str(value).strip()
    return ""


def _positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _price_matches(expected: Any, actual: Any) -> bool:
    expected_px = _positive_float(expected)
    actual_px = _positive_float(actual)
    if expected_px is None or actual_px is None:
        return False
    tolerance = max(1.0, abs(expected_px) * 0.0001)
    return abs(expected_px - actual_px) <= tolerance


def _first_positive_float(payload: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        parsed = _positive_float(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip() if ch.isdigit())


def _quote_epoch_from_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None:
        if parsed > 10_000_000_000:
            parsed /= 1000.0
        if parsed > 946_684_800:
            return parsed
    epoch = _coerce_epoch(value)
    if epoch is not None and epoch > 946_684_800:
        return epoch
    return None


def _quote_timestamp(payload: Mapping[str, Any]) -> Optional[float]:
    for key in (
        "quote_ts",
        "quote_timestamp",
        "provider_ts",
        "provider_timestamp",
        "created_ts",
        "timestamp",
        "stck_cntg_timestamp",
    ):
        epoch = _quote_epoch_from_value(payload.get(key))
        if epoch is not None:
            return epoch

    for key in ("quote_datetime", "provider_datetime", "updated_at", "last_updated_at", "datetime"):
        epoch = _quote_epoch_from_value(payload.get(key))
        if epoch is not None:
            return epoch

    date_digits = ""
    for key in ("stck_bsop_date", "bsop_date", "trading_date", "quote_date", "date"):
        date_digits = _digits(payload.get(key))
        if date_digits:
            break
    time_digits = ""
    for key in ("stck_cntg_hour", "stck_cntg_time", "cntg_hour", "quote_time", "time"):
        time_digits = _digits(payload.get(key))
        if time_digits:
            break
    if len(date_digits) == 6:
        date_digits = f"20{date_digits}"
    if len(date_digits) < 8 or len(time_digits) < 4:
        return None
    time_digits = time_digits[:6].ljust(6, "0")
    try:
        observed_at = datetime.strptime(f"{date_digits[:8]}{time_digits}", "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return observed_at.replace(tzinfo=_KST).timestamp()


class OMSCore:
    """
    OMS Core: Central order management system.

    Processes intents through:
    1. Validation + expiry check
    2. Risk checks
    3. Arbitration
    4. Order planning
    5. Execution → WorkingOrder (allocation updated on FILL, not submit)
    """

    def __init__(
        self,
        kis_api: 'KoreaInvestAPI',
        risk_config: Optional[RiskConfig] = None,
        idempotency_store: Optional[IdempotencyStore] = None,
        persistence: Optional[OMSPersistence] = None,
        event_emitter: Optional[object] = None,
        sector_map: Optional[Mapping[str, str]] = None,
        require_persistence: bool = False,
    ):
        self.state = StateStore()
        self.risk = RiskGateway(
            self.state,
            risk_config or RiskConfig(),
            price_getter=lambda s: kis_api.get_last_price(s),
            sector_map=dict(sector_map or {}),
        )
        self.arbitration = ArbitrationEngine(self.state)
        self.planner = OrderPlanner()
        self.adapter = KISExecutionAdapter(kis_api)
        self.persistence = persistence
        self.event_emitter = event_emitter
        self.require_persistence = require_persistence

        self._idem = idempotency_store or InMemoryIdempotencyStore()
        self._reconcile_task: Optional[asyncio.Task] = None
        self._stop_watcher: Optional[StopWatcher] = None
        self._symbol_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._rejection_counts: Dict[str, int] = {}
        self.stop_protection_status: str = "unknown"
        self.unprotected_positions_count: int = 0
        self.active_stop_count: int = 0
        self.triggered_stop_count: int = 0
        self.stop_watcher_last_check_ts: Optional[float] = None
        self.stop_watcher_price_stale_count: int = 0
        self.stop_protection_last_error: str = ""
        self._stop_protection_status_source: str = ""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _set_stop_protection_status(self, status: str, *, last_error: str = "", source: str) -> None:
        self.stop_protection_status = status
        self.stop_protection_last_error = last_error
        self._stop_protection_status_source = source

    def _has_independent_stop_protection_fault(self) -> bool:
        status = str(self.stop_protection_status or "").lower().strip()
        if status not in {"error", "degraded"}:
            return False
        if self._stop_protection_status_source == "watcher":
            return False
        return bool(self.stop_protection_last_error or getattr(self.risk, "halt_new_entries", False))

    async def submit_intent(self, intent: Intent) -> IntentResult:
        """Submit intent for processing. Main entry point for strategies."""
        oms_received_at = time.time()

        # 1. Idempotency check (outside lock — read-only)
        cached = self._idem.get(intent.idempotency_key)
        if cached is not None:
            logger.debug(f"Duplicate intent: {intent.idempotency_key}")
            self._emit_intent(intent, cached, phase="duplicate")
            return cached

        # 2. Validate (includes expiry enforcement)
        valid, error = intent.validate()
        if not valid:
            return await self._finalize(intent, IntentStatus.REJECTED, f"Validation failed: {error}")

        # Per-symbol mutex: prevents concurrent submits for same symbol
        async with self._symbol_locks[intent.symbol]:
            cached = self._idem.get(intent.idempotency_key)
            if cached is not None:
                logger.debug(f"Duplicate intent after lock: {intent.idempotency_key}")
                self._emit_intent(intent, cached, phase="duplicate")
                return cached
            reserved_result = await self._reserve_idempotency(intent)
            if reserved_result is not None:
                if reserved_result.status in {IntentStatus.EXECUTED, IntentStatus.ACCEPTED}:
                    self._idem.put(intent.idempotency_key, reserved_result)
                self._emit_reconciliation(
                    "IDEMPOTENCY_DUPLICATE",
                    symbol=intent.symbol,
                    payload={
                        "idempotency_key": intent.idempotency_key,
                        "intent_id": intent.intent_id,
                        "existing_intent_id": reserved_result.intent_id,
                        "status": reserved_result.status.name,
                        "message": reserved_result.message,
                    },
                )
                self._emit_intent(intent, reserved_result, phase="duplicate")
                return reserved_result
            self._emit_reconciliation(
                "IDEMPOTENCY_RESERVED",
                symbol=intent.symbol,
                payload={"idempotency_key": intent.idempotency_key, "intent_id": intent.intent_id},
            )
            return await self._process_intent(intent, oms_received_at=oms_received_at)

    async def _process_intent(self, intent: Intent, oms_received_at: float = 0.0) -> IntentResult:
        """Process intent under per-symbol lock."""

        # 1. Dispatch operational intents
        if intent.intent_type == IntentType.CANCEL_ORDERS:
            return await self._handle_cancel_orders(intent)

        if intent.intent_type == IntentType.MODIFY_RISK:
            return await self._handle_modify_risk(intent)

        # 2. Risk check
        risk_result = self.risk.check(intent)
        self._emit_risk_decision(intent, risk_result)

        if risk_result.decision == RiskDecision.REJECT:
            self._release_lock_if_entry(intent)
            return await self._finalize(
                intent, IntentStatus.REJECTED, risk_result.reason,
                cooldown_until=time.time() + (risk_result.cooldown_sec or 0),
                blocking_positions=risk_result.blocking_positions,
                resource_conflict_type=risk_result.resource_conflict_type,
                oms_received_at=oms_received_at,
            )
        if risk_result.decision == RiskDecision.DEFER:
            self._release_lock_if_entry(intent)
            return await self._finalize(intent, IntentStatus.DEFERRED, risk_result.reason, oms_received_at=oms_received_at)

        # 3. Apply risk modifications
        final_qty = risk_result.modified_qty or intent.desired_qty or intent.target_qty

        # 4. Arbitration
        arb_result = self.arbitration.arbitrate(intent)
        if arb_result.result == ArbitrationResult.DEFER:
            return await self._finalize(intent, IntentStatus.DEFERRED, arb_result.reason, oms_received_at=oms_received_at)
        if arb_result.result == ArbitrationResult.CANCEL:
            self._release_lock_if_entry(intent)
            return await self._finalize(intent, IntentStatus.REJECTED, arb_result.reason, oms_received_at=oms_received_at)

        # 5. Plan + Execute
        result = await self._plan_and_execute(intent, final_qty, risk_result.modified_qty, oms_received_at=oms_received_at)

        # Release entry lock on rejection (execution failure)
        if result.status == IntentStatus.REJECTED:
            self._release_lock_if_entry(intent)

        return result

    async def _reserve_idempotency(self, intent: Intent) -> Optional[IntentResult]:
        if self.persistence is None:
            if self.require_persistence:
                return IntentResult(
                    intent_id=intent.intent_id,
                    status=IntentStatus.DEFERRED,
                    message="Durable idempotency reservation unavailable; no persistence backend configured",
                )
            return None
        is_connected = getattr(self.persistence, "_is_connected", None)
        if callable(is_connected) and not is_connected():
            if self.require_persistence:
                return IntentResult(
                    intent_id=intent.intent_id,
                    status=IntentStatus.DEFERRED,
                    message="Durable idempotency reservation unavailable; persistence is disconnected",
                )
            return None
        reserve = getattr(self.persistence, "reserve_intent", None)
        if not callable(reserve):
            if self.require_persistence:
                return IntentResult(
                    intent_id=intent.intent_id,
                    status=IntentStatus.DEFERRED,
                    message="Durable idempotency reservation unavailable; backend lacks reserve_intent",
                )
            return None
        result = reserve(intent)
        if inspect.isawaitable(result):
            return await result
        return result

    # ------------------------------------------------------------------
    # CANCEL_ORDERS handler
    # ------------------------------------------------------------------

    async def _handle_cancel_orders(self, intent: Intent) -> IntentResult:
        """Cancel working orders for strategy_id on symbol."""
        pos = self.state.get_position(intent.symbol)
        cancelled = 0

        # Query broker once for all orders (not per working order)
        orders_result = await self.adapter.get_orders()
        if orders_result.ok:
            broker_by_id = {bo.order_id: bo for bo in orders_result.data}
        else:
            logger.warning(f"Broker orders unavailable during cancel: {orders_result.error_message}")
            broker_by_id = {}

        for wo in list(pos.working_orders):
            if wo.strategy_id == intent.strategy_id:
                broker = broker_by_id.get(wo.order_id)
                prev_status = wo.status
                if broker:
                    final_delta = broker.filled_qty - wo.filled_qty
                    if final_delta > 0:
                        await self._apply_fill(wo, final_delta)
                        wo.filled_qty = broker.filled_qty

                result = await self.adapter.cancel_order(wo.order_id, wo.symbol, wo.qty - wo.filled_qty, branch=wo.branch)
                if result.success:
                    await self._finalize_working_order(
                        wo,
                        OrderStatus.CANCELLED,
                        prev_status,
                        "CANCELLED",
                        payload={"filled_qty": wo.filled_qty, "order_qty": wo.qty},
                    )
                    cancelled += 1

        return await self._finalize(
            intent, IntentStatus.EXECUTED,
            f"Cancelled {cancelled} order(s)",
        )

    # ------------------------------------------------------------------
    # MODIFY_RISK handler
    # ------------------------------------------------------------------

    async def _handle_modify_risk(self, intent: Intent) -> IntentResult:
        """Update risk overlays for a strategy's allocation."""
        pos = self.state.get_position(intent.symbol)
        alloc = pos.allocations.get(intent.strategy_id)

        if not alloc:
            return await self._finalize(intent, IntentStatus.REJECTED, "No allocation to modify")

        rp = intent.risk_payload
        if rp.stop_px is not None:
            alloc.soft_stop_px = rp.stop_px
        if rp.hard_stop_px is not None:
            pos.hard_stop_px = rp.hard_stop_px
        if intent.constraints.expiry_ts is not None:
            alloc.time_stop_ts = intent.constraints.expiry_ts

        # Persist allocation modification
        if self.persistence:
            await self.persistence.sync_allocation(intent.symbol, alloc)

        stop_px = rp.hard_stop_px if rp.hard_stop_px is not None else rp.stop_px
        if stop_px is not None and alloc.qty > 0:
            stored = await self._upsert_durable_stop(
                symbol=intent.symbol,
                strategy_id=intent.strategy_id,
                qty=alloc.qty,
                stop_price=stop_px,
                entry_intent_id=None,
                entry_order_id=None,
                source_metadata={
                    "source": "modify_risk",
                    "intent_id": intent.intent_id,
                    "old_stop_px": None,
                    "new_stop_px": stop_px,
                },
                event_type="STOP_UPDATED",
            )
            if stored is None and self._durable_stop_required():
                self.risk.halt_new_entries = True
                return await self._finalize(
                    intent,
                    IntentStatus.DEFERRED,
                    "Risk overlays updated but durable stop persistence failed; new entries halted",
                )

        return await self._finalize(intent, IntentStatus.EXECUTED, "Risk overlays updated")

    # ------------------------------------------------------------------
    # Plan + Execute (ENTER, EXIT, REDUCE, FLATTEN, SET_TARGET)
    # ------------------------------------------------------------------

    async def _plan_and_execute(
        self, intent: Intent, final_qty: int, was_modified: Optional[int],
        oms_received_at: float = 0.0,
    ) -> IntentResult:
        """Create order plan and execute via adapter."""
        current_price = await self._get_current_price(intent.symbol)

        if intent.intent_type == IntentType.ENTER:
            plan = self.planner.create_plan(
                symbol=intent.symbol, side="BUY", qty=final_qty,
                intent=intent, current_price=current_price,
            )
        elif intent.intent_type in (IntentType.EXIT, IntentType.FLATTEN):
            pos = self.state.get_position(intent.symbol)
            alloc_qty = pos.get_allocation(intent.strategy_id)
            if alloc_qty <= 0:
                # Check working BUY orders — cancel instead of sell
                pending = pos.working_qty(
                    strategy_id=intent.strategy_id, side="BUY"
                )
                if pending > 0:
                    return await self._handle_cancel_orders(intent)
                return await self._finalize(intent, IntentStatus.REJECTED, "No allocation to exit", oms_received_at=oms_received_at)
            # Respect desired_qty for partial exits, capped at allocation
            exit_qty = min(intent.desired_qty, alloc_qty) if intent.desired_qty else alloc_qty
            # Safety cap — never sell more than broker holds
            if exit_qty > pos.real_qty:
                logger.warning(
                    f"EXIT qty capped: {intent.symbol} alloc={alloc_qty} real_qty={pos.real_qty}"
                )
                exit_qty = pos.real_qty
            if exit_qty <= 0:
                return await self._finalize(
                    intent, IntentStatus.REJECTED,
                    f"No sellable shares: real_qty={pos.real_qty} (alloc={alloc_qty})",
                    oms_received_at=oms_received_at,
                )
            plan = self.planner.create_exit_plan(
                symbol=intent.symbol, qty=exit_qty,
                strategy_id=intent.strategy_id,
                intent_id=intent.intent_id, urgency=intent.urgency, intent=intent,
            )
        elif intent.intent_type == IntentType.REDUCE:
            reduce_qty = abs(final_qty)
            # Safety cap — never sell more than broker holds
            pos = self.state.get_position(intent.symbol)
            if reduce_qty > pos.real_qty:
                logger.warning(
                    f"REDUCE qty capped: {intent.symbol} requested={reduce_qty} real_qty={pos.real_qty}"
                )
                reduce_qty = pos.real_qty
            if reduce_qty <= 0:
                return await self._finalize(
                    intent, IntentStatus.REJECTED,
                    f"No sellable shares: real_qty={pos.real_qty}",
                    oms_received_at=oms_received_at,
                )
            plan = self.planner.create_exit_plan(
                symbol=intent.symbol, qty=reduce_qty,
                strategy_id=intent.strategy_id,
                intent_id=intent.intent_id, urgency=intent.urgency, intent=intent,
            )
        elif intent.intent_type == IntentType.SET_TARGET:
            # Compute delta = target_qty - current_allocation
            pos = self.state.get_position(intent.symbol)
            current_alloc = pos.get_allocation(intent.strategy_id)
            target_qty = intent.target_qty or 0
            delta = target_qty - current_alloc
            if delta == 0:
                return await self._finalize(intent, IntentStatus.EXECUTED, "Already at target", oms_received_at=oms_received_at)
            side = "BUY" if delta > 0 else "SELL"
            sell_qty = abs(delta)
            if side == "SELL":
                # Safety cap — never sell more than broker holds
                if sell_qty > pos.real_qty:
                    logger.warning(
                        f"SET_TARGET sell qty capped: {intent.symbol} delta={sell_qty} real_qty={pos.real_qty}"
                    )
                    sell_qty = pos.real_qty
                if sell_qty <= 0:
                    return await self._finalize(
                        intent, IntentStatus.REJECTED,
                        f"No sellable shares: real_qty={pos.real_qty} (alloc={current_alloc})",
                        oms_received_at=oms_received_at,
                    )
            plan = self.planner.create_plan(
                symbol=intent.symbol, side=side, qty=abs(delta),
                intent=intent, current_price=current_price,
            ) if delta > 0 else self.planner.create_exit_plan(
                symbol=intent.symbol, qty=sell_qty,
                strategy_id=intent.strategy_id,
                intent_id=intent.intent_id, urgency=intent.urgency, intent=intent,
            )
        else:
            return await self._finalize(intent, IntentStatus.REJECTED, f"Unsupported intent type: {intent.intent_type}", oms_received_at=oms_received_at)

        if plan.execution_style == "SYNTHETIC_STOP":
            if self._durable_stop_required() and not bool(self.risk.config.allow_synthetic_stop_only):
                self.risk.halt_new_entries = True
                return await self._finalize(
                    intent,
                    IntentStatus.DEFERRED,
                    "Synthetic stop-only routing is not live-safe without emergency override; new entries halted",
                    oms_received_at=oms_received_at,
                )
            return await self._finalize(
                intent,
                IntentStatus.ACCEPTED,
                "Synthetic stop accepted for trigger-then-submit routing; no broker limit order submitted",
                oms_received_at=oms_received_at,
            )

        order_price = plan.limit_price or current_price or 0.0
        sector_reserved = False
        if plan.side == "BUY":
            self.risk.reserve_sector(plan.symbol, plan.qty, order_price)
            sector_reserved = True

        submit_ref = f"OMS-{uuid.uuid4().hex[:12]}"
        if self.persistence:
            update_plan = getattr(self.persistence, "update_intent_submission_plan", None)
            if callable(update_plan):
                maybe = update_plan(
                    intent,
                    side=plan.side,
                    planned_qty=plan.qty,
                    order_type=plan.order_type.name,
                    limit_price=plan.limit_price,
                    stop_price=plan.stop_price,
                    submit_ref=submit_ref,
                )
                if inspect.isawaitable(maybe):
                    await maybe

        # Execute
        try:
            exec_result = await self.adapter.submit_order(
                symbol=plan.symbol, side=plan.side, qty=plan.qty,
                order_type=plan.order_type.name,
                limit_price=plan.limit_price, stop_price=plan.stop_price,
                intent_id=intent.intent_id,
                idempotency_key=intent.idempotency_key,
                submit_ref=submit_ref,
            )
        except Exception:
            if sector_reserved:
                self.risk.unreserve_sector(plan.symbol, plan.qty, order_price)
            raise
        order_submitted_at = time.time()

        if not exec_result.success:
            if sector_reserved:
                self.risk.unreserve_sector(plan.symbol, plan.qty, order_price)
            return await self._finalize(intent, IntentStatus.REJECTED, exec_result.message, oms_received_at=oms_received_at)

        # Track as WorkingOrder — allocation is updated on FILL, not here
        wo = WorkingOrder(
            order_id=exec_result.order_id,
            symbol=plan.symbol,
            side=plan.side,
            qty=plan.qty,
            price=plan.limit_price or current_price,
            order_type=plan.order_type.name,
            status=OrderStatus.WORKING,
            strategy_id=intent.strategy_id,
            cancel_after_sec=plan.cancel_after,
            intent_id=intent.intent_id,
            idempotency_key=intent.idempotency_key,
            submit_ref=submit_ref,
            risk_stop_px=intent.risk_payload.stop_px,
            risk_hard_stop_px=intent.risk_payload.hard_stop_px,
        )

        # Persist the broker order before any slower event/export work.
        order_persistence_failed = False
        if self.persistence:
            wo.oms_order_id = await self.persistence.record_order(wo, intent_id=intent.intent_id)
            order_persistence_failed = not bool(wo.oms_order_id)
            if order_persistence_failed and self.require_persistence:
                self.risk.halt_new_entries = True
                reason = "broker_success_order_persistence_failed"
                marker = getattr(self.persistence, "mark_intent_ambiguous", None)
                if callable(marker):
                    maybe = marker(intent, order_id=wo.order_id, submit_ref=submit_ref, reason=reason)
                    if inspect.isawaitable(maybe):
                        await maybe
                self._emit_reconciliation(
                    "IDEMPOTENCY_AMBIGUOUS",
                    symbol=plan.symbol,
                    payload={
                        "idempotency_key": intent.idempotency_key,
                        "intent_id": intent.intent_id,
                        "order_id": wo.order_id,
                        "submit_ref": submit_ref,
                        "reason": reason,
                    },
                )
            if not order_persistence_failed:
                await self.persistence.record_order_event(
                    "ORDER_SUBMITTED", order_id=wo.order_id, intent_id=intent.intent_id,
                    strategy_id=intent.strategy_id, symbol=plan.symbol,
                    payload={"submit_ref": submit_ref},
                    status_after="WORKING",
                )

        self.state.add_working_order(plan.symbol, wo)
        self._emit_order_event(
            wo,
            "ORDER_SUBMITTED",
            payload={"status_after": "WORKING", "order_submitted_at": order_submitted_at, "submit_ref": submit_ref},
            intent=intent,
        )

        if order_persistence_failed and self.require_persistence:
            return await self._finalize(
                intent,
                IntentStatus.DEFERRED,
                "Broker order accepted but OMS order persistence failed; reconciliation required before retry",
                order_id=exec_result.order_id,
                modified_qty=final_qty if was_modified else None,
                oms_received_at=oms_received_at,
                order_submitted_at=order_submitted_at,
            )

        return await self._finalize(
            intent, IntentStatus.EXECUTED,
            order_id=exec_result.order_id,
            modified_qty=final_qty if was_modified else None,
            oms_received_at=oms_received_at, order_submitted_at=order_submitted_at,
        )

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    async def _apply_fill(
        self,
        wo: WorkingOrder,
        fill_qty: int,
        intent: Optional[Intent] = None,
        *,
        inferred: bool = False,
        fill_price: Optional[float] = None,
    ) -> None:
        """Apply fill to allocation. real_qty is updated from broker sync only."""
        applied_price = float(fill_price if fill_price and fill_price > 0 else wo.price or 0.0)
        if applied_price > 0:
            wo.price = applied_price
        qty_delta = fill_qty if wo.side == "BUY" else -fill_qty
        fill_ts = datetime.now()
        exec_id = f"{wo.order_id}:{wo.filled_qty + fill_qty}"
        resolved_intent_id = intent.intent_id if intent else wo.intent_id
        before_pos = self.state.get_position(wo.symbol)
        before_alloc = _allocation_payload(wo.symbol, before_pos.allocations.get(wo.strategy_id))

        # Record realized P&L for sell fills
        fill_realized_pnl = 0.0
        if wo.side == "SELL":
            pos = self.state.get_position(wo.symbol)
            alloc = pos.allocations.get(wo.strategy_id)
            if alloc and alloc.cost_basis > 0:
                fill_realized_pnl = (applied_price - alloc.cost_basis) * fill_qty
                self.state.record_realized_pnl(fill_realized_pnl, strategy_id=wo.strategy_id)

        self.state.update_allocation(
            wo.symbol, wo.strategy_id, qty_delta,
            cost_basis=applied_price,
        )

        soft_stop_px, hard_stop_px = _working_order_stop_metadata(wo, intent)

        # Persist explicit entry stop metadata from either the live intent or
        # the rehydrated working order on every BUY fill path.
        if wo.side == "BUY" and (soft_stop_px is not None or hard_stop_px is not None):
            pos = self.state.get_position(wo.symbol)
            alloc = pos.allocations.get(wo.strategy_id) if pos else None
            if alloc:
                if soft_stop_px is not None:
                    alloc.soft_stop_px = soft_stop_px
                elif alloc.soft_stop_px is None:
                    alloc.soft_stop_px = hard_stop_px
            if hard_stop_px is not None:
                pos.hard_stop_px = hard_stop_px

        durable_stop_attempt = self.persistence is not None or (
            isinstance(self, OMSCore) and self._durable_stop_required()
        )
        if durable_stop_attempt and wo.side == "BUY":
            stop_px = hard_stop_px or soft_stop_px
            if stop_px is not None:
                pos = self.state.get_position(wo.symbol)
                alloc = pos.allocations.get(wo.strategy_id) if pos else None
                stored = await self._upsert_durable_stop(
                    symbol=wo.symbol,
                    strategy_id=wo.strategy_id,
                    qty=alloc.qty if alloc else fill_qty,
                    stop_price=stop_px,
                    entry_intent_id=resolved_intent_id,
                    entry_order_id=wo.order_id,
                    source_metadata={
                        "source": "buy_fill",
                        "fill_qty": fill_qty,
                        "order_id": wo.order_id,
                        "intent_id": resolved_intent_id,
                        "risk_stop_px": soft_stop_px,
                        "risk_hard_stop_px": hard_stop_px,
                    },
                    event_type="STOP_CREATED",
                )
                if stored is None and self._durable_stop_required():
                    self.risk.halt_new_entries = True
                    self._set_stop_protection_status(
                        "error",
                        last_error="durable stop upsert failed after BUY fill",
                        source="durable_stop",
                    )
        elif durable_stop_attempt and wo.side == "SELL":
            await self._sync_durable_stop_after_exit_fill(wo)

        # Update OMS risk gateway sector exposure on fills
        if wo.side == "BUY":
            self.risk.on_sector_fill(wo.symbol, fill_qty, applied_price)
        else:
            self.risk.on_sector_close(wo.symbol, fill_qty, applied_price)
        after_pos = self.state.get_position(wo.symbol)
        after_alloc = _allocation_payload(wo.symbol, after_pos.allocations.get(wo.strategy_id))
        self._emit_fill(
            wo,
            fill_qty,
            intent=intent,
            inferred=inferred,
            payload={
                "kis_exec_id": exec_id,
                "fill_ts": fill_ts.isoformat(),
                "filled_qty_before": wo.filled_qty,
                "filled_qty_after": wo.filled_qty + fill_qty,
                "order_qty": wo.qty,
                "commission": None,
                "tax": None,
                "fill_price": applied_price,
                "previous_allocation": before_alloc,
                "new_allocation": after_alloc,
                "realized_pnl": fill_realized_pnl,
            },
        )
        self._emit_position_snapshot("fill_applied")
        self._emit_allocation_snapshot("fill_applied")
        self._emit_portfolio_snapshot("fill_applied")

        # Note: real_qty updated from broker position sync in _reconcile to avoid double-credit
        logger.info(f"Fill applied: {wo.symbol} {wo.side} {fill_qty} for {wo.strategy_id}")

        # Persist fill and allocation
        if self.persistence:
            await self.persistence.record_fill(
                kis_exec_id=exec_id, order_id=wo.order_id,
                strategy_id=wo.strategy_id, symbol=wo.symbol,
                side=wo.side, qty=fill_qty, price=applied_price,
                fill_ts=fill_ts,
            )
            pos = self.state.get_position(wo.symbol)
            alloc = pos.allocations.get(wo.strategy_id)
            if alloc:
                await self.persistence.sync_allocation(wo.symbol, alloc)

            # Trade lifecycle tracking
            if wo.side == "BUY":
                # Entry fill → open trade
                setup_type = intent.risk_payload.rationale_code if intent else ""
                confidence = intent.risk_payload.confidence if intent else ""
                if resolved_intent_id:
                    await self.persistence.open_trade(
                        strategy_id=wo.strategy_id,
                        symbol=wo.symbol,
                        direction="LONG",
                        entry_qty=fill_qty,
                        entry_price=applied_price,
                        entry_ts=fill_ts,
                        entry_intent_id=resolved_intent_id,
                        setup_type=setup_type,
                        confidence=confidence,
                    )
            else:
                # Exit fill → close trade
                trade_id = await self.persistence.find_open_trade(wo.strategy_id, wo.symbol)
                if trade_id and resolved_intent_id:
                    exit_reason = intent.risk_payload.rationale_code if intent else "exit"
                    await self.persistence.close_trade(
                        trade_id=trade_id,
                        exit_qty=fill_qty,
                        exit_price=applied_price,
                        exit_ts=fill_ts,
                        exit_intent_id=resolved_intent_id,
                        exit_reason=exit_reason,
                        realized_pnl=fill_realized_pnl,
                    )

    def _remaining_qty(self, wo: WorkingOrder) -> int:
        """Get remaining unfilled quantity for a working order."""
        return max(wo.qty - wo.filled_qty, 0)

    def _release_sector_reservation(self, wo: WorkingOrder, qty: Optional[int] = None) -> None:
        """Release any unfilled BUY reservation held in sector exposure tracking."""
        if wo.side != "BUY":
            return
        release_qty = self._remaining_qty(wo) if qty is None else max(qty, 0)
        if release_qty <= 0:
            return
        self.risk.unreserve_sector(wo.symbol, release_qty, wo.price)

    async def _finalize_working_order(
        self,
        wo: WorkingOrder,
        final_status: OrderStatus,
        prev_status: OrderStatus,
        event_type: str,
        payload: Optional[Dict] = None,
    ) -> None:
        """Finalize a working order and persist its terminal state."""
        wo.status = final_status
        wo.updated_at = datetime.now()
        if final_status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            self._release_sector_reservation(wo)
            # Evict idempotency cache for unfilled SELL orders so exits can be retried,
            # but respect the rejection cap — if _finalize() already cached after 5
            # rejections, evicting here would undo that cap and create an infinite loop.
            if wo.side == "SELL" and wo.idempotency_key and wo.filled_qty == 0:
                if self._rejection_counts.get(wo.idempotency_key, 0) >= 5:
                    logger.warning(
                        f"Idem eviction skipped: {wo.idempotency_key} "
                        f"(rejection cap reached, {self._rejection_counts[wo.idempotency_key]}x)"
                    )
                elif self._idem.remove(wo.idempotency_key):
                    logger.info(
                        f"Idem evicted: {wo.idempotency_key} "
                        f"(SELL {final_status.name}, 0 filled)"
                    )
        self.state.remove_working_order(wo.symbol, wo.order_id)
        self.state.release_entry_lock(wo.symbol, wo.strategy_id)
        self._emit_order_event(
            wo,
            event_type,
            payload={
                "payload": payload or {},
                "status_before": prev_status.name,
                "status_after": final_status.name,
                "filled_qty": wo.filled_qty,
                "order_qty": wo.qty,
            },
        )
        self._emit_reconciliation(
            "ORDER_TERMINAL",
            symbol=wo.symbol,
            payload={
                "action": event_type,
                "order_id": wo.order_id,
                "strategy_id": wo.strategy_id,
                "side": wo.side,
                "before_value": {"status": prev_status.name, "filled_qty": wo.filled_qty},
                "after_value": {"status": final_status.name, "filled_qty": wo.filled_qty},
                "reason": dict(payload or {}).get("reason", event_type),
            },
        )
        if self.persistence:
            await self.persistence.record_order_event(
                event_type,
                order_id=wo.order_id,
                intent_id=wo.intent_id,
                strategy_id=wo.strategy_id,
                symbol=wo.symbol,
                payload=payload,
                status_before=prev_status.name,
                status_after=final_status.name,
            )
            await self.persistence.update_order_status(
                wo.order_id, final_status, wo.filled_qty, wo.price,
            )

    def _durable_stop_required(self) -> bool:
        if bool(getattr(self.risk.config, "stop_protection_emergency_override", False)):
            return False
        if not bool(getattr(self.risk.config, "require_durable_stops", True)):
            return False
        return bool(self.require_persistence)

    def _persistence_callable(self, name: str):
        if self.persistence is None:
            return None
        method = getattr(self.persistence, name, None)
        if not callable(method):
            return None
        if type(self.persistence).__module__.startswith("unittest.mock") and name not in vars(self.persistence):
            return None
        return method

    def _default_stop_mode(self, symbol: str) -> str:
        raw = str(getattr(self.risk.config, "default_stop_protection_mode", "oms_watcher") or "oms_watcher")
        mode = raw.upper().replace("-", "_")
        if mode == StopProtectionMode.BROKER_NATIVE.value:
            if self.adapter.supports_native_stop(symbol):
                return StopProtectionMode.BROKER_NATIVE.value
            logger.warning(f"Broker-native stops unverified for {symbol}; falling back to OMS watcher")
            return StopProtectionMode.OMS_WATCHER.value
        if mode in {"SYNTHETIC", "SYNTHETIC_ONLY"}:
            return StopProtectionMode.SYNTHETIC_ONLY.value
        return StopProtectionMode.OMS_WATCHER.value

    async def _upsert_durable_stop(
        self,
        *,
        symbol: str,
        strategy_id: str,
        qty: int,
        stop_price: float,
        entry_intent_id: Optional[str],
        entry_order_id: Optional[str],
        source_metadata: Mapping[str, Any],
        event_type: str,
    ) -> Optional[ProtectiveStop]:
        if stop_price <= 0 or qty <= 0:
            return None
        mode = self._default_stop_mode(symbol)
        if mode == StopProtectionMode.SYNTHETIC_ONLY.value and self._durable_stop_required():
            self.risk.halt_new_entries = True
            self._set_stop_protection_status(
                "error",
                last_error="synthetic stop-only protection is not live-safe",
                source="durable_stop",
            )
            return None
        if self.persistence is None:
            return None
        oms_id = getattr(self.persistence, "oms_id", "primary")
        existing_active = await self._load_stop_for_allocation(strategy_id, symbol)
        stop = ProtectiveStop.for_allocation(
            oms_id=oms_id,
            strategy_id=strategy_id,
            symbol=symbol,
            qty=qty,
            stop_price=stop_price,
            trigger_price_source=TriggerPriceSource.LAST.value,
            protection_mode=mode,
            status=StopStatus.ACTIVE.value if mode == StopProtectionMode.OMS_WATCHER.value else StopStatus.PENDING.value,
            entry_intent_id=entry_intent_id,
            entry_order_id=entry_order_id,
            config_hash=str(getattr(self.risk.config, "default_stop_protection_mode", "")),
            source_metadata=source_metadata,
        )
        if existing_active is not None and str(existing_active.status).upper() in _ACTIVE_STOP_STATUSES:
            stop.stop_id = existing_active.stop_id
        store_upsert = self._persistence_callable("upsert_stop")
        if not callable(store_upsert):
            return None
        stored = store_upsert(stop)
        if inspect.isawaitable(stored):
            stored = await stored
        terminal_returned = (
            stored is not None
            and str(getattr(stored, "status", "") or "").upper() in _TERMINAL_STOP_STATUSES
        )
        if stored is None or terminal_returned:
            self._set_stop_protection_status(
                "error",
                last_error=(
                    "protective stop upsert returned a terminal row; active protection was not created"
                    if terminal_returned
                    else "protective stop persistence failed"
                ),
                source="durable_stop",
            )
            if self._durable_stop_required():
                self.risk.halt_new_entries = True
            return None
        self._emit_stop_event(event_type, stored, payload=dict(source_metadata or {}))
        await self._record_stop_event(event_type, stored, dict(source_metadata or {}))
        self.active_stop_count = max(self.active_stop_count, 1)
        self._set_stop_protection_status(
            "ok" if self.unprotected_positions_count == 0 else "degraded",
            source="durable_stop",
        )
        return stored

    async def _sync_durable_stop_after_exit_fill(self, wo: WorkingOrder) -> None:
        if self.persistence is None:
            return
        pos = self.state.get_position(wo.symbol)
        alloc = pos.allocations.get(wo.strategy_id)
        remaining_qty = int(alloc.qty) if alloc else 0
        updater = self._persistence_callable("update_stop_quantity")
        if callable(updater):
            result = updater(wo.strategy_id, wo.symbol, remaining_qty)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                self._emit_stop_event(
                    "STOP_FILLED" if remaining_qty <= 0 else "STOP_UPDATED",
                    result,
                    payload={"source": "sell_fill", "remaining_qty": remaining_qty, "order_id": wo.order_id},
                )
        if wo.idempotency_key and str(wo.idempotency_key).startswith("STOP:") and remaining_qty <= 0:
            stop = await self._load_stop_for_allocation(wo.strategy_id, wo.symbol)
            if stop is not None:
                marker = self._persistence_callable("mark_filled")
                if callable(marker):
                    maybe = marker(stop.stop_id)
                    if inspect.isawaitable(maybe):
                        await maybe

    async def _load_stop_for_allocation(self, strategy_id: str, symbol: str) -> Optional[ProtectiveStop]:
        if self.persistence is None:
            return None
        loader = self._persistence_callable("load_stop_for_allocation")
        if not callable(loader):
            return None
        result = loader(strategy_id, symbol)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _submit_stop_exit(self, stop: ProtectiveStop, observation: PriceObservation) -> IntentResult:
        async with self._symbol_locks[stop.symbol]:
            pos = self.state.get_position(stop.symbol)
            alloc_qty = pos.get_allocation(stop.strategy_id)
            exit_qty = min(max(int(stop.qty or 0), 0), max(int(alloc_qty or 0), 0), max(int(pos.real_qty or 0), 0))
            if exit_qty <= 0:
                if self.persistence:
                    marker = self._persistence_callable("mark_cancelled")
                    if callable(marker):
                        maybe = marker(stop.stop_id, "stop_triggered_no_sellable_qty")
                        if inspect.isawaitable(maybe):
                            await maybe
                return IntentResult(stop.exit_intent_id or stop.stop_id, IntentStatus.CANCELLED, "Stop triggered but no sellable quantity")
            triggered_at = _stop_exit_epoch(stop, observation)
            idempotency_key = f"STOP:{getattr(self.persistence, 'oms_id', 'primary')}:{stop.stop_id}:{triggered_at}:{exit_qty}"
            stop.idempotency_key = idempotency_key
            intent = Intent(
                intent_type=IntentType.EXIT,
                strategy_id=stop.strategy_id,
                symbol=stop.symbol,
                desired_qty=exit_qty,
                urgency=Urgency.HIGH,
                risk_payload=RiskPayload(
                    stop_px=stop.stop_price,
                    rationale_code=f"protective_stop:{stop.stop_id}",
                    confidence="RED",
                ),
                idempotency_key=idempotency_key,
                metadata={
                    "stop_id": stop.stop_id,
                    "stop_price": stop.stop_price,
                    "trigger_price": observation.price,
                    "trigger_price_source": observation.source,
                    "stop_protection_mode": stop.protection_mode,
                },
            )
            cached = self._idem.get(intent.idempotency_key)
            if cached is not None:
                return cached
            reserved_result = await self._reserve_idempotency(intent)
            if reserved_result is not None:
                if reserved_result.status in {IntentStatus.EXECUTED, IntentStatus.ACCEPTED}:
                    self._idem.put(intent.idempotency_key, reserved_result)
                self._emit_reconciliation(
                    "IDEMPOTENCY_DUPLICATE",
                    symbol=intent.symbol,
                    payload={
                        "idempotency_key": intent.idempotency_key,
                        "intent_id": intent.intent_id,
                        "existing_intent_id": reserved_result.intent_id,
                        "status": reserved_result.status.name,
                    },
                )
                return reserved_result
            self._emit_reconciliation(
                "IDEMPOTENCY_RESERVED",
                symbol=intent.symbol,
                payload={"idempotency_key": intent.idempotency_key, "intent_id": intent.intent_id},
            )
            result = await self._process_intent(intent, oms_received_at=time.time())
            stop_exit_submitted = bool(result.order_id) and result.status not in {
                IntentStatus.REJECTED,
                IntentStatus.DEFERRED,
                IntentStatus.CANCELLED,
            }
            self._emit_stop_event(
                "STOP_EXIT_SUBMITTED" if stop_exit_submitted else "STOP_FAILED",
                stop,
                payload={
                    "exit_intent_id": result.intent_id,
                    "order_id": result.order_id,
                    "status": result.status.name,
                    "message": result.message,
                    "trigger_price": observation.price,
                },
            )
            return result

    def _emit_stop_event(self, event_type: str, stop: ProtectiveStop, *, payload: Optional[Dict] = None) -> None:
        emitter = self.event_emitter
        row = {
            "stop_id": stop.stop_id,
            "strategy_id": stop.strategy_id,
            "symbol": stop.symbol,
            "qty": stop.qty,
            "stop_price": stop.stop_price,
            "status": stop.status,
            "protection_mode": stop.protection_mode,
            **dict(payload or {}),
        }
        if emitter is not None:
            try:
                emit_stop = getattr(emitter, "emit_stop_event", None)
                if callable(emit_stop):
                    emit_stop(event_type, row)
                else:
                    emitter.emit_reconciliation(event_type, symbol=stop.symbol, payload=row)
            except Exception:
                pass

    async def _record_stop_event(self, event_type: str, stop: ProtectiveStop, payload: Optional[Dict] = None) -> None:
        if self.persistence is None:
            return
        recorder = self._persistence_callable("record_order_event")
        if not callable(recorder):
            return
        maybe = recorder(
            event_type,
            intent_id=stop.entry_intent_id or stop.exit_intent_id,
            strategy_id=stop.strategy_id,
            symbol=stop.symbol,
            payload={"stop_id": stop.stop_id, **dict(payload or {})},
            status_after=stop.status,
        )
        if inspect.isawaitable(maybe):
            await maybe

    async def _start_stop_watcher(self) -> None:
        if self.persistence is None or self._stop_watcher is not None:
            return
        if self._persistence_callable("load_active_stops") is None:
            return
        self._stop_watcher = StopWatcher(
            store=self.persistence,
            price_source=self._price_observation,
            exit_submitter=self._submit_stop_exit,
            trigger_notifier=self._notify_stop_triggered,
            stale_after_sec=float(getattr(self.risk.config, "stop_price_stale_after_sec", 30.0) or 30.0),
            interval_sec=float(getattr(self.risk.config, "stop_watcher_interval_sec", 5.0) or 5.0),
        )
        await self._stop_watcher.start()

    async def _notify_stop_triggered(self, stop: ProtectiveStop, observation: PriceObservation) -> None:
        payload = {
            "trigger_price": observation.price,
            "trigger_price_source": observation.source,
            "observation_ts": observation.timestamp,
            "market_open": observation.market_open,
            "executable": observation.executable,
        }
        self._emit_stop_event("STOP_TRIGGERED", stop, payload=payload)
        await self._record_stop_event("STOP_TRIGGERED", stop, payload)

    async def _price_observation(self, symbol: str) -> PriceObservation:
        quote = await self._get_current_price_payload(symbol)
        if isinstance(quote, Mapping):
            price = _first_positive_float(
                quote,
                "stck_prpr",
                "last",
                "last_price",
                "price",
                "close",
                "stck_prdy_clpr",
            )
            if price is not None:
                quote_ts = _quote_timestamp(quote)
                if quote_ts is not None:
                    market_open = self.adapter._is_order_session_open()
                    return PriceObservation(
                        symbol=str(symbol).zfill(6),
                        price=price,
                        timestamp=quote_ts,
                        source=TriggerPriceSource.LAST.value,
                        market_open=market_open,
                        executable=market_open,
                    )
        price = await self._get_current_price(symbol)
        return PriceObservation(
            symbol=str(symbol).zfill(6),
            price=float(price or 0.0),
            timestamp=0.0,
            source="UNVERIFIED_LAST",
            market_open=False,
            executable=False,
        )

    async def _reconcile_protective_stops_on_startup(self) -> None:
        active_stops: list[ProtectiveStop] = []
        if self.persistence:
            loader = self._persistence_callable("load_active_stops")
            if callable(loader):
                loaded = loader()
                if inspect.isawaitable(loaded):
                    loaded = await loaded
                active_stops = list(loaded or [])
        active_keys = {(stop.strategy_id, stop.symbol) for stop in active_stops}
        unprotected = []
        for symbol, pos in self.state.get_all_positions().items():
            for strategy_id, alloc in pos.allocations.items():
                if int(alloc.qty or 0) <= 0:
                    continue
                required_stop_px = alloc.soft_stop_px or pos.hard_stop_px
                if required_stop_px and (strategy_id, str(symbol).zfill(6)) not in active_keys:
                    unprotected.append((strategy_id, symbol, alloc.qty, required_stop_px))
        self.active_stop_count = len(active_stops)
        self.unprotected_positions_count = len(unprotected)
        self.triggered_stop_count = sum(1 for stop in active_stops if stop.status.startswith("TRIGGERED"))
        if unprotected:
            self.risk.halt_new_entries = True
            self._set_stop_protection_status(
                "error",
                last_error=f"{len(unprotected)} protected allocations lack durable stops",
                source="startup_reconcile",
            )
        else:
            self._set_stop_protection_status("ok", source="startup_reconcile")
        for stop in active_stops:
            self._emit_stop_event("STOP_ACTIVE", stop, payload={"source": "startup"})
            await self._record_stop_event("STOP_ACTIVE", stop, {"source": "startup"})

    def stop_health_payload(self) -> Dict[str, Any]:
        if self._stop_watcher is not None:
            health = self._stop_watcher.health
            if health.last_check_ts is not None:
                self.stop_watcher_last_check_ts = health.last_check_ts
                self.active_stop_count = health.active_stop_count
                self.triggered_stop_count = health.triggered_stop_count
            self.stop_watcher_price_stale_count = health.stale_price_count
            if self.unprotected_positions_count > 0:
                self._set_stop_protection_status(
                    "error",
                    last_error=(
                        self.stop_protection_last_error
                        or f"{self.unprotected_positions_count} protected allocations lack durable stops"
                    ),
                    source="startup_reconcile",
                )
            elif self._has_independent_stop_protection_fault():
                pass
            elif health.status in {"error", "degraded"}:
                self._set_stop_protection_status(health.status, last_error=health.last_error, source="watcher")
            elif health.stale_price_count > 0:
                self._set_stop_protection_status(
                    "degraded",
                    last_error=health.last_error or "stop watcher price stale",
                    source="watcher",
                )
            elif (
                health.status == "ok"
                and health.last_check_ts is not None
                and (
                    self._stop_protection_status_source == "watcher"
                    or str(self.stop_protection_status or "").lower().strip() in {"", "unknown", "ok"}
                )
            ):
                self._set_stop_protection_status("ok", source="watcher")
        age = None
        if self.stop_watcher_last_check_ts is not None:
            age = max(time.time() - self.stop_watcher_last_check_ts, 0.0)
        return {
            "stop_protection_status": self.stop_protection_status,
            "unprotected_positions_count": self.unprotected_positions_count,
            "active_stop_count": self.active_stop_count,
            "triggered_stop_count": self.triggered_stop_count,
            "stop_watcher_last_check_age_sec": age,
            "stop_watcher_price_stale_count": self.stop_watcher_price_stale_count,
            "stop_protection_last_error": self.stop_protection_last_error,
        }

    def _pending_idempotency_match(
        self,
        row: Mapping[str, Any],
        broker_orders: list[Any],
    ) -> tuple[Optional[Any], str, list[str]]:
        symbol = str(row.get("symbol") or "").zfill(6)
        side = str(row.get("planned_side") or ("BUY" if row.get("intent_type") == "ENTER" else "SELL")).upper()
        qty = int(row.get("planned_qty") or row.get("desired_qty") or row.get("target_qty") or 0)
        planned_type = str(row.get("planned_order_type") or "").upper().strip()
        planned_ref = str(row.get("submit_ref") or "").strip()
        planned_limit = row.get("planned_limit_price")
        planned_stop = row.get("planned_stop_price")
        created_ts = _coerce_epoch(row.get("created_ts") or row.get("created_at"))
        price_required = planned_type in {"LIMIT", "MARKETABLE_LIMIT", "CLOSE_AUCTION", "STOP_LIMIT"}
        planned_price = planned_limit if planned_limit is not None and str(planned_limit).strip() else planned_stop
        durable_order_id = str(row.get("order_id") or "").strip()

        base = [
            order
            for order in broker_orders
            if str(getattr(order, "symbol", "") or "").zfill(6) == symbol
            and str(getattr(order, "side", "") or "").upper() == side
            and int(getattr(order, "qty", 0) or 0) == qty
        ]
        if durable_order_id:
            direct = [
                order
                for order in broker_orders
                if str(getattr(order, "order_id", "") or "").strip() == durable_order_id
            ]
            if len(direct) == 1:
                order = direct[0]
                mismatches: list[str] = []
                if str(getattr(order, "symbol", "") or "").zfill(6) != symbol:
                    mismatches.append("symbol")
                if str(getattr(order, "side", "") or "").upper() != side:
                    mismatches.append("side")
                if int(getattr(order, "qty", 0) or 0) != qty:
                    mismatches.append("qty")
                if mismatches:
                    return None, f"order_id_candidate_mismatch:{','.join(mismatches)}", [durable_order_id]
                return order, "matched_order_id", [durable_order_id]
            if len(direct) > 1:
                return None, f"multiple_order_id_candidates:{len(direct)}", [durable_order_id]
            base_order_ids = [str(getattr(order, "order_id", "") or "") for order in base]
            if base_order_ids:
                return None, f"expected_order_id_not_visible:{durable_order_id};base_candidates:{len(base)}", base_order_ids
            return None, f"expected_order_id_not_visible:{durable_order_id}", []

        exact: list[Any] = []
        for order in base:
            broker_ref = _broker_attr(order, "submit_ref", "client_order_id", "client_order_key", "memo", "order_memo")
            if planned_ref and broker_ref != planned_ref:
                continue

            broker_type = _broker_attr(order, "order_type", "ord_type", "type")
            if not planned_type or not broker_type or broker_type.upper().strip() != planned_type:
                continue

            if price_required:
                if not _price_matches(planned_price, getattr(order, "price", None)):
                    continue

            broker_ts = _coerce_epoch(
                getattr(order, "created_ts", None)
                or getattr(order, "created_at", None)
                or getattr(order, "created_time", None)
            )
            if created_ts is None or broker_ts is None:
                continue
            if abs(broker_ts - created_ts) > _IDEMPOTENCY_MATCH_WINDOW_SEC:
                continue

            exact.append(order)

        order_ids = [str(getattr(order, "order_id", "") or "") for order in (exact or base)]
        if len(exact) == 1:
            return exact[0], "matched_exact_plan", order_ids
        if len(exact) > 1:
            return None, f"multiple_exact_candidates:{len(exact)}", order_ids
        if base:
            return None, f"broker_candidates_do_not_match_plan:{len(base)}", order_ids
        return None, "no_broker_candidate", []

    async def _mark_pending_idempotency_ambiguous(
        self,
        row: Mapping[str, Any],
        *,
        reason: str,
        candidate_order_ids: list[str],
    ) -> None:
        idempotency_key = str(row.get("idempotency_key") or "")
        if not idempotency_key:
            return
        marker = self._persistence_callable("mark_idempotency_ambiguous")
        if callable(marker):
            maybe = marker(
                idempotency_key,
                reason=f"{reason}; candidates={candidate_order_ids}",
                submit_ref=str(row.get("submit_ref") or "") or None,
            )
            if inspect.isawaitable(maybe):
                await maybe
        self._emit_reconciliation(
            "IDEMPOTENCY_AMBIGUOUS",
            symbol=str(row.get("symbol") or "").zfill(6),
            payload={
                "idempotency_key": idempotency_key,
                "intent_id": row.get("intent_id"),
                "reason": reason,
                "candidate_order_ids": candidate_order_ids,
                "submit_ref": row.get("submit_ref"),
                "planned_side": row.get("planned_side"),
                "planned_qty": row.get("planned_qty") or row.get("desired_qty") or row.get("target_qty"),
                "planned_order_type": row.get("planned_order_type"),
                "planned_limit_price": row.get("planned_limit_price"),
                "planned_stop_price": row.get("planned_stop_price"),
                "created_ts": row.get("created_ts"),
            },
        )

    async def reconcile_stale_pending_idempotency(self, stale_after_sec: float = 60.0) -> int:
        if self.persistence is None:
            return 0
        lister = self._persistence_callable("list_pending_idempotency")
        resolver = self._persistence_callable("resolve_idempotency")
        if not callable(lister) or not callable(resolver):
            return 0
        rows = lister(stale_after_sec=stale_after_sec)
        if inspect.isawaitable(rows):
            rows = await rows
        if not rows:
            return 0
        orders_result = await self.adapter.get_orders()
        broker_orders = list(orders_result.data) if orders_result.ok else []
        reconciled = 0
        for row in rows:
            symbol = str(row.get("symbol") or "").zfill(6)
            side = str(row.get("planned_side") or ("BUY" if row.get("intent_type") == "ENTER" else "SELL")).upper()
            qty = int(row.get("planned_qty") or row.get("desired_qty") or row.get("target_qty") or 0)
            match, match_reason, candidate_order_ids = self._pending_idempotency_match(row, broker_orders)
            if match is None:
                if candidate_order_ids:
                    await self._mark_pending_idempotency_ambiguous(
                        row,
                        reason=match_reason,
                        candidate_order_ids=candidate_order_ids,
                    )
                continue
            record_order = self._persistence_callable("record_order")
            if callable(record_order):
                reconciled_order = WorkingOrder(
                    order_id=match.order_id,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    filled_qty=int(getattr(match, "filled_qty", 0) or 0),
                    price=float(row.get("planned_limit_price") or getattr(match, "price", 0.0) or 0.0),
                    order_type=str(row.get("planned_order_type") or "LIMIT").upper(),
                    status=(
                        OrderStatus.PARTIAL
                        if int(getattr(match, "filled_qty", 0) or 0) > 0
                        else OrderStatus.WORKING
                    ),
                    strategy_id=str(row.get("strategy_id") or "").upper().strip(),
                    intent_id=str(row.get("intent_id") or "") or None,
                    idempotency_key=str(row.get("idempotency_key") or "") or None,
                    submit_ref=str(row.get("submit_ref") or "") or None,
                    risk_stop_px=float(row["stop_px"]) if row.get("stop_px") is not None else None,
                    risk_hard_stop_px=float(row["hard_stop_px"]) if row.get("hard_stop_px") is not None else None,
                )
                persisted_order_id = record_order(reconciled_order, intent_id=reconciled_order.intent_id)
                if inspect.isawaitable(persisted_order_id):
                    persisted_order_id = await persisted_order_id
                if not persisted_order_id and self.require_persistence:
                    self._emit_reconciliation(
                        "IDEMPOTENCY_AMBIGUOUS",
                        symbol=symbol,
                        payload={
                            "idempotency_key": row.get("idempotency_key"),
                            "order_id": match.order_id,
                            "reason": "reconciled_broker_order_persistence_failed",
                        },
                    )
                    continue
                if self._remaining_qty(reconciled_order) > 0:
                    pos = self.state.get_position(symbol)
                    if not any(existing.order_id == reconciled_order.order_id for existing in pos.working_orders):
                        self.state.add_working_order(symbol, reconciled_order)
            result = resolver(
                str(row.get("idempotency_key") or ""),
                status=IntentStatus.EXECUTED,
                reason=f"IDEMPOTENCY_RECONCILED broker order {match.order_id}",
                order_id=match.order_id,
            )
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                reconciled += 1
                self._idem.put(str(row.get("idempotency_key") or ""), result)
                self._emit_reconciliation(
                    "IDEMPOTENCY_RECONCILED",
                    symbol=symbol,
                    payload={
                        "idempotency_key": row.get("idempotency_key"),
                        "order_id": match.order_id,
                        "match_reason": match_reason,
                        "submit_ref": row.get("submit_ref"),
                        "planned_order_type": row.get("planned_order_type"),
                        "planned_limit_price": row.get("planned_limit_price"),
                        "planned_stop_price": row.get("planned_stop_price"),
                    },
                )
        return reconciled

    async def _sync_working_orders(self) -> Dict[str, 'BrokerOrder']:
        """Poll broker orders and reconcile with working order state.

        Returns:
            broker_by_id dict for reuse by _enforce_order_timeouts.
            Empty dict if broker query failed (sync skipped).
        """
        orders_result = await self.adapter.get_orders()
        if not orders_result.ok:
            logger.warning(f"Skipping order sync: broker query failed ({orders_result.error_message})")
            return {}

        broker_by_id = {bo.order_id: bo for bo in orders_result.data}

        for symbol, pos in self.state.get_all_positions().items():
            async with self._symbol_locks[symbol]:
                for wo in list(pos.working_orders):
                    broker = broker_by_id.get(wo.order_id)
                    prev_status = wo.status

                    if broker:
                        # Capture branch code for cancellation
                        if broker.branch and not wo.branch:
                            wo.branch = broker.branch
                        wo.missing_from_broker_count = 0
                        # Still working — detect partial fills via filled_qty delta
                        new_filled = broker.filled_qty
                        fill_delta = new_filled - wo.filled_qty
                        if fill_delta > 0:
                            fill_price = _incremental_fill_price(wo, new_filled, fill_delta, broker.price)
                            await self._apply_fill(wo, fill_delta, fill_price=fill_price)
                            if broker.price and broker.price > 0:
                                wo.price = broker.price
                            # Record partial fill event
                            if self.persistence and new_filled < wo.qty:
                                await self.persistence.record_order_event(
                                    "PARTIAL_FILL", order_id=wo.order_id,
                                    strategy_id=wo.strategy_id, symbol=wo.symbol,
                                    payload={
                                        "fill_qty": fill_delta,
                                        "fill_price": fill_price,
                                        "total_filled": new_filled,
                                        "order_qty": wo.qty,
                                    },
                                    status_before=prev_status.name, status_after="PARTIAL",
                                )
                        wo.filled_qty = new_filled
                        if new_filled >= wo.qty:
                            await self._finalize_working_order(
                                wo,
                                OrderStatus.FILLED,
                                prev_status,
                                "FILL",
                                payload={"filled_qty": wo.filled_qty, "order_qty": wo.qty},
                            )
                            continue
                        else:
                            wo.status = OrderStatus.PARTIAL if wo.filled_qty > 0 else OrderStatus.WORKING
                            if self.persistence and wo.status == OrderStatus.PARTIAL:
                                await self.persistence.update_order_status(
                                    wo.order_id, OrderStatus.PARTIAL, wo.filled_qty, wo.price,
                                )
                        wo.updated_at = datetime.now()
                    else:
                        # Order disappeared from broker — treat unfilled remainder
                        if wo.filled_qty >= wo.qty:
                            await self._finalize_working_order(
                                wo,
                                OrderStatus.FILLED,
                                prev_status,
                                "FILL",
                                payload={"filled_qty": wo.filled_qty, "order_qty": wo.qty},
                            )
                            continue
                        wo.missing_from_broker_count += 1
                        wo.updated_at = datetime.now()
                        logger.warning(
                            f"Working order missing from broker snapshot: {wo.symbol} "
                            f"{wo.order_id} ({wo.missing_from_broker_count} cycle(s))"
                        )
                        self._emit_reconciliation(
                            "WORKING_ORDER_MISSING",
                            symbol=wo.symbol,
                            payload={
                                "order_id": wo.order_id,
                                "strategy_id": wo.strategy_id,
                                "side": wo.side,
                                "filled_qty": wo.filled_qty,
                                "order_qty": wo.qty,
                                "missing_cycles": wo.missing_from_broker_count,
                            },
                        )

        return broker_by_id

    async def _reconcile_missing_working_orders(self, position_deltas: Dict[str, int]) -> None:
        """Infer missing-order terminal states from broker position deltas."""
        for symbol, pos in self.state.get_all_positions().items():
            missing_orders = [wo for wo in list(pos.working_orders) if wo.missing_from_broker_count > 0]
            if not missing_orders:
                continue

            async with self._symbol_locks[symbol]:
                buy_delta = max(position_deltas.get(symbol, 0), 0)
                sell_delta = max(-position_deltas.get(symbol, 0), 0)

                for wo in missing_orders:
                    prev_status = wo.status
                    fill_budget = buy_delta if wo.side == "BUY" else sell_delta
                    inferred_fill = min(self._remaining_qty(wo), fill_budget)

                    if inferred_fill > 0:
                        logger.warning(
                            f"Inferred fill for missing order {wo.order_id}: "
                            f"{wo.symbol} {wo.side} +{inferred_fill}"
                        )
                        await self._apply_fill(wo, inferred_fill, inferred=True)
                        wo.filled_qty += inferred_fill
                        self._emit_reconciliation(
                            "INFERRED_FILL",
                            symbol=wo.symbol,
                            payload={
                                "order_id": wo.order_id,
                                "strategy_id": wo.strategy_id,
                                "side": wo.side,
                                "fill_qty": inferred_fill,
                                "filled_qty": wo.filled_qty,
                                "order_qty": wo.qty,
                                "missing_cycles": wo.missing_from_broker_count,
                            },
                        )
                        if wo.side == "BUY":
                            buy_delta -= inferred_fill
                        else:
                            sell_delta -= inferred_fill

                        if wo.filled_qty < wo.qty:
                            wo.status = OrderStatus.PARTIAL
                            wo.updated_at = datetime.now()
                            if self.persistence:
                                await self.persistence.record_order_event(
                                    "PARTIAL_FILL",
                                    order_id=wo.order_id,
                                    intent_id=wo.intent_id,
                                    strategy_id=wo.strategy_id,
                                    symbol=wo.symbol,
                                    payload={
                                        "fill_qty": inferred_fill,
                                        "total_filled": wo.filled_qty,
                                        "order_qty": wo.qty,
                                        "inferred": True,
                                    },
                                    status_before=prev_status.name,
                                    status_after="PARTIAL",
                                )
                                await self.persistence.update_order_status(
                                    wo.order_id, OrderStatus.PARTIAL, wo.filled_qty, wo.price,
                                )
                            prev_status = wo.status

                    if wo.filled_qty >= wo.qty:
                        await self._finalize_working_order(
                            wo,
                            OrderStatus.FILLED,
                            prev_status,
                            "INFERRED_FILL",
                            payload={
                                "filled_qty": wo.filled_qty,
                                "order_qty": wo.qty,
                                "missing_cycles": wo.missing_from_broker_count,
                            },
                        )
                        continue

                    if wo.missing_from_broker_count >= BROKER_MISSING_GRACE_CYCLES:
                        if wo.filled_qty > 0:
                            logger.info(f"Partial cancel: {wo.symbol} filled {wo.filled_qty}/{wo.qty}")
                        await self._finalize_working_order(
                            wo,
                            OrderStatus.CANCELLED,
                            prev_status,
                            "CANCELLED",
                            payload={
                                "filled_qty": wo.filled_qty,
                                "order_qty": wo.qty,
                                "missing_cycles": wo.missing_from_broker_count,
                            },
                        )

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def start_reconciliation_loop(self, interval_sec: float = 5.0):
        """Start background reconciliation loop with adaptive interval.

        Interval adapts based on activity:
        - Active (working orders): interval_sec (default 5s)
        - Idle (no working orders): 15s
        - Rate-limited (cycle took >10s): 20s for 2 cycles then back to normal
        """
        consecutive_failures = 0
        max_failures_before_safe_mode = 5

        async def loop():
            nonlocal consecutive_failures
            cycle_count = 0
            rate_limit_cooldown = 0
            while True:
                cycle_start = time.time()
                try:
                    await self._reconcile(cycle_count)
                    consecutive_failures = 0
                    # Warn if equity still not loaded after first successful cycle
                    if cycle_count == 0 and self.state.equity <= 0:
                        logger.warning(
                            "EQUITY_ZERO: Reconciliation completed but equity=0 "
                            "— start() already attempted once; will retry next cycle"
                        )
                except Exception as e:
                    consecutive_failures += 1
                    logger.error(f"Reconciliation error ({consecutive_failures}x): {e}")
                    if consecutive_failures >= max_failures_before_safe_mode:
                        logger.critical(
                            f"Reconciliation failed {consecutive_failures}x consecutively — entering safe mode"
                        )
                        self.risk.safe_mode = True

                cycle_count += 1
                cycle_duration = time.time() - cycle_start

                # Adaptive interval
                if rate_limit_cooldown > 0:
                    sleep_sec = 20.0
                    rate_limit_cooldown -= 1
                elif cycle_duration > 10.0:
                    sleep_sec = 20.0
                    rate_limit_cooldown = 2
                elif not self.state.get_working_orders():
                    sleep_sec = 15.0
                else:
                    sleep_sec = interval_sec

                await asyncio.sleep(sleep_sec)

        self._reconcile_task = asyncio.create_task(loop())

    async def _enforce_order_timeouts(self, broker_by_id: Dict[str, 'BrokerOrder']) -> None:
        """Cancel orders that exceed their timeout.

        Args:
            broker_by_id: Already-fetched broker orders from _sync_working_orders.
                          Reused to avoid redundant API calls.
        """
        now = time.time()
        for pos in self.state.get_all_positions().values():
            for wo in list(pos.working_orders):
                if wo.cancel_after_sec and (now - wo.submit_ts) > wo.cancel_after_sec:
                    logger.info(f"Timeout cancel: {wo.symbol} {wo.order_id} after {wo.cancel_after_sec}s")
                    prev_status = wo.status

                    # Use already-fetched broker data (no extra API call)
                    broker = broker_by_id.get(wo.order_id)
                    if broker:
                        final_delta = broker.filled_qty - wo.filled_qty
                        if final_delta > 0:
                            await self._apply_fill(wo, final_delta)
                            wo.filled_qty = broker.filled_qty

                    result = await self.adapter.cancel_order(wo.order_id, wo.symbol, wo.qty - wo.filled_qty, branch=wo.branch)
                    if result.success:
                        await self._finalize_working_order(
                            wo,
                            OrderStatus.CANCELLED,
                            prev_status,
                            "TIMEOUT_CANCEL",
                            payload={
                                "timeout_sec": wo.cancel_after_sec,
                                "filled_qty": wo.filled_qty,
                                "order_qty": wo.qty,
                            },
                        )

    async def _reconcile(self, cycle_count: int = 0):
        """Full reconciliation cycle: orders → timeouts → positions → drift → account.

        Args:
            cycle_count: Current reconciliation cycle number, used to reduce
                frequency of non-critical API calls (e.g., buyable_cash).
        """
        # 1. Sync working orders (detect fills) — returns broker data for reuse
        broker_by_id = await self._sync_working_orders()

        # 2. Enforce order timeouts (reuse broker data, no extra API call)
        await self._enforce_order_timeouts(broker_by_id)

        # 3. Get positions + equity from a single API call (eliminates duplicate)
        positions_result, equity = await self.adapter.get_balance_snapshot()
        positions_ok = positions_result.ok
        broker_positions = positions_result.data if positions_ok else []

        if not positions_ok:
            logger.warning(f"Skipping position sync: broker query failed ({positions_result.error_message})")
        else:
            # Update equity from the same call that fetched positions
            if equity is not None:
                self.state.equity = equity

            tracked_positions = self.state.get_all_positions()
            tracked_symbols = set(tracked_positions)
            broker_positions_by_symbol = {bp.symbol: bp for bp in broker_positions}
            position_deltas: Dict[str, int] = {}

            for symbol in tracked_symbols | set(broker_positions_by_symbol):
                bp = broker_positions_by_symbol.get(symbol)
                new_qty = bp.qty if bp else 0
                new_avg_price = bp.avg_price if bp else 0.0

                async with self._symbol_locks[symbol]:
                    pos = self.state.get_position(symbol)
                    old_qty = pos.real_qty
                    position_deltas[symbol] = new_qty - old_qty

                    if pos.real_qty != new_qty or pos.avg_price != new_avg_price:
                        logger.info(f"Reconcile {symbol}: {pos.real_qty} -> {new_qty}")
                        self.state.update_position(symbol, real_qty=new_qty, avg_price=new_avg_price)
                        self._emit_reconciliation(
                            "POSITION_SYNC",
                            symbol=symbol,
                            payload={"before_value": {"real_qty": old_qty}, "after_value": {"real_qty": new_qty}},
                        )
                        if self.persistence:
                            await self.persistence.sync_position(pos)
                            await self.persistence.log_recon(
                                "POSITION_SYNC",
                                symbol=symbol,
                                before_value={"real_qty": old_qty},
                                after_value={"real_qty": new_qty},
                                action="UPDATED",
                            )

            await self._reconcile_missing_working_orders(position_deltas)

        # 4. Check allocation drift (only if positions were successfully fetched)
        if positions_ok:
            await self._check_allocation_drift()

            # 4b. Reconcile OMS risk gateway sector exposure from positions
            sector_positions = {
                bp.symbol: (bp.qty, bp.avg_price)
                for bp in broker_positions if bp.qty > 0
            }
            working_entry_orders = [
                (wo.symbol, self._remaining_qty(wo), wo.price)
                for wo in self.state.get_working_orders()
                if wo.side == "BUY" and self._remaining_qty(wo) > 0
            ]
            self.risk.reconcile_sector_exposure(sector_positions, working_entry_orders)
            self._emit_position_snapshot("reconcile")
            self._emit_allocation_snapshot("reconcile")
            self._emit_portfolio_snapshot("reconcile")

        # 5. Update buyable cash (only every 6th cycle — ~30s at 5s interval)
        if cycle_count % 6 == 0:
            buyable = await self.adapter.get_buyable_cash()
            if buyable is not None:
                self.state.buyable_cash = buyable

        # 6. Update daily PnL from broker positions
        prices = {bp.symbol: bp.current_price for bp in broker_positions}
        self.state.update_daily_pnl(prices)

        # 7. Update daily risk metrics
        if self.persistence:
            today = date.today()
            all_positions = self.state.get_all_positions()

            # Compute gross exposure
            gross_exposure = sum(
                pos.real_qty * prices.get(sym, pos.avg_price)
                for sym, pos in all_positions.items()
            )

            # Compute unrealized PnL from live marks
            unrealized_pnl = sum(
                (prices.get(sym, pos.avg_price) - pos.avg_price) * pos.real_qty
                for sym, pos in all_positions.items()
                if pos.real_qty > 0
            )

            # Update portfolio-level daily risk
            await self.persistence.update_daily_risk_portfolio(
                trade_date=today,
                equity_krw=self.state.equity,
                buyable_cash_krw=self.state.buyable_cash,
                realized_pnl_krw=self.state.daily_realized_pnl,
                unrealized_pnl_krw=unrealized_pnl,
                gross_exposure_krw=gross_exposure,
                positions_count=len(all_positions),
                halted=getattr(self.risk, 'halt_new_entries', False),
                safe_mode=getattr(self.risk, 'safe_mode', False),
                regime=getattr(self.risk, '_regime', None),
            )

            # Compute per-strategy unrealized PnL from allocations + live prices
            strategy_unrealized = {}
            for sym, pos in all_positions.items():
                px = prices.get(sym, pos.avg_price)
                for strat_id, alloc in pos.allocations.items():
                    if alloc.qty > 0 and alloc.cost_basis > 0:
                        pnl = (px - alloc.cost_basis) * alloc.qty
                        strategy_unrealized[strat_id] = strategy_unrealized.get(strat_id, 0.0) + pnl

            # Get today's trade stats from DB (wins/losses from correct trade data)
            trade_stats = await self.persistence.get_strategy_trade_stats(today)

            # Merge all strategy IDs that appear in any source
            all_strat_ids = set(self.state.strategy_realized_pnl) | set(strategy_unrealized) | set(trade_stats)

            for strat_id in all_strat_ids:
                ts = trade_stats.get(strat_id, {})
                await self.persistence.update_daily_risk_strategy(
                    trade_date=today,
                    strategy_id=strat_id,
                    realized_pnl_krw=self.state.strategy_realized_pnl.get(strat_id, 0.0),
                    unrealized_pnl_krw=strategy_unrealized.get(strat_id, 0.0),
                    trades_count=ts.get('trades', 0),
                    wins=ts.get('wins', 0),
                    losses=ts.get('losses', 0),
                    halted=strat_id in getattr(self.risk, '_paused_strategies', set()),
                )

        # 8. Heartbeat to database
        if self.persistence:
            drift_count = sum(
                1 for p in self.state.get_all_positions().values()
                if p.frozen
            )
            await self.persistence.heartbeat(
                equity_krw=self.state.equity,
                buyable_cash_krw=self.state.buyable_cash,
                daily_pnl_krw=self.state.daily_pnl,
                daily_pnl_pct=self.state.daily_pnl_pct,
                safe_mode=getattr(self.risk, 'safe_mode', False),
                halt_new_entries=getattr(self.risk, 'halt_new_entries', False),
                kis_connected=True,
                recon_status="warn" if drift_count > 0 else "ok",
                drift_count=drift_count,
            )
        self._emit_heartbeat("reconcile")

    async def _check_allocation_drift(self) -> None:
        """
        Detect and repair allocation drift.

        Policy:
        - If working orders exist: allow temporary drift (orders in flight).
        - Zero position: broker holds 0 shares, clear all allocations, unfreeze.
        - Already frozen: skip re-processing (prevents log spam).
        - Positive drift: assign to _UNKNOWN_, freeze.
        - Negative drift, single strategy: auto-correct allocation, don't freeze.
        - Negative drift, multi-strategy: freeze once, require admin correction.
        """
        for symbol, pos in self.state.get_all_positions().items():
            drift = pos.allocation_drift()
            unknown_qty = pos.get_allocation(UNKNOWN_STRATEGY)

            if abs(drift) <= DRIFT_TOLERANCE:
                # No drift — unfreeze if previously frozen and UNKNOWN cleared
                if pos.frozen:
                    if unknown_qty == 0:
                        pos.frozen = False
                        pos.allocations.pop(UNKNOWN_STRATEGY, None)
                        logger.info(f"Unfroze {symbol}: drift resolved")
                        self._emit_reconciliation(
                            "ALLOCATION_DRIFT",
                            symbol=symbol,
                            payload={
                                "action": "UNFROZEN",
                                "reason": "drift_resolved",
                                "drift": drift,
                                "frozen_before": True,
                                "frozen_after": False,
                            },
                        )
                        self._emit_position_snapshot("drift_unfrozen")
                        self._emit_allocation_snapshot("drift_unfrozen")
                        if self.persistence:
                            await self.persistence.sync_position(pos)
                            await self.persistence.log_recon(
                                "ALLOCATION_DRIFT", symbol=symbol, action="UNFROZEN",
                                details="Drift resolved, symbol unfrozen",
                            )
                continue

            if pos.has_working_orders():
                # Orders in flight — drift is expected, skip
                continue

            # Zero-position auto-cleanup: broker holds no shares
            if pos.real_qty == 0 and pos.total_allocated() > 0:
                cleared = {}
                for sid, alloc in list(pos.allocations.items()):
                    if alloc.qty > 0:
                        old_qty = self.state.set_allocation(symbol, sid, 0)
                        cleared[sid] = old_qty
                        if self.persistence:
                            await self.persistence.sync_allocation(symbol, alloc)
                pos.allocations.clear()
                was_frozen = pos.frozen
                pos.frozen = False
                logger.warning(
                    f"ZERO-POSITION CLEANUP {symbol}: broker holds 0 shares, "
                    f"cleared allocations {cleared}, unfrozen={was_frozen}"
                )
                self._emit_reconciliation(
                    "ALLOCATION_DRIFT",
                    symbol=symbol,
                    payload={
                        "action": "ZERO_POSITION_CLEANUP",
                        "reason": "broker_zero_position",
                        "before_value": {"allocations": cleared, "frozen": was_frozen},
                        "after_value": {"allocations": {}, "frozen": False},
                        "drift": drift,
                    },
                )
                self._emit_position_snapshot("zero_position_cleanup")
                self._emit_allocation_snapshot("zero_position_cleanup")
                if self.persistence:
                    await self.persistence.sync_position(pos)
                    await self.persistence.log_recon(
                        "ALLOCATION_DRIFT", symbol=symbol,
                        before_value={"allocations": cleared, "frozen": was_frozen},
                        after_value={"allocations": {}, "frozen": False},
                        action="ZERO_POSITION_CLEANUP",
                        details="Broker holds 0 shares, all allocations cleared",
                    )
                continue

            non_unknown = {
                sid: a for sid, a in pos.allocations.items()
                if sid != UNKNOWN_STRATEGY and a.qty > 0
            }

            # Already frozen — avoid repeated alerts, but still allow safe self-heal.
            if pos.frozen:
                if drift < 0 and unknown_qty == 0 and len(non_unknown) == 1:
                    strat_id, alloc = next(iter(non_unknown.items()))
                    old_qty = self.state.set_allocation(symbol, strat_id, pos.real_qty)
                    pos.allocations.pop(UNKNOWN_STRATEGY, None)
                    pos.frozen = False
                    logger.warning(
                        f"RECOVERED frozen negative drift {symbol}: "
                        f"{strat_id} allocation {old_qty} -> {alloc.qty}"
                    )
                    self._emit_reconciliation(
                        "ALLOCATION_DRIFT",
                        symbol=symbol,
                        payload={
                            "action": "AUTO_CORRECTED_FROZEN",
                            "reason": "single_strategy_negative_drift",
                            "strategy_id": strat_id,
                            "before_value": {"alloc_qty": old_qty, "frozen": True},
                            "after_value": {"alloc_qty": alloc.qty, "frozen": False},
                            "drift": drift,
                        },
                    )
                    self._emit_position_snapshot("drift_auto_corrected_frozen")
                    self._emit_allocation_snapshot("drift_auto_corrected_frozen")
                    if self.persistence:
                        await self.persistence.sync_allocation(symbol, alloc)
                        await self.persistence.sync_position(pos)
                        await self.persistence.log_recon(
                            "ALLOCATION_DRIFT", symbol=symbol,
                            before_value={"strategy": strat_id, "alloc_qty": old_qty, "frozen": True},
                            after_value={"strategy": strat_id, "alloc_qty": alloc.qty, "frozen": False},
                            action="AUTO_CORRECTED_FROZEN",
                            details=(
                                "Recovered frozen single-strategy negative drift after "
                                "_UNKNOWN_ was cleared"
                            ),
                        )
                continue

            # --- First detection: log and take action ---
            logger.critical(
                f"ALLOCATION DRIFT {symbol}: real={pos.real_qty} "
                f"allocated={pos.total_allocated()} drift={drift}"
            )

            if drift > 0:
                # Positive drift: broker has more shares than allocated — assign to UNKNOWN
                if UNKNOWN_STRATEGY not in pos.allocations:
                    pos.allocations[UNKNOWN_STRATEGY] = StrategyAllocation(strategy_id=UNKNOWN_STRATEGY)
                pos.allocations[UNKNOWN_STRATEGY].qty += drift
                pos.frozen = True
                self._emit_reconciliation(
                    "ALLOCATION_DRIFT",
                    symbol=symbol,
                    payload={
                        "action": "ASSIGNED_UNKNOWN",
                        "reason": "positive_drift",
                        "strategy_ids": list(non_unknown),
                        "before_value": {"total_allocated": pos.total_allocated() - drift, "frozen": False},
                        "after_value": {"total_allocated": pos.total_allocated(), "frozen": True},
                        "drift": drift,
                    },
                )
                self._emit_position_snapshot("drift_assigned_unknown")
                self._emit_allocation_snapshot("drift_assigned_unknown")
                if self.persistence:
                    await self.persistence.sync_allocation(symbol, pos.allocations[UNKNOWN_STRATEGY])
                    await self.persistence.sync_position(pos)
                    await self.persistence.log_recon(
                        "ALLOCATION_DRIFT", symbol=symbol,
                        before_value={"total_allocated": pos.total_allocated() - drift},
                        after_value={"total_allocated": pos.total_allocated()},
                        action="ASSIGNED_UNKNOWN",
                        details=f"Positive drift of {drift} assigned to _UNKNOWN_, symbol frozen",
                    )
            else:
                # Negative drift: allocations exceed real broker qty
                if len(non_unknown) == 1:
                    # Single strategy — safe to auto-correct
                    strat_id, alloc = next(iter(non_unknown.items()))
                    old_qty = self.state.set_allocation(symbol, strat_id, pos.real_qty)
                    logger.warning(
                        f"AUTO-CORRECTED negative drift {symbol}: "
                        f"{strat_id} allocation {old_qty} -> {alloc.qty}"
                    )
                    # If drift persists (e.g., _UNKNOWN_ allocation remains), freeze
                    if abs(pos.allocation_drift()) > DRIFT_TOLERANCE:
                        pos.frozen = True
                        logger.critical(
                            f"Drift persists after auto-correction {symbol}: "
                            f"drift={pos.allocation_drift()} — freezing"
                        )
                    self._emit_reconciliation(
                        "ALLOCATION_DRIFT",
                        symbol=symbol,
                        payload={
                            "action": "AUTO_CORRECTED",
                            "reason": "single_strategy_negative_drift",
                            "strategy_id": strat_id,
                            "before_value": {"alloc_qty": old_qty},
                            "after_value": {"alloc_qty": alloc.qty, "frozen": pos.frozen},
                            "drift": drift,
                        },
                    )
                    self._emit_position_snapshot("drift_auto_corrected")
                    self._emit_allocation_snapshot("drift_auto_corrected")
                    if self.persistence:
                        await self.persistence.sync_allocation(symbol, alloc)
                        await self.persistence.sync_position(pos)
                        await self.persistence.log_recon(
                            "ALLOCATION_DRIFT", symbol=symbol,
                            before_value={"strategy": strat_id, "alloc_qty": old_qty},
                            after_value={"strategy": strat_id, "alloc_qty": alloc.qty},
                            action="AUTO_CORRECTED",
                            details=f"Single-strategy auto-correction: {strat_id} {old_qty} -> {alloc.qty}",
                        )
                else:
                    # Multiple strategies — freeze and log once, require admin
                    pos.frozen = True
                    logger.critical(
                        f"NEGATIVE DRIFT {symbol}: real={pos.real_qty} "
                        f"alloc={pos.total_allocated()} strategies={list(non_unknown.keys())} "
                        f"— manual correction required"
                    )
                    self._emit_reconciliation(
                        "ALLOCATION_DRIFT",
                        symbol=symbol,
                        payload={
                            "action": "NEGATIVE_DRIFT_FROZEN",
                            "reason": "multi_strategy_negative_drift",
                            "strategy_ids": list(non_unknown),
                            "before_value": {"total_allocated": pos.total_allocated()},
                            "after_value": {"real_qty": pos.real_qty, "drift": drift, "frozen": True},
                            "drift": drift,
                        },
                    )
                    self._emit_position_snapshot("drift_frozen")
                    self._emit_allocation_snapshot("drift_frozen")
                    if self.persistence:
                        await self.persistence.sync_position(pos)
                        await self.persistence.log_recon(
                            "ALLOCATION_DRIFT", symbol=symbol,
                            before_value={"total_allocated": pos.total_allocated()},
                            after_value={"real_qty": pos.real_qty, "drift": drift},
                            action="NEGATIVE_DRIFT_FROZEN",
                            details=f"Negative drift of {drift}, {len(non_unknown)} strategies, frozen",
                        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_deployment(self, status: str) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_deployment(
                status,
                {
                    "portfolio_id": "olr_kalcb",
                    "strategy_ids": sorted(self.risk.config.strategy_budgets.keys()),
                    "safe_mode": self.risk.safe_mode,
                    "halt_new_entries": self.risk.halt_new_entries,
                },
            )
        except Exception:
            pass

    def _emit_intent(self, intent: Intent, result: IntentResult, *, phase: str) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_intent(intent, result, phase=phase)
        except Exception:
            pass

    def _emit_risk_decision(self, intent: Intent, risk_result) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_risk_decision(intent, risk_result, trace=list(getattr(risk_result, "trace", []) or []), oms=self)
        except TypeError:
            try:
                emitter.emit_risk_decision(intent, risk_result, trace=list(getattr(risk_result, "trace", []) or []))
            except Exception:
                pass
        except Exception:
            pass

    def _emit_order_event(self, wo: WorkingOrder, event_type: str, *, payload: Optional[Dict] = None, intent: Optional[Intent] = None) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_order_event(wo, event_type, payload=payload or {}, intent=intent)
        except Exception:
            pass

    def _emit_fill(
        self,
        wo: WorkingOrder,
        fill_qty: int,
        *,
        intent: Optional[Intent] = None,
        inferred: bool = False,
        payload: Optional[Dict] = None,
    ) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_fill(wo, fill_qty, intent=intent, inferred=inferred, extra=payload or {})
        except Exception:
            pass

    def _emit_reconciliation(self, event_type: str, *, symbol: str = "", payload: Optional[Dict] = None) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_reconciliation(event_type, symbol=symbol, payload=payload or {})
        except Exception:
            pass

    def _emit_position_snapshot(self, reason: str) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_position_snapshot(self.state, reason=reason)
        except Exception:
            pass

    def _emit_allocation_snapshot(self, reason: str) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_allocation_snapshot(self.state, reason=reason)
        except Exception:
            pass

    def _emit_portfolio_snapshot(self, reason: str) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_portfolio_snapshot(self, reason=reason)
        except Exception:
            pass

    def _emit_heartbeat(self, reason: str) -> None:
        emitter = self.event_emitter
        if emitter is None:
            return
        try:
            emitter.emit_heartbeat(self, reason=reason)
        except Exception:
            pass

    def _release_lock_if_entry(self, intent: Intent) -> None:
        """Release entry lock if this was an ENTER intent."""
        if intent.intent_type == IntentType.ENTER:
            self.state.release_entry_lock(intent.symbol, intent.strategy_id)

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        return self.adapter.api.get_last_price(symbol)

    async def _get_current_price_payload(self, symbol: str) -> Any:
        getter = getattr(self.adapter.api, "get_current_price", None)
        if not callable(getter):
            return None
        return await asyncio.to_thread(getter, symbol)

    async def _finalize(
        self, intent: Intent, status: IntentStatus, message: str = "",
        order_id: Optional[str] = None, modified_qty: Optional[int] = None,
        cooldown_until: Optional[float] = None,
        blocking_positions: Optional[List] = None,
        resource_conflict_type: Optional[str] = None,
        oms_received_at: Optional[float] = None,
        order_submitted_at: Optional[float] = None,
    ) -> IntentResult:
        """Create result, store in idempotency cache, and persist."""
        result = IntentResult(
            intent_id=intent.intent_id,
            status=status,
            message=message,
            order_id=order_id,
            modified_qty=modified_qty,
            cooldown_until=cooldown_until,
            blocking_positions=blocking_positions,
            resource_conflict_type=resource_conflict_type,
            oms_received_at=oms_received_at,
            order_submitted_at=order_submitted_at,
        )

        # Log all intent outcomes for observability
        log_fn = logger.info if status == IntentStatus.EXECUTED else logger.warning
        log_fn(
            f"Intent {intent.strategy_id}:{intent.symbol} "
            f"{intent.intent_type.name} -> {status.name}: {message}"
        )

        if status in {IntentStatus.EXECUTED, IntentStatus.ACCEPTED}:
            self._idem.put(intent.idempotency_key, result)
            self._rejection_counts.pop(intent.idempotency_key, None)
        elif status == IntentStatus.REJECTED:
            key = intent.idempotency_key
            self._rejection_counts[key] = self._rejection_counts.get(key, 0) + 1
            if self._rejection_counts[key] >= 5:
                logger.error(
                    f"Intent {intent.strategy_id}:{intent.symbol} "
                    f"{intent.intent_type.name} rejected {self._rejection_counts[key]}x "
                    f"— caching to stop retries"
                )
                self._idem.put(key, result)

        # Persist intent
        if self.persistence:
            await self.persistence.record_intent(intent, result)
        self._emit_intent(intent, result, phase="finalized")
        self._emit_pre_order_terminal_event(intent, result)

        return result

    def _emit_pre_order_terminal_event(self, intent: Intent, result: IntentResult) -> None:
        if result.order_id or result.status not in {IntentStatus.REJECTED, IntentStatus.DEFERRED, IntentStatus.CANCELLED}:
            return
        event_type = {
            IntentStatus.REJECTED: "ORDER_REJECTED",
            IntentStatus.DEFERRED: "ORDER_DEFERRED",
            IntentStatus.CANCELLED: "ORDER_CANCELLED",
        }[result.status]
        qty = intent.desired_qty if intent.desired_qty is not None else intent.target_qty
        order = SimpleNamespace(
            order_id=str(getattr(intent, "metadata", {}).get("provisional_order_ref") or f"preorder:{intent.intent_id}"),
            oms_order_id="",
            symbol=intent.symbol,
            side=_intent_order_side(intent),
            qty=qty,
            filled_qty=0,
            price=getattr(intent.risk_payload, "entry_px", None) or getattr(intent.constraints, "limit_price", None),
            strategy_id=intent.strategy_id,
            intent_id=intent.intent_id,
            idempotency_key=intent.idempotency_key,
        )
        self._emit_order_event(
            order,
            event_type,
            payload={
                "status_after": result.status.name,
                "reason": result.message,
                "pre_working_order": True,
                "cooldown_until": result.cooldown_until,
                "blocking_positions": result.blocking_positions,
                "resource_conflict_type": result.resource_conflict_type,
                "oms_received_at": result.oms_received_at,
            },
            intent=intent,
        )

    async def flatten_all(self) -> None:
        """Emergency flatten all positions via intent pipeline."""
        self.risk.trigger_flatten()
        positions = self.state.get_all_positions()
        for symbol, pos in positions.items():
            if pos.real_qty > 0:
                for strat_id, alloc in pos.allocations.items():
                    if alloc.qty > 0:
                        intent = Intent(
                            intent_type=IntentType.EXIT,
                            strategy_id=strat_id,
                            symbol=symbol,
                            desired_qty=alloc.qty,
                            urgency=Urgency.HIGH,
                            risk_payload=RiskPayload(rationale_code="emergency_flatten"),
                        )
                        await self.submit_intent(intent)
                # Handle unallocated remainder (drift)
                unallocated = pos.real_qty - pos.total_allocated()
                if unallocated > 0:
                    intent = Intent(
                        intent_type=IntentType.EXIT,
                        strategy_id=UNKNOWN_STRATEGY,
                        symbol=symbol,
                        desired_qty=unallocated,
                        urgency=Urgency.HIGH,
                        risk_payload=RiskPayload(rationale_code="emergency_flatten"),
                    )
                    await self.submit_intent(intent)

    def get_position(self, symbol: str):
        """Get position state for symbol."""
        return self.state.get_position(symbol)

    def get_allocation(self, symbol: str, strategy_id: str) -> int:
        """Get strategy allocation for symbol."""
        return self.state.get_position(symbol).get_allocation(strategy_id)

    async def correct_allocation(self, symbol: str, strategy_id: str, new_qty: int) -> dict:
        """Admin: set a strategy's allocation to an absolute value."""
        pos = self.state.get_position(symbol)
        old_qty = self.state.set_allocation(symbol, strategy_id, new_qty)
        logger.warning(f"ADMIN: Corrected {symbol}/{strategy_id}: {old_qty} -> {new_qty}")

        if self.persistence:
            alloc = pos.allocations.get(strategy_id)
            if alloc:
                await self.persistence.sync_allocation(symbol, alloc)
            await self.persistence.log_recon(
                "ALLOCATION_DRIFT", symbol=symbol,
                before_value={"strategy": strategy_id, "alloc_qty": old_qty},
                after_value={"strategy": strategy_id, "alloc_qty": new_qty},
                action="ADMIN_CORRECTED",
                details=f"Admin set {strategy_id} allocation from {old_qty} to {new_qty}",
            )

        # Auto-unfreeze if drift resolved
        if abs(pos.allocation_drift()) <= DRIFT_TOLERANCE:
            unknown_qty = pos.get_allocation(UNKNOWN_STRATEGY)
            if unknown_qty == 0:
                pos.frozen = False
                pos.allocations.pop(UNKNOWN_STRATEGY, None)
                logger.info(f"Unfroze {symbol} after admin correction")
                if self.persistence:
                    await self.persistence.sync_position(pos)

        self._emit_reconciliation(
            "ALLOCATION_DRIFT",
            symbol=symbol,
            payload={
                "action": "ADMIN_CORRECTED",
                "manual": True,
                "strategy_id": strategy_id,
                "before_value": {"alloc_qty": old_qty},
                "after_value": {"alloc_qty": new_qty, "drift": pos.allocation_drift(), "frozen": pos.frozen},
                "drift": pos.allocation_drift(),
            },
        )
        self._emit_position_snapshot("admin_correct_allocation")
        self._emit_allocation_snapshot("admin_correct_allocation")
        self._emit_portfolio_snapshot("admin_correct_allocation")
        return {
            "symbol": symbol, "strategy_id": strategy_id,
            "old_qty": old_qty, "new_qty": new_qty,
            "drift": pos.allocation_drift(), "frozen": pos.frozen,
        }

    async def eod_cleanup(self) -> None:
        """End-of-day: cancel all working orders and reset daily state."""
        # Query broker for final fill status before cancelling
        orders_result = await self.adapter.get_orders()
        if orders_result.ok:
            broker_by_id = {bo.order_id: bo for bo in orders_result.data}
        else:
            logger.warning(f"EOD: broker orders unavailable ({orders_result.error_message}), proceeding with cancel")
            broker_by_id = {}

        for pos in self.state.get_all_positions().values():
            for wo in list(pos.working_orders):
                broker = broker_by_id.get(wo.order_id)
                if broker:
                    final_delta = broker.filled_qty - wo.filled_qty
                    if final_delta > 0:
                        await self._apply_fill(wo, final_delta)
                        wo.filled_qty = broker.filled_qty

                cancel_result = await self.adapter.cancel_order(wo.order_id, wo.symbol, wo.qty - wo.filled_qty, branch=wo.branch)
                if not cancel_result.success:
                    logger.warning(f"EOD cancel failed for {wo.order_id}: {cancel_result.message}")

                # Re-query broker after cancel to capture any fills that occurred
                # between the initial query and the cancel request
                post_cancel_result = await self.adapter.get_orders()
                if post_cancel_result.ok:
                    post_broker = {bo.order_id: bo for bo in post_cancel_result.data}.get(wo.order_id)
                    if post_broker:
                        late_delta = post_broker.filled_qty - wo.filled_qty
                        if late_delta > 0:
                            logger.info(f"EOD: late fill detected for {wo.order_id}: +{late_delta}")
                            await self._apply_fill(wo, late_delta)
                            wo.filled_qty = post_broker.filled_qty

                await self._finalize_working_order(
                    wo,
                    final_status=OrderStatus.CANCELLED,
                    prev_status=wo.status,
                    event_type="EOD_CANCEL",
                    payload={"reason": "eod_cleanup", "filled_qty": wo.filled_qty},
                )

        self.state.daily_pnl = 0.0
        self.state.daily_pnl_pct = 0.0
        self.state.daily_realized_pnl = 0.0
        self.state.strategy_realized_pnl = {}
        self.risk.halt_new_entries = False
        self.risk.flatten_in_progress = False
        self._rejection_counts.clear()
        self._idem.clear()
        self.adapter.reset()
        self._emit_position_snapshot("eod_cleanup")
        self._emit_allocation_snapshot("eod_cleanup")
        self._emit_portfolio_snapshot("eod_cleanup")
        logger.info("EOD cleanup complete")

    async def start(self) -> None:
        """Initialize OMS: connect persistence, load state, start reconciliation."""
        self._emit_deployment("starting")
        # Connect to database
        if self.persistence:
            await self.persistence.connect()
            if self.require_persistence and not self.persistence._is_connected():
                raise RuntimeError("OMS persistence is required but Postgres is unavailable")
            await self._load_persisted_state()
            await self.reconcile_stale_pending_idempotency(stale_after_sec=60.0)
        elif self.require_persistence:
            raise RuntimeError("OMS persistence is required but no persistence backend is configured")

        # Run first reconciliation synchronously so equity is loaded before
        # the server starts accepting strategy requests (prevents equity=0 gap)
        try:
            await self._reconcile(cycle_count=0)
            if self.state.equity > 0:
                logger.info(f"Initial reconciliation complete — equity={self.state.equity:,.0f}")
            else:
                logger.warning(
                    "Initial reconciliation completed but equity=0 "
                    "— KIS may have returned empty data; will retry in loop"
                )
        except Exception as e:
            logger.error(f"Initial reconciliation failed (will retry in loop): {e}")

        # Start reconciliation loop
        if self.persistence:
            await self._start_stop_watcher()
        await self.start_reconciliation_loop()
        self._emit_deployment("started")
        logger.info("OMS started")

    async def _load_persisted_state(self) -> None:
        """Load state from database on startup."""
        if not self.persistence:
            return

        # Load positions
        positions = await self.persistence.load_positions()
        for symbol, pos in positions.items():
            self.state._positions[symbol] = pos

        # Load allocations into positions
        allocs = await self.persistence.load_allocations()
        for symbol, strategy_allocs in allocs.items():
            pos = self.state.get_position(symbol)
            pos.allocations.update(strategy_allocs)

        # Load working orders
        orders = await self.persistence.load_working_orders()
        for wo in orders:
            self.state.add_working_order(wo.symbol, wo)

        # Rehydrate accepted/executed intent outcomes for restart-safe dedupe.
        load_idem = getattr(self.persistence, "load_idempotency_results", None)
        if inspect.iscoroutinefunction(load_idem):
            idempotency_results = await load_idem()
            for key, result in idempotency_results.items():
                self._idem.put(key, result)
            if idempotency_results:
                logger.info(f"Restored {len(idempotency_results)} idempotency results from database")

        # Load OMS state (safe_mode, halt flags)
        oms_state = await self.persistence.load_oms_state()
        if oms_state:
            if oms_state.get("safe_mode"):
                self.risk.safe_mode = True
            if oms_state.get("halt_new_entries"):
                self.risk.halt_new_entries = True

        await self._reconcile_protective_stops_on_startup()

        # Restore per-strategy realized PnL from today's closed trades (mid-day restart resilience)
        realized = await self.persistence.load_daily_realized_pnl(date.today())
        if realized:
            self.state.strategy_realized_pnl = realized
            self.state.daily_realized_pnl = sum(realized.values())
            logger.info(f"Restored realized PnL for {len(realized)} strategies from DB")

        logger.info("Persisted state loaded")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._emit_deployment("shutdown")
        self._emit_position_snapshot("shutdown")
        self._emit_allocation_snapshot("shutdown")
        self._emit_portfolio_snapshot("shutdown")
        if self._reconcile_task:
            self._reconcile_task.cancel()
        if self._stop_watcher:
            await self._stop_watcher.stop()
        if self.persistence:
            await self.persistence.close()
        logger.info("OMS shutdown complete")
