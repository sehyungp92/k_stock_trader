"""
OMS Core: Main orchestrator that ties everything together.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
import time
from loguru import logger

from collections import defaultdict

from .intent import Intent, IntentResult, IntentStatus, IntentType, Urgency, RiskPayload
from .state import StateStore, WorkingOrder, OrderStatus, StrategyAllocation
from .risk import RiskGateway, RiskConfig, RiskDecision
from .arbitration import ArbitrationEngine, ArbitrationResult
from .planner import OrderPlanner
from .adapter import KISExecutionAdapter
from .persistence import OMSPersistence


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


class InMemoryIdempotencyStore(IdempotencyStore):
    def __init__(self):
        self._store: Dict[str, IntentResult] = {}

    def get(self, key: str) -> Optional[IntentResult]:
        return self._store.get(key)

    def put(self, key: str, result: IntentResult) -> None:
        self._store[key] = result


# ---------------------------------------------------------------------------
# OMS Core
# ---------------------------------------------------------------------------

UNKNOWN_STRATEGY = "_UNKNOWN_"
DRIFT_TOLERANCE = 0  # shares
BROKER_MISSING_GRACE_CYCLES = 2


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
    ):
        self.state = StateStore()
        self.risk = RiskGateway(
            self.state,
            risk_config or RiskConfig(),
            price_getter=lambda s: kis_api.get_last_price(s),
        )
        self.arbitration = ArbitrationEngine(self.state)
        self.planner = OrderPlanner()
        self.adapter = KISExecutionAdapter(kis_api)
        self.persistence = persistence

        self._idem = idempotency_store or InMemoryIdempotencyStore()
        self._reconcile_task: Optional[asyncio.Task] = None
        self._symbol_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def submit_intent(self, intent: Intent) -> IntentResult:
        """Submit intent for processing. Main entry point for strategies."""
        oms_received_at = time.time()

        # 1. Idempotency check (outside lock — read-only)
        cached = self._idem.get(intent.idempotency_key)
        if cached is not None:
            logger.debug(f"Duplicate intent: {intent.idempotency_key}")
            return cached

        # 2. Validate (includes expiry enforcement)
        valid, error = intent.validate()
        if not valid:
            return await self._finalize(intent, IntentStatus.REJECTED, f"Validation failed: {error}")

        # Per-symbol mutex: prevents concurrent submits for same symbol
        async with self._symbol_locks[intent.symbol]:
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
            alloc_qty = self.state.get_position(intent.symbol).get_allocation(intent.strategy_id)
            if alloc_qty <= 0:
                # Check working BUY orders — cancel instead of sell
                pending = self.state.get_position(intent.symbol).working_qty(
                    strategy_id=intent.strategy_id, side="BUY"
                )
                if pending > 0:
                    return await self._handle_cancel_orders(intent)
                return await self._finalize(intent, IntentStatus.REJECTED, "No allocation to exit", oms_received_at=oms_received_at)
            # Respect desired_qty for partial exits, capped at allocation
            exit_qty = min(intent.desired_qty, alloc_qty) if intent.desired_qty else alloc_qty
            plan = self.planner.create_exit_plan(
                symbol=intent.symbol, qty=exit_qty,
                strategy_id=intent.strategy_id,
                intent_id=intent.intent_id, urgency=intent.urgency,
            )
        elif intent.intent_type == IntentType.REDUCE:
            plan = self.planner.create_exit_plan(
                symbol=intent.symbol, qty=abs(final_qty),
                strategy_id=intent.strategy_id,
                intent_id=intent.intent_id, urgency=intent.urgency,
            )
        elif intent.intent_type == IntentType.SET_TARGET:
            # Compute delta = target_qty - current_allocation
            current_alloc = self.state.get_position(intent.symbol).get_allocation(intent.strategy_id)
            target_qty = intent.target_qty or 0
            delta = target_qty - current_alloc
            if delta == 0:
                return await self._finalize(intent, IntentStatus.EXECUTED, "Already at target", oms_received_at=oms_received_at)
            side = "BUY" if delta > 0 else "SELL"
            plan = self.planner.create_plan(
                symbol=intent.symbol, side=side, qty=abs(delta),
                intent=intent, current_price=current_price,
            ) if delta > 0 else self.planner.create_exit_plan(
                symbol=intent.symbol, qty=abs(delta),
                strategy_id=intent.strategy_id,
                intent_id=intent.intent_id, urgency=intent.urgency,
            )
        else:
            return await self._finalize(intent, IntentStatus.REJECTED, f"Unsupported intent type: {intent.intent_type}", oms_received_at=oms_received_at)

        order_price = plan.limit_price or current_price or 0.0
        sector_reserved = False
        if plan.side == "BUY":
            self.risk.reserve_sector(plan.symbol, plan.qty, order_price)
            sector_reserved = True

        # Execute
        try:
            exec_result = await self.adapter.submit_order(
                symbol=plan.symbol, side=plan.side, qty=plan.qty,
                order_type=plan.order_type.name,
                limit_price=plan.limit_price, stop_price=plan.stop_price,
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
        )
        self.state.add_working_order(plan.symbol, wo)

        # Persist order
        if self.persistence:
            wo.oms_order_id = await self.persistence.record_order(wo, intent_id=intent.intent_id)
            await self.persistence.record_order_event(
                "ORDER_SUBMITTED", order_id=wo.order_id, intent_id=intent.intent_id,
                strategy_id=intent.strategy_id, symbol=plan.symbol,
                status_after="WORKING",
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

    async def _apply_fill(self, wo: WorkingOrder, fill_qty: int, intent: Optional[Intent] = None) -> None:
        """Apply fill to allocation. real_qty is updated from broker sync only."""
        qty_delta = fill_qty if wo.side == "BUY" else -fill_qty

        # Record realized P&L for sell fills
        if wo.side == "SELL":
            pos = self.state.get_position(wo.symbol)
            alloc = pos.allocations.get(wo.strategy_id)
            if alloc and alloc.cost_basis > 0:
                realized_pnl = (wo.price - alloc.cost_basis) * fill_qty
                self.state.record_realized_pnl(realized_pnl)

        self.state.update_allocation(
            wo.symbol, wo.strategy_id, qty_delta,
            cost_basis=wo.price,
        )

        # Update OMS risk gateway sector exposure on fills
        if wo.side == "BUY":
            self.risk.on_sector_fill(wo.symbol, fill_qty, wo.price)
        else:
            self.risk.on_sector_close(wo.symbol, fill_qty, wo.price)

        # Note: real_qty updated from broker position sync in _reconcile to avoid double-credit
        logger.info(f"Fill applied: {wo.symbol} {wo.side} {fill_qty} for {wo.strategy_id}")

        # Persist fill and allocation
        if self.persistence:
            exec_id = f"{wo.order_id}:{wo.filled_qty + fill_qty}"
            fill_ts = datetime.now()
            resolved_intent_id = intent.intent_id if intent else wo.intent_id
            await self.persistence.record_fill(
                kis_exec_id=exec_id, order_id=wo.order_id,
                strategy_id=wo.strategy_id, symbol=wo.symbol,
                side=wo.side, qty=fill_qty, price=wo.price,
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
                        entry_price=wo.price,
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
                        exit_price=wo.price,
                        exit_ts=fill_ts,
                        exit_intent_id=resolved_intent_id,
                        exit_reason=exit_reason,
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
        self.state.remove_working_order(wo.symbol, wo.order_id)
        self.state.release_entry_lock(wo.symbol, wo.strategy_id)
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
                            await self._apply_fill(wo, fill_delta)
                            # Record partial fill event
                            if self.persistence and new_filled < wo.qty:
                                await self.persistence.record_order_event(
                                    "PARTIAL_FILL", order_id=wo.order_id,
                                    strategy_id=wo.strategy_id, symbol=wo.symbol,
                                    payload={"fill_qty": fill_delta, "total_filled": new_filled, "order_qty": wo.qty},
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
                        await self._apply_fill(wo, inferred_fill)
                        wo.filled_qty += inferred_fill
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
                        logger.critical(
                            "EQUITY_ZERO: First reconciliation completed but equity=0 "
                            "— all ENTER intents will be deferred until equity is loaded"
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
            from datetime import date
            today = date.today()

            # Compute gross exposure
            gross_exposure = sum(
                pos.real_qty * prices.get(sym, pos.avg_price)
                for sym, pos in self.state.get_all_positions().items()
            )

            # Update portfolio-level daily risk
            await self.persistence.update_daily_risk_portfolio(
                trade_date=today,
                equity_krw=self.state.equity,
                buyable_cash_krw=self.state.buyable_cash,
                realized_pnl_krw=self.state.daily_pnl,  # Approximate
                unrealized_pnl_krw=0,  # TODO: compute unrealized separately if needed
                gross_exposure_krw=gross_exposure,
                positions_count=len(self.state.get_all_positions()),
                halted=getattr(self.risk, 'halt_new_entries', False),
                safe_mode=getattr(self.risk, 'safe_mode', False),
                regime=getattr(self.risk, '_regime', None),
            )

            # Update per-strategy daily risk (aggregate by strategy)
            strategy_stats = {}
            for pos in self.state.get_all_positions().values():
                for strat_id, alloc in pos.allocations.items():
                    if strat_id not in strategy_stats:
                        strategy_stats[strat_id] = {
                            'realized_pnl': 0, 'unrealized_pnl': 0,
                            'trades': 0, 'wins': 0, 'losses': 0
                        }
                    # Count open positions per strategy
                    if alloc.qty > 0:
                        strategy_stats[strat_id]['trades'] += 1

            for strat_id, stats in strategy_stats.items():
                await self.persistence.update_daily_risk_strategy(
                    trade_date=today,
                    strategy_id=strat_id,
                    realized_pnl_krw=stats['realized_pnl'],
                    unrealized_pnl_krw=stats['unrealized_pnl'],
                    trades_count=stats['trades'],
                    wins=stats['wins'],
                    losses=stats['losses'],
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
                recon_status="WARN" if drift_count > 0 else "OK",
                drift_count=drift_count,
            )

    async def _check_allocation_drift(self) -> None:
        """
        Detect and repair allocation drift.

        Policy:
        - If working orders exist: allow temporary drift (orders in flight).
        - If no working orders and drift != 0:
            - Assign drift to _UNKNOWN_ allocation.
            - Freeze symbol for new entries until resolved.
            - Log critical event.
        """
        for symbol, pos in self.state.get_all_positions().items():
            drift = pos.allocation_drift()

            if abs(drift) <= DRIFT_TOLERANCE:
                # No drift — unfreeze if previously frozen and UNKNOWN cleared
                if pos.frozen:
                    unknown_qty = pos.get_allocation(UNKNOWN_STRATEGY)
                    if unknown_qty == 0:
                        pos.frozen = False
                        logger.info(f"Unfroze {symbol}: drift resolved")
                        if self.persistence:
                            await self.persistence.log_recon(
                                "ALLOCATION_DRIFT", symbol=symbol, action="UNFROZEN",
                                details="Drift resolved, symbol unfrozen",
                            )
                continue

            if pos.has_working_orders():
                # Orders in flight — drift is expected, skip
                continue

            # Deterministic repair: assign drift to UNKNOWN
            logger.critical(
                f"ALLOCATION DRIFT {symbol}: real={pos.real_qty} "
                f"allocated={pos.total_allocated()} drift={drift}"
            )

            if drift > 0:
                # Positive drift: broker has more shares than allocated — assign to UNKNOWN
                if UNKNOWN_STRATEGY not in pos.allocations:
                    pos.allocations[UNKNOWN_STRATEGY] = StrategyAllocation(strategy_id=UNKNOWN_STRATEGY)
                pos.allocations[UNKNOWN_STRATEGY].qty += drift
            else:
                # Negative drift: broker has fewer shares than allocated.
                # Do NOT assign negative qty — log for manual review only.
                logger.critical(
                    f"NEGATIVE DRIFT {symbol}: broker has {pos.real_qty} shares but "
                    f"allocations sum to {pos.total_allocated()}. "
                    f"Manual review required — NOT auto-correcting."
                )
            pos.frozen = True

            if self.persistence:
                await self.persistence.log_recon(
                    "ALLOCATION_DRIFT", symbol=symbol,
                    before_value={"total_allocated": pos.total_allocated() - drift},
                    after_value={"total_allocated": pos.total_allocated(), "drift": drift},
                    action="ASSIGNED_UNKNOWN",
                    details=f"Drift of {drift} assigned to _UNKNOWN_, symbol frozen",
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _release_lock_if_entry(self, intent: Intent) -> None:
        """Release entry lock if this was an ENTER intent."""
        if intent.intent_type == IntentType.ENTER:
            self.state.release_entry_lock(intent.symbol, intent.strategy_id)

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        return self.adapter.api.get_last_price(symbol)

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

        # Only cache EXECUTED results — REJECTED/DEFERRED must be retryable
        if status == IntentStatus.EXECUTED:
            self._idem.put(intent.idempotency_key, result)

        # Persist intent
        if self.persistence:
            await self.persistence.record_intent(intent, result)

        return result

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

                self.state.remove_working_order(wo.symbol, wo.order_id)
                self.state.release_entry_lock(wo.symbol, wo.strategy_id)

        self.state.daily_pnl = 0.0
        self.state.daily_pnl_pct = 0.0
        self.state.daily_realized_pnl = 0.0
        self.risk.halt_new_entries = False
        self.risk.flatten_in_progress = False
        logger.info("EOD cleanup complete")

    async def start(self) -> None:
        """Initialize OMS: connect persistence, load state, start reconciliation."""
        # Connect to database
        if self.persistence:
            await self.persistence.connect()
            await self._load_persisted_state()

        # Start reconciliation loop
        await self.start_reconciliation_loop()
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

        # Load OMS state (safe_mode, halt flags)
        oms_state = await self.persistence.load_oms_state()
        if oms_state:
            if oms_state.get("safe_mode"):
                self.risk.safe_mode = True
            if oms_state.get("halt_new_entries"):
                self.risk.halt_new_entries = True

        logger.info("Persisted state loaded")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._reconcile_task:
            self._reconcile_task.cancel()
        if self.persistence:
            await self.persistence.close()
        logger.info("OMS shutdown complete")
