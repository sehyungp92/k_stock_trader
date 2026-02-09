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
from loguru import logger

from .intent import Intent, IntentResult, IntentStatus
from .state import WorkingOrder, SymbolPosition, StrategyAllocation, OrderStatus


class OMSPersistence:
    """Async persistence layer for OMS state."""

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.environ.get(
            "DATABASE_URL",
            "postgresql://trading_writer:changeme@postgres:5432/trading"
        )
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
            logger.info("Postgres connection pool established")
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

    # ------------------------------------------------------------------
    # Intent Recording
    # ------------------------------------------------------------------

    async def record_intent(self, intent: Intent, result: IntentResult) -> None:
        """Record intent and its result."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO intents (
                    intent_id, idempotency_key, strategy_id, symbol,
                    intent_type, desired_qty, target_qty, urgency, time_horizon,
                    max_slippage_bps, max_spread_bps, limit_price, stop_price, expiry_ts,
                    entry_px, stop_px, hard_stop_px, rationale_code, confidence, signal_hash,
                    status, result_message, modified_qty, order_id, cooldown_until, processed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    to_timestamp($14), $15, $16, $17, $18, $19, $20, $21, $22, $23, $24,
                    to_timestamp($25), NOW()
                )
                ON CONFLICT (idempotency_key) DO UPDATE SET
                    status = EXCLUDED.status,
                    result_message = EXCLUDED.result_message,
                    modified_qty = EXCLUDED.modified_qty,
                    order_id = EXCLUDED.order_id,
                    cooldown_until = EXCLUDED.cooldown_until,
                    processed_at = NOW()
                """,
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
            )
        except Exception as e:
            logger.error(f"Failed to record intent: {e}")

    # ------------------------------------------------------------------
    # Order Recording
    # ------------------------------------------------------------------

    async def record_order(
        self,
        order: WorkingOrder,
        intent_id: Optional[str] = None,
        kis_order_id: Optional[str] = None,
        kis_order_date: Optional[str] = None,
    ) -> None:
        """Record order creation or update."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                INSERT INTO orders (
                    oms_order_id, strategy_id, symbol, side, order_type,
                    qty, filled_qty, limit_price, stop_price, status,
                    kis_order_id, kis_order_date, intent_id, cancel_after_sec
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::uuid, $14)
                ON CONFLICT (oms_order_id) DO UPDATE SET
                    filled_qty = EXCLUDED.filled_qty,
                    status = EXCLUDED.status,
                    kis_order_id = COALESCE(EXCLUDED.kis_order_id, orders.kis_order_id),
                    kis_order_date = COALESCE(EXCLUDED.kis_order_date, orders.kis_order_date),
                    last_update_at = NOW()
                """,
                order.order_id,
                order.strategy_id,
                order.symbol,
                order.side,
                order.order_type,
                order.qty,
                order.filled_qty,
                order.price if order.order_type == "LIMIT" else None,
                None,  # stop_price
                order.status.name,
                kis_order_id,
                kis_order_date,
                intent_id,
                int(order.cancel_after_sec) if order.cancel_after_sec else None,
            )
        except Exception as e:
            logger.error(f"Failed to record order: {e}")

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
                order_id, status.name, filled_qty, avg_fill_price,
            )
        except Exception as e:
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
        try:
            await self.pool.execute(
                """
                INSERT INTO order_events (
                    oms_order_id, intent_id, strategy_id, symbol,
                    event_type, payload, status_before, status_after
                ) VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8)
                """,
                order_id,
                intent_id,
                strategy_id,
                symbol,
                event_type,
                json.dumps(payload) if payload else None,
                status_before,
                status_after,
            )
        except Exception as e:
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
        try:
            await self.pool.execute(
                """
                INSERT INTO fills (
                    kis_exec_id, oms_order_id, strategy_id, symbol,
                    side, qty, price, commission, tax, fill_ts
                ) VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (kis_exec_id) DO NOTHING
                """,
                kis_exec_id, order_id, strategy_id, symbol,
                side, qty, price, commission, tax, fill_ts,
            )
        except Exception as e:
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
                    cooldown_until, vi_cooldown_until, frozen
                ) VALUES ($1, $2, $3, $4, $5, to_timestamp($6), to_timestamp($7), to_timestamp($8), $9)
                ON CONFLICT (symbol) DO UPDATE SET
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
            )
        except Exception as e:
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
                    soft_stop_px, time_stop_ts
                ) VALUES ($1, $2, $3, $4, $5, $6, to_timestamp($7))
                ON CONFLICT (symbol, strategy_id) DO UPDATE SET
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
            )
        except Exception as e:
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
                    halted, safe_mode, regime
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (trade_date) DO UPDATE SET
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
            )
        except Exception as e:
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
                    trades_count, wins, losses, halted
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (trade_date, strategy_id) DO UPDATE SET
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
            )
        except Exception as e:
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
                    positions_count, last_error, version, last_heartbeat_ts
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (strategy_id) DO UPDATE SET
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
            )
        except Exception as e:
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
                UPDATE oms_state SET
                    last_heartbeat_ts = NOW(),
                    equity_krw = $1,
                    buyable_cash_krw = $2,
                    daily_pnl_krw = $3,
                    daily_pnl_pct = $4,
                    safe_mode = $5,
                    halt_new_entries = $6,
                    kis_connected = $7,
                    last_recon_ts = NOW(),
                    recon_status = $8,
                    allocation_drift_count = $9,
                    version = $10,
                    last_update_at = NOW()
                WHERE oms_id = 'primary'
                """,
                int(equity_krw), int(buyable_cash_krw), int(daily_pnl_krw),
                daily_pnl_pct, safe_mode, halt_new_entries, kis_connected,
                recon_status, drift_count, version,
            )
        except Exception as e:
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
        """Open a new trade. Returns trade_id."""
        if not self._is_connected():
            return None
        import uuid
        trade_id = str(uuid.uuid4())
        try:
            await self.pool.execute(
                """
                INSERT INTO trades (
                    trade_id, strategy_id, symbol, direction,
                    entry_qty, entry_price, entry_ts, entry_intent_id,
                    setup_type, confidence, status
                ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::uuid, $9, $10, 'OPEN')
                """,
                trade_id, strategy_id, symbol, direction,
                entry_qty, entry_price, entry_ts, entry_intent_id,
                setup_type, confidence,
            )
            logger.debug(f"Opened trade {trade_id}: {symbol} {direction} {entry_qty}@{entry_price}")
            return trade_id
        except Exception as e:
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
    ) -> None:
        """Close a trade, computing P&L."""
        if not self._is_connected():
            return
        try:
            await self.pool.execute(
                """
                UPDATE trades SET
                    exit_qty = $2,
                    exit_price = $3,
                    exit_ts = $4,
                    exit_intent_id = $5::uuid,
                    exit_reason = $6,
                    realized_pnl_krw = ($3 - entry_price) * LEAST(entry_qty, $2)
                        * CASE WHEN direction = 'LONG' THEN 1 ELSE -1 END,
                    status = 'CLOSED',
                    closed_at = NOW()
                WHERE trade_id = $1::uuid
                """,
                trade_id, exit_qty, exit_price, exit_ts, exit_intent_id, exit_reason,
            )
            logger.debug(f"Closed trade {trade_id}: {exit_qty}@{exit_price} reason={exit_reason}")
        except Exception as e:
            logger.error(f"Failed to close trade: {e}")

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
        except Exception as e:
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
                WHERE strategy_id = $1 AND symbol = $2 AND status = 'OPEN'
                ORDER BY entry_ts DESC LIMIT 1
                """,
                strategy_id, symbol,
            )
            return str(row['trade_id']) if row else None
        except Exception as e:
            logger.error(f"Failed to find open trade: {e}")
            return None

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
                    before_value, after_value, action, details
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                recon_type, symbol, strategy_id,
                json.dumps(before_value) if before_value else None,
                json.dumps(after_value) if after_value else None,
                action, details,
            )
        except Exception as e:
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
                "SELECT * FROM positions WHERE real_qty != 0 OR frozen = TRUE"
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
            logger.info(f"Loaded {len(positions)} positions from database")
            return positions
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            return {}

    async def load_allocations(self) -> Dict[str, Dict[str, StrategyAllocation]]:
        """Load allocations from database on startup."""
        if not self._is_connected():
            return {}
        try:
            rows = await self.pool.fetch("SELECT * FROM allocations WHERE qty > 0")
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
            logger.info(f"Loaded allocations for {len(allocs)} symbols from database")
            return allocs
        except Exception as e:
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
                WHERE status IN ('WORKING', 'PARTIAL', 'SUBMITTING')
                """
            )
            orders = []
            for row in rows:
                orders.append(WorkingOrder(
                    order_id=str(row["oms_order_id"]),
                    symbol=row["symbol"],
                    side=row["side"],
                    qty=row["qty"],
                    filled_qty=row["filled_qty"],
                    price=float(row["limit_price"]) if row["limit_price"] else 0.0,
                    order_type=row["order_type"],
                    status=OrderStatus[row["status"]],
                    strategy_id=row["strategy_id"],
                    created_at=row["created_at"],
                    cancel_after_sec=row["cancel_after_sec"],
                ))
            logger.info(f"Loaded {len(orders)} working orders from database")
            return orders
        except Exception as e:
            logger.error(f"Failed to load working orders: {e}")
            return []

    async def load_oms_state(self) -> Optional[Dict[str, Any]]:
        """Load OMS state from database on startup."""
        if not self._is_connected():
            return None
        try:
            row = await self.pool.fetchrow(
                "SELECT * FROM oms_state WHERE oms_id = 'primary'"
            )
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Failed to load OMS state: {e}")
            return None
