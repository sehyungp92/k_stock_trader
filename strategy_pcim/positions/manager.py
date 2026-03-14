"""PCIM Position Manager."""

import time as time_module
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, List, Set
from loguru import logger

from ..config.constants import SIZING, STRATEGY_ID


@dataclass
class PCIMPosition:
    """PCIM position record."""
    symbol: str
    entry_date: date
    entry_price: float
    qty: int
    atr_at_entry: float

    # Stop levels
    initial_stop: float = 0.0
    trailing_stop: float = 0.0
    current_stop: float = 0.0

    # Status
    status: str = "OPEN"
    tp_done: bool = False
    remaining_qty: int = 0

    # Tracking
    max_price: float = 0.0
    close_reason: Optional[str] = None

    # Pending exit tracking (for exit fill confirmation)
    pending_exit_type: Optional[str] = None    # "STOP", "TAKE_PROFIT", "DAY15_EXIT"
    pending_exit_qty: int = 0
    pending_exit_ts: float = 0.0
    pending_exit_intent_id: Optional[str] = None
    pending_exit_price: float = 0.0

    def __post_init__(self):
        self.remaining_qty = self.qty
        self.initial_stop = self.entry_price - (SIZING["STOP_ATR_MULT"] * self.atr_at_entry)
        self.current_stop = self.initial_stop
        self.max_price = self.entry_price


class PositionManager:
    """Manages PCIM positions."""

    def __init__(self):
        self.positions: Dict[str, PCIMPosition] = {}
        self.pending_orders: Dict[str, dict] = {}  # symbol -> {intent_id, intended_qty, atr}
        self.submitted_today: Set[str] = set()  # idempotency: symbols entered today

    def add_position(self, pos: PCIMPosition) -> None:
        self.positions[pos.symbol] = pos
        logger.info(f"Position added: {pos.symbol} @ {pos.entry_price:.0f}, qty={pos.qty}")

    def get_position(self, symbol: str) -> Optional[PCIMPosition]:
        return self.positions.get(symbol)

    def get_open_positions(self) -> List[PCIMPosition]:
        return [p for p in self.positions.values() if p.status == "OPEN"]

    def close_position(self, symbol: str, reason: str) -> None:
        pos = self.positions.get(symbol)
        if pos:
            pos.status = "CLOSED"
            pos.close_reason = reason
            logger.info(f"Position closed: {symbol}, reason={reason}")

    def reduce_position(self, symbol: str, qty_sold: int) -> None:
        pos = self.positions.get(symbol)
        if pos:
            pos.remaining_qty -= qty_sold
            if pos.remaining_qty <= 0:
                pos.status = "CLOSED"
                pos.close_reason = "FULLY_SOLD"

    def submit_exit(self, symbol: str, exit_type: str, qty: int, intent_id: str, price: float) -> None:
        """Mark position as having a pending exit order."""
        pos = self.positions.get(symbol)
        if pos:
            pos.pending_exit_type = exit_type
            pos.pending_exit_qty = qty
            pos.pending_exit_ts = time_module.time()
            pos.pending_exit_intent_id = intent_id
            pos.pending_exit_price = price
            logger.info(f"{symbol}: Pending exit {exit_type} qty={qty}")

    def clear_pending_exit(self, symbol: str) -> None:
        """Clear pending exit state."""
        pos = self.positions.get(symbol)
        if pos:
            pos.pending_exit_type = None
            pos.pending_exit_qty = 0
            pos.pending_exit_ts = 0.0
            pos.pending_exit_intent_id = None
            pos.pending_exit_price = 0.0

    def has_pending_exit(self, symbol: str) -> bool:
        """Check if position has a pending exit order."""
        pos = self.positions.get(symbol)
        return pos is not None and pos.pending_exit_type is not None

    def track_pending(self, symbol: str, intent_id: str, intended_qty: int, atr: float) -> None:
        """Track a pending order until fill confirmed."""
        self.pending_orders[symbol] = {
            'intent_id': intent_id, 'intended_qty': intended_qty, 'atr': atr
        }
        self.submitted_today.add(symbol)

    def was_submitted_today(self, symbol: str) -> bool:
        """Check if symbol was already submitted today (idempotency)."""
        return symbol in self.submitted_today

    def clear_pending(self, symbol: str) -> Optional[dict]:
        """Remove and return pending order info."""
        return self.pending_orders.pop(symbol, None)

    async def reconcile_from_oms(self, oms, api, today: date) -> None:
        """Reconcile positions from OMS allocations at startup."""
        allocations = await oms.get_strategy_allocations(STRATEGY_ID)
        for symbol, alloc in allocations.items():
            if alloc.qty <= 0:
                continue
            if symbol in self.positions:
                # Update existing position with actual qty
                self.positions[symbol].qty = alloc.qty
                self.positions[symbol].remaining_qty = alloc.qty
                if alloc.cost_basis > 0:
                    self.positions[symbol].entry_price = alloc.cost_basis
            else:
                # Create position from OMS allocation
                atr = api.get_atr_20d(symbol) if api else 0.0
                self.positions[symbol] = PCIMPosition(
                    symbol=symbol,
                    entry_date=today,
                    entry_price=alloc.cost_basis or 0.0,
                    qty=alloc.qty,
                    atr_at_entry=atr,
                )
                logger.info(f"Reconciled position from OMS: {symbol} qty={alloc.qty}")
        self.submitted_today.update(self.positions.keys())

    def reset_daily_state(self) -> None:
        """Reset daily tracking state."""
        self.submitted_today.clear()
        self.pending_orders.clear()
        # Warn about stale pending exits
        for pos in self.positions.values():
            if pos.pending_exit_type:
                logger.warning(f"{pos.symbol}: Stale pending exit {pos.pending_exit_type} cleared on daily reset")
                pos.pending_exit_type = None
                pos.pending_exit_qty = 0
                pos.pending_exit_ts = 0.0
                pos.pending_exit_intent_id = None
                pos.pending_exit_price = 0.0
