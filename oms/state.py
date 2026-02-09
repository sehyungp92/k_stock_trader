"""
OMS State Model: Single source of truth for positions and allocations.

Key invariant: sum(allocations) == real_qty (within tolerance)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional
import time
import threading


class OrderStatus(Enum):
    PENDING = auto()
    SUBMITTING = auto()
    WORKING = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    FAILED = auto()


@dataclass
class WorkingOrder:
    """Represents an order in flight."""
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: int
    filled_qty: int = 0
    price: float = 0.0
    order_type: str = "LIMIT"
    status: OrderStatus = OrderStatus.PENDING
    strategy_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    submit_ts: float = field(default_factory=time.time)
    cancel_after_sec: Optional[float] = None
    branch: str = ""  # KRX order branch for cancel/revise


@dataclass
class StrategyAllocation:
    """Per-strategy allocation within a symbol position."""
    strategy_id: str
    qty: int = 0
    cost_basis: float = 0.0
    entry_ts: Optional[datetime] = None

    # Strategy-specific risk overlays
    soft_stop_px: Optional[float] = None
    time_stop_ts: Optional[float] = None


@dataclass
class SymbolPosition:
    """
    Complete position state for a symbol.

    Contains real broker position + virtual strategy allocations.
    """
    symbol: str

    # Real broker position
    real_qty: int = 0
    avg_price: float = 0.0

    # Virtual allocations per strategy
    allocations: Dict[str, StrategyAllocation] = field(default_factory=dict)

    # Risk overlays (tightest wins)
    hard_stop_px: Optional[float] = None

    # Locks
    entry_lock_owner: Optional[str] = None
    entry_lock_until: Optional[float] = None

    # Cooldowns
    cooldown_until: Optional[float] = None
    vi_cooldown_until: Optional[float] = None

    # Working orders
    working_orders: List[WorkingOrder] = field(default_factory=list)

    # Reconciliation
    frozen: bool = False  # True = allocation drift detected, new entries blocked

    # Timestamps
    last_update_ts: datetime = field(default_factory=datetime.now)

    def has_working_orders(self) -> bool:
        """Check if any orders are in flight."""
        return len(self.working_orders) > 0

    def working_qty(self, strategy_id: Optional[str] = None, side: Optional[str] = None) -> int:
        """Sum of unfilled qty in working orders, optionally filtered."""
        total = 0
        for wo in self.working_orders:
            if strategy_id and wo.strategy_id != strategy_id:
                continue
            if side and wo.side != side:
                continue
            total += wo.qty - wo.filled_qty
        return total

    def total_allocated(self) -> int:
        """Sum of all strategy allocations."""
        return sum(a.qty for a in self.allocations.values())

    def allocation_drift(self) -> int:
        """Difference between real position and allocations."""
        return self.real_qty - self.total_allocated()

    def get_allocation(self, strategy_id: str) -> int:
        """Get allocation for a specific strategy."""
        alloc = self.allocations.get(strategy_id)
        return alloc.qty if alloc else 0

    def is_entry_locked(self, now_ts: float) -> bool:
        """Check if entry is locked by another strategy."""
        if self.entry_lock_until is None:
            return False
        return now_ts < self.entry_lock_until

    def can_strategy_enter(self, strategy_id: str, now_ts: float) -> bool:
        """Check if strategy can take entry."""
        if not self.is_entry_locked(now_ts):
            return True
        return self.entry_lock_owner == strategy_id


class StateStore:
    """
    Thread-safe state store for OMS.

    In production, back with Redis/Postgres. This is in-memory reference.
    """

    def __init__(self):
        self._positions: Dict[str, SymbolPosition] = {}
        self._lock = threading.RLock()

        # Account-level state
        self.equity: float = 0.0
        self.buyable_cash: float = 0.0
        self.daily_pnl: float = 0.0
        self.daily_pnl_pct: float = 0.0
        self.daily_realized_pnl: float = 0.0  # Accumulated realized P&L from closed positions

    def get_position(self, symbol: str) -> SymbolPosition:
        """Get or create position for symbol."""
        with self._lock:
            if symbol not in self._positions:
                self._positions[symbol] = SymbolPosition(symbol=symbol)
            return self._positions[symbol]

    def get_all_positions(self) -> Dict[str, SymbolPosition]:
        """Get all positions (copy)."""
        with self._lock:
            return dict(self._positions)

    def update_position(self, symbol: str, **kwargs) -> None:
        """Update position fields."""
        with self._lock:
            pos = self.get_position(symbol)
            for k, v in kwargs.items():
                if hasattr(pos, k):
                    setattr(pos, k, v)
            pos.last_update_ts = datetime.now()

    def update_allocation(
        self, symbol: str, strategy_id: str, qty_delta: int,
        cost_basis: Optional[float] = None
    ) -> None:
        """Update strategy allocation for a symbol."""
        with self._lock:
            pos = self.get_position(symbol)
            if strategy_id not in pos.allocations:
                pos.allocations[strategy_id] = StrategyAllocation(strategy_id=strategy_id)

            alloc = pos.allocations[strategy_id]
            if cost_basis is not None and qty_delta > 0:
                # Weighted average cost basis for buys
                old_notional = alloc.cost_basis * alloc.qty
                new_notional = cost_basis * qty_delta
                alloc.qty += qty_delta
                alloc.cost_basis = (old_notional + new_notional) / max(alloc.qty, 1)
            else:
                alloc.qty += qty_delta
            if alloc.qty > 0 and alloc.entry_ts is None:
                alloc.entry_ts = datetime.now()
            elif alloc.qty <= 0:
                alloc.entry_ts = None

    def set_entry_lock(
        self, symbol: str, strategy_id: str, until_ts: float
    ) -> bool:
        """Attempt to acquire entry lock. Returns True if successful."""
        with self._lock:
            pos = self.get_position(symbol)
            now = time.time()

            if pos.is_entry_locked(now) and pos.entry_lock_owner != strategy_id:
                return False

            pos.entry_lock_owner = strategy_id
            pos.entry_lock_until = until_ts
            return True

    def release_entry_lock(self, symbol: str, strategy_id: str) -> None:
        """Release entry lock if owned by strategy."""
        with self._lock:
            pos = self.get_position(symbol)
            if pos.entry_lock_owner == strategy_id:
                pos.entry_lock_owner = None
                pos.entry_lock_until = None

    def add_working_order(self, symbol: str, order: WorkingOrder) -> None:
        """Add working order to position."""
        with self._lock:
            pos = self.get_position(symbol)
            pos.working_orders.append(order)

    def remove_working_order(self, symbol: str, order_id: str) -> None:
        """Remove working order from position."""
        with self._lock:
            pos = self.get_position(symbol)
            pos.working_orders = [o for o in pos.working_orders if o.order_id != order_id]

    def get_working_orders(self, symbol: Optional[str] = None) -> List[WorkingOrder]:
        """Get working orders, optionally filtered by symbol."""
        with self._lock:
            if symbol:
                pos = self._positions.get(symbol)
                return list(pos.working_orders) if pos else []

            orders = []
            for pos in self._positions.values():
                orders.extend(pos.working_orders)
            return orders

    def get_allocations_for_strategy(self, strategy_id: str) -> Dict[str, StrategyAllocation]:
        """Get all allocations for a strategy across all symbols."""
        with self._lock:
            result = {}
            for symbol, pos in self._positions.items():
                alloc = pos.allocations.get(strategy_id)
                if alloc and alloc.qty > 0:
                    result[symbol] = alloc
            return result

    def record_realized_pnl(self, pnl: float) -> None:
        """Record realized P&L from a closed position."""
        with self._lock:
            self.daily_realized_pnl += pnl

    def update_daily_pnl(self, prices: Dict[str, float]) -> None:
        """Compute daily P&L from positions using live prices + realized P&L."""
        with self._lock:
            unrealized_pnl = 0.0
            for symbol, pos in self._positions.items():
                if pos.real_qty > 0:
                    current_px = prices.get(symbol, pos.avg_price)
                    unrealized_pnl += (current_px - pos.avg_price) * pos.real_qty
            self.daily_pnl = unrealized_pnl + self.daily_realized_pnl
            self.daily_pnl_pct = self.daily_pnl / max(self.equity, 1.0)
