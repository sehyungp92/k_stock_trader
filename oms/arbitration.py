"""
Arbitration Engine: Resolve conflicts between multiple strategies.

Handles:
- Priority ordering (exits > reductions > entries)
- Entry ownership locks
- Virtual allocation netting
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import time

from .intent import Intent, IntentType
from .state import StateStore


class ArbitrationResult(Enum):
    PROCEED = auto()
    DEFER = auto()
    MERGE = auto()
    CANCEL = auto()


@dataclass
class ArbitrationDecision:
    result: ArbitrationResult
    reason: str = ""
    merged_qty: Optional[int] = None
    defer_until: Optional[float] = None


class ArbitrationEngine:
    """
    Arbitration engine for multi-strategy conflict resolution.

    Priority order:
    1. Risk exits (circuit breaker, flatten)
    2. Hard exits (strategy exit)
    3. Reductions
    4. Entries

    Within same class: INTRADAY > SWING, higher urgency first.
    """

    # Entry lock durations by strategy (seconds)
    LOCK_DURATIONS = {
        "KMP": 90,
        "KPR": 180,
        "PCIM": 300,
        "NULRIMOK": 60,
    }

    def __init__(self, state: StateStore):
        self.state = state
        self._pending_intents: Dict[str, List[Intent]] = {}

    def arbitrate(self, intent: Intent) -> ArbitrationDecision:
        """
        Arbitrate intent against existing intents and positions.

        Returns decision on how to proceed.
        """
        symbol = intent.symbol
        pos = self.state.get_position(symbol)
        now = time.time()

        # Exits always proceed
        if intent.intent_type in (IntentType.EXIT, IntentType.FLATTEN):
            return ArbitrationDecision(ArbitrationResult.PROCEED)

        # Reductions proceed but may merge
        if intent.intent_type == IntentType.REDUCE:
            return ArbitrationDecision(ArbitrationResult.PROCEED)

        # Entries need lock arbitration + same-ticker check
        if intent.intent_type == IntentType.ENTER:
            if pos.get_allocation(intent.strategy_id) > 0:
                return ArbitrationDecision(
                    ArbitrationResult.CANCEL,
                    f"{intent.strategy_id} already holds {intent.symbol}",
                )
            return self._arbitrate_entry(intent, pos, now)

        return ArbitrationDecision(ArbitrationResult.PROCEED)

    def _arbitrate_entry(
        self, intent: Intent, pos, now: float
    ) -> ArbitrationDecision:
        """Arbitrate entry intent."""
        strategy_id = intent.strategy_id
        symbol = intent.symbol

        # Check if entry locked by another strategy
        if pos.is_entry_locked(now) and pos.entry_lock_owner != strategy_id:
            remaining = pos.entry_lock_until - now
            return ArbitrationDecision(
                ArbitrationResult.DEFER,
                f"Entry locked by {pos.entry_lock_owner} ({remaining:.0f}s)",
                defer_until=pos.entry_lock_until
            )

        # Try to acquire lock
        lock_duration = self.LOCK_DURATIONS.get(strategy_id, 60)
        lock_until = now + lock_duration

        if not self.state.set_entry_lock(symbol, strategy_id, lock_until):
            return ArbitrationDecision(
                ArbitrationResult.DEFER,
                "Failed to acquire entry lock"
            )

        # Check for conflicting intents on same symbol
        pending = self._pending_intents.get(symbol, [])
        exits_pending = any(i.intent_type == IntentType.EXIT for i in pending)

        if exits_pending:
            self.state.release_entry_lock(symbol, strategy_id)
            return ArbitrationDecision(
                ArbitrationResult.DEFER,
                "Exit intent pending for symbol"
            )

        return ArbitrationDecision(ArbitrationResult.PROCEED)

    def add_pending(self, intent: Intent) -> None:
        """Add intent to pending queue for a symbol."""
        symbol = intent.symbol
        if symbol not in self._pending_intents:
            self._pending_intents[symbol] = []
        self._pending_intents[symbol].append(intent)

    def remove_pending(self, intent: Intent) -> None:
        """Remove intent from pending queue."""
        symbol = intent.symbol
        if symbol in self._pending_intents:
            self._pending_intents[symbol] = [
                i for i in self._pending_intents[symbol]
                if i.intent_id != intent.intent_id
            ]

    def compute_net_target(self, symbol: str) -> Tuple[int, Dict[str, int]]:
        """
        Compute net target position from all strategy allocations.

        Returns (net_target_qty, {strategy_id: allocation_qty})
        """
        pos = self.state.get_position(symbol)
        allocations = {
            sid: alloc.qty
            for sid, alloc in pos.allocations.items()
        }
        net_target = sum(allocations.values())
        return net_target, allocations

    def compute_trade_qty(self, symbol: str, desired_total: int) -> int:
        """Compute trade quantity to reach desired total position."""
        pos = self.state.get_position(symbol)
        return desired_total - pos.real_qty
