"""
Shared Priority Rate Budget for Multi-Strategy KIS API Access.

Provides centralized rate limiting with time-based priority windows
for coordinating API access across multiple strategies.

Features:
- Priority-aware token bucket
- Time-window based priority boost
- File-based coordination for multi-process (Redis optional)
- SharedRateBudgetClient for strategies to connect to
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

from loguru import logger

# Cross-platform file locking
if sys.platform == 'win32':
    import msvcrt

    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Priority Windows Configuration
# ---------------------------------------------------------------------------

# Time windows where specific strategies get priority access
# Format: {strategy_id: [((start_hour, start_min), (end_hour, end_min)), ...]}
PRIORITY_WINDOWS: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {
    "KMP": [((9, 15), (10, 0))],      # KMP gets priority 09:15-10:00 (opening breakdown)
    "KPR": [],                         # No special priority
    "NULRIMOK": [((8, 0), (8, 30))],  # DSE priority 08:00-08:30
    "PCIM": [],                        # No special priority
}

# Priority multipliers
PRIORITY_BOOST = 2.0      # Strategy in priority window gets 2x tokens
PRIORITY_PENALTY = 0.5    # Other strategies get 0.5x during competing priority


@dataclass
class PriorityTokenBucket:
    """Token bucket with priority-aware consumption."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def _get_active_priority_strategy(self) -> Optional[str]:
        """Check if any strategy is currently in its priority window."""
        now = datetime.now()
        current_time = (now.hour, now.minute)

        for strategy_id, windows in PRIORITY_WINDOWS.items():
            for (start_h, start_m), (end_h, end_m) in windows:
                start_time = (start_h, start_m)
                end_time = (end_h, end_m)

                if start_time <= current_time < end_time:
                    return strategy_id
        return None

    def get_effective_multiplier(self, strategy_id: str) -> float:
        """Get the effective token multiplier for a strategy based on priority."""
        active_priority = self._get_active_priority_strategy()

        if active_priority is None:
            return 1.0  # No priority active, normal rate

        if strategy_id.upper() == active_priority:
            return PRIORITY_BOOST  # This strategy has priority

        return PRIORITY_PENALTY  # Another strategy has priority

    def try_consume(self, tokens: int = 1, strategy_id: str = "") -> bool:
        """
        Try to consume tokens with priority adjustment.

        In priority window:
        - Priority strategy: cost is divided by PRIORITY_BOOST (gets more calls)
        - Other strategies: cost is multiplied by 1/PRIORITY_PENALTY (gets fewer calls)
        """
        with self._lock:
            now = time.time()
            # Refill tokens
            self.tokens = min(
                self.capacity,
                self.tokens + (now - self.last_refill) * self.refill_rate
            )
            self.last_refill = now

            # Apply priority multiplier
            multiplier = self.get_effective_multiplier(strategy_id)
            effective_cost = tokens / multiplier

            if self.tokens >= effective_cost:
                self.tokens -= effective_cost
                return True
            return False

    def available_tokens(self, strategy_id: str = "") -> float:
        """Get available tokens for a strategy (accounting for priority)."""
        with self._lock:
            now = time.time()
            current_tokens = min(
                self.capacity,
                self.tokens + (now - self.last_refill) * self.refill_rate
            )
            multiplier = self.get_effective_multiplier(strategy_id)
            return current_tokens * multiplier


class RateLimitedError(Exception):
    """Raised when rate limit is exceeded."""
    pass


# ---------------------------------------------------------------------------
# Shared Rate Budget (Multi-Process Coordination via File Lock)
# ---------------------------------------------------------------------------

@dataclass
class SharedBucketState:
    """Serializable bucket state for file-based sharing."""
    tokens: float
    last_refill: float
    capacity: int
    refill_rate: float


class SharedRateBudget:
    """
    Shared rate budget with file-based coordination for multi-process access.

    Uses file locking (fcntl) to coordinate between processes.
    Each endpoint class has its own bucket.
    """

    DEFAULT_BUDGETS = {
        "QUOTE": (60, 1.0),     # High frequency quotes
        "CHART": (30, 0.5),     # Chart data
        "FLOW": (20, 0.33),     # Order flow
        "ORDER": (30, 0.5),     # Order submission
        "BALANCE": (20, 0.33),  # Balance queries
        "DEFAULT": (30, 0.5),   # Fallback
    }

    def __init__(
        self,
        state_file: Optional[str] = None,
        budgets: Optional[Dict[str, tuple]] = None,
    ):
        """
        Initialize shared rate budget.

        Args:
            state_file: Path to file for multi-process coordination.
                       If None, uses in-memory only (single process).
            budgets: Override default budget configs {class: (capacity, refill_rate)}
        """
        self.state_file = state_file
        self._state_path = Path(state_file) if state_file else None

        budget_config = {**self.DEFAULT_BUDGETS, **(budgets or {})}
        self.buckets: Dict[str, PriorityTokenBucket] = {
            k: PriorityTokenBucket(v[0], v[1])
            for k, v in budget_config.items()
        }

        self._file_lock = threading.Lock()

        # Initialize state file if needed
        if self._state_path:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._state_path.exists():
                self._save_state()

    def _load_state(self) -> Dict[str, SharedBucketState]:
        """Load bucket states from file."""
        if not self._state_path or not self._state_path.exists():
            return {}

        try:
            with open(self._state_path, 'r') as f:
                data = json.load(f)
            return {
                k: SharedBucketState(**v)
                for k, v in data.items()
            }
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save_state(self) -> None:
        """Save bucket states to file."""
        if not self._state_path:
            return

        data = {
            k: {
                "tokens": bucket.tokens,
                "last_refill": bucket.last_refill,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
            }
            for k, bucket in self.buckets.items()
        }

        with open(self._state_path, 'w') as f:
            json.dump(data, f)

    def _sync_from_file(self) -> None:
        """Sync in-memory state from file."""
        if not self._state_path:
            return

        states = self._load_state()
        for k, state in states.items():
            if k in self.buckets:
                bucket = self.buckets[k]
                bucket.tokens = state.tokens
                bucket.last_refill = state.last_refill

    def try_consume(
        self,
        endpoint_class: str,
        strategy_id: str = "",
        cost: int = 1,
    ) -> bool:
        """
        Try to consume tokens for an endpoint class.

        Args:
            endpoint_class: API endpoint class (QUOTE, ORDER, etc.)
            strategy_id: Strategy requesting the tokens (for priority)
            cost: Number of tokens to consume

        Returns:
            True if tokens consumed, False if rate limited
        """
        bucket = self.buckets.get(endpoint_class, self.buckets["DEFAULT"])

        if self._state_path:
            with self._file_lock:
                # Lock file for multi-process coordination
                try:
                    with open(self._state_path, 'r+') as f:
                        lock_file(f)
                        try:
                            self._sync_from_file()
                            result = bucket.try_consume(cost, strategy_id)
                            self._save_state()
                        finally:
                            unlock_file(f)
                except (IOError, OSError):
                    # Fallback to in-memory only if file access fails
                    result = bucket.try_consume(cost, strategy_id)
        else:
            result = bucket.try_consume(cost, strategy_id)

        return result

    def get_priority_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get current priority status for a strategy."""
        bucket = self.buckets["DEFAULT"]
        active_priority = bucket._get_active_priority_strategy()
        multiplier = bucket.get_effective_multiplier(strategy_id)

        return {
            "strategy_id": strategy_id,
            "active_priority_strategy": active_priority,
            "your_multiplier": multiplier,
            "is_priority": active_priority == strategy_id.upper(),
            "priority_windows": PRIORITY_WINDOWS.get(strategy_id.upper(), []),
        }

    async def call_rest(
        self,
        endpoint_class: str,
        fn: Callable,
        *args,
        strategy_id: str = "",
        cost: int = 1,
        **kwargs,
    ) -> Any:
        """
        Call a REST function with rate limiting.

        Args:
            endpoint_class: API endpoint class
            fn: Async function to call
            strategy_id: Strategy making the call
            cost: Token cost

        Returns:
            Result from fn

        Raises:
            RateLimitedError: If rate limit exceeded
        """
        if not self.try_consume(endpoint_class, strategy_id, cost):
            raise RateLimitedError(
                f"{endpoint_class} rate limited for {strategy_id}"
            )
        return await fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Client for Strategies
# ---------------------------------------------------------------------------

class SharedRateBudgetClient:
    """
    Client for strategies to connect to shared rate budget.

    Usage:
        budget = SharedRateBudgetClient("KMP", state_file="/var/run/oms/rate_budget.json")

        if budget.try_consume("QUOTE"):
            result = await api.get_quote(symbol)

        # Or use call_rest wrapper:
        result = await budget.call_rest("QUOTE", api.get_quote, symbol)
    """

    def __init__(
        self,
        strategy_id: str,
        state_file: Optional[str] = None,
        budgets: Optional[Dict[str, tuple]] = None,
    ):
        """
        Initialize client for a specific strategy.

        Args:
            strategy_id: Strategy identifier (KMP, KPR, NULRIMOK, PCIM)
            state_file: Path to shared state file for multi-process coordination
            budgets: Override default budget configs
        """
        self.strategy_id = strategy_id.upper()
        self._budget = SharedRateBudget(state_file=state_file, budgets=budgets)

    def try_consume(self, endpoint_class: str, cost: int = 1) -> bool:
        """Try to consume tokens for an endpoint class."""
        return self._budget.try_consume(endpoint_class, self.strategy_id, cost)

    async def call_rest(
        self,
        endpoint_class: str,
        fn: Callable,
        *args,
        cost: int = 1,
        **kwargs,
    ) -> Any:
        """Call REST function with rate limiting."""
        return await self._budget.call_rest(
            endpoint_class,
            fn,
            *args,
            strategy_id=self.strategy_id,
            cost=cost,
            **kwargs,
        )

    def get_priority_status(self) -> Dict[str, Any]:
        """Get current priority status."""
        return self._budget.get_priority_status(self.strategy_id)

    @property
    def is_priority(self) -> bool:
        """Check if this strategy currently has priority."""
        status = self.get_priority_status()
        return status["is_priority"]

    @property
    def multiplier(self) -> float:
        """Get current rate multiplier for this strategy."""
        status = self.get_priority_status()
        return status["your_multiplier"]


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

_default_budget: Optional[SharedRateBudget] = None


def get_shared_budget(state_file: Optional[str] = None) -> SharedRateBudget:
    """
    Get or create the default shared rate budget.

    For multi-process coordination, pass a state_file path that all
    processes can access.
    """
    global _default_budget

    if _default_budget is None:
        _default_budget = SharedRateBudget(state_file=state_file)

    return _default_budget


def create_strategy_client(
    strategy_id: str,
    state_file: Optional[str] = None,
) -> SharedRateBudgetClient:
    """
    Create a rate budget client for a strategy.

    Args:
        strategy_id: Strategy identifier
        state_file: Path to shared state file (e.g., "/var/run/oms/rate_budget.json")

    Returns:
        SharedRateBudgetClient configured for the strategy
    """
    return SharedRateBudgetClient(strategy_id, state_file=state_file)
