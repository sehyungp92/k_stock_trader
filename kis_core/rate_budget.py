"""Rate budget management for KIS REST API calls."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import time
import threading


@dataclass
class TokenBucket:
    """Token bucket rate limiter."""
    capacity: int
    refill_rate: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def try_consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.refill_rate)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class RateLimitedError(Exception):
    pass


class RateBudget:
    """Manages rate budgets for different API endpoint classes."""

    DEFAULT_BUDGETS = {
        "QUOTE": (60, 1.0), "CHART": (60, 2.0), "FLOW": (40, 1.0),
        "ORDER": (30, 0.5), "BALANCE": (20, 0.33), "DEFAULT": (30, 0.5),
    }

    def __init__(self, budgets: Optional[Dict[str, tuple]] = None):
        budget_config = {**self.DEFAULT_BUDGETS, **(budgets or {})}
        self.buckets = {k: TokenBucket(v[0], v[1]) for k, v in budget_config.items()}

    def try_consume(self, endpoint_class: str, cost: int = 1) -> bool:
        bucket = self.buckets.get(endpoint_class, self.buckets["DEFAULT"])
        return bucket.try_consume(cost)

    async def call_rest(self, endpoint_class: str, fn: Callable, *args, cost: int = 1, **kwargs) -> Any:
        bucket = self.buckets.get(endpoint_class, self.buckets["DEFAULT"])
        if not bucket.try_consume(cost):
            raise RateLimitedError(f"{endpoint_class} rate limited")
        return await fn(*args, **kwargs)
