"""KPR Investor Signal.

Provides investor flow signals (foreign/institutional net) with caching,
age tracking, and budgeted refresh. Stale data triggers acceptance
and sizing penalties.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Optional, Set
from loguru import logger


class InvestorSignal(Enum):
    STRONG = auto()
    NEUTRAL = auto()
    DISTRIBUTE = auto()
    CONFLICT = auto()
    STALE = auto()
    UNAVAILABLE = auto()


@dataclass
class InvestorFlowData:
    ticker: str
    foreign_net: float = 0.0
    inst_net: float = 0.0
    timestamp: Optional[datetime] = None
    epoch_ts: float = 0.0  # For age calculations

    @property
    def is_stale(self) -> bool:
        if self.timestamp is None:
            return True
        return (datetime.now() - self.timestamp).total_seconds() > 300


class InvestorFlowProvider:
    """Provides investor flow signals with caching and budgeted refresh.

    Key features:
    - Cached snapshots with age tracking
    - Non-blocking dispatch_refresh for targeted symbols
    - Budgeted refresh to avoid rate limit exhaustion
    - Only refresh relevant symbols (HOT/SETUP/ACCEPTING/in-position)
    """

    def __init__(self, api, rate_budget=None):
        self.api = api
        self._cache: Dict[str, InvestorFlowData] = {}
        self._rate_budget = rate_budget
        self._inflight: Set[str] = set()

    def age_sec(self, ticker: str, now: float = None) -> float:
        """Return age of cached data in seconds (inf if missing).

        Args:
            ticker: Symbol code.
            now: Current epoch time (defaults to time.time()).

        Returns:
            Age in seconds, or float("inf") if no cached data.
        """
        now = now or time.time()
        cached = self._cache.get(ticker)
        if cached and cached.epoch_ts > 0:
            return now - cached.epoch_ts
        return float("inf")

    def dispatch_refresh(self, ticker: str) -> None:
        """Non-blocking budgeted refresh for symbol.

        Spawns async task to refresh investor data if budget allows.
        Multiple calls for same ticker are deduplicated.

        Args:
            ticker: Symbol to refresh.
        """
        if ticker in self._inflight:
            return
        if self.age_sec(ticker) < 300:
            return
        if self._rate_budget and not self._rate_budget.try_consume("FLOW"):
            return

        self._inflight.add(ticker)

        async def _refresh_task():
            try:
                rows = self.api.get_investor_trend(ticker, days=5)

                if rows:
                    foreign_net = sum(d.get('foreign_net', 0) for d in rows[:5])
                    inst_net = sum(d.get('inst_net', 0) for d in rows[:5])

                    self._cache[ticker] = InvestorFlowData(
                        ticker=ticker,
                        foreign_net=foreign_net,
                        inst_net=inst_net,
                        timestamp=datetime.now(),
                        epoch_ts=time.time(),
                    )
            except Exception as e:
                logger.debug(f"Investor refresh failed for {ticker}: {e}")
            finally:
                self._inflight.discard(ticker)

        asyncio.create_task(_refresh_task())

    async def fetch(self, ticker: str, max_age: float = 120) -> InvestorSignal:
        """Fetch investor signal, refetching if cache older than max_age seconds."""
        cached = self._cache.get(ticker)
        if cached and cached.timestamp:
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < max_age:
                return self._classify(cached)

        # Rate limit REST calls
        if self._rate_budget and not self._rate_budget.try_consume("FLOW"):
            return self._classify(cached) if cached else InvestorSignal.STALE

        try:
            rows = self.api.get_investor_trend(ticker, days=5)
            if not rows:
                return InvestorSignal.UNAVAILABLE

            foreign_net = sum(d.get('foreign_net', 0) for d in rows[:5])
            inst_net = sum(d.get('inst_net', 0) for d in rows[:5])

            self._cache[ticker] = InvestorFlowData(ticker, foreign_net, inst_net, datetime.now(), epoch_ts=time.time())
            return self._classify(self._cache[ticker])
        except Exception:
            return InvestorSignal.UNAVAILABLE

    def _classify(self, data: InvestorFlowData) -> InvestorSignal:
        if data.foreign_net > 0 and data.inst_net > 0:
            return InvestorSignal.STRONG
        if data.foreign_net < 0 and data.inst_net < 0:
            return InvestorSignal.DISTRIBUTE
        # CONFLICT: one positive, one negative (not both near zero)
        if (data.foreign_net > 0) != (data.inst_net > 0):
            return InvestorSignal.CONFLICT
        return InvestorSignal.NEUTRAL

    def is_stale(self, ticker: str, max_age: float) -> bool:
        cached = self._cache.get(ticker)
        if not cached or not cached.timestamp:
            return True
        return (datetime.now() - cached.timestamp).total_seconds() > max_age
