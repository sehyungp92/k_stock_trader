"""KPR Program Provider.

Per-stock program flow is not available via current API (only market-wide).
Mark as UNAVAILABLE to enforce two-pillar mode (investor + micro only).
"""

from enum import Enum, auto
from loguru import logger


class ProgramSignal(Enum):
    ACCUMULATE = auto()
    NEUTRAL = auto()
    DISTRIBUTE = auto()
    STALE = auto()
    UNAVAILABLE = auto()


class ProgramProvider:
    """Program pillar stub - always UNAVAILABLE (per-stock data not available)."""

    def __init__(self, api=None, rate_budget=None):
        self.available: bool = False

    async def probe(self) -> None:
        """Per-stock program data unavailable; force two-pillar mode."""
        self.available = False
        logger.info("ProgramProvider: per-stock unavailable, two-pillar mode active")

    async def fetch(self, ticker: str) -> ProgramSignal:
        return ProgramSignal.UNAVAILABLE
