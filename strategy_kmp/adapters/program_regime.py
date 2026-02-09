"""
Market-wide Program Regime (REST cumulative -> delta -> EWMA).

Uses the program-trade-by-stock endpoint for real cumulative net-buy,
not fluctuation ranking.
"""

from __future__ import annotations
from typing import Dict
from loguru import logger

from ..config.constants import EWMA_ALPHA, PROGRAM_POLL_SEC


class MarketProgramRegime:
    """
    Track market-wide institutional program flow.

    Uses cumulative net-buy values from REST, computes deltas,
    applies EWMA smoothing.
    """

    def __init__(self, alpha: float = EWMA_ALPHA):
        self.alpha = alpha
        self.prev_cum: Dict[str, float] = {}
        self.ewma_delta: Dict[str, float] = {}
        self.last_ok_ts: float = 0.0

    def update(self, market: str, cumulative: float, now_ts: float) -> None:
        if market not in self.prev_cum or cumulative < self.prev_cum[market]:
            # First observation or reset (cumulative went backwards)
            self.prev_cum[market] = cumulative
            self.ewma_delta[market] = 0.0
            return

        delta = cumulative - self.prev_cum[market]
        self.prev_cum[market] = cumulative

        prev = self.ewma_delta.get(market, 0.0)
        self.ewma_delta[market] = self.alpha * delta + (1 - self.alpha) * prev
        self.last_ok_ts = now_ts

    def regime(self) -> str:
        k = self.ewma_delta.get("KOSPI", 0.0)
        q = self.ewma_delta.get("KOSDAQ", 0.0)
        if k > 0 and q > 0:
            return "strong_inflow"
        if k < 0 and q < 0:
            return "outflow"
        return "mixed"

    def multiplier(self) -> float:
        r = self.regime()
        if r == "strong_inflow":
            return 1.10
        if r == "outflow":
            return 0.85
        return 1.00


async def program_poll_task(api, regime: MarketProgramRegime) -> None:
    """
    Background task to poll program trend via the correct endpoint.
    """
    import asyncio
    import time

    while True:
        try:
            now = time.time()
            for mkt in ("KOSPI", "KOSDAQ"):
                data = api.get_program_trend(market=mkt)
                if data:
                    # Prefer ntby_amt (net buy amount) for value-based regime
                    cum = float(data.get("ntby_amt", 0) or data.get("ntby_qty", 0))
                    regime.update(mkt, cum, now)
        except Exception as e:
            logger.debug(f"Program poll error: {e}")

        await asyncio.sleep(PROGRAM_POLL_SEC)
