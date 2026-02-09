"""
WebSocket Subscription Budget Manager for KMP.

Manages KIS WS registration limits (max 20 combined).
Extends BaseSubscriptionManager with KMP-specific focus set logic.
"""

from __future__ import annotations
from typing import Dict, Optional, Set
from loguru import logger

from kis_core import KISWebSocketClient, BaseSubscriptionManager

from ..config.constants import WS_MAX_REGS, FOCUS_MAX
from ..core.state import SymbolState, State
from ..core.tick_table import tick_size


class SubscriptionManager(BaseSubscriptionManager):
    """
    KMP-specific subscription manager.

    Extends BaseSubscriptionManager with:
    - Focus set limit (FOCUS_MAX) for bid/ask subscriptions
    - Eviction priority: tick-only subscriptions first
    """

    def __init__(self, ws_client: KISWebSocketClient):
        """
        Args:
            ws_client: KISWebSocketClient instance.
        """
        super().__init__(ws_client, max_regs=WS_MAX_REGS)
        self._focus_max = FOCUS_MAX

    @property
    def sub_cnt(self) -> Set[str]:
        """Tick subscriptions (H0STCNT0)."""
        return self.tick_subs

    @property
    def sub_asp(self) -> Set[str]:
        """Bid/ask subscriptions (H0STASP0)."""
        return self.asp_subs

    async def ensure_cnt(self, ticker: str) -> bool:
        """Ensure ticker has tick subscription."""
        return await self.ensure_tick(ticker)

    async def ensure_asp(self, ticker: str) -> bool:
        """Ensure ticker has bid/ask subscription (within focus limit)."""
        if ticker in self.asp_subs:
            return True
        if len(self.asp_subs) >= self._focus_max:
            return False
        return await self.ensure_askbid(ticker)

    async def drop_cnt(self, ticker: str) -> None:
        """Drop tick subscription."""
        await self.drop_tick(ticker)

    async def drop_asp(self, ticker: str) -> None:
        """Drop bid/ask subscription."""
        await self.drop_askbid(ticker)

    async def _evict_for_tick(self, incoming: str) -> None:
        """Evict tick-only subscription (not in focus set)."""
        for t in list(self.tick_subs):
            if t not in self.asp_subs:
                await self.drop_tick(t)
                return


async def refresh_focus_list(
    states: Dict[str, SymbolState],
    subs: SubscriptionManager,
    last_prices: Dict[str, float] | None = None,
) -> None:
    """
    Refresh ASP focus set with near-trigger prioritisation.

    Priority 0: ARMED / IN_POSITION (always focus)
    Priority 1: WAIT_ACCEPTANCE within 5 ticks of OR high
    Priority 2: other WAIT_ACCEPTANCE
    """
    if last_prices is None:
        last_prices = {}

    focus: list[tuple[int, str]] = []

    for s in states.values():
        if s.fsm in (State.ARMED, State.IN_POSITION):
            focus.append((0, s.code))
        elif s.fsm == State.WAIT_ACCEPTANCE:
            px = last_prices.get(s.code, 0.0)
            if px > 0 and s.or_high > 0:
                ts = tick_size(px)
                distance_ticks = (s.or_high - px) / ts if ts > 0 else 999
                prio = 1 if distance_ticks <= 5 else 2
            else:
                prio = 2
            focus.append((prio, s.code))

    focus.sort()
    selected = [t for _, t in focus[:FOCUS_MAX]]

    for ticker in selected:
        await subs.ensure_asp(ticker)

    # Drop ASP for symbols no longer in focus
    for ticker in list(subs.sub_asp):
        if ticker not in selected:
            await subs.drop_asp(ticker)

    # Drop CNT+ASP for DONE symbols
    for s in states.values():
        if s.fsm == State.DONE and s.code in subs.sub_cnt:
            await subs.drop_all(s.code)


async def release_non_position_slots(
    states: Dict[str, SymbolState],
    subs: SubscriptionManager,
) -> int:
    """
    Release WS slots for symbols not in position.

    Called after entry cutoff (10:00) to free slots for Nulrimok handoff.
    Keeps subscriptions only for IN_POSITION symbols that need exit monitoring.

    Returns:
        Number of slots released.
    """
    released = 0
    for s in states.values():
        if s.fsm != State.IN_POSITION and s.code in subs.sub_cnt:
            await subs.drop_all(s.code)
            released += 1

    if released > 0:
        in_position = sum(1 for s in states.values() if s.fsm == State.IN_POSITION)
        logger.info(f"Released {released} WS slots after entry cutoff. Keeping {in_position} for position monitoring.")
