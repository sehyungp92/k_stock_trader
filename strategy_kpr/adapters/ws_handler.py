"""
KPR WebSocket Handler.

Handles tick messages for HOT tier symbols, updating VWAPLedger
and providing real-time tick data for MicroPressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Optional, Set

from loguru import logger

from kis_core import KISWebSocketClient, BaseSubscriptionManager, TickMessage, VWAPLedger

# Max HOT tier subscriptions (KIS limit is 41, use 40 to leave room for notifications)
KPR_HOT_MAX_SUBS = 40


@dataclass
class KPRTickState:
    """Per-symbol tick state for KPR strategy."""

    last_price: float = 0.0
    last_volume: float = 0.0
    cum_vol: float = 0.0
    cum_val: float = 0.0
    last_tick_ts: Optional[datetime] = None


class KPRSubscriptionManager(BaseSubscriptionManager):
    """
    KPR-specific subscription manager for HOT tier symbols.

    Only subscribes to tick stream (H0STCNT0), no bid/ask.
    Limited to KPR_HOT_MAX_SUBS subscriptions (KIS limit is 41).
    """

    def __init__(self, ws_client: KISWebSocketClient):
        super().__init__(ws_client, max_regs=KPR_HOT_MAX_SUBS)

    async def ensure_hot(self, ticker: str) -> bool:
        """Ensure HOT tier symbol has tick subscription."""
        return await self.ensure_tick(ticker)

    async def demote_from_hot(self, ticker: str) -> None:
        """Remove tick subscription when symbol demoted from HOT."""
        await self.drop_tick(ticker)


def make_kpr_tick_handler(
    vwap_ledgers: Dict[str, VWAPLedger],
    tick_states: Dict[str, KPRTickState],
    hot_set: Set[str],
    on_hot_tick: Optional[Callable[[str, TickMessage], None]] = None,
):
    """
    Create a tick handler for KPR strategy.

    Updates VWAPLedger from cumulative fields and tracks tick state
    for HOT tier symbols.

    Args:
        vwap_ledgers: Dict of ticker -> VWAPLedger (updates cum_vol/cum_pv).
        tick_states: Dict of ticker -> KPRTickState (updates last price/volume).
        hot_set: Set of HOT tier tickers (only process these).
        on_hot_tick: Optional callback for each HOT tick (for MicroPressure).

    Returns:
        Callback function for KISWebSocketClient.on_tick().
    """

    def handler(msg: TickMessage) -> None:
        # Only process HOT tier symbols
        if msg.ticker not in hot_set:
            return

        # Update tick state
        ts = tick_states.get(msg.ticker)
        if ts is None:
            ts = KPRTickState()
            tick_states[msg.ticker] = ts

        ts.last_price = msg.price
        ts.last_volume = msg.volume
        ts.cum_vol = msg.cum_vol
        ts.cum_val = msg.cum_val
        ts.last_tick_ts = msg.timestamp

        # Update VWAPLedger from cumulative fields (more accurate than bar-based)
        ledger = vwap_ledgers.get(msg.ticker)
        if ledger and msg.cum_vol > 0 and msg.cum_val > 0:
            # Use cumulative from exchange directly (replaces bar accumulation)
            ledger.cum_vol = msg.cum_vol
            ledger.cum_pv = msg.cum_val

        # Optional callback for strategy-specific processing
        if on_hot_tick:
            try:
                on_hot_tick(msg.ticker, msg)
            except Exception as e:
                logger.debug(f"KPR on_hot_tick callback error: {e}")

    return handler


async def sync_hot_subscriptions(
    subs: KPRSubscriptionManager,
    current_hot: Set[str],
    new_hot: Set[str],
) -> None:
    """
    Sync WebSocket subscriptions with current HOT tier set.

    Subscribes new HOT tickers and unsubscribes demoted tickers.
    """
    # Unsubscribe demoted
    for ticker in current_hot - new_hot:
        await subs.demote_from_hot(ticker)
        logger.debug(f"KPR WS: Demoted {ticker} from HOT")

    # Subscribe promoted
    for ticker in new_hot - current_hot:
        if await subs.ensure_hot(ticker):
            logger.debug(f"KPR WS: Promoted {ticker} to HOT")
