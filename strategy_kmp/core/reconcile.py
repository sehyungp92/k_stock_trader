"""OMS reconciliation for KMP strategy.

Rebuilds exposure state from OMS truth to handle fills that arrive
asynchronously and ensure sector caps remain accurate.
"""

from typing import Dict, Set
from loguru import logger

from kis_core import SectorExposure
from .state import SymbolState, State
from ..config.universe_meta import TickerMeta
from ..config.constants import STRATEGY_ID


async def reconcile_exposure(
    oms,
    states: Dict[str, SymbolState],
    exposure: SectorExposure,
    meta: Dict[str, TickerMeta],
) -> None:
    """Rebuild exposure from OMS positions.

    Called periodically (every 1-2s) to ensure exposure tracking
    matches actual fills and prevent sector cap drift.

    Args:
        oms: OMSClient instance.
        states: Dict of ticker -> SymbolState.
        exposure: SectorExposure to rebuild.
        meta: Dict of ticker -> TickerMeta with sector info.
    """
    try:
        positions = await oms.get_all_positions()
    except Exception as e:
        logger.debug(f"Reconciliation failed: {e}")
        return

    # Build position data for reconciliation: symbol -> (qty, price)
    position_data: Dict[str, tuple] = {}
    working_orders: Set[str] = set()

    # Collect open positions
    for ticker, pos in positions.items():
        alloc_qty = pos.get_allocation(STRATEGY_ID)
        if alloc_qty <= 0:
            continue

        alloc = pos.allocations.get(STRATEGY_ID)
        entry_px = getattr(alloc, 'cost_basis', 0.0) if alloc else 0.0
        position_data[ticker] = (alloc_qty, entry_px)

        # Sync local state if position exists but state doesn't reflect it
        s = states.get(ticker)
        if s and s.fsm not in (State.IN_POSITION, State.DONE):
            s.fsm = State.IN_POSITION
            s.qty = alloc_qty
            if entry_px:
                s.entry_px = entry_px
            logger.info(f"{ticker}: Reconciled to IN_POSITION, qty={alloc_qty}")

    # Collect working orders (ARMED states awaiting fill)
    for s in states.values():
        if s.fsm == State.ARMED:
            working_orders.add(s.code)

    # Reconcile exposure state
    exposure.reconcile(position_data, working_orders)

    # Check for positions closed externally
    for ticker, s in states.items():
        if s.fsm == State.IN_POSITION:
            pos = positions.get(ticker)
            alloc_qty = pos.get_allocation(STRATEGY_ID) if pos else 0
            if alloc_qty <= 0:
                # Position closed externally
                s.fsm = State.DONE
                logger.info(f"{ticker}: Position closed externally, DONE")
