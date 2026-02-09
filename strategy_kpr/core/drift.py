"""OMS drift detection and handling for KPR strategy.

Detects divergence between strategy's assumed state and broker truth,
freezes trading when drift detected, and reconciles from truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set
from loguru import logger


@dataclass
class DriftEvent:
    """A single drift event detected."""

    drift_type: str  # POSITION_MISMATCH, MISSING_LOCAL, MISSING_BROKER, ORDER_ORPHAN
    symbol: str
    local_qty: int = 0
    broker_qty: int = 0
    detail: str = ""


class DriftMonitor:
    """Detects and handles drift between local state and OMS/broker truth.

    Attributes:
        global_trade_block: If True, block all new entries.
        reconcile_needed: If True, reconciliation from broker truth pending.
        last_drift_events: Most recent drift events detected.
    """

    def __init__(self):
        """Initialize drift monitor."""
        self.global_trade_block: bool = False
        self.reconcile_needed: bool = False
        self.last_drift_events: List[DriftEvent] = []

    def compute_drift(
        self,
        local_positions: Dict[str, int],  # symbol -> qty
        broker_positions: Dict[str, int],  # symbol -> qty
        local_orders: Set[str] = None,  # order_ids in flight
        broker_orders: Set[str] = None,  # order_ids at broker
    ) -> List[DriftEvent]:
        """Compare local state vs broker truth, return drift events.

        Args:
            local_positions: Local view of positions.
            broker_positions: Broker truth positions.
            local_orders: Local pending order IDs.
            broker_orders: Broker open order IDs.

        Returns:
            List of DriftEvent for any mismatches found.
        """
        events = []
        local_orders = local_orders or set()
        broker_orders = broker_orders or set()

        # Position drift: broker has position that differs from local
        for sym, broker_qty in broker_positions.items():
            local_qty = local_positions.get(sym, 0)
            if local_qty != broker_qty:
                events.append(DriftEvent(
                    drift_type="POSITION_MISMATCH",
                    symbol=sym,
                    local_qty=local_qty,
                    broker_qty=broker_qty,
                    detail=f"local={local_qty} broker={broker_qty}",
                ))

        # Missing at broker: local has position but broker doesn't
        for sym, local_qty in local_positions.items():
            if sym not in broker_positions and local_qty > 0:
                events.append(DriftEvent(
                    drift_type="MISSING_BROKER",
                    symbol=sym,
                    local_qty=local_qty,
                    broker_qty=0,
                    detail="Local has position, broker doesn't",
                ))

        # Order drift: local order not at broker
        for oid in local_orders - broker_orders:
            events.append(DriftEvent(
                drift_type="ORDER_ORPHAN_LOCAL",
                symbol="",
                detail=f"order_id={oid} (local only)",
            ))

        # Broker order not tracked locally
        for oid in broker_orders - local_orders:
            events.append(DriftEvent(
                drift_type="ORDER_ORPHAN_BROKER",
                symbol="",
                detail=f"order_id={oid} (broker only)",
            ))

        return events

    def handle_drift(self, events: List[DriftEvent]) -> bool:
        """Process drift events.

        Args:
            events: List of drift events detected.

        Returns:
            True if trade block was activated, False otherwise.
        """
        self.last_drift_events = events

        if events:
            self.global_trade_block = True
            self.reconcile_needed = True
            for e in events:
                logger.warning(
                    f"Drift detected: {e.drift_type} {e.symbol} - {e.detail}"
                )
            return True
        return False

    def clear_after_reconcile(self) -> None:
        """Clear block after successful reconciliation."""
        self.global_trade_block = False
        self.reconcile_needed = False
        self.last_drift_events = []
        logger.info("Drift cleared after reconciliation")

    def block_on_oms_unavailable(self) -> None:
        """Block trading when OMS is unavailable."""
        self.global_trade_block = True
        logger.warning("Trade block activated: OMS unavailable")
