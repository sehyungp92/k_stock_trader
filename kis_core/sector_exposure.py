"""Unified sector exposure tracking.

Consolidates sector cap enforcement from KMP, KPR, and Nulrimok strategies.
Supports dual-mode (count + percentage), race condition prevention via
reserve/unreserve, and reconciliation from OMS truth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Set


@dataclass
class SectorExposureConfig:
    """Configuration for sector exposure limits.

    Attributes:
        mode: Enforcement mode - "count", "pct", or "both".
        max_positions_per_sector: Maximum positions per sector (count mode).
        max_sector_pct: Maximum sector exposure as fraction of equity (pct mode).
        unknown_sector_policy: How to handle symbols with no sector mapping.
    """

    mode: Literal["count", "pct", "both"] = "both"
    max_positions_per_sector: int = 2
    max_sector_pct: float = 0.30
    unknown_sector_policy: Literal["allow", "block"] = "allow"


class SectorExposure:
    """Unified sector cap with race condition prevention.

    Combines best features from:
    - KMP ExposureState: reserve/unreserve for race prevention, reconciliation
    - KPR SectorLimiter: unknown sector bypass, symbol-to-sector lookup
    - Nulrimok: percentage-based limits for varying position sizes

    Usage:
        exposure = SectorExposure(sym_to_sector, config)

        # Before submitting order
        if exposure.can_enter(symbol, qty, price, equity):
            exposure.reserve(symbol, qty, price)
            try:
                result = await oms.submit_intent(intent)
            except:
                exposure.unreserve(symbol, qty, price)

        # On fill
        exposure.on_fill(symbol, qty, price)

        # On position close
        exposure.on_close(symbol, qty, price)

        # Periodic reconciliation
        exposure.reconcile(positions, working_orders)
    """

    def __init__(
        self,
        sym_to_sector: Dict[str, str],
        config: Optional[SectorExposureConfig] = None,
    ):
        """Initialize sector exposure tracker.

        Args:
            sym_to_sector: Mapping of symbol to sector code.
            config: Exposure limit configuration.
        """
        self.sym_to_sector = sym_to_sector
        self.config = config or SectorExposureConfig()

        # Count-based tracking (from KMP)
        self.sector_open_count: Dict[str, int] = {}
        self.sector_working_count: Dict[str, int] = {}

        # Notional-based tracking (from Nulrimok)
        self.sector_open_notional: Dict[str, float] = {}
        self.sector_working_notional: Dict[str, float] = {}

    def get_sector(self, symbol: str) -> str:
        """Get sector for symbol.

        Args:
            symbol: Symbol code.

        Returns:
            Sector code or "UNKNOWN" if not mapped.
        """
        return self.sym_to_sector.get(symbol, "UNKNOWN")

    def can_enter(
        self,
        symbol: str,
        qty: int,
        price: float,
        equity: float,
    ) -> bool:
        """Check if entry is allowed under sector cap.

        Args:
            symbol: Symbol to check.
            qty: Order quantity.
            price: Entry price.
            equity: Current account equity.

        Returns:
            True if entry allowed, False if cap would be exceeded.
        """
        sector = self.get_sector(symbol)

        # Handle unknown sector
        if sector == "UNKNOWN":
            return self.config.unknown_sector_policy == "allow"

        mode = self.config.mode
        notional = qty * price

        # Count-based check
        if mode in ("count", "both"):
            open_count = self.sector_open_count.get(sector, 0)
            working_count = self.sector_working_count.get(sector, 0)
            total_count = open_count + working_count
            if total_count >= self.config.max_positions_per_sector:
                return False

        # Percentage-based check
        if mode in ("pct", "both") and equity > 0:
            open_notional = self.sector_open_notional.get(sector, 0.0)
            working_notional = self.sector_working_notional.get(sector, 0.0)
            total_notional = open_notional + working_notional + notional
            if total_notional / equity >= self.config.max_sector_pct:
                return False

        return True

    def reserve(self, symbol: str, qty: int = 1, price: float = 0.0) -> None:
        """Reserve a slot BEFORE sending order to prevent races.

        Args:
            symbol: Symbol being ordered.
            qty: Order quantity.
            price: Order price (for notional calculation).
        """
        sector = self.get_sector(symbol)
        if sector == "UNKNOWN":
            return

        # Count reservation
        self.sector_working_count[sector] = (
            self.sector_working_count.get(sector, 0) + 1
        )

        # Notional reservation
        notional = qty * price
        self.sector_working_notional[sector] = (
            self.sector_working_notional.get(sector, 0.0) + notional
        )

    def unreserve(self, symbol: str, qty: int = 1, price: float = 0.0) -> None:
        """Release reservation on order failure/cancel/rejection.

        Args:
            symbol: Symbol that was reserved.
            qty: Reserved quantity.
            price: Reserved price.
        """
        sector = self.get_sector(symbol)
        if sector == "UNKNOWN":
            return

        # Count release
        self.sector_working_count[sector] = max(
            0, self.sector_working_count.get(sector, 0) - 1
        )

        # Notional release
        notional = qty * price
        self.sector_working_notional[sector] = max(
            0.0, self.sector_working_notional.get(sector, 0.0) - notional
        )

    def on_fill(self, symbol: str, qty: int = 1, price: float = 0.0) -> None:
        """Move from working -> open on fill confirmation.

        Args:
            symbol: Symbol that was filled.
            qty: Fill quantity.
            price: Fill price.
        """
        sector = self.get_sector(symbol)
        if sector == "UNKNOWN":
            return

        # Move count from working to open
        self.sector_working_count[sector] = max(
            0, self.sector_working_count.get(sector, 0) - 1
        )
        self.sector_open_count[sector] = (
            self.sector_open_count.get(sector, 0) + 1
        )

        # Move notional from working to open
        notional = qty * price
        self.sector_working_notional[sector] = max(
            0.0, self.sector_working_notional.get(sector, 0.0) - notional
        )
        self.sector_open_notional[sector] = (
            self.sector_open_notional.get(sector, 0.0) + notional
        )

    def on_close(self, symbol: str, qty: int = 1, price: float = 0.0) -> None:
        """Decrement open count on position close.

        Args:
            symbol: Symbol being closed.
            qty: Close quantity.
            price: Close price (or entry price for notional calc).
        """
        sector = self.get_sector(symbol)
        if sector == "UNKNOWN":
            return

        # Decrement count
        self.sector_open_count[sector] = max(
            0, self.sector_open_count.get(sector, 0) - 1
        )

        # Decrement notional
        notional = qty * price
        self.sector_open_notional[sector] = max(
            0.0, self.sector_open_notional.get(sector, 0.0) - notional
        )

    def reset(self) -> None:
        """Clear all exposure counts for reconciliation rebuild."""
        self.sector_open_count.clear()
        self.sector_working_count.clear()
        self.sector_open_notional.clear()
        self.sector_working_notional.clear()

    def reconcile(
        self,
        positions: Dict[str, tuple],
        working_orders: Optional[Set[str]] = None,
    ) -> None:
        """Rebuild exposure state from OMS truth.

        Args:
            positions: Dict of symbol -> (qty, entry_price) for open positions.
            working_orders: Set of symbols with pending entry orders.
        """
        self.reset()

        # Rebuild open positions
        for symbol, (qty, price) in positions.items():
            sector = self.get_sector(symbol)
            if sector == "UNKNOWN":
                continue
            self.sector_open_count[sector] = (
                self.sector_open_count.get(sector, 0) + 1
            )
            notional = qty * price
            self.sector_open_notional[sector] = (
                self.sector_open_notional.get(sector, 0.0) + notional
            )

        # Rebuild working orders (count only, notional unknown)
        if working_orders:
            for symbol in working_orders:
                sector = self.get_sector(symbol)
                if sector == "UNKNOWN":
                    continue
                self.sector_working_count[sector] = (
                    self.sector_working_count.get(sector, 0) + 1
                )

    def count_in_sector(self, sector: str, include_working: bool = True) -> int:
        """Count positions in a given sector.

        Args:
            sector: Sector code.
            include_working: Whether to include working orders.

        Returns:
            Count of positions in the sector.
        """
        count = self.sector_open_count.get(sector, 0)
        if include_working:
            count += self.sector_working_count.get(sector, 0)
        return count

    def notional_in_sector(self, sector: str, include_working: bool = True) -> float:
        """Get notional exposure in a given sector.

        Args:
            sector: Sector code.
            include_working: Whether to include working orders.

        Returns:
            Notional exposure in the sector.
        """
        notional = self.sector_open_notional.get(sector, 0.0)
        if include_working:
            notional += self.sector_working_notional.get(sector, 0.0)
        return notional

    def sector_pct(self, sector: str, equity: float, include_working: bool = True) -> float:
        """Get sector exposure as percentage of equity.

        Args:
            sector: Sector code.
            equity: Account equity.
            include_working: Whether to include working orders.

        Returns:
            Sector exposure as fraction of equity.
        """
        if equity <= 0:
            return 0.0
        return self.notional_in_sector(sector, include_working) / equity
