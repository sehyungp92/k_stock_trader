from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from strategy_common.actions import StrategyAction
from strategy_common.events import DecisionEvent


@dataclass(frozen=True, slots=True)
class OLRFillEvent:
    order_id: str
    symbol: str
    side: str
    qty: int
    price: float
    timestamp: datetime
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OLRExpiredOrderEvent:
    order_id: str
    symbol: str
    side: str
    order_type: str
    qty: int | None
    timestamp: datetime
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OLROrderUpdateEvent:
    order_id: str
    symbol: str
    status: str
    timestamp: datetime
    side: str = ""
    order_type: str = ""
    qty: int | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OLRPortfolioView:
    cash: float
    equity: float
    positions: dict[str, int] = field(default_factory=dict)
    open_positions: int = 0
    open_notional: float = 0.0
    gross_exposure_pct: float = 0.0
    # Intent-only mode: an external allocator must approve and reserve sizes
    # before submission. Default preserves the standalone/live core contract.
    external_entry_allocation: bool = False
    # Explicit research controller: proposals do not consume the session's
    # attempt, and native rank weight is a prior rather than a sizing ceiling.
    entry_proposal_only: bool = False
    economic_size_envelope: bool = False


@dataclass(slots=True)
class OLRCoreResult:
    state: Any
    actions: list[StrategyAction] = field(default_factory=list)
    decisions: list[DecisionEvent] = field(default_factory=list)
