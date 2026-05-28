from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from strategy_common.actions import StrategyAction
from strategy_common.events import DecisionEvent


@dataclass(frozen=True, slots=True)
class KALCBOrderUpdateEvent:
    order_id: str
    symbol: str
    status: str
    timestamp: datetime
    role: str = ""
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KALCBFillEvent:
    order_id: str
    symbol: str
    side: str
    qty: int
    price: float
    timestamp: datetime
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KALCBPortfolioView:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    open_positions: int = 0
    sector_counts: dict[str, int] = field(default_factory=dict)
    open_risk: float = 0.0
    open_notional: float = 0.0
    equity: float = 0.0
    high_water_equity: float = 0.0
    drawdown_pct: float = 0.0
    session_start_equity: float = 0.0
    session_return_pct: float = 0.0


@dataclass(slots=True)
class KALCBCoreResult:
    state: Any
    actions: list[StrategyAction] = field(default_factory=list)
    decisions: list[DecisionEvent] = field(default_factory=list)
