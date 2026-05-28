from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from strategy_olr.models import OLRDailyCandidate


class OLRSymbolStage(str, Enum):
    WATCHING = "WATCHING"
    ENTRY_QUEUED = "ENTRY_QUEUED"
    IN_POSITION = "IN_POSITION"
    EXIT_QUEUED = "EXIT_QUEUED"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


@dataclass(slots=True)
class OLRPositionState:
    symbol: str
    qty_open: int
    entry_price: float
    entry_time: datetime
    candidate_rank: int
    candidate_score: float
    source_artifact_hash: str = ""
    sector: str = "UNKNOWN"
    entry_order_id: str = ""
    exit_order_id: str = ""
    max_favorable_price: float = 0.0
    max_adverse_price: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_favorable_price <= 0.0:
            self.max_favorable_price = self.entry_price
        if self.max_adverse_price <= 0.0:
            self.max_adverse_price = self.entry_price

    def update_mark(self, high: float, low: float) -> None:
        self.max_favorable_price = max(self.max_favorable_price, float(high))
        self.max_adverse_price = min(self.max_adverse_price, float(low))


@dataclass(slots=True)
class OLRSymbolState:
    symbol: str
    stage: OLRSymbolStage = OLRSymbolStage.WATCHING
    session_date: date | None = None
    candidate: OLRDailyCandidate | None = None
    pending_entry_order_id: str = ""
    pending_exit_order_id: str = ""
    pending_entry_metadata: dict[str, Any] = field(default_factory=dict)
    pending_exit_metadata: dict[str, Any] = field(default_factory=dict)
    session_bars: list[Any] = field(default_factory=list)
    position: OLRPositionState | None = None
    entry_attempted: bool = False
    exit_attempted_dates: set[date] = field(default_factory=set)
    last_decision_code: str = ""
    last_decision_details: dict[str, Any] = field(default_factory=dict)

    def reset_for_session(self, session_date: date, candidate: OLRDailyCandidate | None) -> None:
        self.session_date = session_date
        self.candidate = candidate
        self.pending_entry_order_id = ""
        self.pending_entry_metadata.clear()
        self.session_bars.clear()
        self.last_decision_code = ""
        self.last_decision_details.clear()
        self.entry_attempted = False
        if self.position is None:
            self.stage = OLRSymbolStage.WATCHING if candidate is not None and candidate.tradable else OLRSymbolStage.BLOCKED


@dataclass(slots=True)
class OLRState:
    symbols: dict[str, OLRSymbolState] = field(default_factory=dict)
    snapshot_hash: str = ""
    source_fingerprint: str = ""
    session_date: date | None = None
    order_roles: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def symbol_state(self, symbol: str) -> OLRSymbolState:
        key = str(symbol).zfill(6)
        return self.symbols.setdefault(key, OLRSymbolState(symbol=key))
