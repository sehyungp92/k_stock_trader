from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from strategy_common.market import MarketBar
from strategy_kalcb.models import KALCBDailyCandidate


class SymbolStage(str, Enum):
    WATCHING = "WATCHING"
    ENTRY_QUEUED = "ENTRY_QUEUED"
    IN_POSITION = "IN_POSITION"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


@dataclass(slots=True)
class KALCBBarState:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_bar(cls, bar: MarketBar) -> "KALCBBarState":
        return cls(bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume)

    def to_market_bar(self, symbol: str, timeframe: str = "5m") -> MarketBar:
        return MarketBar(
            symbol=symbol,
            timestamp=self.timestamp,
            timeframe=timeframe,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            is_completed=True,
            source="kalcb_state",
        )


@dataclass(slots=True)
class KALCBPositionState:
    symbol: str
    qty_entry: int
    qty_open: int
    entry_price: float
    entry_time: datetime
    initial_stop: float
    current_stop: float
    risk_per_share: float
    entry_type: str
    momentum_score: int
    sector: str = "UNKNOWN"
    regime_tier: str = "A"
    entry_order_id: str = ""
    stop_order_id: str = ""
    partial_order_id: str = ""
    max_favorable_price: float = 0.0
    max_adverse_price: float = 0.0
    hold_bars: int = 0
    partial_taken: bool = False
    exit_in_flight: bool = False
    stop_tightened: bool = False
    vwap_fail_streak: int = 0
    last_exit_reason: str = ""
    avwap_at_entry: float = 0.0
    or_high: float = 0.0
    or_low: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_favorable_price <= 0:
            self.max_favorable_price = self.entry_price
        if self.max_adverse_price <= 0:
            self.max_adverse_price = self.entry_price

    def update_mark(self, *, high: float, low: float) -> None:
        self.max_favorable_price = max(self.max_favorable_price, float(high))
        self.max_adverse_price = min(self.max_adverse_price, float(low))

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return (float(price) - self.entry_price) / self.risk_per_share

    def mfe_r(self) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return (self.max_favorable_price - self.entry_price) / self.risk_per_share

    def mae_r(self) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return (self.max_adverse_price - self.entry_price) / self.risk_per_share


@dataclass(slots=True)
class KALCBSymbolState:
    symbol: str
    stage: SymbolStage = SymbolStage.WATCHING
    candidate: KALCBDailyCandidate | None = None
    candidate_rank: int = 0
    session_date: date | None = None
    bars: list[KALCBBarState] = field(default_factory=list)
    _market_bars: list[MarketBar] = field(default_factory=list, init=False, repr=False)
    opening_range_built: bool = False
    or_high: float = 0.0
    or_low: float = 0.0
    or_volume: float = 0.0
    vwap_value: float = 0.0
    vwap_volume: float = 0.0
    pending_entry_order_id: str = ""
    pending_entry_metadata: dict[str, Any] = field(default_factory=dict)
    touched_vwap: bool = False
    touched_or_mid: bool = False
    touched_or_high: bool = False
    touched_pdh: bool = False
    touched_reclaim_levels: dict[str, bool] = field(default_factory=dict)
    position: KALCBPositionState | None = None
    rejected_reason: str = ""
    entry_attempted: bool = False
    last_decision_code: str = ""
    last_decision_details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._market_bars = [item.to_market_bar(self.symbol) for item in self.bars]

    def reset_for_session(self, session_date: date, candidate: KALCBDailyCandidate | None, candidate_rank: int = 0) -> None:
        self.session_date = session_date
        self.candidate = candidate
        self.candidate_rank = int(candidate_rank or 0)
        self.bars.clear()
        self._market_bars.clear()
        self.opening_range_built = False
        self.or_high = 0.0
        self.or_low = 0.0
        self.or_volume = 0.0
        self.vwap_value = 0.0
        self.vwap_volume = 0.0
        self.pending_entry_order_id = ""
        self.pending_entry_metadata.clear()
        self.touched_vwap = False
        self.touched_or_mid = False
        self.touched_or_high = False
        self.touched_pdh = False
        self.touched_reclaim_levels.clear()
        self.rejected_reason = ""
        self.entry_attempted = False
        self.last_decision_code = ""
        self.last_decision_details.clear()
        if self.position is None:
            self.stage = SymbolStage.WATCHING if candidate is not None and candidate.tradable else SymbolStage.BLOCKED

    def add_bar(self, bar: MarketBar) -> None:
        self.bars.append(KALCBBarState.from_bar(bar))
        self._market_bars.append(bar)
        typical = (bar.high + bar.low + bar.close) / 3.0
        self.vwap_value += typical * bar.volume
        self.vwap_volume += bar.volume

    @property
    def bars_today(self) -> list[MarketBar]:
        return self._market_bars

    @property
    def vwap(self) -> float:
        if self.vwap_volume <= 0:
            return self.bars[-1].close if self.bars else 0.0
        return self.vwap_value / self.vwap_volume


@dataclass(slots=True)
class KALCBState:
    symbols: dict[str, KALCBSymbolState] = field(default_factory=dict)
    snapshot_hash: str = ""
    source_fingerprint: str = ""
    session_date: date | None = None
    order_roles: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def symbol_state(self, symbol: str) -> KALCBSymbolState:
        key = str(symbol)
        return self.symbols.setdefault(key, KALCBSymbolState(symbol=key))
