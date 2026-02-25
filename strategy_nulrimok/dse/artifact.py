"""Nulrimok Watchlist Artifact Schema."""

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class TickerArtifact:
    ticker: str
    regime_tier: str = "C"
    regime_score: float = 0.0
    risk_multiplier: float = 0.0
    sector: str = ""
    sector_rank_weight: float = 0.3
    flow_score: float = 0.0
    flow_persistence: float = 0.0
    flow_pass: bool = False
    rs_percentile: float = 0.0
    leader_pass: bool = False
    trend_pass: bool = False
    sma50: float = 0.0
    anchor_date: Optional[str] = None
    avwap_ref: float = 0.0
    band_lower: float = 0.0
    band_upper: float = 0.0
    acceptance_pass: bool = False
    avwap_proximity: float = 0.0
    daily_rank: float = 0.0
    tradable: bool = False
    recommended_risk: float = 0.005
    setup_type: str = ""
    atr30m_est: float = 0.0


@dataclass
class PositionArtifact:
    ticker: str
    entry_time: str
    avg_price: float
    qty: int
    stop: float = 0.0
    flow_reversal_flag: bool = False
    exit_at_open: bool = False


@dataclass
class WatchlistArtifact:
    date: str
    regime_tier: str = "C"
    regime_score: float = 0.0
    risk_mult: float = 1.0
    candidates: List[TickerArtifact] = field(default_factory=list)
    tradable: List[str] = field(default_factory=list)
    active_set: List[str] = field(default_factory=list)
    overflow: List[str] = field(default_factory=list)
    positions: List[PositionArtifact] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "date": self.date, "regime_tier": self.regime_tier, "regime_score": self.regime_score, "risk_mult": self.risk_mult,
            "candidates": [asdict(c) for c in self.candidates], "tradable": self.tradable,
            "active_set": self.active_set, "overflow": self.overflow,
            "positions": [asdict(p) for p in self.positions],
        }

    def get_ticker(self, ticker: str) -> Optional[TickerArtifact]:
        return next((c for c in self.candidates if c.ticker == ticker), None)

    @property
    def all_tickers(self) -> List[str]:
        """Return all ticker symbols from candidates."""
        return [c.ticker for c in self.candidates]
