"""PCIM Candidate dataclass."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Candidate:
    """Trading candidate from influencer signal."""

    # Source info
    influencer_id: str
    video_id: str
    symbol: str
    company_name: str
    conviction_score: float  # 0.0 to 1.0

    # Consolidation tracking
    influencer_count: int = 1

    # Filter status
    pass_trend_gate: bool = False
    reject_reason: Optional[str] = None

    # Daily stats
    adtv_20d: float = 0.0
    market_cap: float = 0.0
    atr_20d: float = 0.0
    sma20: float = 0.0
    close_prev: float = 0.0

    # Gap reversal
    gap_rev_rate: float = 0.0
    gap_rev_events: int = 0
    gap_rev_insufficient: bool = False

    # Premarket
    expected_open: Optional[float] = None
    gap_pct: Optional[float] = None
    bucket: Optional[str] = None

    # Tier
    tier: Optional[str] = None
    tier_mult: Optional[float] = None

    # Soft multipliers
    soft_mult: float = 1.0

    # Sizing
    raw_qty: Optional[int] = None
    final_qty: Optional[int] = None
    final_notional: Optional[float] = None

    # Priority (for selection when caps bind)
    priority_key: Optional[Tuple] = None

    def is_rejected(self) -> bool:
        return self.reject_reason is not None

    def compute_priority_key(self) -> Tuple:
        """Lower is better: (bucket_rank, -conviction_score, gap_pct, -adtv)"""
        bucket_rank = 0 if self.bucket == "A" else 1
        return (bucket_rank, -self.conviction_score, self.gap_pct if self.gap_pct is not None else 1.0, -self.adtv_20d)
