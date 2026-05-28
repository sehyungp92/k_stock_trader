from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from hashlib import sha256
from typing import Any


@dataclass(frozen=True, slots=True)
class OLRMarketResearch:
    trade_date: date
    breadth_pct_above_20dma: float
    avg_20d_return_pct: float
    market_heat_score: float
    regime_tier: str
    regime: str = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class OLRSectorResearch:
    sector: str
    symbol_count: int
    return_20d_pct: float
    breadth_20d: float
    participation: float
    flow_5d: float
    regime: str = "UNKNOWN"
    foreign_flow_5d: float = 0.0
    institutional_flow_5d: float = 0.0
    flow_agreement_5d: float = 0.0


@dataclass(frozen=True, slots=True)
class OLRResearchSymbol:
    symbol: str
    trade_date: date
    daily_rows: tuple[dict[str, Any], ...]
    flow_rows: tuple[dict[str, Any], ...]
    foreign_flow_rows: tuple[dict[str, Any], ...]
    institutional_flow_rows: tuple[dict[str, Any], ...]
    sector: str
    price: float
    adv20_krw: float
    prior_day_high: float
    prior_day_low: float
    prior_day_close: float
    daily_atr: float
    expected_5m_volume: float
    average_30m_volume: float
    return_5d_pct: float = 0.0
    return_20d_pct: float = 0.0
    return_60d_pct: float = 0.0
    volume_ratio_20d: float = 0.0
    close_location_20d: float = 0.0
    median_spread_pct: float | None = None
    flow_available: bool = False
    flow_1d: float = 0.0
    flow_3d: float = 0.0
    flow_5d: float = 0.0
    flow_20d: float = 0.0
    foreign_flow_1d: float = 0.0
    foreign_flow_3d: float = 0.0
    foreign_flow_5d: float = 0.0
    foreign_flow_20d: float = 0.0
    institutional_flow_1d: float = 0.0
    institutional_flow_3d: float = 0.0
    institutional_flow_5d: float = 0.0
    institutional_flow_20d: float = 0.0
    flow_z: float = 0.0
    foreign_flow_z: float = 0.0
    institutional_flow_z: float = 0.0
    flow_positive_days_5d: float = 0.0
    foreign_positive_days_5d: float = 0.0
    institutional_positive_days_5d: float = 0.0
    flow_acceleration: float = 0.0
    foreign_flow_acceleration: float = 0.0
    institutional_flow_acceleration: float = 0.0
    flow_agreement_5d: float = 0.0
    flow_divergence_5d: float = 0.0
    combined_flow_notional_5d: float = 0.0
    sponsorship_balance_5d: float = 0.0
    etf_flag: bool = False
    preferred_flag: bool = False
    otc_flag: bool = False
    hard_to_borrow_flag: bool = False
    blacklist_flag: bool = False
    halted_flag: bool = False
    severe_news_flag: bool = False


@dataclass(frozen=True, slots=True)
class OLRResearchSnapshot:
    trade_date: date
    market: OLRMarketResearch
    sectors: dict[str, OLRSectorResearch]
    symbols: dict[str, OLRResearchSymbol]
    source_fingerprint: str
    generated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OLRDailyCandidate:
    symbol: str
    trade_date: date
    prior_day_high: float
    prior_day_low: float
    prior_day_close: float
    daily_atr: float
    expected_5m_volume: float
    average_30m_volume: float
    sector: str = "UNKNOWN"
    regime_tier: str = "A"
    selection_score: float = 0.0
    daily_signal_score: float = 0.0
    rank: int = 0
    rank_pct: float = 0.0
    rs_percentile: float = 0.0
    accumulation_score: float = 0.0
    flow_score: float = 0.0
    foreign_flow_5d: float = 0.0
    institutional_flow_5d: float = 0.0
    flow_agreement_5d: float = 0.0
    tradable: bool = True
    reject_reasons: tuple[str, ...] = ()
    source_fingerprint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trade_date"] = self.trade_date.isoformat()
        payload["reject_reasons"] = list(self.reject_reasons)
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "OLRDailyCandidate":
        data = dict(payload)
        data["trade_date"] = date.fromisoformat(str(data["trade_date"]))
        data["reject_reasons"] = tuple(data.get("reject_reasons", ()) or ())
        return cls(**data)


@dataclass(frozen=True, slots=True)
class OLRDailySnapshot:
    trade_date: date
    candidates: tuple[OLRDailyCandidate, ...]
    source_fingerprint: str
    generated_at: datetime
    strategy_id: str = "OLR"
    metadata: dict[str, Any] = field(default_factory=dict)
    _artifact_hash_cache: str = field(default="", init=False, repr=False, compare=False)

    def by_symbol(self) -> dict[str, OLRDailyCandidate]:
        return {candidate.symbol: candidate for candidate in self.candidates}

    @property
    def artifact_hash(self) -> str:
        if self._artifact_hash_cache:
            return self._artifact_hash_cache
        value = snapshot_hash(self)
        object.__setattr__(self, "_artifact_hash_cache", value)
        return value

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "trade_date": self.trade_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "source_fingerprint": self.source_fingerprint,
            "artifact_hash": self.artifact_hash,
            "metadata": dict(self.metadata),
            "candidates": [candidate.to_json_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "OLRDailySnapshot":
        return cls(
            strategy_id=str(payload.get("strategy_id", "OLR")),
            trade_date=date.fromisoformat(str(payload["trade_date"])),
            generated_at=datetime.fromisoformat(str(payload["generated_at"])),
            source_fingerprint=str(payload.get("source_fingerprint", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
            candidates=tuple(OLRDailyCandidate.from_json_dict(row) for row in payload.get("candidates", [])),
        )


@dataclass(frozen=True, slots=True)
class OLRAfternoonContext:
    trade_date: date
    symbol: str
    candidate: OLRDailyCandidate
    afternoon_ret: float
    vwap_ret: float
    gap: float
    rel_volume: float
    close_location: float
    open_drawdown: float
    high_from_open: float
    low_vs_prev_close: float
    range_atr: float
    last_close: float
    bar_count: int
    prior_return_5d: float = 0.0
    prior_return_20d: float = 0.0
    prior_return_60d: float = 0.0
    lagged_flow_5d: float = 0.0
    lagged_foreign_flow_5d: float = 0.0
    lagged_institutional_flow_5d: float = 0.0
    lagged_flow_z: float = 0.0
    lagged_foreign_z: float = 0.0
    lagged_institutional_z: float = 0.0
    lagged_flow_agreement_5d: float = 0.0
    lagged_flow_divergence_5d: float = 0.0
    lagged_sector_flow_5d: float = 0.0
    lagged_sector_foreign_flow_5d: float = 0.0
    lagged_sector_institutional_flow_5d: float = 0.0
    intraday_sector_score_pct: float = 50.0
    intraday_sector_ret: float = 0.0
    intraday_sector_breadth: float = 0.5
    intraday_sector_rel_volume: float = 1.0
    intraday_sector_participation: float = 0.0
    intraday_sector_effective_count: int = 0
    market_score: float = 0.0


def snapshot_hash(snapshot: OLRDailySnapshot) -> str:
    payload = {
        "strategy_id": snapshot.strategy_id,
        "trade_date": snapshot.trade_date.isoformat(),
        "source_fingerprint": snapshot.source_fingerprint,
        "candidates": [candidate.to_json_dict() for candidate in snapshot.candidates],
    }
    hash_bound_metadata = _hash_bound_metadata(snapshot.metadata)
    if hash_bound_metadata:
        payload["hash_bound_metadata"] = hash_bound_metadata
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _hash_bound_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    final_config_hash = str(metadata.get("final_candidate_config_hash") or "").strip()
    if not final_config_hash:
        return {}
    payload = {
        "final_candidate_config_hash": final_config_hash,
        "final_candidate_config_hash_version": str(metadata.get("final_candidate_config_hash_version") or "").strip(),
    }
    candidate_config_hash = str(metadata.get("candidate_config_hash") or "").strip()
    if candidate_config_hash:
        payload["candidate_config_hash"] = candidate_config_hash
    return payload
