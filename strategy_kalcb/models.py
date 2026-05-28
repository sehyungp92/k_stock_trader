from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
from typing import Any


@dataclass(frozen=True, slots=True)
class KALCBMarketResearch:
    trade_date: date
    breadth_pct_above_20dma: float
    avg_20d_return_pct: float
    regime_tier: str
    regime: str


@dataclass(frozen=True, slots=True)
class KALCBSectorResearch:
    sector: str
    symbol_count: int
    return_20d_pct: float
    breadth_20d: float
    participation: float
    regime: str


@dataclass(frozen=True, slots=True)
class KALCBResearchSymbol:
    symbol: str
    trade_date: date
    daily_rows: tuple[dict[str, Any], ...]
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
    daily_flow_rows: tuple[dict[str, Any], ...] = ()
    daily_foreign_flow_rows: tuple[dict[str, Any], ...] = ()
    daily_institutional_flow_rows: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class KALCBResearchSnapshot:
    trade_date: date
    market: KALCBMarketResearch
    sectors: dict[str, KALCBSectorResearch]
    symbols: dict[str, KALCBResearchSymbol]
    source_fingerprint: str
    generated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class EntryType(str, Enum):
    FIRST30_OPEN = "KRX_FIRST30_OPEN"
    OPENING_DRIVE = "KRX_OPENING_DRIVE"
    POST_OR_MOMENTUM = "KRX_POST_OR_MOMENTUM"
    OR_BREAKOUT = "KRX_OR_BREAKOUT"
    PDH_BREAKOUT = "KRX_PDH_BREAKOUT"
    COMBINED_BREAKOUT = "KRX_COMBINED_BREAKOUT"
    AVWAP_RECLAIM = "KRX_AVWAP_RECLAIM"
    PULLBACK_ACCEPTANCE = "KRX_PULLBACK_ACCEPTANCE"
    OR_MID_RECLAIM = "KRX_OR_MID_RECLAIM"
    OR_HIGH_RECLAIM = "KRX_OR_HIGH_RECLAIM"
    PDH_RECLAIM = "KRX_PDH_RECLAIM"
    DEFERRED_CONTINUATION = "KRX_DEFERRED_CONTINUATION"


@dataclass(frozen=True, slots=True)
class KALCBDailyCandidate:
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
    rs_percentile: float = 0.0
    accumulation_score: float = 0.0
    flow_score: float = 0.0
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
    def from_json_dict(cls, payload: dict[str, Any]) -> "KALCBDailyCandidate":
        data = dict(payload)
        data["trade_date"] = date.fromisoformat(str(data["trade_date"]))
        data["reject_reasons"] = tuple(data.get("reject_reasons", ()) or ())
        return cls(**data)


@dataclass(frozen=True, slots=True)
class KALCBDailySnapshot:
    trade_date: date
    candidates: tuple[KALCBDailyCandidate, ...]
    source_fingerprint: str
    generated_at: datetime
    strategy_id: str = "KALCB"
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_symbol(self) -> dict[str, KALCBDailyCandidate]:
        return {candidate.symbol: candidate for candidate in self.candidates}

    @property
    def artifact_hash(self) -> str:
        return snapshot_hash(self)

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
    def from_json_dict(cls, payload: dict[str, Any]) -> "KALCBDailySnapshot":
        return cls(
            strategy_id=str(payload.get("strategy_id", "KALCB")),
            trade_date=date.fromisoformat(str(payload["trade_date"])),
            generated_at=datetime.fromisoformat(str(payload["generated_at"])),
            source_fingerprint=str(payload.get("source_fingerprint", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
            candidates=tuple(KALCBDailyCandidate.from_json_dict(row) for row in payload.get("candidates", [])),
        )


def snapshot_hash(snapshot: KALCBDailySnapshot) -> str:
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


_HASH_BOUND_METADATA_KEYS = (
    "artifact_stage",
    "candidate_config_hash",
    "source",
    "sector_map_hash",
    "sector_map_size",
    "requested_universe_count",
    "data_available_symbol_count",
    "unavailable_symbol_count",
    "source_universe_count",
    "candidate_pool_count",
    "frontier_enabled",
    "frontier_size",
    "frontier_selection_mode",
    "frontier_active_selection_mode",
    "active_symbol_count",
    "frontier_symbol_count",
    "overflow_symbol_count",
)


def _hash_bound_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not str(metadata.get("candidate_config_hash") or "").strip():
        return {}
    return {
        key: metadata[key]
        for key in _HASH_BOUND_METADATA_KEYS
        if key in metadata and metadata[key] not in (None, "")
    }
