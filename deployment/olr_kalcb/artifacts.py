from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from strategy_common.market import MarketBar
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailySnapshot
from strategy_kalcb.research import KALCB_CANONICAL_SOURCE
from strategy_kalcb.research_generator import generate_finalized_candidate_snapshot as generate_kalcb_finalized_candidate_snapshot
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE, OLRArtifactStore
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailySnapshot
from strategy_olr.research import load_candidate_snapshot
from strategy_olr.research_generator import generate_afternoon_candidate_snapshot, generate_candidate_snapshot as generate_olr_candidate_snapshot


@dataclass(frozen=True, slots=True)
class ArtifactGenerationResult:
    strategy_id: str
    trade_date: date
    stage: str
    artifact_hash: str
    source_fingerprint: str
    candidate_count: int
    path: Path | None
    elapsed_s: float


def generate_kalcb_daily(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    *,
    config: KALCBConfig | None = None,
    sector_map: dict[str, str] | None = None,
    daily_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    artifact_root: str | Path = "data/strategy/kalcb",
    source_fingerprint: str | None = None,
    candidate_config_hash: str | None = None,
    config_mutations: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> tuple[KALCBDailySnapshot, ArtifactGenerationResult]:
    started = monotonic()
    cfg = config or KALCBConfig()
    snapshot = generate_kalcb_finalized_candidate_snapshot(
        daily_by_symbol,
        trade_date,
        config=cfg,
        sector_map=sector_map,
        daily_flow_by_symbol=daily_flow_by_symbol,
        daily_foreign_flow_by_symbol=daily_foreign_flow_by_symbol,
        daily_institutional_flow_by_symbol=daily_institutional_flow_by_symbol,
        artifact_root=artifact_root,
        source_fingerprint=source_fingerprint,
        candidate_config_hash=candidate_config_hash,
        config_mutations=config_mutations,
        source=KALCB_CANONICAL_SOURCE,
        generated_at=generated_at,
    )
    path = KALCBArtifactStore(artifact_root).path_for(trade_date)
    return snapshot, _result("KALCB", snapshot, path, monotonic() - started)


def generate_olr_daily(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    *,
    config: OLRConfig | None = None,
    sector_map: dict[str, str] | None = None,
    flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    index_ohlcv_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    artifact_root: str | Path = "data/strategy/olr",
    source_fingerprint: str | None = None,
    generated_at: datetime | None = None,
) -> tuple[OLRDailySnapshot, ArtifactGenerationResult]:
    started = monotonic()
    snapshot = generate_olr_candidate_snapshot(
        daily_by_symbol,
        trade_date,
        config=config or OLRConfig(),
        sector_map=sector_map,
        flow_by_symbol=flow_by_symbol,
        foreign_flow_by_symbol=foreign_flow_by_symbol,
        institutional_flow_by_symbol=institutional_flow_by_symbol,
        index_ohlcv_by_symbol=index_ohlcv_by_symbol,
        artifact_root=artifact_root,
        source_fingerprint=source_fingerprint,
        generated_at=generated_at,
    )
    path = OLRArtifactStore(artifact_root).path_for(trade_date, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
    return snapshot, _result("OLR", snapshot, path, monotonic() - started)


def generate_olr_afternoon(
    trade_date: date,
    bars_by_symbol: dict[object, tuple[MarketBar, ...]] | dict[tuple[date, str], tuple[MarketBar, ...]],
    *,
    candidate_snapshot: OLRDailySnapshot | None = None,
    artifact_root: str | Path = "data/strategy/olr",
    config: OLRConfig | None = None,
    sector_map: dict[str, str] | None = None,
) -> tuple[OLRDailySnapshot, ArtifactGenerationResult]:
    started = monotonic()
    daily = candidate_snapshot or load_candidate_snapshot(
        trade_date,
        artifact_root=artifact_root,
        artifact_stage=OLR_STAGE1_ARTIFACT_STAGE,
    )
    if daily is None:
        raise FileNotFoundError(f"OLR stage1 artifact missing for {trade_date.isoformat()}")
    snapshot = generate_afternoon_candidate_snapshot(
        daily,
        bars_by_symbol,
        config=config or OLRConfig(),
        sector_map=sector_map,
        artifact_root=artifact_root,
    )
    path = OLRArtifactStore(artifact_root).path_for(trade_date, artifact_stage=OLR_FINAL_ARTIFACT_STAGE)
    return snapshot, _result("OLR", snapshot, path, monotonic() - started)


def _result(strategy_id: str, snapshot: Any, path: Path | None, elapsed_s: float) -> ArtifactGenerationResult:
    return ArtifactGenerationResult(
        strategy_id=strategy_id,
        trade_date=snapshot.trade_date,
        stage=str((snapshot.metadata or {}).get("artifact_stage") or ""),
        artifact_hash=snapshot.artifact_hash,
        source_fingerprint=snapshot.source_fingerprint,
        candidate_count=len(snapshot.candidates),
        path=path,
        elapsed_s=elapsed_s,
    )
