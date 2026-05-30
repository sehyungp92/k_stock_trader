from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from .artifact_store import KALCBArtifactStore, save_snapshot_to_lrs
from .config import KALCBConfig
from .models import KALCBDailySnapshot, KALCBResearchSnapshot
from .research import (
    KALCB_CANONICAL_SOURCE,
    build_research_snapshot,
    candidate_config_fingerprint,
    daily_selection_from_snapshot,
    finalize_candidate_snapshot,
    run_daily_selection,
)


def generate_research_snapshot(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    *,
    config: KALCBConfig | None = None,
    sector_map: dict[str, str] | None = None,
    daily_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    source_fingerprint: str | None = None,
    generated_at: datetime | None = None,
) -> KALCBResearchSnapshot:
    """Build today's research snapshot from already prepared daily bars.

    Live KIS fetching is intentionally outside v1; callers provide completed
    daily rows and receive the same pure snapshot used by backtests.
    """

    return build_research_snapshot(
        daily_by_symbol,
        trade_date,
        config or KALCBConfig(),
        sector_map=sector_map,
        daily_flow_by_symbol=daily_flow_by_symbol,
        daily_foreign_flow_by_symbol=daily_foreign_flow_by_symbol,
        daily_institutional_flow_by_symbol=daily_institutional_flow_by_symbol,
        source_fingerprint=source_fingerprint,
        generated_at=generated_at,
    )


def generate_candidate_snapshot(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    *,
    config: KALCBConfig | None = None,
    sector_map: dict[str, str] | None = None,
    daily_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    artifact_root: str | Path | None = None,
    source_fingerprint: str | None = None,
    generated_at: datetime | None = None,
    lrs=None,
) -> KALCBDailySnapshot:
    """Build the research-stage candidate snapshot.

    This helper is intentionally non-persistent by default. Use
    ``generate_finalized_candidate_snapshot()`` for live/paper executable
    artifacts.
    """

    return run_daily_selection(
        daily_by_symbol,
        trade_date,
        config=config or KALCBConfig(),
        sector_map=sector_map,
        daily_flow_by_symbol=daily_flow_by_symbol,
        daily_foreign_flow_by_symbol=daily_foreign_flow_by_symbol,
        daily_institutional_flow_by_symbol=daily_institutional_flow_by_symbol,
        artifact_root=artifact_root,
        source_fingerprint=source_fingerprint,
        generated_at=generated_at,
        lrs=lrs,
    )


def generate_finalized_candidate_snapshot(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    *,
    config: KALCBConfig | None = None,
    sector_map: dict[str, str] | None = None,
    daily_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    artifact_root: str | Path | None = "data/strategy/kalcb",
    source_fingerprint: str | None = None,
    candidate_config_hash: str | None = None,
    config_mutations: dict[str, Any] | None = None,
    source: str = KALCB_CANONICAL_SOURCE,
    generated_at: datetime | None = None,
    lrs=None,
) -> KALCBDailySnapshot:
    config_was_defaulted = config is None
    cfg = config or KALCBConfig()
    resolved_config_hash = _resolve_final_candidate_config_hash(
        cfg,
        candidate_config_hash=candidate_config_hash,
        config_mutations=config_mutations,
        sector_map=sector_map,
        allow_default_hash=config_was_defaulted,
    )
    base = generate_candidate_snapshot(
        daily_by_symbol,
        trade_date,
        config=cfg,
        sector_map=sector_map,
        daily_flow_by_symbol=daily_flow_by_symbol,
        daily_foreign_flow_by_symbol=daily_foreign_flow_by_symbol,
        daily_institutional_flow_by_symbol=daily_institutional_flow_by_symbol,
        artifact_root=None,
        source_fingerprint=source_fingerprint,
        generated_at=generated_at,
    )
    snapshot = finalize_candidate_snapshot(
        base,
        config=cfg,
        candidate_config_hash=resolved_config_hash,
        source=source,
        requested_universe_count=len(daily_by_symbol),
        data_available_symbols=sorted(str(symbol).zfill(6) for symbol in daily_by_symbol),
        unavailable_symbols=(),
        source_universe_count=len(daily_by_symbol),
        sector_map_size=len(sector_map or {}),
        generated_at=generated_at,
    )
    if artifact_root is not None:
        KALCBArtifactStore(artifact_root).save_snapshot(snapshot)
    if lrs is not None:
        save_snapshot_to_lrs(snapshot, lrs)
    return snapshot


def select_from_research_snapshot(
    snapshot: KALCBResearchSnapshot,
    *,
    config: KALCBConfig | None = None,
) -> KALCBDailySnapshot:
    return daily_selection_from_snapshot(snapshot, config or KALCBConfig())


def _resolve_final_candidate_config_hash(
    config: KALCBConfig,
    *,
    candidate_config_hash: str | None,
    config_mutations: dict[str, Any] | None,
    sector_map: dict[str, str] | None,
    allow_default_hash: bool,
) -> str:
    explicit = str(candidate_config_hash or "").strip()
    if explicit:
        if config_mutations is not None:
            expected = candidate_config_fingerprint(config, config_mutations, sector_map)
            if explicit != expected:
                raise ValueError("KALCB candidate_config_hash does not match config_mutations")
        return explicit
    if config_mutations is not None:
        return candidate_config_fingerprint(config, config_mutations, sector_map)
    if allow_default_hash:
        return candidate_config_fingerprint(config, {}, sector_map)
    raise ValueError(
        "KALCB finalized artifact generation requires candidate_config_hash or config_mutations "
        "when an explicit config is supplied"
    )
