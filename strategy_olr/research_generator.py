from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from strategy_common.market import MarketBar

from .config import OLRConfig
from .models import OLRDailySnapshot, OLRResearchSnapshot
from .research import build_research_snapshot, daily_selection_from_snapshot, run_afternoon_selection, run_daily_selection


def generate_research_snapshot(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    *,
    config: OLRConfig | None = None,
    sector_map: dict[str, str] | None = None,
    flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    index_ohlcv_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    source_fingerprint: str | None = None,
    generated_at: datetime | None = None,
) -> OLRResearchSnapshot:
    """Build today's causal OLR research snapshot from prepared completed daily rows."""

    return build_research_snapshot(
        daily_by_symbol,
        trade_date,
        config or OLRConfig(),
        sector_map=sector_map,
        flow_by_symbol=flow_by_symbol,
        foreign_flow_by_symbol=foreign_flow_by_symbol,
        institutional_flow_by_symbol=institutional_flow_by_symbol,
        index_ohlcv_by_symbol=index_ohlcv_by_symbol,
        source_fingerprint=source_fingerprint,
        generated_at=generated_at,
    )


def generate_candidate_snapshot(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    *,
    config: OLRConfig | None = None,
    sector_map: dict[str, str] | None = None,
    flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    index_ohlcv_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    artifact_root: str | Path | None = "data/strategy/olr",
    source_fingerprint: str | None = None,
    generated_at: datetime | None = None,
    lrs=None,
) -> OLRDailySnapshot:
    cfg = config or OLRConfig()
    research_snapshot = generate_research_snapshot(
        daily_by_symbol,
        trade_date,
        config=cfg,
        sector_map=sector_map,
        flow_by_symbol=flow_by_symbol,
        foreign_flow_by_symbol=foreign_flow_by_symbol,
        institutional_flow_by_symbol=institutional_flow_by_symbol,
        index_ohlcv_by_symbol=index_ohlcv_by_symbol,
        source_fingerprint=source_fingerprint,
        generated_at=generated_at,
    )
    return run_daily_selection(
        research_snapshot,
        config=cfg,
        artifact_root=artifact_root,
        lrs=lrs,
    )


def generate_afternoon_candidate_snapshot(
    candidate_snapshot: OLRDailySnapshot,
    bars_by_symbol: dict[object, tuple[MarketBar, ...]] | dict[tuple[date, str], tuple[MarketBar, ...]],
    *,
    config: OLRConfig | None = None,
    sector_map: dict[str, str] | None = None,
    artifact_root: str | Path | None = "data/strategy/olr",
    lrs=None,
) -> OLRDailySnapshot:
    """Build the live 14:30 artifact using the same selector replay uses."""

    return run_afternoon_selection(
        candidate_snapshot,
        bars_by_symbol,
        config=config or OLRConfig(),
        sector_map=sector_map,
        artifact_root=artifact_root,
        lrs=lrs,
    )


def select_from_research_snapshot(
    snapshot: OLRResearchSnapshot,
    *,
    config: OLRConfig | None = None,
) -> OLRDailySnapshot:
    return daily_selection_from_snapshot(snapshot, config or OLRConfig())
