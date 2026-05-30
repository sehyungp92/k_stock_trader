from __future__ import annotations

import json
import bisect
from dataclasses import replace
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any

from strategy_common.clock import KST
from strategy_common.sector_daily import SECTOR_DAILY_VERSION, SectorDailyFeature, SectorDailyMember, SectorDailyPanel, score_sector_daily_members

from .config import KALCBConfig
from .models import (
    KALCBDailyCandidate,
    KALCBDailySnapshot,
    KALCBMarketResearch,
    KALCBResearchSnapshot,
    KALCBResearchSymbol,
    KALCBSectorResearch,
)
from .signals import atr_from_daily_rows
from .structural_campaign import (
    build_structural_campaign,
    campaign_metadata,
    compute_rs_percentiles,
)


RESEARCH_MODEL_VERSION = "kalcb-research-selection-v3-structural-campaign"
KALCB_FINAL_ARTIFACT_STAGE = "daily_finalized_candidate"
KALCB_CANONICAL_SOURCE = "real_kis_krx_parquet"
_MAX_RESEARCH_DAILY_HISTORY = 320
_MAX_RESEARCH_FLOW_HISTORY = 80
_PRIOR_ROW_INDEX_CACHE: dict[tuple[Any, ...], tuple[tuple[date, ...], tuple[dict[str, Any], ...]]] = {}
_SECTOR_DAILY_PANEL_CACHE: dict[tuple[date, tuple[tuple[str, int, str], ...]], SectorDailyPanel] = {}
_CANDIDATE_CONFIG_MUTATION_KEYS = {
    "kalcb.session.opening_range_bars",
    "kalcb.session.ws_budget",
    "kalcb.live.ws_budget",
    "kalcb.frontier.enabled",
    "kalcb.frontier.size",
    "kalcb.frontier.selection_mode",
    "kalcb.frontier.active_selection_mode",
    "kalcb.frontier.rotation_min_frontier_trades",
    "kalcb.frontier.rotation_min_frontier_avg_r",
    "kalcb.frontier.rotation_min_frontier_total_r",
    "kalcb.frontier.rotation_min_proof_symbols",
    "kalcb.discovery.frontier_size",
    "kalcb.discovery.selection_mode",
    "kalcb.discovery.active_selection_mode",
    "kalcb.research.top_long_count",
    "kalcb.research.min_price_krw",
    "kalcb.research.min_adv20_krw",
    "kalcb.research.min_history_days",
    "kalcb.research.weights.relative_strength",
    "kalcb.research.weights.daily_trend",
    "kalcb.research.weights.compression",
    "kalcb.research.weights.accumulation",
    "kalcb.research.weights.stock_regime",
    "kalcb.research.weights.sector_regime",
    "kalcb.research.weights.sector_participation",
    "kalcb.research.min_rs_percentile",
    "kalcb.research.min_trend_score",
    "kalcb.research.min_compression_score",
    "kalcb.research.min_accumulation_score",
    "kalcb.research.min_sector_participation",
    "kalcb.research.min_sector_daily_score_pct",
    "kalcb.research.max_box_range_pct",
    "kalcb.research.structural_frontier_count",
    "kalcb.research.min_structural_campaign_score",
}


def build_research_snapshot(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    config: KALCBConfig | None = None,
    *,
    sector_map: dict[str, str] | None = None,
    daily_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
    source_fingerprint: str | None = None,
    compute_source_fingerprint: bool = True,
    metadata: dict[str, Any] | None = None,
) -> KALCBResearchSnapshot:
    """Build a causal daily research snapshot from completed rows only."""

    cfg = config or KALCBConfig()
    sectors = {str(symbol).zfill(6): _normalize_sector(sector) for symbol, sector in dict(sector_map or {}).items()}
    flow_rows_by_symbol = {str(symbol).zfill(6): rows for symbol, rows in dict(daily_flow_by_symbol or {}).items()}
    foreign_flow_rows_by_symbol = {str(symbol).zfill(6): rows for symbol, rows in dict(daily_foreign_flow_by_symbol or {}).items()}
    institutional_flow_rows_by_symbol = {str(symbol).zfill(6): rows for symbol, rows in dict(daily_institutional_flow_by_symbol or {}).items()}
    symbols: dict[str, KALCBResearchSymbol] = {}
    for raw_symbol, raw_rows in daily_by_symbol.items():
        symbol = str(raw_symbol).zfill(6)
        prior_rows = _recent_prior_rows(raw_rows, trade_date, _MAX_RESEARCH_DAILY_HISTORY)
        if not prior_rows:
            continue
        prior = prior_rows[-1]
        prior_close = _float(prior.get("close"))
        prior_volume = _float(prior.get("volume"))
        if prior_close <= 0 or prior_volume <= 0:
            continue
        adv20 = _adv_krw(prior_rows, 20)
        symbols[symbol] = KALCBResearchSymbol(
            symbol=symbol,
            trade_date=trade_date,
            daily_rows=tuple(dict(row) for row in prior_rows),
            sector=sectors.get(symbol, "UNKNOWN"),
            price=prior_close,
            adv20_krw=adv20,
            prior_day_high=_float(prior.get("high"), prior_close),
            prior_day_low=_float(prior.get("low"), prior_close),
            prior_day_close=prior_close,
            daily_atr=max(atr_from_daily_rows(prior_rows, 14), prior_close * 0.01),
            expected_5m_volume=max(prior_volume / 78.0, 1.0),
            average_30m_volume=max(prior_volume / 13.0, 1.0),
            return_5d_pct=_return_pct(prior_rows, 5),
            return_20d_pct=_return_pct(prior_rows, 20),
            return_60d_pct=_return_pct(prior_rows, 60),
            volume_ratio_20d=_volume_ratio(prior_rows, 20),
            close_location_20d=_close_location(prior_rows[-20:]),
            daily_flow_rows=tuple(_recent_prior_rows(flow_rows_by_symbol.get(symbol, []), trade_date, _MAX_RESEARCH_FLOW_HISTORY)),
            daily_foreign_flow_rows=tuple(_recent_prior_rows(foreign_flow_rows_by_symbol.get(symbol, []), trade_date, _MAX_RESEARCH_FLOW_HISTORY)),
            daily_institutional_flow_rows=tuple(_recent_prior_rows(institutional_flow_rows_by_symbol.get(symbol, []), trade_date, _MAX_RESEARCH_FLOW_HISTORY)),
        )
    if compute_source_fingerprint or not source_fingerprint:
        causal_source_fingerprint = _source_fingerprint(
            daily_by_symbol,
            trade_date,
            daily_flow_by_symbol=flow_rows_by_symbol,
            daily_foreign_flow_by_symbol=foreign_flow_rows_by_symbol,
            daily_institutional_flow_by_symbol=institutional_flow_rows_by_symbol,
        )
        causal_source_fingerprint_policy = "strict_prior_daily_rows_hash"
    else:
        causal_source_fingerprint = str(source_fingerprint)
        causal_source_fingerprint_policy = "provided_prior_daily_dataset_hash"
    sector_daily_panel = _sector_daily_panel_from_symbols(symbols, trade_date)
    sector_research = _build_sector_research(symbols, sector_daily_panel)
    market = _build_market_research(trade_date, symbols)
    return KALCBResearchSnapshot(
        trade_date=trade_date,
        market=market,
        sectors=sector_research,
        symbols=symbols,
        source_fingerprint=source_fingerprint or causal_source_fingerprint,
        generated_at=generated_at or datetime.now(tz=KST),
        metadata={
            "research_model_version": RESEARCH_MODEL_VERSION,
            "requested_symbol_count": len(daily_by_symbol),
            "sector_map_size": len(sectors),
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "research_config_hash": _research_config_fingerprint(cfg),
            "sector_map_hash": _mapping_fingerprint(sectors),
            "research_lookahead_policy": "strict_prior_daily_rows_only",
            "research_trade_date": trade_date.isoformat(),
            "research_as_of_date": _research_as_of_date(symbols),
            "research_causal_source_fingerprint": causal_source_fingerprint,
            "research_causal_source_fingerprint_policy": causal_source_fingerprint_policy,
            **dict(metadata or {}),
        },
    )


def daily_selection_from_snapshot(
    snapshot: KALCBResearchSnapshot,
    config: KALCBConfig | None = None,
) -> KALCBDailySnapshot:
    """Select top long KALCB candidates from a causal research snapshot."""

    cfg = config or KALCBConfig()
    rs_percentiles = _relative_strength_percentiles(snapshot.symbols)
    sector_daily_panel = _sector_daily_panel_from_symbols(snapshot.symbols, snapshot.trade_date)
    scored: list[tuple[float, KALCBDailyCandidate]] = []
    rejected: dict[str, list[str]] = {}
    for symbol, research_symbol in snapshot.symbols.items():
        reasons = _reject_reasons(research_symbol, cfg)
        if reasons:
            rejected[symbol] = reasons
            continue
        sector_daily_feature = sector_daily_panel.feature_for(snapshot.trade_date, symbol, sector=research_symbol.sector)
        score, details = _long_score(
            research_symbol,
            snapshot,
            rs_percentiles.get(symbol, 0.0),
            cfg,
            sector_daily_feature=sector_daily_feature,
        )
        score_reasons = _score_reject_reasons(details, cfg)
        if score_reasons:
            rejected[symbol] = score_reasons
            continue
        flow_score = _sector_flow_score(sector_daily_feature)
        structural_metadata = dict(details.get("structural_campaign_metadata") or {})
        candidate = KALCBDailyCandidate(
            symbol=symbol,
            trade_date=snapshot.trade_date,
            prior_day_high=research_symbol.prior_day_high,
            prior_day_low=research_symbol.prior_day_low,
            prior_day_close=research_symbol.prior_day_close,
            daily_atr=research_symbol.daily_atr,
            expected_5m_volume=research_symbol.expected_5m_volume,
            average_30m_volume=research_symbol.average_30m_volume,
            sector=research_symbol.sector,
            regime_tier=snapshot.market.regime_tier,
            selection_score=score,
            rs_percentile=details["relative_strength_pct"],
            accumulation_score=details["accumulation_score"],
            flow_score=flow_score,
            tradable=True,
            source_fingerprint=snapshot.source_fingerprint,
            metadata={
                **_snapshot_candidate_metadata(snapshot),
                "research_model_version": RESEARCH_MODEL_VERSION,
                "source": "kalcb_research_selection",
                "prior_day_date": _row_date_label(research_symbol.daily_rows[-1]),
                "prior_day_notional": research_symbol.price * _float(research_symbol.daily_rows[-1].get("volume")),
                "frontier_selection_score": score,
                "frontier_score_components": details,
                "research_score_components": details,
                "market_regime": snapshot.market.regime,
                "market_breadth_pct": snapshot.market.breadth_pct_above_20dma,
                "legacy_sector_regime": _legacy_sector_regime(sector_daily_feature),
                "sector_participation": sector_daily_feature.participation,
                **sector_daily_feature.metadata(),
                **structural_metadata,
                "regime_source": "ohlcv_research",
                "optional_features": {
                    "foreign_institutional_flow": bool(research_symbol.daily_flow_rows or research_symbol.daily_foreign_flow_rows or research_symbol.daily_institutional_flow_rows),
                    "program_flow": False,
                    "bid_ask_spread_snapshots": False,
                },
            },
        )
        scored.append((score, candidate))

    scored.sort(
        key=lambda item: (
            -item[0],
            -_float(item[1].metadata.get("first30_confirmation_score")),
            -item[1].rs_percentile,
            -_float(item[1].metadata.get("stock_vs_universe_strength")),
            -_float(item[1].metadata.get("sector_leadership_pct")),
            -_float(item[1].metadata.get("prior_day_notional")),
            item[1].symbol,
        )
    )
    frontier_cap = int(cfg.research_structural_frontier_count or max(cfg.research_top_long_count, cfg.frontier_size, cfg.ws_budget))
    frontier_cap = max(cfg.research_top_long_count, frontier_cap)
    active_count = min(max(1, int(cfg.ws_budget)), len(scored))
    selected_rows = scored[:frontier_cap]
    selected_candidates: list[KALCBDailyCandidate] = []
    for rank, (_score, candidate) in enumerate(selected_rows, start=1):
        role = "active" if rank <= active_count else "overflow"
        frontier_role = "initial_active" if role == "active" else "frontier_shadow"
        selected_candidates.append(
            replace(
                candidate,
                metadata={
                    **dict(candidate.metadata),
                    "candidate_rank": rank,
                    "frontier_rank": rank,
                    "structural_source_rank": rank,
                    "structural_source_role": role,
                    "frontier_role": frontier_role,
                    "frontier_initial_active": role == "active",
                    "frontier_selection_mode": "structural_campaign",
                },
            )
        )
    selected = tuple(selected_candidates)
    active_symbols = [candidate.symbol for candidate in selected[:active_count]]
    frontier_symbols = [candidate.symbol for candidate in selected]
    all_pool_symbols = [candidate.symbol for _, candidate in scored]
    overflow_symbols = [symbol for symbol in all_pool_symbols if symbol not in set(active_symbols)]
    return KALCBDailySnapshot(
        trade_date=snapshot.trade_date,
        candidates=selected,
        source_fingerprint=snapshot.source_fingerprint,
        generated_at=snapshot.generated_at,
        metadata={
            "research_model_version": RESEARCH_MODEL_VERSION,
            "source": "kalcb_research_selection",
            "top_long_count": cfg.research_top_long_count,
            "structural_frontier_count": frontier_cap,
            "candidate_pool_count": len(scored),
            "candidate_pool_symbols": all_pool_symbols,
            "item_symbols": all_pool_symbols,
            "active_symbols": active_symbols,
            "active_symbol_count": len(active_symbols),
            "frontier_symbols": frontier_symbols,
            "frontier_symbol_count": len(frontier_symbols),
            "overflow_symbols": overflow_symbols,
            "overflow_symbol_count": len(overflow_symbols),
            "active_budget_source": "ws_budget",
            "selected_symbols": [candidate.symbol for candidate in selected],
            "rejected_symbol_count": len(rejected),
            "rejected_symbols": rejected,
            "market": {
                "breadth_pct_above_20dma": snapshot.market.breadth_pct_above_20dma,
                "avg_20d_return_pct": snapshot.market.avg_20d_return_pct,
                "regime_tier": snapshot.market.regime_tier,
                "regime": snapshot.market.regime,
            },
            **dict(snapshot.metadata),
        },
    )


def candidate_config_fingerprint(
    config: KALCBConfig,
    mutations: dict[str, Any] | None = None,
    sector_map: dict[str, str] | None = None,
) -> str:
    """Hash the KALCB executable candidate configuration used by replay/live."""

    raw_mutations = dict(mutations or {})
    return _mapping_fingerprint(
        {
            "opening_range_bars": config.opening_range_bars,
            "ws_budget": config.ws_budget,
            "frontier_enabled": config.frontier_enabled,
            "frontier_size": config.frontier_size,
            "frontier_selection_mode": config.frontier_selection_mode,
            "frontier_active_selection_mode": config.frontier_active_selection_mode,
            "research_model_version": RESEARCH_MODEL_VERSION,
            "research_top_long_count": config.research_top_long_count,
            "research_min_price_krw": config.research_min_price_krw,
            "research_min_adv20_krw": config.research_min_adv20_krw,
            "research_min_history_days": config.research_min_history_days,
            "research_weight_relative_strength": config.research_weight_relative_strength,
            "research_weight_daily_trend": config.research_weight_daily_trend,
            "research_weight_compression": config.research_weight_compression,
            "research_weight_accumulation": config.research_weight_accumulation,
            "research_weight_stock_regime": config.research_weight_stock_regime,
            "research_weight_sector_regime": config.research_weight_sector_regime,
            "research_weight_sector_participation": config.research_weight_sector_participation,
            "research_min_rs_percentile": config.research_min_rs_percentile,
            "research_min_trend_score": config.research_min_trend_score,
            "research_min_compression_score": config.research_min_compression_score,
            "research_min_accumulation_score": config.research_min_accumulation_score,
            "research_min_sector_participation": config.research_min_sector_participation,
            "research_min_sector_daily_score_pct": config.research_min_sector_daily_score_pct,
            "research_max_box_range_pct": config.research_max_box_range_pct,
            "research_structural_frontier_count": config.research_structural_frontier_count,
            "research_min_structural_campaign_score": config.research_min_structural_campaign_score,
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "sector_map_hash": _mapping_fingerprint(sector_map or {}),
            "frontier_selector_version": "research_top_longs_v1",
            "universe_contract": "explicit_universe_required_no_parquet_fallback_data_availability_v2",
            "mutations": {
                key: raw_mutations[key]
                for key in sorted(raw_mutations)
                if key in _CANDIDATE_CONFIG_MUTATION_KEYS
            },
        }
    )


def finalize_candidate_snapshot(
    snapshot: KALCBDailySnapshot,
    *,
    config: KALCBConfig | None = None,
    candidate_config_hash: str | None = None,
    source: str = "kalcb_research_selection",
    sector_map_hash: str | None = None,
    requested_universe_count: int | None = None,
    data_available_symbols: list[str] | tuple[str, ...] | None = None,
    unavailable_symbols: list[str] | tuple[str, ...] | None = None,
    source_universe_count: int | None = None,
    sector_map_size: int | None = None,
    generated_at: datetime | None = None,
) -> KALCBDailySnapshot:
    """Materialize the executable active/frontier KALCB candidate artifact.

    The pure selector owns alpha ranking. This helper owns deployment/replay
    materialization: active seed selection, frontier ordering, and audit
    metadata. Keep this in strategy code so live and replay share the same
    executable artifact boundary.
    """

    cfg = config or KALCBConfig()
    candidates = list(snapshot.candidates)
    config_hash = str(candidate_config_hash or snapshot.metadata.get("candidate_config_hash") or "")
    map_hash = str(sector_map_hash or snapshot.metadata.get("sector_map_hash") or "")
    available_source = data_available_symbols if data_available_symbols is not None else (candidate.symbol for candidate in candidates)
    available = sorted(str(symbol).zfill(6) for symbol in available_source)
    unavailable = sorted(str(symbol).zfill(6) for symbol in (unavailable_symbols or ()))
    frontier_limit = max(1, cfg.frontier_size if cfg.frontier_enabled else cfg.ws_budget)
    frontier_limit = max(frontier_limit, cfg.ws_budget)
    hot_ranked = sorted(candidates, key=lambda item: item.selection_score, reverse=True)
    active_seed = _select_active_seed(candidates, cfg)
    frontier = _build_frontier_order(candidates, active_seed, cfg, frontier_limit)
    active_symbols = {candidate.symbol for candidate in active_seed}
    active_rank_by_symbol = {candidate.symbol: index for index, candidate in enumerate(active_seed, start=1)}
    hot_rank_by_symbol = {candidate.symbol: index for index, candidate in enumerate(hot_ranked, start=1)}
    selected = tuple(
        replace(
            candidate,
            metadata={
                **dict(candidate.metadata),
                "candidate_config_hash": config_hash,
                "source": source,
                "frontier_enabled": cfg.frontier_enabled,
                "frontier_size": frontier_limit,
                "frontier_selection_mode": cfg.frontier_selection_mode,
                "frontier_active_selection_mode": cfg.frontier_active_selection_mode,
                "frontier_rank": rank,
                "frontier_hot_rank": hot_rank_by_symbol.get(candidate.symbol, 0),
                "frontier_initial_active": candidate.symbol in active_symbols,
                "frontier_role": "initial_active" if candidate.symbol in active_symbols else "shadow",
                "active_rank": active_rank_by_symbol.get(candidate.symbol, 0),
            },
        )
        for rank, candidate in enumerate(frontier, start=1)
    )
    pool_symbols = list(snapshot.metadata.get("candidate_pool_symbols") or [candidate.symbol for candidate in candidates])
    selected_symbol_set = {item.symbol for item in selected}
    requested_count = int(
        requested_universe_count
        if requested_universe_count is not None
        else snapshot.metadata.get("requested_universe_count")
        or snapshot.metadata.get("requested_symbol_count")
        or len(available)
    )
    source_count = int(
        source_universe_count
        if source_universe_count is not None
        else snapshot.metadata.get("source_universe_count")
        or snapshot.metadata.get("requested_symbol_count")
        or requested_count
    )
    candidate_pool_count = int(snapshot.metadata.get("candidate_pool_count") or len(candidates))
    recorded_sector_map_size = int(
        sector_map_size
        if sector_map_size is not None
        else snapshot.metadata.get("sector_map_size")
        or 0
    )
    return KALCBDailySnapshot(
        trade_date=snapshot.trade_date,
        candidates=selected,
        source_fingerprint=snapshot.source_fingerprint,
        generated_at=generated_at or snapshot.generated_at,
        strategy_id=snapshot.strategy_id,
        metadata={
            **dict(snapshot.metadata),
            "artifact_stage": KALCB_FINAL_ARTIFACT_STAGE,
            "candidate_config_hash": config_hash,
            "source": source,
            "sector_map_hash": map_hash,
            "sector_map_size": recorded_sector_map_size,
            "ws_budget": cfg.ws_budget,
            "frontier_enabled": cfg.frontier_enabled,
            "frontier_size": frontier_limit,
            "frontier_selection_mode": cfg.frontier_selection_mode,
            "frontier_active_selection_mode": cfg.frontier_active_selection_mode,
            "requested_universe_count": requested_count,
            "data_available_symbol_count": len(available),
            "unavailable_symbol_count": len(unavailable),
            "unavailable_symbols": list(unavailable),
            "source_universe_count": source_count,
            "candidate_pool_count": candidate_pool_count,
            "candidate_pool_symbols": pool_symbols,
            "frontier_symbols": [candidate.symbol for candidate in selected],
            "frontier_symbol_count": len(selected),
            "overflow_symbols": [symbol for symbol in pool_symbols if symbol not in selected_symbol_set],
            "overflow_symbol_count": max(0, candidate_pool_count - len(selected)),
            "active_symbols": [candidate.symbol for candidate in selected if candidate.symbol in active_symbols],
            "active_symbol_count": len(active_symbols),
            "frontier_rest_budget_symbols_per_5m": _frontier_rest_budget_symbols_per_5m(cfg),
        },
    )


def run_daily_selection(
    daily_by_symbol: dict[str, list[dict[str, Any]]] | KALCBResearchSnapshot,
    trade_date: date | None = None,
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
    """Build the unfinalized KALCB daily candidate snapshot from rows or snapshot.

    Persisting executable KALCB artifacts requires ``finalize_candidate_snapshot()``
    first. The low-level selector is non-persistent by default to avoid writing
    research-stage artifacts into the live/paper artifact store.
    """

    cfg = config or KALCBConfig()
    research_snapshot = _coerce_kalcb_research_snapshot(
        daily_by_symbol,
        trade_date,
        cfg,
        sector_map=sector_map,
        daily_flow_by_symbol=daily_flow_by_symbol,
        daily_foreign_flow_by_symbol=daily_foreign_flow_by_symbol,
        daily_institutional_flow_by_symbol=daily_institutional_flow_by_symbol,
        source_fingerprint=source_fingerprint,
        generated_at=generated_at,
    )
    candidate_snapshot = daily_selection_from_snapshot(research_snapshot, cfg)
    if artifact_root is not None:
        raise ValueError(
            "KALCB run_daily_selection builds unfinalized research-stage snapshots and does not persist them; "
            "use strategy_kalcb.research_generator.generate_finalized_candidate_snapshot() for executable artifacts."
        )
    if lrs is not None:
        raise ValueError(
            "KALCB run_daily_selection builds unfinalized research-stage snapshots and does not persist them to LRS; "
            "use strategy_kalcb.research_generator.generate_finalized_candidate_snapshot() for executable artifacts."
        )
    return candidate_snapshot


def _coerce_kalcb_research_snapshot(
    daily_by_symbol: dict[str, list[dict[str, Any]]] | KALCBResearchSnapshot,
    trade_date: date | None,
    config: KALCBConfig,
    *,
    sector_map: dict[str, str] | None = None,
    daily_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    source_fingerprint: str | None = None,
    generated_at: datetime | None = None,
) -> KALCBResearchSnapshot:
    if isinstance(daily_by_symbol, KALCBResearchSnapshot):
        return daily_by_symbol
    if trade_date is None:
        raise ValueError("KALCB run_daily_selection requires trade_date when building from rows")
    return build_research_snapshot(
        daily_by_symbol,
        trade_date,
        config,
        sector_map=sector_map,
        daily_flow_by_symbol=daily_flow_by_symbol,
        daily_foreign_flow_by_symbol=daily_foreign_flow_by_symbol,
        daily_institutional_flow_by_symbol=daily_institutional_flow_by_symbol,
        source_fingerprint=source_fingerprint,
        generated_at=generated_at,
    )


def _build_frontier_order(
    candidates: list[KALCBDailyCandidate],
    active_seed: list[KALCBDailyCandidate],
    config: KALCBConfig,
    limit: int,
) -> list[KALCBDailyCandidate]:
    mode = str(config.frontier_selection_mode or "opportunity").lower()
    if mode in {"liquidity", "prior_notional", "notional"}:
        sleeves = [active_seed, _rank_candidates(candidates, "prior_day_notional")]
    elif mode in {"hot", "prior_hot", "hybrid_hot"}:
        sleeves = [
            active_seed,
            sorted(candidates, key=lambda item: item.selection_score, reverse=True),
            _rank_candidates(candidates, "prior_day_notional"),
        ]
    else:
        sleeves = [
            active_seed,
            sorted(candidates, key=lambda item: item.selection_score, reverse=True),
            _rank_candidates(candidates, "volume_ratio_20d"),
            _rank_candidates(candidates, "rs_20d"),
            _rank_candidates(candidates, "range_pct"),
            _rank_candidates(candidates, "prior_day_notional"),
        ]
    frontier: list[KALCBDailyCandidate] = []
    seen: set[str] = set()
    max_len = max((len(sleeve) for sleeve in sleeves), default=0)
    for index in range(max_len):
        for sleeve in sleeves:
            if index >= len(sleeve):
                continue
            candidate = sleeve[index]
            if candidate.symbol in seen:
                continue
            seen.add(candidate.symbol)
            frontier.append(candidate)
            if len(frontier) >= limit:
                return frontier
    return frontier


def _select_active_seed(candidates: list[KALCBDailyCandidate], config: KALCBConfig) -> list[KALCBDailyCandidate]:
    budget = max(1, int(config.ws_budget))
    mode = str(config.frontier_active_selection_mode or config.frontier_selection_mode or "campaign").lower()
    if mode in {"liquidity", "prior_notional", "notional"}:
        ranked = _rank_candidates(candidates, "prior_day_notional")
    elif mode in {"hot", "prior_hot", "score", "research_score", "selection_score"}:
        ranked = sorted(candidates, key=lambda item: item.selection_score, reverse=True)
    else:
        ranked = sorted(candidates, key=_active_campaign_sort_key, reverse=True)
    selected = [candidate for candidate in ranked if _active_campaign_pass(candidate, mode)]
    if len(selected) < budget:
        seen = {candidate.symbol for candidate in selected}
        selected.extend(candidate for candidate in ranked if candidate.symbol not in seen)
    return selected[:budget]


def _active_campaign_sort_key(candidate: KALCBDailyCandidate) -> tuple[float, float, float, float]:
    components = dict(candidate.metadata.get("frontier_score_components") or {})
    return (
        float(candidate.selection_score),
        float(components.get("campaign_setup_score") or 0.0),
        float(components.get("rs_20d") or 0.0),
        float(components.get("prior_notional") or 0.0),
    )


def _active_campaign_pass(candidate: KALCBDailyCandidate, mode: str) -> bool:
    if mode in {"liquidity", "prior_notional", "notional", "hot", "prior_hot", "score", "research_score", "selection_score"}:
        return True
    components = dict(candidate.metadata.get("frontier_score_components") or {})
    prior_notional = float(components.get("prior_notional") or 0.0)
    if prior_notional <= 0:
        return False
    rs_20d = float(components.get("rs_20d") or 0.0)
    close_location_20d = float(components.get("close_location_20d") or 0.0)
    range_pct = float(components.get("range_pct") or 0.0)
    volume_ratio = float(components.get("volume_ratio_20d") or 0.0)
    return (rs_20d > 0.0 or close_location_20d >= 0.65) and range_pct <= 0.12 and volume_ratio >= 0.50


def _rank_candidates(candidates: list[KALCBDailyCandidate], component: str) -> list[KALCBDailyCandidate]:
    return sorted(
        candidates,
        key=lambda item: float((item.metadata.get("frontier_score_components") or {}).get(component) or item.metadata.get(component) or 0.0),
        reverse=True,
    )


def _frontier_rest_budget_symbols_per_5m(config: KALCBConfig) -> int:
    raw = int((5 * 60 / max(float(config.rest_min_interval_paper_s), 1e-9)) * max(min(float(config.frontier_rest_safety_fraction), 1.0), 0.01))
    return max(1, raw)


def _prior_rows(rows: list[dict[str, Any]], trade_date: date) -> list[dict[str, Any]]:
    return _recent_prior_rows(rows, trade_date, None)


def _recent_prior_rows(rows: list[dict[str, Any]], trade_date: date, limit: int | None) -> list[dict[str, Any]]:
    dates, ordered = _prior_row_index(rows or [])
    index = bisect.bisect_left(dates, trade_date)
    start = max(0, index - int(limit)) if limit is not None and int(limit) > 0 else 0
    return [dict(row) for row in ordered[start:index]]


def _prior_row_index(rows: list[dict[str, Any]]) -> tuple[tuple[date, ...], tuple[dict[str, Any], ...]]:
    if not rows:
        return (), ()
    key = _prior_row_index_cache_key(rows)
    cached = _PRIOR_ROW_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    dated_rows: list[tuple[date, dict[str, Any]]] = []
    for row in rows or []:
        parsed = _try_row_date(row)
        if parsed is None:
            continue
        try:
            dated_rows.append((parsed, dict(row)))
        except (TypeError, ValueError):
            continue
    dated_rows.sort(key=lambda item: item[0])
    indexed = (tuple(day for day, _row in dated_rows), tuple(row for _day, row in dated_rows))
    _PRIOR_ROW_INDEX_CACHE[key] = indexed
    return indexed


def _prior_row_index_cache_key(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    first = rows[0] if rows else {}
    middle = rows[len(rows) // 2] if rows else {}
    last = rows[-1] if rows else {}
    return (
        id(rows),
        len(rows),
        _row_cache_marker(first),
        _row_cache_marker(middle),
        _row_cache_marker(last),
    )


def _row_cache_marker(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("date") or row.get("trade_date") or row.get("timestamp"), row.get("close"), row.get("volume"))


def _try_row_date(row: dict[str, Any]) -> date | None:
    try:
        return _row_date(row)
    except (AttributeError, TypeError, ValueError):
        return None


def _row_date(row: dict[str, Any]) -> date:
    value = row.get("date") or row.get("trade_date") or row.get("timestamp")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    if "T" in text:
        return datetime.fromisoformat(text).date()
    return date.fromisoformat(text[:10])


def _build_market_research(trade_date: date, symbols: dict[str, KALCBResearchSymbol]) -> KALCBMarketResearch:
    above = 0
    counted = 0
    returns: list[float] = []
    for item in symbols.values():
        closes = [_float(row.get("close")) for row in item.daily_rows]
        if len(closes) >= 20:
            counted += 1
            above += int(closes[-1] >= _mean(closes[-20:]))
        if len(closes) >= 21:
            returns.append(item.return_20d_pct)
    breadth = 100.0 * above / counted if counted else 50.0
    avg_return = fmean(returns) if returns else 0.0
    if breadth >= 55.0 and avg_return >= 0.0:
        tier = "A"
        regime = "BULL"
    elif breadth < 40.0 and avg_return < -3.0:
        tier = "C"
        regime = "BEAR"
    elif breadth < 45.0:
        tier = "B"
        regime = "CHOP"
    else:
        tier = "B"
        regime = "TRANSITIONAL"
    return KALCBMarketResearch(
        trade_date=trade_date,
        breadth_pct_above_20dma=breadth,
        avg_20d_return_pct=avg_return,
        regime_tier=tier,
        regime=regime,
    )


def _build_sector_research(
    symbols: dict[str, KALCBResearchSymbol],
    sector_daily_panel: SectorDailyPanel | None = None,
) -> dict[str, KALCBSectorResearch]:
    if not symbols:
        return {}
    trade_date = next(iter(symbols.values())).trade_date
    panel = sector_daily_panel or _sector_daily_panel_from_symbols(symbols, trade_date)
    return {
        sector: KALCBSectorResearch(
            sector=sector,
            symbol_count=item.member_count,
            return_20d_pct=item.ret_20d * 100.0,
            breadth_20d=item.breadth_20d,
            participation=item.participation,
            regime=_legacy_sector_regime(item),
        )
        for (_, sector), item in panel.sectors_by_key.items()
    }


def _sector_daily_panel_from_symbols(
    symbols: dict[str, KALCBResearchSymbol],
    trade_date: date,
) -> SectorDailyPanel:
    key = (
        trade_date,
        tuple(
            sorted(
                (symbol, len(item.daily_rows), _row_date_label(item.daily_rows[-1]) if item.daily_rows else "")
                for symbol, item in symbols.items()
            )
        ),
    )
    cached = _SECTOR_DAILY_PANEL_CACHE.get(key)
    if cached is not None:
        return cached
    members = [
        member
        for member in (_sector_daily_member_from_research_symbol(item) for item in symbols.values())
        if member is not None
    ]
    panel = score_sector_daily_members(
        members,
        trade_date=trade_date,
        target_symbols=symbols,
    )
    _SECTOR_DAILY_PANEL_CACHE[key] = panel
    return panel


def _sector_daily_member_from_research_symbol(item: KALCBResearchSymbol) -> SectorDailyMember | None:
    rows = item.daily_rows
    if len(rows) < 21 or _bad_recent_ohlcv(rows[-20:]):
        return None
    closes = [_float(row.get("close")) for row in rows]
    volumes = [_float(row.get("volume")) for row in rows]
    if closes[-1] <= 0.0 or volumes[-1] <= 0.0:
        return None
    flow = _sector_member_flow_values(item)
    return SectorDailyMember(
        symbol=item.symbol,
        sector=item.sector,
        trade_date=item.trade_date,
        ret_5d=item.return_5d_pct / 100.0,
        ret_20d=item.return_20d_pct / 100.0,
        ret_60d=item.return_60d_pct / 100.0,
        above_sma20=closes[-1] >= _mean(closes[-20:]),
        rel_volume=item.volume_ratio_20d,
        flow_5d=flow["flow_5d"],
        foreign_flow_5d=flow["foreign_flow_5d"],
        institutional_flow_5d=flow["institutional_flow_5d"],
        flow_agreement_5d=flow["flow_agreement_5d"],
        flow_available=bool(flow["flow_available"]),
    )


def _sector_member_flow_values(item: KALCBResearchSymbol) -> dict[str, float]:
    daily_by_date = {_try_row_date(row): row for row in item.daily_rows[-5:] if _try_row_date(row) is not None}
    flow_by_date = {_try_row_date(row): row for row in item.daily_flow_rows if _try_row_date(row) is not None}
    foreign_by_date = {_try_row_date(row): row for row in item.daily_foreign_flow_rows if _try_row_date(row) is not None}
    inst_by_date = {_try_row_date(row): row for row in item.daily_institutional_flow_rows if _try_row_date(row) is not None}
    if not (flow_by_date or foreign_by_date or inst_by_date):
        return {"flow_available": 0.0, "flow_5d": 0.0, "foreign_flow_5d": 0.0, "institutional_flow_5d": 0.0, "flow_agreement_5d": 0.0}
    normalized: list[tuple[float, float, float, float]] = []
    for row_date in sorted(day for day in daily_by_date if day is not None):
        daily = daily_by_date[row_date]
        volume = max(_float(daily.get("volume")), 1.0)
        flow_row = flow_by_date.get(row_date, {})
        foreign_row = foreign_by_date.get(row_date, {})
        inst_row = inst_by_date.get(row_date, {})
        foreign = _optional_flow_value(foreign_row, "foreign_net")
        if foreign is None:
            foreign = _optional_flow_value(flow_row, "foreign_net") or 0.0
        inst = _optional_flow_value(inst_row, "institutional_net", "inst_net")
        if inst is None:
            inst = _optional_flow_value(flow_row, "institutional_net", "inst_net") or 0.0
        foreign_norm = float(foreign) / volume
        inst_norm = float(inst) / volume
        normalized.append((foreign_norm + inst_norm, foreign_norm, inst_norm, min(max(foreign_norm, 0.0), max(inst_norm, 0.0))))
    if not normalized:
        return {"flow_available": 0.0, "flow_5d": 0.0, "foreign_flow_5d": 0.0, "institutional_flow_5d": 0.0, "flow_agreement_5d": 0.0}
    return {
        "flow_available": 1.0,
        "flow_5d": fmean(item[0] for item in normalized),
        "foreign_flow_5d": fmean(item[1] for item in normalized),
        "institutional_flow_5d": fmean(item[2] for item in normalized),
        "flow_agreement_5d": fmean(item[3] for item in normalized),
    }


def _optional_flow_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return _float(row.get(key))
    return None


def _reject_reasons(item: KALCBResearchSymbol, config: KALCBConfig) -> list[str]:
    reasons: list[str] = []
    if len(item.daily_rows) < config.research_min_history_days:
        reasons.append("insufficient_history")
    if item.price < config.research_min_price_krw:
        reasons.append("price_below_floor")
    if item.adv20_krw < config.research_min_adv20_krw:
        reasons.append("adv20_below_floor")
    if _bad_recent_ohlcv(item.daily_rows[-20:]):
        reasons.append("bad_recent_ohlcv")
    latest = item.daily_rows[-1] if item.daily_rows else {}
    for key in ("blacklist_flag", "halted_flag", "severe_news_flag", "etf_flag", "preferred_flag", "otc_flag"):
        if bool(latest.get(key, False)):
            reasons.append(key.removesuffix("_flag"))
    return reasons


def _long_score(
    item: KALCBResearchSymbol,
    snapshot: KALCBResearchSnapshot,
    rs_pct: float,
    config: KALCBConfig,
    *,
    sector_daily_feature: SectorDailyFeature | None = None,
) -> tuple[float, dict[str, Any]]:
    trend_component, trend_label = _trend_component(item.daily_rows)
    compression_component, box = _compression_component(item.daily_rows)
    legacy_accumulation = _accumulation_score(item.daily_rows[-20:])
    sector = snapshot.sectors.get(item.sector)
    sector_regime_component = float(sector_daily_feature.score_pct) if sector_daily_feature is not None else _sector_regime_component(sector.regime if sector is not None else "UNKNOWN")
    structural = build_structural_campaign(
        item.symbol,
        snapshot.trade_date,
        item.daily_rows,
        sector=item.sector,
        rs_percentile=rs_pct,
        stock_vs_universe_strength=(0.65 * item.return_20d_pct + 0.35 * item.return_60d_pct),
        sector_daily_score_pct=sector_regime_component,
        sector_participation=sector_daily_feature.participation if sector_daily_feature is not None else (sector.participation if sector is not None else 0.0),
        market_heat_score=snapshot.market.breadth_pct_above_20dma,
        daily_flow_rows=item.daily_flow_rows,
        daily_foreign_flow_rows=item.daily_foreign_flow_rows,
        daily_institutional_flow_rows=item.daily_institutional_flow_rows,
    )
    accumulation_pct = float(structural.accumulation_score)
    accumulation = (accumulation_pct - 50.0) / 50.0
    accumulation_component = accumulation_pct
    stock_regime_component = _stock_regime_component(trend_label)
    sector_participation_component = 100.0 * (sector_daily_feature.participation if sector_daily_feature is not None else (sector.participation if sector is not None else 0.0))
    score = float(structural.structural_campaign_score) * 10.0
    structural_meta = campaign_metadata(structural)
    structural_detail = dict(structural.selection_detail)
    details = {
        "relative_strength_pct": rs_pct,
        "daily_trend_score": trend_component,
        "compression_score": compression_component,
        "accumulation_score": accumulation,
        "legacy_accumulation_score": legacy_accumulation,
        "accumulation_component": accumulation_component,
        "stock_regime_score": stock_regime_component,
        "sector_regime_score": sector_regime_component,
        "sector_participation_score": sector_participation_component,
        "return_5d_pct": item.return_5d_pct,
        "return_20d_pct": item.return_20d_pct,
        "return_60d_pct": item.return_60d_pct,
        "volume_ratio_20d": item.volume_ratio_20d,
        "close_location_20d": item.close_location_20d,
        "box_high": box.get("high", 0.0),
        "box_low": box.get("low", 0.0),
        "box_range_pct": box.get("range_pct", 0.0),
        "box_containment": box.get("containment", 0.0),
        "structural_campaign_score": float(structural.structural_campaign_score),
        "first30_confirmation_score": 0.0,
        "campaign_state": structural.campaign_state,
        "campaign_state_score": structural_meta["campaign_state_score"],
        "campaign_box_high": structural_meta["campaign_box_high"],
        "campaign_box_low": structural_meta["campaign_box_low"],
        "campaign_box_mid": structural_meta["campaign_box_mid"],
        "campaign_box_range_pct": structural_meta["campaign_box_range_pct"],
        "campaign_box_containment": structural_meta["campaign_box_containment"],
        "campaign_avwap": 0.0,
        "campaign_breakout_level": structural_meta["campaign_breakout_level"],
        "campaign_breakout_displacement": structural_meta["campaign_breakout_displacement"],
        "structural_source_score": float(structural.structural_campaign_score),
        "campaign_setup_score": score,
        "hot_score": score,
        "rs_20d": item.return_20d_pct / 100.0,
        "prior_day_notional": item.price * _float(item.daily_rows[-1].get("volume")),
        "prior_notional": item.price * _float(item.daily_rows[-1].get("volume")),
        "range_pct": _range_pct(item.daily_rows[-1]),
        "structural_campaign_metadata": structural_meta,
        **structural_detail,
        **(sector_daily_feature.metadata() if sector_daily_feature is not None else {}),
    }
    return round(max(0.0, min(score, 100.0)), 6), details


def _research_score_weights(config: KALCBConfig) -> dict[str, float]:
    raw = {
        "relative_strength": max(float(config.research_weight_relative_strength), 0.0),
        "daily_trend": max(float(config.research_weight_daily_trend), 0.0),
        "compression": max(float(config.research_weight_compression), 0.0),
        "accumulation": max(float(config.research_weight_accumulation), 0.0),
        "stock_regime": max(float(config.research_weight_stock_regime), 0.0),
        "sector_regime": max(float(config.research_weight_sector_regime), 0.0),
        "sector_participation": max(float(config.research_weight_sector_participation), 0.0),
    }
    total = sum(raw.values())
    if total <= 0:
        return {key: 0.0 for key in raw}
    return {key: value / total for key, value in raw.items()}


def _score_reject_reasons(details: dict[str, Any], config: KALCBConfig) -> list[str]:
    reasons: list[str] = []
    if details["relative_strength_pct"] < config.research_min_rs_percentile:
        reasons.append("rs_below_floor")
    if details["daily_trend_score"] < config.research_min_trend_score:
        reasons.append("trend_below_floor")
    if details["compression_score"] < config.research_min_compression_score:
        reasons.append("compression_below_floor")
    if details["accumulation_score"] < config.research_min_accumulation_score:
        reasons.append("accumulation_below_floor")
    if details["sector_participation_score"] < 100.0 * config.research_min_sector_participation:
        reasons.append("sector_participation_below_floor")
    if details.get("sector_daily_score_pct", 50.0) < config.research_min_sector_daily_score_pct:
        reasons.append("sector_daily_score_below_floor")
    if config.research_max_box_range_pct > 0 and details["box_range_pct"] > config.research_max_box_range_pct:
        reasons.append("box_range_above_cap")
    if float(details.get("structural_campaign_score") or 0.0) < float(config.research_min_structural_campaign_score):
        reasons.append("structural_campaign_score_below_floor")
    return reasons


def _relative_strength_percentiles(symbols: dict[str, KALCBResearchSymbol]) -> dict[str, float]:
    raw = {
        symbol: 0.65 * item.return_20d_pct + 0.35 * item.return_60d_pct
        for symbol, item in symbols.items()
    }
    return compute_rs_percentiles(raw)


def _trend_component(rows: tuple[dict[str, Any], ...]) -> tuple[float, str]:
    closes = [_float(row.get("close")) for row in rows]
    if len(closes) < 20:
        return 0.0, "UNKNOWN"
    sma20 = _mean(closes[-20:])
    sma60 = _mean(closes[-60:]) if len(closes) >= 60 else sma20
    close = closes[-1]
    sma20_rising = len(closes) < 25 or sma20 >= _mean(closes[-25:-5])
    if close >= sma20 >= sma60 and sma20_rising:
        return 100.0, "BULL"
    if close >= sma60:
        return 70.0, "TRANSITIONAL"
    if close >= sma20:
        return 45.0, "CHOP"
    return 10.0, "BEAR"


def _compression_component(rows: tuple[dict[str, Any], ...]) -> tuple[float, dict[str, float]]:
    if len(rows) < 20:
        return 0.0, {}
    window_10 = rows[-10:]
    window_20 = rows[-20:]
    close = _float(rows[-1].get("close"))
    high_10 = max(_float(row.get("high")) for row in window_10)
    low_10 = min(_float(row.get("low")) for row in window_10)
    high_20 = max(_float(row.get("high")) for row in window_20)
    low_20 = min(_float(row.get("low")) for row in window_20)
    range_10 = max(high_10 - low_10, 0.0)
    range_20 = max(high_20 - low_20, 1e-9)
    compression = max(0.0, min(1.0, 1.0 - (range_10 / range_20)))
    containment = sum(1 for row in window_20 if low_10 <= _float(row.get("close")) <= high_10) / len(window_20)
    close_location = (close - low_10) / max(range_10, 1e-9)
    quality = 100.0 * max(0.0, min(1.0, 0.55 * compression + 0.25 * containment + 0.20 * max(0.0, min(close_location, 1.0))))
    return quality, {
        "high": high_10,
        "low": low_10,
        "range_pct": (range_10 / close) if close > 0 else 0.0,
        "containment": containment,
    }


def _stock_regime_component(trend_label: str) -> float:
    return {
        "BULL": 100.0,
        "TRANSITIONAL": 70.0,
        "CHOP": 45.0,
        "BEAR": 10.0,
    }.get(trend_label, 40.0)


def _sector_regime_component(regime: str) -> float:
    return {
        "BULL": 100.0,
        "TRANSITIONAL": 70.0,
        "CHOP": 45.0,
        "BEAR": 10.0,
    }.get(str(regime).upper(), 40.0)


def _legacy_sector_regime(sector: Any) -> str:
    regime = str(getattr(sector, "regime", "UNKNOWN")).upper()
    if regime == "LEADING":
        return "BULL"
    if regime == "LAGGING":
        return "BEAR"
    ret_20d = float(getattr(sector, "ret_20d", 0.0))
    breadth_20d = float(getattr(sector, "breadth_20d", 0.5))
    if abs(ret_20d) < 0.02:
        return "CHOP"
    return "TRANSITIONAL" if breadth_20d >= 0.45 else "CHOP"


def _sector_flow_score(sector: SectorDailyFeature | KALCBSectorResearch | None) -> float:
    if sector is None:
        return 0.0
    if isinstance(sector, SectorDailyFeature):
        return sector.ret_20d + (sector.participation - 0.5)
    return (sector.return_20d_pct / 100.0) + (sector.participation - 0.5)


def _accumulation_score(rows: tuple[dict[str, Any], ...]) -> float:
    signed_volume = 0.0
    total_volume = 0.0
    for row in rows:
        close = _float(row.get("close"))
        open_ = _float(row.get("open"), close)
        volume = _float(row.get("volume"))
        if volume <= 0:
            continue
        if close > open_:
            signed_volume += volume
        elif close < open_:
            signed_volume -= volume
        total_volume += volume
    return signed_volume / total_volume if total_volume > 0 else 0.0


def _bad_recent_ohlcv(rows: tuple[dict[str, Any], ...]) -> bool:
    if not rows:
        return True
    for row in rows:
        open_ = _float(row.get("open"))
        high = _float(row.get("high"))
        low = _float(row.get("low"))
        close = _float(row.get("close"))
        volume = _float(row.get("volume"))
        if min(open_, high, low, close) <= 0 or volume <= 0:
            return True
        if high < low or close > high * 1.0001 or close < low * 0.9999:
            return True
    return False


def _adv_krw(rows: list[dict[str, Any]], period: int) -> float:
    sample = rows[-period:] if len(rows) >= period else rows
    notionals = [_float(row.get("close")) * _float(row.get("volume")) for row in sample]
    return fmean(notionals) if notionals else 0.0


def _return_pct(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...], periods: int) -> float:
    if len(rows) <= periods:
        return 0.0
    current = _float(rows[-1].get("close"))
    prior = _float(rows[-periods - 1].get("close"))
    return ((current / prior) - 1.0) * 100.0 if prior > 0 else 0.0


def _volume_ratio(rows: list[dict[str, Any]], period: int) -> float:
    if len(rows) <= period:
        return 1.0
    prior = [_float(row.get("volume")) for row in rows[-period - 1 : -1]]
    baseline = fmean(prior) if prior else 0.0
    return _float(rows[-1].get("volume")) / baseline if baseline > 0 else 1.0


def _close_location(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    high = max(_float(row.get("high")) for row in rows)
    low = min(_float(row.get("low")) for row in rows)
    close = _float(rows[-1].get("close"))
    return (close - low) / max(high - low, 1e-9)


def _above_sma(rows: tuple[dict[str, Any], ...], period: int) -> bool:
    closes = [_float(row.get("close")) for row in rows]
    return len(closes) >= period and closes[-1] >= _mean(closes[-period:])


def _range_pct(row: dict[str, Any]) -> float:
    close = _float(row.get("close"))
    if close <= 0:
        return 0.0
    return (_float(row.get("high")) - _float(row.get("low"))) / close


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_sector(value: Any) -> str:
    text = str(value or "").strip()
    return text.upper() if text else "UNKNOWN"


def _snapshot_candidate_metadata(snapshot: KALCBResearchSnapshot) -> dict[str, Any]:
    keys = (
        "candidate_config_hash",
        "research_config_hash",
        "sector_map_hash",
        "research_lookahead_policy",
        "research_trade_date",
        "research_as_of_date",
        "research_causal_source_fingerprint",
    )
    return {key: snapshot.metadata[key] for key in keys if key in snapshot.metadata}


def _research_as_of_date(symbols: dict[str, KALCBResearchSymbol]) -> str:
    dates = [_row_date_label(item.daily_rows[-1]) for item in symbols.values() if item.daily_rows]
    return max(dates) if dates else ""


def _research_config_fingerprint(config: KALCBConfig) -> str:
    return _mapping_fingerprint(
        {
            "research_model_version": RESEARCH_MODEL_VERSION,
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "top_long_count": config.research_top_long_count,
            "min_price_krw": config.research_min_price_krw,
            "min_adv20_krw": config.research_min_adv20_krw,
            "min_history_days": config.research_min_history_days,
            "weights": _research_score_weights(config),
            "min_rs_percentile": config.research_min_rs_percentile,
            "min_trend_score": config.research_min_trend_score,
            "min_compression_score": config.research_min_compression_score,
            "min_accumulation_score": config.research_min_accumulation_score,
            "min_sector_participation": config.research_min_sector_participation,
            "min_sector_daily_score_pct": config.research_min_sector_daily_score_pct,
            "max_box_range_pct": config.research_max_box_range_pct,
            "structural_frontier_count": config.research_structural_frontier_count,
            "min_structural_campaign_score": config.research_min_structural_campaign_score,
        }
    )


def _mapping_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(raw.encode("utf-8")).hexdigest()


def _source_fingerprint(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    *,
    daily_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    daily_institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    symbols: dict[str, dict[str, Any]] = {}
    for symbol, rows in sorted(daily_by_symbol.items()):
        normalized_symbol = str(symbol).zfill(6)
        prior_rows = _prior_rows(rows, trade_date)
        flow_rows = _prior_rows((daily_flow_by_symbol or {}).get(normalized_symbol, []), trade_date)
        foreign_rows = _prior_rows((daily_foreign_flow_by_symbol or {}).get(normalized_symbol, []), trade_date)
        institutional_rows = _prior_rows((daily_institutional_flow_by_symbol or {}).get(normalized_symbol, []), trade_date)
        symbols[normalized_symbol] = {
            "rows": len(prior_rows),
            "last": _row_date_label(prior_rows[-1]) if prior_rows else "",
            "bars_hash": _rows_fingerprint(prior_rows),
            "flow_hash": _rows_fingerprint(flow_rows),
            "foreign_flow_hash": _rows_fingerprint(foreign_rows),
            "institutional_flow_hash": _rows_fingerprint(institutional_rows),
        }
    payload = {
        "trade_date": trade_date.isoformat(),
        "sector_daily_version": SECTOR_DAILY_VERSION,
        "symbols": symbols,
    }
    return _mapping_fingerprint(payload)


def _rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = []
    for row in rows:
        payload.append({"date": _row_date_label(row), **{str(key): value for key, value in sorted(row.items()) if str(key) not in {"date", "trade_date", "timestamp"}}})
    return _mapping_fingerprint({"rows": payload})


def _row_date_label(row: dict[str, Any]) -> str:
    parsed = _try_row_date(row)
    if parsed is not None:
        return parsed.isoformat()
    return str(row.get("date") or row.get("trade_date") or row.get("timestamp") or "")
