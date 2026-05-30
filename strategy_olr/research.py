from __future__ import annotations

import bisect
import json
import math
from dataclasses import replace
from datetime import date, datetime, time
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_common.sector_daily import SECTOR_DAILY_VERSION, SectorDailyFeature, SectorDailyPanel, build_sector_daily_panel
from strategy_common.sector_intraday import build_sector_intraday_panel
from strategy_common.sector_map import normalize_sector_map

from .artifact_store import (
    OLR_FINAL_ARTIFACT_STAGE,
    OLR_STAGE1_ARTIFACT_STAGE,
    OLRArtifactStore,
    load_snapshot_from_lrs,
    save_snapshot_to_lrs,
)
from .config import OLRConfig, OLR_CORE_VERSION
from .models import (
    OLRAfternoonContext,
    OLRDailyCandidate,
    OLRDailySnapshot,
    OLRMarketResearch,
    OLRResearchSnapshot,
    OLRResearchSymbol,
    OLRSectorResearch,
)


RESEARCH_MODEL_VERSION = "olr-research-selection-v1"
AFTERNOON_SELECTION_VERSION = "olr-afternoon-selection-v1"
FINAL_CANDIDATE_CONFIG_HASH_VERSION = "olr-final-candidate-config-v1"
INTRADAY_SELECTION_CUTOFF = time(14, 30)
INTRADAY_SELECTION_CUTOFF_LABEL = "timestamp < 14:30 KST"
_EMPTY_ROW_DIGEST = ""
_ROW_INDEX_CACHE: dict[int, tuple[int, Any, tuple[date, ...], tuple[dict[str, Any], ...]]] = {}
_ROW_DIGEST_PREFIX_CACHE: dict[int, tuple[int, Any, tuple[date, ...], tuple[str, ...]]] = {}
_FINAL_CONFIG_HASH_EXCLUDED_FIELDS = {
    "live_parity_fill_timing",
    "entry_mode",
    "exit_mode",
    "allocation_mode",
    "target_gross_exposure",
    "max_position_pct",
    "rank_decay",
    "min_selected",
    "auction_fill_time",
    "auction_limit_offset_bps",
    "auction_adverse_bps",
    "auction_nonfill_rate",
    "market_entry_price_buffer_bps",
    "trade_entry_plan",
    "trade_exit_plan",
    "slippage_bps",
    "commission_bps",
    "tax_bps_on_sell",
}


def load_candidate_snapshot(
    trade_date: date,
    *,
    artifact_root: str | Path = "data/strategy/olr",
    artifact_stage: str | None = None,
    lrs=None,
) -> OLRDailySnapshot | None:
    if lrs is not None:
        snapshot = load_snapshot_from_lrs(trade_date, lrs, artifact_stage=artifact_stage)
        if snapshot is not None:
            return snapshot
    return OLRArtifactStore(artifact_root).load_snapshot(trade_date, artifact_stage=artifact_stage)


def build_research_snapshot(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
    config: OLRConfig | None = None,
    *,
    sector_map: dict[str, str] | None = None,
    flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    index_ohlcv_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    generated_at: datetime | None = None,
    source_fingerprint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> OLRResearchSnapshot:
    """Build a causal daily OLR research snapshot from prior completed rows only."""

    cfg = config or OLRConfig()
    input_metadata = dict(metadata or {})
    daily_lookback = max(0, int(input_metadata.get("research_snapshot_daily_lookback") or 0))
    flow_lookback = max(0, int(input_metadata.get("research_snapshot_flow_lookback") or 0))
    sectors = {str(symbol).zfill(6): _normalize_sector(sector) for symbol, sector in dict(sector_map or {}).items()}
    flow_by_symbol = flow_by_symbol or {}
    foreign_flow_by_symbol = foreign_flow_by_symbol or {}
    institutional_flow_by_symbol = institutional_flow_by_symbol or {}
    symbols: dict[str, OLRResearchSymbol] = {}
    skipped: dict[str, list[str]] = {}
    for raw_symbol, raw_rows in sorted((daily_by_symbol or {}).items()):
        symbol = str(raw_symbol).zfill(6)
        prior_rows = _prior_rows(raw_rows, trade_date, daily_lookback or None)
        prior_flow = _prior_rows(flow_by_symbol.get(symbol, []), trade_date, flow_lookback or None)
        prior_foreign = _prior_rows(foreign_flow_by_symbol.get(symbol, []), trade_date, flow_lookback or None)
        prior_inst = _prior_rows(institutional_flow_by_symbol.get(symbol, []), trade_date, flow_lookback or None)
        stored_daily_rows = prior_rows
        stored_flow = prior_flow
        stored_foreign = prior_foreign
        stored_inst = prior_inst
        if not prior_rows:
            skipped[symbol] = ["missing_completed_daily_ohlcv"]
            continue
        prior = prior_rows[-1]
        prior_close = _float(prior.get("close"))
        prior_volume = _float(prior.get("volume"))
        if prior_close <= 0 or prior_volume <= 0 or _float(prior.get("high"), prior_close) < _float(prior.get("low"), prior_close):
            skipped[symbol] = ["invalid_latest_ohlcv"]
            continue
        flow_metrics = _flow_metrics(prior_rows, prior_flow, prior_foreign, prior_inst)
        symbols[symbol] = OLRResearchSymbol(
            symbol=symbol,
            trade_date=trade_date,
            daily_rows=tuple(dict(row) for row in stored_daily_rows),
            flow_rows=tuple(dict(row) for row in stored_flow),
            foreign_flow_rows=tuple(dict(row) for row in stored_foreign),
            institutional_flow_rows=tuple(dict(row) for row in stored_inst),
            sector=sectors.get(symbol, "UNKNOWN"),
            price=prior_close,
            adv20_krw=_adv_krw(prior_rows, 20),
            prior_day_high=_float(prior.get("high"), prior_close),
            prior_day_low=_float(prior.get("low"), prior_close),
            prior_day_close=prior_close,
            daily_atr=max(_atr(prior_rows, 14), prior_close * 0.01),
            expected_5m_volume=max(prior_volume / 78.0, 1.0),
            average_30m_volume=max(prior_volume / 13.0, 1.0),
            return_5d_pct=_return_pct(prior_rows, 5),
            return_20d_pct=_return_pct(prior_rows, 20),
            return_60d_pct=_return_pct(prior_rows, 60),
            volume_ratio_20d=_volume_ratio(prior_rows, 20),
            close_location_20d=_close_location(prior_rows[-20:]),
            median_spread_pct=_spread_pct(prior),
            flow_available=bool(flow_metrics["flow_available"]),
            flow_1d=flow_metrics["flow_1d"],
            flow_3d=flow_metrics["flow_3d"],
            flow_5d=flow_metrics["flow_5d"],
            flow_20d=flow_metrics["flow_20d"],
            foreign_flow_1d=flow_metrics["foreign_flow_1d"],
            foreign_flow_3d=flow_metrics["foreign_flow_3d"],
            foreign_flow_5d=flow_metrics["foreign_flow_5d"],
            foreign_flow_20d=flow_metrics["foreign_flow_20d"],
            institutional_flow_1d=flow_metrics["institutional_flow_1d"],
            institutional_flow_3d=flow_metrics["institutional_flow_3d"],
            institutional_flow_5d=flow_metrics["institutional_flow_5d"],
            institutional_flow_20d=flow_metrics["institutional_flow_20d"],
            flow_z=flow_metrics["flow_z"],
            foreign_flow_z=flow_metrics["foreign_flow_z"],
            institutional_flow_z=flow_metrics["institutional_flow_z"],
            flow_positive_days_5d=flow_metrics["flow_positive_days_5d"],
            foreign_positive_days_5d=flow_metrics["foreign_positive_days_5d"],
            institutional_positive_days_5d=flow_metrics["institutional_positive_days_5d"],
            flow_acceleration=flow_metrics["flow_acceleration"],
            foreign_flow_acceleration=flow_metrics["foreign_flow_acceleration"],
            institutional_flow_acceleration=flow_metrics["institutional_flow_acceleration"],
            flow_agreement_5d=flow_metrics["flow_agreement_5d"],
            flow_divergence_5d=flow_metrics["flow_divergence_5d"],
            combined_flow_notional_5d=flow_metrics["combined_flow_notional_5d"],
            sponsorship_balance_5d=flow_metrics["sponsorship_balance_5d"],
            etf_flag=_flag(prior, "etf_flag", "is_etf", "etf"),
            preferred_flag=_flag(prior, "preferred_flag", "is_preferred", "preferred"),
            otc_flag=_flag(prior, "otc_flag", "is_otc", "otc"),
            hard_to_borrow_flag=_flag(prior, "hard_to_borrow_flag", "hard_to_borrow"),
            blacklist_flag=_flag(prior, "blacklist_flag", "blacklisted"),
            halted_flag=_flag(prior, "halted_flag", "halted", "is_halted"),
            severe_news_flag=_flag(prior, "severe_news_flag", "severe_news"),
        )
    sector_daily_panel = _sector_daily_panel_from_symbols(symbols, trade_date)
    sector_research = _build_sector_research(symbols, sector_daily_panel)
    market = _build_market_research(trade_date, symbols, index_ohlcv_by_symbol or {})
    causal_source_fingerprint = _source_fingerprint(
        daily_by_symbol,
        flow_by_symbol,
        foreign_flow_by_symbol,
        institutional_flow_by_symbol,
        index_ohlcv_by_symbol or {},
        trade_date,
    )
    return OLRResearchSnapshot(
        trade_date=trade_date,
        market=market,
        sectors=sector_research,
        symbols=symbols,
        source_fingerprint=source_fingerprint or causal_source_fingerprint,
        generated_at=generated_at or datetime.now(tz=KST),
        metadata={
            **input_metadata,
            "research_model_version": RESEARCH_MODEL_VERSION,
            "strategy_core_version": OLR_CORE_VERSION,
            "selection_time_basis": "pre_session_from_prior_completed_daily_rows",
            "data_availability_policy": "prior_completed_daily_rows_only",
            "daily_row_cutoff": "row_date < trade_date",
            "flow_row_cutoff": "row_date < trade_date",
            "same_day_daily_ohlcv_visible": False,
            "same_day_daily_flow_visible": False,
            "official_performance": False,
            "requested_symbol_count": len(daily_by_symbol or {}),
            "source_universe_count": len(symbols),
            "build_rejected_symbols": skipped,
            "sector_map_size": len(sectors),
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "research_trade_date": trade_date.isoformat(),
            "research_as_of_date": _research_as_of_date(symbols),
            "research_config_hash": _research_config_fingerprint(cfg),
            "sector_map_hash": _mapping_fingerprint(sectors),
            "research_causal_source_fingerprint": causal_source_fingerprint,
        },
    )


def daily_selection_from_snapshot(
    snapshot: OLRResearchSnapshot,
    config: OLRConfig | None = None,
) -> OLRDailySnapshot:
    """Select the pre-session OLR candidate pool from a causal research snapshot."""

    cfg = config or OLRConfig()
    rs_percentiles = _relative_strength_percentiles(snapshot.symbols)
    rs60_percentiles = _relative_strength_60d_percentiles(snapshot.symbols)
    sector_strength_percentiles = _sector_strength_percentiles(snapshot.sectors)
    sector_daily_panel = _sector_daily_panel_from_symbols(snapshot.symbols, snapshot.trade_date)
    rejected: dict[str, list[str]] = {key: list(value) for key, value in dict(snapshot.metadata.get("build_rejected_symbols") or {}).items()}
    scored: list[tuple[tuple[Any, ...], OLRDailyCandidate]] = []
    for symbol, research_symbol in snapshot.symbols.items():
        sector_daily_feature = sector_daily_panel.feature_for(snapshot.trade_date, symbol, sector=research_symbol.sector)
        reasons = _reject_reasons(research_symbol, snapshot, cfg, sector_daily_feature=sector_daily_feature)
        if reasons:
            rejected[symbol] = reasons
            continue
        score, details = _long_score(
            research_symbol,
            snapshot,
            rs_percentiles.get(symbol, 0.0),
            rs60_percentiles.get(symbol, rs_percentiles.get(symbol, 0.0)),
            sector_strength_percentiles.get(research_symbol.sector, 0.0),
            cfg,
            sector_daily_feature=sector_daily_feature,
        )
        score_reasons = _score_reject_reasons(details, cfg)
        if score_reasons:
            rejected[symbol] = score_reasons
            continue
        sector = snapshot.sectors.get(research_symbol.sector)
        candidate = OLRDailyCandidate(
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
            daily_signal_score=details["daily_signal_score"],
            rs_percentile=details["relative_strength_pct"],
            accumulation_score=details["accumulation_raw"],
            flow_score=details["flow_score_raw"],
            foreign_flow_5d=research_symbol.foreign_flow_5d,
            institutional_flow_5d=research_symbol.institutional_flow_5d,
            flow_agreement_5d=research_symbol.flow_agreement_5d,
            tradable=True,
            source_fingerprint=snapshot.source_fingerprint,
            metadata={
                **_snapshot_candidate_metadata(snapshot),
                "research_model_version": RESEARCH_MODEL_VERSION,
                "source": "olr_research_selection",
                "prior_day_date": _row_date_label(research_symbol.daily_rows[-1]),
                "prior_day_notional": research_symbol.price * _float(research_symbol.daily_rows[-1].get("volume")),
                "adv20_krw": research_symbol.adv20_krw,
                "median_spread_pct": research_symbol.median_spread_pct,
                "return_5d_pct": research_symbol.return_5d_pct,
                "return_20d_pct": research_symbol.return_20d_pct,
                "return_60d_pct": research_symbol.return_60d_pct,
                "frontier_selection_score": score,
                "frontier_score_components": details,
                "research_score_components": details,
                "market_regime": snapshot.market.regime,
                "market_breadth_pct": snapshot.market.breadth_pct_above_20dma,
                "market_heat_score": snapshot.market.market_heat_score,
                **sector_daily_feature.metadata(),
                "sector_strength_pct": details["sector_strength_pct"],
                "sector_regime": sector.regime if sector is not None else "UNKNOWN",
                "sector_participation": sector.participation if sector is not None else 0.0,
                "sector_flow_5d": sector.flow_5d if sector is not None else 0.0,
                "sector_foreign_flow_5d": sector.foreign_flow_5d if sector is not None else 0.0,
                "sector_institutional_flow_5d": sector.institutional_flow_5d if sector is not None else 0.0,
                "sector_flow_agreement_5d": sector.flow_agreement_5d if sector is not None else 0.0,
                "lagged_flow_available": research_symbol.flow_available,
                "lagged_flow_1d": research_symbol.flow_1d,
                "lagged_flow_3d": research_symbol.flow_3d,
                "lagged_flow_5d": research_symbol.flow_5d,
                "lagged_flow_20d": research_symbol.flow_20d,
                "lagged_foreign_flow_1d": research_symbol.foreign_flow_1d,
                "lagged_foreign_flow_3d": research_symbol.foreign_flow_3d,
                "lagged_foreign_flow_5d": research_symbol.foreign_flow_5d,
                "lagged_foreign_flow_20d": research_symbol.foreign_flow_20d,
                "lagged_institutional_flow_1d": research_symbol.institutional_flow_1d,
                "lagged_institutional_flow_3d": research_symbol.institutional_flow_3d,
                "lagged_institutional_flow_5d": research_symbol.institutional_flow_5d,
                "lagged_institutional_flow_20d": research_symbol.institutional_flow_20d,
                "lagged_flow_z": research_symbol.flow_z,
                "lagged_foreign_z": research_symbol.foreign_flow_z,
                "lagged_institutional_z": research_symbol.institutional_flow_z,
                "lagged_flow_positive_days_5d": research_symbol.flow_positive_days_5d,
                "lagged_foreign_positive_days_5d": research_symbol.foreign_positive_days_5d,
                "lagged_institutional_positive_days_5d": research_symbol.institutional_positive_days_5d,
                "lagged_flow_acceleration": research_symbol.flow_acceleration,
                "lagged_foreign_flow_acceleration": research_symbol.foreign_flow_acceleration,
                "lagged_institutional_flow_acceleration": research_symbol.institutional_flow_acceleration,
                "lagged_flow_agreement_5d": research_symbol.flow_agreement_5d,
                "lagged_flow_divergence_5d": research_symbol.flow_divergence_5d,
                "lagged_combined_flow_notional_5d": research_symbol.combined_flow_notional_5d,
                "lagged_sponsorship_balance_5d": research_symbol.sponsorship_balance_5d,
                "same_day_flow_used": False,
                "official_performance": False,
            },
        )
        scored.append((_daily_sort_key(candidate, cfg), candidate))

    scored.sort(key=lambda item: item[0])
    total = max(1, len(scored))
    ranked = tuple(
        replace(candidate, rank=index, rank_pct=(index / total) * 100.0)
        for index, (_, candidate) in enumerate(scored, start=1)
    )
    selected = ranked[: cfg.research_top_long_count]
    overflow = ranked[cfg.research_top_long_count :]
    return OLRDailySnapshot(
        trade_date=snapshot.trade_date,
        candidates=selected,
        source_fingerprint=snapshot.source_fingerprint,
        generated_at=snapshot.generated_at,
        metadata={
            **dict(snapshot.metadata),
            "research_model_version": RESEARCH_MODEL_VERSION,
            "strategy_core_version": OLR_CORE_VERSION,
            "source": "olr_research_selection",
            "artifact_stage": OLR_STAGE1_ARTIFACT_STAGE,
            "selection_time_basis": "pre_session_from_prior_completed_daily_rows",
            "data_availability_policy": "prior_completed_daily_rows_only",
            "daily_row_cutoff": "row_date < trade_date",
            "flow_row_cutoff": "row_date < trade_date",
            "same_day_daily_ohlcv_visible": False,
            "same_day_daily_flow_visible": False,
            "official_performance": False,
            "top_long_count": cfg.research_top_long_count,
            "candidate_pool_count": len(ranked),
            "candidate_pool_symbols": [candidate.symbol for candidate in ranked],
            "frontier_active_selection_mode": cfg.frontier_active_selection_mode,
            "selected_symbols": [candidate.symbol for candidate in selected],
            "selected_symbol_count": len(selected),
            "overflow_symbols": [candidate.symbol for candidate in overflow],
            "overflow_symbol_count": len(overflow),
            "rejected_symbol_count": len(rejected),
            "rejected_symbols": rejected,
            "market": {
                "breadth_pct_above_20dma": snapshot.market.breadth_pct_above_20dma,
                "avg_20d_return_pct": snapshot.market.avg_20d_return_pct,
                "market_heat_score": snapshot.market.market_heat_score,
                "regime_tier": snapshot.market.regime_tier,
                "regime": snapshot.market.regime,
            },
        },
    )


def build_afternoon_contexts(
    snapshot: OLRDailySnapshot,
    bars_by_symbol: Mapping[Any, Any],
    config: OLRConfig | None = None,
    sector_map: Mapping[str, str] | None = None,
) -> dict[str, OLRAfternoonContext]:
    """Build 14:30-decision contexts from completed 5m bars strictly before 14:30."""

    cfg = config or OLRConfig()
    effective_sector_map = normalize_sector_map(sector_map or {candidate.symbol: candidate.sector for candidate in snapshot.candidates})
    sector_panel = build_sector_intraday_panel(
        bars_by_symbol,
        effective_sector_map,
        trade_dates=(snapshot.trade_date,),
        cutoff=INTRADAY_SELECTION_CUTOFF,
    )
    contexts: dict[str, OLRAfternoonContext] = {}
    for candidate in snapshot.candidates:
        bars = _intraday_selection_bars(
            _bars_for_candidate(bars_by_symbol, snapshot.trade_date, candidate.symbol),
            snapshot.trade_date,
        )
        if not bars:
            continue
        first = bars[0]
        last = bars[-1]
        open_ = max(float(first.open), 1e-9)
        high = max(float(bar.high) for bar in bars)
        low = min(float(bar.low) for bar in bars)
        volume = sum(max(float(bar.volume), 0.0) for bar in bars)
        vwap_num = sum(((float(bar.high) + float(bar.low) + float(bar.close)) / 3.0) * max(float(bar.volume), 0.0) for bar in bars)
        vwap = vwap_num / volume if volume > 0.0 else float(last.close)
        range_ = max(high - low, 0.0)
        prev_close = max(float(candidate.prior_day_close), 1e-9)
        sector_feature = sector_panel.feature_for(snapshot.trade_date, candidate.symbol, sector=candidate.sector)
        contexts[candidate.symbol] = OLRAfternoonContext(
            trade_date=snapshot.trade_date,
            symbol=candidate.symbol,
            candidate=candidate,
            afternoon_ret=float(last.close) / open_ - 1.0,
            vwap_ret=float(last.close) / max(vwap, 1e-9) - 1.0,
            gap=open_ / prev_close - 1.0,
            rel_volume=volume / max(float(candidate.expected_5m_volume) * len(bars), 1.0),
            close_location=(float(last.close) - low) / range_ if range_ > 0.0 else 0.5,
            open_drawdown=low / open_ - 1.0,
            high_from_open=high / open_ - 1.0,
            low_vs_prev_close=low / prev_close - 1.0,
            range_atr=range_ / max(float(candidate.daily_atr), 1e-9),
            last_close=float(last.close),
            bar_count=len(bars),
            prior_return_5d=_candidate_meta_float(candidate, "return_5d_pct"),
            prior_return_20d=_candidate_meta_float(candidate, "return_20d_pct"),
            prior_return_60d=_candidate_meta_float(candidate, "return_60d_pct"),
            lagged_flow_5d=_candidate_meta_float(candidate, "lagged_flow_5d"),
            lagged_foreign_flow_5d=_candidate_meta_float(candidate, "lagged_foreign_flow_5d"),
            lagged_institutional_flow_5d=_candidate_meta_float(candidate, "lagged_institutional_flow_5d"),
            lagged_flow_z=_candidate_meta_float(candidate, "lagged_flow_z"),
            lagged_foreign_z=_candidate_meta_float(candidate, "lagged_foreign_z"),
            lagged_institutional_z=_candidate_meta_float(candidate, "lagged_institutional_z"),
            lagged_flow_agreement_5d=_candidate_meta_float(candidate, "lagged_flow_agreement_5d"),
            lagged_flow_divergence_5d=_candidate_meta_float(candidate, "lagged_flow_divergence_5d"),
            lagged_sector_flow_5d=_candidate_meta_float(candidate, "sector_flow_5d"),
            lagged_sector_foreign_flow_5d=_candidate_meta_float(candidate, "sector_foreign_flow_5d"),
            lagged_sector_institutional_flow_5d=_candidate_meta_float(candidate, "sector_institutional_flow_5d"),
            intraday_sector_score_pct=sector_feature.score_pct,
            intraday_sector_ret=sector_feature.ret,
            intraday_sector_breadth=sector_feature.breadth,
            intraday_sector_rel_volume=sector_feature.rel_volume,
            intraday_sector_participation=sector_feature.participation,
            intraday_sector_effective_count=sector_feature.effective_count,
            market_score=_candidate_meta_float(candidate, "market_heat_score"),
        )
    return contexts


def afternoon_selection_from_snapshot(
    snapshot: OLRDailySnapshot,
    bars_by_symbol: Mapping[Any, Any],
    config: OLRConfig | None = None,
    sector_map: Mapping[str, str] | None = None,
) -> OLRDailySnapshot:
    """Rerank a stage-1 OLR pool using only completed 09:00-14:25 KST bars."""

    cfg = config or OLRConfig()
    contexts = build_afternoon_contexts(snapshot, bars_by_symbol, cfg, sector_map=sector_map)
    return afternoon_selection_from_contexts(snapshot, contexts, cfg)


def afternoon_selection_from_contexts(
    snapshot: OLRDailySnapshot,
    contexts: Mapping[str, OLRAfternoonContext],
    config: OLRConfig | None = None,
) -> OLRDailySnapshot:
    """Rerank a stage-1 OLR pool from prebuilt causal afternoon contexts."""

    cfg = config or OLRConfig()
    if bool(cfg.shadow_reranker_enabled) and cfg.shadow_reranker_profile:
        return _afternoon_shadow_reranker_selection_from_contexts(snapshot, contexts, cfg)
    final_config_hash = final_candidate_config_fingerprint(cfg)
    rejected: dict[str, list[str]] = {}
    scored: list[tuple[float, OLRDailyCandidate, OLRAfternoonContext]] = []
    score_band_rules = _afternoon_score_band_rules(cfg)
    for candidate in snapshot.candidates:
        ctx = contexts.get(candidate.symbol)
        if ctx is None:
            rejected[candidate.symbol] = ["missing_completed_afternoon_bars"]
            continue
        reasons = _afternoon_reject_reasons(ctx, cfg)
        if reasons:
            rejected[candidate.symbol] = reasons
            continue
        score, raw_score, exhaustion_score = _afternoon_score_details(ctx, cfg)
        if score < float(cfg.afternoon_min_score):
            rejected[candidate.symbol] = ["afternoon_score_below_floor"]
            continue
        if score > float(cfg.afternoon_max_score):
            rejected[candidate.symbol] = ["afternoon_score_above_cap"]
            continue
        if (
            float(cfg.afternoon_reject_score_max) > float(cfg.afternoon_reject_score_min)
            and float(cfg.afternoon_reject_score_min) <= score <= float(cfg.afternoon_reject_score_max)
        ):
            rejected[candidate.symbol] = ["afternoon_score_in_reject_band"]
            continue
        matched_score_band_rule = _matching_afternoon_score_band_rule(score, ctx, score_band_rules, exhaustion_score)
        if score_band_rules and not matched_score_band_rule:
            rejected[candidate.symbol] = ["afternoon_score_band_rule_miss"]
            continue
        rule_features = _afternoon_rule_feature_values(ctx, score, exhaustion_score)
        derived_feature_metadata = {
            key: value
            for key, value in rule_features.items()
            if key.startswith("sector_") or key.startswith("stock_")
        }
        updated_metadata = {
            **dict(candidate.metadata),
            "source": "olr_afternoon_selection",
            "afternoon_selection_version": AFTERNOON_SELECTION_VERSION,
            "intraday_selection_cutoff": INTRADAY_SELECTION_CUTOFF_LABEL,
            "same_day_flow_used": False,
            "official_performance": False,
            "afternoon_score": score,
            "afternoon_score_raw": raw_score,
            "afternoon_score_mode": cfg.afternoon_score_mode,
            "afternoon_score_calibration_mode": cfg.afternoon_score_calibration_mode,
            "afternoon_exhaustion_score": exhaustion_score,
            "afternoon_features": {
                "afternoon_ret": ctx.afternoon_ret,
                "vwap_ret": ctx.vwap_ret,
                "gap": ctx.gap,
                "rel_volume": ctx.rel_volume,
                "close_location": ctx.close_location,
                "open_drawdown": ctx.open_drawdown,
                "high_from_open": ctx.high_from_open,
                "low_vs_prev_close": ctx.low_vs_prev_close,
                "range_atr": ctx.range_atr,
                "last_close": ctx.last_close,
                "bar_count": ctx.bar_count,
                "prior_return_5d": ctx.prior_return_5d,
                "prior_return_20d": ctx.prior_return_20d,
                "prior_return_60d": ctx.prior_return_60d,
                "lagged_flow_5d": ctx.lagged_flow_5d,
                "lagged_foreign_flow_5d": ctx.lagged_foreign_flow_5d,
                "lagged_institutional_flow_5d": ctx.lagged_institutional_flow_5d,
                "lagged_flow_z": ctx.lagged_flow_z,
                "lagged_foreign_z": ctx.lagged_foreign_z,
                "lagged_institutional_z": ctx.lagged_institutional_z,
                "lagged_flow_agreement_5d": ctx.lagged_flow_agreement_5d,
                "lagged_flow_divergence_5d": ctx.lagged_flow_divergence_5d,
                "lagged_sector_flow_5d": ctx.lagged_sector_flow_5d,
                "lagged_sector_foreign_flow_5d": ctx.lagged_sector_foreign_flow_5d,
                "lagged_sector_institutional_flow_5d": ctx.lagged_sector_institutional_flow_5d,
                "sector_intraday_score_pct": ctx.intraday_sector_score_pct,
                "sector_intraday_ret": ctx.intraday_sector_ret,
                "sector_intraday_breadth": ctx.intraday_sector_breadth,
                "sector_intraday_rel_volume": ctx.intraday_sector_rel_volume,
                "sector_intraday_participation": ctx.intraday_sector_participation,
                "sector_intraday_effective_count": ctx.intraday_sector_effective_count,
                "market_score": ctx.market_score,
                **derived_feature_metadata,
            },
        }
        if matched_score_band_rule:
            updated_metadata["afternoon_score_band_rule"] = matched_score_band_rule
        updated = replace(
            candidate,
            selection_score=score,
            metadata=updated_metadata,
        )
        scored.append((score, updated, ctx))
    scored.sort(key=lambda item: (-item[0], item[1].rank or 999, item[1].symbol))
    total = max(1, len(scored))
    selected = tuple(
        replace(candidate, rank=index, rank_pct=(index / total) * 100.0)
        for index, (_, candidate, _) in enumerate(scored[: cfg.afternoon_top_n], start=1)
    )
    return OLRDailySnapshot(
        trade_date=snapshot.trade_date,
        candidates=selected,
        source_fingerprint=snapshot.source_fingerprint,
        generated_at=_afternoon_generated_at(snapshot.trade_date),
        metadata={
            **dict(snapshot.metadata),
            "research_model_version": RESEARCH_MODEL_VERSION,
            "afternoon_selection_version": AFTERNOON_SELECTION_VERSION,
            "candidate_config_hash": final_config_hash,
            "final_candidate_config_hash": final_config_hash,
            "final_candidate_config_hash_version": FINAL_CANDIDATE_CONFIG_HASH_VERSION,
            "source": "olr_afternoon_selection",
            "artifact_stage": OLR_FINAL_ARTIFACT_STAGE,
            "intraday_selection_cutoff": INTRADAY_SELECTION_CUTOFF_LABEL,
            "selection_time_basis": "14:30_decision_from_completed_5m_bars",
            "same_day_daily_ohlcv_visible": False,
            "same_day_daily_flow_visible": False,
            "same_day_flow_used": False,
            "official_performance": False,
            "afternoon_top_n": cfg.afternoon_top_n,
            "selected_symbols": [candidate.symbol for candidate in selected],
            "selected_symbol_count": len(selected),
            "afternoon_candidate_pool_count": len(scored),
            "afternoon_rejected_symbol_count": len(rejected),
            "afternoon_rejected_symbols": rejected,
        },
    )


def _afternoon_shadow_reranker_selection_from_contexts(
    snapshot: OLRDailySnapshot,
    contexts: Mapping[str, OLRAfternoonContext],
    config: OLRConfig,
) -> OLRDailySnapshot:
    final_config_hash = final_candidate_config_fingerprint(config)
    profile = dict(config.shadow_reranker_profile or {})
    baseline_config = replace(config, shadow_reranker_enabled=False, shadow_reranker_profile={})
    baseline_snapshot = afternoon_selection_from_contexts(snapshot, contexts, baseline_config)
    baseline_symbols = [
        candidate.symbol
        for candidate in baseline_snapshot.candidates[: max(1, int(config.overnight_slot_count))]
    ]
    rejected: dict[str, list[str]] = {}
    scored: list[tuple[float, OLRDailyCandidate]] = []
    baseline_score_band_rules = _afternoon_score_band_rules(config)
    score_band_rules = () if config.shadow_reranker_replace_score_band_rules else baseline_score_band_rules
    baseline_selected_count = 0
    for candidate in snapshot.candidates:
        ctx = contexts.get(candidate.symbol)
        if ctx is None:
            rejected[candidate.symbol] = ["missing_completed_afternoon_bars"]
            continue
        reasons = _afternoon_reject_reasons(ctx, config)
        score, raw_score, exhaustion_score = _afternoon_score_details(ctx, config)
        baseline_reasons = list(reasons)
        if score < float(config.afternoon_min_score):
            reasons.append("afternoon_score_below_floor")
            baseline_reasons.append("afternoon_score_below_floor")
        if score > float(config.afternoon_max_score):
            reasons.append("afternoon_score_above_cap")
            baseline_reasons.append("afternoon_score_above_cap")
        if (
            float(config.afternoon_reject_score_max) > float(config.afternoon_reject_score_min)
            and float(config.afternoon_reject_score_min) <= score <= float(config.afternoon_reject_score_max)
        ):
            reasons.append("afternoon_score_in_reject_band")
            baseline_reasons.append("afternoon_score_in_reject_band")
        baseline_matched_score_band_rule = _matching_afternoon_score_band_rule(score, ctx, baseline_score_band_rules, exhaustion_score) if baseline_score_band_rules else ""
        if baseline_score_band_rules and not baseline_matched_score_band_rule:
            baseline_reasons.append("afternoon_score_band_rule_miss")
        if not baseline_reasons:
            baseline_selected_count += 1
        matched_score_band_rule = _matching_afternoon_score_band_rule(score, ctx, score_band_rules, exhaustion_score) if score_band_rules else ""
        if score_band_rules and not matched_score_band_rule:
            reasons.append("afternoon_score_band_rule_miss")
        if reasons:
            rejected[candidate.symbol] = reasons
            continue
        feature_values = _shadow_reranker_feature_values(ctx, score, raw_score, exhaustion_score)
        reranker_score, components = _shadow_reranker_score(feature_values, candidate.sector, profile)
        if reranker_score < float(config.shadow_reranker_min_score):
            rejected[candidate.symbol] = ["shadow_reranker_score_below_floor"]
            continue
        updated = replace(
            candidate,
            selection_score=reranker_score,
            metadata={
                **dict(candidate.metadata),
                "source": "olr_shadow_same_day_reranker",
                "afternoon_selection_version": AFTERNOON_SELECTION_VERSION,
                "intraday_selection_cutoff": INTRADAY_SELECTION_CUTOFF_LABEL,
                "same_day_flow_used": False,
                "official_performance": False,
                "afternoon_score": score,
                "afternoon_score_raw": raw_score,
                "afternoon_score_mode": config.afternoon_score_mode,
                "afternoon_score_calibration_mode": config.afternoon_score_calibration_mode,
                "afternoon_exhaustion_score": exhaustion_score,
                "afternoon_score_band_rule": matched_score_band_rule,
                "shadow_reranker_version": str(profile.get("version") or "olr-shadow-same-day-reranker-v1"),
                "shadow_reranker_profile_hash": str(profile.get("profile_hash") or ""),
                "shadow_reranker_score": reranker_score,
                "shadow_reranker_components": components,
                "shadow_reranker_replaced_score_band_rules": bool(config.shadow_reranker_replace_score_band_rules),
                "afternoon_features": feature_values,
            },
        )
        scored.append((reranker_score, updated))
    scored.sort(key=lambda item: (-item[0], item[1].rank or 999, item[1].symbol))
    scored = _shadow_overlay_replace_weakest_preserve_rank(scored, baseline_symbols, profile)
    total = max(1, len(scored))
    selection_limit = int(config.afternoon_top_n)
    if not bool(profile.get("allow_slot_expansion", False)):
        selection_limit = min(selection_limit, baseline_selected_count)
    selected = tuple(
        replace(candidate, rank=index, rank_pct=(index / total) * 100.0)
        for index, (_, candidate) in enumerate(scored[:selection_limit], start=1)
    )
    return OLRDailySnapshot(
        trade_date=snapshot.trade_date,
        candidates=selected,
        source_fingerprint=snapshot.source_fingerprint,
        generated_at=_afternoon_generated_at(snapshot.trade_date),
        metadata={
            **dict(snapshot.metadata),
            "research_model_version": RESEARCH_MODEL_VERSION,
            "afternoon_selection_version": AFTERNOON_SELECTION_VERSION,
            "candidate_config_hash": final_config_hash,
            "final_candidate_config_hash": final_config_hash,
            "final_candidate_config_hash_version": FINAL_CANDIDATE_CONFIG_HASH_VERSION,
            "source": "olr_shadow_same_day_reranker",
            "artifact_stage": OLR_FINAL_ARTIFACT_STAGE,
            "intraday_selection_cutoff": INTRADAY_SELECTION_CUTOFF_LABEL,
            "selection_time_basis": "14:30_decision_from_completed_5m_bars",
            "same_day_daily_ohlcv_visible": False,
            "same_day_daily_flow_visible": False,
            "same_day_flow_used": False,
            "official_performance": False,
            "afternoon_top_n": config.afternoon_top_n,
            "selected_symbols": [candidate.symbol for candidate in selected],
            "selected_symbol_count": len(selected),
            "afternoon_candidate_pool_count": len(scored),
            "afternoon_rejected_symbol_count": len(rejected),
            "afternoon_rejected_symbols": rejected,
            "shadow_reranker_version": str(profile.get("version") or "olr-shadow-same-day-reranker-v1"),
            "shadow_reranker_profile_hash": str(profile.get("profile_hash") or ""),
            "shadow_reranker_replaced_score_band_rules": bool(config.shadow_reranker_replace_score_band_rules),
            "shadow_reranker_slot_policy": str(profile.get("slot_policy") or "replace_existing_trade_slots"),
            "shadow_reranker_baseline_selected_count": baseline_selected_count,
            "shadow_reranker_baseline_symbols": baseline_symbols,
        },
    )


def _shadow_overlay_replace_weakest_preserve_rank(
    scored: list[tuple[float, OLRDailyCandidate]],
    baseline_symbols: list[str],
    profile: Mapping[str, Any],
) -> list[tuple[float, OLRDailyCandidate]]:
    if str(profile.get("slot_policy") or "") == "free_rerank":
        return scored
    by_symbol = {candidate.symbol: (score, candidate) for score, candidate in scored}
    selected = [by_symbol[symbol] for symbol in baseline_symbols if symbol in by_symbol]
    if not selected:
        return []
    selected_symbols = {candidate.symbol for _, candidate in selected}
    shadows = [(score, candidate) for score, candidate in scored if candidate.symbol not in selected_symbols]
    max_replacements = max(0, int(profile.get("max_replacements_per_day", 1) or 1))
    margin = float(profile.get("replacement_margin", 0.0) or 0.0)
    replacements = 0
    for shadow_score, shadow_candidate in shadows:
        if replacements >= max_replacements:
            break
        weakest_index, (weakest_score, weakest_candidate) = min(
            enumerate(selected),
            key=lambda item: (
                float(item[1][0]),
                -int(item[1][1].rank or 999),
                item[1][1].symbol,
            ),
        )
        if float(shadow_score) <= float(weakest_score) + margin:
            break
        selected[weakest_index] = (shadow_score, shadow_candidate)
        replacements += 1
    return selected


def _afternoon_generated_at(trade_date: date) -> datetime:
    return datetime.combine(trade_date, INTRADAY_SELECTION_CUTOFF, tzinfo=KST)


def run_daily_selection(
    daily_by_symbol: dict[str, list[dict[str, Any]]] | OLRResearchSnapshot,
    trade_date: date | None = None,
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
    """Build and persist an OLR daily artifact from rows or a research snapshot.

    Live artifact generators can fetch/build today's ``OLRResearchSnapshot`` and
    pass it here. Replay adapters can pass cached rows and let this helper build
    the snapshot first. In both cases the selection converges on
    ``daily_selection_from_snapshot()``.
    """

    cfg = config or OLRConfig()
    if isinstance(daily_by_symbol, OLRResearchSnapshot):
        research_snapshot = daily_by_symbol
    else:
        if trade_date is None:
            raise ValueError("OLR run_daily_selection requires trade_date when building from rows")
        research_snapshot = build_research_snapshot(
            daily_by_symbol,
            trade_date,
            cfg,
            sector_map=sector_map,
            flow_by_symbol=flow_by_symbol,
            foreign_flow_by_symbol=foreign_flow_by_symbol,
            institutional_flow_by_symbol=institutional_flow_by_symbol,
            index_ohlcv_by_symbol=index_ohlcv_by_symbol,
            source_fingerprint=source_fingerprint,
            generated_at=generated_at,
        )
    candidate_snapshot = daily_selection_from_snapshot(research_snapshot, cfg)
    if artifact_root is not None:
        OLRArtifactStore(artifact_root).save_snapshot(candidate_snapshot, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
    if lrs is not None:
        save_snapshot_to_lrs(candidate_snapshot, lrs, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
    return candidate_snapshot


def run_afternoon_selection(
    candidate_snapshot: OLRDailySnapshot,
    bars_by_symbol: Mapping[Any, Any],
    *,
    config: OLRConfig | None = None,
    sector_map: Mapping[str, str] | None = None,
    artifact_root: str | Path | None = "data/strategy/olr",
    lrs=None,
) -> OLRDailySnapshot:
    """Build and persist the final 14:30 OLR artifact via the shared selector."""

    selected = afternoon_selection_from_snapshot(candidate_snapshot, bars_by_symbol, config or OLRConfig(), sector_map=sector_map)
    if artifact_root is not None:
        OLRArtifactStore(artifact_root).save_snapshot(selected, artifact_stage=OLR_FINAL_ARTIFACT_STAGE)
    if lrs is not None:
        save_snapshot_to_lrs(selected, lrs, artifact_stage=OLR_FINAL_ARTIFACT_STAGE)
    return selected


def _reject_reasons(
    symbol: OLRResearchSymbol,
    snapshot: OLRResearchSnapshot,
    config: OLRConfig,
    *,
    sector_daily_feature: SectorDailyFeature | None = None,
) -> list[str]:
    reasons: list[str] = []
    if len(symbol.daily_rows) < config.research_min_history_days:
        reasons.append("insufficient_history")
    if symbol.price < config.research_min_price_krw:
        reasons.append("price_below_floor")
    if symbol.adv20_krw < config.research_min_adv20_krw:
        reasons.append("adv20_below_floor")
    if symbol.adv20_krw < config.premarket_min_adv20_krw:
        reasons.append("premarket_adv20_below_floor")
    if _bad_recent_ohlcv(symbol.daily_rows[-20:]):
        reasons.append("bad_recent_ohlcv")
    if symbol.etf_flag:
        reasons.append("etf")
    if symbol.preferred_flag:
        reasons.append("preferred")
    if symbol.otc_flag:
        reasons.append("otc")
    if symbol.hard_to_borrow_flag:
        reasons.append("hard_to_borrow")
    if symbol.blacklist_flag:
        reasons.append("blacklisted")
    if symbol.halted_flag:
        reasons.append("halted")
    if symbol.severe_news_flag:
        reasons.append("severe_news")
    if config.research_require_spread and symbol.median_spread_pct is None:
        reasons.append("missing_spread")
    if symbol.median_spread_pct is not None and symbol.median_spread_pct > config.research_max_median_spread_pct:
        reasons.append("spread_above_cap")
    if symbol.return_20d_pct < config.min_parent_20d_return_pct or symbol.return_20d_pct > config.max_parent_20d_return_pct:
        reasons.append("parent_20d_return_outside_bounds")
    if snapshot.market.breadth_pct_above_20dma < config.min_market_breadth_pct:
        reasons.append("market_breadth_below_floor")
    if snapshot.market.market_heat_score < config.min_market_heat_score:
        reasons.append("market_heat_below_floor")
    if symbol.flow_5d < config.research_min_flow_5d:
        reasons.append("lagged_flow_below_floor")
    if symbol.foreign_flow_5d < config.research_min_foreign_flow_5d:
        reasons.append("lagged_foreign_flow_below_floor")
    if symbol.institutional_flow_5d < config.research_min_institutional_flow_5d:
        reasons.append("lagged_institutional_flow_below_floor")
    if symbol.flow_z < config.research_min_flow_z:
        reasons.append("lagged_flow_z_below_floor")
    if symbol.flow_agreement_5d < config.research_min_flow_agreement:
        reasons.append("lagged_flow_agreement_below_floor")
    if symbol.flow_divergence_5d > config.research_max_flow_divergence:
        reasons.append("lagged_flow_divergence_above_cap")
    if symbol.foreign_flow_z < config.premarket_min_foreign5_z:
        reasons.append("premarket_foreign5_z_below_floor")
    sector = snapshot.sectors.get(symbol.sector)
    use_daily_flow_gates = bool(config.research_use_sector_daily_flow_gates and sector_daily_feature is not None)
    sector_flow_5d = sector_daily_feature.flow_5d if use_daily_flow_gates else (sector.flow_5d if sector is not None else 0.0)
    sector_foreign_flow_5d = sector_daily_feature.foreign_flow_5d if use_daily_flow_gates else (sector.foreign_flow_5d if sector is not None else 0.0)
    sector_institutional_flow_5d = sector_daily_feature.institutional_flow_5d if use_daily_flow_gates else (sector.institutional_flow_5d if sector is not None else 0.0)
    sector_flow_agreement_5d = sector_daily_feature.flow_agreement_5d if use_daily_flow_gates else (sector.flow_agreement_5d if sector is not None else 0.0)
    if sector is not None and sector_flow_5d < config.research_min_sector_flow_5d:
        reasons.append("lagged_sector_flow_below_floor")
    if sector is not None and sector_foreign_flow_5d < config.research_min_sector_foreign_flow_5d:
        reasons.append("lagged_sector_foreign_flow_below_floor")
    if sector is not None and sector_institutional_flow_5d < config.research_min_sector_institutional_flow_5d:
        reasons.append("lagged_sector_institutional_flow_below_floor")
    if sector is not None and sector_flow_agreement_5d < config.research_min_sector_flow_agreement:
        reasons.append("lagged_sector_flow_agreement_below_floor")
    return reasons


def _score_reject_reasons(details: dict[str, float], config: OLRConfig) -> list[str]:
    reasons: list[str] = []
    if details["relative_strength_pct"] < config.research_min_rs_percentile:
        reasons.append("rs_below_floor")
    if details["trend_score"] < config.research_min_trend_score:
        reasons.append("trend_below_floor")
    if details["compression_score"] < config.research_min_compression_score:
        reasons.append("compression_below_floor")
    if details["accumulation_raw"] < config.research_min_accumulation_score:
        reasons.append("accumulation_below_floor")
    if details["sector_participation"] < config.research_min_sector_participation:
        reasons.append("sector_participation_below_floor")
    if details.get("sector_daily_score_pct", 50.0) < config.research_min_sector_daily_score_pct:
        reasons.append("sector_daily_score_below_floor")
    if config.research_max_box_range_pct > 0.0 and details["box_range_pct"] > config.research_max_box_range_pct:
        reasons.append("box_range_above_cap")
    if details["daily_signal_score"] < config.daily_signal_min_score:
        reasons.append("score_below_daily_min")
    if details["daily_signal_score"] < config.daily_rescue_min_score:
        reasons.append("score_below_daily_rescue_min")
    if config.signal_floor > 0.0 and details["daily_signal_score"] < config.signal_floor:
        reasons.append("score_below_signal_floor")
    if details["daily_signal_score"] > config.daily_signal_max_score:
        reasons.append("score_above_daily_max")
    flow_policy = str(config.flow_policy or "").lower()
    if flow_policy in {"require_positive", "strict_positive", "positive_only"} and details.get("flow_score_raw", 0.0) < 0.0:
        reasons.append("flow_policy_requires_positive_flow")
    if details.get("cdd", 0.0) > config.cdd_max:
        reasons.append("cdd_too_extended")
    if abs(details.get("gap_pct", 0.0)) > config.gap_max_pct:
        reasons.append("gap_out_of_range")
    if details.get("relative_strength_pct", 0.0) < config.min_relative_strength_pct:
        reasons.append("relative_strength_below_floor")
    if details.get("relative_strength_pct", 0.0) > config.max_relative_strength_pct:
        reasons.append("relative_strength_above_ceiling")
    if details.get("trend_broken", 0.0) > 0.0:
        reasons.append("trend_broken")
    if details.get("secular_trend", 0.0) > 0.0 and not config.allow_secular:
        reasons.append("secular_trend_disabled")
    if details.get("trigger_count", 0.0) <= 0.0:
        reasons.append("no_pullback_trigger")
    return reasons


def _long_score(
    symbol: OLRResearchSymbol,
    snapshot: OLRResearchSnapshot,
    rs_percentile: float,
    rs60_percentile: float,
    sector_strength_percentile: float,
    config: OLRConfig,
    *,
    sector_daily_feature: SectorDailyFeature | None = None,
) -> tuple[float, dict[str, float]]:
    sector = snapshot.sectors.get(symbol.sector)
    trend = _trend_score(symbol.daily_rows)
    compression, box_range_pct = _compression_score(symbol.daily_rows)
    accumulation = _accumulation_score(symbol.daily_rows)
    stock_regime = _stock_regime_score(symbol.daily_rows)
    sector_regime = (
        float(sector_daily_feature.score_pct)
        if config.research_use_sector_daily_regime_score and sector_daily_feature is not None
        else _sector_regime_score(sector)
    )
    participation = float(
        sector_daily_feature.participation
        if config.research_use_sector_daily_participation and sector_daily_feature is not None
        else (sector.participation if sector is not None else 0.0)
    )
    participation_score = 100.0 * participation
    daily_signal, trigger_count, signal_details = _daily_signal_score(
        symbol,
        rs_percentile,
        rs60_percentile,
        sector_strength_percentile,
        snapshot.market.market_heat_score,
        config,
    )
    flow_score = _flow_score(symbol.flow_5d, symbol.flow_z)
    foreign_score = _flow_score(symbol.foreign_flow_5d, 0.0)
    inst_score = _flow_score(symbol.institutional_flow_5d, 0.0)
    agreement_score = 100.0 * _clip01((symbol.flow_agreement_5d + 1.0) / 2.0)
    weights = {
        "relative_strength_score": max(config.research_weight_relative_strength, 0.0),
        "trend_score": max(config.research_weight_daily_trend, 0.0),
        "compression_score": max(config.research_weight_compression, 0.0),
        "accumulation_score": max(config.research_weight_accumulation, 0.0),
        "stock_regime_score": max(config.research_weight_stock_regime, 0.0),
        "sector_regime_score": max(config.research_weight_sector_regime, 0.0),
        "sector_participation_score": max(config.research_weight_sector_participation, 0.0),
        "daily_signal_score": max(config.research_weight_daily_signal, 0.0),
        "flow_score": max(config.research_weight_flow, 0.0),
        "foreign_flow_score": max(config.research_weight_foreign_flow, 0.0),
        "institutional_flow_score": max(config.research_weight_institutional_flow, 0.0),
        "flow_agreement_score": max(config.research_weight_flow_agreement, 0.0),
    }
    values = {
        "relative_strength_score": rs_percentile,
        "trend_score": trend,
        "compression_score": compression,
        "accumulation_score": 50.0 + 50.0 * accumulation,
        "stock_regime_score": stock_regime,
        "sector_regime_score": sector_regime,
        "sector_participation_score": participation_score,
        "daily_signal_score": daily_signal,
        "flow_score": flow_score,
        "foreign_flow_score": foreign_score,
        "institutional_flow_score": inst_score,
        "flow_agreement_score": agreement_score,
    }
    total_weight = max(sum(weights.values()), 1e-9)
    raw_score = sum(values[key] * weights[key] for key in values) / total_weight
    details = {
        **values,
        "relative_strength_pct": rs_percentile,
        "relative_strength_60d_pct": rs60_percentile,
        "sector_strength_pct": sector_strength_percentile,
        "accumulation_raw": accumulation,
        "box_range_pct": box_range_pct,
        "sector_participation": participation,
        "flow_score_raw": symbol.flow_5d,
        "foreign_flow_5d": symbol.foreign_flow_5d,
        "institutional_flow_5d": symbol.institutional_flow_5d,
        "flow_z": symbol.flow_z,
        "flow_agreement_5d": symbol.flow_agreement_5d,
        "flow_divergence_5d": symbol.flow_divergence_5d,
        "trigger_count": float(trigger_count),
        "weighted_score_raw": raw_score,
        **(sector_daily_feature.metadata() if sector_daily_feature is not None else {}),
        **signal_details,
    }
    score_multiplier = _selection_score_multiplier(details, config)
    score = raw_score * score_multiplier
    details["selection_score_multiplier"] = score_multiplier
    details["weighted_score"] = score
    return score, details


def _selection_score_multiplier(details: dict[str, float], config: OLRConfig) -> float:
    multiplier = 1.0
    daily_signal = float(details.get("daily_signal_score", 0.0))
    if 0.0 <= daily_signal < float(config.daily_signal_min_score) and daily_signal >= float(config.daily_rescue_min_score):
        multiplier *= max(0.0, float(config.rescue_size_mult))
    if details.get("secular_trend", 0.0) > 0.0:
        multiplier *= max(0.0, float(config.secular_sizing_mult))
    if config.structure_sizing_enabled:
        structural = max(0.0, min(float(details.get("structural_score", 0.0)) / 100.0, 1.0))
        structure_mult = 0.70 + 0.60 * structural
        multiplier *= max(float(config.structure_size_mult_min), min(float(config.structure_size_mult_max), structure_mult))
    flow_policy = str(config.flow_policy or "").lower()
    if flow_policy in {"soft_penalty_rescue", "soft_penalty"} and details.get("flow_score_raw", 0.0) < 0.0:
        multiplier *= max(0.0, float(config.rescue_size_mult))
    return multiplier


def _daily_signal_score(
    symbol: OLRResearchSymbol,
    rs_percentile: float,
    rs60_percentile: float,
    sector_strength_percentile: float,
    market_heat_score: float,
    config: OLRConfig,
) -> tuple[float, int, dict[str, float]]:
    closes = [_float(row.get("close")) for row in symbol.daily_rows]
    volumes = [_float(row.get("volume")) for row in symbol.daily_rows]
    if len(closes) < 20:
        return 0.0, 0, {"trigger_count": 0.0}
    prev_close = closes[-1]
    sma20 = _mean(closes[-20:])
    sma60 = _mean(closes[-60:]) if len(closes) >= 60 else sma20
    atr14 = max(_atr(list(symbol.daily_rows), 14), prev_close * 0.01)
    rsi2 = _rsi(closes, 2)
    rsi5 = _rsi(closes, 5)
    cdd = _consecutive_down_days(closes)
    gap_pct = ((float(symbol.daily_rows[-1].get("open", prev_close)) / max(_float(symbol.daily_rows[-2].get("close")), 1e-9)) - 1.0) * 100.0 if len(symbol.daily_rows) >= 2 else 0.0
    depth_atr = _pullback_depth_atr(symbol.daily_rows, prev_close, atr14)
    bb_pctb = _bollinger_pctb(closes)
    volume_climax = _volume_climax_ratio(volumes)
    roc5 = _return_pct(symbol.daily_rows, 5)
    trend_tier = _classify_trend(prev_close=prev_close, sma20=sma20, sma60=sma60)
    trigger_weights: dict[str, float] = {}
    if rsi2 < config.rsi2_trigger_thresh:
        trigger_weights["RSI2"] = 20.0
    if rsi5 < config.rsi5_trigger_thresh and cdd >= config.cdd_min_for_rsi5:
        trigger_weights["RSI5_CDD"] = 18.0
    if depth_atr > config.depth_atr_trigger:
        trigger_weights["DEPTH"] = 16.0
    if bb_pctb < config.bb_pctb_trigger:
        trigger_weights["BB_PCTB"] = 14.0
    if volume_climax > config.volume_climax_trigger:
        trigger_weights["VOL_CLIMAX"] = 12.0
    if rs_percentile >= config.relative_strength_trigger_pct:
        trigger_weights["RS_STRONG"] = 10.0
    elif roc5 <= config.roc5_drop_trigger_pct:
        trigger_weights["ROC5_DROP"] = 10.0
    if gap_pct <= config.gap_down_trigger_pct:
        trigger_weights["GAP_DOWN"] = 10.0

    base_score = min(100.0, sum(trigger_weights.values()))
    structural_score = _kiaric_structural_score(
        symbol,
        rs_percentile=rs_percentile,
        rs60_percentile=rs60_percentile,
        sector_strength_percentile=sector_strength_percentile,
        market_heat_score=market_heat_score,
        trend_tier=trend_tier,
    )
    structure_weight = max(0.0, min(float(config.daily_structure_weight), 0.80))
    score = (1.0 - structure_weight) * base_score + structure_weight * structural_score
    return max(0.0, min(100.0, score)), len(trigger_weights), {
        "base_pullback_score": base_score,
        "discrete_trigger_score": base_score,
        "structural_score": structural_score,
        "rsi2": rsi2,
        "rsi5": rsi5,
        "cdd": float(cdd),
        "gap_pct": gap_pct,
        "depth_atr": depth_atr,
        "bb_pctb": bb_pctb,
        "volume_climax_ratio": volume_climax,
        "roc5_pct": roc5,
        "relative_strength_60d_pct": rs60_percentile,
        "sector_strength_pct": sector_strength_percentile,
        "trend_broken": 1.0 if trend_tier == "BROKEN" else 0.0,
        "secular_trend": 1.0 if trend_tier == "SECULAR" else 0.0,
        "trigger_count": float(len(trigger_weights)),
    }


def _daily_sort_key(candidate: OLRDailyCandidate, config: OLRConfig) -> tuple[Any, ...]:
    mode = str(config.frontier_active_selection_mode or "hybrid").lower()
    score = float(candidate.selection_score)
    adv = _float(candidate.metadata.get("adv20_krw"))
    rs = float(candidate.rs_percentile)
    flow = float(candidate.flow_score)
    ret5 = _float(candidate.metadata.get("return_5d_pct"))
    volume = _float((candidate.metadata.get("research_score_components") or {}).get("volume_climax_ratio"), 1.0)
    if mode in {"score", "research_score", "selection_score"}:
        return (-score, -adv, -rs, candidate.symbol)
    if mode == "liquidity":
        return (-adv, -score, -rs, candidate.symbol)
    if mode == "campaign":
        campaign = 0.60 * score + 0.20 * rs + 0.20 * max(flow, 0.0)
        return (-campaign, -score, -adv, candidate.symbol)
    if mode == "hot":
        hot = 0.55 * score + 12.0 * max(ret5, 0.0) + 3.0 * min(volume, 5.0)
        return (-hot, -score, -adv, candidate.symbol)
    return (-(0.70 * score + 0.30 * min(adv / 10_000_000_000.0, 100.0)), -score, -adv, candidate.symbol)


def _kiaric_structural_score(
    symbol: OLRResearchSymbol,
    *,
    rs_percentile: float,
    rs60_percentile: float,
    sector_strength_percentile: float,
    market_heat_score: float,
    trend_tier: str,
) -> float:
    closes = [_float(row.get("close")) for row in symbol.daily_rows]
    sma20 = _mean(closes[-20:])
    sma60 = _mean(closes[-60:]) if len(closes) >= 60 else sma20
    trend_spread = ((sma20 - sma60) / sma60 * 100.0) if sma60 > 0 else 0.0
    trend = 55.0 + 2.5 * symbol.return_20d_pct + 1.5 * trend_spread
    if trend_tier == "STRONG":
        trend += 12.0
    elif trend_tier == "ACCEPTABLE":
        trend += 6.0
    elif trend_tier == "SECULAR":
        trend -= 4.0
    trend = max(0.0, min(trend, 100.0))
    volume = max(0.0, min(symbol.volume_ratio_20d / 2.0, 1.0)) * 100.0
    heat = max(0.0, min(market_heat_score, 100.0))
    rel = max(0.0, min(rs_percentile, 100.0))
    rel60 = max(0.0, min(rs60_percentile, 100.0))
    sector = max(0.0, min(sector_strength_percentile, 100.0))
    return max(0.0, min(100.0, 0.30 * rel + 0.15 * rel60 + 0.15 * sector + 0.20 * trend + 0.10 * volume + 0.10 * heat))


def _classify_trend(*, prev_close: float, sma20: float, sma60: float) -> str:
    if prev_close <= 0 or sma20 <= 0 or sma60 <= 0:
        return "UNKNOWN"
    if prev_close >= sma20 >= sma60:
        return "STRONG"
    if prev_close >= sma60:
        return "ACCEPTABLE"
    if prev_close >= sma60 * 0.96:
        return "SECULAR"
    return "BROKEN"


def _pullback_depth_atr(rows: tuple[dict[str, Any], ...], prev_close: float, atr14: float, lookback: int = 10) -> float:
    if atr14 <= 0:
        return 0.0
    window = rows[-lookback:] if len(rows) >= lookback else rows
    recent_high = max((_float(row.get("high"), prev_close) for row in window), default=prev_close)
    return max(0.0, (recent_high - prev_close) / atr14)


def _bollinger_pctb(closes: list[float], period: int = 20, stdev_mult: float = 2.0) -> float:
    if len(closes) < period:
        return 0.5
    window = [float(value) for value in closes[-period:]]
    mean_value = _mean(window)
    stdev = _std(window)
    upper = mean_value + stdev_mult * stdev
    lower = mean_value - stdev_mult * stdev
    width = upper - lower
    return (float(closes[-1]) - lower) / width if width > 0 else 0.5


def _volume_climax_ratio(volumes: list[float], period: int = 20) -> float:
    if len(volumes) < 2:
        return 0.0
    window = [max(float(value), 0.0) for value in volumes[-period:]]
    baseline = _mean(window[:-1])
    return window[-1] / baseline if baseline > 0 else 0.0


def _consecutive_down_days(closes: list[float]) -> int:
    count = 0
    for index in range(len(closes) - 1, 0, -1):
        if closes[index] < closes[index - 1]:
            count += 1
            continue
        break
    return count


def _rsi(closes: list[float], period: int) -> float:
    if len(closes) <= period:
        return 50.0
    gains: list[float] = []
    losses: list[float] = []
    for index in range(len(closes) - period, len(closes)):
        change = closes[index] - closes[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = _mean(gains)
    avg_loss = _mean(losses)
    if avg_loss <= 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _afternoon_reject_reasons(ctx: OLRAfternoonContext, config: OLRConfig) -> list[str]:
    reasons: list[str] = []
    sector = _normalize_sector(ctx.candidate.sector)
    blocked_sectors = _sector_set(config.afternoon_blocked_sectors)
    allowed_sectors = _sector_set(config.afternoon_allowed_sectors)
    if blocked_sectors and sector in blocked_sectors:
        reasons.append("afternoon_sector_blocked")
    if allowed_sectors and sector not in allowed_sectors:
        reasons.append("afternoon_sector_not_allowed")
    if ctx.afternoon_ret < config.afternoon_min_ret:
        reasons.append("afternoon_ret_below_floor")
    if ctx.afternoon_ret > config.afternoon_max_ret:
        reasons.append("afternoon_ret_above_cap")
    if ctx.vwap_ret < config.afternoon_min_vwap_ret:
        reasons.append("vwap_ret_below_floor")
    if ctx.vwap_ret > config.afternoon_max_vwap_ret:
        reasons.append("vwap_ret_above_cap")
    if ctx.gap < config.afternoon_min_gap or ctx.gap > config.afternoon_max_gap:
        reasons.append("gap_outside_bounds")
    if ctx.rel_volume < config.afternoon_min_rel_volume:
        reasons.append("rel_volume_below_floor")
    if ctx.close_location < config.afternoon_min_close_location:
        reasons.append("close_location_below_floor")
    if abs(min(ctx.open_drawdown, 0.0)) > config.afternoon_max_open_drawdown:
        reasons.append("open_drawdown_above_cap")
    if ctx.range_atr > config.afternoon_max_range_atr:
        reasons.append("range_atr_above_cap")
    if ctx.high_from_open < config.afternoon_min_high_from_open:
        reasons.append("high_from_open_below_floor")
    if ctx.low_vs_prev_close < config.afternoon_min_low_vs_prev_close:
        reasons.append("low_vs_prev_close_below_floor")
    if ctx.bar_count < config.afternoon_min_bar_count:
        reasons.append("afternoon_bar_count_below_floor")
    if ctx.prior_return_5d < config.afternoon_min_prior_ret5:
        reasons.append("prior_ret5_below_floor")
    if ctx.prior_return_20d < config.afternoon_min_prior_ret20 or ctx.prior_return_20d > config.afternoon_max_prior_ret20:
        reasons.append("prior_ret20_outside_bounds")
    if ctx.prior_return_60d < config.afternoon_min_prior_ret60:
        reasons.append("prior_ret60_below_floor")
    if ctx.lagged_flow_5d < config.afternoon_min_flow_5d:
        reasons.append("lagged_flow_5d_below_floor")
    if ctx.lagged_foreign_flow_5d < config.afternoon_min_foreign_flow_5d:
        reasons.append("lagged_foreign_flow_5d_below_floor")
    if ctx.lagged_institutional_flow_5d < config.afternoon_min_institutional_flow_5d:
        reasons.append("lagged_institutional_flow_5d_below_floor")
    if ctx.lagged_flow_z < config.afternoon_min_flow_z:
        reasons.append("lagged_flow_z_below_floor")
    if ctx.lagged_foreign_z < config.afternoon_min_foreign_z:
        reasons.append("lagged_foreign_z_below_floor")
    if ctx.lagged_institutional_z < config.afternoon_min_institutional_z:
        reasons.append("lagged_institutional_z_below_floor")
    if ctx.lagged_flow_agreement_5d < config.afternoon_min_flow_agreement:
        reasons.append("lagged_flow_agreement_below_floor")
    if ctx.lagged_flow_divergence_5d > config.afternoon_max_flow_divergence:
        reasons.append("lagged_flow_divergence_above_cap")
    if ctx.lagged_sector_flow_5d < config.afternoon_min_sector_flow:
        reasons.append("lagged_sector_flow_below_floor")
    if ctx.lagged_sector_foreign_flow_5d < config.afternoon_min_sector_foreign_flow:
        reasons.append("lagged_sector_foreign_flow_below_floor")
    if ctx.lagged_sector_institutional_flow_5d < config.afternoon_min_sector_institutional_flow:
        reasons.append("lagged_sector_institutional_flow_below_floor")
    if ctx.intraday_sector_score_pct < config.afternoon_min_intraday_sector_score_pct:
        reasons.append("intraday_sector_score_below_floor")
    if ctx.market_score < config.afternoon_min_market_score:
        reasons.append("market_score_below_floor")
    if config.afternoon_require_close_above_prev and ctx.last_close <= ctx.candidate.prior_day_close:
        reasons.append("close_not_above_prev_close")
    if float(ctx.candidate.flow_score) < config.afternoon_min_lagged_flow_score:
        reasons.append("lagged_flow_score_below_floor")
    if _afternoon_exhaustion_score(ctx) > config.afternoon_max_exhaustion_score:
        reasons.append("afternoon_exhaustion_above_cap")
    mode = str(config.afternoon_score_mode or "hybrid").lower()
    if mode in {"momentum", "hybrid", "efficient", "flow_confirmed", "daily_plus_intraday"} and ctx.afternoon_ret < 0.0:
        reasons.append("mode_requires_nonnegative_afternoon_ret")
    if mode in {"vwap_strength", "flow_confirmed"} and ctx.vwap_ret < 0.0:
        reasons.append("mode_requires_nonnegative_vwap_ret")
    if mode == "gap_hold" and (ctx.gap < 0.002 or ctx.afternoon_ret < 0.0 or ctx.low_vs_prev_close < -0.02):
        reasons.append("gap_hold_shape_failed")
    if mode == "flow_confirmed" and not (ctx.lagged_flow_5d > 0.0 or ctx.lagged_flow_z > 0.0 or ctx.lagged_sector_flow_5d > 0.0):
        reasons.append("flow_confirmed_without_lagged_confirmation")
    return reasons


def _sector_set(values: Any) -> set[str]:
    if values in (None, "", ()):
        return set()
    if isinstance(values, str):
        raw = values.replace(";", ",").split(",")
    else:
        try:
            raw = list(values)
        except TypeError:
            raw = [values]
    return {_normalize_sector(item) for item in raw if _normalize_sector(item)}


def _afternoon_score_band_rules(config: OLRConfig) -> tuple[dict[str, Any], ...]:
    raw = getattr(config, "afternoon_score_band_rules", ()) or ()
    if isinstance(raw, Mapping):
        raw = (raw,)
    rules: list[dict[str, Any]] = []
    for rule in tuple(raw):
        if isinstance(rule, Mapping):
            rules.append(dict(rule))
    return tuple(rules)


def _matching_afternoon_score_band_rule(
    score: float,
    ctx: OLRAfternoonContext,
    rules: tuple[dict[str, Any], ...],
    exhaustion_score: float,
) -> str:
    for index, rule in enumerate(rules, start=1):
        if _afternoon_score_band_rule_matches(score, ctx, rule, exhaustion_score):
            return str(rule.get("name") or f"rule_{index}")
    return ""


def _afternoon_score_band_rule_matches(
    score: float,
    ctx: OLRAfternoonContext,
    rule: Mapping[str, Any],
    exhaustion_score: float,
) -> bool:
    feature_values = _afternoon_rule_feature_values(ctx, score, exhaustion_score)
    if "min_score" in rule and score < float(rule["min_score"]):
        return False
    if "max_score" in rule and score > float(rule["max_score"]):
        return False
    rank = int(ctx.candidate.rank or 999)
    if "min_rank" in rule and rank < int(rule["min_rank"]):
        return False
    if "max_rank" in rule and rank > int(rule["max_rank"]):
        return False
    sector = _normalize_sector(ctx.candidate.sector)
    allowed_sectors = _sector_set(rule.get("allowed_sectors"))
    if allowed_sectors and sector not in allowed_sectors:
        return False
    blocked_sectors = _sector_set(rule.get("blocked_sectors"))
    if blocked_sectors and sector in blocked_sectors:
        return False
    if "sector_admission" in rule and not _sector_admission_rule_matches(feature_values, rule.get("sector_admission")):
        return False
    min_checks = {
        "min_afternoon_ret": ctx.afternoon_ret,
        "min_vwap_ret": ctx.vwap_ret,
        "min_gap": ctx.gap,
        "min_rel_volume": ctx.rel_volume,
        "min_close_location": ctx.close_location,
        "min_high_from_open": ctx.high_from_open,
        "min_low_vs_prev_close": ctx.low_vs_prev_close,
        "min_prior_ret5": ctx.prior_return_5d,
        "min_prior_ret20": ctx.prior_return_20d,
        "min_prior_ret60": ctx.prior_return_60d,
        "min_flow_5d": ctx.lagged_flow_5d,
        "min_foreign_flow_5d": ctx.lagged_foreign_flow_5d,
        "min_institutional_flow_5d": ctx.lagged_institutional_flow_5d,
        "min_flow_z": ctx.lagged_flow_z,
        "min_foreign_z": ctx.lagged_foreign_z,
        "min_institutional_z": ctx.lagged_institutional_z,
        "min_flow_agreement": ctx.lagged_flow_agreement_5d,
        "min_sector_flow": ctx.lagged_sector_flow_5d,
        "min_sector_foreign_flow": ctx.lagged_sector_foreign_flow_5d,
        "min_sector_institutional_flow": ctx.lagged_sector_institutional_flow_5d,
        "min_intraday_sector_score_pct": ctx.intraday_sector_score_pct,
        "min_market_score": ctx.market_score,
        "min_lagged_flow_score": ctx.candidate.flow_score,
    }
    min_checks.update({f"min_{key}": value for key, value in feature_values.items()})
    for key, value in min_checks.items():
        if key in rule and float(value) < float(rule[key]):
            return False
    max_checks = {
        "max_afternoon_ret": ctx.afternoon_ret,
        "max_vwap_ret": ctx.vwap_ret,
        "max_gap": ctx.gap,
        "max_range_atr": ctx.range_atr,
        "max_prior_ret20": ctx.prior_return_20d,
        "max_flow_divergence": ctx.lagged_flow_divergence_5d,
        "max_exhaustion_score": exhaustion_score,
    }
    max_checks.update({f"max_{key}": value for key, value in feature_values.items()})
    for key, value in max_checks.items():
        if key in rule and float(value) > float(rule[key]):
            return False
    for key, threshold in _rule_feature_thresholds(rule, "min_features").items():
        if feature_values.get(key, _candidate_meta_float(ctx.candidate, key)) < threshold:
            return False
    for key, threshold in _rule_feature_thresholds(rule, "max_features").items():
        if feature_values.get(key, _candidate_meta_float(ctx.candidate, key)) > threshold:
            return False
    if "max_open_drawdown" in rule and abs(min(ctx.open_drawdown, 0.0)) > float(rule["max_open_drawdown"]):
        return False
    if bool(rule.get("require_close_above_prev")) and ctx.last_close <= ctx.candidate.prior_day_close:
        return False
    return True


def _sector_admission_rule_matches(feature_values: Mapping[str, float], raw: Any) -> bool:
    if raw in (None, False):
        return True
    if not isinstance(raw, Mapping):
        return False
    mode = str(raw.get("mode") or "dynamic_confirmed_rotation")
    if mode not in {"dynamic_confirmed_rotation"}:
        return False
    if not _feature_thresholds_match(feature_values, raw):
        return False
    return True


def _feature_thresholds_match(feature_values: Mapping[str, float], thresholds: Mapping[str, Any]) -> bool:
    for key, value in thresholds.items():
        if key.startswith("min_"):
            feature = key[4:]
            if float(feature_values.get(feature, 0.0)) < float(value):
                return False
        elif key.startswith("max_"):
            feature = key[4:]
            if float(feature_values.get(feature, 0.0)) > float(value):
                return False
    return True


def _rule_feature_thresholds(rule: Mapping[str, Any], key: str) -> dict[str, float]:
    raw = rule.get(key)
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for feature, value in raw.items():
        try:
            out[str(feature)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _afternoon_rule_feature_values(ctx: OLRAfternoonContext, score: float, exhaustion_score: float) -> dict[str, float]:
    daily_score = _candidate_meta_float(ctx.candidate, "sector_daily_score_pct", 50.0)
    daily_ret_5d = _candidate_meta_float(ctx.candidate, "sector_daily_ret_5d", 0.0)
    daily_ret_20d = _candidate_meta_float(ctx.candidate, "sector_daily_ret_20d", 0.0)
    daily_ret_60d = _candidate_meta_float(ctx.candidate, "sector_daily_ret_60d", 0.0)
    daily_breadth = _candidate_meta_float(ctx.candidate, "sector_daily_breadth_20d", 0.5)
    daily_participation = _candidate_meta_float(ctx.candidate, "sector_daily_participation", 0.0)
    daily_rel_volume = _candidate_meta_float(ctx.candidate, "sector_daily_rel_volume", 1.0)
    daily_flow = _candidate_meta_float(ctx.candidate, "sector_daily_flow_5d", 0.0)
    daily_flow_agreement = _candidate_meta_float(ctx.candidate, "sector_daily_flow_agreement_5d", 0.0)
    legacy_strength = _candidate_meta_float(ctx.candidate, "sector_strength_pct", 50.0)
    legacy_participation = _candidate_meta_float(ctx.candidate, "sector_participation", 0.0)
    daily_accel = daily_ret_5d - (daily_ret_20d / 4.0)
    stock_sector_ret5_gap_pct = ctx.prior_return_5d - (100.0 * daily_ret_5d)
    stock_sector_ret20_gap_pct = ctx.prior_return_20d - (100.0 * daily_ret_20d)
    stock_sector_ret60_gap_pct = ctx.prior_return_60d - (100.0 * daily_ret_60d)
    stock_intraday_sector_ret_gap = ctx.afternoon_ret - ctx.intraday_sector_ret
    score_delta = ctx.intraday_sector_score_pct - daily_score
    confirm_min = min(daily_score, ctx.intraday_sector_score_pct)
    confirm_mean = 0.5 * (daily_score + ctx.intraday_sector_score_pct)
    quality_confirm = (
        0.35 * daily_score
        + 0.30 * ctx.intraday_sector_score_pct
        + 12.0 * (daily_breadth - 0.5)
        + 10.0 * (ctx.intraday_sector_breadth - 0.5)
        + 8.0 * (daily_participation - 0.5)
        + 6.0 * (ctx.intraday_sector_participation - 0.5)
        + 4.0 * math.tanh(daily_flow * 4.0)
        + 3.0 * math.tanh((daily_rel_volume - 1.0) * 2.0)
    )
    rotation_score = (
        score_delta
        + 100.0 * max(ctx.intraday_sector_ret, 0.0)
        + 15.0 * (ctx.intraday_sector_breadth - 0.5)
        + 5.0 * math.tanh((ctx.intraday_sector_rel_volume - 1.0) * 2.0)
    )
    stock_leadership_score = (
        100.0 * stock_intraday_sector_ret_gap
        + 8.0 * (ctx.close_location - 0.5)
        + 4.0 * math.tanh((ctx.rel_volume - 1.0) * 2.0)
    )
    return {
        "afternoon_score": float(score),
        "afternoon_exhaustion_score": float(exhaustion_score),
        "sector_daily_score_pct": daily_score,
        "sector_daily_ret_5d": daily_ret_5d,
        "sector_daily_ret_20d": daily_ret_20d,
        "sector_daily_ret_60d": daily_ret_60d,
        "sector_daily_ret_5d_pct": 100.0 * daily_ret_5d,
        "sector_daily_ret_20d_pct": 100.0 * daily_ret_20d,
        "sector_daily_ret_60d_pct": 100.0 * daily_ret_60d,
        "sector_daily_breadth_20d": daily_breadth,
        "sector_daily_participation": daily_participation,
        "sector_daily_rel_volume": daily_rel_volume,
        "sector_daily_flow_5d": daily_flow,
        "sector_daily_flow_agreement_5d": daily_flow_agreement,
        "sector_daily_accel_5v20": daily_accel,
        "sector_daily_accel_5v20_pct": 100.0 * daily_accel,
        "sector_strength_pct": legacy_strength,
        "sector_participation": legacy_participation,
        "sector_intraday_score_pct": ctx.intraday_sector_score_pct,
        "intraday_sector_score_pct": ctx.intraday_sector_score_pct,
        "sector_intraday_ret": ctx.intraday_sector_ret,
        "sector_intraday_ret_pct": 100.0 * ctx.intraday_sector_ret,
        "sector_intraday_breadth": ctx.intraday_sector_breadth,
        "sector_intraday_rel_volume": ctx.intraday_sector_rel_volume,
        "sector_intraday_participation": ctx.intraday_sector_participation,
        "sector_intraday_effective_count": float(ctx.intraday_sector_effective_count),
        "sector_intraday_daily_score_delta": score_delta,
        "sector_intraday_daily_score_abs_delta": abs(score_delta),
        "sector_confirm_min_score_pct": confirm_min,
        "sector_confirm_mean_score_pct": confirm_mean,
        "sector_confirm_quality_score": quality_confirm,
        "sector_rotation_score": rotation_score,
        "stock_sector_daily_ret5_gap_pct": stock_sector_ret5_gap_pct,
        "stock_sector_daily_ret20_gap_pct": stock_sector_ret20_gap_pct,
        "stock_sector_daily_ret60_gap_pct": stock_sector_ret60_gap_pct,
        "stock_intraday_sector_ret_gap": stock_intraday_sector_ret_gap,
        "stock_intraday_sector_ret_gap_pct": 100.0 * stock_intraday_sector_ret_gap,
        "stock_intraday_leadership_score": stock_leadership_score,
    }


def _shadow_reranker_feature_values(
    ctx: OLRAfternoonContext,
    score: float,
    raw_score: float,
    exhaustion_score: float,
) -> dict[str, float]:
    candidate = ctx.candidate
    values = _afternoon_rule_feature_values(ctx, score, exhaustion_score)
    values.update(
        {
            "daily_candidate_score": float(candidate.selection_score or 0.0),
            "daily_candidate_rank": float(candidate.rank or 999),
            "daily_rank_pct": float(candidate.rank_pct or 0.0),
            "daily_signal_score": float(candidate.daily_signal_score or _candidate_meta_float(candidate, "daily_signal_score", 0.0)),
            "relative_strength_pct": float(candidate.rs_percentile or _candidate_meta_float(candidate, "rs_percentile", 0.0)),
            "accumulation_score": float(candidate.accumulation_score or 0.0),
            "flow_score": float(candidate.flow_score or 0.0),
            "foreign_flow_5d": float(candidate.foreign_flow_5d or _candidate_meta_float(candidate, "lagged_foreign_flow_5d", 0.0)),
            "institutional_flow_5d": float(candidate.institutional_flow_5d or _candidate_meta_float(candidate, "lagged_institutional_flow_5d", 0.0)),
            "flow_agreement_5d": float(candidate.flow_agreement_5d or _candidate_meta_float(candidate, "lagged_flow_agreement_5d", 0.0)),
            "afternoon_score": float(score),
            "afternoon_score_raw": float(raw_score),
            "afternoon_exhaustion_score": float(exhaustion_score),
            "prior_return_5d": float(ctx.prior_return_5d),
            "prior_return_20d": float(ctx.prior_return_20d),
            "prior_return_60d": float(ctx.prior_return_60d),
            "afternoon_ret": float(ctx.afternoon_ret),
            "vwap_ret": float(ctx.vwap_ret),
            "gap": float(ctx.gap),
            "rel_volume": float(ctx.rel_volume),
            "close_location": float(ctx.close_location),
            "open_drawdown": float(ctx.open_drawdown),
            "high_from_open": float(ctx.high_from_open),
            "low_vs_prev_close": float(ctx.low_vs_prev_close),
            "range_atr": float(ctx.range_atr),
            "lagged_flow_5d": float(ctx.lagged_flow_5d),
            "lagged_foreign_flow_5d": float(ctx.lagged_foreign_flow_5d),
            "lagged_institutional_flow_5d": float(ctx.lagged_institutional_flow_5d),
            "lagged_flow_z": float(ctx.lagged_flow_z),
            "lagged_foreign_z": float(ctx.lagged_foreign_z),
            "lagged_institutional_z": float(ctx.lagged_institutional_z),
            "lagged_flow_agreement_5d": float(ctx.lagged_flow_agreement_5d),
            "lagged_flow_divergence_5d": float(ctx.lagged_flow_divergence_5d),
            "lagged_sector_flow_5d": float(ctx.lagged_sector_flow_5d),
            "lagged_sector_foreign_flow_5d": float(ctx.lagged_sector_foreign_flow_5d),
            "lagged_sector_institutional_flow_5d": float(ctx.lagged_sector_institutional_flow_5d),
            "market_score": float(ctx.market_score),
        }
    )
    return values


def _shadow_reranker_score(features: Mapping[str, float], sector: str, profile: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    stats = dict(profile.get("feature_stats") or {})
    weights = dict(profile.get("weights") or {})
    clip = max(float(profile.get("score_clip", 6.0) or 6.0), 0.1)
    score = 0.0
    components: dict[str, float] = {}
    for key, raw_weight in weights.items():
        if key not in features:
            continue
        stat = stats.get(key) or {}
        std = float(stat.get("std", 0.0) or 0.0)
        if std <= 0.0:
            continue
        z = (float(features[key]) - float(stat.get("mean", 0.0) or 0.0)) / std
        z = max(-clip, min(clip, z))
        component = float(raw_weight) * z
        components[str(key)] = component
        score += component
    sector_priors = dict(profile.get("sector_priors") or {})
    sector_component = float(profile.get("sector_prior_weight", 0.35) or 0.0) * float(sector_priors.get(_normalize_sector(sector), 0.0) or 0.0)
    components["sector_prior"] = sector_component
    score += sector_component
    return score, components


def _afternoon_score(ctx: OLRAfternoonContext, config: OLRConfig) -> float:
    return _afternoon_score_details(ctx, config)[0]


def _afternoon_score_details(ctx: OLRAfternoonContext, config: OLRConfig) -> tuple[float, float, float]:
    raw_score = _raw_afternoon_score(ctx, config)
    exhaustion_score = _afternoon_exhaustion_score(ctx)
    score = raw_score
    if str(config.afternoon_score_calibration_mode or "raw").lower() == "exhaustion_adjusted":
        score -= max(float(config.afternoon_exhaustion_penalty), 0.0) * exhaustion_score
    return score, raw_score, exhaustion_score


def _raw_afternoon_score(ctx: OLRAfternoonContext, config: OLRConfig) -> float:
    rel_volume_bonus = min(ctx.rel_volume, 5.0) * 0.001
    close_bonus = ctx.close_location * 0.002
    flow_bonus = (
        0.025 * max(ctx.lagged_flow_5d, 0.0)
        + 0.015 * max(ctx.lagged_foreign_flow_5d, 0.0)
        + 0.015 * max(ctx.lagged_institutional_flow_5d, 0.0)
        + 0.020 * max(ctx.lagged_flow_agreement_5d, 0.0)
        + 0.001 * max(ctx.lagged_flow_z, 0.0)
        + 0.04 * max(ctx.lagged_sector_flow_5d, 0.0)
        - 0.010 * max(ctx.lagged_flow_divergence_5d, 0.0)
    )
    market_bonus = 0.002 * max(ctx.market_score / 100.0, 0.0)
    lagged_flow_bonus = 0.0
    if config.afternoon_use_lagged_flow_score:
        lagged_flow_bonus = 0.002 * math.tanh(float(ctx.candidate.flow_score) * 10.0)
    sector_intraday_bonus = max(float(config.afternoon_weight_intraday_sector), 0.0) * ((float(ctx.intraday_sector_score_pct) - 50.0) / 50.0)
    rule_features = _afternoon_rule_feature_values(ctx, 0.0, _afternoon_exhaustion_score(ctx))
    sector_context_bonus = (
        max(float(config.afternoon_weight_sector_confirm_quality), 0.0)
        * ((float(rule_features["sector_confirm_quality_score"]) - 35.0) / 35.0)
        + max(float(config.afternoon_weight_sector_rotation), 0.0)
        * (float(rule_features["sector_rotation_score"]) / 50.0)
        + max(float(config.afternoon_weight_stock_sector_leadership), 0.0)
        * (float(rule_features["stock_intraday_leadership_score"]) / 10.0)
    )
    mode = str(config.afternoon_score_mode or "hybrid").lower()
    if mode == "momentum":
        raw = ctx.afternoon_ret + 0.25 * ctx.vwap_ret + 0.05 * max(ctx.gap, 0.0) + rel_volume_bonus + close_bonus + market_bonus
    elif mode == "vwap_strength":
        raw = ctx.vwap_ret + 0.40 * max(ctx.afternoon_ret, 0.0) + 0.05 * max(ctx.gap, 0.0) + rel_volume_bonus + flow_bonus
    elif mode == "gap_hold":
        raw = ctx.gap + ctx.afternoon_ret + 0.25 * ctx.vwap_ret + max(ctx.low_vs_prev_close, 0.0) + close_bonus + flow_bonus
    elif mode == "flow_confirmed":
        raw = 0.55 * max(ctx.afternoon_ret, 0.0) + 0.30 * max(ctx.vwap_ret, 0.0) + flow_bonus + close_bonus
    elif mode == "efficient":
        raw = (ctx.afternoon_ret + 0.50 * ctx.vwap_ret + 0.05 * max(ctx.gap, 0.0)) / max(ctx.range_atr, 0.15) + rel_volume_bonus + flow_bonus
    elif mode == "daily_plus_intraday":
        raw = 0.0005 * ctx.candidate.selection_score + ctx.afternoon_ret + ctx.vwap_ret + rel_volume_bonus + close_bonus + flow_bonus
    else:
        raw = ctx.afternoon_ret + ctx.vwap_ret + 0.20 * max(ctx.gap, 0.0) + rel_volume_bonus + close_bonus + flow_bonus + market_bonus
    return 10_000.0 * (raw + lagged_flow_bonus + sector_intraday_bonus + sector_context_bonus)


def _afternoon_exhaustion_score(ctx: OLRAfternoonContext) -> float:
    ret_excess = max(float(ctx.afternoon_ret) - 0.045, 0.0) / 0.045
    vwap_extension = max(float(ctx.vwap_ret) - 0.025, 0.0) / 0.025
    range_extension = max(float(ctx.range_atr) - 1.35, 0.0) / 1.35
    high_extension = max(float(ctx.high_from_open) - 0.075, 0.0) / 0.075
    weak_close_after_push = max(0.65 - float(ctx.close_location), 0.0) / 0.65 if ctx.high_from_open > 0.035 else 0.0
    flow_divergence = max(float(ctx.lagged_flow_divergence_5d), 0.0)
    return max(0.0, ret_excess + vwap_extension + range_extension + high_extension + 0.5 * weak_close_after_push + 0.5 * flow_divergence)


def _intraday_selection_bars(bars: Any, trade_date: date) -> tuple[MarketBar, ...]:
    out: list[MarketBar] = []
    for bar in sorted(tuple(bars or ()), key=lambda item: item.timestamp):
        ts = bar.timestamp.astimezone(KST)
        if ts.date() != trade_date:
            continue
        if ts.time() < time(9, 0) or ts.time() >= INTRADAY_SELECTION_CUTOFF:
            continue
        if not bool(bar.is_completed):
            continue
        out.append(bar)
    return tuple(out)


def _bars_for_candidate(bars_by_symbol: Mapping[Any, Any], trade_date: date, symbol: str) -> Any:
    if symbol in bars_by_symbol:
        return bars_by_symbol[symbol]
    key = (trade_date, symbol)
    if key in bars_by_symbol:
        return bars_by_symbol[key]
    return ()


def _candidate_meta_float(candidate: OLRDailyCandidate, key: str, default: float = 0.0) -> float:
    return _float(dict(candidate.metadata or {}).get(key), default)


def _prior_rows(rows: list[dict[str, Any]], trade_date: date, lookback: int | None = None) -> list[dict[str, Any]]:
    dates, ordered = _indexed_rows(rows)
    if not dates:
        return []
    end = bisect.bisect_left(dates, trade_date)
    start = max(0, end - int(lookback)) if lookback and int(lookback) > 0 else 0
    return list(ordered[start:end])


def _indexed_rows(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[tuple[date, ...], tuple[dict[str, Any], ...]]:
    if not rows:
        return (), ()
    key = id(rows)
    cached = _ROW_INDEX_CACHE.get(key)
    if cached is not None and cached[0] == len(rows) and cached[1] is rows:
        return cached[2], cached[3]
    dated_rows: list[tuple[date, dict[str, Any]]] = []
    for row in rows or ():
        parsed = _try_row_date(row)
        if parsed is None:
            continue
        try:
            dated_rows.append((parsed, dict(row)))
        except (TypeError, ValueError):
            continue
    dated_rows.sort(key=lambda item: item[0])
    dates = tuple(item[0] for item in dated_rows)
    ordered = tuple(item[1] for item in dated_rows)
    _ROW_INDEX_CACHE[key] = (len(rows), rows, dates, ordered)
    return dates, ordered


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


def _row_date_label(row: dict[str, Any]) -> str:
    return _row_date(row).isoformat()


def _build_market_research(
    trade_date: date,
    symbols: dict[str, OLRResearchSymbol],
    index_ohlcv_by_symbol: dict[str, list[dict[str, Any]]],
) -> OLRMarketResearch:
    above = 0
    counted = 0
    returns: list[float] = []
    for item in symbols.values():
        closes = [_float(row.get("close")) for row in item.daily_rows]
        if len(closes) >= 20:
            counted += 1
            if closes[-1] > fmean(closes[-20:]):
                above += 1
            returns.append(item.return_20d_pct)
    breadth = 100.0 * above / counted if counted else 50.0
    avg20 = fmean(returns) if returns else 0.0
    index_boost = _index_heat(index_ohlcv_by_symbol, trade_date)
    heat = max(0.0, min(100.0, 50.0 + 5.0 * avg20 + 0.35 * (breadth - 50.0) + 15.0 * (index_boost - 0.5)))
    if breadth >= 55.0 and avg20 >= 0.0:
        tier, regime = "A", "BROAD_RISK_ON"
    elif breadth >= 35.0:
        tier, regime = "B", "MIXED"
    else:
        tier, regime = "C", "RISK_OFF"
    return OLRMarketResearch(
        trade_date=trade_date,
        breadth_pct_above_20dma=breadth,
        avg_20d_return_pct=avg20,
        market_heat_score=heat,
        regime_tier=tier,
        regime=regime,
    )


def _build_sector_research(
    symbols: dict[str, OLRResearchSymbol],
    sector_daily_panel: SectorDailyPanel | None = None,
) -> dict[str, OLRSectorResearch]:
    if not symbols:
        return {}
    by_sector: dict[str, list[OLRResearchSymbol]] = {}
    for item in symbols.values():
        by_sector.setdefault(item.sector, []).append(item)

    sector_research: dict[str, OLRSectorResearch] = {}
    for sector, items in by_sector.items():
        counted = 0
        above = 0
        returns: list[float] = []
        participating = 0
        flow_5d: list[float] = []
        foreign_flow_5d: list[float] = []
        institutional_flow_5d: list[float] = []
        flow_agreement_5d: list[float] = []
        for item in items:
            closes = [_float(row.get("close")) for row in item.daily_rows]
            if len(closes) >= 20:
                counted += 1
                above += int(closes[-1] > fmean(closes[-20:]))
                returns.append(item.return_20d_pct)
            if item.return_5d_pct >= 0.0 and item.volume_ratio_20d >= 1.0:
                participating += 1
            flow_5d.append(item.flow_5d)
            foreign_flow_5d.append(item.foreign_flow_5d)
            institutional_flow_5d.append(item.institutional_flow_5d)
            flow_agreement_5d.append(item.flow_agreement_5d)
        breadth = above / counted if counted else 0.5
        ret20 = fmean(returns) if returns else 0.0
        participation = participating / len(items) if items else 0.0
        sector_research[sector] = OLRSectorResearch(
            sector=sector,
            symbol_count=len(items),
            return_20d_pct=ret20,
            breadth_20d=breadth,
            participation=participation,
            flow_5d=fmean(flow_5d) if flow_5d else 0.0,
            foreign_flow_5d=fmean(foreign_flow_5d) if foreign_flow_5d else 0.0,
            institutional_flow_5d=fmean(institutional_flow_5d) if institutional_flow_5d else 0.0,
            flow_agreement_5d=fmean(flow_agreement_5d) if flow_agreement_5d else 0.0,
            regime=_legacy_sector_regime(ret20, breadth),
        )
    return sector_research


def _sector_daily_panel_from_symbols(
    symbols: dict[str, OLRResearchSymbol],
    trade_date: date,
) -> SectorDailyPanel:
    return build_sector_daily_panel(
        {symbol: list(item.daily_rows) for symbol, item in symbols.items()},
        {symbol: item.sector for symbol, item in symbols.items()},
        trade_dates=(trade_date,),
        flow_by_symbol={symbol: list(item.flow_rows) for symbol, item in symbols.items()},
        foreign_flow_by_symbol={symbol: list(item.foreign_flow_rows) for symbol, item in symbols.items()},
        institutional_flow_by_symbol={symbol: list(item.institutional_flow_rows) for symbol, item in symbols.items()},
        symbols=symbols,
    )


def _relative_strength_percentiles(symbols: dict[str, OLRResearchSymbol]) -> dict[str, float]:
    raw = {symbol: 0.65 * item.return_20d_pct + 0.35 * item.return_60d_pct for symbol, item in symbols.items()}
    return _percentile_map(raw)


def _relative_strength_60d_percentiles(symbols: dict[str, OLRResearchSymbol]) -> dict[str, float]:
    raw = {symbol: item.return_60d_pct for symbol, item in symbols.items()}
    return _percentile_map(raw)


def _sector_strength_percentiles(sectors: dict[str, OLRSectorResearch]) -> dict[str, float]:
    raw = {sector: item.return_20d_pct for sector, item in sectors.items()}
    return _percentile_map(raw)


def _percentile_map(raw: dict[str, float]) -> dict[str, float]:
    values = sorted(raw.values())
    if not values:
        return {}
    return {
        symbol: 100.0 * bisect.bisect_right(values, value) / len(values)
        for symbol, value in raw.items()
    }


def _bad_recent_ohlcv(rows: tuple[dict[str, Any], ...]) -> bool:
    if not rows:
        return True
    for row in rows:
        open_ = _float(row.get("open"))
        high = _float(row.get("high"))
        low = _float(row.get("low"))
        close = _float(row.get("close"))
        volume = _float(row.get("volume"))
        if min(open_, high, low, close) <= 0.0 or volume <= 0.0:
            return True
        if high < low or close > high * 1.0001 or close < low * 0.9999:
            return True
    return False


def _adv_krw(rows: list[dict[str, Any]], lookback: int) -> float:
    sample = rows[-lookback:]
    values = [_float(row.get("close")) * _float(row.get("volume")) for row in sample if _float(row.get("close")) > 0 and _float(row.get("volume")) > 0]
    return fmean(values) if values else 0.0


def _return_pct(rows: list[dict[str, Any]], lookback: int) -> float:
    if len(rows) <= lookback:
        return 0.0
    now = _float(rows[-1].get("close"))
    prev = _float(rows[-1 - lookback].get("close"))
    if prev <= 0:
        return 0.0
    return (now / prev - 1.0) * 100.0


def _volume_ratio(rows: list[dict[str, Any]], lookback: int) -> float:
    if len(rows) < lookback + 1:
        return 1.0
    current = _float(rows[-1].get("volume"))
    prior = [_float(row.get("volume")) for row in rows[-lookback - 1 : -1]]
    avg = fmean(value for value in prior if value > 0) if any(value > 0 for value in prior) else 0.0
    return current / avg if avg > 0 else 1.0


def _close_location(rows: list[dict[str, Any]]) -> float:
    highs = [_float(row.get("high")) for row in rows]
    lows = [_float(row.get("low")) for row in rows]
    closes = [_float(row.get("close")) for row in rows]
    if not highs or not lows or not closes:
        return 0.5
    high = max(highs)
    low = min(lows)
    if high <= low:
        return 0.5
    return (closes[-1] - low) / (high - low)


def _trend_score(rows: tuple[dict[str, Any], ...]) -> float:
    closes = [_float(row.get("close")) for row in rows]
    if len(closes) < 20:
        return 0.0
    sma20 = _mean(closes[-20:])
    sma60 = _mean(closes[-60:]) if len(closes) >= 60 else sma20
    sma20_rising = len(closes) < 25 or sma20 >= _mean(closes[-25:-5])
    close = closes[-1]
    if close >= sma20 >= sma60 and sma20_rising:
        return 100.0
    if close >= sma60:
        return 70.0
    if close >= sma20:
        return 45.0
    return 10.0


def _compression_score(rows: tuple[dict[str, Any], ...]) -> tuple[float, float]:
    if len(rows) < 20:
        return 0.0, 0.0
    window_10 = rows[-10:]
    window_20 = rows[-20:]
    close = max(_float(rows[-1].get("close")), 1e-9)
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
    return quality, range_10 / close


def _accumulation_score(rows: tuple[dict[str, Any], ...]) -> float:
    sample = rows[-20:]
    signed_volume = 0.0
    total_volume = 0.0
    for row in sample:
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


def _stock_regime_score(rows: tuple[dict[str, Any], ...]) -> float:
    closes = [_float(row.get("close")) for row in rows]
    if len(closes) < 20:
        return 0.0
    score = 40.0
    if closes[-1] > fmean(closes[-20:]):
        score += 25.0
    if len(closes) >= 50 and closes[-1] > fmean(closes[-50:]):
        score += 20.0
    if len(closes) >= 5 and closes[-1] > closes[-5]:
        score += 15.0
    return min(100.0, score)


def _sector_regime_score(sector: OLRSectorResearch | None) -> float:
    if sector is None:
        return 35.0
    if sector.regime == "LEADING":
        return 100.0
    if sector.regime == "MIXED":
        return 60.0
    return 25.0


def _legacy_sector_regime(return_20d_pct: float, breadth_20d: float) -> str:
    if return_20d_pct >= 0.0 and breadth_20d >= 0.55:
        return "LEADING"
    if breadth_20d >= 0.35:
        return "MIXED"
    return "LAGGING"


def _flow_metrics(
    daily_rows: list[dict[str, Any]],
    flow_rows: list[dict[str, Any]],
    foreign_rows: list[dict[str, Any]],
    inst_rows: list[dict[str, Any]],
) -> dict[str, float]:
    daily_by_date = {_row_date(row): row for row in daily_rows if _try_row_date(row) is not None}
    flow_by_date = {_row_date(row): row for row in flow_rows if _try_row_date(row) is not None}
    foreign_by_date = {_row_date(row): row for row in foreign_rows if _try_row_date(row) is not None}
    inst_by_date = {_row_date(row): row for row in inst_rows if _try_row_date(row) is not None}
    normalized: list[tuple[date, float, float, float, float, float, float]] = []
    for flow_date in sorted(daily_by_date):
        daily = daily_by_date[flow_date]
        volume = max(_float(daily.get("volume")), 1.0)
        close = max(_float(daily.get("close")), 0.0)
        flow_row = flow_by_date.get(flow_date, {})
        foreign_row = foreign_by_date.get(flow_date, {})
        inst_row = inst_by_date.get(flow_date, {})
        foreign = _optional_flow_value(foreign_row, "foreign_net")
        if foreign is None:
            foreign = _optional_flow_value(flow_row, "foreign_net") or 0.0
        inst = _optional_flow_value(inst_row, "institutional_net", "inst_net")
        if inst is None:
            inst = _optional_flow_value(flow_row, "institutional_net", "inst_net") or 0.0
        combined = float(foreign) + float(inst)
        foreign_norm = float(foreign) / volume
        inst_norm = float(inst) / volume
        combined_norm = combined / volume
        agreement = min(max(foreign_norm, 0.0), max(inst_norm, 0.0))
        divergence = max(0.0, -foreign_norm * inst_norm) ** 0.5 if foreign_norm * inst_norm < 0.0 else abs(foreign_norm - inst_norm) * 0.25
        notional = combined * close
        daily_notional = volume * close
        normalized.append((flow_date, combined_norm, foreign_norm, inst_norm, agreement, divergence, notional / max(daily_notional, 1.0)))

    def avg(index: int, lookback: int) -> float:
        values = [item[index] for item in normalized[-lookback:]]
        return fmean(values) if values else 0.0

    def positive(index: int, lookback: int) -> float:
        values = [item[index] for item in normalized[-lookback:]]
        return float(sum(1 for value in values if value > 0.0))

    def z(index: int) -> float:
        values = [item[index] for item in normalized[-20:]]
        std = _std(values)
        return (values[-1] - fmean(values)) / std if values and std > 0.0 else 0.0

    if not normalized:
        return _empty_flow_metrics()
    combined_values = [item[1] for item in normalized]
    foreign_values = [item[2] for item in normalized]
    inst_values = [item[3] for item in normalized]
    combined20 = avg(1, 20)
    foreign20 = avg(2, 20)
    inst20 = avg(3, 20)
    combined_accel_base = fmean(combined_values[-40:-20]) if combined_values[-40:-20] else combined20
    foreign_accel_base = fmean(foreign_values[-40:-20]) if foreign_values[-40:-20] else foreign20
    inst_accel_base = fmean(inst_values[-40:-20]) if inst_values[-40:-20] else inst20
    return {
        "flow_available": 1.0 if (flow_rows or foreign_rows or inst_rows) else 0.0,
        "flow_1d": avg(1, 1),
        "flow_3d": avg(1, 3),
        "flow_5d": avg(1, 5),
        "flow_20d": combined20,
        "foreign_flow_1d": avg(2, 1),
        "foreign_flow_3d": avg(2, 3),
        "foreign_flow_5d": avg(2, 5),
        "foreign_flow_20d": foreign20,
        "institutional_flow_1d": avg(3, 1),
        "institutional_flow_3d": avg(3, 3),
        "institutional_flow_5d": avg(3, 5),
        "institutional_flow_20d": inst20,
        "flow_z": z(1),
        "foreign_flow_z": z(2),
        "institutional_flow_z": z(3),
        "flow_positive_days_5d": positive(1, 5),
        "foreign_positive_days_5d": positive(2, 5),
        "institutional_positive_days_5d": positive(3, 5),
        "flow_acceleration": avg(1, 3) - combined_accel_base,
        "foreign_flow_acceleration": avg(2, 3) - foreign_accel_base,
        "institutional_flow_acceleration": avg(3, 3) - inst_accel_base,
        "flow_agreement_5d": avg(4, 5),
        "flow_divergence_5d": avg(5, 5),
        "combined_flow_notional_5d": sum(item[6] for item in normalized[-5:]),
        "sponsorship_balance_5d": avg(2, 5) - avg(3, 5),
    }


def _optional_flow_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row and row.get(key) is not None:
            return _float(row.get(key))
    return None


def _empty_flow_metrics() -> dict[str, float]:
    keys = (
        "flow_available",
        "flow_1d",
        "flow_3d",
        "flow_5d",
        "flow_20d",
        "foreign_flow_1d",
        "foreign_flow_3d",
        "foreign_flow_5d",
        "foreign_flow_20d",
        "institutional_flow_1d",
        "institutional_flow_3d",
        "institutional_flow_5d",
        "institutional_flow_20d",
        "flow_z",
        "foreign_flow_z",
        "institutional_flow_z",
        "flow_positive_days_5d",
        "foreign_positive_days_5d",
        "institutional_positive_days_5d",
        "flow_acceleration",
        "foreign_flow_acceleration",
        "institutional_flow_acceleration",
        "flow_agreement_5d",
        "flow_divergence_5d",
        "combined_flow_notional_5d",
        "sponsorship_balance_5d",
    )
    return {key: 0.0 for key in keys}


def _flow_score(value: float, z: float) -> float:
    return 50.0 + 35.0 * math.tanh(float(value) * 2.0) + 15.0 * math.tanh(float(z) / 2.0)


def _spread_pct(row: dict[str, Any]) -> float | None:
    for key in ("median_spread_pct", "spread_pct", "bid_ask_spread_pct"):
        if key in row and row.get(key) is not None:
            return _float(row.get(key))
    return None


def _flag(row: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key in row:
            value = row.get(key)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)
    return False


def _index_heat(index_ohlcv_by_symbol: dict[str, list[dict[str, Any]]], trade_date: date) -> float:
    values = []
    for rows in index_ohlcv_by_symbol.values():
        prior = _prior_rows(rows, trade_date)
        if len(prior) < 20:
            continue
        closes = [_float(row.get("close")) for row in prior]
        values.append(1.0 if closes[-1] > fmean(closes[-20:]) else 0.0)
    return fmean(values) if values else 0.0


def _atr(rows: list[dict[str, Any]], period: int) -> float:
    if len(rows) < 2:
        return 0.0
    trs: list[float] = []
    sample = rows[-(period + 1) :]
    for index in range(1, len(sample)):
        high = _float(sample[index].get("high"))
        low = _float(sample[index].get("low"))
        prev_close = _float(sample[index - 1].get("close"))
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return fmean(trs) if trs else 0.0


def _research_config_fingerprint(config: OLRConfig) -> str:
    payload = {key: getattr(config, key) for key in sorted(config.__dataclass_fields__)}
    return _stable_hash(payload)


def final_candidate_config_fingerprint(config: OLRConfig) -> str:
    payload = {
        key: getattr(config, key)
        for key in sorted(config.__dataclass_fields__)
        if key not in _FINAL_CONFIG_HASH_EXCLUDED_FIELDS
    }
    return _stable_hash(
        {
            "version": FINAL_CANDIDATE_CONFIG_HASH_VERSION,
            "core_version": OLR_CORE_VERSION,
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "selection_config": payload,
        }
    )


def _mapping_fingerprint(mapping: dict[str, Any]) -> str:
    return _stable_hash(mapping)


def _source_fingerprint(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    flow_by_symbol: dict[str, list[dict[str, Any]]],
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]],
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]],
    index_ohlcv_by_symbol: dict[str, list[dict[str, Any]]],
    trade_date: date,
) -> str:
    return _stable_hash(
        {
            "model": RESEARCH_MODEL_VERSION,
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "trade_date": trade_date.isoformat(),
            "daily": {str(symbol).zfill(6): _causal_digest(rows, trade_date) for symbol, rows in sorted((daily_by_symbol or {}).items())},
            "flow": {str(symbol).zfill(6): _causal_digest(rows, trade_date) for symbol, rows in sorted((flow_by_symbol or {}).items())},
            "foreign_flow": {str(symbol).zfill(6): _causal_digest(rows, trade_date) for symbol, rows in sorted((foreign_flow_by_symbol or {}).items())},
            "institutional_flow": {str(symbol).zfill(6): _causal_digest(rows, trade_date) for symbol, rows in sorted((institutional_flow_by_symbol or {}).items())},
            "index": {str(code): _causal_digest(rows, trade_date) for code, rows in sorted((index_ohlcv_by_symbol or {}).items())},
        }
    )


def _causal_digest(rows: list[dict[str, Any]], trade_date: date) -> str:
    dates, prefixes = _row_digest_prefixes(rows)
    if not dates:
        return _EMPTY_ROW_DIGEST
    end = bisect.bisect_left(dates, trade_date)
    if end <= 0:
        return _EMPTY_ROW_DIGEST
    return prefixes[end - 1]


def _row_digest_prefixes(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[tuple[date, ...], tuple[str, ...]]:
    if not rows:
        return (), ()
    key = id(rows)
    cached = _ROW_DIGEST_PREFIX_CACHE.get(key)
    if cached is not None and cached[0] == len(rows) and cached[1] is rows:
        return cached[2], cached[3]
    dates, ordered = _indexed_rows(rows)
    prefixes: list[str] = []
    acc = _EMPTY_ROW_DIGEST
    for row in ordered:
        row_hash = _stable_hash(tuple((key, row.get(key)) for key in sorted(row)))
        acc = _stable_hash((acc, row_hash))
        prefixes.append(acc)
    out = tuple(prefixes)
    _ROW_DIGEST_PREFIX_CACHE[key] = (len(rows), rows, dates, out)
    return dates, out


def _snapshot_candidate_metadata(snapshot: OLRResearchSnapshot) -> dict[str, Any]:
    return {
        "research_config_hash": snapshot.metadata.get("research_config_hash"),
        "sector_map_hash": snapshot.metadata.get("sector_map_hash"),
        "research_as_of_date": snapshot.metadata.get("research_as_of_date"),
        "daily_row_cutoff": snapshot.metadata.get("daily_row_cutoff"),
        "flow_row_cutoff": snapshot.metadata.get("flow_row_cutoff"),
        "research_causal_source_fingerprint": snapshot.metadata.get("research_causal_source_fingerprint"),
    }


def _research_as_of_date(symbols: dict[str, OLRResearchSymbol]) -> str | None:
    dates = []
    for symbol in symbols.values():
        if symbol.daily_rows:
            dates.append(_row_date(symbol.daily_rows[-1]))
    return max(dates).isoformat() if dates else None


def _normalize_sector(sector: Any) -> str:
    text = str(sector or "").strip().upper()
    return text or "UNKNOWN"


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(raw.encode("utf-8")).hexdigest()


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return math.sqrt(fmean((value - mean) ** 2 for value in values))


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
