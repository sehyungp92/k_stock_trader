from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.config import load_yaml_config, normalize_runtime_config
from strategy_kalcb.config import KALCBConfig

from .first30_signal_sweep import (
    DEFAULT_HOLDOUT_DAYS,
    First30Context,
    First30Spec,
    KALCBFirst30Dataset,
    OpportunityRow,
    Selection,
    STAGE2_PORTFOLIO_POLICY,
    _avg,
    _clip,
    _compact_summary,
    _dedupe_specs,
    _even_sample,
    _float,
    _num_label,
    _pct_label,
    _return_score,
    _round_trip_cost_pct,
    _spec_signature,
    _training_config,
    build_coarse_specs,
    build_contexts,
    build_refinement_specs,
    evaluate_selections,
    evaluate_spec,
    passes,
    prepare_first30_dataset,
    score_candidate,
    select as select_first30,
    summarize,
)


PREMARKET_FRONTIER_SWEEP_VERSION = "kalcb-premarket-frontier-first30-sweep-v1"
DEFAULT_OUTPUT_DIR = Path("data/backtests/output/kalcb/premarket_frontier_sweeps")
FRONTIER_SIZES = (4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 103)


@dataclass(frozen=True, slots=True)
class PremarketFeature:
    day: date
    symbol: str
    sector: str
    ret5: float
    ret20: float
    ret60: float
    atr_pct: float
    adv20_krw: float
    close20_loc: float
    close60_loc: float
    volume_surge: float
    above_sma20: bool
    above_sma60: bool
    flow_1d: float
    flow_3d: float
    flow_5d: float
    flow_20d: float
    flow_notional_5d: float
    flow_positive_days_5d: float
    flow_acceleration: float
    flow_z: float
    flow_available: bool = False
    foreign_5d: float = 0.0
    foreign_z: float = 0.0
    foreign_acceleration: float = 0.0
    foreign_positive_days_5d: float = 0.0
    inst_5d: float = 0.0
    inst_z: float = 0.0
    inst_acceleration: float = 0.0
    inst_positive_days_5d: float = 0.0
    flow_agreement_5d: float = 0.0
    flow_divergence_5d: float = 0.0
    sponsorship_balance_5d: float = 0.0
    sector_foreign_5d: float = 0.0
    sector_inst_5d: float = 0.0
    sector_agreement_5d: float = 0.0
    sector_flow_5d: float = 0.0
    sector_participation: float = 0.0
    market_score: float = 0.0
    market_ret5: float = 0.0
    market_ret20: float = 0.0
    market_above_sma20: bool = False


@dataclass(frozen=True, slots=True)
class FrontierSpec:
    name: str
    mode: str
    frontier_size: int
    min_ret5: float = -1.0
    min_ret20: float = -1.0
    max_ret20: float = 9.99
    min_ret60: float = -1.0
    min_close20_loc: float = 0.0
    min_adv20_krw: float = 0.0
    max_atr_pct: float = 9.99
    min_volume_surge: float = 0.0
    min_flow_5d: float = -9.99
    min_flow_z: float = -9.99
    min_flow_acceleration: float = -9.99
    min_foreign_flow_5d: float = -9.99
    min_inst_flow_5d: float = -9.99
    min_foreign_z: float = -9.99
    min_inst_z: float = -9.99
    min_flow_agreement: float = -9.99
    max_flow_divergence: float = 9.99
    min_sector_flow: float = -9.99
    min_sector_participation: float = 0.0
    min_market_score: float = -9.99
    require_above_sma20: bool = False
    require_above_sma60: bool = False
    require_flow_available: bool = False


@dataclass(frozen=True, slots=True)
class PairSpec:
    name: str
    frontier: FrontierSpec
    first30: First30Spec


@dataclass(frozen=True, slots=True)
class PairResult:
    spec: PairSpec
    return_score: float
    mfe_score: float
    combined_score: float
    pareto_score: float
    rejected: bool
    reject_reason: str
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class FrontierResult:
    spec: FrontierSpec
    return_score: float
    mfe_score: float
    combined_score: float
    pareto_score: float
    rejected: bool
    reject_reason: str
    metrics: dict[str, float]


def run_premarket_frontier_only_sweep(
    config: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    max_workers: int = 4,
    coarse_frontier_limit: int = 640,
    deep_pair_count: int = 24,
    deep_per_mode_limit: int = 160,
    max_frontier_specs: int | None = None,
) -> dict[str, Any]:
    """Stage 1: rank premarket-only frontiers without optimizing first30 selectors."""
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    training_config = _training_config(dict(config), holdout_days)
    _record_setup_progress(out, "prepare_dataset_started")
    dataset = prepare_first30_dataset(training_config)
    _record_setup_progress(
        out,
        "prepare_dataset_completed",
        sessions=len(dataset.trading_dates),
        symbols=len(dataset.data_available_symbols),
        bars_by_key=len(dataset.bars_by_key),
    )
    cfg = KALCBConfig.from_mapping(training_config, {})
    _record_setup_progress(out, "build_contexts_started")
    contexts = build_contexts(dataset)
    _record_setup_progress(out, "build_contexts_completed", days=len(contexts), contexts=sum(len(items) for items in contexts.values()))
    _record_setup_progress(out, "build_premarket_features_started")
    features = build_premarket_features(contexts)
    _record_setup_progress(out, "build_premarket_features_completed", features=sum(len(items) for items in features.values()))
    _record_setup_progress(out, "build_opportunity_map_started")
    opportunity_by_key = build_opportunity_map(dataset, contexts, cfg)
    opportunity_by_day = _opportunity_by_day(opportunity_by_key.values())
    _record_setup_progress(out, "build_opportunity_map_completed", opportunities=len(opportunity_by_key))

    frontier_specs = build_frontier_specs()
    if max_frontier_specs is not None:
        frontier_specs = _even_sample(frontier_specs, max(1, int(max_frontier_specs)))
    coarse_frontiers = _stratified_frontier_sample(frontier_specs, coarse_frontier_limit)
    frontier_cache: dict[str, dict[date, tuple[str, ...]]] = {}
    score_cache: dict[str, dict[date, dict[str, float]]] = {}
    rows = _evaluate_frontier_specs(
        coarse_frontiers,
        features,
        dataset,
        cfg,
        opportunity_by_key,
        opportunity_by_day,
        frontier_cache,
        score_cache,
        out,
        stage="coarse_frontier",
        completed_offset=0,
        total=len(coarse_frontiers),
        max_workers=max_workers,
    )
    deep_frontiers = build_deep_frontier_specs(
        rows,
        all_frontiers=frontier_specs,
        deep_pair_count=deep_pair_count,
        deep_per_mode_limit=deep_per_mode_limit,
    )
    existing = {_frontier_signature(row.spec) for row in rows}
    deep_frontiers = [spec for spec in deep_frontiers if _frontier_signature(spec) not in existing]
    deep_rows = _evaluate_frontier_specs(
        deep_frontiers,
        features,
        dataset,
        cfg,
        opportunity_by_key,
        opportunity_by_day,
        frontier_cache,
        score_cache,
        out,
        stage="deep_frontier",
        completed_offset=len(rows),
        total=len(rows) + len(deep_frontiers),
        max_workers=max_workers,
        seed_rows=rows,
    )
    rows = _dedupe_frontier_results([*rows, *deep_rows])
    _assign_frontier_pareto_scores(rows)
    rows.sort(key=lambda row: (-row.combined_score, row.rejected, -row.pareto_score, row.spec.name))
    top_portfolio = sorted(
        rows,
        key=lambda row: (
            row.rejected,
            -row.metrics.get("portfolio_proxy_net_return_pct", 0.0),
            row.metrics.get("frontier_avg_size", 999.0),
            row.spec.name,
        ),
    )[:30]
    top_slot = sorted(
        rows,
        key=lambda row: (
            row.rejected,
            -row.metrics.get("slot_cumulative_gross_return_pct", 0.0),
            row.metrics.get("frontier_avg_size", 999.0),
            row.spec.name,
        ),
    )[:30]
    top_mfe = sorted(
        rows,
        key=lambda row: (
            row.rejected,
            -row.metrics.get("avg_mfe_r", 0.0),
            -row.metrics.get("portfolio_proxy_net_return_pct", 0.0),
            row.metrics.get("frontier_avg_size", 999.0),
            row.spec.name,
        ),
    )[:30]
    top_pareto = sorted(rows, key=lambda row: (-row.pareto_score, row.rejected, row.metrics.get("frontier_avg_size", 999.0), row.spec.name))[:30]
    payload = {
        "strategy": "kalcb",
        "sweep_version": f"{PREMARKET_FRONTIER_SWEEP_VERSION}-stage1-frontier-only",
        "stage_contract": "stage1_premarket_frontier_only",
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "training_window": {
            "start": dataset.trading_dates[0].isoformat(),
            "end": dataset.trading_dates[-1].isoformat(),
            "sessions": len(dataset.trading_dates),
        },
        "holdout_days": int(holdout_days),
        "causality_policy": {
            "premarket_frontier": "prior completed daily_ohlcv, daily_flow, index_ohlcv, and sector-map data only",
            "evaluation_labels": "09:30-to-configured-flatten bars are used only after selection for research labels",
            "official_performance": False,
        },
        "metric_contract": {
            "primary_promotion_metric": "none_stage1_research_only",
            "primary_research_metric": "portfolio_proxy_net_return_pct",
            "proxy_metrics": ["portfolio_proxy_net_return_pct", "slot_cumulative_net_return_pct", "avg_mfe_r"],
            "promotion_requires_audit_pass": False,
            "official_performance": False,
        },
        "source_fingerprints": {
            "intraday": dataset.source_fingerprint,
            "daily_lrs": dataset.daily_source_fingerprint,
            "combined": stable_signature([dataset.source_fingerprint, dataset.daily_source_fingerprint]),
        },
        "cost_policy": {
            "round_trip_cost_pct": _round_trip_cost_pct(cfg),
            "slippage_bps_each_side": cfg.slippage_bps,
            "commission_bps_each_side": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
        },
        "selection_policy": {
            "frontier_sizes": list(FRONTIER_SIZES),
            "score_modes": list(_frontier_modes()),
            "objective": "premarket-only frontier quality; first30 selectors are intentionally not optimized in Stage 1",
            "portfolio_proxy_policy": STAGE2_PORTFOLIO_POLICY,
            "coarse_frontier_count": len(coarse_frontiers),
            "deep_frontier_count": len(deep_frontiers),
            "max_workers": min(max(1, int(max_workers)), 4),
        },
        "data_policy": {
            "data_root": str(dataset.data_root),
            "daily_data_root": str(dataset.daily_data_root),
            "symbols": len(dataset.symbols),
            "daily_available_symbols": len(dataset.daily_available_symbols),
            "combined_flow_available_symbols": len(dataset.flow_by_symbol),
            "foreign_flow_available_symbols": len(dataset.foreign_flow_by_symbol),
            "institutional_flow_available_symbols": len(dataset.institutional_flow_by_symbol),
            "unavailable_symbols": list(dataset.unavailable_symbols),
            "index_codes": sorted(dataset.index_by_code),
        },
        "candidate_count": len(rows),
        "top_combined": [_frontier_row_payload(row) for row in rows[:30]],
        "top_portfolio_proxy": [_frontier_row_payload(row) for row in top_portfolio],
        "top_slot_return": [_frontier_row_payload(row) for row in top_slot],
        "top_mfe": [_frontier_row_payload(row) for row in top_mfe],
        "top_pareto": [_frontier_row_payload(row) for row in top_pareto],
        "rows": [_frontier_row_payload(row) for row in rows],
    }
    payload["sweep_hash"] = stable_signature(
        {
            "version": payload["sweep_version"],
            "source_fingerprints": payload["source_fingerprints"],
            "training_window": payload["training_window"],
            "top_portfolio": payload["top_portfolio_proxy"][:10],
            "top_mfe": payload["top_mfe"][:10],
        }
    )
    json_path = out / f"kalcb_premarket_frontier_only_sweep_{payload['sweep_hash'][:12]}.json"
    md_path = out / f"kalcb_premarket_frontier_only_sweep_{payload['sweep_hash'][:12]}.md"
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_frontier_markdown(payload), encoding="utf-8")
    _write_frontier_progress(out, "completed", len(rows), len(rows), rows)
    return payload


def run_premarket_frontier_sweep(
    config: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    max_workers: int = 4,
    first30_top_n: int = 8,
    refine_first30_top_n: int = 6,
    max_first30_coarse_specs: int | None = None,
    first30_artifact: str | Path | None = None,
    coarse_frontier_limit: int = 640,
    deep_pair_count: int = 16,
    deep_per_mode_limit: int = 120,
    max_frontier_specs: int | None = None,
    frontier_artifact: str | Path | None = None,
    frontier_top_n: int = 0,
) -> dict[str, Any]:
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    training_config = _training_config(dict(config), holdout_days)
    _record_setup_progress(out, "prepare_dataset_started")
    dataset = prepare_first30_dataset(training_config)
    _record_setup_progress(
        out,
        "prepare_dataset_completed",
        sessions=len(dataset.trading_dates),
        symbols=len(dataset.data_available_symbols),
        bars_by_key=len(dataset.bars_by_key),
    )
    cfg = KALCBConfig.from_mapping(training_config, {})
    _record_setup_progress(out, "build_contexts_started")
    contexts = build_contexts(dataset)
    _record_setup_progress(out, "build_contexts_completed", days=len(contexts), contexts=sum(len(items) for items in contexts.values()))
    _record_setup_progress(out, "build_premarket_features_started")
    features = build_premarket_features(contexts)
    _record_setup_progress(out, "build_premarket_features_completed", features=sum(len(items) for items in features.values()))
    _record_setup_progress(out, "build_opportunity_map_started")
    opportunity_by_key = build_opportunity_map(dataset, contexts, cfg)
    opportunity_by_day = _opportunity_by_day(opportunity_by_key.values())
    _record_setup_progress(out, "build_opportunity_map_completed", opportunities=len(opportunity_by_key))
    first30_leaderboard_payload: list[dict[str, Any]]
    if first30_artifact:
        _record_setup_progress(out, "load_first30_artifact_started", first30_artifact=str(first30_artifact))
        first30_leaderboard_payload, promoted_first30 = load_first30_artifact(first30_artifact, first30_top_n=first30_top_n)
        _record_setup_progress(out, "load_first30_artifact_completed", promoted_first30=len(promoted_first30))
    else:
        _record_setup_progress(out, "promote_first30_started")
        first30_rows, promoted_first30 = promote_first30_specs(
            dataset,
            contexts,
            cfg,
            output_dir=out,
            first30_top_n=first30_top_n,
            refine_top_n=refine_first30_top_n,
            max_coarse_specs=max_first30_coarse_specs,
            max_workers=max_workers,
        )
        first30_leaderboard_payload = [_first30_row_payload(row) for row in first30_rows[:30]]
        _record_setup_progress(out, "promote_first30_completed", rows=len(first30_rows), promoted_first30=len(promoted_first30))
    full_first30 = {spec.name: select_first30(spec, contexts) for spec in promoted_first30}
    references = build_reference_rows(
        dataset=dataset,
        cfg=cfg,
        promoted_first30=promoted_first30,
        full_first30=full_first30,
        opportunity_by_key=opportunity_by_key,
        opportunity_by_day=opportunity_by_day,
    )
    reference_by_first30 = {item["first30"]["name"]: item for item in references}
    if frontier_artifact:
        frontier_specs = load_frontier_finalists(frontier_artifact, frontier_top_n=max(1, int(frontier_top_n or 5)))
    else:
        frontier_specs = build_frontier_specs()
    if max_frontier_specs is not None:
        frontier_specs = _even_sample(frontier_specs, max(1, int(max_frontier_specs)))
    coarse_frontiers = _stratified_frontier_sample(frontier_specs, coarse_frontier_limit)
    coarse_pairs = build_pair_specs(promoted_first30, coarse_frontiers)
    frontier_cache: dict[str, dict[date, tuple[str, ...]]] = {}
    rows = _evaluate_pair_specs(
        coarse_pairs,
        features,
        contexts,
        dataset,
        cfg,
        full_first30,
        reference_by_first30,
        opportunity_by_key,
        opportunity_by_day,
        frontier_cache,
        out,
        stage="coarse",
        completed_offset=0,
        total=len(coarse_pairs),
        max_workers=max_workers,
    )
    deep_pairs = build_deep_pair_specs(
        rows,
        promoted_first30=promoted_first30,
        all_frontiers=frontier_specs,
        deep_pair_count=deep_pair_count,
        deep_per_mode_limit=deep_per_mode_limit,
    )
    existing_pairs = {_pair_signature(row.spec) for row in rows}
    deep_pairs = [pair for pair in deep_pairs if _pair_signature(pair) not in existing_pairs]
    deep_rows = _evaluate_pair_specs(
        deep_pairs,
        features,
        contexts,
        dataset,
        cfg,
        full_first30,
        reference_by_first30,
        opportunity_by_key,
        opportunity_by_day,
        frontier_cache,
        out,
        stage="deep",
        completed_offset=len(rows),
        total=len(rows) + len(deep_pairs),
        max_workers=max_workers,
        seed_rows=rows,
    )
    rows = [*rows, *deep_rows]
    rows = _dedupe_pair_results(rows)
    _assign_pareto_scores(rows)
    rows.sort(key=lambda row: (-row.combined_score, row.rejected, -row.pareto_score, row.spec.name))
    top_portfolio = sorted(
        rows,
        key=lambda row: (
            row.rejected,
            -row.metrics.get("portfolio_proxy_net_return_pct", 0.0),
            row.metrics.get("frontier_avg_size", 999.0),
            row.spec.name,
        ),
    )[:30]
    top_slot = sorted(
        rows,
        key=lambda row: (
            row.rejected,
            -row.metrics.get("slot_cumulative_gross_return_pct", 0.0),
            row.metrics.get("frontier_avg_size", 999.0),
            row.spec.name,
        ),
    )[:30]
    top_mfe = sorted(
        rows,
        key=lambda row: (
            row.rejected,
            -row.metrics.get("avg_mfe_r", 0.0),
            -row.metrics.get("slot_cumulative_gross_return_pct", 0.0),
            row.metrics.get("frontier_avg_size", 999.0),
            row.spec.name,
        ),
    )[:30]
    top_pareto = sorted(rows, key=lambda row: (-row.pareto_score, row.rejected, row.metrics.get("frontier_avg_size", 999.0), row.spec.name))[:30]
    payload = {
        "strategy": "kalcb",
        "sweep_version": PREMARKET_FRONTIER_SWEEP_VERSION,
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "training_window": {
            "start": dataset.trading_dates[0].isoformat(),
            "end": dataset.trading_dates[-1].isoformat(),
            "sessions": len(dataset.trading_dates),
        },
        "holdout_days": int(holdout_days),
        "causality_policy": {
            "premarket_frontier": "prior completed daily_ohlcv, daily_flow, index_ohlcv, and sector-map data only",
            "first30_gate": "completed 09:00-09:25 KST bars only",
            "entry": "09:30 KST bar open",
            "evaluation": "09:30-to-configured-flatten bars only; post-entry bars are never visible to selectors",
            "official_performance": False,
        },
        "source_fingerprints": {
            "intraday": dataset.source_fingerprint,
            "daily_lrs": dataset.daily_source_fingerprint,
            "combined": stable_signature([dataset.source_fingerprint, dataset.daily_source_fingerprint]),
        },
        "cost_policy": {
            "round_trip_cost_pct": _round_trip_cost_pct(cfg),
            "slippage_bps_each_side": cfg.slippage_bps,
            "commission_bps_each_side": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
        },
        "selection_policy": {
            "frontier_sizes": list(FRONTIER_SIZES),
            "score_modes": list(_frontier_modes()),
            "objective": "portfolio-aware first30 baseline net return first, Avg MFE R second; gross Slot Return remains an opportunity diagnostic",
            "portfolio_proxy_policy": STAGE2_PORTFOLIO_POLICY,
            "first30_promoted_count": len(promoted_first30),
            "first30_artifact": str(first30_artifact) if first30_artifact else "",
            "frontier_artifact": str(frontier_artifact) if frontier_artifact else "",
            "frontier_artifact_top_n": int(frontier_top_n or 0),
            "coarse_frontier_count": len(coarse_frontiers),
            "deep_pair_count": len(deep_pairs),
            "max_workers": min(max(1, int(max_workers)), 4),
        },
        "data_policy": {
            "data_root": str(dataset.data_root),
            "daily_data_root": str(dataset.daily_data_root),
            "symbols": len(dataset.symbols),
            "daily_available_symbols": len(dataset.daily_available_symbols),
            "combined_flow_available_symbols": len(dataset.flow_by_symbol),
            "foreign_flow_available_symbols": len(dataset.foreign_flow_by_symbol),
            "institutional_flow_available_symbols": len(dataset.institutional_flow_by_symbol),
            "unavailable_symbols": list(dataset.unavailable_symbols),
            "index_codes": sorted(dataset.index_by_code),
        },
        "first30_leaderboard": first30_leaderboard_payload,
        "references": references,
        "candidate_count": len(rows),
        "top_combined": [_row_payload(row) for row in rows[:30]],
        "top_portfolio_proxy": [_row_payload(row) for row in top_portfolio],
        "top_slot_return": [_row_payload(row) for row in top_slot],
        "top_mfe": [_row_payload(row) for row in top_mfe],
        "top_pareto": [_row_payload(row) for row in top_pareto],
        "rows": [_row_payload(row) for row in rows],
    }
    payload["sweep_hash"] = stable_signature(
        {
            "version": PREMARKET_FRONTIER_SWEEP_VERSION,
            "source_fingerprints": payload["source_fingerprints"],
            "training_window": payload["training_window"],
            "top_portfolio": payload["top_portfolio_proxy"][:10],
            "top_slot": payload["top_slot_return"][:10],
            "top_mfe": payload["top_mfe"][:10],
        }
    )
    json_path = out / f"kalcb_premarket_frontier_sweep_{payload['sweep_hash'][:12]}.json"
    md_path = out / f"kalcb_premarket_frontier_sweep_{payload['sweep_hash'][:12]}.md"
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    _write_progress(out, "completed", len(rows), len(rows), rows)
    return payload


def promote_first30_specs(
    dataset: KALCBFirst30Dataset,
    contexts: dict[date, tuple[First30Context, ...]],
    cfg: KALCBConfig,
    *,
    output_dir: Path,
    first30_top_n: int,
    refine_top_n: int,
    max_coarse_specs: int | None,
    max_workers: int,
) -> tuple[list[Any], list[First30Spec]]:
    folds = _two_folds(list(dataset.trading_dates))
    coarse_specs = build_coarse_specs()
    if max_coarse_specs is not None:
        coarse_specs = _even_sample(coarse_specs, max(1, int(max_coarse_specs)))
    coarse_rows = _evaluate_first30_specs(
        coarse_specs,
        contexts,
        dataset,
        cfg,
        folds,
        max_workers=max_workers,
        output_dir=output_dir,
        stage="first30_coarse",
        completed_offset=0,
        total=len(coarse_specs),
    )
    coarse_rows.sort(key=lambda row: (-row.score, row.rejected, row.spec.name))
    seeds = [row.spec for row in coarse_rows if not row.rejected][: max(0, int(refine_top_n))]
    refine_specs = build_refinement_specs(seeds, existing={_spec_signature(row.spec) for row in coarse_rows}) if seeds else []
    refine_rows = _evaluate_first30_specs(
        refine_specs,
        contexts,
        dataset,
        cfg,
        folds,
        max_workers=max_workers,
        output_dir=output_dir,
        stage="first30_refinement",
        completed_offset=len(coarse_rows),
        total=len(coarse_rows) + len(refine_specs),
        seed_rows=coarse_rows,
    )
    rows = [*coarse_rows, *refine_rows]
    rows.sort(key=lambda row: (-row.score, row.rejected, row.spec.name))
    promoted = _dedupe_specs([row.spec for row in rows if not row.rejected])[: max(1, int(first30_top_n))]
    if not promoted:
        promoted = _dedupe_specs([row.spec for row in rows])[: max(1, int(first30_top_n))]
    _write_first30_promotion(output_dir, rows, promoted)
    return rows, promoted


def load_first30_artifact(path: str | Path, *, first30_top_n: int) -> tuple[list[dict[str, Any]], list[First30Spec]]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(artifact.get("top_results") or artifact.get("rows") or [])
    specs: list[First30Spec] = []
    for row in rows:
        spec_payload = row.get("spec") if isinstance(row, dict) else None
        if not isinstance(spec_payload, dict):
            continue
        specs.append(_first30_spec_from_payload(spec_payload))
        if len(specs) >= max(1, int(first30_top_n)):
            break
    if not specs:
        raise ValueError(f"No first30 specs found in artifact: {path}")
    return rows[:30], _dedupe_specs(specs)[: max(1, int(first30_top_n))]


def _first30_spec_from_payload(payload: dict[str, Any]) -> First30Spec:
    allowed = set(First30Spec.__dataclass_fields__)
    values = {key: value for key, value in payload.items() if key in allowed}
    return First30Spec(**values)


def build_premarket_features(contexts: dict[date, tuple[First30Context, ...]]) -> dict[date, tuple[PremarketFeature, ...]]:
    by_day: dict[date, list[PremarketFeature]] = {}
    for day, items in contexts.items():
        for ctx in items:
            daily = ctx.daily
            flow = ctx.flow
            market = ctx.market
            close = max(daily.prev_close, 1e-9)
            by_day.setdefault(day, []).append(
                PremarketFeature(
                    day=day,
                    symbol=ctx.symbol,
                    sector=ctx.sector,
                    ret5=daily.return_5d,
                    ret20=daily.return_20d,
                    ret60=daily.return_60d,
                    atr_pct=daily.atr14 / close,
                    adv20_krw=daily.adv20_krw,
                    close20_loc=daily.close20_loc,
                    close60_loc=daily.close60_loc,
                    volume_surge=daily.volume_ratio_20d,
                    above_sma20=daily.above_sma20,
                    above_sma60=daily.above_sma60,
                    flow_1d=flow.combined_1d,
                    flow_3d=flow.combined_3d,
                    flow_5d=flow.combined_5d,
                    flow_20d=flow.combined_20d,
                    flow_notional_5d=flow.combined_notional_5d,
                    flow_positive_days_5d=flow.positive_days_5d,
                    flow_acceleration=flow.acceleration,
                    flow_z=flow.z_score,
                    flow_available=flow.available,
                    foreign_5d=flow.foreign_5d,
                    foreign_z=flow.foreign_z,
                    foreign_acceleration=flow.foreign_acceleration,
                    foreign_positive_days_5d=flow.foreign_positive_days_5d,
                    inst_5d=flow.inst_5d,
                    inst_z=flow.inst_z,
                    inst_acceleration=flow.inst_acceleration,
                    inst_positive_days_5d=flow.inst_positive_days_5d,
                    flow_agreement_5d=flow.agreement_5d,
                    flow_divergence_5d=flow.divergence_5d,
                    sponsorship_balance_5d=flow.sponsorship_balance_5d,
                    sector_flow_5d=flow.sector_flow_5d,
                    sector_foreign_5d=flow.sector_foreign_5d,
                    sector_inst_5d=flow.sector_inst_5d,
                    sector_agreement_5d=flow.sector_agreement_5d,
                    sector_participation=flow.sector_participation,
                    market_score=market.score,
                    market_ret5=(market.kospi_ret_5d + market.kosdaq_ret_5d) / 2.0,
                    market_ret20=(market.kospi_ret_20d + market.kosdaq_ret_20d) / 2.0,
                    market_above_sma20=market.kospi_above_sma20 or market.kosdaq_above_sma20,
                )
            )
    return {day: tuple(sorted(items, key=lambda item: item.symbol)) for day, items in by_day.items()}


def build_frontier_specs() -> list[FrontierSpec]:
    specs: list[FrontierSpec] = []
    filters = (
        {},
        {"min_adv20_krw": 2_000_000_000.0},
        {"min_adv20_krw": 5_000_000_000.0},
        {"min_adv20_krw": 10_000_000_000.0},
        {"min_ret5": 0.0},
        {"min_ret5": 0.03},
        {"min_ret5": 0.05},
        {"min_ret5": 0.10},
        {"min_ret20": 0.0},
        {"min_ret20": 0.03},
        {"min_ret20": 0.05},
        {"min_ret60": 0.0},
        {"min_ret5": 0.03, "min_ret20": 0.0},
        {"min_ret5": 0.05, "min_ret20": 0.03},
        {"min_ret5": 0.10, "min_ret20": 0.03},
        {"max_ret20": 0.30},
        {"max_ret20": 0.50},
        {"min_ret5": 0.03, "max_ret20": 0.50},
        {"min_close20_loc": 0.50},
        {"min_close20_loc": 0.70},
        {"min_close20_loc": 0.85},
        {"min_ret5": 0.03, "min_close20_loc": 0.50},
        {"max_atr_pct": 0.05},
        {"max_atr_pct": 0.08},
        {"max_atr_pct": 0.12},
        {"min_volume_surge": 1.10},
        {"min_volume_surge": 1.50},
        {"min_flow_5d": 0.0, "require_flow_available": True},
        {"min_flow_5d": 0.005, "require_flow_available": True},
        {"min_flow_z": 0.0, "require_flow_available": True},
        {"min_flow_z": 0.5, "require_flow_available": True},
        {"min_flow_acceleration": 0.0, "require_flow_available": True},
        {"min_foreign_flow_5d": 0.0, "require_flow_available": True},
        {"min_inst_flow_5d": 0.0, "require_flow_available": True},
        {"min_foreign_flow_5d": 0.0, "min_inst_flow_5d": 0.0, "require_flow_available": True},
        {"min_foreign_z": 0.0, "require_flow_available": True},
        {"min_inst_z": 0.0, "require_flow_available": True},
        {"min_flow_agreement": 0.0, "require_flow_available": True},
        {"min_flow_agreement": 0.003, "require_flow_available": True},
        {"max_flow_divergence": 0.01, "require_flow_available": True},
        {"min_sector_flow": 0.0, "require_flow_available": True},
        {"min_sector_flow": 0.005, "require_flow_available": True},
        {"min_sector_participation": 0.50, "require_flow_available": True},
        {"min_market_score": 0.0},
        {"min_market_score": 0.2},
        {"require_above_sma20": True},
        {"require_above_sma60": True},
        {"require_above_sma20": True, "min_flow_5d": 0.0, "require_flow_available": True},
        {"min_ret5": 0.03, "min_flow_z": 0.0, "require_flow_available": True},
        {"min_ret5": 0.03, "min_foreign_flow_5d": 0.0, "require_flow_available": True},
        {"min_ret5": 0.03, "min_inst_flow_5d": 0.0, "require_flow_available": True},
        {"min_ret5": 0.03, "min_flow_agreement": 0.0, "require_flow_available": True},
        {"min_ret5": 0.03, "min_sector_flow": 0.0, "require_flow_available": True},
        {"min_adv20_krw": 5_000_000_000.0, "min_flow_5d": 0.0, "require_flow_available": True},
    )
    seen: set[str] = set()
    for mode in _frontier_modes():
        for size in FRONTIER_SIZES:
            for values in filters:
                spec = name_frontier(FrontierSpec(name="", mode=mode, frontier_size=size, **values))
                signature = _frontier_signature(spec)
                if signature in seen:
                    continue
                seen.add(signature)
                specs.append(spec)
    return specs


def build_pair_specs(first30_specs: Iterable[First30Spec], frontier_specs: Iterable[FrontierSpec]) -> list[PairSpec]:
    pairs = []
    for frontier in frontier_specs:
        for first30 in first30_specs:
            pairs.append(PairSpec(name=f"{frontier.name}__{first30.name}", frontier=frontier, first30=first30))
    return pairs


def build_deep_frontier_specs(
    rows: list[FrontierResult],
    *,
    all_frontiers: list[FrontierSpec],
    deep_pair_count: int,
    deep_per_mode_limit: int,
) -> list[FrontierSpec]:
    if not rows or deep_pair_count <= 0:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            row.rejected,
            -row.combined_score,
            -row.metrics.get("portfolio_proxy_net_return_pct", 0.0),
            -row.metrics.get("avg_mfe_r", 0.0),
            row.metrics.get("frontier_avg_size", 999.0),
            row.spec.name,
        ),
    )
    seeds = ranked[: int(deep_pair_count)]
    mode_set = {row.spec.mode for row in seeds}
    size_set = {size for row in seeds for size in _near_size(row.spec.frontier_size)}
    frontiers: list[FrontierSpec] = []
    for mode in sorted(mode_set):
        mode_frontiers = [item for item in all_frontiers if item.mode == mode and item.frontier_size in size_set]
        if deep_per_mode_limit > 0:
            frontiers.extend(_even_sample(mode_frontiers, min(len(mode_frontiers), int(deep_per_mode_limit))))
    frontiers.extend(build_refined_frontiers([row.spec for row in seeds], existing={_frontier_signature(item) for item in frontiers}))
    return _dedupe_frontiers([*frontiers, *(row.spec for row in seeds)])


def build_deep_pair_specs(
    rows: list[PairResult],
    *,
    promoted_first30: list[First30Spec],
    all_frontiers: list[FrontierSpec],
    deep_pair_count: int,
    deep_per_mode_limit: int,
) -> list[PairSpec]:
    if not rows or deep_pair_count <= 0:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            row.rejected,
            -row.combined_score,
            -row.metrics.get("slot_cumulative_gross_return_pct", 0.0),
            -row.metrics.get("avg_mfe_r", 0.0),
            row.metrics.get("frontier_avg_size", 999.0),
        ),
    )
    seeds = ranked[: int(deep_pair_count)]
    first30_names = {row.spec.first30.name for row in seeds}
    seed_first30 = [spec for spec in promoted_first30 if spec.name in first30_names] or promoted_first30[:3]
    mode_set = {row.spec.frontier.mode for row in seeds}
    size_set = {size for row in seeds for size in _near_size(row.spec.frontier.frontier_size)}
    frontiers: list[FrontierSpec] = []
    for mode in sorted(mode_set):
        mode_frontiers = [item for item in all_frontiers if item.mode == mode and item.frontier_size in size_set]
        if deep_per_mode_limit > 0:
            frontiers.extend(_even_sample(mode_frontiers, min(len(mode_frontiers), int(deep_per_mode_limit))))
    frontiers.extend(build_refined_frontiers([row.spec.frontier for row in seeds], existing={_frontier_signature(item) for item in frontiers}))
    frontiers = _dedupe_frontiers([*frontiers, *(row.spec.frontier for row in seeds)])
    return build_pair_specs(seed_first30, frontiers)


def build_refined_frontiers(seeds: list[FrontierSpec], *, existing: set[str]) -> list[FrontierSpec]:
    specs: list[FrontierSpec] = []
    for seed in seeds:
        for size in _near_size(seed.frontier_size):
            specs.append(replace(seed, name="", frontier_size=int(size)))
        for ret5 in _near(seed.min_ret5, (-1.0, 0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15)):
            specs.append(replace(seed, name="", min_ret5=float(ret5)))
        for ret20 in _near(seed.min_ret20, (-1.0, 0.0, 0.03, 0.05, 0.08, 0.12)):
            specs.append(replace(seed, name="", min_ret20=float(ret20)))
        for close20 in _near(seed.min_close20_loc, (0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)):
            specs.append(replace(seed, name="", min_close20_loc=float(close20)))
        for atr in _near(seed.max_atr_pct, (0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 9.99)):
            specs.append(replace(seed, name="", max_atr_pct=float(atr)))
        for flow in _near(seed.min_flow_5d, (-9.99, -0.01, 0.0, 0.003, 0.005, 0.01, 0.02)):
            specs.append(replace(seed, name="", min_flow_5d=float(flow), require_flow_available=seed.require_flow_available or flow > -9))
        for flow_z in _near(seed.min_flow_z, (-9.99, -0.5, 0.0, 0.3, 0.5, 0.8, 1.0)):
            specs.append(replace(seed, name="", min_flow_z=float(flow_z), require_flow_available=seed.require_flow_available or flow_z > -9))
        for foreign_flow in _near(seed.min_foreign_flow_5d, (-9.99, -0.01, 0.0, 0.003, 0.005, 0.01, 0.02)):
            specs.append(replace(seed, name="", min_foreign_flow_5d=float(foreign_flow), require_flow_available=seed.require_flow_available or foreign_flow > -9))
        for inst_flow in _near(seed.min_inst_flow_5d, (-9.99, -0.01, 0.0, 0.003, 0.005, 0.01, 0.02)):
            specs.append(replace(seed, name="", min_inst_flow_5d=float(inst_flow), require_flow_available=seed.require_flow_available or inst_flow > -9))
        for agreement in _near(seed.min_flow_agreement, (-9.99, -0.005, 0.0, 0.002, 0.003, 0.005, 0.01)):
            specs.append(replace(seed, name="", min_flow_agreement=float(agreement), require_flow_available=seed.require_flow_available or agreement > -9))
        for divergence in _near(seed.max_flow_divergence, (0.005, 0.01, 0.02, 0.04, 9.99)):
            specs.append(replace(seed, name="", max_flow_divergence=float(divergence), require_flow_available=seed.require_flow_available or divergence < 9))
        for sector_flow in _near(seed.min_sector_flow, (-9.99, -0.005, 0.0, 0.003, 0.005, 0.01)):
            specs.append(replace(seed, name="", min_sector_flow=float(sector_flow), require_flow_available=seed.require_flow_available or sector_flow > -9))
        for market in _near(seed.min_market_score, (-9.99, -0.2, 0.0, 0.2, 0.4)):
            specs.append(replace(seed, name="", min_market_score=float(market)))
    named = [name_frontier(spec) for spec in specs]
    return [spec for spec in _dedupe_frontiers(named) if _frontier_signature(spec) not in existing]


def build_opportunity_map(
    dataset: KALCBFirst30Dataset,
    contexts: dict[date, tuple[First30Context, ...]],
    cfg: KALCBConfig,
) -> dict[tuple[date, str], OpportunityRow]:
    selections = [
        Selection(day, ctx.symbol, 0.0, "all_context")
        for day, items in contexts.items()
        for ctx in items
    ]
    return {(row.trade_date, row.symbol): row for row in evaluate_selections(dataset, selections, cfg)}


def select_frontier(spec: FrontierSpec, features: dict[date, tuple[PremarketFeature, ...]]) -> dict[date, tuple[str, ...]]:
    ranked = select_frontier_ranked(spec, features)
    return {day: tuple(symbol for symbol, _ in rows) for day, rows in ranked.items()}


def select_frontier_ranked(spec: FrontierSpec, features: dict[date, tuple[PremarketFeature, ...]]) -> dict[date, tuple[tuple[str, float], ...]]:
    out: dict[date, tuple[tuple[str, float], ...]] = {}
    for day, items in features.items():
        scored = []
        for feature in items:
            score = score_frontier(spec, feature)
            if score is not None:
                scored.append((score, feature.symbol))
        scored.sort(key=lambda item: (-item[0], item[1]))
        out[day] = tuple((symbol, float(score)) for score, symbol in scored[: max(1, int(spec.frontier_size))])
    return out


def select_first30_from_frontier(
    frontier: dict[date, tuple[str, ...]],
    first30: First30Spec,
    contexts: dict[date, tuple[First30Context, ...]],
) -> list[Selection]:
    selections: list[Selection] = []
    for day, symbols in frontier.items():
        allowed = set(symbols)
        scored = []
        for ctx in contexts.get(day, ()):
            if ctx.symbol not in allowed or not passes(first30, ctx):
                continue
            scored.append((score_candidate(first30, ctx), ctx.symbol))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selections.extend(Selection(day, symbol, score, first30.score_mode) for score, symbol in scored[: max(1, first30.top_n)])
    return selections


def score_frontier(spec: FrontierSpec, feature: PremarketFeature) -> float | None:
    if feature.ret5 < spec.min_ret5:
        return None
    if feature.ret20 < spec.min_ret20 or feature.ret20 > spec.max_ret20:
        return None
    if feature.ret60 < spec.min_ret60:
        return None
    if feature.close20_loc < spec.min_close20_loc:
        return None
    if feature.adv20_krw < spec.min_adv20_krw:
        return None
    if feature.atr_pct > spec.max_atr_pct:
        return None
    if feature.volume_surge < spec.min_volume_surge:
        return None
    if feature.flow_5d < spec.min_flow_5d:
        return None
    if feature.flow_z < spec.min_flow_z:
        return None
    if feature.flow_acceleration < spec.min_flow_acceleration:
        return None
    if feature.foreign_5d < spec.min_foreign_flow_5d:
        return None
    if feature.inst_5d < spec.min_inst_flow_5d:
        return None
    if feature.foreign_z < spec.min_foreign_z:
        return None
    if feature.inst_z < spec.min_inst_z:
        return None
    if feature.flow_agreement_5d < spec.min_flow_agreement:
        return None
    if feature.flow_divergence_5d > spec.max_flow_divergence:
        return None
    if feature.sector_flow_5d < spec.min_sector_flow:
        return None
    if feature.sector_participation < spec.min_sector_participation:
        return None
    if feature.market_score < spec.min_market_score:
        return None
    if spec.require_above_sma20 and not feature.above_sma20:
        return None
    if spec.require_above_sma60 and not feature.above_sma60:
        return None
    if spec.require_flow_available and not feature.flow_available:
        return None
    liquidity = 0.01 * _clip(math.log10(max(feature.adv20_krw, 1.0) / 2_000_000_000.0 + 1.0), 0.0, 2.0)
    rs = 0.45 * feature.ret5 + 0.35 * feature.ret20 + 0.20 * feature.ret60
    flow = (
        0.38 * feature.flow_5d
        + 0.14 * feature.flow_20d
        + 0.16 * feature.foreign_5d
        + 0.16 * feature.inst_5d
        + 0.01 * feature.flow_z
        + 0.004 * feature.foreign_z
        + 0.004 * feature.inst_z
        + 0.28 * feature.flow_acceleration
        + 0.16 * feature.foreign_acceleration
        + 0.16 * feature.inst_acceleration
        + 0.40 * feature.flow_agreement_5d
        - 0.12 * feature.flow_divergence_5d
        + 0.001 * (feature.foreign_positive_days_5d + feature.inst_positive_days_5d)
    )
    sector_flow = 0.50 * feature.sector_flow_5d + 0.15 * feature.sector_foreign_5d + 0.15 * feature.sector_inst_5d + 0.30 * feature.sector_agreement_5d + 0.02 * feature.sector_participation
    trend_quality = 0.50 * feature.close20_loc + 0.25 * feature.close60_loc - 1.5 * feature.atr_pct
    market = 0.01 * feature.market_score + 0.10 * max(feature.ret5 - feature.market_ret5, 0.0)
    if spec.mode == "rs_trend":
        return rs + 0.02 * trend_quality + liquidity
    if spec.mode == "liquidity_momentum":
        return 0.55 * feature.ret5 + 0.25 * feature.ret20 + 0.01 * min(feature.volume_surge, 5.0) + 2.5 * liquidity
    if spec.mode == "compression_breakout":
        return 0.40 * feature.ret5 + 0.25 * feature.ret20 + 0.04 * feature.close20_loc - 0.80 * feature.atr_pct + 0.005 * min(feature.volume_surge, 5.0)
    if spec.mode == "flow_accumulation":
        return 0.40 * rs + 0.80 * flow + 0.50 * sector_flow + liquidity
    if spec.mode == "flow_inflection":
        return 0.35 * feature.ret5 + 0.80 * feature.flow_acceleration + 0.40 * feature.foreign_acceleration + 0.40 * feature.inst_acceleration + 0.015 * feature.flow_z + 0.30 * feature.flow_1d + 0.20 * sector_flow
    if spec.mode == "foreign_accumulation":
        return 0.42 * rs + 0.90 * feature.foreign_5d + 0.35 * feature.foreign_acceleration + 0.012 * feature.foreign_z + 0.25 * sector_flow + liquidity
    if spec.mode == "institutional_accumulation":
        return 0.42 * rs + 0.90 * feature.inst_5d + 0.35 * feature.inst_acceleration + 0.012 * feature.inst_z + 0.25 * sector_flow + liquidity
    if spec.mode == "flow_synergy":
        return 0.35 * rs + 1.15 * feature.flow_agreement_5d + 0.45 * flow + 0.55 * sector_flow + 0.03 * feature.close20_loc
    if spec.mode == "flow_dissynergy":
        leader_side = max(feature.foreign_5d, feature.inst_5d)
        return 0.35 * rs + 0.70 * leader_side + 0.50 * feature.flow_divergence_5d + 0.08 * max(feature.sponsorship_balance_5d, 0.0) + market
    if spec.mode == "sector_flow_participation":
        return 0.35 * rs + 1.10 * sector_flow + 0.40 * feature.flow_5d + 0.04 * feature.close20_loc
    if spec.mode == "index_confirmed_leadership":
        return 0.65 * rs + market + 0.01 * (1.0 if feature.market_above_sma20 else -1.0) + 0.25 * flow
    if spec.mode == "hybrid":
        return 0.45 * rs + 0.20 * trend_quality + 0.35 * flow + 0.20 * sector_flow + market + liquidity
    return None


def evaluate_frontier_spec(
    spec: FrontierSpec,
    features: dict[date, tuple[PremarketFeature, ...]],
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    opportunity_by_key: dict[tuple[date, str], OpportunityRow],
    opportunity_by_day: dict[date, list[OpportunityRow]],
    frontier_cache: dict[str, dict[date, tuple[str, ...]]],
    score_cache: dict[str, dict[date, dict[str, float]]],
) -> FrontierResult:
    if spec.name not in frontier_cache:
        ranked = select_frontier_ranked(spec, features)
        frontier_cache[spec.name] = {day: tuple(symbol for symbol, _ in rows) for day, rows in ranked.items()}
        score_cache[spec.name] = {day: {symbol: score for symbol, score in rows} for day, rows in ranked.items()}
    frontier = frontier_cache[spec.name]
    scores = score_cache.get(spec.name, {})
    rows: list[OpportunityRow] = []
    for day in dataset.trading_dates:
        day_scores = scores.get(day, {})
        for symbol in frontier.get(day, ()):
            base = opportunity_by_key.get((day, symbol))
            if base is None:
                continue
            rows.append(replace(base, family=spec.mode, score=float(day_scores.get(symbol, 0.0))))
    summary = summarize(spec.name, rows, session_dates=dataset.trading_dates, slot_count=spec.frontier_size)
    metrics = {
        **_compact_summary(summary),
        **frontier_stats(frontier, dataset, cfg),
        **mfe_rank_metrics(rows, opportunity_by_day),
    }
    return_score, mfe_score, reject = score_frontier_metrics(metrics)
    size_bonus = 2.5 * _clip((103.0 - metrics.get("frontier_avg_size", 103.0)) / 103.0)
    feasibility_bonus = 1.0 if metrics.get("ws_hot_feasible", 0.0) >= 1.0 else 0.0
    combined = 0.58 * return_score + 0.37 * mfe_score + 5.0 * _clip(metrics.get("top5_recall", 0.0))
    if not reject:
        combined += size_bonus + feasibility_bonus
    rejected = bool(reject)
    return FrontierResult(
        spec=spec,
        return_score=round(0.0 if rejected else return_score, 6),
        mfe_score=round(0.0 if rejected else mfe_score, 6),
        combined_score=round(0.0 if rejected else combined, 6),
        pareto_score=0.0,
        rejected=rejected,
        reject_reason=reject,
        metrics=metrics,
    )


def score_frontier_metrics(metrics: dict[str, float]) -> tuple[float, float, str]:
    candidate_days = metrics.get("candidate_days", 0.0)
    active_share = metrics.get("active_day_share", 0.0)
    if candidate_days < 60:
        return 0.0, 0.0, f"too_few_candidate_days ({candidate_days:.0f} < 60)"
    if active_share < 0.20:
        return 0.0, 0.0, f"too_sparse ({active_share:.3f} < 0.200)"
    portfolio_net = metrics.get("portfolio_proxy_net_return_pct", metrics.get("slot_cumulative_net_return_pct", 0.0))
    calendar_net = metrics.get("portfolio_proxy_calendar_day_net_pct", metrics.get("calendar_day_net_pct", 0.0))
    active_net = metrics.get("portfolio_proxy_active_day_net_pct", metrics.get("active_day_net_pct", 0.0))
    avg_mfe = metrics.get("avg_mfe_r", 0.0)
    med_mfe = metrics.get("median_mfe_r", 0.0)
    mfe_075 = metrics.get("mfe_ge_0_75_share", 0.0)
    mfe_10 = metrics.get("mfe_ge_1_0_share", 0.0)
    percentile = metrics.get("selected_mfe_percentile", 0.0)
    top5_hit = metrics.get("top5_day_hit_share", 0.0)
    top5_recall = metrics.get("top5_recall", 0.0)
    mae_bad = metrics.get("mae_le_neg_1_share", 0.0)
    dd = abs(min(metrics.get("portfolio_proxy_max_drawdown_pct", metrics.get("slot_max_drawdown_net_pct", 0.0)), 0.0))
    names = metrics.get("avg_candidates_per_active_day", metrics.get("frontier_avg_size", 0.0))
    ws_feasible = metrics.get("ws_hot_feasible", 0.0)
    return_score = 100.0 * (
        0.42 * _clip(portfolio_net / 0.45)
        + 0.16 * _return_score(calendar_net, target=0.0025)
        + 0.12 * _return_score(active_net, target=0.004)
        + 0.12 * _clip(avg_mfe / 1.50)
        + 0.08 * _clip(top5_hit)
        + 0.05 * _clip(active_share / 0.70)
        + 0.05 * _clip(metrics.get("net_win_share", 0.0))
    )
    mfe_score = 100.0 * (
        0.30 * _clip(avg_mfe / 1.70)
        + 0.14 * _clip(med_mfe / 1.00)
        + 0.15 * _clip(mfe_075)
        + 0.10 * _clip(mfe_10)
        + 0.17 * _clip((percentile - 0.50) / 0.25)
        + 0.09 * _clip(top5_hit)
        + 0.05 * _clip(top5_recall)
    )
    penalty = 100.0 * (
        0.05 * _clip(mae_bad / 0.40)
        + 0.04 * _clip(dd / 0.12)
        + 0.02 * _clip(max(0.0, names - STAGE2_PORTFOLIO_POLICY["max_positions"]) / 103.0)
        + 0.01 * (0.0 if ws_feasible >= 1.0 else _clip(names / 103.0))
    )
    return max(0.0, return_score - penalty), max(0.0, mfe_score - 0.50 * penalty), ""


def evaluate_pair_spec(
    spec: PairSpec,
    features: dict[date, tuple[PremarketFeature, ...]],
    contexts: dict[date, tuple[First30Context, ...]],
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    full_first30: dict[str, list[Selection]],
    reference_by_first30: dict[str, dict[str, Any]],
    opportunity_by_key: dict[tuple[date, str], OpportunityRow],
    opportunity_by_day: dict[date, list[OpportunityRow]],
    frontier_cache: dict[str, dict[date, tuple[str, ...]]],
) -> PairResult:
    frontier = frontier_cache[spec.frontier.name]
    selections = select_first30_from_frontier(frontier, spec.first30, contexts)
    rows = [
        opportunity_by_key[(selection.trade_date, selection.symbol)]
        for selection in selections
        if (selection.trade_date, selection.symbol) in opportunity_by_key
    ]
    summary = summarize(spec.name, rows, session_dates=dataset.trading_dates, slot_count=spec.first30.top_n)
    metrics = {
        **_compact_summary(summary),
        **frontier_stats(frontier, dataset, cfg),
        **full_selector_recall(selections, full_first30.get(spec.first30.name, [])),
        **mfe_rank_metrics(rows, opportunity_by_day),
    }
    reference = reference_by_first30.get(spec.first30.name, {})
    if reference:
        metrics["full103_slot_cumulative_gross_return_pct"] = reference.get("slot_cumulative_gross_return_pct", 0.0)
        metrics["opportunity_loss_vs_full103_gross_slot_pct"] = reference.get("slot_cumulative_gross_return_pct", 0.0) - metrics.get("slot_cumulative_gross_return_pct", 0.0)
    return_score, mfe_score, reject = score_pair_metrics(metrics)
    recall_score = _clip(metrics.get("full_first30_candidate_recall", 0.0))
    combined = 0.52 * return_score + 0.40 * mfe_score + 0.08 * 100.0 * recall_score
    if not reject:
        combined += 2.5 * _clip((103.0 - metrics.get("frontier_avg_size", 103.0)) / 103.0)
    rejected = bool(reject)
    return PairResult(
        spec=spec,
        return_score=round(0.0 if rejected else return_score, 6),
        mfe_score=round(0.0 if rejected else mfe_score, 6),
        combined_score=round(0.0 if rejected else combined, 6),
        pareto_score=0.0,
        rejected=rejected,
        reject_reason=reject,
        metrics=metrics,
    )


def score_pair_metrics(metrics: dict[str, float]) -> tuple[float, float, str]:
    candidate_days = metrics.get("candidate_days", 0.0)
    active_share = metrics.get("active_day_share", 0.0)
    if candidate_days < 60:
        return 0.0, 0.0, f"too_few_candidate_days ({candidate_days:.0f} < 60)"
    if active_share < 0.20:
        return 0.0, 0.0, f"too_sparse ({active_share:.3f} < 0.200)"
    portfolio_net = metrics.get("portfolio_proxy_net_return_pct", metrics.get("slot_cumulative_net_return_pct", 0.0))
    calendar_net = metrics.get("portfolio_proxy_calendar_day_net_pct", metrics.get("calendar_day_net_pct", 0.0))
    active_net = metrics.get("portfolio_proxy_active_day_net_pct", metrics.get("active_day_net_pct", 0.0))
    avg_mfe = metrics.get("avg_mfe_r", 0.0)
    med_mfe = metrics.get("median_mfe_r", 0.0)
    mfe_075 = metrics.get("mfe_ge_0_75_share", 0.0)
    mfe_10 = metrics.get("mfe_ge_1_0_share", 0.0)
    percentile = metrics.get("selected_mfe_percentile", 0.0)
    top5_hit = metrics.get("top5_day_hit_share", 0.0)
    top5_recall = metrics.get("top5_recall", 0.0)
    mae_bad = metrics.get("mae_le_neg_1_share", 0.0)
    dd = abs(min(metrics.get("portfolio_proxy_max_drawdown_pct", metrics.get("slot_max_drawdown_net_pct", 0.0)), 0.0))
    opportunity_loss = max(metrics.get("opportunity_loss_vs_full103_gross_slot_pct", 0.0), 0.0)
    names = metrics.get("avg_candidates_per_active_day", 0.0)
    return_score = 100.0 * (
        0.45 * _clip(portfolio_net / 0.45)
        + 0.18 * _return_score(calendar_net, target=0.0025)
        + 0.12 * _return_score(active_net, target=0.004)
        + 0.13 * _clip(avg_mfe / 1.50)
        + 0.07 * _clip(active_share / 0.70)
        + 0.05 * _clip(metrics.get("net_win_share", 0.0))
    )
    mfe_score = 100.0 * (
        0.32 * _clip(avg_mfe / 1.70)
        + 0.14 * _clip(med_mfe / 1.00)
        + 0.16 * _clip(mfe_075)
        + 0.10 * _clip(mfe_10)
        + 0.16 * _clip((percentile - 0.50) / 0.25)
        + 0.08 * _clip(top5_hit)
        + 0.04 * _clip(top5_recall)
    )
    penalty = 100.0 * (
        0.05 * _clip(mae_bad / 0.40)
        + 0.04 * _clip(dd / 0.12)
        + 0.02 * _clip(opportunity_loss / 0.50)
        + 0.015 * _clip(max(0.0, names - STAGE2_PORTFOLIO_POLICY["max_positions"]) / STAGE2_PORTFOLIO_POLICY["max_positions"])
    )
    return max(0.0, return_score - penalty), max(0.0, mfe_score - 0.50 * penalty), ""


def build_reference_rows(
    *,
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    promoted_first30: list[First30Spec],
    full_first30: dict[str, list[Selection]],
    opportunity_by_key: dict[tuple[date, str], OpportunityRow],
    opportunity_by_day: dict[date, list[OpportunityRow]],
) -> list[dict[str, Any]]:
    refs = []
    for spec in promoted_first30:
        selections = full_first30.get(spec.name, [])
        rows = [
            opportunity_by_key[(selection.trade_date, selection.symbol)]
            for selection in selections
            if (selection.trade_date, selection.symbol) in opportunity_by_key
        ]
        summary = summarize(f"full103__{spec.name}", rows, session_dates=dataset.trading_dates, slot_count=spec.top_n)
        refs.append(
            {
                "name": f"full103__{spec.name}",
                "first30": asdict(spec),
                **_compact_summary(summary),
                **mfe_rank_metrics(rows, opportunity_by_day),
                "round_trip_cost_pct": _round_trip_cost_pct(cfg),
            }
        )
    return refs


def mfe_rank_metrics(rows: list[OpportunityRow], opportunity_by_day: dict[date, list[OpportunityRow]]) -> dict[str, float]:
    if not rows:
        return {
            "selected_mfe_percentile": 0.0,
            "top5_day_hit_share": 0.0,
            "top5_recall": 0.0,
            "avg_mfe_pct_of_day_best": 0.0,
        }
    percentiles = []
    best_ratios = []
    selected_keys = {(row.trade_date, row.symbol) for row in rows}
    day_hits = 0
    possible_days = 0
    top5_hits = 0
    top5_total = 0
    selected_by_day: dict[date, list[OpportunityRow]] = {}
    for row in rows:
        selected_by_day.setdefault(row.trade_date, []).append(row)
    for day, selected in selected_by_day.items():
        universe = sorted(opportunity_by_day.get(day, []), key=lambda item: (-item.mfe_r, item.symbol))
        if not universe:
            continue
        possible_days += 1
        top5 = universe[:5]
        top5_keys = {(item.trade_date, item.symbol) for item in top5}
        hit_count = len(top5_keys & selected_keys)
        if hit_count:
            day_hits += 1
        top5_hits += hit_count
        top5_total += len(top5)
        best = max(item.mfe_r for item in universe)
        sorted_mfe = sorted(item.mfe_r for item in universe)
        for row in selected:
            less_equal = sum(1 for value in sorted_mfe if value <= row.mfe_r)
            percentiles.append(less_equal / max(float(len(sorted_mfe)), 1.0))
            best_ratios.append(row.mfe_r / max(best, 1e-9))
    return {
        "selected_mfe_percentile": _avg(percentiles),
        "top5_day_hit_share": day_hits / max(float(possible_days), 1.0),
        "top5_recall": top5_hits / max(float(top5_total), 1.0),
        "avg_mfe_pct_of_day_best": _avg(best_ratios),
    }


def full_selector_recall(selections: list[Selection], full: list[Selection]) -> dict[str, float]:
    selected_keys = {(selection.trade_date, selection.symbol) for selection in selections}
    full_keys = {(selection.trade_date, selection.symbol) for selection in full}
    full_days: dict[date, set[str]] = {}
    selected_days: dict[date, set[str]] = {}
    for day, symbol in full_keys:
        full_days.setdefault(day, set()).add(symbol)
    for day, symbol in selected_keys:
        selected_days.setdefault(day, set()).add(symbol)
    hit_days = sum(1 for day, symbols in full_days.items() if symbols & selected_days.get(day, set()))
    return {
        "full_first30_candidate_recall": len(selected_keys & full_keys) / max(float(len(full_keys)), 1.0),
        "full_first30_active_day_hit_share": hit_days / max(float(len(full_days)), 1.0),
    }


def frontier_stats(frontier: dict[date, tuple[str, ...]], dataset: KALCBFirst30Dataset, cfg: KALCBConfig) -> dict[str, float]:
    counts = [len(frontier.get(day, ())) for day in dataset.trading_dates]
    capacity = hot_symbol_capacity(cfg)
    return {
        "frontier_avg_size": _avg(counts),
        "frontier_max_size": float(max(counts) if counts else 0),
        "frontier_target_size": float(max(counts) if counts else 0),
        "ws_budget": float(cfg.ws_budget),
        "ws_hot_symbol_capacity": float(capacity),
        "ws_hot_feasible": 1.0 if max(counts or [0]) <= capacity else 0.0,
        "ws_hot_regs_required": float(max(counts or [0]) * cfg.ws_hot_regs_per_symbol),
    }


def hot_symbol_capacity(cfg: KALCBConfig) -> int:
    return max(0, (cfg.ws_max_registrations - cfg.ws_reserved_execution_regs) // max(cfg.ws_hot_regs_per_symbol, 1))


def name_frontier(spec: FrontierSpec) -> FrontierSpec:
    parts = [
        spec.mode,
        f"fx{spec.frontier_size}",
        f"r5{_pct_label(spec.min_ret5)}",
        f"r20{_pct_label(spec.min_ret20)}to{_pct_label(spec.max_ret20)}",
        f"r60{_pct_label(spec.min_ret60)}",
        f"cl20{_num_label(spec.min_close20_loc)}",
        f"adv{_adv_label(spec.min_adv20_krw)}",
        f"atr{_pct_label(spec.max_atr_pct)}",
        f"vol{_num_label(spec.min_volume_surge)}",
        f"flow5{_num_label(spec.min_flow_5d)}",
        f"flowz{_num_label(spec.min_flow_z)}",
        f"flowacc{_num_label(spec.min_flow_acceleration)}",
        f"for5{_num_label(spec.min_foreign_flow_5d)}",
        f"inst5{_num_label(spec.min_inst_flow_5d)}",
        f"forz{_num_label(spec.min_foreign_z)}",
        f"instz{_num_label(spec.min_inst_z)}",
        f"agree{_num_label(spec.min_flow_agreement)}",
        f"div{_num_label(spec.max_flow_divergence)}",
        f"secflow{_num_label(spec.min_sector_flow)}",
        f"secpart{_num_label(spec.min_sector_participation)}",
        f"mkt{_num_label(spec.min_market_score)}",
    ]
    if spec.require_above_sma20:
        parts.append("sma20")
    if spec.require_above_sma60:
        parts.append("sma60")
    if spec.require_flow_available:
        parts.append("flowdata")
    return replace(spec, name="_".join(parts).replace("-", "m").replace(".", "p"))


def _evaluate_first30_specs(
    specs: list[First30Spec],
    contexts: dict[date, tuple[First30Context, ...]],
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    folds: list[tuple[date, date]],
    *,
    max_workers: int,
    output_dir: Path | None = None,
    stage: str = "first30",
    completed_offset: int = 0,
    total: int | None = None,
    seed_rows: list[Any] | None = None,
) -> list[Any]:
    if not specs:
        return []
    total_count = int(total if total is not None else len(specs))
    rows: list[Any] = []
    # First30 evaluation is CPU-bound and reuses large in-memory feature caches.
    # Deterministic in-process iteration avoids thread contention while preserving
    # the same causal inputs and ranking semantics.
    for spec in specs:
        row = evaluate_spec(spec, contexts, dataset, cfg, folds)
        rows.append(row)
        if output_dir is not None:
            _record_setup_progress(
                output_dir,
                stage,
                completed=completed_offset + len(rows),
                total=total_count,
                current=_first30_progress_row(row),
                best=_best_first30_progress_row([*(seed_rows or []), *rows]),
            )
    return rows


def _evaluate_pair_specs(
    specs: list[PairSpec],
    features: dict[date, tuple[PremarketFeature, ...]],
    contexts: dict[date, tuple[First30Context, ...]],
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    full_first30: dict[str, list[Selection]],
    reference_by_first30: dict[str, dict[str, Any]],
    opportunity_by_key: dict[tuple[date, str], OpportunityRow],
    opportunity_by_day: dict[date, list[OpportunityRow]],
    frontier_cache: dict[str, dict[date, tuple[str, ...]]],
    output_dir: Path,
    *,
    stage: str,
    completed_offset: int,
    total: int,
    max_workers: int,
    seed_rows: list[PairResult] | None = None,
) -> list[PairResult]:
    if not specs:
        return []
    for frontier in _unique_frontiers(specs):
        if frontier.name not in frontier_cache:
            frontier_cache[frontier.name] = select_frontier(frontier, features)
    rows: list[PairResult] = []
    if max_workers <= 1:
        for spec in specs:
            row = evaluate_pair_spec(spec, features, contexts, dataset, cfg, full_first30, reference_by_first30, opportunity_by_key, opportunity_by_day, frontier_cache)
            rows.append(row)
            _record_progress(output_dir, stage, completed_offset + len(rows), total, [*(seed_rows or []), *rows], row)
        return rows
    max_workers = max(1, min(int(max_workers), 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_pair_spec,
                spec,
                features,
                contexts,
                dataset,
                cfg,
                full_first30,
                reference_by_first30,
                opportunity_by_key,
                opportunity_by_day,
                frontier_cache,
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            _record_progress(output_dir, stage, completed_offset + len(rows), total, [*(seed_rows or []), *rows], row)
    return rows


def _evaluate_frontier_specs(
    specs: list[FrontierSpec],
    features: dict[date, tuple[PremarketFeature, ...]],
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    opportunity_by_key: dict[tuple[date, str], OpportunityRow],
    opportunity_by_day: dict[date, list[OpportunityRow]],
    frontier_cache: dict[str, dict[date, tuple[str, ...]]],
    score_cache: dict[str, dict[date, dict[str, float]]],
    output_dir: Path,
    *,
    stage: str,
    completed_offset: int,
    total: int,
    max_workers: int,
    seed_rows: list[FrontierResult] | None = None,
) -> list[FrontierResult]:
    if not specs:
        return []
    rows: list[FrontierResult] = []
    # Frontier evaluation is a research proxy pass over compact in-memory data.
    # Keep it deterministic and in-process; Stage 3 remains the official
    # shared-core path where worker parallelism matters most.
    for spec in specs:
        row = evaluate_frontier_spec(spec, features, dataset, cfg, opportunity_by_key, opportunity_by_day, frontier_cache, score_cache)
        rows.append(row)
        _record_frontier_progress(output_dir, stage, completed_offset + len(rows), total, [*(seed_rows or []), *rows], row)
    return rows


def _dedupe_frontier_results(rows: list[FrontierResult]) -> list[FrontierResult]:
    out: list[FrontierResult] = []
    seen: set[str] = set()
    for row in rows:
        signature = _frontier_signature(row.spec)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(row)
    return out


def _assign_frontier_pareto_scores(rows: list[FrontierResult]) -> None:
    if not rows:
        return
    best_portfolio = max((row.metrics.get("portfolio_proxy_net_return_pct", 0.0) for row in rows if not row.rejected), default=0.0)
    best_mfe = max((row.metrics.get("avg_mfe_r", 0.0) for row in rows if not row.rejected), default=0.0)
    updated: list[FrontierResult] = []
    for row in rows:
        if row.rejected:
            score = 0.0
        else:
            portfolio_component = _clip(row.metrics.get("portfolio_proxy_net_return_pct", 0.0) / max(best_portfolio, 1e-9))
            mfe_component = _clip(row.metrics.get("avg_mfe_r", 0.0) / max(best_mfe, 1e-9))
            top5_component = _clip(row.metrics.get("top5_recall", 0.0))
            size_component = _clip((103.0 - row.metrics.get("frontier_avg_size", 103.0)) / 103.0)
            score = 100.0 * (0.46 * portfolio_component + 0.34 * mfe_component + 0.12 * top5_component + 0.08 * size_component)
        updated.append(replace(row, pareto_score=round(score, 6)))
    rows[:] = updated


def _assign_pareto_scores(rows: list[PairResult]) -> None:
    if not rows:
        return
    best_portfolio = max((row.metrics.get("portfolio_proxy_net_return_pct", 0.0) for row in rows if not row.rejected), default=0.0)
    best_mfe = max((row.metrics.get("avg_mfe_r", 0.0) for row in rows if not row.rejected), default=0.0)
    updated: list[PairResult] = []
    for row in rows:
        if row.rejected:
            score = 0.0
        else:
            portfolio_component = _clip(row.metrics.get("portfolio_proxy_net_return_pct", 0.0) / max(best_portfolio, 1e-9))
            mfe_component = _clip(row.metrics.get("avg_mfe_r", 0.0) / max(best_mfe, 1e-9))
            size_component = _clip((103.0 - row.metrics.get("frontier_avg_size", 103.0)) / 103.0)
            score = 100.0 * (0.48 * portfolio_component + 0.42 * mfe_component + 0.07 * _clip(row.metrics.get("full_first30_candidate_recall", 0.0)) + 0.03 * size_component)
        updated.append(replace(row, pareto_score=round(score, 6)))
    rows[:] = updated


def _opportunity_by_day(rows: Iterable[OpportunityRow]) -> dict[date, list[OpportunityRow]]:
    by_day: dict[date, list[OpportunityRow]] = {}
    for row in rows:
        by_day.setdefault(row.trade_date, []).append(row)
    return by_day


def _stratified_frontier_sample(specs: list[FrontierSpec], limit: int) -> list[FrontierSpec]:
    if limit <= 0 or limit >= len(specs):
        return list(specs)
    buckets: dict[tuple[str, int], list[FrontierSpec]] = {}
    for spec in specs:
        buckets.setdefault((spec.mode, spec.frontier_size), []).append(spec)
    per_bucket = max(1, int(math.ceil(limit / max(float(len(buckets)), 1.0))))
    sampled: list[FrontierSpec] = []
    for key in sorted(buckets):
        sampled.extend(_even_sample(buckets[key], min(len(buckets[key]), per_bucket)))
    return _even_sample(_dedupe_frontiers(sampled), limit)


def _dedupe_frontiers(specs: Iterable[FrontierSpec]) -> list[FrontierSpec]:
    out: list[FrontierSpec] = []
    seen: set[str] = set()
    for spec in specs:
        named = name_frontier(spec)
        signature = _frontier_signature(named)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(named)
    return out


def _dedupe_pair_results(rows: list[PairResult]) -> list[PairResult]:
    out: list[PairResult] = []
    seen: set[str] = set()
    for row in rows:
        signature = _pair_signature(row.spec)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(row)
    return out


def _unique_frontiers(specs: list[PairSpec]) -> list[FrontierSpec]:
    out: list[FrontierSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.frontier.name in seen:
            continue
        seen.add(spec.frontier.name)
        out.append(spec.frontier)
    return out


def _frontier_modes() -> tuple[str, ...]:
    return (
        "rs_trend",
        "liquidity_momentum",
        "compression_breakout",
        "flow_accumulation",
        "flow_inflection",
        "foreign_accumulation",
        "institutional_accumulation",
        "flow_synergy",
        "flow_dissynergy",
        "sector_flow_participation",
        "index_confirmed_leadership",
        "hybrid",
    )


def _frontier_signature(spec: FrontierSpec) -> str:
    data = asdict(spec)
    data.pop("name", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _pair_signature(spec: PairSpec) -> str:
    return stable_signature({"frontier": _frontier_signature(spec.frontier), "first30": _spec_signature(spec.first30)})


def _near(value: float, grid: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sorted(grid, key=lambda item: (abs(float(item) - float(value)), float(item)))[:4])


def _near_size(value: int) -> tuple[int, ...]:
    return tuple(sorted(FRONTIER_SIZES, key=lambda item: (abs(int(item) - int(value)), int(item)))[:4])


def _two_folds(dates: list[date]) -> list[tuple[date, date]]:
    if len(dates) < 2:
        return []
    pivot = len(dates) // 2
    return [(dates[0], dates[pivot - 1]), (dates[pivot], dates[-1])] if pivot > 0 else []


def load_frontier_finalists(path: str | Path, *, frontier_top_n: int = 5) -> list[FrontierSpec]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for section in ("top_portfolio_proxy", "top_pareto", "top_combined", "top_slot_return", "top_mfe"):
        rows.extend(_read_json_section_rows(source, section, max(10, int(frontier_top_n) * 4)))
        if len(rows) >= int(frontier_top_n) * 4:
            break
    finalists: list[FrontierSpec] = []
    seen: set[str] = set()
    for row in rows:
        payload = dict(row.get("frontier") or {})
        if not payload:
            continue
        spec = name_frontier(FrontierSpec(**{key: value for key, value in payload.items() if key in FrontierSpec.__dataclass_fields__}))
        key = _frontier_signature(spec)
        if key in seen:
            continue
        seen.add(key)
        finalists.append(spec)
        if len(finalists) >= max(1, int(frontier_top_n)):
            break
    if not finalists:
        raise ValueError(f"No frontier finalists found in {source}")
    return finalists


def _read_json_section_rows(path: Path, section: str, limit: int) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    marker = f'"{section}"'
    rows: list[dict[str, Any]] = []
    buffer = ""
    marker_seen = False
    array_seen = False
    with path.open("rb") as handle:
        for raw in iter(lambda: handle.read(1024 * 1024), b""):
            buffer += raw.decode("utf-8", errors="ignore")
            while len(rows) < max(1, int(limit)):
                if not marker_seen:
                    marker_index = buffer.find(marker)
                    if marker_index < 0:
                        if len(buffer) > 2_000_000:
                            buffer = buffer[-2_000_000:]
                        break
                    marker_seen = True
                    buffer = buffer[marker_index + len(marker) :]
                if not array_seen:
                    array_index = buffer.find("[")
                    if array_index < 0:
                        break
                    array_seen = True
                    buffer = buffer[array_index + 1 :]
                buffer = buffer.lstrip()
                while buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                if buffer.startswith("]"):
                    return rows
                try:
                    obj, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if len(buffer) > 8_000_000:
                        buffer = buffer[-8_000_000:]
                    break
                if isinstance(obj, dict):
                    rows.append(obj)
                buffer = buffer[end:]
            if len(rows) >= max(1, int(limit)):
                break
    return rows


def _first30_row_payload(row: Any) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "score": row.score,
        "full_score": row.full_score,
        "median_fold_score": row.median_fold_score,
        "worst_fold_score": row.worst_fold_score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "spec": asdict(row.spec),
        "metrics": row.metrics,
    }


def _row_payload(row: PairResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "return_score": row.return_score,
        "mfe_score": row.mfe_score,
        "combined_score": row.combined_score,
        "pareto_score": row.pareto_score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "frontier": asdict(row.spec.frontier),
        "first30": asdict(row.spec.first30),
        "metrics": row.metrics,
    }


def _frontier_row_payload(row: FrontierResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "return_score": row.return_score,
        "mfe_score": row.mfe_score,
        "combined_score": row.combined_score,
        "pareto_score": row.pareto_score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "frontier": asdict(row.spec),
        "metrics": row.metrics,
    }


def _write_first30_promotion(output_dir: Path, rows: list[Any], promoted: list[First30Spec]) -> None:
    payload = {
        "updated_at": _utc_now_iso(),
        "candidate_count": len(rows),
        "promoted": [asdict(spec) for spec in promoted],
        "top_first30": [_first30_row_payload(row) for row in rows[:30]],
    }
    path = output_dir / "first30_promotion.json"
    tmp = output_dir / "first30_promotion.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _record_progress(output_dir: Path, stage: str, completed: int, total: int, rows: list[PairResult], row: PairResult) -> None:
    if completed not in {1, 2, 3, 5, 10, total} and completed % 50 != 0:
        return
    _write_progress(output_dir, stage, completed, total, rows)
    event = {"updated_at": _utc_now_iso(), "stage": stage, "completed": int(completed), "total": int(total), "row": _progress_row(row)}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    print(
        "[kalcb-premarket-frontier] "
        f"{stage} {completed}/{total} {row.spec.name} combined={row.combined_score:.3f} "
        f"portfolio={100.0 * row.metrics.get('portfolio_proxy_net_return_pct', 0.0):.1f}% "
        f"slot={100.0 * row.metrics.get('slot_cumulative_gross_return_pct', 0.0):.1f}% "
        f"mfe={row.metrics.get('avg_mfe_r', 0.0):.3f} "
        f"frontier={row.metrics.get('frontier_avg_size', 0.0):.1f} reject={row.reject_reason}",
        flush=True,
    )


def _record_frontier_progress(output_dir: Path, stage: str, completed: int, total: int, rows: list[FrontierResult], row: FrontierResult) -> None:
    if completed not in {1, 2, 3, 5, 10, total} and completed % 50 != 0:
        return
    _assign_frontier_pareto_scores(rows)
    _write_frontier_progress(output_dir, stage, completed, total, rows)
    event = {"updated_at": _utc_now_iso(), "stage": stage, "completed": int(completed), "total": int(total), "row": _frontier_progress_row(row)}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    print(
        "[kalcb-premarket-frontier-only] "
        f"{stage} {completed}/{total} {row.spec.name} combined={row.combined_score:.3f} "
        f"portfolio={100.0 * row.metrics.get('portfolio_proxy_net_return_pct', 0.0):.1f}% "
        f"slot={100.0 * row.metrics.get('slot_cumulative_gross_return_pct', 0.0):.1f}% "
        f"mfe={row.metrics.get('avg_mfe_r', 0.0):.3f} "
        f"frontier={row.metrics.get('frontier_avg_size', 0.0):.1f} reject={row.reject_reason}",
        flush=True,
    )


def _record_setup_progress(output_dir: Path, stage: str, **extra: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": _utc_now_iso(), "stage": stage, **extra}
    path = output_dir / "setup_progress.json"
    tmp = output_dir / "setup_progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    with (output_dir / "setup_progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _first30_progress_row(row: Any) -> dict[str, Any]:
    metrics = getattr(row, "metrics", {}) or {}
    spec = getattr(row, "spec", None)
    return {
        "name": str(getattr(spec, "name", "")),
        "score": _float(getattr(row, "score", 0.0)),
        "rejected": bool(getattr(row, "rejected", False)),
        "reject_reason": str(getattr(row, "reject_reason", "")),
        "portfolio_proxy_net_return_pct": _float(metrics.get("portfolio_proxy_net_return_pct")),
        "avg_mfe_r": _float(metrics.get("avg_mfe_r")),
        "active_day_share": _float(metrics.get("active_day_share")),
    }


def _best_first30_progress_row(rows: list[Any]) -> dict[str, Any]:
    accepted = [row for row in rows if not bool(getattr(row, "rejected", False))]
    source = accepted or rows
    if not source:
        return {}
    best = sorted(source, key=lambda row: (-_float(getattr(row, "score", 0.0)), str(getattr(getattr(row, "spec", None), "name", ""))))[0]
    return _first30_progress_row(best)


def _write_progress(output_dir: Path, stage: str, completed: int, total: int, rows: list[PairResult]) -> None:
    ranked_portfolio = sorted(
        (_progress_row(row) for row in rows),
        key=lambda item: (bool(item["rejected"]), -float(item["metrics"].get("portfolio_proxy_net_return_pct", 0.0)), float(item["metrics"].get("frontier_avg_size", 999.0))),
    )
    ranked_slot = sorted(
        (_progress_row(row) for row in rows),
        key=lambda item: (bool(item["rejected"]), -float(item["metrics"].get("slot_cumulative_gross_return_pct", 0.0)), float(item["metrics"].get("frontier_avg_size", 999.0))),
    )
    ranked_mfe = sorted(
        (_progress_row(row) for row in rows),
        key=lambda item: (bool(item["rejected"]), -float(item["metrics"].get("avg_mfe_r", 0.0)), float(item["metrics"].get("frontier_avg_size", 999.0))),
    )
    payload = {
        "updated_at": _utc_now_iso(),
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "percent": round(100.0 * completed / total, 3) if total else 100.0,
        "best_portfolio_so_far": ranked_portfolio[0] if ranked_portfolio else None,
        "best_slot_so_far": ranked_slot[0] if ranked_slot else None,
        "best_mfe_so_far": ranked_mfe[0] if ranked_mfe else None,
        "top_portfolio": ranked_portfolio[:15],
        "top_slot": ranked_slot[:15],
        "top_mfe": ranked_mfe[:15],
    }
    path = output_dir / "progress.json"
    tmp = output_dir / "progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _write_frontier_progress(output_dir: Path, stage: str, completed: int, total: int, rows: list[FrontierResult]) -> None:
    ranked_portfolio = sorted(
        (_frontier_progress_row(row) for row in rows),
        key=lambda item: (bool(item["rejected"]), -float(item["metrics"].get("portfolio_proxy_net_return_pct", 0.0)), float(item["metrics"].get("frontier_avg_size", 999.0))),
    )
    ranked_slot = sorted(
        (_frontier_progress_row(row) for row in rows),
        key=lambda item: (bool(item["rejected"]), -float(item["metrics"].get("slot_cumulative_gross_return_pct", 0.0)), float(item["metrics"].get("frontier_avg_size", 999.0))),
    )
    ranked_mfe = sorted(
        (_frontier_progress_row(row) for row in rows),
        key=lambda item: (bool(item["rejected"]), -float(item["metrics"].get("avg_mfe_r", 0.0)), float(item["metrics"].get("frontier_avg_size", 999.0))),
    )
    payload = {
        "updated_at": _utc_now_iso(),
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "percent": round(100.0 * completed / total, 3) if total else 100.0,
        "best_portfolio_so_far": ranked_portfolio[0] if ranked_portfolio else None,
        "best_slot_so_far": ranked_slot[0] if ranked_slot else None,
        "best_mfe_so_far": ranked_mfe[0] if ranked_mfe else None,
        "top_portfolio": ranked_portfolio[:15],
        "top_slot": ranked_slot[:15],
        "top_mfe": ranked_mfe[:15],
    }
    path = output_dir / "progress.json"
    tmp = output_dir / "progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _progress_row(row: PairResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "return_score": row.return_score,
        "mfe_score": row.mfe_score,
        "combined_score": row.combined_score,
        "pareto_score": row.pareto_score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "frontier": asdict(row.spec.frontier),
        "first30": asdict(row.spec.first30),
        "metrics": {
            "active_days": row.metrics.get("active_days", 0.0),
            "avg_candidates_per_session": row.metrics.get("avg_candidates_per_session", 0.0),
            "portfolio_proxy_net_return_pct": row.metrics.get("portfolio_proxy_net_return_pct", 0.0),
            "portfolio_proxy_active_day_net_pct": row.metrics.get("portfolio_proxy_active_day_net_pct", 0.0),
            "portfolio_proxy_max_drawdown_pct": row.metrics.get("portfolio_proxy_max_drawdown_pct", 0.0),
            "slot_cumulative_gross_return_pct": row.metrics.get("slot_cumulative_gross_return_pct", 0.0),
            "slot_cumulative_net_return_pct": row.metrics.get("slot_cumulative_net_return_pct", 0.0),
            "avg_mfe_r": row.metrics.get("avg_mfe_r", 0.0),
            "selected_mfe_percentile": row.metrics.get("selected_mfe_percentile", 0.0),
            "top5_day_hit_share": row.metrics.get("top5_day_hit_share", 0.0),
            "full_first30_candidate_recall": row.metrics.get("full_first30_candidate_recall", 0.0),
            "opportunity_loss_vs_full103_gross_slot_pct": row.metrics.get("opportunity_loss_vs_full103_gross_slot_pct", 0.0),
            "frontier_avg_size": row.metrics.get("frontier_avg_size", 0.0),
            "ws_hot_feasible": row.metrics.get("ws_hot_feasible", 0.0),
        },
    }


def _frontier_progress_row(row: FrontierResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "return_score": row.return_score,
        "mfe_score": row.mfe_score,
        "combined_score": row.combined_score,
        "pareto_score": row.pareto_score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "frontier": asdict(row.spec),
        "metrics": {
            "active_days": row.metrics.get("active_days", 0.0),
            "avg_candidates_per_session": row.metrics.get("avg_candidates_per_session", 0.0),
            "portfolio_proxy_net_return_pct": row.metrics.get("portfolio_proxy_net_return_pct", 0.0),
            "portfolio_proxy_active_day_net_pct": row.metrics.get("portfolio_proxy_active_day_net_pct", 0.0),
            "portfolio_proxy_max_drawdown_pct": row.metrics.get("portfolio_proxy_max_drawdown_pct", 0.0),
            "slot_cumulative_gross_return_pct": row.metrics.get("slot_cumulative_gross_return_pct", 0.0),
            "slot_cumulative_net_return_pct": row.metrics.get("slot_cumulative_net_return_pct", 0.0),
            "avg_mfe_r": row.metrics.get("avg_mfe_r", 0.0),
            "selected_mfe_percentile": row.metrics.get("selected_mfe_percentile", 0.0),
            "top5_day_hit_share": row.metrics.get("top5_day_hit_share", 0.0),
            "top5_recall": row.metrics.get("top5_recall", 0.0),
            "frontier_avg_size": row.metrics.get("frontier_avg_size", 0.0),
            "ws_hot_feasible": row.metrics.get("ws_hot_feasible", 0.0),
        },
    }


def _render_frontier_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB Premarket Frontier-Only Sweep",
        "",
        f"Sweep hash: `{payload['sweep_hash']}`",
        f"Window: {payload['training_window']['start']} to {payload['training_window']['end']} ({payload['training_window']['sessions']} sessions)",
        f"Holdout days excluded: {payload['holdout_days']}",
        "",
        "This is Stage 1 research only. It ranks premarket frontier candidates and does not optimize first30 selectors or official trade plans.",
        "",
        "## Top Portfolio Proxy",
        "",
        _frontier_table(payload.get("top_portfolio_proxy", [])),
        "",
        "## Top Slot Return",
        "",
        _frontier_table(payload["top_slot_return"]),
        "",
        "## Top Avg MFE",
        "",
        _frontier_table(payload["top_mfe"]),
        "",
        "## Pareto",
        "",
        _frontier_table(payload["top_pareto"]),
    ]
    return "\n".join(lines) + "\n"


def _frontier_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Frontier | Combined | Portfolio Net | DD | Slot | Net Slot | Avg MFE | MFE Pctl | Top5 Hit | Top5 Recall | Frontier N | WS Feasible |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows[:25], start=1):
        metrics = row["metrics"]
        lines.append(
            "| "
            f"{index} | {row['frontier']['name']} | {row['combined_score']:.3f} | "
            f"{100.0 * metrics.get('portfolio_proxy_net_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('portfolio_proxy_max_drawdown_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('slot_cumulative_gross_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('slot_cumulative_net_return_pct', 0.0):.1f}% | "
            f"{metrics.get('avg_mfe_r', 0.0):.3f} | "
            f"{metrics.get('selected_mfe_percentile', 0.0):.3f} | "
            f"{metrics.get('top5_day_hit_share', 0.0):.3f} | "
            f"{metrics.get('top5_recall', 0.0):.3f} | "
            f"{metrics.get('frontier_avg_size', 0.0):.1f} | "
            f"{metrics.get('ws_hot_feasible', 0.0):.0f} |"
        )
    return "\n".join(lines)


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB Premarket Frontier + Causal First30 Sweep",
        "",
        f"Sweep hash: `{payload['sweep_hash']}`",
        f"Window: {payload['training_window']['start']} to {payload['training_window']['end']} ({payload['training_window']['sessions']} sessions)",
        f"Holdout days excluded: {payload['holdout_days']}",
        "",
        "## Top Slot Return",
        "",
        _table(payload["top_slot_return"]),
        "",
        "## Top Portfolio Proxy",
        "",
        _table(payload.get("top_portfolio_proxy", [])),
        "",
        "## Top Avg MFE",
        "",
        _table(payload["top_mfe"]),
        "",
        "## Pareto",
        "",
        _table(payload["top_pareto"]),
        "",
        "## Full-103 First30 References",
        "",
        "| Selector | Active Days | Names/Session | Gross Slot | Net Slot | Avg MFE | MFE Pctl | Top5 Hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["references"]:
        lines.append(
            "| "
            f"{item['name']} | {item.get('active_days', 0):.0f} | {item.get('avg_candidates_per_session', 0.0):.3f} | "
            f"{100.0 * item.get('slot_cumulative_gross_return_pct', 0.0):.1f}% | "
            f"{100.0 * item.get('slot_cumulative_net_return_pct', 0.0):.1f}% | "
            f"{item.get('avg_mfe_r', 0.0):.3f} | "
            f"{item.get('selected_mfe_percentile', 0.0):.3f} | {item.get('top5_day_hit_share', 0.0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Frontier | First30 | Combined | Portfolio Net | DD | Slot | Net Slot | Avg MFE | MFE Pctl | Top5 Hit | Recall | Frontier N |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows[:25], start=1):
        metrics = row["metrics"]
        lines.append(
            "| "
            f"{index} | {row['frontier']['name']} | {row['first30']['name']} | {row['combined_score']:.3f} | "
            f"{100.0 * metrics.get('portfolio_proxy_net_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('portfolio_proxy_max_drawdown_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('slot_cumulative_gross_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('slot_cumulative_net_return_pct', 0.0):.1f}% | "
            f"{metrics.get('avg_mfe_r', 0.0):.3f} | "
            f"{metrics.get('selected_mfe_percentile', 0.0):.3f} | "
            f"{metrics.get('top5_day_hit_share', 0.0):.3f} | "
            f"{metrics.get('full_first30_candidate_recall', 0.0):.3f} | "
            f"{metrics.get('frontier_avg_size', 0.0):.1f} |"
        )
    return "\n".join(lines)


def _adv_label(value: float) -> str:
    if value <= 0:
        return "0"
    return f"{value / 1_000_000_000.0:.0f}b"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep KALCB premarket frontiers feeding causal first30 selectors.")
    parser.add_argument("--config", default="config/optimization/kalcb.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--first30-top-n", type=int, default=8)
    parser.add_argument("--refine-first30-top-n", type=int, default=6)
    parser.add_argument("--max-first30-coarse-specs", type=int, default=None)
    parser.add_argument("--first30-artifact", default=None)
    parser.add_argument("--coarse-frontier-limit", type=int, default=640)
    parser.add_argument("--deep-pair-count", type=int, default=16)
    parser.add_argument("--deep-per-mode-limit", type=int, default=120)
    parser.add_argument("--max-frontier-specs", type=int, default=None)
    parser.add_argument("--frontier-artifact", default=None)
    parser.add_argument("--frontier-top-n", type=int, default=0)
    args = parser.parse_args(argv)
    config = normalize_runtime_config("kalcb", load_yaml_config(args.config))
    payload = run_premarket_frontier_sweep(
        config,
        output_dir=args.output_dir,
        holdout_days=args.holdout_days,
        max_workers=args.max_workers,
        first30_top_n=args.first30_top_n,
        refine_first30_top_n=args.refine_first30_top_n,
        max_first30_coarse_specs=args.max_first30_coarse_specs,
        first30_artifact=args.first30_artifact,
        coarse_frontier_limit=args.coarse_frontier_limit,
        deep_pair_count=args.deep_pair_count,
        deep_per_mode_limit=args.deep_per_mode_limit,
        max_frontier_specs=args.max_frontier_specs,
        frontier_artifact=args.frontier_artifact,
        frontier_top_n=args.frontier_top_n,
    )
    print(
        json.dumps(
            {
                "sweep_hash": payload["sweep_hash"],
                "artifact_paths": payload["artifact_paths"],
                "top_portfolio_proxy": payload["top_portfolio_proxy"][:5],
                "top_slot_return": payload["top_slot_return"][:5],
                "top_mfe": payload["top_mfe"][:5],
                "top_pareto": payload["top_pareto"][:5],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
