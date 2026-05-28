from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from backtests.analysis.metrics import compute_trade_metrics
from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.engine.replay import run_replay
from backtests.engine.sim_broker import BrokerCosts
from strategy_kalcb.config import KALCBConfig

from .fixed_trade_plan_phase import _broker_trade_rows, _candidate_snapshot_metadata, _configured_entry_routes, _route_candidate_passes
from .first30_signal_sweep import Selection, build_contexts, prepare_first30_dataset
from .runner import KALCBReplayAdapter, _collapse_exit_legs
from .shadow_ledger_reranker import read_jsonl, write_jsonl
from .trade_plan_sweep import (
    _add_compiled_candidate_pool_metrics,
    _add_portfolio_equivalent_metrics,
    _add_return_divergence_metrics,
    _broker_trades_to_slot_outcomes,
    _candidate_from_context,
    _clone_snapshots_for_replay,
    _fold_metrics_from_outcomes_for_dates,
    _replay_digest,
    _resolve_folds,
    _selection_counts,
    compile_core_replay,
    summarize_outcomes,
)


CANDIDATE_SURFACING_RECOVERY_VERSION = "kalcb-causal-candidate-surfacing-recovery-v1"
CANDIDATE_SURFACING_USAGE_CONTRACT = "research_only_oracle_labels_train_only_not_live_entry_feature"

ACTIVE_COUNT = 8
LEADING_SECTOR_QUOTA = 12
LEADING_SECTOR_BASE_COUNT = 28
LEADING_SECTOR_MAX_PER_SECTOR = 4
RECALL_KS = (8, 16, 32, 64)

CAUSAL_FEATURE_KEYS = (
    "first30_ret",
    "first30_vwap_ret",
    "first30_gap",
    "first30_rel_volume",
    "first30_signal_bar_cpr",
    "first30_open_drawdown",
    "first30_low_vs_prev_close",
    "first30_range_atr",
    "first30_gap_retention_ratio",
    "first30_gap_relvol",
    "first30_low_vs_prev_relvol",
    "first30_quality_pct",
    "daily_return_5d",
    "daily_return_20d",
    "daily_return_60d",
    "daily_volume_ratio_20d",
    "daily_close20_loc",
    "daily_close60_loc",
    "daily_acceleration_5v20",
    "daily_momentum_pct",
    "daily_adv20_krw_log",
    "stock_sector_daily_ret5_spread",
    "stock_sector_daily_ret20_spread",
    "daily_sector_alignment_pct",
    "structural_campaign_score",
    "campaign_state_score",
    "campaign_box_range_pct",
    "campaign_box_containment",
    "campaign_breakout_displacement",
    "sector_daily_score_pct",
    "sector_daily_participation",
    "sector_daily_breadth_20d",
    "sector_intraday_score_pct",
    "sector_intraday_ret",
    "sector_intraday_breadth",
    "sector_intraday_participation",
    "sector_intraday_rel_volume",
    "first30_sector_ret_spread",
    "first30_sector_relvol_ratio",
    "first30_sector_leadership_pct",
    "first30_gap_relvol_sector_breadth",
    "first30_gap_retention_sector_breadth",
    "continuation_joint_quality_pct",
    "leading_sector_cluster",
)

LABEL_KEYS = (
    "label_best_route_shadow_r",
    "label_top_decile_oracle",
    "label_routeable_positive_mfe_controlled_mae",
    "label_same_day_replacement_value_r",
    "label_composite_oracle_recall",
)

LEAKAGE_FEATURE_BLOCKLIST = {
    "net_r",
    "gross_r",
    "mfe_r",
    "mae_r",
    "mfe_capture",
    "oracle_score",
    "same_day_actual_total_r",
    "same_day_weakest_actual_r",
    "same_day_replacement_value_r",
    "best_route_shadow_r",
    *LABEL_KEYS,
}


@dataclass(frozen=True, slots=True)
class PoolVariant:
    name: str
    pool_size: int
    active_count: int = ACTIVE_COUNT
    kind: str = "topn"


POOL_VARIANTS = (
    PoolVariant("top24_causal_oracle_ranker", 24),
    PoolVariant("top40_causal_oracle_ranker", 40),
    PoolVariant("top64_causal_oracle_ranker", 64),
    PoolVariant("top40_with_leading_sector_quota", 40, kind="leading_sector_quota"),
    PoolVariant("blend_existing_frontier_50pct_plus_causal_ranker_50pct", 40, kind="blend_existing_50pct"),
)

DELAYED_ROUTE_FAMILY_MODES = ("pullback_acceptance", "avwap_reclaim", "or_high_reclaim")
STAGE08_ROUTE_BUNDLE_NAME = "delayed_pullback_avwap_or_high_rank8_split_r0p015"
MATCHED_INCUMBENT_ROUTE_BUNDLE_NAME = "matched_incumbent_seed_execution_first30_anchor"


def build_conservative_route_family_mutations(seed_mutations: Mapping[str, Any]) -> dict[str, Any]:
    """Build the stage-08 replay bundle: delayed route families only, no first30 expansion."""
    routes = _configured_entry_routes(seed_mutations)
    delayed_seed = next(
        (
            dict(route)
            for route in routes
            if str(route.get("mode") or "").strip().lower() != "first30_open"
        ),
        {},
    )
    if not delayed_seed:
        delayed_seed = {"max_frontier_rank": 8, "risk_mult": 0.015, "notional_mult": 0.015}

    max_rank = int(_num(delayed_seed.get("max_frontier_rank"), 8.0) or 8)
    seed_risk = _num(delayed_seed.get("risk_mult", delayed_seed.get("notional_mult")), 0.015) or 0.015
    per_route_risk = min(max(seed_risk, 0.0), 0.015) / float(len(DELAYED_ROUTE_FAMILY_MODES))
    context_min = dict(delayed_seed.get("context_min") or {})
    context_max = dict(delayed_seed.get("context_max") or {})

    bundle_routes: list[dict[str, Any]] = []
    for priority, mode in enumerate(DELAYED_ROUTE_FAMILY_MODES):
        route = dict(delayed_seed)
        route.update(
            {
                "name": f"stage08_{mode}_rank{max_rank}_r{per_route_risk:.4f}",
                "mode": mode,
                "priority": priority,
                "require_initial_active": False,
                "max_frontier_rank": max_rank,
                "max_session_trades": 1,
                "risk_mult": per_route_risk,
                "notional_mult": per_route_risk,
            }
        )
        route["context_min"] = dict(context_min)
        route["context_max"] = dict(context_max)
        route.setdefault("after_bar", 1)
        route.setdefault("max_signal_bars", 18)
        route.setdefault("min_reclaim_ret", 0.0005)
        route.setdefault("min_vwap_ret", 0.0)
        route.setdefault("max_pullback_from_vwap_pct", 0.008)
        bundle_routes.append(route)

    mutations = dict(seed_mutations)
    mutations["kalcb.entry.routes"] = bundle_routes
    mutations["kalcb.entry.frontier_branch_universe"] = True
    mutations["kalcb.frontier.shadow_enabled"] = False
    mutations["_kalcb.stage08.route_bundle"] = STAGE08_ROUTE_BUNDLE_NAME
    return mutations


def build_matched_incumbent_route_mutations(seed_mutations: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve incumbent execution so candidate-surfacing can be sanity-checked apples-to-apples."""
    mutations = dict(seed_mutations)
    mutations["_kalcb.stage08.route_bundle"] = MATCHED_INCUMBENT_ROUTE_BUNDLE_NAME
    return mutations


def describe_stage08_route_bundle(mutations: Mapping[str, Any]) -> dict[str, Any]:
    routes = _configured_entry_routes(mutations)
    return {
        "name": str(mutations.get("_kalcb.stage08.route_bundle") or STAGE08_ROUTE_BUNDLE_NAME),
        "modes": [str(route.get("mode") or "") for route in routes],
        "first30_open_enabled": any(str(route.get("mode") or "") == "first30_open" for route in routes),
        "max_frontier_rank": max((int(_num(route.get("max_frontier_rank"), 0.0) or 0) for route in routes), default=0),
        "max_session_trades_by_route": {
            str(route.get("mode") or ""): int(_num(route.get("max_session_trades"), 0.0) or 0)
            for route in routes
        },
        "risk_mult_by_route": {
            str(route.get("mode") or ""): round(_num(route.get("risk_mult"), 0.0) or 0.0, 6)
            for route in routes
        },
    }


def build_candidate_surfacing_recovery_artifacts(
    *,
    train_config: dict[str, Any],
    holdout_config: dict[str, Any],
    train_existing_context: Any,
    holdout_existing_context: Any,
    train_oracle_path: str | Path,
    holdout_oracle_path: str | Path,
    seed_mutations: dict[str, Any],
    baseline_metrics: dict[str, dict[str, Any]],
    output_dir: str | Path,
    max_replay_variants: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_oracle = read_jsonl(train_oracle_path)
    holdout_oracle = read_jsonl(holdout_oracle_path)
    delayed_route_mutations = build_conservative_route_family_mutations(seed_mutations)
    matched_route_mutations = build_matched_incumbent_route_mutations(seed_mutations)
    route_bundle = describe_stage08_route_bundle(delayed_route_mutations)
    matched_route_bundle = describe_stage08_route_bundle(matched_route_mutations)

    train_bundle = _build_window_bundle(
        "train",
        train_config,
        train_oracle,
        train_existing_context,
        delayed_route_mutations,
    )
    holdout_bundle = _build_window_bundle(
        "holdout",
        holdout_config,
        holdout_oracle,
        holdout_existing_context,
        delayed_route_mutations,
    )

    profile = fit_causal_ranker_profile(train_bundle["features"])
    train_scored = score_causal_feature_rows(train_bundle["features"], profile)
    holdout_scored = score_causal_feature_rows(holdout_bundle["features"], profile)

    variants = list(POOL_VARIANTS)
    replay_variants = variants[: max_replay_variants if max_replay_variants is not None else len(variants)]
    train_pools = build_candidate_pools(train_scored, train_bundle["existing_pool_by_day"], variants)
    holdout_pools = build_candidate_pools(holdout_scored, holdout_bundle["existing_pool_by_day"], variants)

    routes = _configured_entry_routes(delayed_route_mutations)
    train_recall = summarize_pool_recall(train_scored, train_pools, routes, delayed_route_mutations)
    holdout_recall = summarize_pool_recall(holdout_scored, holdout_pools, routes, delayed_route_mutations)
    train_replay = _evaluate_replay_variants(
        "train",
        train_config,
        train_bundle,
        train_pools,
        delayed_route_mutations,
        baseline_metrics.get("train") or {},
        out,
        replay_variants,
        replay_name="delayed_route_family",
    )
    holdout_replay = _evaluate_replay_variants(
        "holdout",
        holdout_config,
        holdout_bundle,
        holdout_pools,
        delayed_route_mutations,
        baseline_metrics.get("holdout") or {},
        out,
        replay_variants,
        replay_name="delayed_route_family",
    )
    train_matched_replay = _evaluate_replay_variants(
        "train",
        train_config,
        train_bundle,
        train_pools,
        matched_route_mutations,
        baseline_metrics.get("train") or {},
        out,
        replay_variants,
        replay_name="matched_incumbent_execution",
    )
    holdout_matched_replay = _evaluate_replay_variants(
        "holdout",
        holdout_config,
        holdout_bundle,
        holdout_pools,
        matched_route_mutations,
        baseline_metrics.get("holdout") or {},
        out,
        replay_variants,
        replay_name="matched_incumbent_execution",
    )
    train_matched_active_replay = _evaluate_replay_variants(
        "train",
        train_config,
        train_bundle,
        train_pools,
        matched_route_mutations,
        baseline_metrics.get("train") or {},
        out,
        replay_variants,
        replay_name="matched_active_budget_incumbent_execution",
        active_limit_by_day=train_bundle["existing_active_count_by_day"],
    )
    holdout_matched_active_replay = _evaluate_replay_variants(
        "holdout",
        holdout_config,
        holdout_bundle,
        holdout_pools,
        matched_route_mutations,
        baseline_metrics.get("holdout") or {},
        out,
        replay_variants,
        replay_name="matched_active_budget_incumbent_execution",
        active_limit_by_day=holdout_bundle["existing_active_count_by_day"],
    )

    train_feature_path = out / "candidate_surfacing_train_features.jsonl"
    holdout_feature_path = out / "candidate_surfacing_holdout_features.jsonl"
    train_rank_path = out / "candidate_pool_rankings_train.jsonl"
    holdout_rank_path = out / "candidate_pool_rankings_holdout.jsonl"
    profile_path = out / "causal_candidate_ranker_profile.json"
    recall_path = out / "candidate_pool_recall_summary.json"
    replay_path = out / "candidate_pool_replay_summary.json"
    manifest_path = out / "candidate_pool_variant_manifest.json"
    summary_path = out / "candidate_surfacing_recovery_summary.json"
    report_path = out / "candidate_surfacing_recovery_report.md"

    write_jsonl(train_feature_path, train_scored)
    write_jsonl(holdout_feature_path, holdout_scored)
    write_jsonl(train_rank_path, _flatten_pool_rows(train_pools))
    write_jsonl(holdout_rank_path, _flatten_pool_rows(holdout_pools))
    _write_json(profile_path, profile)
    recall_summary = {"train": train_recall, "holdout": holdout_recall}
    replay_summary = {
        "train": train_replay,
        "holdout": holdout_replay,
        "delayed_route_family": {"train": train_replay, "holdout": holdout_replay},
        "matched_incumbent_execution": {"train": train_matched_replay, "holdout": holdout_matched_replay},
        "matched_active_budget_incumbent_execution": {"train": train_matched_active_replay, "holdout": holdout_matched_active_replay},
    }
    _write_json(recall_path, recall_summary)
    _write_json(replay_path, replay_summary)
    manifest = {
        "version": CANDIDATE_SURFACING_RECOVERY_VERSION,
        "usage_contract": CANDIDATE_SURFACING_USAGE_CONTRACT,
        "variants": [asdict(item) for item in variants],
        "active_count": ACTIVE_COUNT,
        "route_bundle": route_bundle,
        "matched_incumbent_route_bundle": matched_route_bundle,
        "route_config_hash": stable_signature(delayed_route_mutations.get("kalcb.entry.routes") or []),
        "matched_incumbent_route_config_hash": stable_signature(matched_route_mutations.get("kalcb.entry.routes") or []),
        "seed_mutation_hash": stable_signature(seed_mutations),
        "route_mutation_hash": stable_signature(delayed_route_mutations),
        "matched_incumbent_mutation_hash": stable_signature(matched_route_mutations),
        "train_pool_hash": stable_signature(_pool_hash_payload(train_pools)),
        "holdout_pool_hash": stable_signature(_pool_hash_payload(holdout_pools)),
    }
    _write_json(manifest_path, manifest)

    summary = {
        "created_at": _utc_now_iso(),
        "version": CANDIDATE_SURFACING_RECOVERY_VERSION,
        "usage_contract": CANDIDATE_SURFACING_USAGE_CONTRACT,
        "feature_keys": list(CAUSAL_FEATURE_KEYS),
        "label_keys": list(LABEL_KEYS),
        "leakage_feature_blocklist": sorted(LEAKAGE_FEATURE_BLOCKLIST),
        "profile": profile,
        "route_bundle": route_bundle,
        "matched_incumbent_route_bundle": matched_route_bundle,
        "variant_manifest": manifest,
        "train": {
            "feature_row_count": len(train_scored),
            "oracle_labeled_row_count": sum(1 for row in train_scored if row.get("oracle_label_available")),
            "feature_coverage": feature_coverage(train_scored, CAUSAL_FEATURE_KEYS),
            "recall": train_recall,
            "replay": train_replay,
            "matched_incumbent_replay": train_matched_replay,
            "matched_active_budget_replay": train_matched_active_replay,
        },
        "holdout": {
            "feature_row_count": len(holdout_scored),
            "oracle_labeled_row_count": sum(1 for row in holdout_scored if row.get("oracle_label_available")),
            "feature_coverage": feature_coverage(holdout_scored, CAUSAL_FEATURE_KEYS),
            "recall": holdout_recall,
            "replay": holdout_replay,
            "matched_incumbent_replay": holdout_matched_replay,
            "matched_active_budget_replay": holdout_matched_active_replay,
        },
        "artifact_paths": {
            "train_features_jsonl": str(train_feature_path),
            "holdout_features_jsonl": str(holdout_feature_path),
            "train_rankings_jsonl": str(train_rank_path),
            "holdout_rankings_jsonl": str(holdout_rank_path),
            "ranker_profile_json": str(profile_path),
            "recall_summary_json": str(recall_path),
            "replay_summary_json": str(replay_path),
            "variant_manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
            "report_md": str(report_path),
        },
        "root_cause_layer_attribution": _root_cause_attribution(
            train_recall,
            holdout_recall,
            train_replay,
            holdout_replay,
            train_matched_replay,
            holdout_matched_replay,
            train_matched_active_replay,
            holdout_matched_active_replay,
            baseline_metrics,
        ),
    }
    _write_json(summary_path, summary)
    report_path.write_text(render_candidate_surfacing_report(summary), encoding="utf-8")
    return summary


def _build_window_bundle(
    window: str,
    config: dict[str, Any],
    oracle_rows: list[dict[str, Any]],
    existing_context: Any,
    seed_mutations: dict[str, Any],
) -> dict[str, Any]:
    dataset = prepare_first30_dataset(dict(config))
    contexts = build_contexts(dataset)
    context_by_key = {(day, ctx.symbol): ctx for day, items in contexts.items() for ctx in items}
    cfg = KALCBConfig.from_mapping(config, seed_mutations)
    existing_meta_by_key, existing_pool_by_day, existing_active_count_by_day = _existing_candidate_context(existing_context)
    features = build_causal_feature_rows(
        window=window,
        dataset=dataset,
        contexts=contexts,
        cfg=cfg,
        oracle_rows=oracle_rows,
        existing_meta_by_key=existing_meta_by_key,
    )
    return {
        "dataset": dataset,
        "contexts": contexts,
        "context_by_key": context_by_key,
        "cfg": cfg,
        "features": features,
        "existing_meta_by_key": existing_meta_by_key,
        "existing_pool_by_day": existing_pool_by_day,
        "existing_active_count_by_day": existing_active_count_by_day,
    }


def build_causal_feature_rows(
    *,
    window: str,
    dataset: Any,
    contexts: dict[date, tuple[Any, ...]],
    cfg: KALCBConfig,
    oracle_rows: Iterable[dict[str, Any]],
    existing_meta_by_key: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    existing_meta_by_key = dict(existing_meta_by_key or {})
    oracle_by_key = _best_oracle_by_key(oracle_rows)
    top_decile = _top_decile_oracle_keys(oracle_by_key.values())
    rows: list[dict[str, Any]] = []
    for day in dataset.trading_dates:
        for ctx in contexts.get(day, ()):
            day_label = day.isoformat()
            symbol = str(ctx.symbol)
            existing_meta = dict(existing_meta_by_key.get((day_label, symbol)) or {})
            existing_rank = int(_num(existing_meta.get("candidate_rank"), 999))
            selection_score = _num(existing_meta.get("frontier_selection_score"))
            selection = Selection(day, symbol, selection_score, "causal_candidate_surfacing_feature")
            candidate = _candidate_from_context(
                ctx,
                selection,
                dataset,
                cfg,
                existing_rank,
                frontier_rank=int(_num(existing_meta.get("frontier_rank"), 999)),
                frontier_score=selection_score,
                frontier_initial_active=bool(existing_meta.get("frontier_initial_active", False)),
                frontier_role=str(existing_meta.get("frontier_role") or "out_of_pool"),
                source_calibration_metadata={"candidate_surfacing_recovery": CANDIDATE_SURFACING_RECOVERY_VERSION},
            )
            meta = dict(candidate.metadata)
            oracle = dict(oracle_by_key.get((day_label, symbol)) or {})
            row: dict[str, Any] = {
                "window": window,
                "trade_date": day_label,
                "symbol": symbol,
                "sector": str(candidate.sector or "UNKNOWN"),
                "oracle_label_available": bool(oracle),
                "current_in_candidate_pool": bool(existing_meta or oracle.get("in_candidate_pool")),
                "current_frontier_rank": _optional_num(existing_meta.get("frontier_rank")),
                "current_frontier_role": str(existing_meta.get("frontier_role") or "out_of_pool"),
                "current_frontier_selection_score": _optional_num(existing_meta.get("frontier_selection_score")),
                "daily_adv20_krw": float(getattr(ctx.daily, "adv20_krw", 0.0) or 0.0),
                "daily_adv20_krw_log": math.log1p(max(float(getattr(ctx.daily, "adv20_krw", 0.0) or 0.0), 0.0)),
                "oracle_route_family": str(oracle.get("route_family") or ""),
                "oracle_score": _optional_num(oracle.get("oracle_score")),
                "net_r": _optional_num(oracle.get("net_r")),
                "mfe_r": _optional_num(oracle.get("mfe_r")),
                "mae_r": _optional_num(oracle.get("mae_r")),
                "mfe_capture": _optional_num(oracle.get("mfe_capture")),
                "same_day_actual_total_r": _optional_num(oracle.get("same_day_actual_total_r")),
                "same_day_weakest_actual_r": _optional_num(oracle.get("same_day_weakest_actual_r")),
            }
            for key in CAUSAL_FEATURE_KEYS:
                if key == "leading_sector_cluster":
                    row[key] = bool(_num(meta.get("sector_daily_score_pct")) >= 80.0 and _num(meta.get("sector_intraday_score_pct")) >= 70.0)
                elif key == "first30_signal_bar_cpr":
                    row[key] = meta.get("first30_signal_bar_cpr", meta.get("first30_close_location", meta.get("first30_range_close_location")))
                elif key == "daily_adv20_krw_log":
                    row[key] = math.log1p(max(float(getattr(ctx.daily, "adv20_krw", 0.0) or 0.0), 0.0))
                else:
                    row[key] = meta.get(key, existing_meta.get(key))
            _attach_labels(row, top_decile)
            rows.append(row)
    _attach_day_sector_context(rows)
    rows.sort(key=lambda item: (str(item.get("trade_date") or ""), str(item.get("symbol") or "")))
    return rows


def fit_causal_ranker_profile(train_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in train_rows]
    _assert_no_label_leakage(CAUSAL_FEATURE_KEYS)
    target = [_num(row.get("label_composite_oracle_recall")) for row in rows]
    feature_stats: dict[str, dict[str, Any]] = {}
    raw_weights: dict[str, float] = {}
    for key in CAUSAL_FEATURE_KEYS:
        values = [_optional_num(row.get(key)) for row in rows]
        observed = [value for value in values if value is not None]
        coverage = len(observed) / max(len(rows), 1)
        if coverage < 0.60:
            feature_stats[key] = {"coverage": coverage, "status": "dropped_low_coverage"}
            continue
        med = _median(observed)
        q1 = _quantile(observed, 0.25)
        q3 = _quantile(observed, 0.75)
        iqr = max(q3 - q1, 1e-9)
        if iqr <= 1e-8:
            feature_stats[key] = {"coverage": coverage, "median": med, "iqr": iqr, "status": "dropped_zero_iqr"}
            continue
        filled = [med if value is None else value for value in values]
        corr = _pearson(filled, target)
        if not math.isfinite(corr) or abs(corr) < 0.01:
            feature_stats[key] = {"coverage": coverage, "median": med, "iqr": iqr, "correlation": corr, "status": "dropped_low_signal"}
            continue
        weight = max(-0.25, min(0.25, corr))
        raw_weights[key] = weight
        feature_stats[key] = {"coverage": coverage, "median": med, "iqr": iqr, "correlation": corr, "status": "used"}
    denom = sum(abs(value) for value in raw_weights.values())
    weights = {key: value / denom for key, value in sorted(raw_weights.items())} if denom > 0.0 else _fallback_weights()
    for key, weight in weights.items():
        feature_stats.setdefault(key, {"status": "used_fallback", "median": 0.0, "iqr": 1.0, "coverage": 0.0})
        feature_stats[key]["weight"] = weight
    return {
        "version": CANDIDATE_SURFACING_RECOVERY_VERSION,
        "usage_contract": CANDIDATE_SURFACING_USAGE_CONTRACT,
        "source_window": "train",
        "feature_keys": list(CAUSAL_FEATURE_KEYS),
        "label_keys": list(LABEL_KEYS),
        "used_feature_count": len(weights),
        "weights": weights,
        "feature_stats": feature_stats,
        "normalization": "train_median_iqr_missing_to_train_median_zclip4",
        "target": "composite_train_only_oracle_recall_label",
        "label_policy": {
            "best_route_shadow_r_clip": [-5.0, 30.0],
            "same_day_replacement_value_r_clip": [-10.0, 30.0],
            "controlled_path": "mfe_r>=3 net_r>0 mae_r>=-8",
        },
    }


def score_causal_feature_rows(rows: Iterable[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    weights = dict(profile.get("weights") or {})
    stats = dict(profile.get("feature_stats") or {})
    scored: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        components: dict[str, float] = {}
        missing = 0
        total = 0.0
        for key, weight in weights.items():
            stat = dict(stats.get(key) or {})
            med = _num(stat.get("median"))
            iqr = max(_num(stat.get("iqr"), 1.0), 1e-9)
            value = _optional_num(row.get(key))
            if value is None:
                missing += 1
                value = med
            z = max(-4.0, min(4.0, (value - med) / iqr))
            comp = float(weight) * z
            components[key] = comp
            total += comp
        score = total - 0.01 * missing
        row["causal_ranker_score_raw"] = float(total)
        row["causal_ranker_missing_feature_count"] = int(missing)
        row["causal_ranker_score"] = float(score)
        row["causal_ranker_components"] = dict(sorted(components.items(), key=lambda item: abs(item[1]), reverse=True)[:12])
        scored.append(row)
    ranked: list[dict[str, Any]] = []
    for _day, day_rows in _rows_by_day(scored).items():
        ordered = sorted(day_rows, key=_rank_sort_key)
        for rank, row in enumerate(ordered, start=1):
            out = dict(row)
            out["causal_rank_in_day"] = rank
            ranked.append(out)
    ranked.sort(key=lambda item: (str(item.get("trade_date") or ""), int(item.get("causal_rank_in_day") or 0), str(item.get("symbol") or "")))
    return ranked


def build_candidate_pools(
    scored_rows: Iterable[dict[str, Any]],
    existing_pool_by_day: dict[str, tuple[str, ...]] | None,
    variants: Iterable[PoolVariant] = POOL_VARIANTS,
) -> dict[str, list[dict[str, Any]]]:
    existing_pool_by_day = dict(existing_pool_by_day or {})
    by_day = _rows_by_day(scored_rows)
    pools: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        rows: list[dict[str, Any]] = []
        for day, day_rows in by_day.items():
            ordered = sorted(day_rows, key=_rank_sort_key)
            if variant.kind == "leading_sector_quota":
                symbols = _leading_sector_pool(ordered, variant.pool_size)
            elif variant.kind == "blend_existing_50pct":
                symbols = _blend_existing_pool(ordered, existing_pool_by_day.get(day, ()), variant.pool_size)
            else:
                symbols = [str(row.get("symbol") or "") for row in ordered[: variant.pool_size]]
            row_by_symbol = {str(row.get("symbol") or ""): row for row in day_rows}
            for rank, symbol in enumerate(symbols[: variant.pool_size], start=1):
                source = row_by_symbol.get(symbol)
                if source is None:
                    continue
                out = dict(source)
                out["pool_variant"] = variant.name
                out["pool_size"] = variant.pool_size
                out["pool_rank"] = rank
                out["pool_active"] = rank <= variant.active_count
                out["frontier_role_for_replay"] = "initial_active" if rank <= variant.active_count else "frontier_shadow"
                rows.append(out)
        rows.sort(key=lambda item: (str(item.get("trade_date") or ""), int(item.get("pool_rank") or 0), str(item.get("symbol") or "")))
        pools[variant.name] = rows
    return pools


def summarize_pool_recall(
    scored_rows: Iterable[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    routes: list[dict[str, Any]],
    mutations: dict[str, Any],
) -> dict[str, Any]:
    scored = [dict(row) for row in scored_rows]
    ranker_recall = _ranker_recall(scored)
    variants: dict[str, Any] = {}
    for name, pool_rows in pools.items():
        variants[name] = _variant_recall(scored, pool_rows, routes, mutations)
    return {
        "ranker_recall_at": ranker_recall,
        "variants": variants,
        "best_variant_by_quality_delayed_recall": _best_variant(variants, "best_quality_delayed_oracle_in_pool_share"),
        "best_variant_by_route_eligible_share": _best_variant(variants, "route_eligible_share"),
    }


def render_candidate_surfacing_report(summary: dict[str, Any]) -> str:
    route_bundle = dict(summary.get("route_bundle") or {})
    matched_route_bundle = dict(summary.get("matched_incumbent_route_bundle") or {})
    lines = [
        "# KALCB Candidate Surfacing Recovery",
        "",
        f"- Version: `{summary.get('version')}`",
        f"- Usage: {summary.get('usage_contract')}",
        "- Oracle labels are ex-post research labels; live scores use only causal first30/daily/sector inputs.",
        f"- Replay route bundle: `{route_bundle.get('name', '')}`; modes: {', '.join(route_bundle.get('modes') or [])}; first30/open enabled: {route_bundle.get('first30_open_enabled')}.",
        f"- Matched incumbent sanity bundle: `{matched_route_bundle.get('name', '')}`; first30/open enabled: {matched_route_bundle.get('first30_open_enabled')}.",
        "",
    ]
    for window in ("train", "holdout"):
        block = dict(summary.get(window) or {})
        recall = dict(block.get("recall") or {})
        replay = dict(block.get("replay") or {})
        matched_replay = dict(block.get("matched_incumbent_replay") or {})
        matched_active_replay = dict(block.get("matched_active_budget_replay") or {})
        lines.extend([f"## {window.title()}", ""])
        lines.append(f"- Feature rows: {block.get('feature_row_count', 0)}; oracle-labeled rows: {block.get('oracle_labeled_row_count', 0)}")
        lines.append(f"- Best quality-delayed recall variant: `{(recall.get('best_variant_by_quality_delayed_recall') or {}).get('name', '')}`")
        lines.append(f"- Best route-eligible variant: `{(recall.get('best_variant_by_route_eligible_share') or {}).get('name', '')}`")
        lines.append("")
        lines.append("| Variant | Best in pool | Quality delayed best | Route eligible | Trades | Net | DD | MFE capture |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        variants = dict((recall.get("variants") or {}))
        replay_rows = {row.get("variant"): row for row in replay.get("rows") or []}
        for name, row in variants.items():
            rr = dict(replay_rows.get(name) or {})
            metrics = dict(rr.get("metrics") or {})
            lines.append(
                f"| {name} | {_pct(row.get('best_oracle_in_pool_share'))} | "
                f"{_pct(row.get('best_quality_delayed_oracle_in_pool_share'))} | {_pct(row.get('route_eligible_share'))} | "
                f"{_num(metrics.get('trade_count')):.0f} | {_pct(metrics.get('broker_net_return_pct'))} | "
                f"{_pct(metrics.get('broker_max_drawdown_pct'))} | {_pct(metrics.get('avg_mfe_capture'))} |"
            )
        lines.append("")
        best_replay = dict(replay.get("best_by_frequency_adjusted_net") or replay.get("best_by_net") or {})
        route_modes = dict(best_replay.get("entry_route_mode_summary") or {})
        if route_modes:
            pieces = []
            for mode, metrics in route_modes.items():
                mode_metrics = dict(metrics or {})
                pieces.append(
                    f"{mode}: {int(_num(mode_metrics.get('trades')))} trades, "
                    f"capture {_pct(mode_metrics.get('avg_mfe_capture'))}, "
                    f"EOD {_pct(mode_metrics.get('eod_flatten_share'))}"
                )
            lines.append(f"- Best replay actual route mix: {'; '.join(pieces)}")
            lines.append("")
        lines.append("Replay note: constrained replay uses delayed pullback, AVWAP, and OR-high routes only. Top-N variants mainly diagnose recall beyond the active top 8; executable replay changes only when route-eligible candidates enter the rank-capped branch universe.")
        matched_best = dict(matched_replay.get("best_by_frequency_adjusted_net") or matched_replay.get("best_by_net") or {})
        matched_metrics = dict(matched_best.get("metrics") or {})
        if matched_best:
            lines.append(
                f"Matched incumbent execution sanity: `{matched_best.get('variant', '')}` "
                f"trades={_num(matched_metrics.get('trade_count')):.0f}, "
                f"net={_pct(matched_metrics.get('broker_net_return_pct'))}, "
                f"DD={_pct(matched_metrics.get('broker_max_drawdown_pct'))}, "
                f"capture={_pct(matched_metrics.get('avg_mfe_capture'))}."
            )
            matched_modes = dict(matched_best.get("entry_route_mode_summary") or {})
            if matched_modes:
                pieces = []
                for mode, metrics in matched_modes.items():
                    mode_metrics = dict(metrics or {})
                    pieces.append(
                        f"{mode}: {int(_num(mode_metrics.get('trades')))} trades, "
                        f"capture {_pct(mode_metrics.get('avg_mfe_capture'))}, "
                        f"EOD {_pct(mode_metrics.get('eod_flatten_share'))}"
                    )
                lines.append(f"- Matched route mix: {'; '.join(pieces)}")
            lines.append("Matched note: this keeps the incumbent first30/open anchor and sizing, so it is the candidate-source-only sanity check; it is not the conservative delayed-route promotion path.")
        matched_active_best = dict(matched_active_replay.get("best_by_frequency_adjusted_net") or matched_active_replay.get("best_by_net") or {})
        matched_active_metrics = dict(matched_active_best.get("metrics") or {})
        if matched_active_best:
            lines.append(
                f"Active-budget matched sanity: `{matched_active_best.get('variant', '')}` "
                f"trades={_num(matched_active_metrics.get('trade_count')):.0f}, "
                f"net={_pct(matched_active_metrics.get('broker_net_return_pct'))}, "
                f"DD={_pct(matched_active_metrics.get('broker_max_drawdown_pct'))}, "
                f"capture={_pct(matched_active_metrics.get('avg_mfe_capture'))}, "
                f"active_budget={_num(matched_active_metrics.get('active_budget_candidate_count')):.0f}."
            )
            lines.append("Active-budget note: this is the closest same-frequency candidate-source test because daily active counts are inherited from the incumbent snapshot.")
        lines.append("")
    root = dict(summary.get("root_cause_layer_attribution") or {})
    lines.extend(
        [
            "## Layer Attribution",
            "",
            f"- Candidate surfacing: {root.get('candidate_surfacing', '')}",
            f"- Candidate selection: {root.get('candidate_selection', '')}",
            f"- Entry route: {root.get('entry_route', '')}",
            f"- Exit/path management: {root.get('exit_path_management', '')}",
            "",
        ]
    )
    return "\n".join(lines)


def feature_coverage(rows: Iterable[dict[str, Any]], keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    data = [dict(row) for row in rows]
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        present = sum(1 for row in data if _optional_num(row.get(key)) is not None)
        out[key] = {"present": present, "total": len(data), "coverage": present / max(len(data), 1)}
    return out


def _evaluate_replay_variants(
    window: str,
    config: dict[str, Any],
    bundle: dict[str, Any],
    pools: dict[str, list[dict[str, Any]]],
    seed_mutations: dict[str, Any],
    baseline_metrics: dict[str, Any],
    output_dir: Path,
    variants: list[PoolVariant],
    *,
    replay_name: str,
    active_limit_by_day: dict[str, int] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        pool_rows = pools.get(variant.name, [])
        result = evaluate_compiled_candidate_pool(
            window=window,
            variant=variant,
            config=config,
            dataset=bundle["dataset"],
            context_by_key=bundle["context_by_key"],
            pool_rows=pool_rows,
            seed_mutations=seed_mutations,
            output_dir=output_dir,
            replay_name=replay_name,
            active_limit_by_day=active_limit_by_day,
        )
        result["baseline_delta"] = _metric_delta(result.get("metrics") or {}, baseline_metrics)
        rows.append(result)
    return {
        "replay_name": replay_name,
        "baseline_metrics": _compact_metrics(baseline_metrics),
        "rows": rows,
        "best_by_net": _best_replay_row(rows, "broker_net_return_pct"),
        "best_by_frequency_adjusted_net": _best_frequency_adjusted_replay(rows),
    }


def evaluate_compiled_candidate_pool(
    *,
    window: str,
    variant: PoolVariant,
    config: dict[str, Any],
    dataset: Any,
    context_by_key: dict[tuple[date, str], Any],
    pool_rows: list[dict[str, Any]],
    seed_mutations: dict[str, Any],
    output_dir: str | Path | None = None,
    replay_name: str = "candidate_pool",
    active_limit_by_day: dict[str, int] | None = None,
) -> dict[str, Any]:
    dates = tuple(dataset.trading_dates)
    cfg = KALCBConfig.from_mapping(config, seed_mutations)
    routes = _configured_entry_routes(seed_mutations)
    by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        by_day[date.fromisoformat(str(row.get("trade_date"))[:10])].append(row)
    selections: list[Selection] = []
    frontier_by_day: dict[date, tuple[str, ...]] = {}
    frontier_scores_by_day: dict[date, dict[str, float]] = {}
    eligible_keys: set[tuple[date, str]] = set()
    full_pool_count = 0
    active_budget_count = 0
    for day in dates:
        ordered = sorted(by_day.get(day, ()), key=lambda item: (int(item.get("pool_rank") or 999), str(item.get("symbol") or "")))
        frontier_by_day[day] = tuple(str(row.get("symbol") or "") for row in ordered)
        frontier_scores_by_day[day] = {str(row.get("symbol") or ""): _num(row.get("causal_ranker_score")) for row in ordered}
        full_pool_count += len(ordered)
        for row in ordered:
            symbol = str(row.get("symbol") or "")
            if symbol and _pool_row_static_route_eligible(row, routes, seed_mutations):
                eligible_keys.add((day, symbol))
        active_limit = int(variant.active_count)
        if active_limit_by_day is not None:
            active_limit = max(0, int(active_limit_by_day.get(day.isoformat(), 0)))
        active_budget_count += min(active_limit, len(ordered))
        for row in ordered[:active_limit]:
            selections.append(Selection(day, str(row.get("symbol") or ""), _num(row.get("causal_ranker_score")), "causal_candidate_surfacing"))
    counts = _selection_counts(selections, dates)
    filtered_bars = {key: bars for key, bars in dataset.bars_by_key.items() if key in eligible_keys}
    replay_dataset = replace(dataset, bars_by_key=filtered_bars)
    compiled = compile_core_replay(
        selections,
        replay_dataset,
        context_by_key,
        dates,
        counts,
        cfg,
        frontier_by_day=frontier_by_day,
        frontier_scores_by_day=frontier_scores_by_day,
        source_calibration_metadata={
            "candidate_surfacing_recovery_version": CANDIDATE_SURFACING_RECOVERY_VERSION,
            "replay_name": replay_name,
            "pool_variant": variant.name,
            "active_count": variant.active_count,
            "active_limit_contract": "incumbent_daily_active_count" if active_limit_by_day is not None else "variant_active_count",
            "active_budget_candidate_count": active_budget_count,
            "pool_size": variant.pool_size,
            "replay_static_prefilter": "route_static_eligible_only_preserve_original_frontier_rank",
            "full_pool_count": full_pool_count,
            "static_route_eligible_count": len(eligible_keys),
        },
    )
    if not compiled.snapshots or not compiled.bars:
        return {
            "window": window,
            "variant": variant.name,
            "replay_name": replay_name,
            "metrics": {
                "trade_count": 0.0,
                "candidate_pool_count": 0.0,
                "full_candidate_pool_count": float(full_pool_count),
                "static_route_eligible_count": float(len(eligible_keys)),
            },
            "fold_rows": tuple(),
            "trade_rows_path": "",
            "trade_count": 0,
            "entry_route_mode_summary": {},
            "replay_digest": {},
            "compiled_replay": {
                "session_count": len(dates),
                "selection_count": len(selections),
                "active_budget_candidate_count": active_budget_count,
                "candidate_pool_count": 0,
                "full_candidate_pool_count": full_pool_count,
                "static_route_eligible_count": len(eligible_keys),
            },
        }
    costs = BrokerCosts(commission_bps=cfg.commission_bps, tax_bps_on_sell=cfg.tax_bps_on_sell, slippage_bps=cfg.slippage_bps)
    adapter = KALCBReplayAdapter(cfg, _clone_snapshots_for_replay(compiled.snapshots), initial_equity=compiled.initial_equity, costs=costs)
    replay = run_replay(
        compiled.bars,
        adapter,
        initial_equity=compiled.initial_equity,
        costs=costs,
        close_open_positions=False,
        bars_are_ordered=True,
        buying_power_leverage=max(float(cfg.intraday_leverage), 1.0),
    )
    replay.decisions.extend(adapter._sync_new_fills(replay.broker))
    adapter.finalize_frontier_shadow(compiled.bars[-1] if compiled.bars else None)
    trades = _collapse_exit_legs(replay.trades)
    outcomes = _broker_trades_to_slot_outcomes(trades, cfg)
    metrics = summarize_outcomes(outcomes, session_dates=dates, selection_counts=counts)
    _add_compiled_candidate_pool_metrics(metrics, compiled, dates, len(outcomes))
    broker_metrics = compute_trade_metrics(trades, replay.equity_curve, initial_equity=compiled.initial_equity)
    final_equity = float(replay.equity_curve[-1]) if replay.equity_curve else compiled.initial_equity
    metrics.update(
        {
            "broker_net_return_pct": float(broker_metrics.get("net_return_pct", 0.0)),
            "official_mtm_net_return_pct": final_equity / max(float(compiled.initial_equity), 1.0) - 1.0,
            "final_equity": final_equity,
            "end_open_position_count": float(len(replay.broker.positions)),
            "broker_net_profit": float(broker_metrics.get("net_profit", 0.0)),
            "broker_max_drawdown_pct": float(broker_metrics.get("max_drawdown_pct", 0.0)),
            "broker_expected_total_r": float(broker_metrics.get("expected_total_r", 0.0)),
            "broker_avg_r": float(broker_metrics.get("avg_r", 0.0)),
            "broker_mfe_capture": float(broker_metrics.get("mfe_capture", 0.0)),
            "broker_trade_count": float(broker_metrics.get("total_trades", 0.0)),
            "same_bar_fill_count": float(replay.broker.same_bar_fill_violations),
            "mark_to_market_equity_points": float(len(replay.equity_curve)),
            "candidate_snapshot_hash": compiled.candidate_artifact_hash,
            "source_fingerprint": compiled.source_fingerprint,
            "full_candidate_pool_count": float(full_pool_count),
            "active_budget_candidate_count": float(active_budget_count),
            "static_route_eligible_count": float(len(eligible_keys)),
            "replay_static_prefilter": "route_static_eligible_only_preserve_original_frontier_rank",
        }
    )
    metrics.update(adapter.frontier_metrics())
    _add_portfolio_equivalent_metrics(metrics, outcomes, dates, compiled.initial_equity)
    _add_return_divergence_metrics(metrics)
    folds = _resolve_folds(dates, 2)
    fold_rows = _fold_metrics_from_outcomes_for_dates(outcomes, dates, folds, counts, initial_equity=compiled.initial_equity)
    _add_fold_metrics(metrics, fold_rows)
    trade_rows = _broker_trade_rows(trades)
    entry_route_mode_summary = _entry_route_mode_summary(trade_rows)
    trades_path = ""
    if output_dir is not None:
        trades_path = str(Path(output_dir) / f"candidate_pool_replay_trades_{replay_name}_{window}_{variant.name}.jsonl")
        write_jsonl(trades_path, trade_rows)
    return {
        "window": window,
        "variant": variant.name,
        "replay_name": replay_name,
        "metrics": _compact_metrics(metrics),
        "fold_rows": fold_rows,
        "trade_rows_path": trades_path,
        "trade_count": len(trade_rows),
        "entry_route_mode_summary": entry_route_mode_summary,
        "replay_digest": _replay_digest(replay, trades),
        "compiled_replay": {
            "session_count": len(dates),
            "selection_count": len(selections),
            "active_budget_candidate_count": active_budget_count,
            "candidate_pool_count": sum(len(items) for items in frontier_by_day.values()),
            "compiled_candidate_pool_count": sum(int((compiled.snapshots.get(day).metadata or {}).get("candidate_pool_count", 0) or 0) for day in dates if compiled.snapshots.get(day) is not None),
            "full_candidate_pool_count": full_pool_count,
            "static_route_eligible_count": len(eligible_keys),
            "candidate_artifact_hash": compiled.candidate_artifact_hash,
            "source_fingerprint": compiled.source_fingerprint,
        },
    }


def _existing_candidate_context(existing_context: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, tuple[str, ...]], dict[str, int]]:
    meta_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    pool_by_day: dict[str, tuple[str, ...]] = {}
    active_count_by_day: dict[str, int] = {}
    compiled = getattr(existing_context, "compiled_replay", None)
    snapshots = dict(getattr(compiled, "snapshots", {}) or {})
    for day, snapshot in sorted(snapshots.items(), key=lambda item: str(item[0])):
        day_label = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
        ordered: list[tuple[int, str]] = []
        active_symbols = {
            str(symbol)
            for symbol in (getattr(snapshot, "metadata", {}) or {}).get("active_symbols", ())
            if str(symbol)
        }
        inferred_active: set[str] = set()
        for candidate in tuple(getattr(snapshot, "candidates", ()) or ()):
            meta = _candidate_snapshot_metadata(candidate, day_label)
            symbol = str(meta.get("symbol") or getattr(candidate, "symbol", ""))
            meta_by_key[(day_label, symbol)] = meta
            ordered.append((int(_num(meta.get("frontier_rank"), len(ordered) + 1)), symbol))
            if bool(meta.get("frontier_initial_active")):
                inferred_active.add(symbol)
        pool_by_day[day_label] = tuple(symbol for _rank, symbol in sorted(ordered))
        active_count_by_day[day_label] = len(active_symbols or inferred_active)
    return meta_by_key, pool_by_day, active_count_by_day


def _best_oracle_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        day = str(row.get("trade_date") or row.get("entry_date") or "")[:10]
        symbol = str(row.get("symbol") or "")
        if not day or not symbol:
            continue
        key = (day, symbol)
        current = out.get(key)
        score = (_num(row.get("oracle_score")), _num(row.get("net_r")), _num(row.get("mfe_r")))
        current_score = (_num((current or {}).get("oracle_score")), _num((current or {}).get("net_r")), _num((current or {}).get("mfe_r")))
        if current is None or score > current_score:
            out[key] = dict(row)
    return out


def _top_decile_oracle_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[str(row.get("trade_date") or "")[:10]].append(dict(row))
    out: set[tuple[str, str]] = set()
    for day, items in by_day.items():
        ordered = sorted(items, key=lambda item: (_num(item.get("oracle_score")), _num(item.get("mfe_r"))), reverse=True)
        top_n = max(1, math.ceil(0.10 * len(ordered)))
        for row in ordered[:top_n]:
            out.add((day, str(row.get("symbol") or "")))
    return out


def _attach_labels(row: dict[str, Any], top_decile: set[tuple[str, str]]) -> None:
    day = str(row.get("trade_date") or "")[:10]
    symbol = str(row.get("symbol") or "")
    net_r = _optional_num(row.get("net_r"))
    mfe_r = _optional_num(row.get("mfe_r"))
    mae_r = _optional_num(row.get("mae_r"))
    actual = _optional_num(row.get("same_day_actual_total_r"))
    available = net_r is not None
    best_r = _clip(net_r or 0.0, -5.0, 30.0) if available else 0.0
    replacement = _clip((net_r or 0.0) - (actual or 0.0), -10.0, 30.0) if available else 0.0
    top_decile_label = 1.0 if (day, symbol) in top_decile else 0.0
    controlled = 1.0 if available and (mfe_r or 0.0) >= 3.0 and (net_r or 0.0) > 0.0 and (mae_r or 0.0) >= -8.0 else 0.0
    best_scaled = (best_r + 5.0) / 35.0
    replacement_scaled = (replacement + 10.0) / 40.0
    row["label_best_route_shadow_r"] = best_r
    row["label_top_decile_oracle"] = top_decile_label
    row["label_routeable_positive_mfe_controlled_mae"] = controlled
    row["label_same_day_replacement_value_r"] = replacement
    row["label_composite_oracle_recall"] = 0.35 * top_decile_label + 0.30 * controlled + 0.25 * best_scaled + 0.10 * replacement_scaled


def _attach_day_sector_context(rows: list[dict[str, Any]]) -> None:
    for _day, day_rows in _rows_by_day(rows).items():
        sector_counts = Counter(str(row.get("sector") or "UNKNOWN") for row in day_rows)
        leading_counts = Counter(str(row.get("sector") or "UNKNOWN") for row in day_rows if bool(row.get("leading_sector_cluster")))
        max_count = max(sector_counts.values(), default=1)
        for row in day_rows:
            sector = str(row.get("sector") or "UNKNOWN")
            row["same_day_sector_candidate_count"] = int(sector_counts.get(sector, 0))
            row["same_day_sector_candidate_share"] = sector_counts.get(sector, 0) / max(len(day_rows), 1)
            row["same_day_sector_crowding_pressure"] = sector_counts.get(sector, 0) / max(max_count, 1)
            row["same_day_leading_sector_candidate_count"] = int(leading_counts.get(sector, 0))


def _variant_recall(
    scored_rows: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    mutations: dict[str, Any],
) -> dict[str, Any]:
    pool_by_day_symbols: dict[str, set[str]] = defaultdict(set)
    pool_by_day_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        day = str(row.get("trade_date") or "")[:10]
        pool_by_day_symbols[day].add(str(row.get("symbol") or ""))
        pool_by_day_rows[day].append(row)
    by_day = _rows_by_day(scored_rows)
    best_hits = 0
    best_days = 0
    quality_hits = 0
    quality_days = 0
    top_decile_fracs: list[float] = []
    in_pool_r: list[float] = []
    out_pool_r: list[float] = []
    in_pool_mfe: list[float] = []
    out_pool_mfe: list[float] = []
    for day, day_rows in by_day.items():
        oracle_rows = [row for row in day_rows if row.get("oracle_label_available")]
        if not oracle_rows:
            continue
        pool = pool_by_day_symbols.get(day, set())
        best = max(oracle_rows, key=lambda row: (_num(row.get("oracle_score")), _num(row.get("mfe_r")), str(row.get("symbol") or "")))
        best_days += 1
        best_hits += int(str(best.get("symbol") or "") in pool)
        quality = [
            row
            for row in oracle_rows
            if str(row.get("oracle_route_family") or "") != "first30_open"
            and _num(row.get("first30_rel_volume")) >= 2.0
            and _num(row.get("first30_signal_bar_cpr")) >= 0.55
        ]
        if quality:
            qbest = max(quality, key=lambda row: (_num(row.get("oracle_score")), _num(row.get("mfe_r")), str(row.get("symbol") or "")))
            quality_days += 1
            quality_hits += int(str(qbest.get("symbol") or "") in pool)
        top_decile = [row for row in oracle_rows if _num(row.get("label_top_decile_oracle")) > 0.0]
        if top_decile:
            top_decile_fracs.append(sum(1 for row in top_decile if str(row.get("symbol") or "") in pool) / len(top_decile))
        for row in oracle_rows:
            target_r = in_pool_r if str(row.get("symbol") or "") in pool else out_pool_r
            target_mfe = in_pool_mfe if str(row.get("symbol") or "") in pool else out_pool_mfe
            target_r.append(_num(row.get("net_r")))
            target_mfe.append(_num(row.get("mfe_r")))
    route = _route_eligibility_summary(pool_rows, routes, mutations)
    sector = _sector_pool_summary(pool_by_day_rows)
    return {
        "pool_candidate_count": len(pool_rows),
        "pool_day_count": len(pool_by_day_symbols),
        "avg_pool_size": len(pool_rows) / max(len(pool_by_day_symbols), 1),
        "best_oracle_in_pool_share": best_hits / max(best_days, 1),
        "best_oracle_days": best_days,
        "best_quality_delayed_oracle_in_pool_share": quality_hits / max(quality_days, 1),
        "best_quality_delayed_oracle_days": quality_days,
        "top_decile_oracle_recall": sum(top_decile_fracs) / max(len(top_decile_fracs), 1),
        "avg_in_pool_net_r": _avg(in_pool_r),
        "avg_out_of_pool_net_r": _avg(out_pool_r),
        "avg_in_pool_mfe_r": _avg(in_pool_mfe),
        "avg_out_of_pool_mfe_r": _avg(out_pool_mfe),
        **route,
        **sector,
    }


def _ranker_recall(scored_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_day = _rows_by_day(scored_rows)
    out: dict[str, dict[str, float]] = {}
    for k in RECALL_KS:
        best_hits = 0
        best_days = 0
        overlap_fracs: list[float] = []
        for _day, day_rows in by_day.items():
            oracle = [row for row in day_rows if row.get("oracle_label_available")]
            if not oracle:
                continue
            ranked = sorted(day_rows, key=_rank_sort_key)[:k]
            ranker_symbols = {str(row.get("symbol") or "") for row in ranked}
            oracle_order = sorted(oracle, key=lambda row: (_num(row.get("oracle_score")), _num(row.get("mfe_r"))), reverse=True)
            best_days += 1
            best_hits += int(str(oracle_order[0].get("symbol") or "") in ranker_symbols)
            oracle_top = {str(row.get("symbol") or "") for row in oracle_order[: min(k, len(oracle_order))]}
            overlap_fracs.append(len(ranker_symbols & oracle_top) / max(len(oracle_top), 1))
        out[str(k)] = {
            "best_oracle_in_ranker_topk_share": best_hits / max(best_days, 1),
            "oracle_topk_overlap_share": sum(overlap_fracs) / max(len(overlap_fracs), 1),
        }
    return out


def _route_eligibility_summary(pool_rows: list[dict[str, Any]], routes: list[dict[str, Any]], mutations: dict[str, Any]) -> dict[str, Any]:
    eligible_count = 0
    route_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for row in pool_rows:
        meta = _pool_route_meta(row)
        passed_any = False
        first_reason = ""
        for route in routes:
            passed, reason = _route_candidate_passes(route, mutations, meta)
            if passed:
                passed_any = True
                route_counts[str(route.get("mode") or route.get("name") or "route")] += 1
            elif not first_reason:
                first_reason = reason
        if passed_any:
            eligible_count += 1
        else:
            blocker_counts[first_reason or "not_route_eligible"] += 1
    return {
        "route_eligible_candidate_count": eligible_count,
        "route_eligible_share": eligible_count / max(len(pool_rows), 1),
        "route_eligible_counts_by_mode": dict(route_counts),
        "top_route_blockers": blocker_counts.most_common(10),
    }


def _pool_row_static_route_eligible(row: dict[str, Any], routes: list[dict[str, Any]], mutations: dict[str, Any]) -> bool:
    meta = _pool_route_meta(row)
    return any(_route_candidate_passes(route, mutations, meta)[0] for route in routes)


def _pool_route_meta(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    rank = int(row.get("pool_rank") or row.get("causal_rank_in_day") or 999)
    out["candidate_rank"] = rank
    out["frontier_rank"] = rank
    out["frontier_initial_active"] = bool(row.get("pool_active"))
    out["frontier_role"] = str(row.get("frontier_role_for_replay") or ("initial_active" if row.get("pool_active") else "frontier_shadow"))
    out["frontier_selection_score"] = _num(row.get("causal_ranker_score"))
    return out


def _sector_pool_summary(pool_by_day_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    max_shares: list[float] = []
    sector_counts: list[int] = []
    leading_shares: list[float] = []
    for rows in pool_by_day_rows.values():
        counts = Counter(str(row.get("sector") or "UNKNOWN") for row in rows)
        max_shares.append(max(counts.values(), default=0) / max(len(rows), 1))
        sector_counts.append(len(counts))
        leading_shares.append(sum(1 for row in rows if bool(row.get("leading_sector_cluster"))) / max(len(rows), 1))
    return {
        "avg_max_sector_share": _avg(max_shares),
        "avg_sector_count": _avg(sector_counts),
        "avg_leading_sector_cluster_share": _avg(leading_shares),
    }


def _leading_sector_pool(ordered: list[dict[str, Any]], pool_size: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    sector_counts: Counter[str] = Counter()
    for row in ordered[:LEADING_SECTOR_BASE_COUNT]:
        symbol = str(row.get("symbol") or "")
        if symbol and symbol not in seen:
            selected.append(symbol)
            seen.add(symbol)
            sector_counts[str(row.get("sector") or "UNKNOWN")] += 1
    quota = 0
    for row in ordered:
        if quota >= LEADING_SECTOR_QUOTA:
            break
        symbol = str(row.get("symbol") or "")
        sector = str(row.get("sector") or "UNKNOWN")
        if not symbol or symbol in seen or not bool(row.get("leading_sector_cluster")):
            continue
        if sector_counts[sector] >= LEADING_SECTOR_MAX_PER_SECTOR:
            continue
        selected.append(symbol)
        seen.add(symbol)
        sector_counts[sector] += 1
        quota += 1
    for row in ordered:
        if len(selected) >= pool_size:
            break
        symbol = str(row.get("symbol") or "")
        if symbol and symbol not in seen:
            selected.append(symbol)
            seen.add(symbol)
    return selected[:pool_size]


def _blend_existing_pool(ordered: list[dict[str, Any]], existing: tuple[str, ...], pool_size: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    keep_existing = max(1, pool_size // 2)
    row_symbols = {str(row.get("symbol") or "") for row in ordered}
    for symbol in existing[:keep_existing]:
        if symbol in row_symbols and symbol not in seen:
            selected.append(symbol)
            seen.add(symbol)
    for row in ordered:
        if len(selected) >= pool_size:
            break
        symbol = str(row.get("symbol") or "")
        if symbol and symbol not in seen:
            selected.append(symbol)
            seen.add(symbol)
    return selected[:pool_size]


def _flatten_pool_rows(pools: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, items in sorted(pools.items()):
        for item in items:
            rows.append(
                {
                    key: item.get(key)
                    for key in (
                        "pool_variant",
                        "window",
                        "trade_date",
                        "symbol",
                        "sector",
                        "pool_rank",
                        "pool_active",
                        "frontier_role_for_replay",
                        "causal_ranker_score",
                        "causal_rank_in_day",
                        "first30_rel_volume",
                        "first30_signal_bar_cpr",
                        "sector_daily_score_pct",
                        "sector_intraday_score_pct",
                        "leading_sector_cluster",
                        "oracle_label_available",
                        "oracle_route_family",
                        "oracle_score",
                        "net_r",
                        "mfe_r",
                        "mae_r",
                    )
                    if key in item
                }
            )
    rows.sort(key=lambda item: (str(item.get("pool_variant") or ""), str(item.get("trade_date") or ""), int(item.get("pool_rank") or 0), str(item.get("symbol") or "")))
    return rows


def _entry_route_mode_summary(trade_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trade_rows:
        mode = str(row.get("entry_route_mode") or row.get("entry_type") or "unknown")
        by_mode[mode].append(row)
    out: dict[str, dict[str, float]] = {}
    for mode, rows in sorted(by_mode.items()):
        out[mode] = {
            "trades": float(len(rows)),
            "net_return_pct": sum(_num(row.get("net_return_pct")) for row in rows),
            "avg_mfe_capture": _avg(_num(row.get("mfe_capture")) for row in rows),
            "avg_mfe_r": _avg(_num(row.get("mfe_r")) for row in rows),
            "avg_mae_r": _avg(_num(row.get("mae_r")) for row in rows),
            "eod_flatten_share": sum(1 for row in rows if str(row.get("exit_reason") or "") == "eod_flatten") / max(len(rows), 1),
            "win_share": sum(1 for row in rows if _num(row.get("r")) > 0.0) / max(len(rows), 1),
        }
    return out


def _pool_hash_payload(pools: dict[str, list[dict[str, Any]]]) -> dict[str, list[tuple[str, str, int]]]:
    return {
        name: [
            (str(row.get("trade_date") or ""), str(row.get("symbol") or ""), int(row.get("pool_rank") or 0))
            for row in rows
        ]
        for name, rows in sorted(pools.items())
    }


def _root_cause_attribution(
    train_recall: dict[str, Any],
    holdout_recall: dict[str, Any],
    train_replay: dict[str, Any],
    holdout_replay: dict[str, Any],
    train_matched_replay: dict[str, Any] | None = None,
    holdout_matched_replay: dict[str, Any] | None = None,
    train_matched_active_replay: dict[str, Any] | None = None,
    holdout_matched_active_replay: dict[str, Any] | None = None,
    baseline_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    train_best = dict((train_recall.get("best_variant_by_quality_delayed_recall") or {}).get("metrics") or {})
    holdout_best = dict((holdout_recall.get("best_variant_by_quality_delayed_recall") or {}).get("metrics") or {})
    train_replay_best = dict((train_replay.get("best_by_frequency_adjusted_net") or {}).get("metrics") or {})
    holdout_replay_best = dict((holdout_replay.get("best_by_frequency_adjusted_net") or {}).get("metrics") or {})
    train_matched_best = dict(((train_matched_replay or {}).get("best_by_frequency_adjusted_net") or {}).get("metrics") or {})
    holdout_matched_best = dict(((holdout_matched_replay or {}).get("best_by_frequency_adjusted_net") or {}).get("metrics") or {})
    train_matched_active_best = dict(((train_matched_active_replay or {}).get("best_by_frequency_adjusted_net") or {}).get("metrics") or {})
    holdout_matched_active_best = dict(((holdout_matched_active_replay or {}).get("best_by_frequency_adjusted_net") or {}).get("metrics") or {})
    baseline_metrics = dict(baseline_metrics or {})
    train_baseline = dict(baseline_metrics.get("train") or {})
    holdout_baseline = dict(baseline_metrics.get("holdout") or {})
    train_recall_ok = _num(train_best.get("best_quality_delayed_oracle_in_pool_share")) >= 0.75
    holdout_recall_ok = _num(holdout_best.get("best_quality_delayed_oracle_in_pool_share")) >= 0.75
    drawdown_fail = _num(train_replay_best.get("broker_max_drawdown_pct")) > 0.08 or _num(holdout_replay_best.get("broker_max_drawdown_pct")) > 0.08
    capture_fail = _num(train_replay_best.get("avg_mfe_capture")) < 0.38 or _num(holdout_replay_best.get("avg_mfe_capture")) < 0.30
    matched_train_beats_baseline = _num(train_matched_best.get("broker_net_return_pct")) >= _num(train_baseline.get("broker_net_return_pct"))
    matched_holdout_beats_baseline = _num(holdout_matched_best.get("broker_net_return_pct")) >= _num(holdout_baseline.get("broker_net_return_pct"))
    matched_active_train_beats_baseline = _num(train_matched_active_best.get("broker_net_return_pct")) >= _num(train_baseline.get("broker_net_return_pct"))
    matched_active_holdout_beats_baseline = _num(holdout_matched_active_best.get("broker_net_return_pct")) >= _num(holdout_baseline.get("broker_net_return_pct"))
    return {
        "candidate_surfacing": (
            "causal_ranker_improves_recall_and_active_budget_matched_execution_does_not_show_candidate_value_destruction"
            if train_recall_ok and holdout_recall_ok and (matched_active_train_beats_baseline or matched_active_holdout_beats_baseline)
            else "causal_ranker_improves_recall_but_broad_matched_execution_was_overactive_not_apples_to_apples"
            if train_recall_ok and holdout_recall_ok and (matched_train_beats_baseline or matched_holdout_beats_baseline)
            else "causal_ranker_materially_improves_oracle_recall_but_delayed_route_replay_does_not_convert_it"
            if train_recall_ok and holdout_recall_ok
            else "causal_ranker_recall_still_weak_add_targeted_causal_features"
        ),
        "candidate_selection": "ranker_surfaces_better_names_but_rank_capped_delayed_routes_only_convert_when_route_eligible_names_enter_the_branch_universe",
        "entry_route": (
            "route_eligible_share_improved_above_round5_but_rank_relvol_quality_gates_still_constrain_conversion"
            if _num(train_best.get("route_eligible_share")) > 0.06
            else "delayed_route_bottleneck_still_present"
        ),
        "exit_path_management": (
            "replay_fails_risk_or_capture_gate_drawdown_and_mfe_leakage_remain_primary_conversion_failures"
            if drawdown_fail or capture_fail
            else "exit_management_not_primary_for_this_stage"
        ),
    }


def _best_variant(variants: dict[str, Any], key: str) -> dict[str, Any]:
    if not variants:
        return {}
    name, metrics = max(variants.items(), key=lambda item: (_num((item[1] or {}).get(key)), _num((item[1] or {}).get("route_eligible_share")), str(item[0])))
    return {"name": name, "metrics": metrics}


def _best_replay_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda row: (_num((row.get("metrics") or {}).get(metric)), _num((row.get("metrics") or {}).get("trade_count")), str(row.get("variant") or "")))


def _best_frequency_adjusted_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def key(row: dict[str, Any]) -> tuple[float, float, str]:
        metrics = dict(row.get("metrics") or {})
        net = _num(metrics.get("broker_net_return_pct"))
        trades = _num(metrics.get("trade_count"))
        dd = _num(metrics.get("broker_max_drawdown_pct"))
        return (net + 0.0005 * min(trades, 140.0) - 0.25 * max(dd - 0.08, 0.0), trades, str(row.get("variant") or ""))

    return max(rows, key=key)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "broker_net_return_pct",
        "official_mtm_net_return_pct",
        "broker_max_drawdown_pct",
        "trade_count",
        "active_days",
        "avg_trade_net_pct",
        "avg_mfe_capture",
        "avg_mfe_r",
        "avg_mae_r",
        "mae_le_neg_1_share",
        "worst_fold_net",
        "median_fold_net",
        "same_bar_fill_count",
        "end_open_position_count",
        "candidate_pool_count",
        "initial_active_candidate_count",
        "active_budget_candidate_count",
        "frontier_expansion_candidate_count",
        "candidate_pool_conversion",
        "initial_active_conversion",
        "full_candidate_pool_count",
        "static_route_eligible_count",
        "replay_static_prefilter",
        "candidate_snapshot_hash",
        "source_fingerprint",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _metric_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: _num(metrics.get(key)) - _num(baseline.get(key))
        for key in (
            "broker_net_return_pct",
            "broker_max_drawdown_pct",
            "trade_count",
            "avg_trade_net_pct",
            "avg_mfe_capture",
            "mae_le_neg_1_share",
        )
    }


def _add_fold_metrics(metrics: dict[str, Any], fold_rows: tuple[dict[str, Any], ...]) -> None:
    values = [
        _num((row.get("metrics") or {}).get("portfolio_equivalent_net_return_pct", (row.get("metrics") or {}).get("broker_net_return_pct")))
        for row in fold_rows
    ]
    dds = [
        _num((row.get("metrics") or {}).get("portfolio_equivalent_max_drawdown_pct", (row.get("metrics") or {}).get("broker_max_drawdown_pct")))
        for row in fold_rows
    ]
    metrics["worst_fold_net"] = min(values) if values else _num(metrics.get("broker_net_return_pct"))
    metrics["median_fold_net"] = median(values) if values else _num(metrics.get("broker_net_return_pct"))
    metrics["worst_fold_drawdown_pct"] = max(dds) if dds else _num(metrics.get("broker_max_drawdown_pct"))
    metrics["fold_count"] = float(len(fold_rows))


def _assert_no_label_leakage(keys: Iterable[str]) -> None:
    leaked = sorted(set(keys) & LEAKAGE_FEATURE_BLOCKLIST)
    if leaked:
        raise ValueError(f"causal ranker feature set contains leakage fields: {leaked}")


def _fallback_weights() -> dict[str, float]:
    keys = (
        "first30_rel_volume",
        "first30_signal_bar_cpr",
        "first30_vwap_ret",
        "daily_momentum_pct",
        "sector_daily_score_pct",
        "sector_intraday_score_pct",
        "stock_sector_daily_ret20_spread",
        "first30_sector_leadership_pct",
    )
    weight = 1.0 / len(keys)
    return {key: weight for key in keys}


def _rows_by_day(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("trade_date") or "")[:10]].append(dict(row))
    return dict(out)


def _rank_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        -_num(row.get("causal_ranker_score")),
        -_num(row.get("first30_rel_volume")),
        -_num(row.get("first30_signal_bar_cpr")),
        str(row.get("symbol") or ""),
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    y_var = sum((value - y_mean) ** 2 for value in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return 0.0
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / math.sqrt(x_var * y_var)


def _quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def _median(values: Iterable[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(median(data)) if data else 0.0


def _avg(values: Iterable[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(data) / len(data)) if data else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _optional_num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _num(value: Any, default: float = 0.0) -> float:
    out = _optional_num(value)
    return float(default) if out is None else out


def _pct(value: Any) -> str:
    return f"{100.0 * _num(value):.2f}%"


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
