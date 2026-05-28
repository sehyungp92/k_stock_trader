from __future__ import annotations

import json
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.auto.shared.types import Experiment
from backtests.core.replay_bundle import EventReplayBundle
from backtests.strategies.common.synthetic import make_synthetic_replay_bundle
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot
from strategy_kalcb.signals import classify_raw_breakout, close_location_value, compute_bar_rvol, compute_opening_range

from .phase_candidates import BASE_MUTATIONS
from .features import snapshots_from_bundle
from .replay_cache import load_kalcb_real_replay_bundle
from .runner import _synthetic_snapshot


DEFAULT_RESEARCH_HOLDOUT_DAYS = 42
DEFAULT_FOLD_COUNT = 2
DEFAULT_TOP_N = 10
DEFAULT_REFINE_TOP_N = 3
DEFAULT_MAX_REFINEMENT_CANDIDATES = 96
RAW_OPPORTUNITY_MODEL_VERSION = "kalcb_candidate_opportunity_v1"

RESEARCH_MUTATION_KEYS = {
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
    "kalcb.frontier.active_selection_mode",
}


def run_research_sweep(
    config: dict[str, Any] | None = None,
    *,
    mutations: dict[str, Any] | None = None,
    output_dir: str | Path = "data/backtests/output/kalcb/research_sweeps",
    holdout_days: int = DEFAULT_RESEARCH_HOLDOUT_DAYS,
    fold_days: int | None = None,
    fold_count: int = DEFAULT_FOLD_COUNT,
    top_n: int = DEFAULT_TOP_N,
    max_candidates: int | None = None,
    refine_top_n: int = DEFAULT_REFINE_TOP_N,
    max_refinement_candidates: int | None = DEFAULT_MAX_REFINEMENT_CANDIDATES,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Run a training-only KALCB research-parameter sweep.

    Candidates may only move research-selection knobs and the active-selection
    ordering mode. The evaluator scores raw candidate opportunity and does not
    execute the KALCB entry, exit, sizing, or trade-management core.
    """

    t0 = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / f"kalcb_research_sweep_progress_{int(time.time())}.jsonl"
    training_config = _training_config(dict(config or {}), holdout_days)
    base_mutations = _base_mutations(training_config, mutations)
    candidates = build_research_sweep_candidates()
    if max_candidates is not None:
        candidates = candidates[: max(0, int(max_candidates))]
    baseline = Experiment("__baseline__", {})
    _append_progress(
        progress_path,
        {
            "event": "started",
            "created_at": _utc_now_iso(),
            "coarse_candidates": len(candidates) + 1,
            "refine_top_n": int(refine_top_n),
            "max_refinement_candidates": int(max_refinement_candidates) if max_refinement_candidates is not None else None,
            "max_workers": int(max_workers),
        },
    )

    baseline_result = _run_candidate_opportunity(training_config, base_mutations, baseline, out, "full")
    folds = _resolve_folds(_result_dates(baseline_result), fold_days=fold_days, fold_count=fold_count)
    baseline_row = _evaluate_candidate(baseline, training_config, base_mutations, out, folds, full_result=baseline_result, stage="baseline")
    _append_progress(
        progress_path,
        {
            "event": "candidate_completed",
            "stage": "baseline",
            "completed": 1,
            "total": 1,
            "name": baseline_row["name"],
            "score": baseline_row["score"],
            "rejected": baseline_row["rejected"],
            "reject_reason": baseline_row["reject_reason"],
        },
    )
    coarse_rows = [
        baseline_row,
        *_evaluate_candidates(
            candidates,
            training_config=training_config,
            base_mutations=base_mutations,
            output_dir=out,
            folds=folds,
            max_workers=max_workers,
            stage="coarse",
            progress_path=progress_path,
        ),
    ]
    coarse_rows.sort(key=lambda row: (-float(row["score"]), bool(row["rejected"]), str(row["name"])))
    refinement_seeds = [row for row in coarse_rows if not row["rejected"]][: max(0, int(refine_top_n))]
    refinement_candidates = build_research_refinement_candidates(
        refinement_seeds,
        existing_mutations=[row.get("mutations") or {} for row in coarse_rows],
        max_candidates=max_refinement_candidates,
    )
    _append_progress(
        progress_path,
        {
            "event": "refinement_planned",
            "refinement_candidates": len(refinement_candidates),
            "seed_names": [row["name"] for row in refinement_seeds],
        },
    )
    refinement_rows = _evaluate_candidates(
        refinement_candidates,
        training_config=training_config,
        base_mutations=base_mutations,
        output_dir=out,
        folds=folds,
        max_workers=max_workers,
        stage="refinement",
        progress_path=progress_path,
    )
    rows = [*coarse_rows, *refinement_rows]
    rows.sort(key=lambda row: (-float(row["score"]), bool(row["rejected"]), str(row["name"])))
    selected = [row for row in rows if not row["rejected"]][:top_n]
    payload = {
        "strategy": "kalcb",
        "sweep_type": "research_candidate_opportunity_training_only",
        "opportunity_model_version": RAW_OPPORTUNITY_MODEL_VERSION,
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "holdout_contract": {
            "holdout_days": int(holdout_days),
            "selection_uses_holdout": False,
            "policy": "Last holdout_days are excluded by removing explicit end/holdout_start overrides before replay window resolution.",
        },
        "evaluation_policy": (
            "Scores the research-selected candidate list directly from next-session intraday bars. "
            "The evaluator does not run the KALCB order/exit engine and does not use realized PnL."
        ),
        "fold_days": int(fold_days) if fold_days is not None else None,
        "fold_count": int(fold_count),
        "folds": [{"start": start.isoformat(), "end": end.isoformat()} for start, end in folds],
        "training_config": training_config,
        "base_mutations": base_mutations,
        "candidate_count": len(rows),
        "coarse_candidate_count": len(coarse_rows),
        "refinement_candidate_count": len(refinement_rows),
        "refine_top_n": int(refine_top_n),
        "max_refinement_candidates": int(max_refinement_candidates) if max_refinement_candidates is not None else None,
        "refinement_seeds": [
            {"name": row["name"], "score": row["score"], "mutations": row.get("mutations") or {}}
            for row in refinement_seeds
        ],
        "selected_count": len(selected),
        "selection_frontier": selected,
        "rows": rows,
        "phase_auto_seed": _phase_auto_seed(training_config, base_mutations, selected[0] if selected else rows[0]),
    }
    payload["sweep_hash"] = stable_signature(
        {
            "training_config": training_config,
            "base_mutations": base_mutations,
            "rows": [{key: row.get(key) for key in ("name", "score", "mutations", "rejected")} for row in rows],
        }
    )
    json_path = out / f"kalcb_research_sweep_{payload['sweep_hash'][:12]}.json"
    md_path = out / f"kalcb_research_sweep_{payload['sweep_hash'][:12]}.md"
    seed_path = out / f"kalcb_phase_auto_seed_{payload['sweep_hash'][:12]}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    seed_path.write_text(json.dumps(payload["phase_auto_seed"], indent=2, sort_keys=True, default=str), encoding="utf-8")
    payload["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "phase_auto_seed": str(seed_path),
        "progress": str(progress_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _append_progress(
        progress_path,
        {
            "event": "completed",
            "sweep_hash": payload["sweep_hash"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "json": str(json_path),
        },
    )
    return payload


def build_research_sweep_candidates() -> list[Experiment]:
    raw: list[tuple[str, dict[str, Any]]] = []
    raw.extend(
        (f"top{count}", {"kalcb.research.top_long_count": count})
        for count in (10, 15, 20, 25, 30, 40)
    )
    raw.extend(
        (f"adv{int(value / 1_000_000_000)}b", {"kalcb.research.min_adv20_krw": value})
        for value in (1_000_000_000, 2_000_000_000, 5_000_000_000, 10_000_000_000)
    )
    raw.extend(
        [
            ("rs_heavy", _weights(rs=0.36, trend=0.18, compression=0.12, accumulation=0.12, stock=0.08, sector=0.06, participation=0.08)),
            ("trend_rs", _weights(rs=0.30, trend=0.28, compression=0.10, accumulation=0.12, stock=0.10, sector=0.05, participation=0.05)),
            ("compression_accum", _weights(rs=0.20, trend=0.17, compression=0.24, accumulation=0.22, stock=0.07, sector=0.04, participation=0.06)),
            ("sector_participation", _weights(rs=0.22, trend=0.18, compression=0.12, accumulation=0.12, stock=0.08, sector=0.10, participation=0.18)),
            ("stock_regime_balanced", _weights(rs=0.22, trend=0.22, compression=0.13, accumulation=0.13, stock=0.16, sector=0.07, participation=0.07)),
            ("score_active", {"kalcb.frontier.active_selection_mode": "score"}),
            ("liquidity_active", {"kalcb.frontier.active_selection_mode": "liquidity"}),
            ("hybrid_active", {"kalcb.frontier.active_selection_mode": "hybrid"}),
            ("campaign_active", {"kalcb.frontier.active_selection_mode": "campaign"}),
            ("hot_active", {"kalcb.frontier.active_selection_mode": "hot"}),
        ]
    )
    raw.extend(
        [
            ("rs_floor50", {"kalcb.research.min_rs_percentile": 50.0}),
            ("rs_floor60", {"kalcb.research.min_rs_percentile": 60.0}),
            ("rs_floor70", {"kalcb.research.min_rs_percentile": 70.0}),
            ("trend_floor45", {"kalcb.research.min_trend_score": 45.0}),
            ("trend_floor70", {"kalcb.research.min_trend_score": 70.0}),
            ("compression_floor20", {"kalcb.research.min_compression_score": 20.0}),
            ("compression_floor35", {"kalcb.research.min_compression_score": 35.0}),
            ("accum_floor0", {"kalcb.research.min_accumulation_score": 0.0}),
            ("accum_floor020", {"kalcb.research.min_accumulation_score": 0.20}),
            ("sector_participation50", {"kalcb.research.min_sector_participation": 0.50}),
            ("sector_participation67", {"kalcb.research.min_sector_participation": 0.67}),
            ("sector_daily45", {"kalcb.research.min_sector_daily_score_pct": 45.0}),
            ("sector_daily55", {"kalcb.research.min_sector_daily_score_pct": 55.0}),
            ("sector_daily65", {"kalcb.research.min_sector_daily_score_pct": 65.0}),
            ("box_range08", {"kalcb.research.max_box_range_pct": 0.08}),
            ("box_range12", {"kalcb.research.max_box_range_pct": 0.12}),
        ]
    )
    raw.extend(
        [
            ("top15_rs60_trend45", {"kalcb.research.top_long_count": 15, "kalcb.research.min_rs_percentile": 60.0, "kalcb.research.min_trend_score": 45.0}),
            ("top20_rs60_accum0", {"kalcb.research.top_long_count": 20, "kalcb.research.min_rs_percentile": 60.0, "kalcb.research.min_accumulation_score": 0.0}),
            ("top25_rs50_box12", {"kalcb.research.top_long_count": 25, "kalcb.research.min_rs_percentile": 50.0, "kalcb.research.max_box_range_pct": 0.12}),
            ("top20_compress_accum", {"kalcb.research.top_long_count": 20, **_weights(rs=0.20, trend=0.17, compression=0.24, accumulation=0.22, stock=0.07, sector=0.04, participation=0.06)}),
            ("top20_rs_heavy_floor60", {"kalcb.research.top_long_count": 20, "kalcb.research.min_rs_percentile": 60.0, **_weights(rs=0.36, trend=0.18, compression=0.12, accumulation=0.12, stock=0.08, sector=0.06, participation=0.08)}),
            ("top30_sector_campaign", {"kalcb.research.top_long_count": 30, "kalcb.frontier.active_selection_mode": "campaign", **_weights(rs=0.22, trend=0.18, compression=0.12, accumulation=0.12, stock=0.08, sector=0.10, participation=0.18)}),
        ]
    )
    return [Experiment(name, mutations) for name, mutations in raw]


def build_research_refinement_candidates(
    seed_rows: list[dict[str, Any]],
    *,
    existing_mutations: list[dict[str, Any]] | None = None,
    max_candidates: int | None = DEFAULT_MAX_REFINEMENT_CANDIDATES,
) -> list[Experiment]:
    """Build a bounded second-stage grid around the best coarse candidates."""

    limit = None if max_candidates is None else max(0, int(max_candidates))
    if limit == 0 or not seed_rows:
        return []
    existing = {_mutation_signature(mutations) for mutations in (existing_mutations or [])}
    buckets: dict[str, list[tuple[float, str, dict[str, Any]]]] = {
        "core": [],
        "gates": [],
        "filters": [],
        "weights": [],
    }

    for seed_index, row in enumerate(seed_rows, start=1):
        seed = _research_only_mutations(dict(row.get("mutations") or {}))
        merged = dict(row.get("merged_mutations") or {})
        seed_name = _safe_name(str(row.get("name") or f"seed{seed_index}"))
        base_top = int(seed.get("kalcb.research.top_long_count") or merged.get("kalcb.research.top_long_count") or 20)
        base_sector = _optional_float(seed.get("kalcb.research.min_sector_participation"))
        if base_sector is None:
            base_sector = _optional_float(merged.get("kalcb.research.min_sector_participation")) or 0.0
        base_active = str(seed.get("kalcb.frontier.active_selection_mode") or merged.get("kalcb.frontier.active_selection_mode") or "liquidity")
        top_values = _near_grid_values(base_top, (10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50), limit=6)
        sector_values = _near_grid_values(base_sector, (0.40, 0.50, 0.60, 0.67, 0.75, 0.80), limit=4)
        active_values = _ordered_active_modes(base_active)

        for mode in active_values:
            _append_refinement(
                buckets["core"],
                8.0 + (0.0 if mode == base_active else 0.10) + seed_index * 0.05,
                f"ref{seed_index}_{seed_name}_{mode}_active",
                {**seed, "kalcb.frontier.active_selection_mode": mode},
            )

        for top in top_values:
            for sector in sector_values:
                for mode in active_values:
                    distance = _distance(top, base_top, scale=10.0) + _distance(sector, base_sector, scale=0.25) + (0.0 if mode == base_active else 0.30)
                    _append_refinement(
                        buckets["core"],
                        10.0 + distance + seed_index * 0.05,
                        f"ref{seed_index}_{seed_name}_top{top}_sec{_pct_label(sector)}_{mode}",
                        {
                            **seed,
                            "kalcb.research.top_long_count": int(top),
                            "kalcb.research.min_sector_participation": float(sector),
                            "kalcb.frontier.active_selection_mode": mode,
                        },
                    )

        for key, values, label, scale in _soft_gate_grids():
            base_value = _optional_float(seed.get(key))
            if base_value is None:
                base_value = _optional_float(merged.get(key)) or 0.0
            for value in _near_grid_values(base_value, values, limit=4):
                _append_refinement(
                    buckets["gates"],
                    20.0 + _distance(value, base_value, scale=scale) + seed_index * 0.05,
                    f"ref{seed_index}_{seed_name}_{label}{_numeric_label(value)}",
                    {**seed, key: value},
                )

        for key, values, label, scale in _liquidity_filter_grids():
            base_value = _optional_float(seed.get(key))
            if base_value is None:
                base_value = _optional_float(merged.get(key)) or _filter_default(key)
            for value in _near_grid_values(base_value, values, limit=3):
                _append_refinement(
                    buckets["filters"],
                    30.0 + _distance(value, base_value, scale=scale) + seed_index * 0.05,
                    f"ref{seed_index}_{seed_name}_{label}{_numeric_label(value)}",
                    {**seed, key: value},
                )

        anchor = {
            **seed,
            "kalcb.research.top_long_count": base_top,
            "kalcb.research.min_sector_participation": base_sector,
            "kalcb.frontier.active_selection_mode": base_active,
        }
        for family, weights in _refinement_weight_families():
            _append_refinement(
                buckets["weights"],
                40.0 + seed_index * 0.05,
                f"ref{seed_index}_{seed_name}_{family}",
                {**anchor, **weights},
            )
            for mode in active_values[:2]:
                _append_refinement(
                    buckets["weights"],
                    42.0 + (0.0 if mode == base_active else 0.2) + seed_index * 0.05,
                    f"ref{seed_index}_{seed_name}_{family}_{mode}",
                    {**anchor, **weights, "kalcb.frontier.active_selection_mode": mode},
                )

    return _select_refinement_experiments(buckets, existing=existing, limit=limit)


@dataclass(frozen=True, slots=True)
class CandidateOpportunityResult:
    metrics: dict[str, float]
    dates: tuple[date, ...]
    source_fingerprint: str
    candidate_snapshot_hash: str


@dataclass(frozen=True, slots=True)
class CandidateOpportunity:
    symbol: str
    trade_date: date
    rank: int
    active: bool
    valid: bool
    raw_signal: bool
    signal_type: str
    mfe_r: float
    mae_r: float
    intraday_high_from_open_pct: float
    close_from_open_pct: float
    signal_rvol: float
    signal_close_location: float


def score_research_metrics(metrics: dict[str, float]) -> tuple[float, str]:
    valid_days = float(metrics.get("valid_candidate_days", 0.0) or 0.0)
    active_valid_days = float(metrics.get("active_valid_candidate_days", 0.0) or 0.0)
    min_valid = 1.0 if float(metrics.get("snapshot_count", 0.0) or 0.0) <= 2.0 else 50.0
    if valid_days < min_valid:
        return 0.0, f"too_few_candidate_days ({valid_days:.0f} < {min_valid:.0f})"
    if active_valid_days <= 0.0:
        return 0.0, "no_active_candidate_days"
    if float(metrics.get("raw_signal_count", 0.0) or 0.0) <= 0.0:
        return 0.0, "no_raw_or_pdh_opportunities"

    active_signal_rate = _clip(float(metrics.get("active_raw_signal_rate", 0.0)))
    active_good_05 = _clip(float(metrics.get("active_mfe_ge_0_5_per_valid", 0.0)))
    active_good_10 = _clip(float(metrics.get("active_mfe_ge_1_0_per_valid", 0.0)))
    active_avg_mfe = _clip(float(metrics.get("active_avg_signal_mfe_r", 0.0)) / 1.50)
    active_median_mfe = _clip(float(metrics.get("active_median_signal_mfe_r", 0.0)) / 1.00)
    active_good_days = _clip(float(metrics.get("active_days_with_good_signal_share", 0.0)))
    pool_good_05 = _clip(float(metrics.get("mfe_ge_0_5_per_valid", 0.0)))
    pool_good_10 = _clip(float(metrics.get("mfe_ge_1_0_per_valid", 0.0)))
    pool_signal_rate = _clip(float(metrics.get("raw_signal_rate", 0.0)))
    pool_avg_mfe = _clip(float(metrics.get("avg_signal_mfe_r", 0.0)) / 1.50)
    active_capture = _clip(float(metrics.get("active_good_signal_capture_share", 0.0)))
    good_per_day = _clip(float(metrics.get("avg_good_signals_per_day", 0.0)) / 3.0)

    score = (
        0.20 * active_good_05
        + 0.12 * active_good_10
        + 0.14 * active_signal_rate
        + 0.13 * active_avg_mfe
        + 0.09 * active_median_mfe
        + 0.08 * active_good_days
        + 0.08 * pool_good_05
        + 0.05 * pool_good_10
        + 0.04 * pool_signal_rate
        + 0.03 * pool_avg_mfe
        + 0.02 * active_capture
        + 0.02 * good_per_day
    )
    penalty = (
        0.05 * _clip(float(metrics.get("active_low_mfe_lt_0_3_signal_share", 0.0)))
        + 0.04 * _clip(float(metrics.get("active_bad_mae_le_neg_1_0_signal_share", 0.0)))
        + 0.03 * _clip(float(metrics.get("bad_mae_le_neg_1_0_signal_share", 0.0)))
    )
    return max(0.0, 100.0 * (score - penalty)), ""


def _evaluate_candidates(
    candidates: list[Experiment],
    *,
    training_config: dict[str, Any],
    base_mutations: dict[str, Any],
    output_dir: Path,
    folds: list[tuple[date, date]],
    max_workers: int,
    stage: str,
    progress_path: Path | None = None,
) -> list[dict[str, Any]]:
    if max_workers <= 1:
        rows = []
        for index, candidate in enumerate(candidates, start=1):
            row = _evaluate_candidate(candidate, training_config, base_mutations, output_dir, folds, stage=stage)
            rows.append(row)
            _append_candidate_progress(progress_path, row, stage=stage, completed=index, total=len(candidates))
        return rows
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_evaluate_candidate, candidate, training_config, base_mutations, output_dir, folds, None, stage): candidate.name
            for candidate in candidates
        }
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            completed += 1
            _append_candidate_progress(progress_path, row, stage=stage, completed=completed, total=len(candidates))
    return rows


def _evaluate_candidate(
    candidate: Experiment,
    training_config: dict[str, Any],
    base_mutations: dict[str, Any],
    output_dir: Path,
    folds: list[tuple[date, date]],
    full_result: CandidateOpportunityResult | None = None,
    stage: str = "coarse",
) -> dict[str, Any]:
    _validate_research_mutations(candidate)
    merged = dict(base_mutations)
    merged.update(candidate.mutations)
    if full_result is None:
        full_result = _run_candidate_opportunity(training_config, merged, candidate, output_dir, "full")
    full_metrics = dict(full_result.metrics)
    full_score, reject_reason = score_research_metrics(full_metrics)
    fold_rows = []
    for index, (start, end) in enumerate(folds, start=1):
        fold_config = _fold_config(training_config, start, end)
        fold_result = _run_candidate_opportunity(fold_config, merged, candidate, output_dir, f"fold{index}")
        fold_metrics = dict(fold_result.metrics)
        fold_score, fold_reject = score_research_metrics(fold_metrics)
        fold_rows.append(
            {
                "fold": index,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "score": fold_score,
                "rejected": bool(fold_reject),
                "reject_reason": fold_reject,
                "metrics": _compact_metrics(fold_metrics),
            }
        )
    fold_scores = [float(row["score"]) for row in fold_rows]
    median_fold = float(median(fold_scores)) if fold_scores else full_score
    worst_fold = float(min(fold_scores)) if fold_scores else full_score
    stability_score = 0.50 * full_score + 0.35 * median_fold + 0.15 * worst_fold
    fold_reject_count = sum(1 for row in fold_rows if row["rejected"])
    rejected = bool(reject_reason) or (len(fold_rows) >= 2 and fold_reject_count > len(fold_rows) // 2)
    if not reject_reason and rejected:
        reject_reason = "unstable_across_folds"
    return {
        "name": candidate.name,
        "stage": stage,
        "score": round(0.0 if rejected else stability_score, 6),
        "full_score": round(full_score, 6),
        "median_fold_score": round(median_fold, 6),
        "worst_fold_score": round(worst_fold, 6),
        "rejected": rejected,
        "reject_reason": reject_reason,
        "mutations": dict(candidate.mutations),
        "merged_mutations": merged,
        "source_fingerprint": full_result.source_fingerprint,
        "candidate_snapshot_hash": full_result.candidate_snapshot_hash,
        "metrics": _compact_metrics(full_metrics),
        "folds": fold_rows,
    }


def _run_candidate_opportunity(
    config: dict[str, Any],
    mutations: dict[str, Any],
    candidate: Experiment,
    output_dir: Path,
    scope: str,
) -> CandidateOpportunityResult:
    scoped_config = dict(config)
    scoped_config["artifact_root"] = str(output_dir / "_candidate_artifacts" / _safe_name(candidate.name) / scope)
    cfg = KALCBConfig.from_mapping(scoped_config, mutations)
    bundle = _load_candidate_replay_bundle(scoped_config, mutations)
    snapshots = _snapshots_for_bundle(bundle)
    metrics = candidate_opportunity_metrics(bundle, snapshots, cfg)
    return CandidateOpportunityResult(
        metrics=metrics,
        dates=tuple(sorted(snapshots)),
        source_fingerprint=bundle.source_fingerprint,
        candidate_snapshot_hash=str((bundle.metadata or {}).get("kalcb_candidate_artifact_hash") or _snapshot_set_hash(snapshots)),
    )


def _load_candidate_replay_bundle(config: dict[str, Any], mutations: dict[str, Any]) -> EventReplayBundle:
    capability_level = str(config.get("capability_level", "real_replay")).lower()
    if capability_level == "synthetic":
        return make_synthetic_replay_bundle("kalcb", config)
    return load_kalcb_real_replay_bundle(config, mutations)


def _snapshots_for_bundle(bundle: EventReplayBundle) -> dict[date, KALCBDailySnapshot]:
    snapshots = snapshots_from_bundle(bundle)
    if snapshots:
        return snapshots
    bars = [event.bar for event in bundle.events if event.bar is not None]
    if str((bundle.metadata or {}).get("capability_level", "")).lower() == "synthetic" or any(bar.source == "synthetic" for bar in bars):
        snapshot = _synthetic_snapshot(bars, bundle.source_fingerprint)
        return {snapshot.trade_date: snapshot}
    raise ValueError("KALCB research sweep requires source-fingerprinted candidate snapshots")


def candidate_opportunity_metrics(
    bundle: EventReplayBundle,
    snapshots: dict[date, KALCBDailySnapshot],
    config: KALCBConfig,
) -> dict[str, float]:
    bars_by_key = _bars_by_session_symbol(bundle)
    rows: list[CandidateOpportunity] = []
    candidate_counts: list[int] = []
    active_counts: list[int] = []
    for session, snapshot in sorted(snapshots.items()):
        candidates = [candidate for candidate in snapshot.candidates if candidate.tradable]
        active_symbols = _active_symbols(snapshot, config)
        candidate_counts.append(len(candidates))
        active_counts.append(len(active_symbols & {candidate.symbol for candidate in candidates}))
        for rank, candidate in enumerate(candidates, start=1):
            bars = bars_by_key.get((session, candidate.symbol), ())
            rows.append(_candidate_day_opportunity(candidate, bars, rank=rank, active=candidate.symbol in active_symbols, config=config))

    valid = [row for row in rows if row.valid]
    active_valid = [row for row in valid if row.active]
    signals = [row for row in valid if row.raw_signal]
    active_signals = [row for row in signals if row.active]
    good_05 = [row for row in signals if row.mfe_r >= 0.5]
    good_10 = [row for row in signals if row.mfe_r >= 1.0]
    active_good_05 = [row for row in active_signals if row.mfe_r >= 0.5]
    active_good_10 = [row for row in active_signals if row.mfe_r >= 1.0]
    days = sorted(snapshots)
    day_count = len(days)
    signal_days = {row.trade_date for row in signals}
    good_signal_days = {row.trade_date for row in good_05}
    active_signal_days = {row.trade_date for row in active_signals}
    active_good_signal_days = {row.trade_date for row in active_good_05}

    metrics: dict[str, float] = {
        "snapshot_count": float(day_count),
        "selected_candidate_days": float(len(rows)),
        "valid_candidate_days": float(len(valid)),
        "active_candidate_days": float(sum(1 for row in rows if row.active)),
        "active_valid_candidate_days": float(len(active_valid)),
        "candidate_pool_max": float(max(candidate_counts, default=0)),
        "candidate_pool_avg": float(mean(candidate_counts)) if candidate_counts else 0.0,
        "active_symbol_max": float(max(active_counts, default=0)),
        "active_symbol_avg": float(mean(active_counts)) if active_counts else 0.0,
        "raw_signal_count": float(len(signals)),
        "active_raw_signal_count": float(len(active_signals)),
        "raw_signal_rate": _ratio(len(signals), len(valid)),
        "active_raw_signal_rate": _ratio(len(active_signals), len(active_valid)),
        "days_with_signal_share": _ratio(len(signal_days), day_count),
        "days_with_good_signal_share": _ratio(len(good_signal_days), day_count),
        "active_days_with_signal_share": _ratio(len(active_signal_days), day_count),
        "active_days_with_good_signal_share": _ratio(len(active_good_signal_days), day_count),
        "avg_signals_per_day": _ratio(len(signals), day_count),
        "avg_good_signals_per_day": _ratio(len(good_05), day_count),
        "avg_active_signals_per_day": _ratio(len(active_signals), day_count),
        "avg_active_good_signals_per_day": _ratio(len(active_good_05), day_count),
        "avg_signal_mfe_r": _avg(row.mfe_r for row in signals),
        "median_signal_mfe_r": _med(row.mfe_r for row in signals),
        "avg_signal_mae_r": _avg(row.mae_r for row in signals),
        "active_avg_signal_mfe_r": _avg(row.mfe_r for row in active_signals),
        "active_median_signal_mfe_r": _med(row.mfe_r for row in active_signals),
        "active_avg_signal_mae_r": _avg(row.mae_r for row in active_signals),
        "mfe_ge_0_5_signal_share": _ratio(len(good_05), len(signals)),
        "mfe_ge_1_0_signal_share": _ratio(len(good_10), len(signals)),
        "mfe_ge_0_5_per_valid": _ratio(len(good_05), len(valid)),
        "mfe_ge_1_0_per_valid": _ratio(len(good_10), len(valid)),
        "active_mfe_ge_0_5_signal_share": _ratio(len(active_good_05), len(active_signals)),
        "active_mfe_ge_1_0_signal_share": _ratio(len(active_good_10), len(active_signals)),
        "active_mfe_ge_0_5_per_valid": _ratio(len(active_good_05), len(active_valid)),
        "active_mfe_ge_1_0_per_valid": _ratio(len(active_good_10), len(active_valid)),
        "low_mfe_lt_0_3_signal_share": _ratio(sum(1 for row in signals if row.mfe_r < 0.3), len(signals)),
        "active_low_mfe_lt_0_3_signal_share": _ratio(sum(1 for row in active_signals if row.mfe_r < 0.3), len(active_signals)),
        "bad_mae_le_neg_1_0_signal_share": _ratio(sum(1 for row in signals if row.mae_r <= -1.0), len(signals)),
        "active_bad_mae_le_neg_1_0_signal_share": _ratio(sum(1 for row in active_signals if row.mae_r <= -1.0), len(active_signals)),
        "avg_intraday_high_from_open_pct": _avg(row.intraday_high_from_open_pct for row in valid),
        "active_avg_intraday_high_from_open_pct": _avg(row.intraday_high_from_open_pct for row in active_valid),
        "avg_close_from_open_pct": _avg(row.close_from_open_pct for row in valid),
        "active_avg_close_from_open_pct": _avg(row.close_from_open_pct for row in active_valid),
        "avg_signal_rvol": _avg(row.signal_rvol for row in signals),
        "active_avg_signal_rvol": _avg(row.signal_rvol for row in active_signals),
        "avg_signal_close_location": _avg(row.signal_close_location for row in signals),
        "active_avg_signal_close_location": _avg(row.signal_close_location for row in active_signals),
        "avg_signal_rank": _avg(row.rank for row in signals),
        "avg_good_signal_rank": _avg(row.rank for row in good_05),
        "active_good_signal_capture_share": _ratio(len(active_good_05), len(good_05)),
        "symbol_count_with_signal": float(len({row.symbol for row in signals})),
        "active_symbol_count_with_signal": float(len({row.symbol for row in active_signals})),
    }
    return metrics


def _bars_by_session_symbol(bundle: EventReplayBundle) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    grouped: dict[tuple[date, str], list[MarketBar]] = {}
    for event in bundle.events:
        bar = event.bar
        if bar is None:
            continue
        grouped.setdefault((bar.timestamp.date(), bar.symbol), []).append(bar)
    return {
        key: tuple(sorted(values, key=lambda bar: bar.timestamp))
        for key, values in grouped.items()
    }


def _active_symbols(snapshot: KALCBDailySnapshot, config: KALCBConfig) -> set[str]:
    by_symbol = snapshot.by_symbol()
    raw = snapshot.metadata.get("active_symbols") or []
    symbols = [str(symbol) for symbol in raw if str(symbol) in by_symbol]
    if not symbols:
        symbols = [candidate.symbol for candidate in snapshot.candidates[: max(1, int(config.ws_budget))]]
    return set(_dedupe(symbols)[: max(1, int(config.ws_budget))])


def _candidate_day_opportunity(
    candidate: KALCBDailyCandidate,
    bars: tuple[MarketBar, ...],
    *,
    rank: int,
    active: bool,
    config: KALCBConfig,
) -> CandidateOpportunity:
    if len(bars) <= config.opening_range_bars:
        return _empty_opportunity(candidate, rank=rank, active=active)
    opening_price = max(float(bars[0].open), 1e-9)
    intraday_high_pct = max(float(bar.high) for bar in bars) / opening_price - 1.0
    close_pct = float(bars[-1].close) / opening_price - 1.0
    or_high, or_low, _ = compute_opening_range(list(bars), config.opening_range_bars)
    if or_high <= 0 or or_low <= 0:
        return _empty_opportunity(candidate, rank=rank, active=active, valid=False, intraday_high_pct=intraday_high_pct, close_pct=close_pct)

    post_or = list(bars[config.opening_range_bars :])
    signal_index = -1
    signal_type = ""
    for offset, bar in enumerate(post_or, start=config.opening_range_bars):
        entry_type = classify_raw_breakout(bar, prior_day_high=candidate.prior_day_high, or_high=or_high)
        if entry_type is not None:
            signal_index = offset
            signal_type = entry_type.value
            break
    if signal_index < 0:
        return CandidateOpportunity(
            symbol=candidate.symbol,
            trade_date=candidate.trade_date,
            rank=rank,
            active=active,
            valid=True,
            raw_signal=False,
            signal_type="",
            mfe_r=0.0,
            mae_r=0.0,
            intraday_high_from_open_pct=intraday_high_pct,
            close_from_open_pct=close_pct,
            signal_rvol=0.0,
            signal_close_location=0.0,
        )

    signal_bar = bars[signal_index]
    entry_price = float(signal_bar.close)
    risk = _raw_opportunity_risk(entry_price, or_low, float(signal_bar.low), float(candidate.daily_atr), config)
    future = bars[signal_index:]
    mfe_r = max(0.0, (max(float(bar.high) for bar in future) - entry_price) / risk) if risk > 0 else 0.0
    mae_r = (min(float(bar.low) for bar in future) - entry_price) / risk if risk > 0 else 0.0
    expected_volume = float(candidate.expected_5m_volume or candidate.average_30m_volume / 6.0 or 0.0)
    return CandidateOpportunity(
        symbol=candidate.symbol,
        trade_date=candidate.trade_date,
        rank=rank,
        active=active,
        valid=True,
        raw_signal=True,
        signal_type=signal_type,
        mfe_r=float(mfe_r),
        mae_r=float(mae_r),
        intraday_high_from_open_pct=intraday_high_pct,
        close_from_open_pct=close_pct,
        signal_rvol=compute_bar_rvol(signal_bar.volume, expected_volume),
        signal_close_location=close_location_value(signal_bar),
    )


def _raw_opportunity_risk(entry_price: float, or_low: float, signal_low: float, daily_atr: float, config: KALCBConfig) -> float:
    structural_stop = min(float(or_low), float(signal_low)) if or_low > 0 else float(signal_low)
    atr_stop = float(entry_price) - config.stop_atr_multiple * max(float(daily_atr), 0.0)
    stop = max(structural_stop, atr_stop)
    if stop >= entry_price:
        stop = float(entry_price) * 0.985
    return max(float(entry_price) - stop, float(entry_price) * 0.001)


def _empty_opportunity(
    candidate: KALCBDailyCandidate,
    *,
    rank: int,
    active: bool,
    valid: bool = False,
    intraday_high_pct: float = 0.0,
    close_pct: float = 0.0,
) -> CandidateOpportunity:
    return CandidateOpportunity(
        symbol=candidate.symbol,
        trade_date=candidate.trade_date,
        rank=rank,
        active=active,
        valid=valid,
        raw_signal=False,
        signal_type="",
        mfe_r=0.0,
        mae_r=0.0,
        intraday_high_from_open_pct=float(intraday_high_pct),
        close_from_open_pct=float(close_pct),
        signal_rvol=0.0,
        signal_close_location=0.0,
    )


def _training_config(config: dict[str, Any], holdout_days: int) -> dict[str, Any]:
    out = dict(config)
    out["holdout_days"] = int(holdout_days)
    out.pop("end", None)
    out.pop("holdout_start", None)
    date_range = dict(out.get("date_range") or {})
    date_range.pop("end", None)
    if date_range:
        out["date_range"] = date_range
    else:
        out.pop("date_range", None)
    return out


def _fold_config(config: dict[str, Any], start: date, end: date) -> dict[str, Any]:
    out = dict(config)
    out["start"] = start.isoformat()
    out["end"] = end.isoformat()
    out["use_full_available_window"] = False
    return out


def _base_mutations(config: dict[str, Any], mutations: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(BASE_MUTATIONS)
    if isinstance(config.get("initial_mutations"), dict):
        out.update(config["initial_mutations"])
    out.update(dict(mutations or {}))
    return out


def _resolve_folds(dates: list[date], *, fold_days: int | None, fold_count: int) -> list[tuple[date, date]]:
    if fold_days is not None and int(fold_days) > 0:
        return _fold_ranges_by_days(dates, int(fold_days))
    return _fold_ranges_by_count(dates, int(fold_count))


def _fold_ranges_by_count(dates: list[date], fold_count: int) -> list[tuple[date, date]]:
    if fold_count <= 0 or not dates:
        return []
    ordered = sorted(set(dates))
    if fold_count == 1:
        return [(ordered[0], ordered[-1])]
    folds: list[tuple[date, date]] = []
    total = len(ordered)
    for index in range(fold_count):
        start_idx = round(index * total / fold_count)
        end_idx = round((index + 1) * total / fold_count) - 1
        if start_idx >= total:
            continue
        end_idx = min(max(end_idx, start_idx), total - 1)
        folds.append((ordered[start_idx], ordered[end_idx]))
    return folds


def _fold_ranges_by_days(dates: list[date], fold_days: int) -> list[tuple[date, date]]:
    if not dates:
        return []
    start = min(dates)
    latest = max(dates)
    folds: list[tuple[date, date]] = []
    while start <= latest:
        end = min(start + timedelta(days=max(1, int(fold_days)) - 1), latest)
        members = [day for day in dates if start <= day <= end]
        if members:
            folds.append((members[0], members[-1]))
        start = end + timedelta(days=1)
    return folds or [(dates[0], dates[-1])]


def _result_dates(result: CandidateOpportunityResult) -> list[date]:
    return sorted(set(result.dates))


def _phase_auto_seed(training_config: dict[str, Any], base_mutations: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    initial = dict(base_mutations)
    initial.update(dict(row.get("mutations") or {}))
    return {
        **dict(training_config),
        "initial_mutations": initial,
        "research_sweep_seed": {
            "candidate": row.get("name"),
            "score": row.get("score"),
            "metrics": row.get("metrics"),
            "policy": "Use this research candidate-list config as the later phased-auto seed; holdout remains excluded until final validation.",
        },
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    keep = (
        "snapshot_count",
        "selected_candidate_days",
        "valid_candidate_days",
        "active_candidate_days",
        "active_valid_candidate_days",
        "candidate_pool_max",
        "candidate_pool_avg",
        "active_symbol_max",
        "active_symbol_avg",
        "raw_signal_count",
        "active_raw_signal_count",
        "raw_signal_rate",
        "active_raw_signal_rate",
        "days_with_signal_share",
        "days_with_good_signal_share",
        "active_days_with_signal_share",
        "active_days_with_good_signal_share",
        "avg_signals_per_day",
        "avg_good_signals_per_day",
        "avg_active_signals_per_day",
        "avg_active_good_signals_per_day",
        "avg_signal_mfe_r",
        "median_signal_mfe_r",
        "avg_signal_mae_r",
        "active_avg_signal_mfe_r",
        "active_median_signal_mfe_r",
        "active_avg_signal_mae_r",
        "mfe_ge_0_5_signal_share",
        "mfe_ge_1_0_signal_share",
        "mfe_ge_0_5_per_valid",
        "mfe_ge_1_0_per_valid",
        "active_mfe_ge_0_5_signal_share",
        "active_mfe_ge_1_0_signal_share",
        "active_mfe_ge_0_5_per_valid",
        "active_mfe_ge_1_0_per_valid",
        "low_mfe_lt_0_3_signal_share",
        "active_low_mfe_lt_0_3_signal_share",
        "bad_mae_le_neg_1_0_signal_share",
        "active_bad_mae_le_neg_1_0_signal_share",
        "avg_intraday_high_from_open_pct",
        "active_avg_intraday_high_from_open_pct",
        "avg_close_from_open_pct",
        "active_avg_close_from_open_pct",
        "avg_signal_rvol",
        "active_avg_signal_rvol",
        "avg_signal_close_location",
        "active_avg_signal_close_location",
        "avg_signal_rank",
        "avg_good_signal_rank",
        "active_good_signal_capture_share",
        "symbol_count_with_signal",
        "active_symbol_count_with_signal",
    )
    return {key: _float(metrics.get(key)) for key in keep if key in metrics}


def _weights(*, rs: float, trend: float, compression: float, accumulation: float, stock: float, sector: float, participation: float) -> dict[str, float]:
    return {
        "kalcb.research.weights.relative_strength": rs,
        "kalcb.research.weights.daily_trend": trend,
        "kalcb.research.weights.compression": compression,
        "kalcb.research.weights.accumulation": accumulation,
        "kalcb.research.weights.stock_regime": stock,
        "kalcb.research.weights.sector_regime": sector,
        "kalcb.research.weights.sector_participation": participation,
    }


def _append_refinement(specs: list[tuple[float, str, dict[str, Any]]], priority: float, name: str, mutations: dict[str, Any]) -> None:
    clean = _research_only_mutations(mutations)
    if clean:
        specs.append((float(priority), name, clean))


def _append_candidate_progress(
    progress_path: Path | None,
    row: dict[str, Any],
    *,
    stage: str,
    completed: int,
    total: int,
) -> None:
    _append_progress(
        progress_path,
        {
            "event": "candidate_completed",
            "stage": stage,
            "completed": int(completed),
            "total": int(total),
            "name": row.get("name"),
            "score": row.get("score"),
            "rejected": row.get("rejected"),
            "reject_reason": row.get("reject_reason"),
            "metrics": {
                key: (row.get("metrics") or {}).get(key)
                for key in (
                    "valid_candidate_days",
                    "raw_signal_count",
                    "active_raw_signal_count",
                    "active_mfe_ge_0_5_per_valid",
                    "active_avg_signal_mfe_r",
                )
            },
        },
    )


def _append_progress(progress_path: Path | None, payload: dict[str, Any]) -> None:
    if progress_path is None:
        return
    payload = {"ts": _utc_now_iso(), **payload}
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _select_refinement_experiments(
    buckets: dict[str, list[tuple[float, str, dict[str, Any]]]],
    *,
    existing: set[str],
    limit: int | None,
) -> list[Experiment]:
    for specs in buckets.values():
        specs.sort(key=lambda item: (item[0], item[1]))
    if limit is None:
        quotas = {key: len(value) for key, value in buckets.items()}
    else:
        quotas = {
            "core": max(1, int(limit * 0.45)),
            "gates": max(1, int(limit * 0.25)),
            "filters": max(1, int(limit * 0.15)),
            "weights": max(1, int(limit * 0.15)),
        }
        quotas["core"] += max(0, limit - sum(quotas.values()))
    selected: list[Experiment] = []
    seen = set(existing)

    def add_from(specs: list[tuple[float, str, dict[str, Any]]], quota: int | None) -> None:
        nonlocal selected
        added = 0
        for _, name, mutations in specs:
            if limit is not None and len(selected) >= limit:
                return
            if quota is not None and added >= quota:
                return
            signature = _mutation_signature(mutations)
            if signature in seen:
                continue
            seen.add(signature)
            selected.append(Experiment(name, mutations))
            added += 1

    for bucket in ("core", "gates", "filters", "weights"):
        add_from(buckets.get(bucket, []), None if limit is None else quotas.get(bucket, 0))
    leftovers = [item for specs in buckets.values() for item in specs]
    leftovers.sort(key=lambda item: (item[0], item[1]))
    add_from(leftovers, None)
    return selected


def _research_only_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(mutations or {}).items() if key in RESEARCH_MUTATION_KEYS}


def _soft_gate_grids() -> tuple[tuple[str, tuple[float, ...], str, float], ...]:
    return (
        ("kalcb.research.min_rs_percentile", (0.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0), "rs", 25.0),
        ("kalcb.research.min_trend_score", (0.0, 35.0, 45.0, 55.0, 65.0, 70.0, 75.0), "trend", 30.0),
        ("kalcb.research.min_compression_score", (0.0, 10.0, 20.0, 30.0, 40.0, 50.0), "comp", 30.0),
        ("kalcb.research.min_accumulation_score", (-1.0, -0.20, 0.0, 0.10, 0.20, 0.35), "acc", 0.60),
        ("kalcb.research.min_sector_daily_score_pct", (0.0, 45.0, 50.0, 55.0, 60.0, 65.0), "secd", 30.0),
        ("kalcb.research.max_box_range_pct", (0.0, 0.08, 0.10, 0.12, 0.15, 0.20), "box", 0.08),
    )


def _liquidity_filter_grids() -> tuple[tuple[str, tuple[float, ...], str, float], ...]:
    return (
        ("kalcb.research.min_adv20_krw", (1_000_000_000.0, 1_500_000_000.0, 2_000_000_000.0, 3_000_000_000.0, 5_000_000_000.0, 7_500_000_000.0), "adv", 2_000_000_000.0),
        ("kalcb.research.min_price_krw", (500.0, 1_000.0, 2_000.0, 3_000.0, 5_000.0, 10_000.0), "px", 3_000.0),
        ("kalcb.research.min_history_days", (40.0, 50.0, 60.0, 80.0, 100.0), "hist", 30.0),
    )


def _refinement_weight_families() -> tuple[tuple[str, dict[str, float]], ...]:
    return (
        ("sector_rs_weights", _weights(rs=0.28, trend=0.18, compression=0.10, accumulation=0.10, stock=0.07, sector=0.09, participation=0.18)),
        ("sector_trend_weights", _weights(rs=0.22, trend=0.24, compression=0.10, accumulation=0.10, stock=0.08, sector=0.10, participation=0.16)),
        ("sector_compression_weights", _weights(rs=0.20, trend=0.18, compression=0.20, accumulation=0.14, stock=0.06, sector=0.08, participation=0.14)),
        ("accum_participation_weights", _weights(rs=0.20, trend=0.16, compression=0.12, accumulation=0.22, stock=0.06, sector=0.08, participation=0.16)),
        ("rs_trend_clean_weights", _weights(rs=0.34, trend=0.26, compression=0.08, accumulation=0.10, stock=0.08, sector=0.06, participation=0.08)),
    )


def _ordered_active_modes(base: str) -> tuple[str, ...]:
    ordered = [str(base or "liquidity").lower(), "score", "hybrid", "liquidity", "campaign", "hot"]
    out: list[str] = []
    for mode in ordered:
        normalized = "score" if mode in {"research_score", "selection_score"} else mode
        if normalized not in {"score", "hybrid", "liquidity", "campaign", "hot"}:
            continue
        if normalized not in out:
            out.append(normalized)
    return tuple(out)


def _near_grid_values(base: float, grid: tuple[float, ...], *, limit: int) -> tuple[float, ...]:
    ordered = sorted(set(float(value) for value in grid), key=lambda value: (abs(float(value) - float(base)), value))
    return tuple(_clean_grid_value(value) for value in ordered[: max(1, int(limit))])


def _clean_grid_value(value: float) -> float | int:
    if abs(float(value) - round(float(value))) < 1e-9:
        return int(round(float(value)))
    return round(float(value), 6)


def _distance(value: float, base: float, *, scale: float) -> float:
    return abs(float(value) - float(base)) / max(float(scale), 1e-9)


def _filter_default(key: str) -> float:
    return {
        "kalcb.research.min_adv20_krw": 2_000_000_000.0,
        "kalcb.research.min_price_krw": 1_000.0,
        "kalcb.research.min_history_days": 60.0,
    }.get(key, 0.0)


def _pct_label(value: Any) -> str:
    return str(int(round(float(value) * 100.0)))


def _numeric_label(value: Any) -> str:
    number = float(value)
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}b".replace(".", "p")
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    text = f"{number:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _mutation_signature(mutations: dict[str, Any]) -> str:
    return json.dumps(_research_only_mutations(mutations), sort_keys=True, separators=(",", ":"), default=str)


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_research_mutations(candidate: Experiment) -> None:
    invalid = sorted(set(candidate.mutations) - RESEARCH_MUTATION_KEYS)
    if invalid:
        raise ValueError(f"KALCB research sweep candidate {candidate.name!r} contains non-research mutations: {invalid}")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else 0.0


def _avg(values: Any) -> float:
    items = [float(value) for value in values]
    return float(mean(items)) if items else 0.0


def _med(values: Any) -> float:
    items = [float(value) for value in values]
    return float(median(items)) if items else 0.0


def _snapshot_set_hash(snapshots: dict[date, KALCBDailySnapshot]) -> str:
    return stable_signature({day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshots.items())})


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:80] or "candidate"


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB Research Candidate Opportunity Sweep",
        "",
        f"Sweep hash: `{payload['sweep_hash']}`",
        f"Opportunity model: `{payload['opportunity_model_version']}`",
        f"Holdout days excluded: {payload['holdout_contract']['holdout_days']}",
        f"Candidates tested: {payload['candidate_count']}",
        f"Folds: {len(payload['folds'])}",
        "",
        payload["evaluation_policy"],
        "",
        "## Top Frontier",
        "",
        "| Rank | Candidate | Score | Full | Median Fold | Worst Fold | Candidate Days | Raw Signals | Active Signals | Active +0.5R/Valid | Active Avg MFE R | Reject |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(payload["rows"][: max(10, int(payload["selected_count"]))], start=1):
        metrics = row.get("metrics") or {}
        lines.append(
            "| "
            f"{rank} | {row['name']} | {_fmt(row['score'])} | {_fmt(row['full_score'])} | "
            f"{_fmt(row['median_fold_score'])} | {_fmt(row['worst_fold_score'])} | "
            f"{_fmt(metrics.get('valid_candidate_days'), 0)} | {_fmt(metrics.get('raw_signal_count'), 0)} | "
            f"{_fmt(metrics.get('active_raw_signal_count'), 0)} | {_pct(metrics.get('active_mfe_ge_0_5_per_valid'))} | "
            f"{_fmt(metrics.get('active_avg_signal_mfe_r'))} | "
            f"{row.get('reject_reason') or ''} |"
        )
    if payload.get("selection_frontier"):
        best = payload["selection_frontier"][0]
        lines.extend(
            [
                "",
                "## Phased Auto Seed",
                "",
                f"Selected seed: `{best['name']}`",
                "",
                "Use the generated `phase_auto_seed` config as the research-list seed for the later KALCB entry, exit, sizing, and trade-management optimiser.",
            ]
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{_float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return f"{_float(value):.1%}"
