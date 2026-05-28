from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.config import load_yaml_config, normalize_runtime_config
from strategy_olr.config import OLRConfig, OLR_CORE_VERSION
from strategy_olr.execution import (
    EXECUTION_CORE_VERSION,
    OLREntryPlan,
    OLRExitPlan,
    OLRTradeOutcome,
    round_trip_cost_pct,
    simulate_olr_trade,
    summarize_olr_outcomes,
    olr_outcome_hash,
)
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.research import afternoon_selection_from_contexts

from .research_sweep import (
    DEFAULT_EXPECTED_UNIVERSE_SIZE,
    DEFAULT_HOLDOUT_DAYS,
    OLRResearchSweepDataset,
    _resolve_folds,
    _training_config,
    afternoon_contexts_for_snapshots,
    prepare_research_sweep_dataset,
    research_snapshots_for_dataset,
    snapshots_for_experiment,
)


TRADE_PLAN_SWEEP_VERSION = "olr-fixed-top3-execution-sweep-v2"
DEFAULT_OUTPUT_DIR = Path("data/backtests/output/olr/trade_plan_sweeps")


@dataclass(frozen=True, slots=True)
class CandidateSource:
    rank: int
    name: str
    stage1_name: str
    stage2_name: str
    score: float
    mutations: dict[str, Any]
    artifact_hash: str = ""
    stage1_mutations: dict[str, Any] = field(default_factory=dict)
    stage2_mutations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FixedSelection:
    trade_date: date
    symbol: str
    candidate_source_name: str
    candidate: OLRDailyCandidate


@dataclass(frozen=True, slots=True)
class OLRTradePlanSpec:
    name: str
    candidate_source_name: str
    entry: OLREntryPlan
    exit: OLRExitPlan


@dataclass(frozen=True, slots=True)
class PlanResult:
    spec: OLRTradePlanSpec
    score: float
    rejected: bool
    reject_reason: str
    train_metrics: dict[str, float]
    fold_metrics: tuple[dict[str, Any], ...]
    promotion_pass: bool = False
    replay_digest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompiledExecutionSet:
    dataset: OLRResearchSweepDataset
    sources: tuple[CandidateSource, ...]
    snapshots_by_source: dict[str, dict[date, OLRDailySnapshot]]
    selections_by_source: dict[str, tuple[FixedSelection, ...]]
    selection_counts_by_source: dict[str, dict[date, int]]
    eligible_dates: tuple[date, ...]
    next_session_by_date: dict[date, date]
    source_fingerprint: str
    candidate_artifact_hash: str
    fast_cache_enabled: bool


def run_trade_plan_sweep(
    config: dict[str, Any] | None = None,
    *,
    research_sweep_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    top_source_count: int = 3,
    fold_count: int = 2,
    coarse_entry_limit: int = 320,
    coarse_exit_limit: int = 220,
    entry_seed_top_n: int = 10,
    deep_refine_top_n: int = 12,
    deep_refine_max_specs: int = 3_000,
    finalist_count: int = 40,
    audit_finalist_count: int = 5,
    max_workers: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clear_progress_files(out)
    research_path = Path(research_sweep_path) if research_sweep_path else _latest_research_sweep_path()
    research_payload = json.loads(research_path.read_text(encoding="utf-8"))
    sources = load_candidate_sources(research_payload, top_n=top_source_count)
    entry_specs = _limit_specs(build_entry_plans(), coarse_entry_limit)
    exit_specs = _limit_specs(build_exit_plans(), coarse_exit_limit)
    if dry_run:
        payload = {
            "strategy": "olr",
            "dry_run": True,
            "research_sweep_path": str(research_path),
            "top_source_count": len(sources),
            "source_names": [source.name for source in sources],
            "entry_candidate_count": len(sources) * len(entry_specs),
            "max_exit_candidate_count": len(sources) * max(1, entry_seed_top_n) * len(exit_specs),
            "max_deep_refine_specs": int(deep_refine_max_specs),
            "max_workers": max(1, min(int(max_workers), 2)),
            "official_performance": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return payload

    _write_run_status(out, "preparing_fixed_candidates", top_source_count=len(sources), max_workers=max(1, min(int(max_workers), 2)))
    training_config = _training_config(dict(config or {}), holdout_days)
    compiled = build_compiled_execution_set(training_config, research_payload, sources, holdout_days=holdout_days, use_fast_cache=True)
    cfg = OLRConfig.from_mapping(compiled.dataset.config, {})
    folds = _resolve_folds(list(compiled.eligible_dates), fold_days=None, fold_count=fold_count)
    baseline_specs = [_name_plan(OLRTradePlanSpec("", source.name, _close_auction_entry(), _next_close_exit())) for source in compiled.sources]
    baseline_rows = [
        _evaluate_spec(spec, compiled, cfg, folds, {}, ())
        for spec in baseline_specs
    ]
    baseline_rows = [replace(row, promotion_pass=False) for row in baseline_rows]
    baseline_by_source = {row.spec.candidate_source_name: row.train_metrics for row in baseline_rows}
    baseline_fold_by_source = {row.spec.candidate_source_name: row.fold_metrics for row in baseline_rows}
    _write_run_status(
        out,
        "baseline_complete",
        sources=len(compiled.sources),
        best_baseline_equal_slot=round(100.0 * max((row.train_metrics.get("equal_slot_net_return_pct", row.train_metrics.get("slot_cumulative_net_return_pct", 0.0)) for row in baseline_rows), default=0.0), 3),
    )

    entry_exit = _next_close_exit()
    entry_plans = [
        _name_plan(OLRTradePlanSpec("", source.name, entry, entry_exit))
        for source in compiled.sources
        for entry in entry_specs
    ]
    _write_run_status(out, "entry_coarse", total=len(entry_plans))
    entry_rows = _evaluate_specs(entry_plans, compiled, cfg, folds, baseline_by_source, baseline_fold_by_source, out, "entry_coarse", max_workers=max_workers)
    entry_seed_rows = _select_entry_seed_rows(entry_rows, entry_seed_top_n)

    exit_plans = [
        _name_plan(OLRTradePlanSpec("", seed.spec.candidate_source_name, seed.spec.entry, exit_spec))
        for seed in entry_seed_rows
        for exit_spec in exit_specs
    ]
    _write_run_status(out, "exit_management", total=len(exit_plans), entry_seeds=len(entry_seed_rows), exit_specs=len(exit_specs))
    exit_rows = _evaluate_specs(exit_plans, compiled, cfg, folds, baseline_by_source, baseline_fold_by_source, out, "exit_management", max_workers=max_workers)

    broad_rows = _merge_results(baseline_rows, entry_rows, exit_rows)
    broad_rows.sort(key=_plan_sort_key)
    refine_specs = build_refined_trade_plan_specs(
        broad_rows[: max(0, int(deep_refine_top_n))],
        exclude_names={row.spec.name for row in broad_rows},
        max_specs=deep_refine_max_specs,
    )
    refine_rows: list[PlanResult] = []
    if refine_specs:
        _write_run_status(out, "deep_refine", total=len(refine_specs))
        refine_rows = _evaluate_specs(refine_specs, compiled, cfg, folds, baseline_by_source, baseline_fold_by_source, out, "deep_refine", max_workers=max_workers)
    rows = _merge_results(broad_rows, refine_rows)
    rows.sort(key=_plan_sort_key)
    finalists = rows[: max(1, int(finalist_count))]
    promotion_rows = [row for row in finalists if row.promotion_pass]
    audit_targets = (promotion_rows or finalists)[: max(1, int(audit_finalist_count))]
    _write_run_status(out, "full_audit_finalists", total=len(audit_targets), max_workers=max(1, min(int(max_workers), 2)))
    audit_rows = _audit_rows(
        audit_targets,
        dict(config or {}),
        research_payload,
        sources,
        holdout_days,
        folds,
        baseline_by_source,
        baseline_fold_by_source,
        max_workers=max_workers,
    )
    audit_pass = all(bool(row.get("audit_pass")) for row in audit_rows) if audit_rows else True
    payload = {
        "strategy": "olr",
        "sweep_version": TRADE_PLAN_SWEEP_VERSION,
        "strategy_core_version": OLR_CORE_VERSION,
        "execution_core_version": EXECUTION_CORE_VERSION,
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "official_performance": False,
        "research_sweep_path": str(research_path),
        "training_window": {
            "start": compiled.eligible_dates[0].isoformat() if compiled.eligible_dates else "",
            "end": compiled.eligible_dates[-1].isoformat() if compiled.eligible_dates else "",
            "sessions": len(compiled.eligible_dates),
        },
        "holdout_policy": {
            "holdout_days": int(holdout_days),
            "train_only": True,
            "next_session_exit_policy": "A trade_date is eligible only when the next intraday session is also inside the training window.",
            "selection_uses_holdout": False,
            "execution_uses_holdout": False,
        },
        "implementation_lessons_contract": _implementation_lessons_contract(),
        "causality_policy": {
            "candidate_generation": "Stage 1 daily rows and lagged flow use row_date < trade_date; Stage 2 selector uses timestamp < 14:30 KST.",
            "entry": "Post-14:30 signals from completed 5m bar t fill no earlier than bar t+1 open through the last continuous 5m bar; close_auction is a resting order submitted after the 14:30 decision for the configured close-auction print.",
            "exit": "Protective/target orders are modeled as resting orders; discretionary exits from completed bars fill next bar open.",
            "same_bar_signal_fill": False,
            "official_performance": False,
        },
        "timing_policy": {
            "stage2_selector_cutoff": "timestamp < 14:30 KST",
            "last_continuous_5m_bar": "15:15 KST",
            "default_close_auction_fill_time": str(cfg.auction_fill_time),
            "non_auction_entry_fill_policy": "next bar open, capped before the close auction",
            "close_auction_entry_fill_policy": "configured close-auction print",
            "metric_basis": "training execution research via strategy_olr.execution.simulate_olr_trade; not official SimBroker MTM",
        },
        "fast_replay_policy": {
            "enabled": True,
            "mode": "compiled_fixed_candidate_execution_replay",
            "suppressed_work": "research snapshot rebuilds, afternoon context construction, and fixed candidate selection are cached; trade decisions use strategy_olr.execution in both fast and audit modes",
            "selection_functions": [
                "strategy_olr.research.daily_selection_from_snapshot",
                "strategy_olr.research.afternoon_selection_from_contexts",
            ],
            "execution_function": "strategy_olr.execution.simulate_olr_trade",
            "full_audit_finalist_count": len(audit_rows),
            "audit_metric_tolerance": 1e-10,
        },
        "candidate_sources": [_source_payload(source) for source in compiled.sources],
        "source_fingerprints": {
            "research_sweep_hash": str(research_payload.get("sweep_hash") or ""),
            "research_sweep_file": _file_hash(research_path),
            "compiled_execution": compiled.source_fingerprint,
            "candidate_artifacts": compiled.candidate_artifact_hash,
            "daily_intraday": compiled.dataset.source_fingerprint,
        },
        "selection_stats": _selection_stats(compiled),
        "cost_policy": {
            "round_trip_cost_pct": round_trip_cost_pct(cfg),
            "slippage_bps_each_side": cfg.slippage_bps,
            "commission_bps_each_side": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
        },
        "sweep_counts": {
            "entry_coarse": len(entry_rows),
            "exit_management": len(exit_rows),
            "deep_refine": len(refine_rows),
            "candidate_count": len(rows),
            "entry_seed_count": len(entry_seed_rows),
            "promoted_count": len(promotion_rows),
        },
        "baseline": [_row_payload(row) for row in baseline_rows],
        "entry_seeds": [_row_payload(row) for row in entry_seed_rows],
        "promotion_policy": {
            "slot_net_relative_improvement_min": 0.15,
            "metric_contract": {
                "slot_cumulative_net_return_pct": "legacy costed equal-slot execution metric; not a broker-account portfolio return",
                "equal_slot_net_return_pct": "alias of slot_cumulative_net_return_pct for clearer reporting",
                "selected_day_net_pct": "average costed equal-selected-candidate return on selected days",
                "official_performance": "false; OLR trade-plan sweep is execution/research until rerun through a portfolio/broker-equivalent path",
            },
            "selected_day_net_must_improve": True,
            "mfe_capture_must_improve_or_slot_strong": True,
            "worst_fold_slot_must_be_non_negative": True,
            "result": "promote" if promotion_rows else "no_promotion",
        },
        "audit_replays": audit_rows,
        "fast_full_audit": _audit_summary(audit_rows),
        "audit_pass": audit_pass,
        "top_train": [_row_payload(row) for row in rows[:100]],
        "top_promoted": [_row_payload(row) for row in promotion_rows[:25]],
        "rows": [_row_payload(row) for row in rows],
    }
    payload["sweep_hash"] = stable_signature(
        {
            "version": TRADE_PLAN_SWEEP_VERSION,
            "sources": [_source_payload(source) for source in compiled.sources],
            "training_window": payload["training_window"],
            "top_train": payload["top_train"][:10],
            "audit_pass": audit_pass,
        }
    )
    json_path = out / f"olr_trade_plan_sweep_{payload['sweep_hash'][:12]}.json"
    md_path = out / f"olr_trade_plan_sweep_{payload['sweep_hash'][:12]}.md"
    seed_path = out / f"olr_trade_plan_seed_{payload['sweep_hash'][:12]}.json"
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path), "phase_auto_seed": str(seed_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    seed_path.write_text(json.dumps(_phase_seed_payload(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_progress(out, "completed", len(rows), len(rows), rows)
    _write_run_status(out, "completed", sweep_hash=payload["sweep_hash"], json=str(json_path), markdown=str(md_path), audit_pass=audit_pass)
    return payload


def load_candidate_sources(payload: dict[str, Any], *, top_n: int = 3) -> tuple[CandidateSource, ...]:
    base = dict(payload.get("base_mutations") or {})
    default_stage1 = dict((payload.get("selected_stage1_seed") or {}).get("mutations") or {})
    default_stage1_name = str((payload.get("selected_stage1_seed") or {}).get("name") or "stage1")
    rows = list(payload.get("stage2_frontier") or payload.get("selection_frontier") or [])[: max(1, int(top_n))]
    if not rows:
        raise ValueError("OLR trade sweep requires at least one Stage 2 candidate source")
    sources: list[CandidateSource] = []
    for index, row in enumerate(rows, start=1):
        row_stage1 = dict(row.get("stage1_seed") or {})
        stage1 = dict(row_stage1.get("mutations") or default_stage1)
        stage1_name = str(row_stage1.get("name") or default_stage1_name)
        stage2 = dict(row.get("mutations") or {})
        mutations = dict(base)
        mutations.update(stage1)
        mutations.update(stage2)
        stage2_name = str(row.get("name") or f"stage2_{index}")
        sources.append(
            CandidateSource(
                rank=index,
                name=_safe_name(f"src{index}_{stage2_name}"),
                stage1_name=stage1_name,
                stage2_name=stage2_name,
                score=float(row.get("score", 0.0) or 0.0),
                mutations=mutations,
                artifact_hash=str(row.get("artifact_hash") or ""),
                stage1_mutations=stage1,
                stage2_mutations=stage2,
            )
        )
    return tuple(sources)


def build_compiled_execution_set(
    config: dict[str, Any],
    research_payload: dict[str, Any],
    sources: Sequence[CandidateSource],
    *,
    holdout_days: int,
    use_fast_cache: bool,
    include_holdout: bool = False,
) -> CompiledExecutionSet:
    dataset = prepare_research_sweep_dataset(
        config,
        holdout_days=holdout_days,
        expected_universe_size=DEFAULT_EXPECTED_UNIVERSE_SIZE,
        include_holdout=include_holdout,
    )
    base = dict(research_payload.get("base_mutations") or {})
    research_cache = research_snapshots_for_dataset(dataset, base) if use_fast_cache else None
    default_stage1 = dict((research_payload.get("selected_stage1_seed") or {}).get("mutations") or {})
    stage1_cache: dict[str, tuple[dict[date, OLRDailySnapshot], dict[date, dict[str, Any]]]] = {}
    eligible_dates, next_by_date = _eligible_execution_dates(dataset)
    snapshots_by_source: dict[str, dict[date, OLRDailySnapshot]] = {}
    selections_by_source: dict[str, tuple[FixedSelection, ...]] = {}
    counts_by_source: dict[str, dict[date, int]] = {}
    source_payloads = []
    for source in sources:
        source_stage1 = dict(source.stage1_mutations or default_stage1)
        stage1_mutations = dict(base)
        stage1_mutations.update(source_stage1)
        stage1_key = stable_signature(stage1_mutations)
        if stage1_key not in stage1_cache:
            stage1_snapshots = snapshots_for_experiment(dataset, stage1_mutations, research_snapshots=research_cache)
            stage1_cfg = OLRConfig.from_mapping(dataset.config, stage1_mutations)
            contexts = afternoon_contexts_for_snapshots(dataset, stage1_snapshots, stage1_cfg)
            stage1_cache[stage1_key] = (stage1_snapshots, contexts)
        stage1_snapshots, contexts = stage1_cache[stage1_key]
        cfg = OLRConfig.from_mapping(dataset.config, source.mutations)
        selected_by_day: dict[date, OLRDailySnapshot] = {}
        selections: list[FixedSelection] = []
        counts = {day: 0 for day in eligible_dates}
        for day in eligible_dates:
            base_snapshot = stage1_snapshots.get(day)
            if base_snapshot is None:
                continue
            selected = afternoon_selection_from_contexts(base_snapshot, contexts.get(day, {}), cfg)
            selected_by_day[day] = selected
            for candidate in selected.candidates[: max(1, int(cfg.overnight_slot_count))]:
                next_day = next_by_date.get(day)
                if next_day is None:
                    continue
                if not dataset.bars_by_key.get((day, candidate.symbol)) or not dataset.bars_by_key.get((next_day, candidate.symbol)):
                    continue
                selections.append(FixedSelection(day, candidate.symbol, source.name, candidate))
                counts[day] = counts.get(day, 0) + 1
        snapshots_by_source[source.name] = selected_by_day
        selections_by_source[source.name] = tuple(sorted(selections, key=lambda item: (item.trade_date, item.symbol)))
        counts_by_source[source.name] = counts
        source_payloads.append({"name": source.name, "hash": _snapshot_hash(selected_by_day), "count": len(selections)})
    candidate_hash = stable_signature(source_payloads)
    return CompiledExecutionSet(
        dataset=dataset,
        sources=tuple(replace(source, artifact_hash=source.artifact_hash or _snapshot_hash(snapshots_by_source.get(source.name, {}))) for source in sources),
        snapshots_by_source=snapshots_by_source,
        selections_by_source=selections_by_source,
        selection_counts_by_source=counts_by_source,
        eligible_dates=eligible_dates,
        next_session_by_date=next_by_date,
        source_fingerprint=stable_signature([dataset.source_fingerprint, candidate_hash, use_fast_cache, include_holdout]),
        candidate_artifact_hash=candidate_hash,
        fast_cache_enabled=bool(use_fast_cache),
    )


def build_entry_plans() -> list[OLREntryPlan]:
    specs: list[OLREntryPlan] = [_close_auction_entry(), OLREntryPlan("", "decision_next_open")]
    for bars in (1, 2, 4, 6, 8):
        for min_ret in (-0.001, 0.0, 0.001, 0.002, 0.004):
            for min_vwap in (-0.002, -0.001, 0.0, 0.001):
                for close_loc in (0.0, 0.50, 0.60, 0.70):
                    specs.append(OLREntryPlan("", "confirm_next_bar", max_signal_bars=bars, min_bar_ret=min_ret, min_vwap_ret=min_vwap, min_close_location=close_loc))
                    specs.append(OLREntryPlan("", "late_continuation", max_signal_bars=bars, min_bar_ret=min_ret, min_vwap_ret=min_vwap, min_close_location=close_loc, min_breakout_pct=0.0))
    for mode in ("momentum_breakout", "decision_high_breakout", "pdh_breakout"):
        for bars in (2, 4, 8, 12):
            for breakout in (0.0, 0.0005, 0.001, 0.002, 0.004):
                for close_loc in (0.50, 0.60, 0.70):
                    specs.append(OLREntryPlan("", mode, max_signal_bars=bars, min_breakout_pct=breakout, min_close_location=close_loc))
    for mode in ("vwap_reclaim", "pullback_acceptance"):
        for bars in (3, 6, 10, 14):
            for pullback in (0.002, 0.004, 0.008, 0.012):
                for reclaim in (-0.001, 0.0, 0.001, 0.002):
                    specs.append(OLREntryPlan("", mode, max_signal_bars=bars, max_pullback_from_vwap_pct=pullback, min_reclaim_ret=reclaim, min_vwap_ret=reclaim, min_close_location=0.50))
    for after in (1, 2, 4):
        for bars in (4, 8, 12):
            specs.append(OLREntryPlan("", "late_continuation", max_signal_bars=bars, after_bar=after, min_breakout_pct=0.001, min_vwap_ret=0.0, min_close_location=0.50))
    return _dedupe_entries(_name_entry(spec) for spec in specs)


def build_exit_plans() -> list[OLRExitPlan]:
    specs: list[OLRExitPlan] = [_next_close_exit(), _name_exit(OLRExitPlan("", mode="next_open", hard_stop_enabled=False))]
    stop_modes = ("atr", "fixed_pct", "decision_low", "signal_low", "vwap")
    for stop_mode in stop_modes:
        specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True))
        for target in (0.35, 0.50, 0.75, 1.00, 1.50, 2.00):
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, target_r=target))
        for trigger, fraction, stop_r in ((0.25, 0.50, -0.10), (0.35, 0.50, 0.0), (0.50, 0.50, 0.10), (0.75, 0.33, 0.20), (1.00, 0.33, 0.35)):
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, partial_trigger_r=trigger, partial_fraction=fraction, partial_stop_r=stop_r))
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, partial_trigger_r=trigger, partial_fraction=fraction, partial_stop_r=stop_r, target_r=1.50))
        for trigger, stop_r in ((0.25, 0.0), (0.35, 0.0), (0.50, 0.10), (0.75, 0.20), (1.00, 0.35)):
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, breakeven_trigger_r=trigger, breakeven_stop_r=stop_r, target_r=1.50))
        for start, gap in ((0.35, 0.30), (0.50, 0.35), (0.75, 0.45), (1.00, 0.60), (1.25, 0.75)):
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, trail_start_r=start, trail_gap_r=gap))
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, partial_trigger_r=0.50, partial_fraction=0.50, partial_stop_r=0.0, trail_start_r=start, trail_gap_r=gap))
        for bars in (3, 4, 6, 8, 12, 18, 24):
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, max_hold_bars=bars))
        for bars in (2, 3, 4):
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, vwap_fail_bars=bars, vwap_fail_pct=0.001, target_r=1.0))
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, no_mfe_bars=bars, no_mfe_thresh_r=0.15, target_r=1.0))
            specs.append(OLRExitPlan("", mode="managed", stop_mode=stop_mode, hard_stop_enabled=True, failed_followthrough_bars=bars, failed_followthrough_mfe_r=0.20, failed_followthrough_close_r=0.0))
    return _dedupe_exits(_name_exit(spec) for spec in specs)


def build_refined_trade_plan_specs(
    seed_rows: Sequence[PlanResult],
    *,
    exclude_names: set[str] | None = None,
    max_specs: int,
) -> list[OLRTradePlanSpec]:
    excluded = set(exclude_names or set())
    out: dict[str, OLRTradePlanSpec] = {}
    for row in seed_rows:
        for entry in _refined_entries(row.spec.entry):
            for exit_spec in _refined_exits(row.spec.exit):
                spec = _name_plan(OLRTradePlanSpec("", row.spec.candidate_source_name, entry, exit_spec))
                if spec.name in excluded or spec.name in out:
                    continue
                reason = _validate_spec(spec)
                if reason:
                    continue
                out[spec.name] = spec
                if len(out) >= max(0, int(max_specs)):
                    return list(out.values())
    return list(out.values())


def _evaluate_specs(
    specs: Sequence[OLRTradePlanSpec],
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    folds: list[tuple[date, date]],
    baseline_by_source: dict[str, dict[str, float]],
    baseline_fold_by_source: dict[str, tuple[dict[str, Any], ...]],
    output_dir: Path,
    stage: str,
    *,
    max_workers: int,
) -> list[PlanResult]:
    rows: list[PlanResult] = []
    worker_count = max(1, min(int(max_workers), 2))
    if worker_count <= 1 or len(specs) <= 1:
        for spec in specs:
            row = _evaluate_spec(spec, compiled, cfg, folds, baseline_by_source, baseline_fold_by_source.get(spec.candidate_source_name, ()))
            rows.append(row)
            _record_progress(output_dir, stage, len(rows), len(specs), rows, row)
        return rows
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_evaluate_spec, spec, compiled, cfg, folds, baseline_by_source, baseline_fold_by_source.get(spec.candidate_source_name, ())): spec
            for spec in specs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            _record_progress(output_dir, stage, len(rows), len(specs), rows, row)
    return rows


def _evaluate_spec(
    spec: OLRTradePlanSpec,
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    folds: list[tuple[date, date]],
    baseline_by_source: dict[str, dict[str, float]],
    baseline_folds: tuple[dict[str, Any], ...],
) -> PlanResult:
    reject = _validate_spec(spec)
    outcomes: list[OLRTradeOutcome] = []
    metrics: dict[str, float] = {}
    fold_rows: tuple[dict[str, Any], ...] = ()
    digest: dict[str, Any] = {}
    if not reject:
        outcomes = collect_outcomes(spec, compiled, cfg, compiled.eligible_dates)
        counts = compiled.selection_counts_by_source.get(spec.candidate_source_name, {})
        metrics = summarize_olr_outcomes(outcomes, session_dates=compiled.eligible_dates, selection_counts=counts, slot_count=cfg.overnight_slot_count)
        fold_rows = _fold_metrics_from_outcomes(outcomes, compiled.eligible_dates, folds, counts, cfg.overnight_slot_count)
        digest = {"outcome_hash": olr_outcome_hash(outcomes), "trade_count": len(outcomes), "metric_hash": stable_signature(metrics)}
        score, reject = _score_plan(metrics, fold_rows, baseline_by_source.get(spec.candidate_source_name, {}))
    else:
        score = 0.0
        metrics = {"trade_count": 0.0, "selected_count": 0.0}
    promoted = (not reject) and _promotion_pass(metrics, fold_rows, baseline_by_source.get(spec.candidate_source_name, {}), baseline_folds)
    return PlanResult(
        spec=spec,
        score=round(0.0 if reject else score, 6),
        rejected=bool(reject),
        reject_reason=str(reject or ""),
        train_metrics=metrics,
        fold_metrics=fold_rows,
        promotion_pass=promoted,
        replay_digest=digest,
    )


def collect_outcomes(
    spec: OLRTradePlanSpec,
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    dates: Iterable[date],
) -> list[OLRTradeOutcome]:
    date_set = set(dates)
    outcomes: list[OLRTradeOutcome] = []
    for selection in compiled.selections_by_source.get(spec.candidate_source_name, ()):
        if selection.trade_date not in date_set:
            continue
        next_day = compiled.next_session_by_date.get(selection.trade_date)
        if next_day is None:
            continue
        entry_bars = compiled.dataset.bars_by_key.get((selection.trade_date, selection.symbol), ())
        next_bars = compiled.dataset.bars_by_key.get((next_day, selection.symbol), ())
        outcome = simulate_olr_trade(selection.trade_date, selection.symbol, entry_bars, next_bars, selection.candidate, spec.entry, spec.exit, cfg)
        if outcome is not None:
            outcomes.append(outcome)
    return outcomes


def _score_plan(metrics: dict[str, float], fold_rows: tuple[dict[str, Any], ...], baseline: dict[str, float]) -> tuple[float, str]:
    selected = float(metrics.get("selected_count", 0.0))
    trades = float(metrics.get("trade_count", 0.0))
    if selected <= 0:
        return 0.0, "no_selected_candidates"
    if trades < 50:
        return 0.0, f"too_few_trades ({trades:.0f} < 50)"
    if float(metrics.get("signal_conversion", 0.0)) < 0.12:
        return 0.0, f"too_low_conversion ({metrics.get('signal_conversion', 0.0):.3f} < 0.120)"
    slot = float(metrics.get("slot_cumulative_net_return_pct", 0.0))
    selected_net = float(metrics.get("selected_day_net_pct", 0.0))
    active_net = float(metrics.get("active_day_net_pct", 0.0))
    fold_slots = [float(row["metrics"].get("slot_cumulative_net_return_pct", 0.0)) for row in fold_rows]
    worst_fold = min(fold_slots) if fold_slots else slot
    median_fold = median(fold_slots) if fold_slots else slot
    baseline_slot = float(baseline.get("slot_cumulative_net_return_pct", 0.0))
    dd = abs(float(metrics.get("max_drawdown_net_pct", 0.0)))
    capture = float(metrics.get("avg_mfe_capture", 0.0))
    score = (
        145.0 * slot
        + 85.0 * selected_net
        + 35.0 * active_net
        + 35.0 * worst_fold
        + 20.0 * median_fold
        + 12.0 * capture
        + 8.0 * float(metrics.get("net_win_share", 0.0))
        + 4.0 * float(metrics.get("mfe_ge_1_share", 0.0))
        - 70.0 * dd
        - 10.0 * float(metrics.get("mae_le_neg_1_share", 0.0))
        - 5.0 * float(metrics.get("ambiguous_bar_count", 0.0)) / max(trades, 1.0)
    )
    if slot < baseline_slot:
        score -= 60.0 * min(1.0, (baseline_slot - slot) / max(abs(baseline_slot), 1e-9))
    return score, ""


def _promotion_pass(
    metrics: dict[str, float],
    fold_rows: tuple[dict[str, Any], ...],
    baseline: dict[str, float],
    baseline_folds: tuple[dict[str, Any], ...],
) -> bool:
    baseline_slot = float(baseline.get("slot_cumulative_net_return_pct", 0.0))
    slot = float(metrics.get("slot_cumulative_net_return_pct", 0.0))
    if baseline_slot > 0.0 and slot < baseline_slot * 1.15:
        return False
    if baseline_slot <= 0.0 and slot < baseline_slot + 0.10:
        return False
    if float(metrics.get("selected_day_net_pct", 0.0)) <= float(baseline.get("selected_day_net_pct", 0.0)):
        return False
    fold_slots = [float(row["metrics"].get("slot_cumulative_net_return_pct", 0.0)) for row in fold_rows]
    if fold_slots and min(fold_slots) < 0.0:
        return False
    baseline_fold_slots = [float(row["metrics"].get("slot_cumulative_net_return_pct", 0.0)) for row in baseline_folds]
    if fold_slots and baseline_fold_slots and median(fold_slots) < median(baseline_fold_slots):
        return False
    if float(metrics.get("avg_mfe_capture", 0.0)) < float(baseline.get("avg_mfe_capture", 0.0)) and slot < baseline_slot * 1.35:
        return False
    return True


def _audit_rows(
    rows: Sequence[PlanResult],
    config: dict[str, Any],
    research_payload: dict[str, Any],
    sources: Sequence[CandidateSource],
    holdout_days: int,
    folds: list[tuple[date, date]],
    baseline_by_source: dict[str, dict[str, float]],
    baseline_fold_by_source: dict[str, tuple[dict[str, Any], ...]],
    *,
    max_workers: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    full_compiled = build_compiled_execution_set(_training_config(dict(config), holdout_days), research_payload, sources, holdout_days=holdout_days, use_fast_cache=False)
    cfg = OLRConfig.from_mapping(full_compiled.dataset.config, {})
    worker_count = max(1, min(int(max_workers), 2))
    if worker_count <= 1 or len(rows) == 1:
        return [_audit_row(row, full_compiled, cfg, folds, baseline_by_source, baseline_fold_by_source) for row in rows]
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_audit_row, row, full_compiled, cfg, folds, baseline_by_source, baseline_fold_by_source): row
            for row in rows
        }
        for future in as_completed(futures):
            out.append(future.result())
    out.sort(key=lambda item: item.get("name", ""))
    return out


def _audit_row(
    fast_row: PlanResult,
    full_compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    folds: list[tuple[date, date]],
    baseline_by_source: dict[str, dict[str, float]],
    baseline_fold_by_source: dict[str, tuple[dict[str, Any], ...]],
) -> dict[str, Any]:
    full_row = _evaluate_spec(
        fast_row.spec,
        full_compiled,
        cfg,
        folds,
        baseline_by_source,
        baseline_fold_by_source.get(fast_row.spec.candidate_source_name, ()),
    )
    metric_keys = sorted(set(fast_row.train_metrics) | set(full_row.train_metrics))
    metric_deltas = {key: float(full_row.train_metrics.get(key, 0.0) or 0.0) - float(fast_row.train_metrics.get(key, 0.0) or 0.0) for key in metric_keys}
    max_abs_metric_delta = max((abs(value) for value in metric_deltas.values()), default=0.0)
    score_delta = float(full_row.score) - float(fast_row.score)
    digest_match = fast_row.replay_digest.get("outcome_hash") == full_row.replay_digest.get("outcome_hash")
    reject_match = fast_row.rejected == full_row.rejected and fast_row.reject_reason == full_row.reject_reason
    audit_pass = digest_match and reject_match and max_abs_metric_delta <= 1e-10 and abs(score_delta) <= 1e-10
    return {
        "name": fast_row.spec.name,
        "audit_pass": audit_pass,
        "outcome_hash_match": digest_match,
        "rejection_match": reject_match,
        "max_abs_metric_delta": max_abs_metric_delta,
        "score_delta": score_delta,
        "metric_deltas": metric_deltas,
        "fast_score": fast_row.score,
        "full_score": full_row.score,
        "fast_outcome_hash": fast_row.replay_digest.get("outcome_hash"),
        "full_outcome_hash": full_row.replay_digest.get("outcome_hash"),
        "scope": "fast mode suppresses candidate rebuild/context construction only; strategy_olr.execution decisions and fills must match full rebuild",
    }


def _eligible_execution_dates(dataset: OLRResearchSweepDataset) -> tuple[tuple[date, ...], dict[date, date]]:
    dates = tuple(dataset.trading_dates)
    next_by_date = {day: dates[index + 1] for index, day in enumerate(dates[:-1])}
    eligible = tuple(day for day in dates[:-1] if next_by_date.get(day) in set(dates))
    return eligible, next_by_date


def _fold_metrics_from_outcomes(
    outcomes: Sequence[OLRTradeOutcome],
    session_dates: Sequence[date],
    folds: list[tuple[date, date]],
    selection_counts: dict[date, int],
    slot_count: int,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for index, (start, end) in enumerate(folds, start=1):
        dates = tuple(day for day in session_dates if start <= day <= end)
        rows.append({"fold": index, "start": start.isoformat(), "end": end.isoformat(), "metrics": summarize_olr_outcomes(outcomes, session_dates=dates, selection_counts=selection_counts, slot_count=slot_count)})
    return tuple(rows)


def _close_auction_entry() -> OLREntryPlan:
    return _name_entry(OLREntryPlan("", "close_auction"))


def _next_close_exit() -> OLRExitPlan:
    return _name_exit(OLRExitPlan("", mode="next_close", hard_stop_enabled=False))


def _name_entry(plan: OLREntryPlan) -> OLREntryPlan:
    if plan.name:
        return plan
    parts = [
        plan.mode,
        f"b{plan.max_signal_bars}",
        f"a{plan.after_bar}",
        f"ret{_label(plan.min_bar_ret)}",
        f"vw{_label(plan.min_vwap_ret)}",
        f"bo{_label(plan.min_breakout_pct)}",
        f"cl{_label(plan.min_close_location)}",
        f"pb{_label(plan.max_pullback_from_vwap_pct)}",
        f"rec{_label(plan.min_reclaim_ret)}",
        f"dc{int(plan.require_above_decision_close)}",
    ]
    return replace(plan, name="_".join(parts))


def _name_exit(plan: OLRExitPlan) -> OLRExitPlan:
    if plan.name:
        return plan
    parts = [
        plan.mode,
        plan.stop_mode,
        f"hs{int(plan.hard_stop_enabled)}",
        f"atr{_label(plan.stop_atr_mult)}",
        f"sp{_label(plan.stop_pct)}",
        f"t{_label(plan.target_r)}",
        f"p{_label(plan.partial_trigger_r)}x{_label(plan.partial_fraction)}",
        f"ps{_label(plan.partial_stop_r)}",
        f"be{_label(plan.breakeven_trigger_r)}",
        f"tr{_label(plan.trail_start_r)}x{_label(plan.trail_gap_r)}",
        f"mf{_label(plan.mfe_fade_start_r)}x{_label(plan.mfe_fade_gap_r)}f{_label(plan.mfe_fade_floor_r)}",
        f"vf{plan.vwap_fail_bars}",
        f"ff{plan.failed_followthrough_bars}",
        f"nm{plan.no_mfe_bars}",
        f"mh{plan.max_hold_bars}",
    ]
    return replace(plan, name="_".join(parts))


def _name_plan(spec: OLRTradePlanSpec) -> OLRTradePlanSpec:
    if spec.name:
        return spec
    return replace(spec, name=f"{spec.candidate_source_name}__{spec.entry.name}__{spec.exit.name}")


def _validate_spec(spec: OLRTradePlanSpec) -> str:
    exit_plan = spec.exit
    if exit_plan.mode in {"next_close", "next_open"}:
        return ""
    if exit_plan.mode != "managed":
        return f"unknown_exit_mode_{exit_plan.mode}"
    if exit_plan.partial_trigger_r > 0.0 and exit_plan.partial_fraction > 0.0:
        if not 0.0 < exit_plan.partial_fraction < 1.0:
            return "partial_fraction_out_of_range"
        if exit_plan.partial_stop_r > exit_plan.partial_trigger_r:
            return "partial_stop_above_trigger"
    if exit_plan.trail_start_r > 0.0 and exit_plan.trail_gap_r <= 0.0:
        return "trail_gap_missing"
    if spec.entry.mode == "close_auction" and exit_plan.max_hold_bars == 1:
        return "close_auction_max_hold_1_redundant"
    return ""


def _select_entry_seed_rows(rows: Sequence[PlanResult], count: int) -> list[PlanResult]:
    target = max(1, int(count))
    ranked = sorted(rows, key=_plan_sort_key)
    selected: list[PlanResult] = []
    seen: set[str] = set()
    for row in ranked:
        key = f"{row.spec.candidate_source_name}:{_entry_key(row.spec.entry)}"
        if key in seen:
            continue
        if row.rejected and len(selected) >= max(1, target // 2):
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= target:
            break
    return selected


def _refined_entries(entry: OLREntryPlan) -> list[OLREntryPlan]:
    if entry.mode in {"close_auction", "decision_next_open"}:
        return [entry]
    specs: list[OLREntryPlan] = [entry]
    bars_values = _near_int(entry.max_signal_bars, (1, 2, 4), 1, 16)
    close_values = _near_float(entry.min_close_location, (0.0, 0.50, 0.60, 0.70), 0.05, 0.0, 0.95)
    if entry.mode in {"confirm_next_bar", "late_continuation"}:
        for bars in bars_values:
            for ret in _near_float(entry.min_bar_ret, (-0.001, 0.0, 0.001, 0.002), 0.001, -0.005, 0.01):
                for vw in _near_float(entry.min_vwap_ret, (-0.002, 0.0, 0.001), 0.001, -0.005, 0.01):
                    for cl in close_values:
                        specs.append(_name_entry(replace(entry, max_signal_bars=bars, min_bar_ret=ret, min_vwap_ret=vw, min_close_location=cl)))
        if entry.mode == "late_continuation":
            for bo in _near_float(entry.min_breakout_pct, (0.0, 0.001, 0.002), 0.0005, 0.0, 0.01):
                specs.append(_name_entry(replace(entry, min_breakout_pct=bo)))
    elif entry.mode in {"momentum_breakout", "decision_high_breakout", "pdh_breakout"}:
        for bars in bars_values:
            for bo in _near_float(entry.min_breakout_pct, (0.0, 0.001, 0.002), 0.0005, 0.0, 0.01):
                for cl in close_values:
                    specs.append(_name_entry(replace(entry, max_signal_bars=bars, min_breakout_pct=bo, min_close_location=cl)))
    elif entry.mode in {"vwap_reclaim", "pullback_acceptance"}:
        for bars in bars_values:
            for pb in _near_float(entry.max_pullback_from_vwap_pct, (0.004, 0.008, 0.012), 0.002, 0.0, 0.03):
                for rec in _near_float(entry.min_reclaim_ret, (-0.001, 0.0, 0.001), 0.001, -0.005, 0.01):
                    for cl in close_values:
                        specs.append(_name_entry(replace(entry, max_signal_bars=bars, max_pullback_from_vwap_pct=pb, min_reclaim_ret=rec, min_vwap_ret=rec, min_close_location=cl)))
    return _dedupe_entries(specs)


def _refined_exits(exit_plan: OLRExitPlan) -> list[OLRExitPlan]:
    if exit_plan.mode in {"next_close", "next_open"}:
        return [exit_plan, _name_exit(OLRExitPlan("", mode="managed", stop_mode="atr", hard_stop_enabled=True, target_r=1.0))]
    specs: list[OLRExitPlan] = [exit_plan]
    for atr in _near_float(exit_plan.stop_atr_mult, (0.5, 0.8, 1.0, 1.25), 0.1, 0.2, 2.5):
        specs.append(_name_exit(replace(exit_plan, stop_atr_mult=atr)))
    for stop in _near_float(exit_plan.stop_pct, (0.005, 0.008, 0.012), 0.002, 0.002, 0.05):
        specs.append(_name_exit(replace(exit_plan, stop_pct=stop)))
    for target in _near_float(exit_plan.target_r, (0.0, 0.75, 1.0, 1.5, 2.0), 0.25, 0.0, 4.0):
        specs.append(_name_exit(replace(exit_plan, target_r=target)))
    for partial in _near_float(exit_plan.partial_trigger_r, (0.0, 0.35, 0.5, 0.75), 0.1, 0.0, 2.0):
        if partial <= 0:
            continue
        for fraction in (0.33, 0.50):
            for partial_stop in _near_float(exit_plan.partial_stop_r, (-0.1, 0.0, 0.1, 0.2), 0.1, -0.5, min(partial, 1.5)):
                specs.append(_name_exit(replace(exit_plan, partial_trigger_r=partial, partial_fraction=fraction, partial_stop_r=partial_stop)))
    for be in _near_float(exit_plan.breakeven_trigger_r, (0.0, 0.35, 0.5, 0.75), 0.1, 0.0, 2.0):
        if be > 0:
            for be_stop in _near_float(exit_plan.breakeven_stop_r, (0.0, 0.1, 0.2), 0.1, -0.2, min(be, 1.0)):
                specs.append(_name_exit(replace(exit_plan, breakeven_trigger_r=be, breakeven_stop_r=be_stop)))
    for trail in _near_float(exit_plan.trail_start_r, (0.0, 0.5, 0.75, 1.0), 0.1, 0.0, 3.0):
        if trail > 0:
            for gap in _near_float(exit_plan.trail_gap_r or 0.45, (0.3, 0.45, 0.6), 0.1, 0.05, 2.0):
                specs.append(_name_exit(replace(exit_plan, trail_start_r=trail, trail_gap_r=gap)))
    for fade in _near_float(exit_plan.mfe_fade_start_r, (0.0, 1.0, 1.5, 2.0, 2.5), 0.25, 0.0, 4.0):
        if fade > 0:
            for gap in _near_float(exit_plan.mfe_fade_gap_r or 0.75, (0.5, 0.75, 1.0, 1.25), 0.25, 0.05, 3.0):
                for floor in _near_float(exit_plan.mfe_fade_floor_r, (0.0, 0.25, 0.50), 0.25, -1.0, 2.0):
                    specs.append(_name_exit(replace(exit_plan, mfe_fade_start_r=fade, mfe_fade_gap_r=gap, mfe_fade_floor_r=floor)))
    for hold in _near_int(exit_plan.max_hold_bars, (2, 4, 8), 0, 36):
        if hold > 0:
            specs.append(_name_exit(replace(exit_plan, max_hold_bars=hold)))
    return _dedupe_exits(specs)


def _merge_results(*groups: Sequence[PlanResult]) -> list[PlanResult]:
    out: dict[str, PlanResult] = {}
    for group in groups:
        for row in group:
            current = out.get(row.spec.name)
            if current is None or _plan_sort_key(row) < _plan_sort_key(current):
                out[row.spec.name] = row
    return list(out.values())


def _plan_sort_key(row: PlanResult) -> tuple[Any, ...]:
    metrics = row.train_metrics
    worst_fold = min((float(fold["metrics"].get("slot_cumulative_net_return_pct", 0.0)) for fold in row.fold_metrics), default=metrics.get("slot_cumulative_net_return_pct", 0.0))
    return (
        row.rejected,
        -float(metrics.get("slot_cumulative_net_return_pct", 0.0)),
        -float(metrics.get("selected_day_net_pct", 0.0)),
        -worst_fold,
        abs(float(metrics.get("max_drawdown_net_pct", 0.0))),
        -float(metrics.get("avg_mfe_capture", 0.0)),
        row.spec.name,
    )


def _row_payload(row: PlanResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "score": row.score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "promotion_pass": row.promotion_pass,
        "spec": _spec_payload(row.spec),
        "train_metrics": row.train_metrics,
        "fold_metrics": list(row.fold_metrics),
        "fast_replay_digest": row.replay_digest,
    }


def _spec_payload(spec: OLRTradePlanSpec) -> dict[str, Any]:
    return {"name": spec.name, "candidate_source_name": spec.candidate_source_name, "entry": asdict(spec.entry), "exit": asdict(spec.exit)}


def _source_payload(source: CandidateSource) -> dict[str, Any]:
    return {
        "rank": source.rank,
        "name": source.name,
        "stage1_name": source.stage1_name,
        "stage2_name": source.stage2_name,
        "score": source.score,
        "mutations": source.mutations,
        "stage1_mutations": source.stage1_mutations,
        "stage2_mutations": source.stage2_mutations,
        "artifact_hash": source.artifact_hash,
    }


def _selection_stats(compiled: CompiledExecutionSet) -> dict[str, Any]:
    rows = {}
    for source in compiled.sources:
        counts = compiled.selection_counts_by_source.get(source.name, {})
        values = [counts.get(day, 0) for day in compiled.eligible_dates]
        rows[source.name] = {
            "selected_count": sum(values),
            "selected_days": sum(1 for value in values if value > 0),
            "avg_selected_per_session": sum(values) / max(float(len(values)), 1.0),
            "snapshot_hash": _snapshot_hash(compiled.snapshots_by_source.get(source.name, {})),
        }
    return rows


def _audit_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_run"}
    return {
        "status": "pass" if all(bool(row.get("audit_pass")) for row in rows) else "fail",
        "count": len(rows),
        "max_abs_metric_delta": max((float(row.get("max_abs_metric_delta", 0.0) or 0.0) for row in rows), default=0.0),
        "max_abs_score_delta": max((abs(float(row.get("score_delta", 0.0) or 0.0)) for row in rows), default=0.0),
        "outcome_hash_mismatches": sum(1 for row in rows if not row.get("outcome_hash_match")),
    }


def _implementation_lessons_contract() -> dict[str, Any]:
    return {
        "status": "shared_olr_execution_core_training_only",
        "shared_selection_api": {
            "stage1": "strategy_olr.research.daily_selection_from_snapshot",
            "stage2": "strategy_olr.research.afternoon_selection_from_contexts",
            "fixed_candidate_generation": "top 3 OLR research Stage 2 selectors are replayed without touching holdout dates",
        },
        "shared_execution_api": {
            "core": "strategy_olr.execution.simulate_olr_trade",
            "entry_policy": "completed-bar signal -> next continuous bar fill; explicitly modeled close_auction fills at the configured auction print",
            "exit_policy": "resting stops/targets, next-bar discretionary exits, next-session flatten variants",
        },
        "live_backtest_divergence_policy": "Live and replay may differ in data acquisition/OMS transport only; candidate snapshots and execution decisions are strategy_olr APIs.",
        "metric_policy": "Training-only execution evidence; no OOS or official performance claim until holdout/paper parity.",
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OLR Fixed-Candidate Execution Sweep",
        "",
        f"- Sweep hash: `{payload.get('sweep_hash')}`",
        f"- Train: {payload['training_window']['start']} to {payload['training_window']['end']} ({payload['training_window']['sessions']} sessions)",
        f"- Candidate sources: {', '.join(source['stage2_name'] for source in payload.get('candidate_sources', []))}",
        f"- Official performance: `{payload.get('official_performance')}`",
        f"- Fast/full audit: `{payload.get('fast_full_audit', {}).get('status', 'not_run')}`",
        "",
        "## Baseline",
        "",
    ]
    for row in payload.get("baseline", []):
        lines.append(_metric_line(row["train_metrics"], row["spec"]["candidate_source_name"]))
    lines.extend(
        [
            "",
            "## Top Train",
            "",
            "| Rank | Promote | Equal-Slot Net | Selected-Day Net | Active-Day Net | Trades | MFE Cap | Max DD | Plan |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(payload.get("top_train", [])[:30], start=1):
        metrics = row.get("train_metrics") or {}
        lines.append(
            f"| {rank} | {int(bool(row.get('promotion_pass')))} | "
            f"{100.0 * metrics.get('equal_slot_net_return_pct', metrics.get('slot_cumulative_net_return_pct', 0.0)):.1f}% | "
            f"{100.0 * metrics.get('selected_day_net_pct', 0.0):.3f}% | "
            f"{100.0 * metrics.get('active_day_net_pct', 0.0):.3f}% | "
            f"{metrics.get('trade_count', 0.0):.0f} | "
            f"{metrics.get('avg_mfe_capture', 0.0):.3f} | "
            f"{100.0 * metrics.get('max_drawdown_net_pct', 0.0):.1f}% | {row.get('name')} |"
        )
    return "\n".join(lines) + "\n"


def _metric_line(metrics: dict[str, float], label: str) -> str:
    return (
        f"- {label}: equal-slot net {100.0 * metrics.get('equal_slot_net_return_pct', metrics.get('slot_cumulative_net_return_pct', 0.0)):.1f}%, "
        f"selected-day net {100.0 * metrics.get('selected_day_net_pct', 0.0):.3f}%, "
        f"trades {metrics.get('trade_count', 0.0):.0f}, "
        f"MFE capture {metrics.get('avg_mfe_capture', 0.0):.3f}"
    )


def _phase_seed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    best = (payload.get("top_train") or [{}])[0]
    return {
        "strategy": "olr",
        "artifact_promotion_policy": "training_only_until_holdout_and_paper_parity",
        "research_sweep_path": payload.get("research_sweep_path"),
        "execution_sweep_seed": {
            "candidate_source": (best.get("spec") or {}).get("candidate_source_name"),
            "trade_plan": best.get("spec"),
            "metrics": best.get("train_metrics"),
            "policy": "Use as the fixed OLR execution seed for OOS/paper validation; this artifact is not official performance.",
        },
    }


def _latest_research_sweep_path() -> Path:
    candidates = sorted(Path("data/backtests/output/olr/research_sweeps").glob("olr_research_sweep_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No OLR research sweep artifact found; pass --research-sweep-path")
    return candidates[0]


def _snapshot_hash(snapshots: dict[date, OLRDailySnapshot]) -> str:
    return stable_signature({day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshots.items())})


def _file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dedupe_entries(specs: Iterable[OLREntryPlan]) -> list[OLREntryPlan]:
    out: list[OLREntryPlan] = []
    seen: set[str] = set()
    for spec in specs:
        data = asdict(_name_entry(spec))
        data.pop("name", None)
        key = json.dumps(data, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(_name_entry(spec))
    return out


def _dedupe_exits(specs: Iterable[OLRExitPlan]) -> list[OLRExitPlan]:
    out: list[OLRExitPlan] = []
    seen: set[str] = set()
    for spec in specs:
        data = asdict(_name_exit(spec))
        data.pop("name", None)
        key = json.dumps(data, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(_name_exit(spec))
    return out


def _limit_specs(items: list[Any], limit: int) -> list[Any]:
    count = int(limit or 0)
    if count <= 0 or count >= len(items):
        return list(items)
    return _even_sample(items, count)


def _even_sample(items: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    selected = []
    used: set[int] = set()
    last = len(items) - 1
    for index in range(count):
        position = round(index * last / max(count - 1, 1))
        while position in used and position < last:
            position += 1
        while position in used and position > 0:
            position -= 1
        used.add(position)
        selected.append(items[position])
    return selected


def _near_int(value: int, deltas: tuple[int, ...], floor: int, ceiling: int) -> list[int]:
    return [int(item) for item in _ordered_unique(max(floor, min(ceiling, int(value) + int(delta))) for delta in (0, *deltas, *(-delta for delta in deltas)))]


def _near_float(value: float, anchors: tuple[float, ...], delta: float, floor: float, ceiling: float) -> list[float]:
    raw = [float(value), *anchors, float(value) - delta, float(value) + delta]
    return [float(item) for item in _ordered_unique(round(max(floor, min(ceiling, float(item))), 6) for item in raw if math.isfinite(float(item)))]


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _entry_key(entry: OLREntryPlan) -> str:
    data = asdict(entry)
    data.pop("name", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _label(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 9.0:
        return "na"
    return f"{number:.4g}".replace("-", "m").replace(".", "p")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))[:120] or "source"


def _clear_progress_files(output_dir: Path) -> None:
    for name in ("progress.jsonl", "run_status.json"):
        try:
            (output_dir / name).unlink()
        except FileNotFoundError:
            pass
    for path in output_dir.glob("progress_*.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_run_status(output_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"updated_at": _utc_now_iso(), "stage": stage, **extra}
    tmp = output_dir / "run_status.json.tmp"
    path = output_dir / "run_status.json"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "status", **payload}, sort_keys=True, default=str) + "\n")
    details = " ".join(f"{key}={value}" for key, value in extra.items())
    print(f"[olr-trade-plan-sweep] status {stage} {details}".rstrip(), flush=True)


def _record_progress(output_dir: Path, stage: str, completed: int, total: int, rows: list[PlanResult], row: PlanResult) -> None:
    if completed not in {1, 2, 3, 5, 10, total} and completed % 100 != 0:
        return
    _write_progress(output_dir, stage, completed, total, rows)
    event = {"updated_at": _utc_now_iso(), "stage": stage, "completed": completed, "total": total, "row": _progress_row(row)}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    print(
        "[olr-trade-plan-sweep] "
        f"{stage} {completed}/{total} {row.spec.name} "
        f"equal_slot={100.0 * row.train_metrics.get('equal_slot_net_return_pct', row.train_metrics.get('slot_cumulative_net_return_pct', 0.0)):.1f}% "
        f"net/day={100.0 * row.train_metrics.get('selected_day_net_pct', 0.0):.3f}% "
        f"trades={row.train_metrics.get('trade_count', 0.0):.0f} reject={row.reject_reason}",
        flush=True,
    )


def _write_progress(output_dir: Path, stage: str, completed: int, total: int, rows: Sequence[PlanResult]) -> None:
    ranked = sorted(rows, key=_plan_sort_key)
    payload = {
        "updated_at": _utc_now_iso(),
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "percent": round(100.0 * completed / total, 3) if total else 100.0,
        "best_train_so_far": _progress_row(ranked[0]) if ranked else None,
        "top_train": [_progress_row(row) for row in ranked[:15]],
    }
    tmp = output_dir / f"progress_{stage}.json.tmp"
    path = output_dir / f"progress_{stage}.json"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _progress_row(row: PlanResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "score": row.score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "promotion_pass": row.promotion_pass,
        "slot_cumulative_net_return_pct": row.train_metrics.get("slot_cumulative_net_return_pct", 0.0),
        "equal_slot_net_return_pct": row.train_metrics.get("equal_slot_net_return_pct", row.train_metrics.get("slot_cumulative_net_return_pct", 0.0)),
        "selected_day_net_pct": row.train_metrics.get("selected_day_net_pct", 0.0),
        "active_day_net_pct": row.train_metrics.get("active_day_net_pct", 0.0),
        "trade_count": row.train_metrics.get("trade_count", 0.0),
        "max_drawdown_net_pct": row.train_metrics.get("max_drawdown_net_pct", 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep OLR post-14:30 execution plans over fixed top research selectors.")
    parser.add_argument("--config", default="config/optimization/olr.yaml")
    parser.add_argument("--research-sweep-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--top-source-count", type=int, default=3)
    parser.add_argument("--fold-count", type=int, default=2)
    parser.add_argument("--coarse-entry-limit", type=int, default=320)
    parser.add_argument("--coarse-exit-limit", type=int, default=220)
    parser.add_argument("--entry-seed-top-n", type=int, default=10)
    parser.add_argument("--deep-refine-top-n", type=int, default=12)
    parser.add_argument("--deep-refine-max-specs", type=int, default=3000)
    parser.add_argument("--finalist-count", type=int, default=40)
    parser.add_argument("--audit-finalist-count", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = normalize_runtime_config("olr", load_yaml_config(args.config))
    payload = run_trade_plan_sweep(
        config,
        research_sweep_path=args.research_sweep_path,
        output_dir=args.output_dir,
        holdout_days=args.holdout_days,
        top_source_count=args.top_source_count,
        fold_count=args.fold_count,
        coarse_entry_limit=args.coarse_entry_limit,
        coarse_exit_limit=args.coarse_exit_limit,
        entry_seed_top_n=args.entry_seed_top_n,
        deep_refine_top_n=args.deep_refine_top_n,
        deep_refine_max_specs=args.deep_refine_max_specs,
        finalist_count=args.finalist_count,
        audit_finalist_count=args.audit_finalist_count,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(json.dumps({"strategy": "olr", "sweep_hash": payload["sweep_hash"], "artifact_paths": payload["artifact_paths"], "top_train": payload["top_train"][:3]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
