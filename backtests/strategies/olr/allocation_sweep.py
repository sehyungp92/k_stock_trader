from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.config import load_yaml_config, normalize_runtime_config
from strategy_olr.config import OLRConfig, OLR_CORE_VERSION
from strategy_olr.execution import (
    EXECUTION_CORE_VERSION,
    OLRAllocationPlan,
    OLREntryPlan,
    OLRExitPlan,
    olr_outcome_hash,
    round_trip_cost_pct,
    summarize_olr_portfolio_proxy,
    summarize_olr_outcomes_with_allocation,
)

from .research_sweep import (
    DEFAULT_HOLDOUT_DAYS,
    Experiment,
    _resolve_folds,
    _training_config,
    build_afternoon_sweep_candidates,
)
from .runner import attach_overnight_labels_to_snapshots, compile_olr_replay_bundle, run_olr_backtest
from .trade_plan_sweep import (
    CandidateSource,
    CompiledExecutionSet,
    OLRTradePlanSpec,
    _close_auction_entry,
    _file_hash,
    _name_plan,
    _next_close_exit,
    _safe_name,
    _spec_payload,
    _source_payload,
    build_compiled_execution_set,
    collect_outcomes,
)


ALLOCATION_SWEEP_VERSION = "olr-close-to-close-selection-allocation-sweep-v7"
DEFAULT_OUTPUT_DIR = Path("data/backtests/output/olr/allocation_sweeps")
ALLOCATION_CACHE_VERSION = "olr-allocation-portfolio-proxy-cache-v6"
ALLOCATION_OFFICIAL_AUDIT_CACHE_VERSION = "olr-allocation-official-mtm-audit-cache-v10"
ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION = "olr-official-audit-keys-v2"
OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS = 2


@dataclass(frozen=True, slots=True)
class AllocationSweepRow:
    name: str
    source: CandidateSource
    trade_spec: OLRTradePlanSpec
    allocation: OLRAllocationPlan
    score: float
    rejected: bool
    reject_reason: str
    train_metrics: dict[str, float]
    fold_metrics: tuple[dict[str, Any], ...]
    replay_digest: dict[str, Any]
    beats_reference: bool


def run_allocation_sweep(
    config: dict[str, Any],
    *,
    research_sweep_path: str | Path,
    trade_plan_sweep_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    max_sources: int = 0,
    fold_count: int = 2,
    finalist_count: int = 40,
    audit_finalist_count: int = 8,
    max_workers: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clear_progress_files(out)
    _write_run_status(out, "preparing_inputs")
    research_path = Path(research_sweep_path)
    research_payload = json.loads(research_path.read_text(encoding="utf-8"))
    trade_payload = json.loads(Path(trade_plan_sweep_path).read_text(encoding="utf-8")) if trade_plan_sweep_path else {}
    trade_specs = _trade_specs_from_payload(trade_payload)
    sources = _sources_for_trade_specs(trade_payload, trade_specs) if trade_specs else build_stage2_candidate_sources(research_payload, max_sources=max_sources)
    trade_specs_by_source = _trade_specs_by_source(sources, trade_specs)
    allocations = build_allocation_plans()
    if dry_run:
        payload = {
            "strategy": "olr",
            "dry_run": True,
            "candidate_sources": len(sources),
            "trade_plan_specs": sum(len(items) for items in trade_specs_by_source.values()),
            "allocation_plans": len(allocations),
            "candidate_count": sum(max(1, len(trade_specs_by_source.get(source.name, ()))) * len(allocations) for source in sources),
            "max_workers": max(1, min(int(max_workers), 2)),
            "cache_policy": {
                "enabled": True,
                "scope": "deterministic portfolio proxy metrics by source/allocation",
                "official_audit_cached": True,
                "official_audit_cache_version": ALLOCATION_OFFICIAL_AUDIT_CACHE_VERSION,
                "official_audit_evidence_version": ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION,
            },
            "official_performance": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    training_config = _training_config(dict(config or {}), holdout_days)
    compiled = build_compiled_execution_set(
        training_config,
        research_payload,
        sources,
        holdout_days=holdout_days,
        use_fast_cache=True,
    )
    cfg = OLRConfig.from_mapping(compiled.dataset.config, {})
    folds = _resolve_folds(list(compiled.eligible_dates), fold_days=None, fold_count=fold_count)
    reference = _reference_metrics(trade_payload)
    _write_run_status(out, "evaluating_portfolio_proxy", sources=len(compiled.sources), allocations=len(allocations), max_workers=max(1, min(int(max_workers), 2)))
    rows = _evaluate_sources(compiled, cfg, folds, allocations, reference, trade_specs_by_source=trade_specs_by_source, output_dir=out, max_workers=max_workers)
    rows.sort(key=_row_sort_key)
    finalists = rows[: max(1, int(finalist_count))]
    _write_run_status(out, "compiling_full_audit_bundle", finalists=len(finalists))
    full_compiled = build_compiled_execution_set(
        training_config,
        research_payload,
        [row.source for row in finalists],
        holdout_days=holdout_days,
        use_fast_cache=False,
    )
    proxy_rank_by_name = {row.name: index for index, row in enumerate(rows, start=1)}
    requested_audit_count = min(max(1, int(audit_finalist_count)), len(finalists)) if finalists else 0
    _write_run_status(
        out,
        "auditing_official_mtm",
        audit_finalists=requested_audit_count,
        max_workers=max(1, min(int(max_workers), 2)),
        effective_workers=1,
    )
    audits = _audit_rows(
        finalists[:requested_audit_count],
        full_compiled,
        cfg,
        folds,
        reference,
        max_workers=max_workers,
        output_dir=out,
        stage="auditing_official_mtm",
    )
    _annotate_proxy_official_ranks(audits, proxy_rank_by_name)
    rank_diagnostics = _proxy_official_rank_diagnostics(audits)
    expanded_audit_count = _expanded_audit_count(requested_audit_count, len(finalists), rank_diagnostics)
    expansion_reason = str(rank_diagnostics.get("expansion_reason") or "")
    if expanded_audit_count > requested_audit_count:
        _write_run_status(
            out,
            "expanding_official_mtm_audit",
            audit_finalists=expanded_audit_count,
            prior_audit_finalists=requested_audit_count,
            reason=rank_diagnostics.get("expansion_reason", "proxy_official_alignment"),
        )
        audit_by_name = {str(row.get("name")): row for row in audits}
        additional = _audit_rows(
            finalists[requested_audit_count:expanded_audit_count],
            full_compiled,
            cfg,
            folds,
            reference,
            max_workers=max_workers,
            output_dir=out,
            stage="expanding_official_mtm_audit",
            completed_offset=len(audits),
            total_override=expanded_audit_count,
        )
        for row in additional:
            audit_by_name[str(row.get("name"))] = row
        audits = list(audit_by_name.values())
        _annotate_proxy_official_ranks(audits, proxy_rank_by_name)
        rank_diagnostics = _proxy_official_rank_diagnostics(audits)
    audits.sort(key=lambda row: (int(row.get("proxy_rank", 10**9) or 10**9), str(row.get("name") or "")))
    rank_diagnostics["requested_audit_finalist_count"] = requested_audit_count
    rank_diagnostics["final_audit_finalist_count"] = len(audits)
    rank_diagnostics["audit_coverage_expanded"] = expanded_audit_count > requested_audit_count
    if expansion_reason:
        rank_diagnostics["expansion_reason"] = expansion_reason
    audit_pass = all(bool(row.get("audit_pass")) for row in audits) if audits else False
    official_audit_by_name = {str(row.get("name")): row for row in audits}
    official_by_name = {
        str(row.get("name")): dict(row.get("official_mtm_metrics") or {})
        for row in audits
        if row.get("official_mtm_metrics")
    }
    top_train = [_row_payload(row, official_by_name.get(row.name), official_audit_by_name.get(row.name)) for row in rows[:100]]
    top_beats_reference = [_row_payload(row, official_by_name.get(row.name), official_audit_by_name.get(row.name)) for row in rows if row.beats_reference][:50]
    top_official_train = sorted(
        (
            _row_payload(row, official_by_name.get(row.name), official_audit_by_name.get(row.name))
            for row in finalists
            if row.name in official_by_name
        ),
        key=_official_row_sort_key,
    )
    payload = {
        "strategy": "olr",
        "sweep_version": ALLOCATION_SWEEP_VERSION,
        "strategy_core_version": OLR_CORE_VERSION,
        "execution_core_version": EXECUTION_CORE_VERSION,
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "official_performance": False,
        "research_sweep_path": str(research_path),
        "trade_plan_sweep_path": str(trade_plan_sweep_path or ""),
        "training_window": {
            "start": compiled.eligible_dates[0].isoformat() if compiled.eligible_dates else "",
            "end": compiled.eligible_dates[-1].isoformat() if compiled.eligible_dates else "",
            "sessions": len(compiled.eligible_dates),
        },
        "holdout_policy": {
            "holdout_days": int(holdout_days),
            "train_only": True,
            "selection_uses_holdout": False,
            "execution_uses_holdout": False,
            "sizing_uses_holdout": False,
        },
        "causality_policy": {
            "candidate_generation": "Stage 1 daily rows and lagged flow use row_date < trade_date; Stage 2 selector uses timestamp < 14:30 KST.",
            "execution": "Fixed close-auction resting entry submitted after the 14:30 decision and filled only on the configured close-auction print; exit is next-session close.",
            "trade_plan_execution": "When a trade-plan sweep artifact is supplied, allocation rows and official audits consume the swept OLRTradePlanSpec instead of falling back to close_auction_next_close.",
            "allocation": "Sizing uses only same-day selected count/rank/score known at the 14:30 artifact plus fixed allocation parameters.",
        },
        "timing_policy": {
            "stage2_selector_cutoff": "timestamp < 14:30 KST",
            "last_continuous_5m_bar": "15:15 KST",
            "close_auction_fill_time": str(cfg.auction_fill_time),
            "stage2_proxy_basis": "close-to-next-close opportunity label; not official executable performance",
            "stage3_allocation_audit_basis": "swept trade-plan spec when trade_plan_sweep_path is supplied; otherwise fixed close-auction entry at configured auction_fill_time to next-session close",
        },
        "risk_stance_policy": {
            "stance": "aggressive_but_not_unbounded",
            "broad_objective": "portfolio_proxy_net_return_pct",
            "promotion_objective": "official_mtm_net_return_pct",
            "min_active_gross_exposure_pct": 0.45,
            "max_train_proxy_drawdown_abs_pct": 0.30,
            "single_name_policy": "High-concentration plans require at least two selected symbols; no one-name all-in promotion path.",
            "rationale": "Avoid underdeployed low-drawdown artifacts while rejecting portfolio paths whose drawdown is already uncontrolled in train.",
        },
        "reference_train_metrics": reference,
        "cost_policy": {
            "round_trip_cost_pct": round_trip_cost_pct(cfg),
            "slippage_bps_each_side": cfg.slippage_bps,
            "commission_bps_each_side": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
        },
        "sweep_counts": {
            "candidate_sources": len(sources),
            "trade_plan_specs": sum(len(items) for items in trade_specs_by_source.values()),
            "allocation_plans": len(allocations),
            "candidate_count": len(rows),
            "finalist_count": len(finalists),
            "beats_reference_count": sum(1 for row in rows if row.beats_reference),
        },
        "source_fingerprints": {
            "research_sweep_hash": str(research_payload.get("sweep_hash") or ""),
            "research_sweep_file": _file_hash(research_path),
            "compiled_execution": compiled.source_fingerprint,
            "candidate_artifacts": compiled.candidate_artifact_hash,
            "daily_intraday": compiled.dataset.source_fingerprint,
        },
        "fast_replay_policy": {
            "enabled": True,
            "mode": "compiled_fixed_candidate_execution_replay",
            "full_audit_finalist_count": len(audits),
            "requested_audit_finalist_count": requested_audit_count,
            "audit_metric_tolerance": 1e-10,
        },
        "cache_policy": {
            "enabled": True,
            "cache_version": ALLOCATION_CACHE_VERSION,
            "cache_dir": str(_allocation_cache_dir(out)),
            "scope": "Deterministic source/allocation train metrics, folds, and portfolio proxy only.",
            "invalidates_on": [
                "compiled source fingerprint",
                "candidate artifact hash",
                "source mutations",
                "swept trade-plan spec",
                "allocation plan",
                "fold boundaries",
                "initial equity",
                "cost policy",
                "execution/allocation cache version",
            ],
            "official_audit_cached": True,
            "official_audit_cache_version": ALLOCATION_OFFICIAL_AUDIT_CACHE_VERSION,
            "official_audit_evidence_version": ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION,
            "official_audit_cache_dir": str(_official_audit_cache_dir(out)),
            "official_audit_cache_scope": "Deterministic audited OLR core + SimBroker MTM metrics and selected/order/fill/nonfill/open-exposure key evidence for train-only finalists.",
            "official_audit_bar_scope": "Selected Stage-2 snapshot symbols on trade date plus train-only follow-up bars for close-auction exit recovery; noncandidate bars are suppressed because OLR core cannot act on symbols absent from the source-fingerprinted candidate snapshot.",
            "lookahead_policy": "Cache keys are built from train-only compiled artifacts; no holdout or same-day unavailable data is introduced.",
        },
        "audit_replays": audits,
        "fast_full_audit": _audit_summary(audits),
        "proxy_official_rank_diagnostics": rank_diagnostics,
        "audit_pass": audit_pass,
        "optimization_metric_basis": "Broad allocation pruning uses portfolio_proxy_net_return_pct, a cash/integer-quantity proxy for official MTM; legacy slot labels are diagnostics only.",
        "promotion_metric_basis": "top_official_train uses audited OLR core + SimBroker official_mtm_net_return_pct; top_train is the fast proxy pruning frontier.",
        "metric_contract": {
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_basis": "OLRReplayAdapter -> OLR core -> neutral actions -> SimBroker.equity_curve_bar_level_mtm",
            "promotion_requires_audit_pass": True,
            "official_replay_pass": bool(audits),
            "audit_pass": audit_pass,
            "audit_status": "audited_full_bundle_passed" if audit_pass else "audit_failed",
            "official_metrics": [
                "official_mtm_net_return_pct",
                "official_mtm_max_drawdown_pct",
                "official_mtm_sharpe",
                "entry_fill_count",
                "exit_fill_count",
                "end_open_position_count",
                "open_order_count",
                "official_trade_plan_supported",
            ],
            "proxy_metrics": ["portfolio_proxy_net_return_pct", "portfolio_proxy_max_drawdown_pct"],
            "legacy_closed_trade_metrics": ["slot_cumulative_net_return_pct"],
        },
        "execution_contract": {
            "strategy": "olr",
            "phase_framework_version": "custom-olr-allocation-sweep",
            "strategy_core_version": OLR_CORE_VERSION,
            "execution_core_version": EXECUTION_CORE_VERSION,
            "source_fingerprint": compiled.source_fingerprint,
            "feature_manifest_hash": compiled.dataset.source_fingerprint,
            "candidate_snapshot_hash": compiled.candidate_artifact_hash,
            "date_window": {
                "start": compiled.eligible_dates[0].isoformat() if compiled.eligible_dates else "",
                "end": compiled.eligible_dates[-1].isoformat() if compiled.eligible_dates else "",
                "sessions": len(compiled.eligible_dates),
            },
            "cost_policy": {
                "round_trip_cost_pct": round_trip_cost_pct(cfg),
                "slippage_bps_each_side": cfg.slippage_bps,
                "commission_bps_each_side": cfg.commission_bps,
                "tax_bps_on_sell": cfg.tax_bps_on_sell,
            },
            "fill_timing": f"close_auction_{cfg.auction_fill_time}_to_next_close",
            "auction_mode": "close_auction",
            "capability_level": "compiled",
            "replay_mode": "compiled_fixed_candidate_execution_replay",
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_basis": "OLRReplayAdapter -> OLR core -> neutral actions -> SimBroker.equity_curve_bar_level_mtm",
        },
        "top_official_train": top_official_train,
        "top_train": top_train,
        "top_beats_reference": top_beats_reference,
    }
    payload["sweep_hash"] = stable_signature(
        {
            "version": ALLOCATION_SWEEP_VERSION,
            "training_window": payload["training_window"],
            "reference": reference,
            "top_official_train": payload["top_official_train"][:10],
            "top_train": payload["top_train"][:10],
            "audit_pass": audit_pass,
        }
    )
    json_path = out / f"olr_allocation_sweep_{payload['sweep_hash'][:12]}.json"
    md_path = out / f"olr_allocation_sweep_{payload['sweep_hash'][:12]}.md"
    seed_path = out / f"olr_allocation_seed_{payload['sweep_hash'][:12]}.json"
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path), "phase_auto_seed": str(seed_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    seed_path.write_text(json.dumps(_phase_seed_payload(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_run_status(out, "completed", sweep_hash=payload["sweep_hash"], json=str(json_path), markdown=str(md_path), audit_pass=audit_pass)
    return payload


def build_stage2_candidate_sources(payload: dict[str, Any], *, max_sources: int = 0) -> tuple[CandidateSource, ...]:
    base = dict(payload.get("base_mutations") or {})
    stage1_seed = dict(payload.get("selected_stage1_seed") or {})
    stage1 = dict(stage1_seed.get("mutations") or {})
    stage1_name = str(stage1_seed.get("name") or "stage1")
    experiments = build_afternoon_sweep_candidates()
    if int(max_sources or 0) > 0:
        experiments = experiments[: int(max_sources)]
    sources: list[CandidateSource] = []
    for index, experiment in enumerate(experiments, start=1):
        mutations = dict(base)
        mutations.update(stage1)
        mutations.update(dict(experiment.mutations or {}))
        sources.append(
            CandidateSource(
                rank=index,
                name=_safe_name(f"stage2_{experiment.name}"),
                stage1_name=stage1_name,
                stage2_name=experiment.name,
                score=0.0,
                mutations=mutations,
                stage1_mutations=stage1,
                stage2_mutations=dict(experiment.mutations or {}),
            )
        )
    return tuple(sources)


def _trade_specs_from_payload(payload: dict[str, Any]) -> tuple[OLRTradePlanSpec, ...]:
    if not payload:
        return ()
    rows = list(payload.get("top_promoted") or payload.get("top_train") or [])
    specs: list[OLRTradePlanSpec] = []
    seen: set[str] = set()
    for row in rows:
        spec_payload = row.get("spec") if isinstance(row, dict) else None
        if not isinstance(spec_payload, dict):
            continue
        spec = _trade_spec_from_payload(spec_payload)
        if spec.name in seen:
            continue
        seen.add(spec.name)
        specs.append(spec)
    return tuple(specs)


def _trade_spec_from_payload(payload: dict[str, Any]) -> OLRTradePlanSpec:
    entry_payload = dict(payload.get("entry") or {})
    exit_payload = dict(payload.get("exit") or {})
    entry_allowed = set(OLREntryPlan.__dataclass_fields__)
    exit_allowed = set(OLRExitPlan.__dataclass_fields__)
    spec = OLRTradePlanSpec(
        name=str(payload.get("name") or ""),
        candidate_source_name=str(payload.get("candidate_source_name") or ""),
        entry=OLREntryPlan(**{key: value for key, value in entry_payload.items() if key in entry_allowed}),
        exit=OLRExitPlan(**{key: value for key, value in exit_payload.items() if key in exit_allowed}),
    )
    return _name_plan(spec)


def _sources_for_trade_specs(payload: dict[str, Any], specs: Sequence[OLRTradePlanSpec]) -> tuple[CandidateSource, ...]:
    source_payloads = payload.get("candidate_sources") or []
    lookup = {
        str(row.get("name") or ""): _candidate_source_from_payload(row)
        for row in source_payloads
        if isinstance(row, dict) and row.get("name")
    }
    out: list[CandidateSource] = []
    seen: set[str] = set()
    for spec in specs:
        source = lookup.get(spec.candidate_source_name)
        if source is None:
            raise ValueError(f"Trade-plan sweep source not found for allocation audit: {spec.candidate_source_name}")
        if source.name in seen:
            continue
        seen.add(source.name)
        out.append(source)
    return tuple(out)


def _candidate_source_from_payload(payload: dict[str, Any]) -> CandidateSource:
    return CandidateSource(
        rank=int(payload.get("rank", 0) or 0),
        name=str(payload.get("name") or ""),
        stage1_name=str(payload.get("stage1_name") or ""),
        stage2_name=str(payload.get("stage2_name") or ""),
        score=float(payload.get("score", 0.0) or 0.0),
        mutations=dict(payload.get("mutations") or {}),
        artifact_hash=str(payload.get("artifact_hash") or ""),
        stage1_mutations=dict(payload.get("stage1_mutations") or {}),
        stage2_mutations=dict(payload.get("stage2_mutations") or {}),
    )


def _trade_specs_by_source(
    sources: Sequence[CandidateSource],
    specs: Sequence[OLRTradePlanSpec],
) -> dict[str, tuple[OLRTradePlanSpec, ...]]:
    if not specs:
        return {source.name: (_name_plan(OLRTradePlanSpec("", source.name, _close_auction_entry(), _next_close_exit())),) for source in sources}
    grouped: dict[str, list[OLRTradePlanSpec]] = {source.name: [] for source in sources}
    for spec in specs:
        grouped.setdefault(spec.candidate_source_name, []).append(spec)
    return {name: tuple(items) for name, items in grouped.items() if items}


def build_allocation_plans() -> tuple[OLRAllocationPlan, ...]:
    plans = [OLRAllocationPlan("fixed_slots", mode="fixed_slots", max_position_pct=0.25)]
    for cap in (0.30, 0.35, 0.40, 0.50):
        plans.append(OLRAllocationPlan(_allocation_name("capped_equal", cap), mode="capped_equal", max_position_pct=cap))
    for cap in (0.67, 1.00):
        plans.append(OLRAllocationPlan(_allocation_name("capped_equal_min2", cap), mode="capped_equal", max_position_pct=cap, min_selected=2))
    plans.append(OLRAllocationPlan("selected_equal_full_min2", mode="selected_equal", max_position_pct=1.0, min_selected=2))
    for cap in (0.35, 0.50):
        plans.append(OLRAllocationPlan(_allocation_name("capped_equal_min2", cap), mode="capped_equal", max_position_pct=cap, min_selected=2))
    for cap in (0.40, 0.50):
        for decay in (0.5, 1.0, 1.5):
            plans.append(
                OLRAllocationPlan(
                    f"rank_weighted_cap{_label(cap)}_d{_label(decay)}",
                    mode="rank_weighted",
                    max_position_pct=cap,
                    rank_decay=decay,
                )
            )
    for cap in (0.35, 0.50):
        plans.append(OLRAllocationPlan(f"score_weighted_cap{_label(cap)}", mode="score_weighted", max_position_pct=cap))
    for cap in (0.35,):
        for decay in (0.5, 1.0):
            plans.append(
                OLRAllocationPlan(
                    f"rank_score_weighted_cap{_label(cap)}_d{_label(decay)}",
                    mode="rank_score_weighted",
                    max_position_pct=cap,
                    rank_decay=decay,
                )
            )
    return tuple(plans)


def _evaluate_sources(
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    folds: list[tuple[date, date]],
    allocations: Sequence[OLRAllocationPlan],
    reference: dict[str, float],
    *,
    trade_specs_by_source: dict[str, tuple[OLRTradePlanSpec, ...]],
    output_dir: Path,
    max_workers: int,
) -> list[AllocationSweepRow]:
    worker_count = max(1, min(int(max_workers), 2))
    cache_dir = _allocation_cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_context = _allocation_cache_context(compiled, cfg, folds)
    total_sources = len(compiled.sources)
    if worker_count <= 1 or len(compiled.sources) <= 1:
        rows: list[AllocationSweepRow] = []
        for completed, source in enumerate(compiled.sources, start=1):
            source_rows = _evaluate_source(source, compiled, cfg, folds, allocations, reference, trade_specs_by_source=trade_specs_by_source, cache_dir=cache_dir, cache_context=cache_context)
            rows.extend(source_rows)
            _record_source_progress(output_dir, completed, total_sources, rows, source, source_rows)
        return rows
    rows = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_evaluate_source, source, compiled, cfg, folds, allocations, reference, trade_specs_by_source=trade_specs_by_source, cache_dir=cache_dir, cache_context=cache_context): source
            for source in compiled.sources
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            source_rows = future.result()
            rows.extend(source_rows)
            _record_source_progress(output_dir, completed, total_sources, rows, source, source_rows)
    return rows


def _evaluate_source(
    source: CandidateSource,
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    folds: list[tuple[date, date]],
    allocations: Sequence[OLRAllocationPlan],
    reference: dict[str, float],
    *,
    trade_specs_by_source: dict[str, tuple[OLRTradePlanSpec, ...]] | None = None,
    cache_dir: Path | None = None,
    cache_context: dict[str, Any] | None = None,
) -> list[AllocationSweepRow]:
    specs = tuple((trade_specs_by_source or {}).get(source.name, ())) or (_name_plan(OLRTradePlanSpec("", source.name, _close_auction_entry(), _next_close_exit())),)
    counts = compiled.selection_counts_by_source.get(source.name, {})
    rows = []
    initial_equity = float(compiled.dataset.config.get("initial_equity", 10_000_000.0) or 10_000_000.0)
    for spec in specs:
        cached_payloads = {
            allocation.name: _load_allocation_cache(cache_dir, cache_context, source, spec, allocation)
            for allocation in allocations
        }
        missing_allocations = [allocation for allocation in allocations if cached_payloads.get(allocation.name) is None]
        outcomes: Sequence[Any] = ()
        digest = {"outcome_hash": "", "trade_count": 0}
        if missing_allocations:
            outcomes = collect_outcomes(spec, compiled, cfg, compiled.eligible_dates)
            digest = {"outcome_hash": olr_outcome_hash(outcomes), "trade_count": len(outcomes), "trade_plan_hash": stable_signature(_spec_payload(spec))}
        for allocation in allocations:
            cached = cached_payloads.get(allocation.name)
            if cached is not None:
                metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in dict(cached.get("train_metrics") or {}).items()}
                fold_rows = tuple(dict(row) for row in cached.get("fold_metrics") or ())
                replay_digest = dict(cached.get("replay_digest") or {})
                replay_digest["portfolio_proxy_cache_hit"] = True
            else:
                metrics = summarize_olr_outcomes_with_allocation(
                    outcomes,
                    session_dates=compiled.eligible_dates,
                    selection_counts=counts,
                    slot_count=cfg.overnight_slot_count,
                    allocation=allocation,
                )
                metrics.update(
                    summarize_olr_portfolio_proxy(
                        outcomes,
                        session_dates=compiled.eligible_dates,
                        selection_counts=counts,
                        slot_count=cfg.overnight_slot_count,
                        allocation=allocation,
                        initial_equity=initial_equity,
                        config=cfg,
                    )
                )
                fold_rows = _allocation_fold_metrics(outcomes, compiled.eligible_dates, folds, counts, cfg.overnight_slot_count, allocation, initial_equity, cfg)
                replay_digest = {**digest, "metric_hash": stable_signature(metrics), "portfolio_proxy_cache_hit": False}
                _write_allocation_cache(cache_dir, cache_context, source, spec, allocation, metrics, fold_rows, replay_digest)
            score, reject = _score_allocation(metrics, fold_rows, reference)
            beats = _beats_reference(metrics, reference)
            rows.append(
                AllocationSweepRow(
                    name=f"{spec.name}__{allocation.name}",
                    source=source,
                    trade_spec=spec,
                    allocation=allocation,
                    score=round(0.0 if reject else score, 6),
                    rejected=bool(reject),
                    reject_reason=reject,
                    train_metrics=metrics,
                    fold_metrics=fold_rows,
                    replay_digest=replay_digest,
                    beats_reference=beats,
                )
            )
    return rows


def _allocation_fold_metrics(
    outcomes,
    dates: Sequence[date],
    folds: list[tuple[date, date]],
    counts: dict[date, int],
    slot_count: int,
    allocation: OLRAllocationPlan,
    initial_equity: float,
    cfg: OLRConfig,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for index, (start, end) in enumerate(folds, start=1):
        fold_dates = tuple(day for day in dates if start <= day <= end)
        rows.append(
            {
                "fold": index,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "metrics": summarize_olr_outcomes_with_allocation(
                    outcomes,
                    session_dates=fold_dates,
                    selection_counts=counts,
                    slot_count=slot_count,
                    allocation=allocation,
                )
                | summarize_olr_portfolio_proxy(
                    outcomes,
                    session_dates=fold_dates,
                    selection_counts=counts,
                    slot_count=slot_count,
                    allocation=allocation,
                    initial_equity=initial_equity,
                    config=cfg,
                ),
            }
        )
    return tuple(rows)


def _allocation_cache_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "cache" / "portfolio_proxy"


def _official_audit_cache_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "cache" / "official_mtm"


def _allocation_cache_context(compiled: CompiledExecutionSet, cfg: OLRConfig, folds: list[tuple[date, date]]) -> dict[str, Any]:
    return {
        "cache_version": ALLOCATION_CACHE_VERSION,
        "sweep_version": ALLOCATION_SWEEP_VERSION,
        "strategy_core_version": OLR_CORE_VERSION,
        "execution_core_version": EXECUTION_CORE_VERSION,
        "compiled_source_fingerprint": compiled.source_fingerprint,
        "candidate_artifact_hash": compiled.candidate_artifact_hash,
        "daily_intraday_fingerprint": compiled.dataset.source_fingerprint,
        "training_window": {
            "start": compiled.eligible_dates[0].isoformat() if compiled.eligible_dates else "",
            "end": compiled.eligible_dates[-1].isoformat() if compiled.eligible_dates else "",
            "sessions": len(compiled.eligible_dates),
        },
        "folds": [{"start": start.isoformat(), "end": end.isoformat()} for start, end in folds],
        "initial_equity": float(compiled.dataset.config.get("initial_equity", 10_000_000.0) or 10_000_000.0),
        "slot_count": int(cfg.overnight_slot_count),
        "cost_policy": {
            "slippage_bps_each_side": cfg.slippage_bps,
            "commission_bps_each_side": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
            "market_entry_price_buffer_bps": cfg.market_entry_price_buffer_bps,
            "auction_limit_offset_bps": cfg.auction_limit_offset_bps,
        },
        "shared_core_sizing_policy": "equity_weight_cash_cap_submission_price_v3",
        "causality_policy": {
            "selection": "compiled train-only OLR candidate snapshots",
            "labels": "train-only close-to-close/outcome labels already present in compiled execution set",
            "holdout": "excluded from cache context",
        },
    }


def _allocation_cache_key(
    cache_context: dict[str, Any] | None,
    source: CandidateSource,
    spec: OLRTradePlanSpec,
    allocation: OLRAllocationPlan,
) -> str:
    return stable_signature(
        {
            "context": cache_context or {},
            "source": _source_payload(source),
            "trade_plan": _spec_payload(spec),
            "allocation": asdict(allocation),
        }
    )


def _allocation_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def _load_allocation_cache(
    cache_dir: Path | None,
    cache_context: dict[str, Any] | None,
    source: CandidateSource,
    spec: OLRTradePlanSpec,
    allocation: OLRAllocationPlan,
) -> dict[str, Any] | None:
    if cache_dir is None or cache_context is None:
        return None
    cache_key = _allocation_cache_key(cache_context, source, spec, allocation)
    path = _allocation_cache_path(cache_dir, cache_key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if payload.get("cache_key") != cache_key:
        return None
    if payload.get("cache_version") != ALLOCATION_CACHE_VERSION:
        return None
    return payload


def _write_allocation_cache(
    cache_dir: Path | None,
    cache_context: dict[str, Any] | None,
    source: CandidateSource,
    spec: OLRTradePlanSpec,
    allocation: OLRAllocationPlan,
    metrics: dict[str, Any],
    fold_rows: Sequence[dict[str, Any]],
    replay_digest: dict[str, Any],
) -> None:
    if cache_dir is None or cache_context is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _allocation_cache_key(cache_context, source, spec, allocation)
    payload = {
        "cache_key": cache_key,
        "cache_version": ALLOCATION_CACHE_VERSION,
        "created_at": _utc_now_iso(),
        "official_performance": False,
        "cache_context_hash": stable_signature(cache_context),
        "source": _source_payload(source),
        "trade_plan": _spec_payload(spec),
        "allocation": asdict(allocation),
        "train_metrics": metrics,
        "fold_metrics": list(fold_rows),
        "replay_digest": {**dict(replay_digest), "portfolio_proxy_cache_key": cache_key},
        "metric_basis": "deterministic_train_only_portfolio_proxy_not_official_performance",
    }
    path = _allocation_cache_path(cache_dir, cache_key)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _score_allocation(metrics: dict[str, float], folds: tuple[dict[str, Any], ...], reference: dict[str, float]) -> tuple[float, str]:
    trades = float(metrics.get("trade_count", 0.0) or 0.0)
    if trades < 80.0:
        return 0.0, f"too_few_trades ({trades:.0f} < 80)"
    if float(metrics.get("selected_days", 0.0) or 0.0) < 80.0:
        return 0.0, "too_few_selected_days"
    slot = float(metrics.get("portfolio_proxy_net_return_pct", metrics.get("allocation_cumulative_net_return_pct", 0.0)) or 0.0)
    ref = float(reference.get("slot_cumulative_net_return_pct", 0.0) or 0.0)
    fold_slots = [
        float(row["metrics"].get("portfolio_proxy_net_return_pct", row["metrics"].get("allocation_cumulative_net_return_pct", 0.0)) or 0.0)
        for row in folds
    ]
    worst_fold = min(fold_slots) if fold_slots else slot
    dd = abs(float(metrics.get("portfolio_proxy_max_drawdown_pct", metrics.get("allocation_max_drawdown_net_pct", 0.0)) or 0.0))
    selected = float(metrics.get("portfolio_proxy_active_day_net_pct", metrics.get("selected_day_net_pct", 0.0)) or 0.0)
    active_gross = float(metrics.get("portfolio_proxy_avg_active_gross_exposure_pct", metrics.get("allocation_avg_active_gross_exposure", 0.0)) or 0.0)
    if active_gross < 0.45:
        return 0.0, f"underdeployed_active_gross ({active_gross:.3f} < 0.450)"
    if dd > 0.30:
        return 0.0, f"drawdown_too_large ({dd:.3f} > 0.300)"
    alpha_capture = float(metrics.get("avg_mfe_capture", metrics.get("mfe_capture", 0.0)) or 0.0)
    score = (
        120.0 * slot
        + 60.0 * worst_fold
        + 25.0 * selected
        + 12.0 * min(active_gross, 1.0)
        + 10.0 * _clip(float(metrics.get("net_win_share", 0.0) or 0.0))
        + 8.0 * _clip(alpha_capture / 0.55)
        - 85.0 * dd
    )
    if slot <= ref:
        score -= 100.0 * min(1.0, (ref - slot) / max(abs(ref), 1e-9))
    if worst_fold < 0.0:
        score -= 50.0
    return score, ""


def _beats_reference(metrics: dict[str, float], reference: dict[str, float]) -> bool:
    return (
        float(metrics.get("portfolio_proxy_net_return_pct", metrics.get("allocation_cumulative_net_return_pct", 0.0)) or 0.0)
        > float(reference.get("slot_cumulative_net_return_pct", 0.0) or 0.0)
        and float(metrics.get("portfolio_proxy_active_day_net_pct", metrics.get("selected_day_net_pct", 0.0)) or 0.0)
        >= float(reference.get("selected_day_net_pct", 0.0) or 0.0)
    )


def _audit_rows(
    rows: Sequence[AllocationSweepRow],
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    folds: list[tuple[date, date]],
    reference: dict[str, float],
    *,
    max_workers: int,
    output_dir: Path | None = None,
    stage: str = "auditing_official_mtm",
    completed_offset: int = 0,
    total_override: int | None = None,
) -> list[dict[str, Any]]:
    audits = []
    cache_dir = _official_audit_cache_dir(output_dir) if output_dir is not None else None
    cache_context = _official_audit_cache_context(compiled, cfg, folds, reference)
    total = int(total_override if total_override is not None else completed_offset + len(rows))
    cache_hits = 0
    # Official MTM audit builds large replay bundles and runs the live-parity core.
    # Keep it row-serial to avoid Windows native crashes from concurrent bundle pressure.
    for index, row in enumerate(rows, start=1):
        audit = _read_official_audit_cache(cache_dir, cache_context, row)
        if audit is not None:
            cache_hits += 1
        else:
            audit = _audit_row(row, compiled, cfg, folds, reference)
            _write_official_audit_cache(cache_dir, cache_context, row, audit)
        audits.append(audit)
        if output_dir is not None:
            _write_official_audit_progress(
                output_dir,
                stage,
                completed=completed_offset + index,
                total=total,
                cache_hits=cache_hits,
                latest=row.name,
                audits=audits,
                max_workers=max_workers,
            )
    audits.sort(key=lambda row: row.get("name", ""))
    return audits


def _audit_row(
    fast: AllocationSweepRow,
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    folds: list[tuple[date, date]],
    reference: dict[str, float],
) -> dict[str, Any]:
    full = _evaluate_source(
        fast.source,
        compiled,
        cfg,
        folds,
        [fast.allocation],
        reference,
        trade_specs_by_source={fast.source.name: (fast.trade_spec,)},
    )[0]
    metric_keys = sorted(set(fast.train_metrics) | set(full.train_metrics))
    deltas = {
        key: float(full.train_metrics.get(key, 0.0) or 0.0) - float(fast.train_metrics.get(key, 0.0) or 0.0)
        for key in metric_keys
    }
    max_abs_delta = max((abs(value) for value in deltas.values()), default=0.0)
    score_delta = float(full.score) - float(fast.score)
    outcome_match = fast.replay_digest.get("outcome_hash") == full.replay_digest.get("outcome_hash")
    reject_match = fast.rejected == full.rejected and fast.reject_reason == full.reject_reason
    official, official_keys = _official_allocation_report(fast, compiled, cfg)
    official_replay_pass = _official_replay_pass(official)
    audit_pass = outcome_match and reject_match and max_abs_delta <= 1e-10 and abs(score_delta) <= 1e-10 and official_replay_pass
    proxy_net = float(fast.train_metrics.get("portfolio_proxy_net_return_pct", fast.train_metrics.get("allocation_cumulative_net_return_pct", 0.0)) or 0.0)
    official_net = float(official.get("official_mtm_net_return_pct", 0.0) or 0.0)
    return {
        "name": fast.name,
        "audit_pass": audit_pass,
        "fast_full_proxy_audit_pass": outcome_match and reject_match and max_abs_delta <= 1e-10 and abs(score_delta) <= 1e-10,
        "official_replay_pass": official_replay_pass,
        "outcome_hash_match": outcome_match,
        "rejection_match": reject_match,
        "max_abs_metric_delta": max_abs_delta,
        "score_delta": score_delta,
        "metric_deltas": deltas,
        "fast_score": fast.score,
        "full_score": full.score,
        "fast_outcome_hash": fast.replay_digest.get("outcome_hash"),
        "full_outcome_hash": full.replay_digest.get("outcome_hash"),
        "official_mtm_metrics": official,
        "official_audit_keys": official_keys,
        "portfolio_proxy_net_return_pct": proxy_net,
        "official_proxy_net_delta": official_net - proxy_net,
        "official_proxy_abs_net_delta": abs(official_net - proxy_net),
    }


def _official_replay_pass(metrics: dict[str, float]) -> bool:
    return (
        float(metrics.get("official_trade_plan_supported", 1.0) or 0.0) == 1.0
        and float(metrics.get("same_bar_fill_count", 0.0) or 0.0) == 0.0
        and float(metrics.get("end_open_position_count", 0.0) or 0.0) == 0.0
        and float(metrics.get("open_order_count", 0.0) or 0.0) == 0.0
        and float(metrics.get("entry_fill_count", 0.0) or 0.0) == float(metrics.get("exit_fill_count", 0.0) or 0.0)
    )


def _official_audit_cache_context(
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
    folds: list[tuple[date, date]],
    reference: dict[str, float],
) -> dict[str, Any]:
    return {
        "cache_version": ALLOCATION_OFFICIAL_AUDIT_CACHE_VERSION,
        "evidence_version": ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION,
        "compiled_source_fingerprint": compiled.source_fingerprint,
        "candidate_artifact_hash": compiled.candidate_artifact_hash,
        "daily_intraday_fingerprint": compiled.dataset.source_fingerprint,
        "strategy_core_version": OLR_CORE_VERSION,
        "execution_core_version": EXECUTION_CORE_VERSION,
        "dates": {
            "start": compiled.eligible_dates[0].isoformat() if compiled.eligible_dates else "",
            "end": compiled.eligible_dates[-1].isoformat() if compiled.eligible_dates else "",
            "count": len(compiled.eligible_dates),
        },
        "folds": [{"start": start.isoformat(), "end": end.isoformat()} for start, end in folds],
        "costs": {
            "commission_bps": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
            "slippage_bps": cfg.slippage_bps,
            "auction_adverse_bps": cfg.auction_adverse_bps,
            "market_entry_price_buffer_bps": cfg.market_entry_price_buffer_bps,
            "auction_limit_offset_bps": cfg.auction_limit_offset_bps,
        },
        "reference_signature": stable_signature(reference),
        "bar_scope": "selected_trade_date_and_next_session_symbol_bars",
        "official_metric_basis": "OLRReplayAdapter -> OLR core -> neutral actions -> SimBroker.equity_curve_bar_level_mtm",
        "lookahead_policy": "Train-only candidate snapshots; Stage 1 row_date < trade_date; Stage 2 timestamp < 14:30 KST; holdout excluded.",
    }


def _official_audit_cache_key(context: dict[str, Any], row: AllocationSweepRow) -> str:
    return stable_signature(
        {
            "context": context,
            "row": {
                "name": row.name,
                "source": _source_payload(row.source),
                "trade_plan": _spec_payload(row.trade_spec),
                "allocation": asdict(row.allocation),
                "train_metrics": row.train_metrics,
                "fold_metrics": list(row.fold_metrics),
                "replay_digest": row.replay_digest,
                "score": row.score,
                "rejected": row.rejected,
                "reject_reason": row.reject_reason,
            },
        }
    )


def _read_official_audit_cache(
    cache_dir: Path | None,
    context: dict[str, Any],
    row: AllocationSweepRow,
) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    key = _official_audit_cache_key(context, row)
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("cache_key") != key:
        return None
    if payload.get("cache_version") != ALLOCATION_OFFICIAL_AUDIT_CACHE_VERSION:
        return None
    if payload.get("evidence_version") != ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION:
        return None
    audit = payload.get("audit")
    return dict(audit) if isinstance(audit, dict) else None


def _write_official_audit_cache(
    cache_dir: Path | None,
    context: dict[str, Any],
    row: AllocationSweepRow,
    audit: dict[str, Any],
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _official_audit_cache_key(context, row)
    payload = {
        "cache_key": key,
        "cache_version": ALLOCATION_OFFICIAL_AUDIT_CACHE_VERSION,
        "evidence_version": ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION,
        "context": context,
        "row_name": row.name,
        "created_at": _utc_now_iso(),
        "audit": audit,
    }
    tmp = cache_dir / f"{key}.json.tmp"
    path = cache_dir / f"{key}.json"
    tmp.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _annotate_proxy_official_ranks(rows: list[dict[str, Any]], proxy_rank_by_name: dict[str, int]) -> None:
    for sample_rank, row in enumerate(sorted(rows, key=lambda item: proxy_rank_by_name.get(str(item.get("name")), 10**9)), start=1):
        row["proxy_rank"] = int(proxy_rank_by_name.get(str(row.get("name")), 0))
        row["proxy_sample_rank"] = int(sample_rank)
    official_ranked = sorted(
        rows,
        key=lambda item: (
            -float((item.get("official_mtm_metrics") or {}).get("official_mtm_net_return_pct", 0.0) or 0.0),
            str(item.get("name") or ""),
        ),
    )
    for official_rank, row in enumerate(official_ranked, start=1):
        row["official_rank"] = int(official_rank)
        row["rank_change"] = int(official_rank) - int(row.get("proxy_sample_rank", official_rank))


def _proxy_official_rank_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_run", "count": 0}
    correlation = _spearman_rank_correlation(rows)
    max_abs_delta = max((float(row.get("official_proxy_abs_net_delta", 0.0) or 0.0) for row in rows), default=0.0)
    return {
        "status": "complete",
        "count": len(rows),
        "proxy_official_rank_correlation": correlation,
        "max_official_proxy_abs_net_delta": max_abs_delta,
        "avg_official_proxy_abs_net_delta": sum(float(row.get("official_proxy_abs_net_delta", 0.0) or 0.0) for row in rows) / max(len(rows), 1),
        "largest_rank_change": max((abs(int(row.get("rank_change", 0) or 0)) for row in rows), default=0),
    }


def _expanded_audit_count(current: int, total: int, diagnostics: dict[str, Any]) -> int:
    if current >= total:
        return current
    count = int(diagnostics.get("count", 0) or 0)
    corr = float(diagnostics.get("proxy_official_rank_correlation", 0.0) or 0.0)
    max_delta = float(diagnostics.get("max_official_proxy_abs_net_delta", 0.0) or 0.0)
    reason = ""
    if count < 3 and total >= 3:
        reason = "too_few_rank_pairs"
    elif corr < 0.35:
        reason = "weak_proxy_official_rank_correlation"
    elif max_delta > 0.05:
        reason = "wide_proxy_official_net_delta"
    if not reason:
        return current
    diagnostics["expansion_reason"] = reason
    return min(total, max(current + 4, current * 2, 3))


def _spearman_rank_correlation(rows: Sequence[dict[str, Any]]) -> float:
    pairs = [
        (int(row.get("proxy_sample_rank", 0) or 0), int(row.get("official_rank", 0) or 0))
        for row in rows
        if row.get("proxy_sample_rank") and row.get("official_rank")
    ]
    n = len(pairs)
    if n < 2:
        return 0.0
    d2 = sum((proxy_rank - official_rank) ** 2 for proxy_rank, official_rank in pairs)
    return float(1.0 - (6.0 * d2) / (n * ((n * n) - 1.0)))


def _official_allocation_metrics(row: AllocationSweepRow, compiled: CompiledExecutionSet) -> dict[str, float]:
    cfg = OLRConfig.from_mapping(compiled.dataset.config, {})
    metrics, _evidence = _official_allocation_report(row, compiled, cfg)
    return metrics


def _official_allocation_report(
    row: AllocationSweepRow,
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    mutations = {
        "olr.allocation.mode": row.allocation.mode,
        "olr.allocation.target_gross_exposure": row.allocation.target_gross_exposure,
        "olr.allocation.max_position_pct": row.allocation.max_position_pct,
        "olr.allocation.rank_decay": row.allocation.rank_decay,
        "olr.allocation.min_selected": row.allocation.min_selected,
        "olr.trade_plan.entry": asdict(row.trade_spec.entry),
        "olr.trade_plan.exit": asdict(row.trade_spec.exit),
    }
    unsupported = _official_trade_plan_reject_reason(row.trade_spec)
    if unsupported:
        return _unsupported_official_plan_metrics(unsupported), {
            "trade_plan": _spec_payload(row.trade_spec),
            "official_trade_plan_supported": False,
            "unsupported_reason": unsupported,
        }
    audit_cfg = OLRConfig.from_mapping(compiled.dataset.config, mutations)
    dates = tuple(compiled.eligible_dates)
    snapshots = {
        day: snapshot
        for day, snapshot in compiled.snapshots_by_source.get(row.source.name, {}).items()
        if day in set(dates)
    }
    snapshots = attach_overnight_labels_to_snapshots(snapshots, compiled.dataset.overnight_labels_by_key)
    selected_pairs = _official_selected_pairs(snapshots, compiled, audit_cfg)
    needed_pairs = set(selected_pairs)
    for day, symbol in selected_pairs:
        for followup_day in _official_followup_session_dates(compiled.eligible_dates, day):
            needed_pairs.add((followup_day, symbol))
    bars = [
        bar
        for (day, symbol), day_bars in compiled.dataset.bars_by_key.items()
        if (day, str(symbol).zfill(6)) in needed_pairs
        for bar in day_bars
    ]
    bar_scope_hash = stable_signature(
        {
            "selected_pairs": [(day.isoformat(), symbol) for day, symbol in sorted(selected_pairs)],
            "needed_pairs": [(day.isoformat(), symbol) for day, symbol in sorted(needed_pairs)],
            "bar_count": len(bars),
            "auction_exit_recovery_sessions": OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS,
        }
    )
    bundle = compile_olr_replay_bundle(
        bars=bars,
        snapshots=snapshots,
        source_fingerprint=stable_signature([compiled.source_fingerprint, row.name, _spec_payload(row.trade_spec), "official_mtm_audit", bar_scope_hash]),
    )
    result = run_olr_backtest({**compiled.dataset.config, "capability_level": "compiled"}, mutations, replay_bundle=bundle)
    keys = (
        "total_trades",
        "win_rate",
        "avg_r",
        "expected_total_r",
        "profit_factor",
        "expectancy",
        "net_profit",
        "net_gains",
        "net_losses",
        "gross_profit",
        "gross_loss",
        "mfe_capture",
        "net_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "official_mtm_net_return_pct",
        "official_mtm_max_drawdown_pct",
        "official_mtm_sharpe",
        "gross_exposure_avg_pct",
        "rejected_order_count",
        "same_bar_fill_count",
        "auction_order_count",
        "auction_nonfill_count",
        "open_order_count",
        "expired_order_count",
        "entry_fill_count",
        "exit_fill_count",
        "end_open_position_count",
        "forced_replay_close_count",
        "close_to_close_alpha_capture_pct",
        "official_trade_plan_supported",
    )
    metrics = {key: float(result.metrics.get(key, 0.0) or 0.0) for key in keys}
    metrics["official_trade_plan_supported"] = 1.0
    evidence = _official_audit_key_evidence(result, snapshots, audit_cfg, selected_pairs, needed_pairs, len(bars), bar_scope_hash, row.trade_spec)
    return metrics, evidence


def _official_trade_plan_reject_reason(spec: OLRTradePlanSpec) -> str:
    if spec.exit.mode == "next_close":
        return ""
    if spec.exit.mode != "managed":
        return f"official_core_unsupported_exit_mode_{spec.exit.mode}"
    return ""


def _official_followup_session_dates(eligible_dates: Sequence[date], trade_date: date) -> tuple[date, ...]:
    ordered = tuple(sorted(eligible_dates))
    try:
        index = ordered.index(trade_date)
    except ValueError:
        return ()
    stop = min(len(ordered), index + 1 + OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS)
    return ordered[index + 1 : stop]


def _unsupported_official_plan_metrics(reason: str) -> dict[str, float]:
    return {
        "total_trades": 0.0,
        "win_rate": 0.0,
        "avg_r": 0.0,
        "expected_total_r": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "net_profit": 0.0,
        "net_gains": 0.0,
        "net_losses": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "mfe_capture": 0.0,
        "net_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "official_mtm_net_return_pct": 0.0,
        "official_mtm_max_drawdown_pct": 0.0,
        "official_mtm_sharpe": 0.0,
        "gross_exposure_avg_pct": 0.0,
        "rejected_order_count": 0.0,
        "same_bar_fill_count": 0.0,
        "auction_order_count": 0.0,
        "auction_nonfill_count": 0.0,
        "open_order_count": 0.0,
        "expired_order_count": 0.0,
        "entry_fill_count": 0.0,
        "exit_fill_count": 0.0,
        "end_open_position_count": 1.0,
        "forced_replay_close_count": 0.0,
        "close_to_close_alpha_capture_pct": 0.0,
        "official_trade_plan_supported": 0.0,
    }


def _official_selected_pairs(
    snapshots: dict[date, Any],
    compiled: CompiledExecutionSet,
    cfg: OLRConfig,
) -> set[tuple[date, str]]:
    pairs: set[tuple[date, str]] = set()
    for day, snapshot in snapshots.items():
        if day not in set(compiled.eligible_dates):
            continue
        selected = tuple(candidate for candidate in snapshot.candidates[: int(cfg.overnight_slot_count)] if candidate.tradable)
        if len(selected) < int(cfg.min_selected):
            continue
        for candidate in selected:
            pairs.add((day, str(candidate.symbol).zfill(6)))
    return pairs


def _official_audit_key_evidence(
    result: Any,
    snapshots: dict[date, Any],
    cfg: OLRConfig,
    selected_pairs: set[tuple[date, str]],
    needed_pairs: set[tuple[date, str]],
    bar_count: int,
    bar_scope_hash: str,
    trade_spec: OLRTradePlanSpec,
) -> dict[str, Any]:
    selected_candidate_keys = []
    for day, snapshot in sorted(snapshots.items()):
        selected = tuple(candidate for candidate in snapshot.candidates[: int(cfg.overnight_slot_count)] if candidate.tradable)
        if len(selected) < int(cfg.min_selected):
            continue
        for candidate in selected:
            selected_candidate_keys.append(
                "|".join(
                    [
                        day.isoformat(),
                        str(candidate.symbol).zfill(6),
                        f"rank={int(candidate.rank or 0)}",
                        f"artifact={snapshot.artifact_hash}",
                    ]
                )
            )
    submitted_order_keys = [
        _decision_key(decision)
        for decision in result.decisions
        if decision.strategy_id == "OLR" and str(decision.decision_code).endswith("_SUBMITTED")
    ]
    fill_keys = [_fill_key(fill) for fill in result.replay_result.broker.fills if fill.strategy_id == "OLR"]
    rejected_order_keys = [_order_key(order) for order in result.replay_result.broker.rejected_orders if order.strategy_id == "OLR"]
    nonfill_order_keys = [
        _order_key(order)
        for order in result.replay_result.broker.expired_orders
        if order.strategy_id == "OLR" and order.order_type == "CLOSE_AUCTION"
    ]
    return {
        "version": ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION,
        "trade_plan": _spec_payload(trade_spec),
        "bar_scope": "selected_trade_date_and_train_followup_symbol_bars",
        "bar_scope_hash": bar_scope_hash,
        "replay_bar_count": int(bar_count),
        "selected_pair_count": len(selected_pairs),
        "needed_pair_count": len(needed_pairs),
        "auction_exit_recovery_sessions": OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS,
        "selected_candidate_keys": _key_summary(selected_candidate_keys),
        "submitted_order_keys": _key_summary(submitted_order_keys),
        "fill_keys": _key_summary(fill_keys),
        "rejected_order_keys": _key_summary(rejected_order_keys),
        "nonfill_order_keys": _key_summary(nonfill_order_keys),
        "open_order_keys": _key_summary(_order_key(order) for order in result.replay_result.broker.orders if order.strategy_id == "OLR"),
        "open_position_keys": _key_summary(_position_key(position) for position in result.replay_result.broker.positions.values() if position.strategy_id == "OLR"),
    }


def _key_summary(keys: Sequence[str]) -> dict[str, Any]:
    ordered = sorted(str(key) for key in keys)
    return {
        "count": len(ordered),
        "hash": stable_signature(ordered),
        "keys": ordered,
    }


def _decision_key(decision: Any) -> str:
    meta = dict(getattr(decision, "metadata", {}) or {})
    parts = [
        str(getattr(decision, "timestamp", "")).replace("+09:00", ""),
        str(getattr(decision, "decision_code", "")),
        str(getattr(decision, "symbol", "")).zfill(6),
        str(getattr(decision, "reason", "")),
        f"qty={int(meta.get('qty', 0) or 0)}",
        f"rank={int(meta.get('candidate_rank', 0) or 0)}",
        f"artifact={meta.get('source_artifact_hash', '')}",
    ]
    return "|".join(parts)


def _fill_key(fill: Any) -> str:
    meta = dict(getattr(fill, "metadata", {}) or {})
    return "|".join(
        [
            str(getattr(fill, "timestamp", "")).replace("+09:00", ""),
            str(getattr(fill, "side", "")),
            str(getattr(fill, "symbol", "")).zfill(6),
            str(getattr(fill, "reason", "")),
            f"qty={int(getattr(fill, 'qty', 0) or 0)}",
            f"price={float(getattr(fill, 'price', 0.0) or 0.0):.6f}",
            f"artifact={meta.get('source_artifact_hash', '')}",
            f"nonfill_key={meta.get('auction_nonfill_key', '')}",
        ]
    )


def _order_key(order: Any) -> str:
    meta = dict(getattr(order, "metadata", {}) or {})
    return "|".join(
        [
            str(getattr(order, "submitted_at", "")).replace("+09:00", ""),
            str(getattr(order, "side", "")),
            str(getattr(order, "order_type", "")),
            str(getattr(order, "symbol", "")).zfill(6),
            str(getattr(order, "reason", "")),
            f"qty={int(getattr(order, 'qty', 0) or 0)}",
            f"artifact={meta.get('source_artifact_hash', '')}",
            f"nonfill_key={meta.get('auction_nonfill_key', '')}",
        ]
    )


def _position_key(position: Any) -> str:
    return "|".join(
        [
            str(getattr(position, "symbol", "")).zfill(6),
            f"qty={int(getattr(position, 'qty', 0) or 0)}",
            f"avg={float(getattr(position, 'avg_price', 0.0) or 0.0):.6f}",
            f"entry={str(getattr(position, 'entry_fill_time', '')).replace('+09:00', '')}",
        ]
    )


def _reference_metrics(payload: dict[str, Any]) -> dict[str, float]:
    rows = list(payload.get("top_train") or [])
    if rows:
        metrics = rows[0].get("train_metrics") or {}
        return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    return {}


def _row_sort_key(row: AllocationSweepRow) -> tuple[Any, ...]:
    metrics = row.train_metrics
    worst_fold = min(
        (
            float(fold["metrics"].get("portfolio_proxy_net_return_pct", fold["metrics"].get("allocation_cumulative_net_return_pct", 0.0)) or 0.0)
            for fold in row.fold_metrics
        ),
        default=float(metrics.get("portfolio_proxy_net_return_pct", metrics.get("allocation_cumulative_net_return_pct", 0.0)) or 0.0),
    )
    return (
        row.rejected,
        -float(metrics.get("portfolio_proxy_net_return_pct", metrics.get("allocation_cumulative_net_return_pct", 0.0)) or 0.0),
        -worst_fold,
        abs(float(metrics.get("portfolio_proxy_max_drawdown_pct", metrics.get("allocation_max_drawdown_net_pct", 0.0)) or 0.0)),
        -float(metrics.get("portfolio_proxy_active_day_net_pct", metrics.get("selected_day_net_pct", 0.0)) or 0.0),
        row.name,
    )


def _row_payload(
    row: AllocationSweepRow,
    official_mtm_metrics: dict[str, Any] | None = None,
    official_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "name": row.name,
        "score": row.score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "beats_reference": row.beats_reference,
        "source": _source_payload(row.source),
        "trade_plan": _spec_payload(row.trade_spec),
        "allocation": asdict(row.allocation),
        "train_metrics": row.train_metrics,
        "train_metric_basis": "legacy_slot_research_labels_non_official",
        "fold_metrics": list(row.fold_metrics),
        "fast_replay_digest": row.replay_digest,
    }
    if official_mtm_metrics:
        payload["official_mtm_metrics"] = dict(official_mtm_metrics)
        payload["official_metric_basis"] = "OLRReplayAdapter -> OLR core -> neutral actions -> SimBroker.equity_curve_bar_level_mtm"
    if official_audit:
        for key in (
            "audit_pass",
            "proxy_rank",
            "proxy_sample_rank",
            "official_rank",
            "rank_change",
            "official_proxy_net_delta",
            "official_proxy_abs_net_delta",
            "official_audit_keys",
        ):
            if key in official_audit:
                payload[key] = official_audit[key]
    return payload


def _official_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = dict(row.get("official_mtm_metrics") or {})
    return (
        bool(row.get("rejected")),
        float(metrics.get("same_bar_fill_count", 0.0) or 0.0) > 0.0,
        -float(metrics.get("official_mtm_net_return_pct", 0.0) or 0.0),
        abs(float(metrics.get("official_mtm_max_drawdown_pct", 0.0) or 0.0)),
        -float(metrics.get("official_mtm_sharpe", 0.0) or 0.0),
        row.get("name", ""),
    )


def _audit_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_run"}
    return {
        "status": "pass" if all(bool(row.get("audit_pass")) for row in rows) else "fail",
        "count": len(rows),
        "max_abs_metric_delta": max((float(row.get("max_abs_metric_delta", 0.0) or 0.0) for row in rows), default=0.0),
        "max_abs_score_delta": max((abs(float(row.get("score_delta", 0.0) or 0.0)) for row in rows), default=0.0),
        "max_official_proxy_abs_net_delta": max((float(row.get("official_proxy_abs_net_delta", 0.0) or 0.0) for row in rows), default=0.0),
        "proxy_official_rank_correlation": _spearman_rank_correlation(rows),
        "outcome_hash_mismatches": sum(1 for row in rows if not row.get("outcome_hash_match")),
    }


def _clear_progress_files(output_dir: Path) -> None:
    for name in ("run_status.json", "progress.jsonl", "progress_allocation_sources.json", "progress_official_audit.json"):
        try:
            (output_dir / name).unlink()
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
    print(f"[olr-allocation-sweep] status {stage} {details}".rstrip(), flush=True)


def _record_source_progress(
    output_dir: Path,
    completed: int,
    total: int,
    rows: Sequence[AllocationSweepRow],
    source: CandidateSource,
    source_rows: Sequence[AllocationSweepRow],
) -> None:
    if completed not in {1, 2, 3, 5, 10, total} and completed % 25 != 0:
        return
    cache_hits = sum(1 for row in rows if bool(row.replay_digest.get("portfolio_proxy_cache_hit")))
    cache_misses = sum(1 for row in rows if row.replay_digest.get("portfolio_proxy_cache_hit") is False)
    ranked = sorted(rows, key=_row_sort_key)
    best = ranked[0] if ranked else None
    payload = {
        "updated_at": _utc_now_iso(),
        "stage": "evaluating_portfolio_proxy",
        "completed_sources": int(completed),
        "total_sources": int(total),
        "percent": round(100.0 * completed / total, 3) if total else 100.0,
        "rows_completed": len(rows),
        "cache_hit_rows": cache_hits,
        "cache_miss_rows": cache_misses,
        "latest_source": source.name,
        "latest_source_rows": len(source_rows),
        "best_proxy_so_far": _progress_row(best) if best is not None else None,
        "top_proxy": [_progress_row(row) for row in ranked[:10]],
    }
    tmp = output_dir / "progress_allocation_sources.json.tmp"
    path = output_dir / "progress_allocation_sources.json"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "allocation_source_progress", **payload}, sort_keys=True, default=str) + "\n")
    best_name = payload["best_proxy_so_far"]["name"] if payload["best_proxy_so_far"] else ""
    best_return = payload["best_proxy_so_far"]["portfolio_proxy_net_return_pct"] if payload["best_proxy_so_far"] else 0.0
    print(
        "[olr-allocation-sweep] "
        f"sources {completed}/{total} rows={len(rows)} cache_hits={cache_hits} "
        f"best={best_name} proxy={100.0 * float(best_return):.1f}%",
        flush=True,
    )


def _write_official_audit_progress(
    output_dir: Path,
    stage: str,
    *,
    completed: int,
    total: int,
    cache_hits: int,
    latest: str,
    audits: Sequence[dict[str, Any]],
    max_workers: int,
) -> None:
    ranked = sorted(audits, key=_official_row_sort_key)
    payload = {
        "updated_at": _utc_now_iso(),
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "percent": round(100.0 * completed / total, 3) if total else 100.0,
        "cache_hits": int(cache_hits),
        "cache_misses": int(max(0, len(audits) - cache_hits)),
        "latest": latest,
        "max_workers": max(1, min(int(max_workers), 2)),
        "effective_workers": 1,
        "top_official_so_far": [_audit_progress_row(row) for row in ranked[:10]],
    }
    tmp = output_dir / "progress_official_audit.json.tmp"
    path = output_dir / "progress_official_audit.json"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "official_audit_progress", **payload}, sort_keys=True, default=str) + "\n")
    best = payload["top_official_so_far"][0] if payload["top_official_so_far"] else {}
    print(
        "[olr-allocation-sweep] "
        f"official_audit {completed}/{total} cache_hits={cache_hits} "
        f"latest={latest} best={best.get('name', '')} "
        f"net={100.0 * float(best.get('official_mtm_net_return_pct', 0.0) or 0.0):.1f}%",
        flush=True,
    )


def _audit_progress_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("official_mtm_metrics") or {})
    return {
        "name": row.get("name", ""),
        "audit_pass": bool(row.get("audit_pass")),
        "official_mtm_net_return_pct": float(metrics.get("official_mtm_net_return_pct", 0.0) or 0.0),
        "official_mtm_max_drawdown_pct": float(metrics.get("official_mtm_max_drawdown_pct", 0.0) or 0.0),
        "official_mtm_sharpe": float(metrics.get("official_mtm_sharpe", 0.0) or 0.0),
        "portfolio_proxy_net_return_pct": float(row.get("portfolio_proxy_net_return_pct", 0.0) or 0.0),
        "official_proxy_net_delta": float(row.get("official_proxy_net_delta", 0.0) or 0.0),
        "same_bar_fill_count": float(metrics.get("same_bar_fill_count", 0.0) or 0.0),
    }


def _progress_row(row: AllocationSweepRow) -> dict[str, Any]:
    metrics = row.train_metrics
    return {
        "name": row.name,
        "score": row.score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "portfolio_proxy_net_return_pct": metrics.get("portfolio_proxy_net_return_pct", 0.0),
        "portfolio_proxy_max_drawdown_pct": metrics.get("portfolio_proxy_max_drawdown_pct", 0.0),
        "portfolio_proxy_avg_active_gross_exposure_pct": metrics.get("portfolio_proxy_avg_active_gross_exposure_pct", 0.0),
        "trade_count": metrics.get("trade_count", 0.0),
        "source": row.source.name,
        "allocation": row.allocation.name,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OLR Selection Allocation Sweep",
        "",
        f"- Sweep hash: `{payload.get('sweep_hash')}`",
        f"- Train: {payload['training_window']['start']} to {payload['training_window']['end']} ({payload['training_window']['sessions']} sessions)",
        f"- Official performance: `{payload.get('official_performance')}`",
        f"- Fast/full audit: `{payload.get('fast_full_audit', {}).get('status', 'not_run')}`",
        f"- Proxy/official rank correlation: {float((payload.get('proxy_official_rank_diagnostics') or {}).get('proxy_official_rank_correlation', 0.0) or 0.0):.3f}",
        "",
        "## Audited Official-MTM Finalists",
        "",
        "| Rank | Official MTM | Max DD | Sharpe | Same-Bar | Source | Allocation |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(payload.get("top_official_train", [])[:20], start=1):
        metrics = row.get("official_mtm_metrics") or {}
        lines.append(
            f"| {rank} | "
            f"{100.0 * metrics.get('official_mtm_net_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('official_mtm_max_drawdown_pct', 0.0):.1f}% | "
            f"{metrics.get('official_mtm_sharpe', 0.0):.2f} | "
            f"{metrics.get('same_bar_fill_count', 0.0):.0f} | "
            f"{row.get('source', {}).get('stage2_name')} | {row.get('allocation', {}).get('name')} |"
        )
    lines.extend(
        [
            "",
            "## Fast Legacy-Label Pruning Frontier",
            "",
        "| Rank | Beats Ref | Portfolio Proxy | Fractional Alloc | Selected-Day | Gross Exposure | Max DD | Trades | Source | Allocation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for rank, row in enumerate(payload.get("top_train", [])[:40], start=1):
        metrics = row.get("train_metrics") or {}
        lines.append(
            f"| {rank} | {int(bool(row.get('beats_reference')))} | "
            f"{100.0 * metrics.get('portfolio_proxy_net_return_pct', metrics.get('allocation_cumulative_net_return_pct', 0.0)):.1f}% | "
            f"{100.0 * metrics.get('slot_cumulative_net_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('portfolio_proxy_active_day_net_pct', metrics.get('selected_day_net_pct', 0.0)):.3f}% | "
            f"{100.0 * metrics.get('portfolio_proxy_avg_gross_exposure_pct', metrics.get('allocation_avg_gross_exposure', 0.0)):.1f}% | "
            f"{100.0 * metrics.get('portfolio_proxy_max_drawdown_pct', metrics.get('allocation_max_drawdown_net_pct', 0.0)):.1f}% | "
            f"{metrics.get('trade_count', 0.0):.0f} | "
            f"{row.get('source', {}).get('stage2_name')} | {row.get('allocation', {}).get('name')} |"
        )
    return "\n".join(lines) + "\n"


def _phase_seed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    official_rows = payload.get("top_official_train") or []
    best = (official_rows or payload.get("top_train") or [{}])[0]
    metric_source = "audited_official_mtm" if official_rows else "proxy_fallback_no_official_audit"
    return {
        "strategy": "olr",
        "artifact_promotion_policy": "training_only_until_holdout_and_paper_parity",
        "official_performance": False,
        "research_sweep_path": payload.get("research_sweep_path"),
        "allocation_sweep_path": (payload.get("artifact_paths") or {}).get("json"),
        "phase_auto_seed": {
            "candidate_source": (best.get("source") or {}).get("name"),
            "candidate_source_mutations": (best.get("source") or {}).get("mutations"),
            "entry": (best.get("trade_plan") or {}).get("entry") or asdict(_close_auction_entry()),
            "exit": (best.get("trade_plan") or {}).get("exit") or asdict(_next_close_exit()),
            "trade_plan": best.get("trade_plan"),
            "allocation": best.get("allocation"),
            "official_mtm_metrics": best.get("official_mtm_metrics"),
            "legacy_train_metrics": best.get("train_metrics"),
            "metric_source": metric_source,
            "policy": "Training-only seed; prefer audited official-MTM finalists and evaluate OOS/paper before promotion.",
        },
    }


def _allocation_name(mode: str, cap: float) -> str:
    return f"{mode}_cap{_label(cap)}"


def _label(value: Any) -> str:
    text = f"{float(value):.6g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep OLR close-to-close candidate breadth and allocation plans.")
    parser.add_argument("--config", default="config/optimization/olr.yaml")
    parser.add_argument("--research-sweep-path", required=True)
    parser.add_argument("--trade-plan-sweep-path", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--fold-count", type=int, default=2)
    parser.add_argument("--finalist-count", type=int, default=40)
    parser.add_argument("--audit-finalist-count", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = normalize_runtime_config("olr", load_yaml_config(args.config))
    payload = run_allocation_sweep(
        config,
        research_sweep_path=args.research_sweep_path,
        trade_plan_sweep_path=args.trade_plan_sweep_path or None,
        output_dir=args.output_dir,
        holdout_days=args.holdout_days,
        max_sources=args.max_sources,
        fold_count=args.fold_count,
        finalist_count=args.finalist_count,
        audit_finalist_count=args.audit_finalist_count,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(
            json.dumps(
                {
                    "strategy": "olr",
                    "sweep_hash": payload["sweep_hash"],
                    "artifact_paths": payload["artifact_paths"],
                    "top_train": payload["top_train"][:5],
                    "audit_pass": payload["audit_pass"],
                },
                indent=2,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
