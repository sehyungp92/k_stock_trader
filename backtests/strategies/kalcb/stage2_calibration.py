from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from strategy_kalcb.config import KALCBConfig, KALCB_CORE_VERSION

from .premarket_frontier_sweep import _read_json_section_rows
from .kalcb_path_quality_v1 import (
    INTERACTION_REGIME_MODEL_VERSION,
    PATH_CALIBRATION_SCORE_VERSION,
    PATH_QUALITY_USAGE_CONTRACT,
    PATH_QUALITY_MODEL_VERSION,
    build_path_quality_observations,
    fit_interaction_regime_model,
    fit_path_quality_model,
    fold_path_risk_metrics,
    score_path_calibrated_row,
    summarize_path_risk,
)
from .trade_plan_sweep import (
    FixedCandidateSource,
    PRIMARY_OBJECTIVE_METRIC,
    PlanResult,
    _build_contexts,
    _audit_replay_row,
    _core_outcomes_metrics_digest,
    _first30_from_payload,
    _fold_metrics_from_outcomes_for_dates,
    _frontier_scores_by_day,
    _frontier_from_payload,
    _portfolio_risk_mutations,
    _resolve_optimized_source_path,
    _resolve_folds,
    _selection_counts,
    _training_only_config,
    baseline_trade_plan_spec,
    build_fixed_candidate_selections,
    compile_core_replay,
    prepare_first30_dataset,
)


DEFAULT_CALIBRATION_SECTION = "top_portfolio_proxy"
STAGE2_CALIBRATION_VERSION = "kalcb-stage2-core-calibration-v4"
_CALIBRATION_FOLD_COUNT = 2
PATH_CALIBRATED_STAGE2_SECTION = "top_path_calibrated_stage2"


@dataclass(frozen=True, slots=True)
class _CalibrationBase:
    training_config: dict[str, Any]
    cfg: KALCBConfig
    dataset: Any
    contexts: dict[date, tuple[Any, ...]]
    context_by_key: dict[tuple[date, str], Any]
    train_dates: tuple[date, ...]


def run_stage2_core_calibration(
    config: dict[str, Any],
    *,
    stage2_artifact: str | Path,
    output_dir: str | Path,
    candidate_section: str = DEFAULT_CALIBRATION_SECTION,
    candidate_limit: int = 20,
    finalist_count: int = 5,
    max_workers: int = 2,
    compiled_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    del compiled_cache_dir  # The calibration path builds one shared in-process base, then compiles official replays per row.
    worker_count = max(1, int(max_workers or 1))
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage2_path = _resolve_optimized_source_path(Path(stage2_artifact))
    source_hash = _file_sha256(stage2_path)
    stage2_payload = json.loads(stage2_path.read_text(encoding="utf-8"))
    source_rows = _read_json_section_rows(stage2_path, candidate_section, max(1, int(candidate_limit)))
    cfg = KALCBConfig.from_mapping(config, {})
    baseline_spec = baseline_trade_plan_spec()
    calibration_input_hash = _calibration_input_hash(config, stage2_payload)
    cache_root = out / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    uncached: list[tuple[int, dict[str, Any], str, Path]] = []
    total_rows = len(source_rows)
    for rank, source_row in enumerate(source_rows):
        _write_progress(out, "fast_calibration_row_running", completed=len(rows), total=total_rows, source_rank=rank, source_name=str(source_row.get("name") or ""))
        cache_key = _row_cache_key(source_hash, calibration_input_hash, source_row, candidate_section, rank, cfg, baseline_spec)
        cache_path = _row_cache_path(cache_root, cache_key)
        if cache_path.exists():
            rows.append(json.loads(cache_path.read_text(encoding="utf-8")))
            _write_progress(out, "fast_calibration_row_cache_hit", completed=len(rows), total=total_rows, source_rank=rank, source_name=str(source_row.get("name") or ""))
        else:
            uncached.append((rank, source_row, cache_key, cache_path))

    base: _CalibrationBase | None = None
    if uncached:
        _write_progress(out, "calibration_base_building", completed=0, total=len(uncached), uncached_rows=len(uncached))
        base = _prepare_calibration_base(config)
        _write_progress(
            out,
            "calibration_base_built",
            sessions=len(base.train_dates),
            contexts=len(base.context_by_key),
            uncached_rows=len(uncached),
        )
        if worker_count == 1 or len(uncached) == 1:
            for rank, source_row, cache_key, cache_path in uncached:
                payload = _calibrate_row(
                    base,
                    stage2_path=stage2_path,
                    stage2_payload=stage2_payload,
                    stage2_source_hash=source_hash,
                    source_row=source_row,
                    candidate_section=candidate_section,
                    rank=rank,
                    cache_key=cache_key,
                    cache_path=cache_path,
                    baseline_spec=baseline_spec,
                )
                rows.append(payload)
                _write_progress(out, "fast_calibration_row_completed", completed=len(rows), total=total_rows, source_rank=rank, source_name=str(source_row.get("name") or ""))
        else:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(uncached)), thread_name_prefix="kalcb-stage2-cal") as executor:
                futures = {
                    executor.submit(
                        _calibrate_row,
                        base,
                        stage2_path=stage2_path,
                        stage2_payload=stage2_payload,
                        stage2_source_hash=source_hash,
                        source_row=source_row,
                        candidate_section=candidate_section,
                        rank=rank,
                        cache_key=cache_key,
                        cache_path=cache_path,
                        baseline_spec=baseline_spec,
                    ): (rank, source_row)
                    for rank, source_row, cache_key, cache_path in uncached
                }
                for future in as_completed(futures):
                    rank, source_row = futures[future]
                    rows.append(future.result())
                    _write_progress(
                        out,
                        "fast_calibration_row_completed",
                        completed=len(rows),
                        total=total_rows,
                        source_rank=rank,
                        source_name=str(source_row.get("name") or ""),
                        max_workers=worker_count,
                    )
    rows.sort(key=lambda row: int(row.get("source_rank", 0)))

    provisional = select_calibrated_stage2_rows(rows, finalist_count=finalist_count)
    _write_progress(out, "audit_calibration_running", completed=0, total=len(provisional), finalist_count=len(provisional))
    if provisional and base is None:
        _write_progress(out, "calibration_base_building_for_audit", completed=0, total=len(provisional), finalist_count=len(provisional))
        base = _prepare_calibration_base(config)
        _write_progress(out, "calibration_base_built_for_audit", sessions=len(base.train_dates), contexts=len(base.context_by_key))
    audited = _audit_selected_rows(base, stage2_path, stage2_payload, source_hash, provisional, candidate_section, out, baseline_spec) if base else []
    _write_progress(out, "audit_calibration_completed", completed=len(audited), total=len(provisional), finalist_count=len(provisional))
    audited_by_rank = {int(row["source_rank"]): row for row in audited}
    merged_rows = [_merge_audit(row, audited_by_rank.get(int(row["source_rank"]))) for row in rows]
    top_calibrated_source_rows = select_calibrated_stage2_rows(merged_rows, finalist_count=finalist_count, require_audit_pass=True)
    fast_only_source_rows = select_calibrated_stage2_rows(merged_rows, finalist_count=finalist_count, require_audit_pass=False)
    top_path_calibrated = _as_path_calibrated_source_rows(top_calibrated_source_rows)
    fast_only_calibrated = _as_path_calibrated_source_rows(fast_only_source_rows)

    payload = {
        "strategy": "kalcb",
        "calibration_version": STAGE2_CALIBRATION_VERSION,
        "stage_contract": "stage2_shared_core_candidate_pool_calibration",
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stage2_artifact": str(stage2_path),
        "stage2_artifact_hash": source_hash,
        "candidate_artifact_hash": source_hash,
        "calibration_input_hash": calibration_input_hash,
        "candidate_section": candidate_section,
        "candidate_limit": int(candidate_limit),
        "finalist_count": int(finalist_count),
        "max_workers": worker_count,
        "shared_decision_core": "live_shared_core",
        "strategy_core_version": KALCB_CORE_VERSION,
        "research_only": False,
        "source_fingerprints": dict(stage2_payload.get("source_fingerprints") or {}),
        "causality_policy": dict(stage2_payload.get("causality_policy") or {}),
        "calibration_plan": {
            "entry": "first30_open",
            "exit": "EOD flatten",
            "trade_management": "baseline_no_tuned_exit",
            "candidate_pool": "fixed active first30 selections plus frontier-shadow candidates, matching full prepared-context replay",
            "spec": {"name": baseline_spec.name, "entry": asdict(baseline_spec.entry), "exit": asdict(baseline_spec.exit)},
        },
        "metric_contract": {
            "primary_calibration_metric": PRIMARY_OBJECTIVE_METRIC,
            "official_performance": "baseline candidate-pool calibration only; Stage 3 still performs full trade-plan optimization",
            "calibration_universe": "compiled shared-core replay candidate pool including frontier shadows",
            "path_quality_usage_contract": PATH_QUALITY_USAGE_CONTRACT,
            "score_version": PATH_CALIBRATION_SCORE_VERSION,
            "score_components": [
                "broker_net_return_pct",
                "worst_fold_net",
                "avg_mfe_capture",
                "trade_count_frequency",
                "broker_max_drawdown_pct",
                "mae_tail_loss",
                "giveback_loss",
            ],
            "proxy_metrics": ["carried_for_audit_only"],
            "promotion_requires_audit_pass": True,
        },
        "cost_policy": {
            "slippage_bps_each_side": cfg.slippage_bps,
            "commission_bps_each_side": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
        },
        "portfolio_risk_policy": _portfolio_risk_mutations(cfg),
        "rows": merged_rows,
        "audit_replays": audited,
        "top_fast_calibrated_stage2": fast_only_calibrated,
        "top_path_calibrated_stage2": top_path_calibrated,
        "top_calibrated_stage2": top_path_calibrated,
    }
    payload["calibration_hash"] = stable_signature(
        {
            "stage2_artifact_hash": source_hash,
            "calibration_version": STAGE2_CALIBRATION_VERSION,
            "calibration_input_hash": calibration_input_hash,
            "core_version": KALCB_CORE_VERSION,
            "score_version": PATH_CALIBRATION_SCORE_VERSION,
            "top_calibrated": top_path_calibrated,
            "baseline_spec": payload["calibration_plan"]["spec"],
        }
    )
    json_path = out / f"kalcb_stage2_core_calibration_{payload['calibration_hash'][:12]}.json"
    md_path = out / f"kalcb_stage2_core_calibration_{payload['calibration_hash'][:12]}.md"
    _attach_calibrated_source_refs(payload, json_path)
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def select_calibrated_stage2_rows(
    rows: list[dict[str, Any]],
    *,
    finalist_count: int = 5,
    require_audit_pass: bool = False,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if (not require_audit_pass or bool(row.get("audit_pass")))
        and not str(row.get("reject_reason") or "")
    ]
    if not eligible:
        return []
    primary = sorted(eligible, key=_calibration_sort_key)[: min(3, max(1, int(finalist_count)))]
    selected = {int(row["source_rank"]) for row in primary}
    diversity_pool = [
        row
        for row in eligible
        if int(row["source_rank"]) not in selected
        and float(row.get("calibrated_broker_net_return_pct", 0.0) or 0.0) >= 0.0
        and float(row.get("filled_selected_rate", 0.0) or 0.0) >= 0.15
    ]
    diversity_pool.sort(
        key=lambda row: (
            -float(row.get("path_calibrated_score", row.get("calibrated_broker_net_return_pct", 0.0)) or 0.0),
            -float((row.get("path_risk_metrics") or {}).get("avg_mfe_capture", 0.0) or 0.0),
            float((row.get("path_risk_metrics") or {}).get("avg_giveback_r", 999.0) or 999.0),
            str(row.get("name") or ""),
        )
    )
    out = [*primary]
    for row in diversity_pool:
        if len(out) >= max(1, int(finalist_count)):
            break
        if int(row["source_rank"]) in selected:
            continue
        out.append(row)
        selected.add(int(row["source_rank"]))
    return out[: max(1, int(finalist_count))]


def _as_path_calibrated_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["original_source_section"] = str(row.get("source_section") or DEFAULT_CALIBRATION_SECTION)
        item["original_source_rank"] = int(row.get("source_rank", index) or 0)
        item["original_source_name"] = str(row.get("name") or "")
        item["source_section"] = PATH_CALIBRATED_STAGE2_SECTION
        item["source_rank"] = int(index)
        item["calibrated_source_section"] = PATH_CALIBRATED_STAGE2_SECTION
        item["calibrated_source_rank"] = int(index)
        out.append(item)
    return out


def _attach_calibrated_source_refs(payload: dict[str, Any], json_path: Path) -> None:
    for section in ("top_path_calibrated_stage2", "top_calibrated_stage2", "top_fast_calibrated_stage2"):
        rows = list(payload.get(section) or [])
        for index, row in enumerate(rows):
            row["calibrated_source_path"] = str(json_path)
            row["calibrated_source_section"] = PATH_CALIBRATED_STAGE2_SECTION
            row["calibrated_source_rank"] = int(index)
        payload[section] = rows


def _calibrate_row(
    base: _CalibrationBase,
    *,
    stage2_path: Path,
    stage2_payload: dict[str, Any],
    stage2_source_hash: str,
    source_row: dict[str, Any],
    candidate_section: str,
    rank: int,
    cache_key: str,
    cache_path: Path,
    baseline_spec: Any,
) -> dict[str, Any]:
    candidate_source = _candidate_source_from_row(stage2_path, stage2_payload, stage2_source_hash, source_row, candidate_section, rank)
    selections, frontier = build_fixed_candidate_selections(candidate_source, base.dataset, base.contexts)
    frontier_scores = _frontier_scores_by_day(candidate_source, base.contexts)
    selection_counts = _selection_counts(selections, base.train_dates)
    compiled = compile_core_replay(
        selections,
        base.dataset,
        base.context_by_key,
        base.train_dates,
        selection_counts,
        base.cfg,
        frontier_by_day=frontier,
        frontier_scores_by_day=frontier_scores,
        source_calibration_metadata=candidate_source.calibration_metadata,
    )
    outcomes, metrics, digest = _core_outcomes_metrics_digest(
        baseline_spec,
        compiled,
        base.cfg,
        base.train_dates,
        selection_counts,
        audit=False,
    )
    folds = _resolve_folds(base.train_dates, _CALIBRATION_FOLD_COUNT)
    fold_rows = _fold_metrics_from_outcomes_for_dates(
        outcomes,
        base.train_dates,
        folds,
        selection_counts,
        initial_equity=compiled.initial_equity,
    )
    observations = build_path_quality_observations(
        outcomes,
        compiled,
        base.context_by_key,
        base.dataset.bars_by_key,
    )
    path_risk_metrics = summarize_path_risk(observations, selected_count=float(metrics.get("selected_count", 0.0) or 0.0))
    fold_path_metrics = fold_path_risk_metrics(observations, folds)
    path_quality_model = fit_path_quality_model(observations, folds)
    interaction_regime_model = fit_interaction_regime_model(observations, folds)
    payload = _calibration_payload(
        source_row,
        rank=rank,
        candidate_section=candidate_section,
        cache_key=cache_key,
        metrics=metrics,
        digest=digest,
        fold_rows=fold_rows,
        path_risk_metrics=path_risk_metrics,
        fold_path_metrics=fold_path_metrics,
        path_quality_model=path_quality_model,
        interaction_regime_model=interaction_regime_model,
        audit=None,
    )
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _audit_selected_rows(
    base: _CalibrationBase,
    stage2_path: Path,
    stage2_payload: dict[str, Any],
    stage2_source_hash: str,
    rows: list[dict[str, Any]],
    candidate_section: str,
    output_dir: Path,
    baseline_spec: Any,
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for row in rows:
        rank = int(row["source_rank"])
        source_row = dict(row.get("source_row") or {})
        if not source_row:
            source_row = {"name": row.get("name", ""), "frontier": row.get("frontier", {}), "first30": row.get("first30", {})}
        candidate_source = _candidate_source_from_row(stage2_path, stage2_payload, stage2_source_hash, source_row, candidate_section, rank)
        selections, frontier = build_fixed_candidate_selections(candidate_source, base.dataset, base.contexts)
        frontier_scores = _frontier_scores_by_day(candidate_source, base.contexts)
        selection_counts = _selection_counts(selections, base.train_dates)
        compiled = compile_core_replay(
            selections,
            base.dataset,
            base.context_by_key,
            base.train_dates,
            selection_counts,
            base.cfg,
            frontier_by_day=frontier,
            frontier_scores_by_day=frontier_scores,
            source_calibration_metadata=candidate_source.calibration_metadata,
        )
        plan_row = PlanResult(
            spec=baseline_spec,
            score=0.0,
            rejected=False,
            reject_reason="",
            train_metrics=dict(row.get("calibration_metrics") or {}),
            fold_metrics=(),
            promotion_pass=True,
            replay_digest=dict(row.get("fast_replay_digest") or {}),
        )
        audit = _audit_replay_row(plan_row, compiled, base.cfg, base.train_dates, selection_counts)
        audited.append({"source_rank": rank, "source_name": row.get("name", ""), **audit})
        _write_progress(output_dir, "audit_calibration_row_completed", completed=len(audited), total=len(rows), source_rank=rank, source_name=str(row.get("name", "")))
    audited.sort(key=lambda item: int(item.get("source_rank", 0)))
    return audited


def _calibration_payload(
    source_row: dict[str, Any],
    *,
    rank: int,
    candidate_section: str,
    cache_key: str,
    metrics: dict[str, float],
    digest: dict[str, Any],
    fold_rows: tuple[dict[str, Any], ...],
    path_risk_metrics: dict[str, float],
    fold_path_metrics: tuple[dict[str, Any], ...],
    path_quality_model: dict[str, Any],
    interaction_regime_model: dict[str, Any],
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    proxy_metrics = dict(source_row.get("metrics") or {})
    broker_net = float(metrics.get(PRIMARY_OBJECTIVE_METRIC, 0.0) or 0.0)
    proxy_net = float(proxy_metrics.get("portfolio_proxy_net_return_pct", proxy_metrics.get("slot_cumulative_net_return_pct", 0.0)) or 0.0)
    selected = float(metrics.get("selected_count", 0.0) or 0.0)
    trades = float(metrics.get("trade_count", 0.0) or 0.0)
    audit_pass = bool(audit.get("audit_pass")) if audit else False
    path_score, path_components = score_path_calibrated_row(metrics, path_risk_metrics, fold_rows)
    return {
        "name": str(source_row.get("name") or f"{candidate_section}_{rank}"),
        "source_section": candidate_section,
        "source_rank": int(rank),
        "frontier": dict(source_row.get("frontier") or {}),
        "first30": dict(source_row.get("first30") or {}),
        "source_row": {
            "name": str(source_row.get("name") or f"{candidate_section}_{rank}"),
            "frontier": dict(source_row.get("frontier") or {}),
            "first30": dict(source_row.get("first30") or {}),
        },
        "proxy_metrics": proxy_metrics,
        "proxy_net_return_pct": proxy_net,
        "calibrated_broker_net_return_pct": broker_net,
        "calibrated_official_mtm_net_return_pct": float(metrics.get("official_mtm_net_return_pct", broker_net) or 0.0),
        "calibrated_broker_max_drawdown_pct": float(metrics.get("broker_max_drawdown_pct", 0.0) or 0.0),
        "trade_count": trades,
        "selected_count": selected,
        "signal_conversion": float(metrics.get("signal_conversion", 0.0) or 0.0),
        "filled_selected_rate": trades / max(selected, 1.0),
        "same_bar_fill_count": float(metrics.get("same_bar_fill_count", 0.0) or 0.0),
        "proxy_minus_broker_net_return_pct": proxy_net - broker_net,
        "calibration_metrics": dict(metrics),
        "calibration_fold_metrics": tuple(fold_rows),
        "path_calibration_score_version": PATH_CALIBRATION_SCORE_VERSION,
        "path_calibrated_score": float(path_score),
        "path_score_components": dict(path_components),
        "path_risk_metrics": dict(path_risk_metrics),
        "fold_path_risk_metrics": tuple(fold_path_metrics),
        "path_quality_model": dict(path_quality_model),
        "interaction_regime_model_version": INTERACTION_REGIME_MODEL_VERSION,
        "interaction_regime_model": dict(interaction_regime_model),
        "fast_replay_digest": dict(digest),
        "audit_pass": audit_pass,
        "audit_status": "pass" if audit_pass else "not_run",
        "audit": audit or {},
        "cache_key": cache_key,
        "reject_reason": "",
    }


def _merge_audit(row: dict[str, Any], audit: dict[str, Any] | None) -> dict[str, Any]:
    if not audit:
        return row
    merged = dict(row)
    audit_metrics = dict(audit.get("audit_metrics") or {})
    if audit_metrics:
        merged["calibrated_broker_net_return_pct"] = float(audit_metrics.get(PRIMARY_OBJECTIVE_METRIC, merged.get("calibrated_broker_net_return_pct", 0.0)) or 0.0)
        merged["calibrated_official_mtm_net_return_pct"] = float(audit_metrics.get("official_mtm_net_return_pct", merged.get("calibrated_official_mtm_net_return_pct", 0.0)) or 0.0)
        merged["calibrated_broker_max_drawdown_pct"] = float(audit_metrics.get("broker_max_drawdown_pct", merged.get("calibrated_broker_max_drawdown_pct", 0.0)) or 0.0)
        merged["calibration_metrics"] = audit_metrics
    merged["audit_pass"] = bool(audit.get("audit_pass"))
    merged["audit_status"] = "pass" if audit.get("audit_pass") else "fail"
    merged["audit"] = audit
    if not audit.get("audit_pass"):
        merged["reject_reason"] = "calibration_audit_failed"
    _refresh_path_score(merged)
    return merged


def _refresh_path_score(row: dict[str, Any]) -> None:
    score, components = score_path_calibrated_row(
        dict(row.get("calibration_metrics") or {}),
        dict(row.get("path_risk_metrics") or {}),
        tuple(row.get("calibration_fold_metrics") or ()),
    )
    row["path_calibrated_score"] = float(score)
    row["path_score_components"] = dict(components)


def _calibration_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    score = row.get("path_calibrated_score")
    return (
        bool(row.get("reject_reason")),
        not bool(row.get("audit_pass")),
        -float(score if score is not None else row.get("calibrated_broker_net_return_pct", 0.0) or 0.0),
        -float(row.get("calibrated_broker_net_return_pct", 0.0) or 0.0),
        -float(row.get("calibrated_official_mtm_net_return_pct", 0.0) or 0.0),
        float(row.get("calibrated_broker_max_drawdown_pct", 0.0) or 0.0),
        -float(row.get("filled_selected_rate", 0.0) or 0.0),
        str(row.get("name") or ""),
    )


def _row_cache_key(
    stage2_source_hash: str,
    calibration_input_hash: str,
    source_row: dict[str, Any],
    candidate_section: str,
    rank: int,
    cfg: KALCBConfig,
    baseline_spec: Any,
) -> str:
    return stable_signature(
        {
            "stage2_source_hash": stage2_source_hash,
            "calibration_version": STAGE2_CALIBRATION_VERSION,
            "path_quality_model_version": PATH_QUALITY_MODEL_VERSION,
            "interaction_regime_model_version": INTERACTION_REGIME_MODEL_VERSION,
            "path_quality_usage_contract": PATH_QUALITY_USAGE_CONTRACT,
            "path_calibration_score_version": PATH_CALIBRATION_SCORE_VERSION,
            "calibration_input_hash": calibration_input_hash,
            "row_name": str(source_row.get("name") or ""),
            "source_row_hash": stable_signature(source_row),
            "candidate_section": candidate_section,
            "rank": int(rank),
            "core_version": KALCB_CORE_VERSION,
            "fold_contract": {"fold_count": _CALIBRATION_FOLD_COUNT, "split": "chronological_equal_session_chunks"},
            "risk_policy": _portfolio_risk_mutations(cfg),
            "cost_policy": {
                "commission_bps": cfg.commission_bps,
                "slippage_bps": cfg.slippage_bps,
                "tax_bps_on_sell": cfg.tax_bps_on_sell,
            },
            "baseline_spec": {"name": baseline_spec.name, "entry": asdict(baseline_spec.entry), "exit": asdict(baseline_spec.exit)},
        }
    )


def _row_cache_path(cache_root: Path, cache_key: str) -> Path:
    return cache_root / f"stage2_calibration_{cache_key[:16]}.json"


def _calibration_input_hash(config: dict[str, Any], stage2_payload: dict[str, Any]) -> str:
    return stable_signature(
        {
            "calibration_version": STAGE2_CALIBRATION_VERSION,
            "path_quality_model_version": PATH_QUALITY_MODEL_VERSION,
            "interaction_regime_model_version": INTERACTION_REGIME_MODEL_VERSION,
            "path_quality_usage_contract": PATH_QUALITY_USAGE_CONTRACT,
            "path_calibration_score_version": PATH_CALIBRATION_SCORE_VERSION,
            "fold_count": _CALIBRATION_FOLD_COUNT,
            "training_config": _training_only_config(dict(config), train_only=True),
            "stage2_training_window": dict(stage2_payload.get("training_window") or {}),
            "stage2_source_fingerprints": dict(stage2_payload.get("source_fingerprints") or {}),
            "stage2_causality_policy": dict(stage2_payload.get("causality_policy") or {}),
        }
    )


def _prepare_calibration_base(config: dict[str, Any]) -> _CalibrationBase:
    training_config = _training_only_config(dict(config), train_only=True)
    cfg = KALCBConfig.from_mapping(training_config, {})
    dataset = prepare_first30_dataset(training_config)
    contexts = _build_contexts(dataset)
    context_by_key = {(day, ctx.symbol): ctx for day, items in contexts.items() for ctx in items}
    train_dates = tuple(dataset.trading_dates)
    return _CalibrationBase(
        training_config=training_config,
        cfg=cfg,
        dataset=dataset,
        contexts=contexts,
        context_by_key=context_by_key,
        train_dates=train_dates,
    )


def _candidate_source_from_row(
    stage2_path: Path,
    stage2_payload: dict[str, Any],
    stage2_source_hash: str,
    source_row: dict[str, Any],
    candidate_section: str,
    rank: int,
) -> FixedCandidateSource:
    frontier = _frontier_from_payload(source_row.get("frontier") or {})
    first30 = _first30_from_payload(source_row.get("first30") or {})
    return FixedCandidateSource(
        source_path=str(stage2_path),
        source_file_hash=stage2_source_hash,
        source_sweep_hash=str(stage2_payload.get("sweep_hash") or ""),
        source_row_name=str(source_row.get("name") or f"{frontier.name}__{first30.name}"),
        frontier=frontier,
        first30=first30,
        source_section=str(candidate_section),
        source_rank=int(rank),
        calibration_metadata={
            key: source_row.get(key)
            for key in (
                "calibration_version",
                "path_calibration_score_version",
                "path_calibrated_score",
                "path_score_components",
                "path_risk_metrics",
                "path_quality_model",
                "interaction_regime_model_version",
                "interaction_regime_model",
                "fold_path_risk_metrics",
                "original_source_section",
                "original_source_rank",
                "original_source_name",
            )
            if key in source_row
        },
    )


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB Stage 2 Path-Risk Core Calibration",
        "",
        f"Calibration hash: `{payload['calibration_hash']}`",
        f"Stage 2 artifact: `{payload['stage2_artifact']}`",
        "",
        "| Rank | Audit | Path Score | Broker Net | Worst Fold | Capture | MAE<=-1R | Giveback R | Trades | DD | Path Model | Regime Model | Config |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(payload.get("top_calibrated_stage2", []), start=1):
        path = dict(row.get("path_risk_metrics") or {})
        model = dict(row.get("path_quality_model") or {})
        regime_model = dict(row.get("interaction_regime_model") or {})
        fold_rows = tuple(row.get("calibration_fold_metrics") or ())
        worst_fold = min(
            (
                float((fold.get("metrics") or {}).get("portfolio_equivalent_net_return_pct", 0.0) or 0.0)
                for fold in fold_rows
            ),
            default=float(row.get("calibrated_broker_net_return_pct", 0.0) or 0.0),
        )
        lines.append(
            f"| {index} | {int(bool(row.get('audit_pass')))} | "
            f"{float(row.get('path_calibrated_score', 0.0) or 0.0):.2f} | "
            f"{100.0 * float(row.get('calibrated_broker_net_return_pct', 0.0) or 0.0):.2f}% | "
            f"{100.0 * worst_fold:.2f}% | "
            f"{100.0 * float(path.get('avg_mfe_capture', 0.0) or 0.0):.2f}% | "
            f"{100.0 * float(path.get('mae_le_neg_1_share', 0.0) or 0.0):.1f}% | "
            f"{float(path.get('avg_giveback_r', 0.0) or 0.0):.2f} | "
            f"{float(row.get('trade_count', 0.0) or 0.0):.0f} | "
            f"{100.0 * float(row.get('calibrated_broker_max_drawdown_pct', 0.0) or 0.0):.2f}% | "
            f"{int(bool(model.get('accepted')))} | "
            f"{int(bool(regime_model.get('accepted')))} | "
            f"{row.get('name', '')} |"
        )
    return "\n".join(lines) + "\n"


def _write_progress(output_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"updated_at": _utc_now_iso(), "stage": stage, **extra}
    path = output_dir / "progress.json"
    tmp = output_dir / "progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
