from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.config import load_yaml_config, normalize_runtime_config

from .first30_signal_sweep import run_first30_signal_sweep
from .premarket_frontier_sweep import run_premarket_frontier_only_sweep, run_premarket_frontier_sweep
from .stage2_calibration import DEFAULT_CALIBRATION_SECTION, run_stage2_core_calibration
from .trade_plan_sweep import DEFAULT_OPTIMIZED_SOURCE, run_trade_plan_sweep


DEFAULT_OUTPUT_DIR = Path("data/backtests/output/kalcb/three_stage_portfolio_aware")


def run_three_stage_pipeline(
    config: dict[str, Any],
    *,
    stage1_artifact: str | Path = DEFAULT_OPTIMIZED_SOURCE,
    stage1_from_scratch: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_workers: int = 2,
    stage1_top_n: int = 5,
    stage2_top_n: int = 5,
    stage2_calibration_limit: int = 12,
    holdout_days: int = 42,
    first30_top_n: int = 8,
    refine_first30_top_n: int = 8,
    max_first30_coarse_specs: int | None = None,
    stage1_coarse_frontier_limit: int = 640,
    stage1_deep_pair_count: int = 24,
    stage1_deep_per_mode_limit: int = 160,
    stage1_max_frontier_specs: int | None = None,
    deep_pair_count: int = 24,
    deep_per_mode_limit: int = 160,
    coarse_entry_limit: int = 720,
    coarse_exit_limit: int = 240,
    entry_seed_top_n: int = 32,
    deep_refine_top_n: int = 32,
    deep_refine_max_specs: int = 15_000,
    stage3_successive_halving: bool = True,
    stage3_screen_deep_refine_max_specs: int = 3_000,
    stage3_deep_winner_count: int = 2,
    finalist_count: int = 50,
    compiled_cache_dir: str | Path | None = None,
    stage3_only: bool = False,
    stage2_artifact: str | Path | None = None,
    stage2_calibration_artifact: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_stage1_artifact = Path(stage1_artifact)
    stage1_payload: dict[str, Any] | None = None
    stage2_payload: dict[str, Any] | None = None
    calibration_payload: dict[str, Any] | None = None
    calibration_json: Path | None = None
    if stage3_only:
        if not stage2_calibration_artifact:
            raise ValueError("--stage3-only requires --stage2-calibration-artifact")
        calibration_json = Path(stage2_calibration_artifact)
        calibration_payload = _read_json(calibration_json)
        stage2_json_value = stage2_artifact or calibration_payload.get("stage2_artifact") or ""
        if not stage2_json_value:
            raise ValueError("--stage3-only requires --stage2-artifact or a stage2_artifact field in the calibration artifact")
        stage2_json = Path(stage2_json_value)
        top_stage2 = list(calibration_payload.get("top_calibrated_stage2") or [])[: max(1, int(stage2_top_n))]
        if not top_stage2:
            raise ValueError("KALCB Stage 2 calibration artifact has no top_calibrated_stage2 rows")
        _write_status(
            out,
            "stage3_only_loaded",
            stage2_json=str(stage2_json),
            calibration_json=str(calibration_json),
            stage2_top_n=len(top_stage2),
        )
    elif stage1_from_scratch:
        _write_status(out, "stage1_running", stage1_top_n=int(stage1_top_n))
        stage1_dir = out / "stage1_premarket_frontier"
        stage1_payload = run_premarket_frontier_only_sweep(
            config,
            output_dir=stage1_dir,
            holdout_days=holdout_days,
            max_workers=max_workers,
            coarse_frontier_limit=stage1_coarse_frontier_limit,
            deep_pair_count=stage1_deep_pair_count,
            deep_per_mode_limit=stage1_deep_per_mode_limit,
            max_frontier_specs=stage1_max_frontier_specs,
        )
        resolved_stage1_artifact = Path(stage1_payload["artifact_paths"]["json"])
        _write_status(
            out,
            "stage1_completed",
            stage1_artifact=str(resolved_stage1_artifact),
            top_portfolio=[str(row.get("name") or "") for row in stage1_payload.get("top_portfolio_proxy", [])[:5]],
        )

    if not stage3_only:
        _write_status(out, "stage2_first30_running", stage1_artifact=str(resolved_stage1_artifact), stage1_top_n=int(stage1_top_n))
        stage2_first30_dir = out / "stage2_first30_selector"
        first30_payload = run_first30_signal_sweep(
            config,
            output_dir=stage2_first30_dir,
            holdout_days=holdout_days,
            max_workers=max_workers,
            refine_top_n=refine_first30_top_n,
            max_coarse_specs=max_first30_coarse_specs,
        )
        first30_json = Path(first30_payload["artifact_paths"]["json"])
        _write_status(
            out,
            "stage2_first30_completed",
            first30_json=str(first30_json),
            top_portfolio=[str(row.get("name") or "") for row in first30_payload.get("top_portfolio_proxy", [])[:5]],
        )

        _write_status(out, "stage2_frontier_running", stage1_artifact=str(resolved_stage1_artifact), first30_artifact=str(first30_json), stage1_top_n=int(stage1_top_n))
        stage2_dir = out / "stage2_portfolio_aware_first30"
        stage2_payload = run_premarket_frontier_sweep(
            config,
            output_dir=stage2_dir,
            holdout_days=holdout_days,
            max_workers=max_workers,
            first30_top_n=first30_top_n,
            refine_first30_top_n=refine_first30_top_n,
            max_first30_coarse_specs=max_first30_coarse_specs,
            first30_artifact=first30_json,
            frontier_artifact=resolved_stage1_artifact,
            frontier_top_n=stage1_top_n,
            deep_pair_count=deep_pair_count,
            deep_per_mode_limit=deep_per_mode_limit,
        )
        stage2_json = Path(stage2_payload["artifact_paths"]["json"])
        _write_status(
            out,
            "stage2_completed",
            stage2_json=str(stage2_json),
            raw_stage2_top_names=[str(row.get("name") or "") for row in (stage2_payload.get("top_portfolio_proxy") or stage2_payload.get("top_combined") or [])[:5]],
        )

        _write_status(
            out,
            "stage2_calibration_running",
            stage2_json=str(stage2_json),
            calibration_limit=int(stage2_calibration_limit),
            finalist_count=int(stage2_top_n),
        )
        calibration_dir = out / "stage2_core_calibration"
        calibration_payload = run_stage2_core_calibration(
            config,
            stage2_artifact=stage2_json,
            output_dir=calibration_dir,
            candidate_section=DEFAULT_CALIBRATION_SECTION,
            candidate_limit=stage2_calibration_limit,
            finalist_count=stage2_top_n,
            max_workers=max_workers,
            compiled_cache_dir=compiled_cache_dir,
        )
        calibration_json = Path(calibration_payload["artifact_paths"]["json"])
        top_stage2 = list(calibration_payload.get("top_calibrated_stage2") or [])[: max(1, int(stage2_top_n))]
        if not top_stage2:
            raise ValueError("KALCB Stage 2 calibration produced no audit-passed shared-core finalists")
        _write_status(
            out,
            "stage2_calibration_completed",
            calibration_json=str(calibration_json),
            stage2_top_n=len(top_stage2),
            stage2_top_names=[str(row.get("name") or "") for row in top_stage2],
        )

    stage3_root = out / "stage3_trade_plan"
    stage3_root.mkdir(parents=True, exist_ok=True)
    stage3_payloads = _run_stage3_sweeps(
        config,
        top_stage2=top_stage2,
        stage2_json=stage2_json,
        output_dir=out,
        stage3_root=stage3_root,
        max_workers=max_workers,
        coarse_entry_limit=coarse_entry_limit,
        coarse_exit_limit=coarse_exit_limit,
        entry_seed_top_n=entry_seed_top_n,
        deep_refine_top_n=deep_refine_top_n,
        deep_refine_max_specs=deep_refine_max_specs,
        finalist_count=finalist_count,
        compiled_cache_dir=compiled_cache_dir,
        successive_halving=stage3_successive_halving,
        screen_deep_refine_max_specs=stage3_screen_deep_refine_max_specs,
        deep_winner_count=stage3_deep_winner_count,
    )

    stage3_ranked = sorted(
        stage3_payloads,
        key=_stage3_sort_key,
    )
    payload = {
        "strategy": "kalcb",
        "pipeline": "stage1_premarket_frontier_only_to_stage2_first30_selection_to_stage2_shared_core_calibration_to_stage3_shared_core_trade_plan",
        "created_at": _utc_now_iso(),
        "stage1_from_scratch": bool(stage1_from_scratch),
        "stage3_only": bool(stage3_only),
        "stage1_artifact": str(resolved_stage1_artifact),
        "stage1_discovery_artifact": str((stage1_payload or {}).get("artifact_paths", {}).get("json", "")),
        "stage1_top_n": int(stage1_top_n),
        "stage2_artifact": str(stage2_json),
        "stage2_calibration_artifact": str(calibration_json or ""),
        "stage2_calibration_limit": int(stage2_calibration_limit),
        "stage2_calibration_rows": top_stage2,
        "stage2_top_n": len(top_stage2),
        "stage3_scheduler": {
            "successive_halving": bool(stage3_successive_halving),
            "screen_deep_refine_max_specs": int(stage3_screen_deep_refine_max_specs),
            "deep_winner_count": int(stage3_deep_winner_count),
            "deep_refine_max_specs": int(deep_refine_max_specs),
        },
        "stage3_count": len(stage3_payloads),
        "stage3_ranked": stage3_ranked,
    }
    path = out / "three_stage_pipeline_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_status(out, "completed", summary=str(path))
    return payload


def _stage3_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "sweep_hash": payload.get("sweep_hash"),
        "artifact_paths": payload.get("artifact_paths"),
        "candidate_source": (payload.get("fixed_candidate_source") or {}).get("source_row_name", ""),
        "candidate_section": (payload.get("fixed_candidate_source") or {}).get("source_section", ""),
        "candidate_rank": (payload.get("fixed_candidate_source") or {}).get("source_rank", 0),
        "baseline": payload.get("baseline"),
        "top_train": payload.get("top_train", [])[:10],
        "top_promoted": payload.get("top_promoted", [])[:10],
        "audit_replays": payload.get("audit_replays", [])[:25],
        "audit_pass": payload.get("audit_pass"),
        "fast_suppression_audit": payload.get("fast_suppression_audit"),
        "sweep_counts": payload.get("sweep_counts"),
    }
    summary["stage3_ranking_basis"] = _stage3_ranking_basis(summary)
    return summary


def _run_stage3_sweeps(
    config: dict[str, Any],
    *,
    top_stage2: list[dict[str, Any]],
    stage2_json: Path,
    output_dir: Path,
    stage3_root: Path,
    max_workers: int,
    coarse_entry_limit: int,
    coarse_exit_limit: int,
    entry_seed_top_n: int,
    deep_refine_top_n: int,
    deep_refine_max_specs: int,
    finalist_count: int,
    compiled_cache_dir: str | Path | None,
    successive_halving: bool,
    screen_deep_refine_max_specs: int,
    deep_winner_count: int,
) -> list[dict[str, Any]]:
    if not top_stage2:
        return []
    if not successive_halving:
        return [
            _run_stage3_candidate(
                config,
                row=row,
                candidate_index=index,
                candidate_count=len(top_stage2),
                stage2_json=stage2_json,
                output_dir=output_dir,
                root=stage3_root,
                pass_name="full",
                max_workers=max_workers,
                coarse_entry_limit=coarse_entry_limit,
                coarse_exit_limit=coarse_exit_limit,
                entry_seed_top_n=entry_seed_top_n,
                deep_refine_top_n=deep_refine_top_n,
                deep_refine_max_specs=deep_refine_max_specs,
                finalist_count=finalist_count,
                compiled_cache_dir=compiled_cache_dir,
            )
            for index, row in enumerate(top_stage2, start=1)
        ]

    screen_limit = max(1, min(int(screen_deep_refine_max_specs), int(deep_refine_max_specs)))
    screen_top_n = max(1, min(int(deep_refine_top_n), 12))
    screen_rows: list[dict[str, Any]] = []
    for index, row in enumerate(top_stage2, start=1):
        screen_rows.append(
            _run_stage3_candidate(
                config,
                row=row,
                candidate_index=index,
                candidate_count=len(top_stage2),
                stage2_json=stage2_json,
                output_dir=output_dir,
                root=stage3_root,
                pass_name="screen",
                max_workers=max_workers,
                coarse_entry_limit=coarse_entry_limit,
                coarse_exit_limit=coarse_exit_limit,
                entry_seed_top_n=entry_seed_top_n,
                deep_refine_top_n=screen_top_n,
                deep_refine_max_specs=screen_limit,
                finalist_count=min(int(finalist_count), 25),
                compiled_cache_dir=compiled_cache_dir,
            )
        )

    ranked_screen = sorted(screen_rows, key=_stage3_sort_key)
    winner_count = max(1, min(int(deep_winner_count), len(ranked_screen)))
    deep_winner_keys = {
        (str(row.get("candidate_section") or ""), int(row.get("candidate_rank") or 0))
        for row in ranked_screen[:winner_count]
    }
    (stage3_root / "stage3_successive_halving_screen_summary.json").write_text(
        json.dumps(
            {
                "created_at": _utc_now_iso(),
                "screen_deep_refine_max_specs": screen_limit,
                "screen_deep_refine_top_n": screen_top_n,
                "deep_refine_max_specs": int(deep_refine_max_specs),
                "deep_refine_top_n": int(deep_refine_top_n),
                "deep_winner_count": winner_count,
                "screen_ranked": ranked_screen,
                "deep_winner_keys": sorted([list(key) for key in deep_winner_keys]),
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    _write_status(
        output_dir,
        "stage3_screen_completed",
        screened=len(screen_rows),
        deep_winner_count=winner_count,
        deep_winners=[str(row.get("candidate_source") or "") for row in ranked_screen[:winner_count]],
    )

    deep_rows: list[dict[str, Any]] = []
    for screen_row in ranked_screen[:winner_count]:
        source_index = int(screen_row.get("stage2_candidate_index") or 0)
        source_row = top_stage2[source_index - 1] if 0 < source_index <= len(top_stage2) else {}
        if not source_row:
            continue
        deep_rows.append(
            _run_stage3_candidate(
                config,
                row=source_row,
                candidate_index=source_index,
                candidate_count=len(top_stage2),
                stage2_json=stage2_json,
                output_dir=output_dir,
                root=stage3_root,
                pass_name="deep",
                max_workers=max_workers,
                coarse_entry_limit=coarse_entry_limit,
                coarse_exit_limit=coarse_exit_limit,
                entry_seed_top_n=entry_seed_top_n,
                deep_refine_top_n=deep_refine_top_n,
                deep_refine_max_specs=deep_refine_max_specs,
                finalist_count=finalist_count,
                compiled_cache_dir=compiled_cache_dir,
            )
        )

    deep_keys = {(str(row.get("candidate_section") or ""), int(row.get("candidate_rank") or 0)) for row in deep_rows}
    carry_forward_screen = [
        row
        for row in ranked_screen
        if (str(row.get("candidate_section") or ""), int(row.get("candidate_rank") or 0)) not in deep_keys
    ]
    final_rows = deep_rows + carry_forward_screen
    (stage3_root / "stage3_successive_halving_final_summary.json").write_text(
        json.dumps(
            {
                "created_at": _utc_now_iso(),
                "deep_rows": deep_rows,
                "carried_screen_rows": carry_forward_screen,
                "final_ranked": sorted(final_rows, key=_stage3_sort_key),
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return final_rows


def _run_stage3_candidate(
    config: dict[str, Any],
    *,
    row: dict[str, Any],
    candidate_index: int,
    candidate_count: int,
    stage2_json: Path,
    output_dir: Path,
    root: Path,
    pass_name: str,
    max_workers: int,
    coarse_entry_limit: int,
    coarse_exit_limit: int,
    entry_seed_top_n: int,
    deep_refine_top_n: int,
    deep_refine_max_specs: int,
    finalist_count: int,
    compiled_cache_dir: str | Path | None,
) -> dict[str, Any]:
    optimized_source = Path(str(row.get("calibrated_source_path") or stage2_json))
    candidate_section = str(row.get("calibrated_source_section") or row.get("source_section") or DEFAULT_CALIBRATION_SECTION)
    candidate_rank = int(row.get("calibrated_source_rank") if row.get("calibrated_source_rank") is not None else row.get("source_rank") or 0)
    candidate_name = str(row.get("name") or f"{candidate_section}_{candidate_rank}")
    candidate_dir = root / pass_name / f"candidate_{candidate_index:02d}_{_safe_name(candidate_name)[:96]}"
    _write_status(
        output_dir,
        f"stage3_{pass_name}_running",
        candidate=candidate_index,
        total=candidate_count,
        candidate_name=candidate_name,
        optimized_source=str(optimized_source),
        candidate_section=candidate_section,
        candidate_rank=candidate_rank,
        deep_refine_max_specs=int(deep_refine_max_specs),
        deep_refine_top_n=int(deep_refine_top_n),
    )
    payload = run_trade_plan_sweep(
        config,
        optimized_source=optimized_source,
        candidate_section=candidate_section,
        candidate_rank=candidate_rank,
        strict_candidate_source=False,
        output_dir=candidate_dir,
        train_only=True,
        max_workers=max_workers,
        fold_count=2,
        coarse_entry_limit=coarse_entry_limit,
        coarse_exit_limit=coarse_exit_limit,
        entry_seed_top_n=entry_seed_top_n,
        deep_refine_top_n=deep_refine_top_n,
        deep_refine_max_specs=deep_refine_max_specs,
        finalist_count=finalist_count,
        audit_max_workers=max_workers,
        worker_backend="thread",
        compiled_cache_dir=compiled_cache_dir,
    )
    summary = _stage3_summary(payload)
    summary.update(
        {
            "stage3_pass": pass_name,
            "stage2_candidate_index": int(candidate_index),
            "stage2_calibration_row": row,
        }
    )
    basis = _stage3_ranking_basis(summary)
    _write_status(
        output_dir,
        f"stage3_{pass_name}_candidate_completed",
        candidate=candidate_index,
        total=candidate_count,
        candidate_name=candidate_name,
        ranking_source=basis.get("source", ""),
        broker_net_return_pct=round(float(basis.get("broker_net_return_pct", 0.0) or 0.0), 3),
        audit_pass=bool(basis.get("audit_pass", False)),
    )
    return summary


def _stage3_sort_key(item: dict[str, Any]) -> tuple[int, float, str, str]:
    basis = _stage3_ranking_basis(item)
    tier = basis.get("tier", 99)
    return (
        int(tier if tier is not None else 99),
        -float(basis.get("broker_net_return_pct", 0.0) or 0.0),
        str(basis.get("source") or ""),
        str(item.get("candidate_source") or ""),
    )


def _stage3_ranking_basis(item: dict[str, Any]) -> dict[str, Any]:
    promoted_names = {str(row.get("name") or "") for row in item.get("top_promoted", []) if row.get("name")}
    audit_rows = [dict(row) for row in item.get("audit_replays", []) if row.get("audit_pass")]
    promoted_audits = [row for row in audit_rows if not promoted_names or str(row.get("name") or "") in promoted_names]
    if promoted_audits:
        best = max(promoted_audits, key=lambda row: float((row.get("audit_metrics") or {}).get("broker_net_return_pct", 0.0) or 0.0))
        return {
            "tier": 0,
            "source": "audit_passed_top_promoted" if promoted_names else "audit_passed_replay",
            "name": best.get("name", ""),
            "audit_pass": True,
            "broker_net_return_pct": float((best.get("audit_metrics") or {}).get("broker_net_return_pct", 0.0) or 0.0),
        }
    if audit_rows:
        best = max(audit_rows, key=lambda row: float((row.get("audit_metrics") or {}).get("broker_net_return_pct", 0.0) or 0.0))
        return {
            "tier": 1,
            "source": "audit_passed_replay",
            "name": best.get("name", ""),
            "audit_pass": True,
            "broker_net_return_pct": float((best.get("audit_metrics") or {}).get("broker_net_return_pct", 0.0) or 0.0),
        }
    promoted = item.get("top_promoted", [])
    if promoted:
        best = max(promoted, key=lambda row: float((row.get("train_metrics") or {}).get("broker_net_return_pct", 0.0) or 0.0))
        return {
            "tier": 2,
            "source": "top_promoted_audit_missing",
            "name": best.get("name", ""),
            "audit_pass": False,
            "broker_net_return_pct": float((best.get("train_metrics") or {}).get("broker_net_return_pct", 0.0) or 0.0),
        }
    train = item.get("top_train", [])
    best = train[0] if train else {}
    return {
        "tier": 3,
        "source": "top_train_diagnostic_fallback",
        "name": best.get("name", ""),
        "audit_pass": False,
        "broker_net_return_pct": float((best.get("train_metrics") or {}).get("broker_net_return_pct", 0.0) or 0.0),
    }


def _write_status(output_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"updated_at": _utc_now_iso(), "stage": stage, **extra}
    path = output_dir / "run_status.json"
    tmp = output_dir / "run_status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    print(f"[kalcb-three-stage] {stage} {json.dumps(extra, sort_keys=True, default=str)}", flush=True)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run KALCB Stage 1 finalist -> portfolio-aware Stage 2 -> shared-core Stage 3 pipeline.")
    parser.add_argument("--config", default="config/optimization/kalcb.yaml")
    parser.add_argument("--stage1-artifact", default=str(DEFAULT_OPTIMIZED_SOURCE))
    parser.add_argument("--stage1-from-scratch", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--stage1-top-n", type=int, default=5)
    parser.add_argument("--stage2-top-n", type=int, default=5)
    parser.add_argument("--stage2-calibration-limit", type=int, default=12)
    parser.add_argument("--holdout-days", type=int, default=42)
    parser.add_argument("--first30-top-n", type=int, default=8)
    parser.add_argument("--refine-first30-top-n", type=int, default=8)
    parser.add_argument("--max-first30-coarse-specs", type=int, default=None)
    parser.add_argument("--stage1-coarse-frontier-limit", type=int, default=640)
    parser.add_argument("--stage1-deep-pair-count", type=int, default=24)
    parser.add_argument("--stage1-deep-per-mode-limit", type=int, default=160)
    parser.add_argument("--stage1-max-frontier-specs", type=int, default=None)
    parser.add_argument("--deep-pair-count", type=int, default=24)
    parser.add_argument("--deep-per-mode-limit", type=int, default=160)
    parser.add_argument("--coarse-entry-limit", type=int, default=720)
    parser.add_argument("--coarse-exit-limit", type=int, default=240)
    parser.add_argument("--entry-seed-top-n", type=int, default=32)
    parser.add_argument("--deep-refine-top-n", type=int, default=32)
    parser.add_argument("--deep-refine-max-specs", type=int, default=15_000)
    parser.add_argument("--stage3-only", action="store_true")
    parser.add_argument("--stage2-artifact", default=None)
    parser.add_argument("--stage2-calibration-artifact", default=None)
    parser.add_argument("--stage3-screen-deep-refine-max-specs", type=int, default=3_000)
    parser.add_argument("--stage3-deep-winner-count", type=int, default=2)
    parser.add_argument("--no-stage3-successive-halving", dest="stage3_successive_halving", action="store_false")
    parser.add_argument("--finalist-count", type=int, default=50)
    parser.add_argument("--compiled-cache-dir", default=None)
    parser.set_defaults(stage3_successive_halving=True)
    args = parser.parse_args(argv)
    config = normalize_runtime_config("kalcb", load_yaml_config(args.config))
    payload = run_three_stage_pipeline(
        config,
        stage1_artifact=args.stage1_artifact,
        stage1_from_scratch=args.stage1_from_scratch,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        stage1_top_n=args.stage1_top_n,
        stage2_top_n=args.stage2_top_n,
        stage2_calibration_limit=args.stage2_calibration_limit,
        holdout_days=args.holdout_days,
        first30_top_n=args.first30_top_n,
        refine_first30_top_n=args.refine_first30_top_n,
        max_first30_coarse_specs=args.max_first30_coarse_specs,
        stage1_coarse_frontier_limit=args.stage1_coarse_frontier_limit,
        stage1_deep_pair_count=args.stage1_deep_pair_count,
        stage1_deep_per_mode_limit=args.stage1_deep_per_mode_limit,
        stage1_max_frontier_specs=args.stage1_max_frontier_specs,
        deep_pair_count=args.deep_pair_count,
        deep_per_mode_limit=args.deep_per_mode_limit,
        coarse_entry_limit=args.coarse_entry_limit,
        coarse_exit_limit=args.coarse_exit_limit,
        entry_seed_top_n=args.entry_seed_top_n,
        deep_refine_top_n=args.deep_refine_top_n,
        deep_refine_max_specs=args.deep_refine_max_specs,
        stage3_successive_halving=args.stage3_successive_halving,
        stage3_screen_deep_refine_max_specs=args.stage3_screen_deep_refine_max_specs,
        stage3_deep_winner_count=args.stage3_deep_winner_count,
        finalist_count=args.finalist_count,
        compiled_cache_dir=args.compiled_cache_dir,
        stage3_only=args.stage3_only,
        stage2_artifact=args.stage2_artifact,
        stage2_calibration_artifact=args.stage2_calibration_artifact,
    )
    print(json.dumps({"summary": "three_stage_pipeline_summary.json", "stage3_count": payload["stage3_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
