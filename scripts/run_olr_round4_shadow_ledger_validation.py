from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "olr_round2_oos_deep_ablation.py"
ROUND4_CONFIG = ROOT / "data" / "backtests" / "output" / "olr" / "round_4" / "optimized_config.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "backtests" / "output" / "olr" / "shadow_ledger" / "round_4"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate the OLR round-4 shadow opportunity ledger.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--holdout-days", type=int, default=42)
    parser.add_argument("--max-days", type=int, default=0, help="Optional smoke cap per window; 0 runs the full train/OOS split.")
    parser.add_argument("--max-shadow-candidates-per-day", type=int, default=0)
    args = parser.parse_args()

    started = time.monotonic()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    def status(stage: str, **fields: Any) -> None:
        payload = {"stage": stage, "elapsed_seconds": round(time.monotonic() - started, 3)}
        payload.update(fields)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str))
            handle.write("\n")
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)

    from backtests.config import load_yaml_config, normalize_runtime_config
    from backtests.strategies.olr import trade_plan_sweep as tps
    from backtests.strategies.olr.shadow_ledger_reranker import (
        DEFAULT_FEATURE_KEYS,
        build_shadow_opportunity_ledger,
        evaluate_same_day_reranker_with_replay,
        feature_coverage,
        fit_same_day_reranker_profile,
        write_shadow_reranker_artifacts,
    )
    from strategy_olr.config import OLRConfig

    helper = load_helper()
    optimized = read_json(ROUND4_CONFIG)
    mutations = dict(optimized["mutations"])
    runtime_config = normalize_runtime_config("olr", load_yaml_config(str(ROOT / "config" / "optimization" / "olr.yaml")))
    runtime_config["capability_level"] = "real_replay"
    runtime_config["holdout_days"] = int(args.holdout_days)
    runtime_config["use_full_available_window"] = True
    candidate = helper.Candidate("round4_soft_combo_current", "round4_current", mutations, "Current round-4 optimized mutations.")
    source = helper.build_sources([candidate])[1][0]

    status("prepare_dataset_start", holdout_days=int(args.holdout_days))
    dataset = tps.prepare_research_sweep_dataset(
        runtime_config,
        holdout_days=int(args.holdout_days),
        expected_universe_size=tps.DEFAULT_EXPECTED_UNIVERSE_SIZE,
        include_holdout=True,
    )
    eligible_dates, next_by_date = tps._eligible_execution_dates(dataset)
    train_dates = tuple(day for day in eligible_dates if day < dataset.holdout_start)
    oos_dates = tuple(day for day in eligible_dates if day >= dataset.holdout_start)
    if int(args.max_days) > 0:
        train_dates = train_dates[-int(args.max_days) :]
        oos_dates = oos_dates[: int(args.max_days)]
    status(
        "prepare_dataset_done",
        dataset_dates=len(dataset.trading_dates),
        train_dates=len(train_dates),
        oos_dates=len(oos_dates),
        holdout_start=dataset.holdout_start.isoformat(),
    )

    status("research_cache_start")
    research_cache = tps.research_snapshots_for_dataset(dataset, {})
    status("research_cache_done", snapshots=len(research_cache))

    stage1_mutations = dict(source.stage1_mutations or {})
    status("stage1_snapshots_start", stage1_mutations=len(stage1_mutations))
    stage1_snapshots = tps.snapshots_for_experiment(dataset, stage1_mutations, research_snapshots=research_cache)
    stage1_cfg = OLRConfig.from_mapping(dataset.config, stage1_mutations)
    contexts = tps.afternoon_contexts_for_snapshots(dataset, stage1_snapshots, stage1_cfg)
    status("stage1_snapshots_done", snapshots=len(stage1_snapshots), contexts=len(contexts))

    cfg = OLRConfig.from_mapping(dataset.config, mutations)
    status("ledger_train_start")
    train_rows = build_shadow_opportunity_ledger(
        stage1_snapshots,
        contexts,
        dataset.bars_by_key,
        next_by_date,
        cfg,
        train_dates,
        window="train",
        source_label=source.name,
        max_shadow_candidates_per_day=int(args.max_shadow_candidates_per_day),
    )
    status("ledger_train_done", rows=len(train_rows), actual_slots=sum(1 for row in train_rows if row.get("actual_trade_slot")))

    status("ledger_oos_start")
    oos_rows = build_shadow_opportunity_ledger(
        stage1_snapshots,
        contexts,
        dataset.bars_by_key,
        next_by_date,
        cfg,
        oos_dates,
        window="oos",
        source_label=source.name,
        max_shadow_candidates_per_day=int(args.max_shadow_candidates_per_day),
    )
    status("ledger_oos_done", rows=len(oos_rows), actual_slots=sum(1 for row in oos_rows if row.get("actual_trade_slot")))

    status("replay_validation_start")
    summary = evaluate_same_day_reranker_with_replay(
        train_rows,
        oos_rows,
        dataset.bars_by_key,
        cfg,
        mutations,
        runtime_config=runtime_config,
    )
    summary["source_round"] = 4
    summary["round4_optimized_config"] = str(ROUND4_CONFIG)
    summary["dataset_window"] = {
        "train_start": train_dates[0].isoformat() if train_dates else "",
        "train_end": train_dates[-1].isoformat() if train_dates else "",
        "train_sessions": len(train_dates),
        "oos_start": oos_dates[0].isoformat() if oos_dates else "",
        "oos_end": oos_dates[-1].isoformat() if oos_dates else "",
        "oos_sessions": len(oos_dates),
    }
    summary["ledger_counts"] = {
        "train_rows": len(train_rows),
        "oos_rows": len(oos_rows),
        "train_actual_slots": sum(1 for row in train_rows if row.get("actual_trade_slot")),
        "oos_actual_slots": sum(1 for row in oos_rows if row.get("actual_trade_slot")),
        "train_fill_feasible": sum(1 for row in train_rows if row.get("fill_feasible")),
        "oos_fill_feasible": sum(1 for row in oos_rows if row.get("fill_feasible")),
    }
    summary["feature_coverage"] = {
        "train": feature_coverage(train_rows),
        "oos": feature_coverage(oos_rows),
    }
    status("variant_sweep_start")
    variant_sweep = run_variant_sweep(
        train_rows,
        oos_rows,
        dataset.bars_by_key,
        cfg,
        mutations,
        runtime_config,
        feature_keys=DEFAULT_FEATURE_KEYS,
        fit_same_day_reranker_profile=fit_same_day_reranker_profile,
        evaluate_same_day_reranker_with_replay=evaluate_same_day_reranker_with_replay,
    )
    summary["variant_sweep"] = variant_sweep
    best_variant = best_variant_result(variant_sweep)
    if best_variant:
        summary["best_variant"] = best_variant
        if bool(best_variant.get("promotion_pass")):
            summary["promotion_pass"] = True
            summary["candidate_mutation"] = dict(best_variant.get("candidate_mutation") or {})
    (out / "shadow_reranker_variant_sweep.json").write_text(json.dumps(variant_sweep, indent=2, sort_keys=True, default=str), encoding="utf-8")
    status(
        "variant_sweep_done",
        variants=len(variant_sweep),
        best_variant=str((best_variant or {}).get("name") or ""),
        best_oos_net=metric_net(((best_variant or {}).get("oos") or {}).get("reranked_metrics") or {}),
        best_promotion_pass=bool((best_variant or {}).get("promotion_pass")),
    )
    artifact_paths = write_shadow_reranker_artifacts(train_rows, oos_rows, summary, output_dir=out)
    summary["artifact_paths"] = artifact_paths
    (out / "shadow_reranker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    status(
        "complete",
        promotion_pass=bool(summary.get("promotion_pass")),
        train_actual_net=metric_net(summary["train"]["actual"]["metrics"]),
        train_reranked_net=metric_net(summary["train"]["reranked"]["metrics"]),
        oos_actual_net=metric_net(summary["oos"]["actual"]["metrics"]),
        oos_reranked_net=metric_net(summary["oos"]["reranked"]["metrics"]),
        summary_path=str(out / "shadow_reranker_summary.json"),
        report_path=str(out / "shadow_reranker_report.md"),
    )
    return 0


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("olr_round2_oos_deep_ablation", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_net(metrics: dict[str, Any]) -> float:
    return float(metrics.get("official_mtm_net_return_pct", metrics.get("net_return_pct", 0.0)) or 0.0)


def run_variant_sweep(
    train_rows: Sequence[dict[str, Any]],
    oos_rows: Sequence[dict[str, Any]],
    bars_by_key: Any,
    cfg: Any,
    mutations: dict[str, Any],
    runtime_config: dict[str, Any],
    *,
    feature_keys: Sequence[str],
    fit_same_day_reranker_profile: Any,
    evaluate_same_day_reranker_with_replay: Any,
) -> list[dict[str, Any]]:
    active_train_rows = [row for row in train_rows if int(row.get("same_day_selected_count", 0) or 0) > 0]
    fit_rows = active_train_rows or list(train_rows)
    sector_keys = tuple(key for key in feature_keys if "sector" in key)
    non_sector_keys = tuple(key for key in feature_keys if "sector" not in key)
    price_flow_keys = tuple(
        key
        for key in (
            "daily_candidate_score",
            "daily_candidate_rank",
            "daily_rank_pct",
            "daily_signal_score",
            "relative_strength_pct",
            "accumulation_score",
            "flow_score",
            "foreign_flow_5d",
            "institutional_flow_5d",
            "flow_agreement_5d",
            "afternoon_score",
            "afternoon_score_raw",
            "afternoon_exhaustion_score",
            "afternoon_ret",
            "vwap_ret",
            "gap",
            "rel_volume",
            "close_location",
            "open_drawdown",
            "high_from_open",
            "range_atr",
            "lagged_flow_5d",
            "lagged_foreign_flow_5d",
            "lagged_institutional_flow_5d",
            "lagged_flow_z",
            "lagged_foreign_z",
            "lagged_institutional_z",
            "lagged_flow_agreement_5d",
            "lagged_flow_divergence_5d",
        )
        if key in feature_keys
    )
    sector_plus_price_keys = tuple(dict.fromkeys(price_flow_keys + sector_keys))
    variants = [
        {
            "name": "active_all_default",
            "rows": fit_rows,
            "feature_keys": tuple(feature_keys),
            "target_key": "same_day_replacement_value_r",
            "min_abs_correlation": 0.025,
            "max_abs_weight": 0.75,
            "sector_prior_weight": 0.35,
        },
        {
            "name": "active_all_no_sector_prior",
            "rows": fit_rows,
            "feature_keys": tuple(feature_keys),
            "target_key": "same_day_replacement_value_r",
            "min_abs_correlation": 0.025,
            "max_abs_weight": 0.75,
            "sector_prior_weight": 0.0,
        },
        {
            "name": "active_all_no_sector_prior_margin125",
            "rows": fit_rows,
            "feature_keys": tuple(feature_keys),
            "target_key": "same_day_replacement_value_r",
            "min_abs_correlation": 0.025,
            "max_abs_weight": 0.75,
            "sector_prior_weight": 0.0,
            "replacement_margin": 1.25,
        },
        {
            "name": "active_all_no_sector_prior_margin150",
            "rows": fit_rows,
            "feature_keys": tuple(feature_keys),
            "target_key": "same_day_replacement_value_r",
            "min_abs_correlation": 0.025,
            "max_abs_weight": 0.75,
            "sector_prior_weight": 0.0,
            "replacement_margin": 1.5,
        },
        {
            "name": "active_all_no_sector_prior_margin200",
            "rows": fit_rows,
            "feature_keys": tuple(feature_keys),
            "target_key": "same_day_replacement_value_r",
            "min_abs_correlation": 0.025,
            "max_abs_weight": 0.75,
            "sector_prior_weight": 0.0,
            "replacement_margin": 2.0,
        },
        {
            "name": "active_all_conservative",
            "rows": fit_rows,
            "feature_keys": tuple(feature_keys),
            "target_key": "same_day_replacement_value_r",
            "min_abs_correlation": 0.075,
            "max_abs_weight": 0.25,
            "sector_prior_weight": 0.10,
        },
        {
            "name": "active_route_net_no_sector_prior",
            "rows": fit_rows,
            "feature_keys": tuple(feature_keys),
            "target_key": "route_net_r",
            "min_abs_correlation": 0.025,
            "max_abs_weight": 0.75,
            "sector_prior_weight": 0.0,
        },
        {
            "name": "active_non_sector_context",
            "rows": fit_rows,
            "feature_keys": non_sector_keys,
            "target_key": "same_day_replacement_value_r",
            "min_abs_correlation": 0.025,
            "max_abs_weight": 0.75,
            "sector_prior_weight": 0.0,
        },
        {
            "name": "active_sector_price_context",
            "rows": fit_rows,
            "feature_keys": sector_plus_price_keys,
            "target_key": "same_day_replacement_value_r",
            "min_abs_correlation": 0.05,
            "max_abs_weight": 0.35,
            "sector_prior_weight": 0.15,
        },
    ]
    out: list[dict[str, Any]] = []
    for variant in variants:
        profile = fit_same_day_reranker_profile(
            variant["rows"],
            feature_keys=variant["feature_keys"],
            target_key=str(variant["target_key"]),
            min_abs_correlation=float(variant["min_abs_correlation"]),
            max_abs_weight=float(variant["max_abs_weight"]),
            sector_prior_weight=float(variant["sector_prior_weight"]),
            allow_slot_expansion=False,
            max_replacements_per_day=int(variant.get("max_replacements_per_day", 1) or 1),
            replacement_margin=float(variant.get("replacement_margin", 0.0) or 0.0),
        )
        result = evaluate_same_day_reranker_with_replay(
            train_rows,
            oos_rows,
            bars_by_key,
            cfg,
            mutations,
            runtime_config=runtime_config,
            profile=profile,
        )
        out.append(compact_variant_result(str(variant["name"]), result, fit_row_count=len(variant["rows"])))
    return out


def compact_variant_result(name: str, result: dict[str, Any], *, fit_row_count: int) -> dict[str, Any]:
    profile = dict(result.get("profile") or {})
    return {
        "name": name,
        "promotion_pass": bool(result.get("promotion_pass")),
        "fit_row_count": fit_row_count,
        "profile_hash": str(profile.get("profile_hash") or ""),
        "target_key": str(profile.get("target_key") or ""),
        "slot_policy": str(profile.get("slot_policy") or ""),
        "max_replacements_per_day": int(profile.get("max_replacements_per_day", 0) or 0),
        "replacement_margin": float(profile.get("replacement_margin", 0.0) or 0.0),
        "feature_weight_count": len(dict(profile.get("weights") or {})),
        "sector_prior_weight": float(profile.get("sector_prior_weight", 0.0) or 0.0),
        "candidate_mutation": dict(result.get("candidate_mutation") or {}),
        "train": {
            "actual_metrics": dict(((result.get("train") or {}).get("actual") or {}).get("metrics") or {}),
            "reranked_metrics": dict(((result.get("train") or {}).get("reranked") or {}).get("metrics") or {}),
            "delta": dict((result.get("train") or {}).get("delta") or {}),
        },
        "oos": {
            "actual_metrics": dict(((result.get("oos") or {}).get("actual") or {}).get("metrics") or {}),
            "reranked_metrics": dict(((result.get("oos") or {}).get("reranked") or {}).get("metrics") or {}),
            "delta": dict((result.get("oos") or {}).get("delta") or {}),
        },
    }


def best_variant_result(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    promoted = [item for item in results if bool(item.get("promotion_pass"))]
    candidates = promoted or list(results)
    return max(
        candidates,
        key=lambda item: (
            metric_net(((item.get("oos") or {}).get("reranked_metrics") or {})),
            metric_net(((item.get("train") or {}).get("reranked_metrics") or {})),
            -abs(float(((item.get("oos") or {}).get("delta") or {}).get("entry_fill_count", 0.0) or 0.0)),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
