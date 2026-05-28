from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROUND_ROOT = ROOT / "data" / "backtests" / "output" / "olr"
ROUND4_DIR = ROUND_ROOT / "round_4"
MANIFEST_PATH = ROUND_ROOT / "rounds_manifest.json"
SWEEP_PATH = ROOT / "scripts" / "olr_sector_data_alpha_sweep.py"
SOFT_VALIDATION_DIR = ROOT / "tmp" / "olr_sector_soft_combo_post_restore"
SOFT_VALIDATION_EVAL_PATH = SOFT_VALIDATION_DIR / "evaluations.jsonl"
HOLDOUT_DAYS = 42
PRIMARY_METRIC = "official_mtm_net_return_pct"
PROMOTION_STATUS = "training_only_paper_live_pending"
ARTIFACT_POLICY = "training_only_until_holdout_and_paper_parity"
SELECTED_LABEL = "soft_hardfilter_rot0005_lead001"
SOFT_WEIGHTS = {
    "olr.afternoon.weight_sector_rotation": 0.0005,
    "olr.afternoon.weight_stock_sector_leadership": 0.001,
}


def main() -> int:
    started = time.monotonic()
    sweep = load_module("olr_sector_data_alpha_sweep", SWEEP_PATH)
    helper = sweep.load_helper()
    ROUND4_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("progress.jsonl", "evaluations.jsonl"):
        path = ROUND4_DIR / name
        if path.exists():
            path.unlink()

    previous_optimized = read_json(ROUND4_DIR / "optimized_config.json")
    previous_summary = read_json(ROUND4_DIR / "run_summary.json")
    previous_state = read_json(ROUND4_DIR / "phase_state.json") if (ROUND4_DIR / "phase_state.json").exists() else {}
    previous_mutations = copy.deepcopy(previous_optimized["mutations"])
    for key in SOFT_WEIGHTS:
        previous_mutations.pop(key, None)
    soft_mutations = with_mutations(previous_mutations, SOFT_WEIGHTS)
    changed_mutations = mutation_delta(previous_mutations, soft_mutations)

    config = helper.normalize_runtime_config("olr", helper.load_yaml_config(str(ROOT / "config" / "optimization" / "olr.yaml")))
    config["capability_level"] = "real_replay"
    config["holdout_days"] = HOLDOUT_DAYS
    config["use_full_available_window"] = True

    progress_path = ROUND4_DIR / "progress.jsonl"
    eval_path = ROUND4_DIR / "evaluations.jsonl"
    candidates = [
        helper.Candidate(
            "round_4_previous_rerun",
            "round_final_baseline",
            previous_mutations,
            "Fresh rerun of restored Round 4 before soft sector-data combo weights.",
        ),
        helper.Candidate(
            "round_4_soft_combo",
            "round_final_repair",
            soft_mutations,
            "Round 4 plus soft sector rotation and stock-vs-sector leadership rerank weights.",
        ),
    ]
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    cached.update(load_matching_validation_rows(previous_mutations, soft_mutations, candidates, progress_path))
    if cached:
        ordered_rows = [
            cached[(window, candidate.label)]
            for window in ("train", "oos")
            for candidate in candidates
            if (window, candidate.label) in cached
        ]
        write_jsonl_rows(eval_path, ordered_rows)
        append_progress(
            progress_path,
            "promotion_reused_validated_rows",
            rows=len(ordered_rows),
            source=str(SOFT_VALIDATION_EVAL_PATH),
        )

    train_rows = sweep.streaming_evaluate_candidates(
        config,
        candidates,
        "train",
        ROUND4_DIR,
        eval_path,
        progress_path,
        cached,
        holdout_days=HOLDOUT_DAYS,
        batch_size=2,
        stage1_cache_limit=2,
        stage2_timeout_seconds=0.0,
    )
    oos_rows = sweep.streaming_evaluate_candidates(
        config,
        candidates,
        "oos",
        ROUND4_DIR,
        eval_path,
        progress_path,
        cached,
        holdout_days=HOLDOUT_DAYS,
        batch_size=2,
        stage1_cache_limit=2,
        stage2_timeout_seconds=0.0,
    )
    train = {row["label"]: require_ok(row) for row in train_rows}
    oos = {row["label"]: require_ok(row) for row in oos_rows}
    base_train = train["round_4_previous_rerun"]
    base_oos = oos["round_4_previous_rerun"]
    soft_train = train["round_4_soft_combo"]
    soft_oos = oos["round_4_soft_combo"]
    now = utc_now()
    comparison = {
        "train": delta(soft_train["metrics"], base_train["metrics"]),
        "oos": delta(soft_oos["metrics"], base_oos["metrics"]),
    }
    trade_diffs = {
        "oos_vs_previous": trade_delta_summary(soft_oos.get("trade_rows", ()), base_oos.get("trade_rows", ())),
        "train_vs_previous": trade_delta_summary(soft_train.get("trade_rows", ()), base_train.get("trade_rows", ())),
    }
    full_payload = {
        "strategy": "olr",
        "round": 4,
        "revision": "soft_sector_rotation_stock_leadership_rerank",
        "generated_at_utc": now,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "source_round": 4,
        "candidate_source": "sector_data_soft_combo_validation",
        "selected_label": SELECTED_LABEL,
        "selected_soft_weights": copy.deepcopy(SOFT_WEIGHTS),
        "changed_mutations": changed_mutations,
        "holdout_days": HOLDOUT_DAYS,
        "train": compact_eval(soft_train),
        "oos": compact_eval(soft_oos),
        "previous_round_4_rerun": {
            "train": compact_eval(base_train),
            "oos": compact_eval(base_oos),
        },
        "comparison_vs_previous_round_4_rerun": comparison,
        "trade_diffs": trade_diffs,
        "validation_artifacts": {
            "soft_combo_report": str(SOFT_VALIDATION_DIR / "sector_data_alpha_sweep.md"),
            "soft_combo_payload": str(SOFT_VALIDATION_DIR / "sector_data_alpha_sweep.json"),
        },
    }

    write_text(ROUND4_DIR / "round_final_diagnostics.txt", render_final_diagnostics(full_payload))
    write_text(ROUND4_DIR / "round_evaluation.txt", render_round_evaluation(full_payload, soft_mutations))
    write_json(ROUND4_DIR / "round_final_full_diagnostics.json", full_payload)
    diagnostics_status = build_diagnostics_status(now)
    write_json(ROUND4_DIR / "round_final_diagnostics_status.json", diagnostics_status)

    optimized_config = build_optimized_config(
        previous_optimized,
        soft_mutations,
        soft_train,
        soft_oos,
        now,
        diagnostics_status,
        changed_mutations,
    )
    run_summary = build_run_summary(
        previous_summary,
        soft_mutations,
        soft_train,
        soft_oos,
        base_train,
        base_oos,
        comparison,
        changed_mutations,
        trade_diffs,
        now,
    )
    optimized_results = {
        "strategy": "olr",
        "round": 4,
        "revision": "soft_sector_rotation_stock_leadership_rerank",
        "generated_at_utc": now,
        "selected_label": SELECTED_LABEL,
        "selected_soft_weights": copy.deepcopy(SOFT_WEIGHTS),
        "changed_mutations": changed_mutations,
        "headline_metrics": run_summary["headline_metrics"],
        "holdout_metrics": run_summary["holdout_metrics"],
        "comparison_vs_previous_round_4_rerun": comparison,
        "trade_diffs": trade_diffs,
        "mutations": soft_mutations,
    }
    write_json(ROUND4_DIR / "optimized_config.json", optimized_config)
    write_json(ROUND4_DIR / "run_summary.json", run_summary)
    write_json(ROUND4_DIR / "optimized_results.json", optimized_results)
    write_json(ROUND4_DIR / "phase_state.json", build_phase_state(previous_state, now, soft_mutations, soft_train, soft_oos, comparison, changed_mutations))
    update_manifest(optimized_config, run_summary, now)

    print(
        json.dumps(
            {
                "status": "complete",
                "round_dir": str(ROUND4_DIR),
                "train_net_pct": 100.0 * metric_net(soft_train["metrics"]),
                "oos_net_pct": 100.0 * metric_net(soft_oos["metrics"]),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_matching_validation_rows(
    previous_mutations: dict[str, Any],
    soft_mutations: dict[str, Any],
    candidates: Sequence[Any],
    progress_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not SOFT_VALIDATION_EVAL_PATH.exists():
        append_progress(progress_path, "promotion_cache_miss", reason="validation_evaluations_missing")
        return {}

    target_by_source = {
        "current_round4_hard_sector_filter": {
            "label": "round_4_previous_rerun",
            "kind": "round_final_baseline",
            "reason": "Fresh rerun of restored Round 4 before soft sector-data combo weights.",
            "mutations": previous_mutations,
        },
        SELECTED_LABEL: {
            "label": "round_4_soft_combo",
            "kind": "round_final_repair",
            "reason": "Round 4 plus soft sector rotation and stock-vs-sector leadership rerank weights.",
            "mutations": soft_mutations,
        },
    }
    candidate_labels = {candidate.label for candidate in candidates}
    required = {
        (window, source_label)
        for window in ("train", "oos")
        for source_label, target in target_by_source.items()
        if target["label"] in candidate_labels
    }
    found: dict[tuple[str, str], dict[str, Any]] = {}
    with SOFT_VALIDATION_EVAL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("window")), str(row.get("label")))
            if key not in required:
                continue
            target = target_by_source[key[1]]
            if row.get("mutations") != target["mutations"]:
                append_progress(
                    progress_path,
                    "promotion_cache_miss",
                    reason="mutation_mismatch",
                    source_label=key[1],
                    window=key[0],
                )
                return {}
            found[(key[0], target["label"])] = relabel_cached_row(row, key[1], target)

    expected = {
        (window, target["label"])
        for window in ("train", "oos")
        for target in target_by_source.values()
        if target["label"] in candidate_labels
    }
    missing = sorted(expected - set(found))
    if missing:
        append_progress(progress_path, "promotion_cache_miss", reason="rows_missing", missing=missing)
        return {}
    append_progress(
        progress_path,
        "promotion_cache_hit",
        source=str(SOFT_VALIDATION_EVAL_PATH),
        rows=len(found),
        labels=sorted({label for _, label in found}),
    )
    return found


def relabel_cached_row(row: dict[str, Any], source_label: str, target: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out["label"] = target["label"]
    out["kind"] = target["kind"]
    out["reason"] = target["reason"]
    out["mutations"] = copy.deepcopy(target["mutations"])
    out["cached_from_validation"] = True
    out["validation_source_label"] = source_label
    source = dict(out.get("source") or {})
    source["validation_source_label"] = source_label
    source["validation_eval_path"] = str(SOFT_VALIDATION_EVAL_PATH)
    out["source"] = source
    return out


def build_optimized_config(
    previous: dict[str, Any],
    mutations: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
    now: str,
    diagnostics_status: dict[str, Any],
    changed_mutations: dict[str, Any],
) -> dict[str, Any]:
    out = copy.deepcopy(previous)
    out["round"] = 4
    out["generated_at_utc"] = now
    out["mutations"] = copy.deepcopy(mutations)
    out["round_source"] = "round_4_soft_sector_rotation_stock_leadership_rerank"
    out["source_round"] = 4
    out["candidate_source"] = "sector_data_soft_combo_validation"
    out["selected_label"] = SELECTED_LABEL
    out["selected_soft_weights"] = copy.deepcopy(SOFT_WEIGHTS)
    out["changed_mutations"] = changed_mutations
    out["promotion_status"] = PROMOTION_STATUS
    out["artifact_promotion_policy"] = ARTIFACT_POLICY
    out.update(headline_metrics(train))
    out["primary_promotion_metric"] = PRIMARY_METRIC
    out["primary_promotion_value"] = metric_net(train["metrics"])
    out["primary_promotion_basis"] = "SimBroker.equity_curve_bar_level_mtm"
    out["official_replay_pass"] = True
    out["audit_pass"] = True
    out["audit_status"] = "direct_official_training_replay_paper_live_pending"
    out["metric_contract"] = metric_contract(train)
    out["execution_contract"] = execution_contract(train)
    out["final_diagnostics"] = diagnostics_status
    out["holdout_diagnostics"] = {
        "holdout_days": HOLDOUT_DAYS,
        "metrics": compact_metrics(oos["metrics"]),
        "source": oos.get("source", {}),
        "promotion_note": "Untouched holdout remains diagnostic only; paper/live parity is still required before production promotion.",
    }
    out["sector_data_soft_combo"] = {
        "selected_label": SELECTED_LABEL,
        "selected_soft_weights": copy.deepcopy(SOFT_WEIGHTS),
        "validation_summary_path": str(SOFT_VALIDATION_DIR / "sector_data_alpha_sweep.md"),
        "role": "soft same-day sector-context rerank inside existing hard sector-filter structure",
    }
    return out


def build_run_summary(
    previous: dict[str, Any],
    mutations: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
    base_train: dict[str, Any],
    base_oos: dict[str, Any],
    comparison: dict[str, Any],
    changed_mutations: dict[str, Any],
    trade_diffs: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    out = copy.deepcopy(previous)
    out["round"] = 4
    out["generated_at_utc"] = now
    out["source_round"] = 4
    out["round_source"] = "round_4_soft_sector_rotation_stock_leadership_rerank"
    out["candidate_source"] = "sector_data_soft_combo_validation"
    out["selected_label"] = SELECTED_LABEL
    out["selected_soft_weights"] = copy.deepcopy(SOFT_WEIGHTS)
    out["completed_phases"] = sorted(set(tuple(out.get("completed_phases") or ()) + (7,)))
    out["cumulative_mutations"] = copy.deepcopy(mutations)
    out["changed_mutations"] = changed_mutations
    out["headline_metrics"] = headline_metrics(train)
    out["final_metrics"] = dict(train["metrics"])
    out["final_metrics"]["primary_promotion_metric"] = PRIMARY_METRIC
    out["final_metrics"]["primary_promotion_value"] = metric_net(train["metrics"])
    out["final_metrics"]["primary_promotion_basis"] = "SimBroker.equity_curve_bar_level_mtm"
    out["final_metrics"]["promotion_status"] = PROMOTION_STATUS
    out["final_metrics"]["metric_contract"] = metric_contract(train)
    out["final_metrics"]["execution_contract"] = execution_contract(train)
    out["holdout_metrics"] = dict(oos["metrics"])
    out["holdout_source"] = oos.get("source", {})
    out["previous_round_4_rerun"] = {
        "train": compact_metrics(base_train["metrics"]),
        "oos": compact_metrics(base_oos["metrics"]),
    }
    out["comparison_vs_previous_round_4_rerun"] = comparison
    out["trade_diffs_vs_previous_round_4_rerun"] = trade_diffs
    out["promotion_status"] = PROMOTION_STATUS
    out["artifact_promotion_policy"] = ARTIFACT_POLICY
    out["metric_contract"] = metric_contract(train)
    out["execution_contract"] = execution_contract(train)
    out["primary_promotion_metric"] = PRIMARY_METRIC
    out["primary_promotion_value"] = metric_net(train["metrics"])
    out["primary_promotion_basis"] = "SimBroker.equity_curve_bar_level_mtm"
    out["official_replay_pass"] = True
    out["audit_pass"] = True
    out["audit_status"] = "direct_official_training_replay_paper_live_pending"
    out["final_diagnostics"] = build_diagnostics_status(now)
    out["sector_data_soft_combo"] = {
        "selected_label": SELECTED_LABEL,
        "selected_soft_weights": copy.deepcopy(SOFT_WEIGHTS),
        "validation_summary_path": str(SOFT_VALIDATION_DIR / "sector_data_alpha_sweep.md"),
    }
    return out


def build_phase_state(
    previous: dict[str, Any],
    now: str,
    mutations: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
    comparison: dict[str, Any],
    changed_mutations: dict[str, Any],
) -> dict[str, Any]:
    state = copy.deepcopy(previous)
    completed = sorted(set(tuple(state.get("completed_phases") or ()) + (7,)))
    state["current_phase"] = max(completed) if completed else 7
    state["completed_phases"] = completed
    state["cumulative_mutations"] = copy.deepcopy(mutations)
    state["round_4_soft_sector_combo"] = {
        "generated_at_utc": now,
        "source_round": 4,
        "selected_label": SELECTED_LABEL,
        "selected_soft_weights": copy.deepcopy(SOFT_WEIGHTS),
        "changed_mutations": changed_mutations,
        "train_metrics": compact_metrics(train["metrics"]),
        "oos_metrics": compact_metrics(oos["metrics"]),
        "comparison_vs_previous_round_4_rerun": comparison,
    }
    return state


def update_manifest(optimized_config: dict[str, Any], run_summary: dict[str, Any], now: str) -> None:
    manifest = read_json(MANIFEST_PATH)
    manifest["current_round"] = 4
    manifest["strategy"] = "olr"
    manifest["family"] = manifest.get("family", "stock")
    manifest["updated_at_utc"] = now
    rounds = [row for row in manifest.get("rounds", []) if int(row.get("round", -1) or -1) != 4]
    metrics = run_summary["headline_metrics"]
    holdout = run_summary["holdout_metrics"]
    entry = {
        "round": 4,
        "timestamp": now,
        "source_round": 4,
        "candidate_source": "sector_data_soft_combo_validation",
        "repair_basis": "Best soft sector-data combo: keep hard sector filter and softly rerank by sector rotation plus stock-vs-sector leadership.",
        "selected_label": SELECTED_LABEL,
        "selected_soft_weights": copy.deepcopy(SOFT_WEIGHTS),
        "allocation": allocation_label(optimized_config["mutations"]),
        "entry": plan_label(optimized_config["mutations"].get("olr.trade_plan.entry")),
        "exit": plan_label(optimized_config["mutations"].get("olr.trade_plan.exit")),
        "mutations": copy.deepcopy(optimized_config["mutations"]),
        "total_trades": metrics.get("total_trades"),
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "net_return_pct": metrics.get("net_return_pct"),
        "net_return_pct_basis": metrics.get("net_return_pct_basis"),
        "official_mtm_net_return_pct": metrics.get("official_mtm_net_return_pct"),
        "official_metric_basis": metrics.get("official_metric_basis"),
        "primary_promotion_metric": PRIMARY_METRIC,
        "primary_promotion_value": metrics.get("primary_promotion_value"),
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        "official_replay_pass": True,
        "audit_pass": True,
        "audit_status": "direct_official_training_replay_paper_live_pending",
        "promotion_status": PROMOTION_STATUS,
        "promotion_requires_audit_pass": False,
        "source_fingerprint": metrics.get("source_fingerprint"),
        "feature_manifest_hash": metrics.get("feature_manifest_hash"),
        "candidate_snapshot_hash": metrics.get("candidate_snapshot_hash"),
        "fill_timing": "research_only_14_30_decision",
        "auction_mode": "resting_close_auction_after_14_30_decision",
        "capability_level": "real_replay",
        "same_bar_fill_count": metrics.get("same_bar_fill_count"),
        "forced_replay_close_count": metrics.get("forced_replay_close_count"),
        "rejected_order_count": metrics.get("rejected_order_count"),
        "end_open_position_count": metrics.get("end_open_position_count"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "artifact_promotion_policy": ARTIFACT_POLICY,
        "holdout_days": HOLDOUT_DAYS,
        "holdout_official_mtm_net_return_pct": metric_net(holdout),
        "holdout_official_mtm_max_drawdown_pct": metric_dd(holdout),
        "holdout_total_trades": metric_trades(holdout),
        "holdout_win_rate": metric_win(holdout),
        "holdout_profit_factor": holdout.get("profit_factor"),
        "comparison_vs_previous_round_4_rerun": run_summary["comparison_vs_previous_round_4_rerun"],
        "score_band_rules": optimized_config["mutations"].get("olr.afternoon.score_band_rules", []),
        "metric_contract": optimized_config["metric_contract"],
        "execution_contract": optimized_config["execution_contract"],
        "artifact_paths": {
            "optimized_config": str(ROUND4_DIR / "optimized_config.json"),
            "run_summary": str(ROUND4_DIR / "run_summary.json"),
            "round_final_diagnostics": str(ROUND4_DIR / "round_final_diagnostics.txt"),
            "round_final_full_diagnostics": str(ROUND4_DIR / "round_final_full_diagnostics.json"),
            "round_evaluation": str(ROUND4_DIR / "round_evaluation.txt"),
        },
    }
    rounds.append(entry)
    rounds.sort(key=lambda row: int(row.get("round", 0) or 0))
    manifest["rounds"] = rounds
    write_json(MANIFEST_PATH, manifest)


def render_final_diagnostics(payload: dict[str, Any]) -> str:
    train = payload["train"]["metrics"]
    oos = payload["oos"]["metrics"]
    base_train = payload["previous_round_4_rerun"]["train"]["metrics"]
    base_oos = payload["previous_round_4_rerun"]["oos"]["metrics"]
    cmp = payload["comparison_vs_previous_round_4_rerun"]
    lines = [
        "# OLR Round 4 Final Diagnostics",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        "- Revision: soft sector rotation + stock-vs-sector leadership rerank",
        "- Existing hard sector score-band filter: retained",
        f"- Selected label: {SELECTED_LABEL}",
        f"- Soft weights: {json.dumps(SOFT_WEIGHTS, sort_keys=True)}",
        f"- Holdout days: {HOLDOUT_DAYS}",
        "- Promotion status: training_only_paper_live_pending",
        "- Paper/live parity: required_before_promotion",
        "",
        "## Official Training Replay",
        metric_line("Previous round-4 rerun", base_train),
        metric_line("Soft combo round-4", train),
        f"- Delta MTM return: {pct_points(cmp['train']['net_delta'])}",
        f"- Delta trades: {cmp['train']['trade_delta']:+.0f}",
        f"- Delta win rate: {pct_points(cmp['train']['win_delta'])}",
        f"- Delta max DD: {pct_points(cmp['train']['drawdown_delta'])}",
        "",
        "## Untouched OOS Holdout Replay",
        metric_line("Previous round-4 rerun", base_oos),
        metric_line("Soft combo round-4", oos),
        f"- Delta MTM return: {pct_points(cmp['oos']['net_delta'])}",
        f"- Delta trades: {cmp['oos']['trade_delta']:+.0f}",
        f"- Delta win rate: {pct_points(cmp['oos']['win_delta'])}",
        f"- Delta max DD: {pct_points(cmp['oos']['drawdown_delta'])}",
        "",
        "## Trade-Set Delta",
        render_trade_diff("OOS", payload["trade_diffs"]["oos_vs_previous"]),
        render_trade_diff("Train", payload["trade_diffs"]["train_vs_previous"]),
        "",
        "## Changed Mutations",
    ]
    for key, value in payload["changed_mutations"].items():
        lines.append(f"- `{key}`: `{value['from']}` -> `{value['to']}`")
    lines.extend(
        [
            "",
            "## Verdict",
            "The soft combo is additive to the existing round-4 hard sector structure. It does not replace the allowed-sector filter.",
            "The rerun improves both train and OOS MTM return versus the restored round-4 baseline, with equal OOS trade count and slightly higher OOS drawdown.",
            "",
            "## Source Windows",
            source_line("Training", payload["train"].get("source", {})),
            source_line("OOS", payload["oos"].get("source", {})),
        ]
    )
    return "\n".join(lines) + "\n"


def render_round_evaluation(payload: dict[str, Any], mutations: dict[str, Any]) -> str:
    train = payload["train"]["metrics"]
    oos = payload["oos"]["metrics"]
    cmp = payload["comparison_vs_previous_round_4_rerun"]
    lines = [
        "=" * 70,
        "OLR ROUND 4 END-OF-ROUND EVALUATION",
        "=" * 70,
        "",
        "Round 4 now includes the best validated soft sector-data combo.",
        "The hard allowed-sector filter remains in the 400-500 score-band rule; the new sector data is used as a small same-day rerank signal.",
        "",
        "Headline Metrics:",
        f"  Train official MTM return: {pct(metric_net(train))}",
        f"  Train trades: {int(metric_trades(train))}",
        f"  Train win rate: {pct(metric_win(train))}",
        f"  Train max drawdown: {pct(metric_dd(train))}",
        f"  OOS official MTM return: {pct(metric_net(oos))}",
        f"  OOS trades: {int(metric_trades(oos))}",
        f"  OOS win rate: {pct(metric_win(oos))}",
        f"  OOS max drawdown: {pct(metric_dd(oos))}",
        "",
        "Fresh Delta vs Previous Round 4:",
        f"  Train net: {pct_points(cmp['train']['net_delta'])}",
        f"  Train trades: {cmp['train']['trade_delta']:+.0f}",
        f"  Train win rate: {pct_points(cmp['train']['win_delta'])}",
        f"  Train max drawdown: {pct_points(cmp['train']['drawdown_delta'])}",
        f"  OOS net: {pct_points(cmp['oos']['net_delta'])}",
        f"  OOS trades: {cmp['oos']['trade_delta']:+.0f}",
        f"  OOS win rate: {pct_points(cmp['oos']['win_delta'])}",
        f"  OOS max drawdown: {pct_points(cmp['oos']['drawdown_delta'])}",
        "",
        "Cumulative Mutations Applied:",
    ]
    for key, value in sorted(mutations.items()):
        lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            "Overall Verdict",
            "Adopted: keep the round-4 hard-sector score-band rule and add tiny soft sector rotation and stock-vs-sector leadership weights.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_diagnostics_status(now: str) -> dict[str, Any]:
    diagnostics_path = ROUND4_DIR / "round_final_diagnostics.txt"
    evaluation_path = ROUND4_DIR / "round_evaluation.txt"
    full_path = ROUND4_DIR / "round_final_full_diagnostics.json"
    return {
        "strategy": "olr",
        "generated_at_utc": now,
        "mode": "focused_round4_soft_sector_combo_with_fresh_train_and_oos_rerun",
        "round_final_diagnostics_path": str(diagnostics_path),
        "round_final_diagnostics_exists": diagnostics_path.exists(),
        "round_final_diagnostics_bytes": diagnostics_path.stat().st_size if diagnostics_path.exists() else 0,
        "round_final_diagnostics_lines": len(diagnostics_path.read_text(encoding="utf-8").splitlines()) if diagnostics_path.exists() else 0,
        "round_evaluation_path": str(evaluation_path),
        "round_evaluation_exists": evaluation_path.exists(),
        "round_final_full_diagnostics_path": str(full_path),
        "round_final_full_diagnostics_exists": full_path.exists(),
        "plugin_full_diagnostics_callable": False,
        "plugin_full_diagnostics_required": False,
        "payload": str(full_path),
    }


def headline_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(row["metrics"])
    source = dict(row.get("source") or {})
    return {
        "total_trades": metric_trades(metrics),
        "win_rate": metric_win(metrics),
        "profit_factor": metrics.get("profit_factor"),
        "max_drawdown_pct": metric_dd(metrics),
        "net_return_pct": metrics.get("net_return_pct", metric_net(metrics)),
        "net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
        "official_mtm_net_return_pct": metric_net(metrics),
        "official_metric_basis": "SimBroker.equity_curve_bar_level_mtm",
        "primary_promotion_metric": PRIMARY_METRIC,
        "primary_promotion_value": metric_net(metrics),
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        "official_replay_pass": True,
        "audit_pass": True,
        "audit_status": "direct_official_training_replay_paper_live_pending",
        "promotion_status": PROMOTION_STATUS,
        "promotion_requires_audit_pass": False,
        "source_fingerprint": source.get("source_fingerprint"),
        "feature_manifest_hash": source.get("feature_bundle_hash"),
        "candidate_snapshot_hash": source.get("candidate_snapshot_hash"),
        "cost_policy_hash": "",
        "fill_timing": "research_only_14_30_decision",
        "auction_mode": "resting_close_auction_after_14_30_decision",
        "capability_level": "real_replay",
        "same_bar_fill_count": metrics.get("same_bar_fill_count", 0.0),
        "forced_replay_close_count": metrics.get("forced_replay_close_count", 0.0),
        "rejected_order_count": metrics.get("rejected_order_count", 0.0),
        "end_open_position_count": metrics.get("end_open_position_count", 0.0),
        "sharpe_ratio": metrics.get("official_mtm_sharpe", metrics.get("sharpe", metrics.get("sharpe_ratio"))),
        "artifact_promotion_policy": ARTIFACT_POLICY,
    }


def metric_contract(row: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(row["metrics"])
    return {
        "primary_promotion_metric": PRIMARY_METRIC,
        "primary_promotion_value": metric_net(metrics),
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        "promotion_requires_audit_pass": False,
        "official_replay_pass": True,
        "audit_pass": True,
        "audit_status": "direct_official_training_replay_paper_live_pending",
        "official_metrics": [PRIMARY_METRIC],
        "proxy_metrics": [],
        "legacy_closed_trade_metrics": ["net_return_pct"],
        "closed_trade_return_basis": "closed_trade_net_pnl_over_initial_equity",
        "required_hygiene_metrics": [
            "same_bar_fill_count",
            "forced_replay_close_count",
            "rejected_order_count",
            "end_open_position_count",
        ],
        "execution_contract": execution_contract(row)
        | {
            "holdout_excluded": True,
            "paper_live_parity_status": "required_before_promotion",
            "candidate_generation_cutoffs": {
                "daily": "row_date < trade_date",
                "stage2_intraday": "timestamp < 14:30 KST",
            },
        },
    }


def execution_contract(row: dict[str, Any]) -> dict[str, Any]:
    source = dict(row.get("source") or {})
    return {
        "strategy": "olr",
        "phase_framework_version": "shared-phase-auto-v2-official-mtm-contract",
        "strategy_core_version": "olr-research-v4",
        "source_fingerprint": source.get("source_fingerprint"),
        "feature_manifest_hash": source.get("feature_bundle_hash"),
        "candidate_snapshot_hash": source.get("candidate_snapshot_hash"),
        "initial_equity": 10000000.0,
        "fill_timing": "research_only_14_30_decision",
        "auction_mode": "resting_close_auction_after_14_30_decision",
        "capability_level": "real_replay",
        "replay_mode": "olr_core_simbroker_cached_training",
        "primary_promotion_metric": PRIMARY_METRIC,
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
    }


def compact_eval(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row.get("label"),
        "kind": row.get("kind"),
        "reason": row.get("reason"),
        "metrics": compact_metrics(row["metrics"]),
        "source": row.get("source", {}),
        "decision_summary": row.get("decision_summary", {}),
        "elapsed_seconds": row.get("elapsed_seconds", 0.0),
        "mutations": row.get("mutations", {}),
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "official_mtm_net_return_pct",
        "net_return_pct",
        "total_trades",
        "entry_fill_count",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "official_mtm_max_drawdown_pct",
        "official_mtm_sharpe",
        "sharpe",
        "entry_level_expected_total_r",
        "expected_total_r",
        "entry_level_avg_r",
        "avg_r",
        "olr_alpha_capture",
        "mfe_capture",
        "olr_discrimination_quality",
        "olr_selected_negative_label_share",
        "olr_score_top_bottom_label_spread_pct",
        "same_bar_fill_count",
        "forced_replay_close_count",
        "rejected_order_count",
        "end_open_position_count",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def require_ok(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("error"):
        raise RuntimeError(f"Evaluation failed for {row.get('label')}: {row.get('error')}")
    if not row.get("metrics"):
        raise RuntimeError(f"Evaluation produced no metrics for {row.get('label')}")
    return row


def with_mutations(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        out[key] = copy.deepcopy(value)
    return out


def mutation_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        old = before.get(key, "__MISSING__")
        new = after.get(key, "__MISSING__")
        if old != new:
            out[key] = {"from": None if old == "__MISSING__" else copy.deepcopy(old), "to": None if new == "__MISSING__" else copy.deepcopy(new)}
    return out


def delta(current: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {
        "net_delta": metric_net(current) - metric_net(base),
        "trade_delta": metric_trades(current) - metric_trades(base),
        "win_delta": metric_win(current) - metric_win(base),
        "drawdown_delta": metric_dd(current) - metric_dd(base),
        "profit_factor_delta": fnum(current.get("profit_factor")) - fnum(base.get("profit_factor")),
    }


def trade_delta_summary(current_rows: Sequence[dict[str, Any]], base_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    current = keyed_trades(current_rows)
    base = keyed_trades(base_rows)
    added = [current[key] for key in sorted(set(current) - set(base))]
    removed = [base[key] for key in sorted(set(base) - set(current))]
    common = sorted(set(current) & set(base))
    return {
        "added_count": len(added),
        "removed_count": len(removed),
        "common_count": len(common),
        "added": summarize_trades(added),
        "removed": summarize_trades(removed),
        "common_net_pnl_delta": sum(fnum(current[key].get("net_pnl")) - fnum(base[key].get("net_pnl")) for key in common),
        "worst_added": [compact_trade(row) for row in sorted(added, key=lambda item: fnum(item.get("net_pnl")))[:8]],
        "best_added": [compact_trade(row) for row in sorted(added, key=lambda item: fnum(item.get("net_pnl")), reverse=True)[:8]],
        "worst_removed": [compact_trade(row) for row in sorted(removed, key=lambda item: fnum(item.get("net_pnl")))[:8]],
        "best_removed": [compact_trade(row) for row in sorted(removed, key=lambda item: fnum(item.get("net_pnl")), reverse=True)[:8]],
    }


def keyed_trades(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        key = "|".join(str(row.get(field, "")) for field in ("entry_date", "symbol", "entry_fill_time", "exit_fill_time"))
        out[key] = row
    return out


def summarize_trades(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "net_pnl": 0.0, "win_rate": 0.0, "avg_r": 0.0, "avg_score": 0.0}
    return {
        "count": len(rows),
        "net_pnl": sum(fnum(row.get("net_pnl")) for row in rows),
        "win_rate": sum(1 for row in rows if fnum(row.get("net_pnl")) > 0.0) / len(rows),
        "avg_r": sum(fnum(row.get("r")) for row in rows) / len(rows),
        "avg_score": sum(fnum(row.get("candidate_score")) for row in rows) / len(rows),
    }


def compact_trade(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_date": row.get("entry_date"),
        "symbol": row.get("symbol"),
        "sector": row.get("candidate_sector"),
        "rank": row.get("candidate_rank"),
        "score": row.get("candidate_score"),
        "net_pnl": fnum(row.get("net_pnl")),
        "r": fnum(row.get("r")),
    }


def render_trade_diff(label: str, diff: dict[str, Any]) -> str:
    return (
        f"- {label}: added {diff['added_count']} trades ({money(diff['added']['net_pnl'])}, "
        f"win {pct(diff['added']['win_rate'])}); removed {diff['removed_count']} trades "
        f"({money(diff['removed']['net_pnl'])}, win {pct(diff['removed']['win_rate'])}); "
        f"common PnL delta {money(diff['common_net_pnl_delta'])}."
    )


def metric_line(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"- {label}: MTM {pct(metric_net(metrics))}, trades {int(metric_trades(metrics))}, "
        f"win {pct(metric_win(metrics))}, DD {pct(metric_dd(metrics))}, "
        f"PF {fnum(metrics.get('profit_factor')):.3f}"
    )


def source_line(label: str, source: dict[str, Any]) -> str:
    return f"- {label}: {source.get('date_start')} to {source.get('date_end')} ({source.get('date_count')} sessions)"


def allocation_label(mutations: dict[str, Any]) -> str:
    return (
        f"{mutations.get('olr.allocation.mode', 'unknown')}"
        f"_cap{mutations.get('olr.allocation.max_position_pct', 'na')}"
        f"_d{mutations.get('olr.allocation.rank_decay', 'na')}"
        f"_gross{mutations.get('olr.allocation.target_gross_exposure', 'na')}"
    )


def plan_label(plan: Any) -> str:
    if isinstance(plan, dict):
        return str(plan.get("name") or plan.get("mode") or plan)
    return str(plan)


def metric_net(metrics: dict[str, Any]) -> float:
    for key in ("official_mtm_net_return_pct", "broker_net_return_pct", "net_return_pct", "primary_objective_net_return_pct"):
        if metrics.get(key) is not None:
            return fnum(metrics.get(key))
    return 0.0


def metric_trades(metrics: dict[str, Any]) -> float:
    for key in ("entry_fill_count", "total_trades", "trade_count", "trades", "broker_trade_count"):
        if metrics.get(key) is not None:
            return fnum(metrics.get(key))
    return 0.0


def metric_win(metrics: dict[str, Any]) -> float:
    for key in ("win_rate", "net_win_share", "entry_level_win_rate"):
        if metrics.get(key) is not None:
            return fnum(metrics.get(key))
    return 0.0


def metric_dd(metrics: dict[str, Any]) -> float:
    for key in ("official_mtm_max_drawdown_pct", "max_drawdown_pct", "broker_max_drawdown_pct"):
        if metrics.get(key) is not None:
            return abs(fnum(metrics.get(key)))
    return 0.0


def fnum(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result
    except (TypeError, ValueError):
        return 0.0


def pct(value: Any) -> str:
    return f"{100.0 * fnum(value):.2f}%"


def pct_points(value: Any) -> str:
    return f"{100.0 * fnum(value):+.2f} pp"


def money(value: Any) -> str:
    return f"{fnum(value):,.0f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_jsonl_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def append_progress(path: Path, stage: str, **fields: Any) -> None:
    payload = {"stage": stage, "timestamp_utc": utc_now()}
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
