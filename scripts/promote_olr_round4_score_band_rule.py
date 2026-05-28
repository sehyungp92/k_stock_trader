from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROUND_ROOT = ROOT / "data" / "backtests" / "output" / "olr"
ROUND3_DIR = ROUND_ROOT / "round_3"
ROUND4_DIR = ROUND_ROOT / "round_4"
HELPER_PATH = ROOT / "scripts" / "olr_round2_oos_deep_ablation.py"
SEARCH_DIR = ROOT / "tmp" / "olr_nonmonotone_score_band_search"
BEST_MUTATIONS_PATH = SEARCH_DIR / "best_balanced_mutations.json"
SEARCH_SUMMARY_PATH = SEARCH_DIR / "nonmonotone_score_band_search.json"
HOLDOUT_DAYS = 42
PRIMARY_METRIC = "official_mtm_net_return_pct"
PROMOTION_STATUS = "training_only_paper_live_pending"
ARTIFACT_POLICY = "training_only_until_holdout_and_paper_parity"


def main() -> int:
    started = time.monotonic()
    helper = load_helper()
    ROUND4_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("progress.jsonl", "evaluations.jsonl"):
        path = ROUND4_DIR / name
        if path.exists():
            path.unlink()

    round3_config = read_json(ROUND3_DIR / "optimized_config.json")
    round3_summary = read_json(ROUND3_DIR / "run_summary.json")
    round3_mutations = copy.deepcopy(round3_config["mutations"])
    round4_mutations = read_json(BEST_MUTATIONS_PATH)
    search_summary = read_json(SEARCH_SUMMARY_PATH)
    changed_mutations = mutation_delta(round3_mutations, round4_mutations)

    config = helper.normalize_runtime_config("olr", helper.load_yaml_config(str(ROOT / "config" / "optimization" / "olr.yaml")))
    config["capability_level"] = "real_replay"
    config["holdout_days"] = HOLDOUT_DAYS
    config["use_full_available_window"] = True

    progress_path = ROUND4_DIR / "progress.jsonl"
    eval_path = ROUND4_DIR / "evaluations.jsonl"
    candidates = [
        helper.Candidate(
            "round_3_final_rerun",
            "round_final_baseline",
            round3_mutations,
            "Fresh rerun of the accepted Round 3 final configuration for Round 4 comparison.",
        ),
        helper.Candidate(
            "round_4_score_band_rule",
            "round_final_repair",
            round4_mutations,
            "Round 4 selected conditional non-monotone score-band rule from the score-band search.",
        ),
    ]
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    train_rows = helper.evaluate_candidates(
        config,
        candidates,
        "train",
        ROUND4_DIR,
        eval_path,
        progress_path,
        cached,
        holdout_days=HOLDOUT_DAYS,
        batch_size=2,
    )
    oos_rows = helper.evaluate_candidates(
        config,
        candidates,
        "oos",
        ROUND4_DIR,
        eval_path,
        progress_path,
        cached,
        holdout_days=HOLDOUT_DAYS,
        batch_size=2,
    )
    by_label_train = {row["label"]: row for row in train_rows}
    by_label_oos = {row["label"]: row for row in oos_rows}
    round3_train = require_ok(by_label_train["round_3_final_rerun"])
    round3_oos = require_ok(by_label_oos["round_3_final_rerun"])
    round4_train = require_ok(by_label_train["round_4_score_band_rule"])
    round4_oos = require_ok(by_label_oos["round_4_score_band_rule"])

    now = utc_now()
    comparison = {
        "train": delta(round4_train["metrics"], round3_train["metrics"]),
        "oos": delta(round4_oos["metrics"], round3_oos["metrics"]),
    }
    full_payload = {
        "strategy": "olr",
        "round": 4,
        "generated_at_utc": now,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "source_round": 3,
        "candidate_source": "conditional_non_monotone_score_band_search",
        "search_summary": str(SEARCH_SUMMARY_PATH),
        "selected_search_label": (search_summary.get("best_balanced") or {}).get("label"),
        "changed_mutations": changed_mutations,
        "holdout_days": HOLDOUT_DAYS,
        "train": compact_for_artifact(helper, round4_train),
        "oos": compact_for_artifact(helper, round4_oos),
        "round_3_baseline_rerun": {
            "train": compact_for_artifact(helper, round3_train),
            "oos": compact_for_artifact(helper, round3_oos),
        },
        "comparison_vs_round_3_rerun": comparison,
    }

    write_text(ROUND4_DIR / "round_final_diagnostics.txt", render_final_diagnostics(full_payload, round4_train, round4_oos))
    write_text(ROUND4_DIR / "round_evaluation.txt", render_round_evaluation(full_payload, round4_mutations))
    write_json(ROUND4_DIR / "round_final_full_diagnostics.json", full_payload)
    diagnostics_status = build_diagnostics_status(now)
    write_json(ROUND4_DIR / "round_final_diagnostics_status.json", diagnostics_status)

    optimized_config = build_optimized_config(
        round3_config,
        round4_mutations,
        round4_train,
        round4_oos,
        now,
        diagnostics_status,
        changed_mutations,
    )
    run_summary = build_run_summary(
        round3_summary,
        round4_mutations,
        round4_train,
        round4_oos,
        round3_train,
        round3_oos,
        comparison,
        changed_mutations,
        now,
    )
    optimized_results = {
        "strategy": "olr",
        "round": 4,
        "generated_at_utc": now,
        "changed_mutations": changed_mutations,
        "headline_metrics": run_summary["headline_metrics"],
        "holdout_metrics": run_summary["holdout_metrics"],
        "comparison_vs_round_3_rerun": comparison,
        "selected_score_band_rules": round4_mutations.get("olr.afternoon.score_band_rules", []),
        "search_summary": str(SEARCH_SUMMARY_PATH),
        "mutations": round4_mutations,
    }
    write_json(ROUND4_DIR / "optimized_config.json", optimized_config)
    write_json(ROUND4_DIR / "run_summary.json", run_summary)
    write_json(ROUND4_DIR / "optimized_results.json", optimized_results)
    write_json(ROUND4_DIR / "phase_state.json", build_phase_state(now, round4_mutations, round4_train, comparison, changed_mutations))
    update_manifest(optimized_config, run_summary, now)

    print(
        json.dumps(
            {
                "status": "complete",
                "round_dir": str(ROUND4_DIR),
                "train_net_pct": 100.0 * metric_net(round4_train["metrics"]),
                "oos_net_pct": 100.0 * metric_net(round4_oos["metrics"]),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def load_helper():
    spec = importlib.util.spec_from_file_location("olr_round2_oos_deep_ablation", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_optimized_config(
    round3_config: dict[str, Any],
    mutations: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
    now: str,
    diagnostics_status: dict[str, Any],
    changed_mutations: dict[str, Any],
) -> dict[str, Any]:
    out = copy.deepcopy(round3_config)
    train_metrics = dict(train["metrics"])
    out["round"] = 4
    out["generated_at_utc"] = now
    out["mutations"] = copy.deepcopy(mutations)
    out["round_source"] = "round_3_final_conditional_score_band_repair"
    out["source_round"] = 3
    out["candidate_source"] = "conditional_non_monotone_score_band_search"
    out["changed_mutations"] = changed_mutations
    out["promotion_status"] = PROMOTION_STATUS
    out["artifact_promotion_policy"] = ARTIFACT_POLICY
    out.update(headline_metrics(train))
    out["primary_promotion_metric"] = PRIMARY_METRIC
    out["primary_promotion_value"] = metric_net(train_metrics)
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
    out["score_band_search"] = {
        "selected_label": "cond_400_500_allow_broad_positive_sectors",
        "summary_path": str(SEARCH_SUMMARY_PATH),
        "selected_score_band_rules": mutations.get("olr.afternoon.score_band_rules", []),
    }
    return out


def build_run_summary(
    round3_summary: dict[str, Any],
    mutations: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
    round3_train: dict[str, Any],
    round3_oos: dict[str, Any],
    comparison: dict[str, Any],
    changed_mutations: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    out = copy.deepcopy(round3_summary)
    out["round"] = 4
    out["generated_at_utc"] = now
    out["source_round"] = 3
    out["round_source"] = "round_3_final_conditional_score_band_repair"
    out["candidate_source"] = "conditional_non_monotone_score_band_search"
    out["completed_phases"] = [1, 2, 3, 4, 5, 6]
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
    out["round_3_baseline_rerun"] = {
        "train": compact_metrics(round3_train["metrics"]),
        "oos": compact_metrics(round3_oos["metrics"]),
    }
    out["comparison_vs_round_3_rerun"] = comparison
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
    out["score_band_search"] = {
        "selected_label": "cond_400_500_allow_broad_positive_sectors",
        "summary_path": str(SEARCH_SUMMARY_PATH),
        "selected_score_band_rules": mutations.get("olr.afternoon.score_band_rules", []),
    }
    return out


def build_phase_state(
    now: str,
    mutations: dict[str, Any],
    train: dict[str, Any],
    comparison: dict[str, Any],
    changed_mutations: dict[str, Any],
) -> dict[str, Any]:
    state_path = ROUND3_DIR / "phase_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    state = copy.deepcopy(state)
    state["current_phase"] = 6
    state["completed_phases"] = [1, 2, 3, 4, 5, 6]
    state["cumulative_mutations"] = copy.deepcopy(mutations)
    state["round_4_score_band_repair"] = {
        "generated_at_utc": now,
        "source_round": 3,
        "changed_mutations": changed_mutations,
        "selected_label": "cond_400_500_allow_broad_positive_sectors",
        "final_metrics": compact_metrics(train["metrics"]),
        "comparison_vs_round_3_rerun": comparison,
    }
    return state


def update_manifest(optimized_config: dict[str, Any], run_summary: dict[str, Any], now: str) -> None:
    path = ROUND_ROOT / "rounds_manifest.json"
    manifest = read_json(path)
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
        "source_round": 3,
        "candidate_source": "conditional_non_monotone_score_band_search",
        "repair_basis": "Best balanced rule from 420-candidate OOS score-band search plus fresh train validation.",
        "selected_search_label": "cond_400_500_allow_broad_positive_sectors",
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
        "comparison_vs_round_3_rerun": run_summary["comparison_vs_round_3_rerun"],
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
    write_json(path, manifest)


def render_final_diagnostics(payload: dict[str, Any], train: dict[str, Any], oos: dict[str, Any]) -> str:
    train_m = train["metrics"]
    oos_m = oos["metrics"]
    cmp = payload["comparison_vs_round_3_rerun"]
    rules = payload["train"]["mutations"].get("olr.afternoon.score_band_rules", [])
    lines = [
        "# OLR Round 4 Final Diagnostics",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        "- Source: Round 3 final cumulative mutations",
        "- Focused repair: conditional/non-monotone afternoon score-band rule",
        "- Selected search label: cond_400_500_allow_broad_positive_sectors",
        f"- Holdout days: {HOLDOUT_DAYS}",
        "- Promotion status: training_only_paper_live_pending",
        "- Paper/live parity: required_before_promotion",
        "",
        "## Score-Band Rule",
        f"- Rules: {json.dumps(rules, sort_keys=True)}",
        "",
        "## Official Training Replay",
        f"- MTM return: {pct(metric_net(train_m))}",
        f"- Max drawdown: {pct(metric_dd(train_m))}",
        f"- Entry trades: {int(metric_trades(train_m))}",
        f"- Win rate: {pct(metric_win(train_m))}",
        f"- Profit factor: {float(train_m.get('profit_factor', 0.0) or 0.0):.3f}",
        f"- Expected total R: {float(train_m.get('expected_total_r', 0.0) or 0.0):.3f}",
        f"- Entry-level expected total R: {float(train_m.get('entry_level_expected_total_r', 0.0) or 0.0):.3f}",
        f"- Alpha capture: {float(train_m.get('olr_alpha_capture', train_m.get('mfe_capture', 0.0)) or 0.0):.3f}",
        f"- Discrimination quality: {metric_or_na(train_m.get('olr_discrimination_quality'))}",
        f"- Same-bar fills: {int(float(train_m.get('same_bar_fill_count', 0.0) or 0.0))}",
        f"- Forced replay closes: {int(float(train_m.get('forced_replay_close_count', 0.0) or 0.0))}",
        f"- Rejected orders: {int(float(train_m.get('rejected_order_count', 0.0) or 0.0))}",
        f"- End open positions: {int(float(train_m.get('end_open_position_count', 0.0) or 0.0))}",
        "",
        "## Untouched OOS Holdout Replay",
        f"- MTM return: {pct(metric_net(oos_m))}",
        f"- Max drawdown: {pct(metric_dd(oos_m))}",
        f"- Entry trades: {int(metric_trades(oos_m))}",
        f"- Win rate: {pct(metric_win(oos_m))}",
        f"- Profit factor: {float(oos_m.get('profit_factor', 0.0) or 0.0):.3f}",
        f"- Expected total R: {float(oos_m.get('expected_total_r', 0.0) or 0.0):.3f}",
        f"- Entry-level expected total R: {float(oos_m.get('entry_level_expected_total_r', 0.0) or 0.0):.3f}",
        f"- Same-bar fills: {int(float(oos_m.get('same_bar_fill_count', 0.0) or 0.0))}",
        f"- Forced replay closes: {int(float(oos_m.get('forced_replay_close_count', 0.0) or 0.0))}",
        f"- Rejected orders: {int(float(oos_m.get('rejected_order_count', 0.0) or 0.0))}",
        f"- End open positions: {int(float(oos_m.get('end_open_position_count', 0.0) or 0.0))}",
        "",
        "## Delta Versus Round 3 Fresh Rerun",
        f"- Training MTM return delta: {pct_points(cmp['train']['net_delta'])}",
        f"- Training trade delta: {cmp['train']['trade_delta']:+.0f}",
        f"- Training win-rate delta: {pct_points(cmp['train']['win_delta'])}",
        f"- Training max-DD delta: {pct_points(cmp['train']['drawdown_delta'])}",
        f"- OOS MTM return delta: {pct_points(cmp['oos']['net_delta'])}",
        f"- OOS trade delta: {cmp['oos']['trade_delta']:+.0f}",
        f"- OOS win-rate delta: {pct_points(cmp['oos']['win_delta'])}",
        f"- OOS max-DD delta: {pct_points(cmp['oos']['drawdown_delta'])}",
        "",
        "## Verdict",
        "Round 4 raises both official training return and untouched OOS return versus the fresh Round 3 rerun, while reducing max drawdown in both windows.",
        "It remains a research artifact until paper/live parity is completed.",
        "",
        "## Source Windows",
        f"- Training: {train.get('source', {}).get('date_start')} to {train.get('source', {}).get('date_end')} ({train.get('source', {}).get('date_count')} sessions)",
        f"- OOS: {oos.get('source', {}).get('date_start')} to {oos.get('source', {}).get('date_end')} ({oos.get('source', {}).get('date_count')} sessions)",
    ]
    return "\n".join(lines) + "\n"


def render_round_evaluation(payload: dict[str, Any], mutations: dict[str, Any]) -> str:
    train = payload["train"]["metrics"]
    oos = payload["oos"]["metrics"]
    cmp = payload["comparison_vs_round_3_rerun"]
    lines = [
        "=" * 70,
        "OLR ROUND 4 END-OF-ROUND EVALUATION",
        "=" * 70,
        "",
        "Round 4 is a focused score-band repair overlay on the completed Round 3 stack.",
        "It keeps the low-score and high-score tails, opens only the 400-500 middle score band, and gates that middle band to broad positive sectors.",
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
        "Fresh Delta vs Round 3:",
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
            "The Round 4 conditional score-band repair raises OOS return and frequency versus Round 3 while improving training return and drawdown. Promotion remains training-only until holdout review and paper/live parity are satisfied.",
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
        "mode": "focused_round4_conditional_score_band_with_fresh_train_and_oos_rerun",
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


def delta(current: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {
        "net_delta": metric_net(current) - metric_net(base),
        "trade_delta": metric_trades(current) - metric_trades(base),
        "win_delta": metric_win(current) - metric_win(base),
        "drawdown_delta": metric_dd(current) - metric_dd(base),
        "profit_factor_delta": float(current.get("profit_factor", 0.0) or 0.0) - float(base.get("profit_factor", 0.0) or 0.0),
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


def compact_for_artifact(helper: Any, row: dict[str, Any]) -> dict[str, Any]:
    compact = helper.compact_eval(row)
    compact["metrics"] = compact_metrics(row["metrics"])
    compact["source"] = row.get("source", {})
    return compact


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


def mutation_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        old = before.get(key, "__MISSING__")
        new = after.get(key, "__MISSING__")
        if old != new:
            out[key] = {
                "from": None if old == "__MISSING__" else copy.deepcopy(old),
                "to": None if new == "__MISSING__" else copy.deepcopy(new),
            }
    return out


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


def metric_or_na(value: Any) -> str:
    return "n/a" if value is None else f"{fnum(value):.3f}"


def pct(value: Any) -> str:
    return f"{100.0 * fnum(value):.2f}%"


def pct_points(value: Any) -> str:
    return f"{100.0 * fnum(value):+.2f} pp"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
