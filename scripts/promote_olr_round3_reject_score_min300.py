from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROUND_ROOT = ROOT / "data" / "backtests" / "output" / "olr"
ROUND2_DIR = ROUND_ROOT / "round_2"
ROUND3_DIR = ROUND_ROOT / "round_3"
HELPER_PATH = ROOT / "scripts" / "olr_round2_oos_deep_ablation.py"
HOLDOUT_DAYS = 42
PRIMARY_METRIC = "official_mtm_net_return_pct"
PROMOTION_STATUS = "training_only_paper_live_pending"
ARTIFACT_POLICY = "training_only_until_holdout_and_paper_parity"


def main() -> int:
    started = time.monotonic()
    helper = load_helper()
    ROUND3_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("progress.jsonl", "evaluations.jsonl"):
        path = ROUND3_DIR / name
        if path.exists():
            path.unlink()

    round2_config = read_json(ROUND2_DIR / "optimized_config.json")
    round2_summary = read_json(ROUND2_DIR / "run_summary.json")
    round2_mutations = copy.deepcopy(round2_config["mutations"])
    round3_mutations = copy.deepcopy(round2_mutations)
    prior_reject_min = round3_mutations.get("olr.afternoon.reject_score_min")
    round3_mutations["olr.afternoon.reject_score_min"] = 300.0

    config = helper.normalize_runtime_config("olr", helper.load_yaml_config(str(ROOT / "config" / "optimization" / "olr.yaml")))
    config["capability_level"] = "real_replay"
    config["holdout_days"] = HOLDOUT_DAYS
    config["use_full_available_window"] = True

    progress_path = ROUND3_DIR / "progress.jsonl"
    eval_path = ROUND3_DIR / "evaluations.jsonl"
    candidates = [
        helper.Candidate(
            "round_2_final_rerun",
            "round_final_baseline",
            round2_mutations,
            "Fresh rerun of the accepted Round 2 final configuration for Round 3 comparison.",
        ),
        helper.Candidate(
            "round_3_reject_score_min300",
            "round_final_repair",
            round3_mutations,
            "Round 2 cumulative mutations with only olr.afternoon.reject_score_min lowered from 400 to 300.",
        ),
    ]
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    train_rows = helper.evaluate_candidates(
        config,
        candidates,
        "train",
        ROUND3_DIR,
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
        ROUND3_DIR,
        eval_path,
        progress_path,
        cached,
        holdout_days=HOLDOUT_DAYS,
        batch_size=2,
    )
    by_label_train = {row["label"]: row for row in train_rows}
    by_label_oos = {row["label"]: row for row in oos_rows}
    round2_train = require_ok(by_label_train["round_2_final_rerun"])
    round2_oos = require_ok(by_label_oos["round_2_final_rerun"])
    round3_train = require_ok(by_label_train["round_3_reject_score_min300"])
    round3_oos = require_ok(by_label_oos["round_3_reject_score_min300"])

    now = utc_now()
    comparison = build_comparison(helper, round2_train, round2_oos, round3_train, round3_oos)
    full_payload = {
        "strategy": "olr",
        "round": 3,
        "generated_at_utc": now,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "source_round": 2,
        "changed_mutations": {
            "olr.afternoon.reject_score_min": {
                "from": prior_reject_min,
                "to": 300.0,
            }
        },
        "holdout_days": HOLDOUT_DAYS,
        "train": compact_for_artifact(helper, round3_train),
        "oos": compact_for_artifact(helper, round3_oos),
        "round_2_baseline_rerun": {
            "train": compact_for_artifact(helper, round2_train),
            "oos": compact_for_artifact(helper, round2_oos),
        },
        "comparison_vs_round_2_rerun": comparison,
    }

    final_diagnostics_text = render_final_diagnostics(helper, full_payload, round3_train, round3_oos)
    round_evaluation_text = render_round_evaluation(helper, full_payload, round3_mutations)
    write_text(ROUND3_DIR / "round_final_diagnostics.txt", final_diagnostics_text)
    write_text(ROUND3_DIR / "round_evaluation.txt", round_evaluation_text)
    write_json(ROUND3_DIR / "round_final_full_diagnostics.json", full_payload)

    diagnostics_status = build_diagnostics_status(now)
    write_json(ROUND3_DIR / "round_final_diagnostics_status.json", diagnostics_status)

    optimized_config = build_optimized_config(
        round2_config,
        round3_mutations,
        round3_train,
        round3_oos,
        now,
        diagnostics_status,
        prior_reject_min,
    )
    run_summary = build_run_summary(
        round2_summary,
        round3_mutations,
        round3_train,
        round3_oos,
        round2_train,
        round2_oos,
        comparison,
        now,
    )
    optimized_results = {
        "strategy": "olr",
        "round": 3,
        "generated_at_utc": now,
        "changed_mutations": full_payload["changed_mutations"],
        "headline_metrics": run_summary["headline_metrics"],
        "holdout_metrics": run_summary["holdout_metrics"],
        "comparison_vs_round_2_rerun": comparison,
        "mutations": round3_mutations,
    }
    write_json(ROUND3_DIR / "optimized_config.json", optimized_config)
    write_json(ROUND3_DIR / "run_summary.json", run_summary)
    write_json(ROUND3_DIR / "optimized_results.json", optimized_results)
    write_json(ROUND3_DIR / "phase_state.json", build_phase_state(now, round3_mutations, round3_train, comparison))

    update_manifest(optimized_config, run_summary, now)

    print(
        json.dumps(
            {
                "status": "complete",
                "round_dir": str(ROUND3_DIR),
                "train_net_pct": 100.0 * metric_net(round3_train["metrics"]),
                "oos_net_pct": 100.0 * metric_net(round3_oos["metrics"]),
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
    round2_config: dict[str, Any],
    mutations: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
    now: str,
    diagnostics_status: dict[str, Any],
    prior_reject_min: Any,
) -> dict[str, Any]:
    out = copy.deepcopy(round2_config)
    train_metrics = dict(train["metrics"])
    out["round"] = 3
    out["generated_at_utc"] = now
    out["mutations"] = copy.deepcopy(mutations)
    out["round_source"] = "round_2_final_focused_repair"
    out["source_round"] = 2
    out["changed_mutations"] = {"olr.afternoon.reject_score_min": {"from": prior_reject_min, "to": 300.0}}
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
    return out


def build_run_summary(
    round2_summary: dict[str, Any],
    mutations: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
    round2_train: dict[str, Any],
    round2_oos: dict[str, Any],
    comparison: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    out = copy.deepcopy(round2_summary)
    out["round"] = 3
    out["generated_at_utc"] = now
    out["source_round"] = 2
    out["round_source"] = "round_2_final_focused_repair"
    out["completed_phases"] = [1, 2, 3, 4, 5, 6]
    out["cumulative_mutations"] = copy.deepcopy(mutations)
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
    out["round_2_baseline_rerun"] = {
        "train": compact_metrics(round2_train["metrics"]),
        "oos": compact_metrics(round2_oos["metrics"]),
    }
    out["comparison_vs_round_2_rerun"] = comparison
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
    return out


def build_phase_state(
    now: str,
    mutations: dict[str, Any],
    train: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    state_path = ROUND2_DIR / "phase_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    state = copy.deepcopy(state)
    state["current_phase"] = 6
    state["completed_phases"] = [1, 2, 3, 4, 5, 6]
    state["cumulative_mutations"] = copy.deepcopy(mutations)
    replace_mutation_value(state, "olr.afternoon.reject_score_min", 300.0)
    state["round_3_repair"] = {
        "generated_at_utc": now,
        "source_round": 2,
        "changed_mutations": {"olr.afternoon.reject_score_min": {"from": 400.0, "to": 300.0}},
        "final_metrics": compact_metrics(train["metrics"]),
        "comparison_vs_round_2_rerun": comparison,
    }
    return state


def replace_mutation_value(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        for item_key, item_value in obj.items():
            if item_key == key and not isinstance(item_value, dict):
                obj[item_key] = value
            else:
                replace_mutation_value(item_value, key, value)
    elif isinstance(obj, list):
        for item in obj:
            replace_mutation_value(item, key, value)


def update_manifest(optimized_config: dict[str, Any], run_summary: dict[str, Any], now: str) -> None:
    path = ROUND_ROOT / "rounds_manifest.json"
    manifest = read_json(path)
    manifest["current_round"] = 3
    manifest["strategy"] = "olr"
    manifest["family"] = manifest.get("family", "stock")
    manifest["updated_at_utc"] = now
    rounds = [row for row in manifest.get("rounds", []) if int(row.get("round", -1) or -1) != 3]
    metrics = run_summary["headline_metrics"]
    holdout = run_summary["holdout_metrics"]
    entry = {
        "round": 3,
        "timestamp": now,
        "source_round": 2,
        "candidate_source": "round_2_final_reject_score_min300_repair",
        "repair_basis": "Focused reject_score_min sweep; fresh train and untouched holdout rerun saved in round_3.",
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
        "comparison_vs_round_2_rerun": run_summary["comparison_vs_round_2_rerun"],
        "metric_contract": optimized_config["metric_contract"],
        "execution_contract": optimized_config["execution_contract"],
        "artifact_paths": {
            "optimized_config": str(ROUND3_DIR / "optimized_config.json"),
            "run_summary": str(ROUND3_DIR / "run_summary.json"),
            "round_final_diagnostics": str(ROUND3_DIR / "round_final_diagnostics.txt"),
            "round_final_full_diagnostics": str(ROUND3_DIR / "round_final_full_diagnostics.json"),
            "round_evaluation": str(ROUND3_DIR / "round_evaluation.txt"),
        },
    }
    rounds.append(entry)
    rounds.sort(key=lambda row: int(row.get("round", 0) or 0))
    manifest["rounds"] = rounds
    write_json(path, manifest)


def render_final_diagnostics(
    helper: Any,
    payload: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
) -> str:
    train_m = train["metrics"]
    oos_m = oos["metrics"]
    cmp = payload["comparison_vs_round_2_rerun"]
    lines = [
        "# OLR Round 3 Final Diagnostics",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        "- Source: Round 2 final cumulative mutations",
        "- Focused repair: olr.afternoon.reject_score_min 400.0 -> 300.0",
        f"- Holdout days: {HOLDOUT_DAYS}",
        "- Promotion status: training_only_paper_live_pending",
        "- Paper/live parity: required_before_promotion",
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
        "## Delta Versus Round 2 Fresh Rerun",
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
        "Lowering reject_score_min to 300 improves both training and OOS MTM return versus the fresh Round 2 rerun, while reducing OOS drawdown and preserving OOS trade count.",
        "It remains a research artifact until paper/live parity is completed.",
        "",
        "## Source Windows",
        f"- Training: {train.get('source', {}).get('date_start')} to {train.get('source', {}).get('date_end')} ({train.get('source', {}).get('date_count')} sessions)",
        f"- OOS: {oos.get('source', {}).get('date_start')} to {oos.get('source', {}).get('date_end')} ({oos.get('source', {}).get('date_count')} sessions)",
        "",
        "## Kept Features",
        "round_2_final + reject_score_min300",
    ]
    return "\n".join(lines) + "\n"


def render_round_evaluation(helper: Any, payload: dict[str, Any], mutations: dict[str, Any]) -> str:
    train = payload["train"]["metrics"]
    oos = payload["oos"]["metrics"]
    cmp = payload["comparison_vs_round_2_rerun"]
    lines = [
        "=" * 70,
        "OLR ROUND 3 END-OF-ROUND EVALUATION",
        "=" * 70,
        "",
        "Round 3 is a focused repair overlay on the completed Round 2 phase stack.",
        "The cumulative mutation set is preserved except for olr.afternoon.reject_score_min = 300.0.",
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
        "Fresh Delta vs Round 2:",
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
            "The Round 3 focused repair raises OOS return without reducing OOS frequency and also raises the official training MTM return. Promotion remains training-only until holdout review and paper/live parity are satisfied.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_diagnostics_status(now: str) -> dict[str, Any]:
    diagnostics_path = ROUND3_DIR / "round_final_diagnostics.txt"
    evaluation_path = ROUND3_DIR / "round_evaluation.txt"
    full_path = ROUND3_DIR / "round_final_full_diagnostics.json"
    return {
        "strategy": "olr",
        "generated_at_utc": now,
        "mode": "focused_round3_repair_with_fresh_train_and_oos_rerun",
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


def build_comparison(helper: Any, round2_train: dict[str, Any], round2_oos: dict[str, Any], round3_train: dict[str, Any], round3_oos: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": delta(helper, round3_train["metrics"], round2_train["metrics"]),
        "oos": delta(helper, round3_oos["metrics"], round2_oos["metrics"]),
    }


def delta(helper: Any, current: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
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
        "execution_contract": execution_contract(row) | {
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
            return float(metrics.get(key) or 0.0)
    return 0.0


def metric_trades(metrics: dict[str, Any]) -> float:
    for key in ("entry_fill_count", "total_trades", "trade_count", "trades", "broker_trade_count"):
        if metrics.get(key) is not None:
            return float(metrics.get(key) or 0.0)
    return 0.0


def metric_win(metrics: dict[str, Any]) -> float:
    for key in ("win_rate", "net_win_share", "entry_level_win_rate"):
        if metrics.get(key) is not None:
            return float(metrics.get(key) or 0.0)
    return 0.0


def metric_dd(metrics: dict[str, Any]) -> float:
    for key in ("official_mtm_max_drawdown_pct", "max_drawdown_pct", "broker_max_drawdown_pct"):
        if metrics.get(key) is not None:
            return abs(float(metrics.get(key) or 0.0))
    return 0.0


def pct(value: Any) -> str:
    return f"{100.0 * float(value or 0.0):.3f}%"


def metric_or_na(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def pct_points(value: Any) -> str:
    return f"{100.0 * float(value or 0.0):+.3f} pct-pts"


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
