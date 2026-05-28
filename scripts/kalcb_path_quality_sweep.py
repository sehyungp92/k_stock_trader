from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.fixed_trade_plan_phase import (
    KALCBFixedTradePlanOptimizationPlugin,
    SOURCE_PATH_MUTATION,
    SOURCE_RANK_MUTATION,
    SOURCE_SECTION_MUTATION,
    _mutation_key,
    score_fixed,
)


ROOT = Path(".")
ROUND4_DIR = ROOT / "data/backtests/output/kalcb/round_4"
ROUND5_DIR = ROOT / "data/backtests/output/kalcb/round_5"
OUT_DIR = ROUND5_DIR / "path_quality_sweep"
OUT_JSON = OUT_DIR / "path_quality_sweep_results.json"
OUT_MD = OUT_DIR / "path_quality_sweep_report.md"

METRIC_KEYS = (
    "broker_net_return_pct",
    "official_mtm_net_return_pct",
    "broker_max_drawdown_pct",
    "trade_count",
    "active_days",
    "avg_trade_net_pct",
    "net_win_share",
    "avg_mfe_capture",
    "mae_le_neg_1_share",
    "avg_mfe_r",
    "avg_mae_r",
    "worst_fold_net",
    "candidate_pool_conversion",
    "initial_active_conversion",
    "same_bar_fill_count",
    "end_open_position_count",
    "immutable_score",
)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "config/optimization/kalcb.yaml").read_text(encoding="utf-8")) or {}
    config["workers"] = 2
    config["validation_gate_enabled"] = False
    config["skip_initial_baseline_eval"] = True

    round4 = _read_json(ROUND4_DIR / "optimized_config.json")
    round5 = _read_json(ROUND5_DIR / "optimized_config.json")
    source = _fixed_source(round5["mutations"])
    train_config = deepcopy(config)
    train_config["fixed_candidate_source"] = source
    validation_config = deepcopy(config)
    holdout = dict(validation_config.get("baseline") or {})
    validation_config["start"] = str(holdout["holdout_start"])
    validation_config["end"] = str(holdout["holdout_end"])
    validation_config["use_full_available_window"] = True
    validation_config["validation_gate_enabled"] = False
    validation_config["fixed_candidate_source"] = source

    _progress(started, "initialising_train")
    train_plugin = KALCBFixedTradePlanOptimizationPlugin(train_config, output_dir=OUT_DIR / "train_replay", max_workers=2)
    _progress(started, "initialising_holdout")
    holdout_plugin = KALCBFixedTradePlanOptimizationPlugin(validation_config, output_dir=OUT_DIR / "holdout_replay", max_workers=2)

    base_mutations = dict(round5["mutations"])
    round4_mutations = dict(round4["mutations"])
    _progress(started, "evaluating_baselines")
    base_train, base_train_rows = _evaluate(train_plugin, base_mutations)
    base_holdout, base_holdout_rows = _evaluate(holdout_plugin, base_mutations)
    round4_train, _ = _evaluate(train_plugin, round4_mutations)
    round4_holdout, _ = _evaluate(holdout_plugin, round4_mutations)

    experiments = _experiments(base_mutations)
    rows: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments, start=1):
        name = experiment["name"]
        mutations = dict(base_mutations)
        mutations.update(experiment["mutations"])
        _progress(started, f"evaluating_{index:02d}_{name}", index=index, total=len(experiments))
        row = _evaluate_experiment(
            train_plugin,
            holdout_plugin,
            name,
            mutations,
            experiment,
            base_train,
            base_holdout,
        )
        rows.append(row)
        _write_payload(started, source, base_train, base_holdout, round4_train, round4_holdout, base_train_rows, base_holdout_rows, rows)

    payload = _write_payload(started, source, base_train, base_holdout, round4_train, round4_holdout, base_train_rows, base_holdout_rows, rows)
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    _progress(started, "done", index=len(experiments), total=len(experiments))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_source(mutations: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": mutations[SOURCE_PATH_MUTATION],
        "section": mutations[SOURCE_SECTION_MUTATION],
        "rank": int(mutations[SOURCE_RANK_MUTATION]),
    }


def _evaluate(plugin: KALCBFixedTradePlanOptimizationPlugin, mutations: dict[str, Any]) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    metrics = plugin.evaluate_mutations(mutations)
    detail = plugin._evaluation_details[_mutation_key(mutations)]
    return dict(metrics), tuple(detail.trade_rows)


def _evaluate_experiment(
    train_plugin: KALCBFixedTradePlanOptimizationPlugin,
    holdout_plugin: KALCBFixedTradePlanOptimizationPlugin,
    name: str,
    mutations: dict[str, Any],
    experiment: dict[str, Any],
    base_train: dict[str, Any],
    base_holdout: dict[str, Any],
) -> dict[str, Any]:
    try:
        train, train_rows = _evaluate(train_plugin, mutations)
        holdout, holdout_rows = _evaluate(holdout_plugin, mutations)
        train_compact = _compact(train)
        holdout_compact = _compact(holdout)
        train_delta = _delta(train_compact, _compact(base_train))
        holdout_delta = _delta(holdout_compact, _compact(base_holdout))
        return {
            "name": name,
            "family": experiment.get("family", ""),
            "thesis": experiment.get("thesis", ""),
            "mutation_hash": stable_signature(experiment["mutations"]),
            "mutations": experiment["mutations"],
            "train": train_compact,
            "holdout": holdout_compact,
            "train_delta": train_delta,
            "holdout_delta": holdout_delta,
            "train_path": _path_summary(train_rows),
            "holdout_path": _path_summary(holdout_rows),
            "score": _research_score(train_delta, holdout_delta),
            "promotable_research_candidate": _promotable(train_compact, holdout_compact, train_delta, holdout_delta),
        }
    except Exception as exc:
        return {
            "name": name,
            "family": experiment.get("family", ""),
            "thesis": experiment.get("thesis", ""),
            "mutations": experiment["mutations"],
            "error": f"{type(exc).__name__}: {exc}",
            "score": -999.0,
            "promotable_research_candidate": False,
        }


def _anchor_route() -> dict[str, Any]:
    return {
        "name": "first30_anchor",
        "priority": 0,
        "mode": "first30_open",
        "require_initial_active": True,
    }


def _frontier_route(name: str, mode: str, **kwargs: Any) -> dict[str, Any]:
    route = {
        "name": name,
        "priority": 1,
        "mode": mode,
        "require_initial_active": False,
        "route_risk_mult": 0.30,
        "route_notional_mult": 0.35,
        "route_participation_mult": 0.50,
        "route_max_session_trades": 1,
        "max_frontier_rank": 12,
        "min_first30_rel_volume": 4.0,
        "min_first30_signal_cpr": 0.80,
        "min_first30_range_atr": 0.75,
        "min_quality_votes": 7,
        "quality_min_bar_ret": 0.015,
        "quality_min_first30_rel_volume": 3.0,
        "quality_min_first30_signal_cpr": 0.80,
        "quality_min_first30_range_atr": 0.75,
        "quality_min_flow_score": -0.05,
        "quality_min_accumulation_score": 0.0,
        "quality_max_frontier_rank": 12,
        "max_signal_bars": 18,
        "after_bar": 1,
    }
    route.update(kwargs)
    return route


def _experiments(base: dict[str, Any]) -> list[dict[str, Any]]:
    del base
    anchor = _anchor_route()
    deferred_12 = _frontier_route("frontier_deferred_rank12", "deferred_continuation", min_bar_ret=0.0)
    deferred_20_regime = _frontier_route(
        "frontier_deferred_regime_rank20",
        "deferred_continuation",
        max_frontier_rank=20,
        route_risk_mult=0.22,
        route_notional_mult=0.28,
        min_first30_rel_volume=5.0,
        quality_max_frontier_rank=20,
        context_min={"session_first30_positive_share": 0.52, "session_first30_rel_volume_median": 2.2},
        context_max={"session_first30_gap_dispersion": 0.12},
    )
    or_reclaim_12 = _frontier_route(
        "frontier_or_high_reclaim_rank12",
        "or_high_reclaim",
        min_reclaim_ret=0.0,
        max_pullback_from_vwap_pct=0.006,
    )
    avwap_reclaim_12 = _frontier_route(
        "frontier_avwap_reclaim_rank12",
        "avwap_reclaim",
        min_reclaim_ret=0.001,
        max_pullback_from_vwap_pct=0.004,
        min_first30_rel_volume=4.5,
    )

    return [
        {
            "name": "path_stale_mfe_vwap",
            "family": "path_management",
            "thesis": "Exit proven moves only after stale MFE and VWAP failure.",
            "mutations": {
                "kalcb.exit.path_quality_enabled": True,
                "kalcb.exit.path_quality_min_hold_bars": 18,
                "kalcb.exit.path_quality_min_mfe_r": 8.0,
                "kalcb.exit.path_quality_min_giveback_r": 6.0,
                "kalcb.exit.path_quality_min": {"bars_since_mfe": 5, "below_vwap_streak": 1},
                "kalcb.exit.path_quality_max": {"vwap_ret": -0.001},
            },
        },
        {
            "name": "path_stale_mfe_or_mid",
            "family": "path_management",
            "thesis": "Exit after proof when price loses the opening range midline.",
            "mutations": {
                "kalcb.exit.path_quality_enabled": True,
                "kalcb.exit.path_quality_min_hold_bars": 18,
                "kalcb.exit.path_quality_min_mfe_r": 8.0,
                "kalcb.exit.path_quality_min_giveback_r": 7.0,
                "kalcb.exit.path_quality_min": {"bars_since_mfe": 5, "below_or_mid_streak": 1},
                "kalcb.exit.path_quality_max": {"or_position": 0.52},
            },
        },
        {
            "name": "path_recent_down_after_tail",
            "family": "path_management",
            "thesis": "Protect tail winners after a three-bar downshift without cutting early noise.",
            "mutations": {
                "kalcb.exit.path_quality_enabled": True,
                "kalcb.exit.path_quality_min_hold_bars": 24,
                "kalcb.exit.path_quality_min_mfe_r": 10.0,
                "kalcb.exit.path_quality_min_giveback_r": 8.0,
                "kalcb.exit.path_quality_min": {"recent3_down_count": 2, "bars_since_mfe": 3},
            },
        },
        {
            "name": "path_early_vwap_failure",
            "family": "discrimination_management",
            "thesis": "Reject the worst accepted paths after entry if they quickly lose VWAP and OR structure.",
            "mutations": {
                "kalcb.exit.path_quality_enabled": True,
                "kalcb.exit.path_quality_min_hold_bars": 8,
                "kalcb.exit.path_quality_max_hold_bars": 18,
                "kalcb.exit.path_quality_max": {"current_r": -1.5, "vwap_ret": -0.001, "or_position": 0.45},
            },
        },
        {
            "name": "path_early_mae_or_mid_failure",
            "family": "discrimination_management",
            "thesis": "Cut structurally painful entries if early MAE and OR-mid failure agree.",
            "mutations": {
                "kalcb.exit.path_quality_enabled": True,
                "kalcb.exit.path_quality_min_hold_bars": 8,
                "kalcb.exit.path_quality_max_hold_bars": 20,
                "kalcb.exit.path_quality_min": {"below_or_mid_streak": 2},
                "kalcb.exit.path_quality_max": {"current_r": -2.0, "or_position": 0.50},
            },
        },
        {
            "name": "path_rank4_10_stale_vwap",
            "family": "cohort_management",
            "thesis": "Target the giveback-prone middle-rank cohort without touching rank-1 tails.",
            "mutations": {
                "kalcb.exit.path_quality_enabled": True,
                "kalcb.exit.path_quality_min_hold_bars": 18,
                "kalcb.exit.path_quality_min_mfe_r": 5.0,
                "kalcb.exit.path_quality_min_giveback_r": 5.0,
                "kalcb.exit.path_quality_min": {"frontier_rank": 4, "bars_since_mfe": 4},
                "kalcb.exit.path_quality_max": {"frontier_rank": 10, "vwap_ret": -0.001},
            },
        },
        {
            "name": "conditional_stop_8_4_h18",
            "family": "path_management",
            "thesis": "Use a real protective stop after proof instead of a market giveback exit.",
            "mutations": {
                "kalcb.exit.conditional_stop_activate_r": 8.0,
                "kalcb.exit.conditional_stop_gap_r": 4.0,
                "kalcb.exit.conditional_stop_min_hold_bars": 18,
            },
        },
        {
            "name": "conditional_stop_12_6_h24",
            "family": "path_management",
            "thesis": "Looser proof stop aimed at protecting only established tail winners.",
            "mutations": {
                "kalcb.exit.conditional_stop_activate_r": 12.0,
                "kalcb.exit.conditional_stop_gap_r": 6.0,
                "kalcb.exit.conditional_stop_min_hold_bars": 24,
            },
        },
        {
            "name": "vwap_fail_after_5r",
            "family": "path_management",
            "thesis": "Exit proved moves on VWAP loss after 5R MFE.",
            "mutations": {
                "kalcb.exit.vwap_fail_bars": 1,
                "kalcb.exit.vwap_fail_pct": 0.0,
                "kalcb.exit.vwap_fail_after_mfe_r": 5.0,
            },
        },
        {
            "name": "vwap_fail_after_10r_two_bars",
            "family": "path_management",
            "thesis": "Require two-bar VWAP failure after large proof to avoid cutting tails.",
            "mutations": {
                "kalcb.exit.vwap_fail_bars": 2,
                "kalcb.exit.vwap_fail_pct": 0.0,
                "kalcb.exit.vwap_fail_after_mfe_r": 10.0,
            },
        },
        {
            "name": "failed_followthrough_persistent_same",
            "family": "early_discrimination",
            "thesis": "Existing failed-followthrough should remain live after the checkpoint, not one exact bar only.",
            "mutations": {
                "kalcb.exit.failed_followthrough_persistent": True,
            },
        },
        {
            "name": "failed_followthrough_soft_persistent",
            "family": "early_discrimination",
            "thesis": "Softer persistent followthrough cut, previously train-positive, retested under the new structural sweep.",
            "mutations": {
                "kalcb.exit.failed_followthrough_bars": 8,
                "kalcb.exit.failed_followthrough_mfe_r": 1.0,
                "kalcb.exit.failed_followthrough_close_r": -0.50,
                "kalcb.exit.failed_followthrough_persistent": True,
            },
        },
        {
            "name": "partial_12r_25pct",
            "family": "path_management",
            "thesis": "Take a small partial only after large proof, trying to lift capture while preserving tail.",
            "mutations": {
                "kalcb.exit.use_partial_takes": True,
                "kalcb.exit.partial_r_trigger": 12.0,
                "kalcb.exit.partial_fraction": 0.25,
                "kalcb.exit.partial_stop_to_breakeven": True,
                "kalcb.exit.partial_breakeven_buffer_r": 0.0,
            },
        },
        {
            "name": "partial_20r_20pct",
            "family": "path_management",
            "thesis": "Even later partial for large tails only.",
            "mutations": {
                "kalcb.exit.use_partial_takes": True,
                "kalcb.exit.partial_r_trigger": 20.0,
                "kalcb.exit.partial_fraction": 0.20,
                "kalcb.exit.partial_stop_to_breakeven": True,
                "kalcb.exit.partial_breakeven_buffer_r": 0.0,
            },
        },
        {
            "name": "frontier_deferred_rank12_cap1",
            "family": "routing",
            "thesis": "Add a small reduced-risk frontier continuation branch after confirmation, not at 09:30.",
            "mutations": {
                "kalcb.entry.frontier_branch_universe": True,
                "kalcb.entry.routes": [anchor, deferred_12],
            },
        },
        {
            "name": "frontier_or_high_reclaim_rank12_cap1",
            "family": "routing",
            "thesis": "Add only pullback/reclaim continuation for non-active frontier names.",
            "mutations": {
                "kalcb.entry.frontier_branch_universe": True,
                "kalcb.entry.routes": [anchor, or_reclaim_12],
            },
        },
        {
            "name": "frontier_avwap_reclaim_rank12_cap1",
            "family": "routing",
            "thesis": "Add AVWAP reclaim branch for non-active names with strong first30 proof.",
            "mutations": {
                "kalcb.entry.frontier_branch_universe": True,
                "kalcb.entry.routes": [anchor, avwap_reclaim_12],
            },
        },
        {
            "name": "frontier_deferred_rank20_regime_cap1",
            "family": "routing",
            "thesis": "Wider frontier route only when session first30 breadth/relvol context is constructive.",
            "mutations": {
                "kalcb.entry.frontier_branch_universe": True,
                "kalcb.entry.routes": [anchor, deferred_20_regime],
            },
        },
        {
            "name": "frontier_deferred_rank12_with_path_exit",
            "family": "routing_path_combo",
            "thesis": "Only add frontier continuation if path-quality leak is also controlled.",
            "mutations": {
                "kalcb.entry.frontier_branch_universe": True,
                "kalcb.entry.routes": [anchor, deferred_12],
                "kalcb.exit.path_quality_enabled": True,
                "kalcb.exit.path_quality_min_hold_bars": 12,
                "kalcb.exit.path_quality_min_mfe_r": 4.0,
                "kalcb.exit.path_quality_min_giveback_r": 4.0,
                "kalcb.exit.path_quality_min": {"bars_since_mfe": 4, "below_vwap_streak": 1},
                "kalcb.exit.path_quality_max": {"vwap_ret": -0.001},
            },
        },
        {
            "name": "frontier_deferred_rank12_cond_stop",
            "family": "routing_path_combo",
            "thesis": "Reduced-risk frontier continuation plus proof stop.",
            "mutations": {
                "kalcb.entry.frontier_branch_universe": True,
                "kalcb.entry.routes": [anchor, deferred_12],
                "kalcb.exit.conditional_stop_activate_r": 8.0,
                "kalcb.exit.conditional_stop_gap_r": 4.0,
                "kalcb.exit.conditional_stop_min_hold_bars": 18,
            },
        },
    ]


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}
    out["immutable_score"] = metrics.get("immutable_score", score_fixed(metrics))
    return out


def _delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value) - float(baseline[key])
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float))
    }


def _research_score(train_delta: dict[str, float], holdout_delta: dict[str, float]) -> float:
    train_net = float(train_delta.get("broker_net_return_pct", 0.0))
    holdout_net = float(holdout_delta.get("broker_net_return_pct", 0.0))
    train_dd = max(float(train_delta.get("broker_max_drawdown_pct", 0.0)), 0.0)
    holdout_dd = max(float(holdout_delta.get("broker_max_drawdown_pct", 0.0)), 0.0)
    train_avg = float(train_delta.get("avg_trade_net_pct", 0.0))
    holdout_avg = float(holdout_delta.get("avg_trade_net_pct", 0.0))
    train_cap = float(train_delta.get("avg_mfe_capture", 0.0))
    holdout_cap = float(holdout_delta.get("avg_mfe_capture", 0.0))
    freq = 0.65 * float(train_delta.get("trade_count", 0.0)) / 72.0 + 0.35 * float(holdout_delta.get("trade_count", 0.0)) / 11.0
    return 100.0 * (
        0.26 * _squash(train_net, 0.07)
        + 0.26 * _squash(holdout_net, 0.035)
        + 0.13 * _squash(train_avg, 0.003)
        + 0.11 * _squash(holdout_avg, 0.003)
        + 0.10 * _squash(0.5 * train_cap + 0.5 * holdout_cap, 0.04)
        + 0.06 * _squash(freq, 0.20)
        - 0.08 * _squash(train_dd + holdout_dd, 0.012)
    )


def _promotable(train: dict[str, Any], holdout: dict[str, Any], train_delta: dict[str, float], holdout_delta: dict[str, float]) -> bool:
    del train, holdout
    if train_delta.get("broker_net_return_pct", 0.0) <= 0 or holdout_delta.get("broker_net_return_pct", 0.0) <= 0:
        return False
    if train_delta.get("avg_trade_net_pct", 0.0) < -0.0005 or holdout_delta.get("avg_trade_net_pct", 0.0) < -0.0005:
        return False
    if train_delta.get("broker_max_drawdown_pct", 0.0) > 0.006 or holdout_delta.get("broker_max_drawdown_pct", 0.0) > 0.006:
        return False
    if train_delta.get("worst_fold_net", 0.0) < -0.004:
        return False
    return True


def _path_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    r_values = [_num(row.get("r")) for row in rows]
    mfe_values = [max(_num(row.get("mfe_r")), 0.0) for row in rows]
    losers = [row for row in rows if _num(row.get("r")) < 0]
    return {
        "trades": len(rows),
        "total_r": sum(r_values),
        "avg_r": _mean(r_values),
        "win_share": sum(1 for value in r_values if value > 0) / max(len(rows), 1),
        "avg_mfe_r": _mean(mfe_values),
        "avg_mae_r": _mean([_num(row.get("mae_r")) for row in rows]),
        "loser_share": len(losers) / max(len(rows), 1),
        "loser_avg_mae_r": _mean([_num(row.get("mae_r")) for row in losers]),
        "loser_avg_mfe_r": _mean([_num(row.get("mfe_r")) for row in losers]),
        "actual_positive_mfe_capture": sum(max(value, 0.0) for value in r_values) / max(sum(mfe_values), 1e-9),
        "oracle_50pct_mfe_delta_r": sum(0.5 * value for value in mfe_values) - sum(r_values),
        "exit_counts": dict(sorted(Counter(str(row.get("exit_reason") or "unknown") for row in rows).items())),
    }


def _write_payload(
    started: float,
    source: dict[str, Any],
    base_train: dict[str, Any],
    base_holdout: dict[str, Any],
    round4_train: dict[str, Any],
    round4_holdout: dict[str, Any],
    base_train_rows: tuple[dict[str, Any], ...],
    base_holdout_rows: tuple[dict[str, Any], ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    best = sorted(valid, key=lambda item: float(item.get("score", -999.0)), reverse=True)[:10]
    promotable = [row for row in valid if row.get("promotable_research_candidate")]
    payload = {
        "elapsed_seconds": round(time.time() - started, 3),
        "source_ref": source,
        "baseline_round5": {
            "train": _compact(base_train),
            "holdout": _compact(base_holdout),
            "train_path": _path_summary(base_train_rows),
            "holdout_path": _path_summary(base_holdout_rows),
        },
        "round4_no_drift_check": {
            "train_broker_net_delta_vs_artifact": _num(round4_train.get("broker_net_return_pct")) - _num((_read_json(ROUND4_DIR / "optimized_config.json").get("metric_contract") or {}).get("primary_promotion_value")),
            "holdout_broker_net_delta_vs_artifact": _num(round4_holdout.get("broker_net_return_pct")) - _num((_read_json(ROUND4_DIR / "optimized_config.json").get("oos_validation") or {}).get("broker_net_return_pct")),
            "holdout_trade_count_delta_vs_artifact": _num(round4_holdout.get("trade_count")) - _num((_read_json(ROUND4_DIR / "optimized_config.json").get("oos_validation") or {}).get("trade_count")),
        },
        "rows": rows,
        "summary": {
            "tested": len(rows),
            "valid": len(valid),
            "promotable_research_candidates": len(promotable),
            "best": [_summary_row(row) for row in best],
            "promotable": [_summary_row(row) for row in promotable],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name"),
        "family": row.get("family"),
        "score": row.get("score"),
        "train_net_delta": (row.get("train_delta") or {}).get("broker_net_return_pct"),
        "holdout_net_delta": (row.get("holdout_delta") or {}).get("broker_net_return_pct"),
        "train_dd_delta": (row.get("train_delta") or {}).get("broker_max_drawdown_pct"),
        "holdout_dd_delta": (row.get("holdout_delta") or {}).get("broker_max_drawdown_pct"),
        "train_trade_delta": (row.get("train_delta") or {}).get("trade_count"),
        "holdout_trade_delta": (row.get("holdout_delta") or {}).get("trade_count"),
        "promotable_research_candidate": row.get("promotable_research_candidate"),
    }


def _markdown(payload: dict[str, Any]) -> str:
    base = payload["baseline_round5"]
    lines = [
        "# KALCB path-quality and routing sweep",
        "",
        f"Tested {payload['summary']['tested']} hypotheses in {payload['elapsed_seconds']}s.",
        "",
        "## Baseline",
        "",
        "| Window | Net | DD | Trades | Win | Avg MFE R | MFE capture |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Train | {_pct(base['train']['broker_net_return_pct'])} | {_pct(base['train']['broker_max_drawdown_pct'])} | {_n(base['train']['trade_count'])} | {_pct(base['train']['net_win_share'])} | {_f(base['train']['avg_mfe_r'])} | {_pct(base['train']['avg_mfe_capture'])} |",
        f"| Holdout | {_pct(base['holdout']['broker_net_return_pct'])} | {_pct(base['holdout']['broker_max_drawdown_pct'])} | {_n(base['holdout']['trade_count'])} | {_pct(base['holdout']['net_win_share'])} | {_f(base['holdout']['avg_mfe_r'])} | {_pct(base['holdout']['avg_mfe_capture'])} |",
        "",
        "## Best Rows",
        "",
        "| Rank | Candidate | Family | Train net d | Holdout net d | Train DD d | Holdout DD d | Trade d T/H | Promotable |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(payload["summary"]["best"], start=1):
        lines.append(
            f"| {index} | {row['name']} | {row['family']} | {_pct(row['train_net_delta'])} | {_pct(row['holdout_net_delta'])} | {_pct(row['train_dd_delta'])} | {_pct(row['holdout_dd_delta'])} | {_f(row['train_trade_delta'], 0)}/{_f(row['holdout_trade_delta'], 0)} | {row['promotable_research_candidate']} |"
        )
    lines.extend(["", "## All Rows", "", "| Candidate | Family | Train net d | Holdout net d | Train avg d | Holdout avg d | Train cap d | Holdout cap d | Train DD d | Holdout DD d |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in payload["rows"]:
        if "error" in row:
            lines.append(f"| {row['name']} | {row['family']} | error | error | error | error | error | error | error | error |")
            continue
        td = row["train_delta"]
        hd = row["holdout_delta"]
        lines.append(
            f"| {row['name']} | {row['family']} | {_pct(td.get('broker_net_return_pct'))} | {_pct(hd.get('broker_net_return_pct'))} | {_pct(td.get('avg_trade_net_pct'))} | {_pct(hd.get('avg_trade_net_pct'))} | {_pct(td.get('avg_mfe_capture'))} | {_pct(hd.get('avg_mfe_capture'))} | {_pct(td.get('broker_max_drawdown_pct'))} | {_pct(hd.get('broker_max_drawdown_pct'))} |"
        )
    return "\n".join(lines) + "\n"


def _squash(value: float, scale: float) -> float:
    return math.tanh(float(value) / max(float(scale), 1e-9))


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(values)) if values else 0.0


def _pct(value: Any) -> str:
    return f"{100.0 * _num(value):.2f}%"


def _f(value: Any, places: int = 2) -> str:
    return f"{_num(value):.{places}f}"


def _n(value: Any) -> str:
    return str(int(round(_num(value))))


def _progress(started: float, stage: str, **extra: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "elapsed_seconds": round(time.time() - started, 3), **extra}
    (OUT_DIR / "progress.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
