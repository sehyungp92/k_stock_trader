from __future__ import annotations

import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.types import Experiment
from backtests.strategies.kalcb.fixed_trade_plan_phase import (
    KALCBFixedTradePlanOptimizationPlugin,
    SOURCE_PATH_MUTATION,
    SOURCE_RANK_MUTATION,
    SOURCE_SECTION_MUTATION,
    _validation_gate_reject_reason,
    get_phase_candidates,
    score_fixed,
)


ROOT = Path(".")
ROUND4_DIR = ROOT / "data/backtests/output/kalcb/round_4"
ROUND5_DIR = ROOT / "data/backtests/output/kalcb/round_5"
OUT_DIR = ROUND5_DIR / "structural_rerun"
OUT_JSON = OUT_DIR / "round5_structural_rerun_train_validation.json"
OUT_CSV = OUT_DIR / "round5_structural_rerun_train_validation.csv"
PROGRESS = OUT_DIR / "progress.json"

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
    "same_bar_fill_count",
    "forced_replay_close_count",
    "rejected_order_count",
    "end_open_position_count",
    "immutable_score",
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = yaml.safe_load((ROOT / "config/optimization/kalcb.yaml").read_text(encoding="utf-8")) or {}
    config["workers"] = 2
    config["validation_gate_enabled"] = False
    config["skip_initial_baseline_eval"] = True

    round4 = json.loads((ROUND4_DIR / "optimized_config.json").read_text(encoding="utf-8"))
    round5 = json.loads((ROUND5_DIR / "optimized_config.json").read_text(encoding="utf-8"))
    round4_mutations = dict(round4["mutations"])
    round5_mutations = dict(round5["mutations"])
    source = _fixed_source(round4_mutations)

    train_config = deepcopy(config)
    train_config["fixed_candidate_source"] = source
    validation_config = deepcopy(config)
    holdout = dict(validation_config.get("baseline") or {})
    validation_config["start"] = str(holdout["holdout_start"])
    validation_config["end"] = str(holdout["holdout_end"])
    validation_config["use_full_available_window"] = True
    validation_config["fixed_candidate_source"] = source
    validation_config["validation_gate_enabled"] = False

    write_progress(started, "initialising_train_plugin", 0, 0)
    train_plugin = KALCBFixedTradePlanOptimizationPlugin(train_config, output_dir=OUT_DIR / "train_replay", max_workers=2)
    write_progress(started, "initialising_validation_plugin", 0, 0)
    validation_plugin = KALCBFixedTradePlanOptimizationPlugin(validation_config, output_dir=OUT_DIR / "validation_replay", max_workers=2)

    write_progress(started, "evaluating_round4_round5_baselines", 0, 0)
    round4_train = compact(train_plugin.evaluate_mutations(round4_mutations))
    round4_validation = compact(validation_plugin.evaluate_mutations(round4_mutations))
    round5_train = compact(train_plugin.evaluate_mutations(round5_mutations))
    round5_validation = compact(validation_plugin.evaluate_mutations(round5_mutations))
    no_drift = {
        "round4_train_broker_net_delta_vs_artifact": round4_train["broker_net_return_pct"] - float((round4.get("metric_contract") or {}).get("primary_promotion_value") or 0.0),
        "round4_validation_broker_net_delta_vs_artifact": round4_validation["broker_net_return_pct"] - float(((round4.get("oos_validation") or {}).get("broker_net_return_pct") or 0.0)),
        "round4_validation_trade_count_delta_vs_artifact": round4_validation["trade_count"] - float(((round4.get("oos_validation") or {}).get("trade_count") or 0.0)),
    }

    current = dict(round5_mutations)
    accepted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    candidates = [(phase, candidate) for phase in (9, 10, 11) for candidate in get_phase_candidates(phase)]
    total = len(candidates)

    for phase in (9, 10, 11):
        phase_candidates = get_phase_candidates(phase)
        phase_base_train = compact(train_plugin.evaluate_mutations(current))
        phase_base_validation = compact(validation_plugin.evaluate_mutations(current))
        phase_rows: list[dict[str, Any]] = []
        for candidate in phase_candidates:
            write_progress(started, f"evaluating_phase{phase}_{candidate.name}", len(rows), total)
            mutations = dict(current)
            mutations.update(candidate.mutations or {})
            row = evaluate_pair(
                phase,
                candidate,
                train_plugin,
                validation_plugin,
                mutations,
                phase_base_train,
                phase_base_validation,
            )
            phase_rows.append(row)
            rows.append(row)
            write_payload(
                started,
                round4,
                round5,
                round4_train,
                round4_validation,
                round5_train,
                round5_validation,
                no_drift,
                rows,
                accepted,
                current,
            )
        viable = [row for row in phase_rows if row.get("accepted_candidate")]
        if viable:
            best = max(viable, key=lambda row: float(row.get("validation_probe_score", 0.0) or 0.0))
            current.update(best["mutations"])
            accepted.append(
                {
                    "phase": phase,
                    "candidate": best["candidate"],
                    "mutation_hash": best["mutation_hash"],
                    "validation_probe_score": best["validation_probe_score"],
                    "train_delta": best["train_delta"],
                    "validation_delta": best["validation_delta"],
                    "mutations": best["mutations"],
                }
            )

    final_train = compact(train_plugin.evaluate_mutations(current))
    final_validation = compact(validation_plugin.evaluate_mutations(current))
    payload = write_payload(
        started,
        round4,
        round5,
        round4_train,
        round4_validation,
        round5_train,
        round5_validation,
        no_drift,
        rows,
        accepted,
        current,
        final_train=final_train,
        final_validation=final_validation,
    )
    write_csv(rows)
    write_progress(started, "done", total, total)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")


def _fixed_source(mutations: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": mutations[SOURCE_PATH_MUTATION],
        "section": mutations[SOURCE_SECTION_MUTATION],
        "rank": mutations[SOURCE_RANK_MUTATION],
    }


def evaluate_pair(
    phase: int,
    candidate: Experiment,
    train_plugin: KALCBFixedTradePlanOptimizationPlugin,
    validation_plugin: KALCBFixedTradePlanOptimizationPlugin,
    mutations: dict[str, Any],
    baseline_train: dict[str, Any],
    baseline_validation: dict[str, Any],
) -> dict[str, Any]:
    try:
        train = compact(train_plugin.evaluate_mutations(mutations))
        validation = compact(validation_plugin.evaluate_mutations(mutations))
        train_delta = metric_delta(train, baseline_train)
        validation_delta = metric_delta(validation, baseline_validation)
        reject = _validation_gate_reject_reason(train, baseline_train, validation, baseline_validation)
        score = validation_probe_score(train_delta, validation_delta, baseline_train, baseline_validation)
        return {
            "phase": phase,
            "candidate": candidate.name,
            "mutation_hash": stable_signature(candidate.mutations),
            "mutations": dict(candidate.mutations or {}),
            "train_metrics": train,
            "validation_metrics": validation,
            "train_delta": train_delta,
            "validation_delta": validation_delta,
            "validation_gate_reject_reason": reject,
            "accepted_candidate": not bool(reject) and score > 0.0,
            "validation_probe_score": score,
        }
    except Exception as exc:
        return {
            "phase": phase,
            "candidate": candidate.name,
            "mutation_hash": stable_signature(candidate.mutations),
            "mutations": dict(candidate.mutations or {}),
            "invalid": True,
            "error": f"{type(exc).__name__}: {exc}",
            "accepted_candidate": False,
            "validation_probe_score": -999.0,
        }


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}
    out["immutable_score"] = metrics.get("immutable_score", score_fixed(metrics))
    return out


def metric_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float)):
            out[key] = float(value) - float(baseline[key])
    return out


def validation_probe_score(
    train_delta: dict[str, Any],
    validation_delta: dict[str, Any],
    baseline_train: dict[str, Any],
    baseline_validation: dict[str, Any],
) -> float:
    train_trades = max(float(baseline_train.get("trade_count", 0.0) or 0.0), 1.0)
    validation_trades = max(float(baseline_validation.get("trade_count", 0.0) or 0.0), 1.0)
    train_freq = float(train_delta.get("trade_count", 0.0) or 0.0) / train_trades
    validation_freq = float(validation_delta.get("trade_count", 0.0) or 0.0) / validation_trades
    dd_penalty = max(float(train_delta.get("broker_max_drawdown_pct", 0.0) or 0.0), 0.0) + max(
        float(validation_delta.get("broker_max_drawdown_pct", 0.0) or 0.0), 0.0
    )
    return 100.0 * (
        0.28 * squash(float(train_delta.get("broker_net_return_pct", 0.0) or 0.0), 0.08)
        + 0.28 * squash(float(validation_delta.get("broker_net_return_pct", 0.0) or 0.0), 0.04)
        + 0.14 * squash(float(train_delta.get("avg_trade_net_pct", 0.0) or 0.0), 0.0025)
        + 0.12 * squash(float(validation_delta.get("avg_trade_net_pct", 0.0) or 0.0), 0.0025)
        + 0.10 * squash(0.65 * train_freq + 0.35 * validation_freq, 0.20)
        - 0.08 * squash(dd_penalty, 0.012)
    )


def squash(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    x = max(min(float(value) / float(scale), 5.0), -5.0)
    return x / (1.0 + abs(x))


def write_payload(
    started: float,
    round4: dict[str, Any],
    round5: dict[str, Any],
    round4_train: dict[str, Any],
    round4_validation: dict[str, Any],
    round5_train: dict[str, Any],
    round5_validation: dict[str, Any],
    no_drift: dict[str, Any],
    rows: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    current_mutations: dict[str, Any],
    *,
    final_train: dict[str, Any] | None = None,
    final_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_count = len(rows)
    accepted_candidates = [row for row in rows if row.get("accepted_candidate")]
    payload = {
        "artifact": "kalcb_round5_structural_rerun_train_validation",
        "elapsed_seconds": round(time.time() - started, 3),
        "round4_baseline": {"train": round4_train, "validation": round4_validation},
        "round5_baseline": {"train": round5_train, "validation": round5_validation},
        "round5_vs_round4_delta": {
            "train": metric_delta(round5_train, round4_train),
            "validation": metric_delta(round5_validation, round4_validation),
        },
        "round4_no_drift_check": no_drift,
        "round4_mutation_hash": stable_signature(round4.get("mutations") or {}),
        "round5_mutation_hash": stable_signature(round5.get("mutations") or {}),
        "final_mutation_hash": stable_signature(current_mutations),
        "accepted": accepted,
        "final_train": final_train,
        "final_validation": final_validation,
        "rows": rows,
        "summary": {
            "candidate_pairs": pair_count,
            "accepted_candidate_rows": len(accepted_candidates),
            "accepted_phase_mutations": len(accepted),
            "invalid_candidate_rows": sum(1 for row in rows if row.get("invalid")),
            "best_by_validation_probe_score": _best_row(rows),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("invalid")]
    if not valid:
        return {}
    best = max(valid, key=lambda row: float(row.get("validation_probe_score", -999.0) or -999.0))
    return {
        "phase": best.get("phase"),
        "candidate": best.get("candidate"),
        "validation_probe_score": best.get("validation_probe_score"),
        "validation_gate_reject_reason": best.get("validation_gate_reject_reason"),
        "train_delta": best.get("train_delta"),
        "validation_delta": best.get("validation_delta"),
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "candidate",
        "mutation_hash",
        "accepted_candidate",
        "validation_gate_reject_reason",
        "validation_probe_score",
        "train_broker_net_delta",
        "validation_broker_net_delta",
        "train_dd_delta",
        "validation_dd_delta",
        "train_trade_delta",
        "validation_trade_delta",
        "train_avg_trade_delta",
        "validation_avg_trade_delta",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            train_delta = row.get("train_delta") or {}
            validation_delta = row.get("validation_delta") or {}
            writer.writerow(
                {
                    "phase": row.get("phase"),
                    "candidate": row.get("candidate"),
                    "mutation_hash": row.get("mutation_hash"),
                    "accepted_candidate": row.get("accepted_candidate"),
                    "validation_gate_reject_reason": row.get("validation_gate_reject_reason") or row.get("error"),
                    "validation_probe_score": row.get("validation_probe_score"),
                    "train_broker_net_delta": train_delta.get("broker_net_return_pct"),
                    "validation_broker_net_delta": validation_delta.get("broker_net_return_pct"),
                    "train_dd_delta": train_delta.get("broker_max_drawdown_pct"),
                    "validation_dd_delta": validation_delta.get("broker_max_drawdown_pct"),
                    "train_trade_delta": train_delta.get("trade_count"),
                    "validation_trade_delta": validation_delta.get("trade_count"),
                    "train_avg_trade_delta": train_delta.get("avg_trade_net_pct"),
                    "validation_avg_trade_delta": validation_delta.get("avg_trade_net_pct"),
                }
            )


def write_progress(started: float, stage: str, done: int, total: int) -> None:
    PROGRESS.write_text(
        json.dumps(
            {
                "stage": stage,
                "done": done,
                "total": total,
                "elapsed_seconds": round(time.time() - started, 3),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
