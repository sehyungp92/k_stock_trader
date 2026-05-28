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
    get_phase_candidates,
    score_fixed,
)


ROOT = Path(".")
ROUND_DIR = ROOT / "data/backtests/output/kalcb/round_4"
OUT_DIR = ROUND_DIR / "redesigned_validation_probe"
OUT_JSON = OUT_DIR / "round4_redesigned_phase_train_validation_probe.json"
OUT_CSV = OUT_DIR / "round4_redesigned_phase_train_validation_probe.csv"
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

    optimized = json.loads((ROUND_DIR / "optimized_config.json").read_text(encoding="utf-8"))
    baseline_mutations = dict(optimized["mutations"])
    fixed_source = {
        "path": baseline_mutations[SOURCE_PATH_MUTATION],
        "section": baseline_mutations[SOURCE_SECTION_MUTATION],
        "rank": baseline_mutations[SOURCE_RANK_MUTATION],
    }

    train_config = deepcopy(config)
    train_config["fixed_candidate_source"] = fixed_source
    validation_config = deepcopy(config)
    holdout_window = dict(validation_config.get("baseline") or {})
    validation_config["start"] = str(holdout_window["holdout_start"])
    validation_config["end"] = str(holdout_window["holdout_end"])
    validation_config["use_full_available_window"] = True
    validation_config["fixed_candidate_source"] = fixed_source

    write_progress(started, "initialising_train_plugin", 0, 0)
    train_plugin = KALCBFixedTradePlanOptimizationPlugin(train_config, output_dir=OUT_DIR / "train_replay", max_workers=2)
    write_progress(started, "initialising_validation_plugin", 0, 0)
    validation_plugin = KALCBFixedTradePlanOptimizationPlugin(validation_config, output_dir=OUT_DIR / "validation_replay", max_workers=2)

    baseline_train = compact(train_plugin.evaluate_mutations(baseline_mutations))
    baseline_validation = compact(validation_plugin.evaluate_mutations(baseline_mutations))
    baseline_drift = {
        "train_broker_net_delta_vs_round4_artifact": float(baseline_train.get("broker_net_return_pct", 0.0) or 0.0)
        - float(((optimized.get("metric_contract") or {}).get("primary_promotion_value") or 0.0)),
        "validation_broker_net_delta_vs_round4_artifact": float(baseline_validation.get("broker_net_return_pct", 0.0) or 0.0)
        - float(((optimized.get("oos_validation") or {}).get("broker_net_return_pct") or 0.0),
        ),
        "validation_trade_count_delta_vs_round4_artifact": float(baseline_validation.get("trade_count", 0.0) or 0.0)
        - float(((optimized.get("oos_validation") or {}).get("trade_count") or 0.0),
        ),
    }

    candidates = collect_candidates()
    rows: list[dict[str, Any]] = []
    write_payload(started, optimized, baseline_train, baseline_validation, baseline_drift, rows, candidates)

    total = len(candidates)
    for index, item in enumerate(candidates, start=1):
        phase = int(item["phase"])
        candidate: Experiment = item["candidate"]
        write_progress(started, f"evaluating_p{phase}_{candidate.name}", index - 1, total)
        mutations = dict(baseline_mutations)
        mutations.update(candidate.mutations or {})
        train_row = evaluate_one("train", phase, candidate, train_plugin, mutations, baseline_train)
        validation_row = evaluate_one("validation", phase, candidate, validation_plugin, mutations, baseline_validation)
        pair = combine_pair(train_row, validation_row, baseline_train, baseline_validation)
        rows.extend([train_row, validation_row, pair])
        write_payload(started, optimized, baseline_train, baseline_validation, baseline_drift, rows, candidates)
        write_progress(started, f"completed_p{phase}_{candidate.name}", index, total)

    summary = summary_from_rows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")


def collect_candidates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for phase in range(1, 12):
        for candidate in get_phase_candidates(phase):
            key = (phase, candidate.name)
            if key in seen:
                continue
            seen.add(key)
            out.append({"phase": phase, "candidate": candidate})
    return out


def evaluate_one(
    window: str,
    phase: int,
    candidate: Experiment,
    plugin: KALCBFixedTradePlanOptimizationPlugin,
    mutations: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    try:
        metrics = compact(plugin.evaluate_mutations(mutations))
        delta = metric_delta(metrics, baseline)
        return {
            "row_type": "window",
            "phase": phase,
            "candidate": candidate.name,
            "window": window,
            "mutation_hash": stable_signature(candidate.mutations),
            "mutations": dict(candidate.mutations or {}),
            "metrics": metrics,
            "delta": delta,
            "passes_quality": passes_quality(delta),
            "passes_frequency_aware": passes_frequency_aware(delta, baseline, window),
        }
    except Exception as exc:
        return {
            "row_type": "window",
            "phase": phase,
            "candidate": candidate.name,
            "window": window,
            "mutation_hash": stable_signature(candidate.mutations),
            "mutations": dict(candidate.mutations or {}),
            "invalid": True,
            "error": f"{type(exc).__name__}: {exc}",
            "passes_quality": False,
            "passes_frequency_aware": False,
        }


def combine_pair(
    train_row: dict[str, Any],
    validation_row: dict[str, Any],
    baseline_train: dict[str, Any],
    baseline_validation: dict[str, Any],
) -> dict[str, Any]:
    train_delta = dict(train_row.get("delta") or {})
    validation_delta = dict(validation_row.get("delta") or {})
    invalid = bool(train_row.get("invalid") or validation_row.get("invalid"))
    return {
        "row_type": "pair",
        "phase": train_row.get("phase"),
        "candidate": train_row.get("candidate"),
        "mutation_hash": train_row.get("mutation_hash"),
        "mutations": dict(train_row.get("mutations") or {}),
        "train_delta": train_delta,
        "validation_delta": validation_delta,
        "train_metrics": dict(train_row.get("metrics") or {}),
        "validation_metrics": dict(validation_row.get("metrics") or {}),
        "invalid": invalid,
        "train_passes_quality": bool(train_row.get("passes_quality")),
        "validation_passes_quality": bool(validation_row.get("passes_quality")),
        "train_passes_frequency_aware": bool(train_row.get("passes_frequency_aware")),
        "validation_passes_frequency_aware": bool(validation_row.get("passes_frequency_aware")),
        "both_positive_return": (not invalid)
        and float(train_delta.get("broker_net_return_pct", 0.0) or 0.0) > 0.0
        and float(validation_delta.get("broker_net_return_pct", 0.0) or 0.0) > 0.0,
        "both_quality": (not invalid) and bool(train_row.get("passes_quality")) and bool(validation_row.get("passes_quality")),
        "both_frequency_aware": (not invalid)
        and bool(train_row.get("passes_frequency_aware"))
        and bool(validation_row.get("passes_frequency_aware")),
        "validation_probe_score": validation_probe_score(train_delta, validation_delta, baseline_train, baseline_validation),
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


def passes_quality(delta: dict[str, Any]) -> bool:
    return (
        float(delta.get("broker_net_return_pct", 0.0) or 0.0) > 0.0
        and float(delta.get("avg_trade_net_pct", 0.0) or 0.0) >= -0.0005
        and float(delta.get("broker_max_drawdown_pct", 0.0) or 0.0) <= 0.005
        and float(delta.get("same_bar_fill_count", 0.0) or 0.0) <= 0.0
        and float(delta.get("forced_replay_close_count", 0.0) or 0.0) <= 0.0
        and float(delta.get("end_open_position_count", 0.0) or 0.0) <= 0.0
    )


def passes_frequency_aware(delta: dict[str, Any], baseline: dict[str, Any], window: str) -> bool:
    base_trades = max(float(baseline.get("trade_count", 0.0) or 0.0), 1.0)
    min_trade_delta = -0.10 * base_trades if window == "train" else -max(2.0, 0.20 * base_trades)
    return passes_quality(delta) and float(delta.get("trade_count", 0.0) or 0.0) >= min_trade_delta


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
    components = {
        "train_net_delta": 0.28 * squash(float(train_delta.get("broker_net_return_pct", 0.0) or 0.0), 0.08),
        "validation_net_delta": 0.28 * squash(float(validation_delta.get("broker_net_return_pct", 0.0) or 0.0), 0.04),
        "train_avg_trade_delta": 0.14 * squash(float(train_delta.get("avg_trade_net_pct", 0.0) or 0.0), 0.0025),
        "validation_avg_trade_delta": 0.12 * squash(float(validation_delta.get("avg_trade_net_pct", 0.0) or 0.0), 0.0025),
        "frequency_delta": 0.10 * squash(0.65 * train_freq + 0.35 * validation_freq, 0.20),
        "drawdown_delta": -0.08 * squash(dd_penalty, 0.012),
    }
    return 100.0 * sum(components.values())


def squash(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    x = float(value) / float(scale)
    return max(-1.0, min(1.0, x))


def write_payload(
    started: float,
    optimized: dict[str, Any],
    baseline_train: dict[str, Any],
    baseline_validation: dict[str, Any],
    baseline_drift: dict[str, float],
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    payload = {
        "elapsed_seconds": time.time() - started,
        "strategy": "kalcb",
        "source_round": 4,
        "scope": "redesigned round_4 phase menu rerun against round_4 optimized baseline with train and validation scoring",
        "max_workers": 2,
        "score_components": {
            "official_immutable_score": "score_fixed uses seven components or fewer",
            "validation_probe_score": [
                "train_net_delta",
                "validation_net_delta",
                "train_avg_trade_delta",
                "validation_avg_trade_delta",
                "frequency_delta",
                "drawdown_delta",
            ],
        },
        "round4_artifact_hashes": {
            "config_hash": optimized.get("config_hash", ""),
            "strategy_code_hash": optimized.get("strategy_code_hash", ""),
            "source_data_fingerprint": optimized.get("source_data_fingerprint", ""),
        },
        "baseline_train": baseline_train,
        "baseline_validation": baseline_validation,
        "baseline_drift_check": baseline_drift,
        "total_candidate_definitions": len(candidates),
        "completed_candidate_definitions": len({row.get("candidate") for row in rows if row.get("row_type") == "pair"}),
        "rows": rows,
        "summary": summary_from_rows(rows),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_csv(rows)


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "row_type",
        "phase",
        "window",
        "candidate",
        "both_positive_return",
        "both_quality",
        "both_frequency_aware",
        "validation_probe_score",
        "passes_quality",
        "passes_frequency_aware",
        "net",
        "delta_net",
        "dd",
        "delta_dd",
        "trades",
        "delta_trades",
        "avg_trade",
        "delta_avg_trade",
        "capture",
        "delta_capture",
        "score",
        "delta_score",
        "mutation_hash",
        "mutations_json",
        "invalid",
        "error",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") or {}
            delta = row.get("delta") or {}
            if row.get("row_type") == "pair":
                metrics = row.get("train_metrics") or {}
                delta = row.get("train_delta") or {}
            writer.writerow(
                {
                    "row_type": row.get("row_type", ""),
                    "phase": row.get("phase", ""),
                    "window": row.get("window", ""),
                    "candidate": row.get("candidate", ""),
                    "both_positive_return": row.get("both_positive_return", ""),
                    "both_quality": row.get("both_quality", ""),
                    "both_frequency_aware": row.get("both_frequency_aware", ""),
                    "validation_probe_score": row.get("validation_probe_score", ""),
                    "passes_quality": row.get("passes_quality", ""),
                    "passes_frequency_aware": row.get("passes_frequency_aware", ""),
                    "net": metrics.get("broker_net_return_pct", ""),
                    "delta_net": delta.get("broker_net_return_pct", ""),
                    "dd": metrics.get("broker_max_drawdown_pct", ""),
                    "delta_dd": delta.get("broker_max_drawdown_pct", ""),
                    "trades": metrics.get("trade_count", ""),
                    "delta_trades": delta.get("trade_count", ""),
                    "avg_trade": metrics.get("avg_trade_net_pct", ""),
                    "delta_avg_trade": delta.get("avg_trade_net_pct", ""),
                    "capture": metrics.get("avg_mfe_capture", ""),
                    "delta_capture": delta.get("avg_mfe_capture", ""),
                    "score": metrics.get("immutable_score", ""),
                    "delta_score": delta.get("immutable_score", ""),
                    "mutation_hash": row.get("mutation_hash", ""),
                    "mutations_json": json.dumps(row.get("mutations") or {}, sort_keys=True, default=str),
                    "invalid": row.get("invalid", ""),
                    "error": row.get("error", ""),
                }
            )


def write_progress(started: float, status: str, completed: int, total: int) -> None:
    PROGRESS.write_text(
        json.dumps(
            {
                "elapsed_seconds": time.time() - started,
                "status": status,
                "completed": completed,
                "total": total,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [row for row in rows if row.get("row_type") == "pair"]
    completed = [row for row in pairs if not row.get("invalid")]
    positive = [row for row in completed if row.get("both_positive_return")]
    quality = [row for row in completed if row.get("both_quality")]
    freq = [row for row in completed if row.get("both_frequency_aware")]
    ranked = sorted(completed, key=lambda row: float(row.get("validation_probe_score", -999.0) or -999.0), reverse=True)
    train_only = [
        row
        for row in completed
        if float((row.get("train_delta") or {}).get("broker_net_return_pct", 0.0) or 0.0) > 0.0
        and float((row.get("validation_delta") or {}).get("broker_net_return_pct", 0.0) or 0.0) <= 0.0
    ]
    validation_only = [
        row
        for row in completed
        if float((row.get("validation_delta") or {}).get("broker_net_return_pct", 0.0) or 0.0) > 0.0
        and float((row.get("train_delta") or {}).get("broker_net_return_pct", 0.0) or 0.0) <= 0.0
    ]
    return {
        "completed_pair_count": len(completed),
        "invalid_pair_count": len(pairs) - len(completed),
        "both_positive_return_count": len(positive),
        "both_quality_count": len(quality),
        "both_frequency_aware_count": len(freq),
        "train_only_positive_count": len(train_only),
        "validation_only_positive_count": len(validation_only),
        "top_both_positive": [brief(row) for row in sorted(positive, key=lambda item: float(item.get("validation_probe_score", 0.0) or 0.0), reverse=True)[:10]],
        "top_both_quality": [brief(row) for row in sorted(quality, key=lambda item: float(item.get("validation_probe_score", 0.0) or 0.0), reverse=True)[:10]],
        "top_ranked": [brief(row) for row in ranked[:15]],
    }


def brief(row: dict[str, Any]) -> dict[str, Any]:
    train_delta = dict(row.get("train_delta") or {})
    validation_delta = dict(row.get("validation_delta") or {})
    train_metrics = dict(row.get("train_metrics") or {})
    validation_metrics = dict(row.get("validation_metrics") or {})
    return {
        "phase": row.get("phase"),
        "candidate": row.get("candidate"),
        "score": row.get("validation_probe_score"),
        "train_net_delta": train_delta.get("broker_net_return_pct"),
        "validation_net_delta": validation_delta.get("broker_net_return_pct"),
        "train_dd_delta": train_delta.get("broker_max_drawdown_pct"),
        "validation_dd_delta": validation_delta.get("broker_max_drawdown_pct"),
        "train_trades": train_metrics.get("trade_count"),
        "validation_trades": validation_metrics.get("trade_count"),
        "train_avg_trade_delta": train_delta.get("avg_trade_net_pct"),
        "validation_avg_trade_delta": validation_delta.get("avg_trade_net_pct"),
        "both_positive_return": row.get("both_positive_return"),
        "both_quality": row.get("both_quality"),
        "both_frequency_aware": row.get("both_frequency_aware"),
    }


if __name__ == "__main__":
    main()
