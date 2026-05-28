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

from backtests.auto.shared.types import Experiment
from backtests.strategies.kalcb.fixed_trade_plan_phase import (
    KALCBFixedTradePlanOptimizationPlugin,
    SOURCE_PATH_MUTATION,
    SOURCE_RANK_MUTATION,
    SOURCE_SECTION_MUTATION,
    get_phase_candidates,
)
from tmp.kalcb_round4_redesigned_phase_probe import combine_pair, compact, evaluate_one, summary_from_rows


ROOT = Path(".")
ROUND_DIR = ROOT / "data/backtests/output/kalcb/round_4"
OUT_DIR = ROUND_DIR / "capped_route_phase9_probe"
OUT_JSON = OUT_DIR / "round4_capped_route_phase9_train_validation_probe.json"
OUT_CSV = OUT_DIR / "round4_capped_route_phase9_train_validation_probe.csv"
PROGRESS = OUT_DIR / "progress.json"


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
        - float(((optimized.get("oos_validation") or {}).get("broker_net_return_pct") or 0.0)),
        "validation_trade_count_delta_vs_round4_artifact": float(baseline_validation.get("trade_count", 0.0) or 0.0)
        - float(((optimized.get("oos_validation") or {}).get("trade_count") or 0.0)),
    }

    candidates = list(get_phase_candidates(9))
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        write_progress(started, f"evaluating_{candidate.name}", index - 1, len(candidates))
        mutations = dict(baseline_mutations)
        mutations.update(candidate.mutations or {})
        train_row = evaluate_one("train", 9, candidate, train_plugin, mutations, baseline_train)
        validation_row = evaluate_one("validation", 9, candidate, validation_plugin, mutations, baseline_validation)
        rows.extend([train_row, validation_row, combine_pair(train_row, validation_row, baseline_train, baseline_validation)])
        write_payload(started, baseline_train, baseline_validation, baseline_drift, rows, candidates)
    write_progress(started, "complete", len(candidates), len(candidates))
    print(json.dumps(summary_from_rows(rows), indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")


def write_payload(
    started: float,
    baseline_train: dict[str, Any],
    baseline_validation: dict[str, Any],
    baseline_drift: dict[str, float],
    rows: list[dict[str, Any]],
    candidates: list[Experiment],
) -> None:
    payload = {
        "elapsed_seconds": time.time() - started,
        "strategy": "kalcb",
        "source_round": 4,
        "scope": "phase 9 capped secondary-route rerun against unchanged round_4 baseline",
        "max_workers": 2,
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
        "net",
        "delta_net",
        "dd",
        "delta_dd",
        "trades",
        "delta_trades",
        "avg_trade",
        "delta_avg_trade",
        "invalid",
        "error",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") or row.get("train_metrics") or {}
            delta = row.get("delta") or row.get("train_delta") or {}
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
                    "net": metrics.get("broker_net_return_pct", ""),
                    "delta_net": delta.get("broker_net_return_pct", ""),
                    "dd": metrics.get("broker_max_drawdown_pct", ""),
                    "delta_dd": delta.get("broker_max_drawdown_pct", ""),
                    "trades": metrics.get("trade_count", ""),
                    "delta_trades": delta.get("trade_count", ""),
                    "avg_trade": metrics.get("avg_trade_net_pct", ""),
                    "delta_avg_trade": delta.get("avg_trade_net_pct", ""),
                    "invalid": row.get("invalid", ""),
                    "error": row.get("error", ""),
                }
            )


def write_progress(started: float, status: str, completed: int, total: int) -> None:
    PROGRESS.write_text(
        json.dumps({"elapsed_seconds": time.time() - started, "status": status, "completed": completed, "total": total}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
