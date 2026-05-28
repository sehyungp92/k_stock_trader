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
ROUND_DIR = ROOT / "data/backtests/output/kalcb/round_5"
BACKUP_DIR = ROUND_DIR / "pre_holdout_corrected_final_backup_20260523_231835"
OUT_DIR = ROUND_DIR / "all_phase_candidate_audit"
OUT_JSON = OUT_DIR / "round5_all_phase_candidate_baseline_train_holdout_audit.json"
OUT_CSV = OUT_DIR / "round5_all_phase_candidate_baseline_train_holdout_audit.csv"
PROGRESS = OUT_DIR / "baseline_progress.json"

METRIC_KEYS = (
    "broker_net_return_pct",
    "official_mtm_net_return_pct",
    "broker_max_drawdown_pct",
    "trade_count",
    "avg_trade_net_pct",
    "net_win_share",
    "avg_mfe_capture",
    "mae_le_neg_1_share",
    "avg_mfe_r",
    "avg_mae_r",
    "worst_fold_net",
    "same_bar_fill_count",
    "end_open_position_count",
    "immutable_score",
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = yaml.safe_load((ROOT / "config/optimization/kalcb.yaml").read_text(encoding="utf-8")) or {}
    config["workers"] = 2
    state = json.loads((BACKUP_DIR / "phase_state.json").read_text(encoding="utf-8"))
    round5_baseline = dict(state["phase_results"]["1"]["base_mutations"])
    r4_optimized = json.loads((ROOT / "data/backtests/output/kalcb/round_4/optimized_config.json").read_text(encoding="utf-8"))["mutations"]
    fixed_source = {
        "path": round5_baseline[SOURCE_PATH_MUTATION],
        "section": round5_baseline[SOURCE_SECTION_MUTATION],
        "rank": round5_baseline[SOURCE_RANK_MUTATION],
    }

    train_config = deepcopy(config)
    train_config["fixed_candidate_source"] = fixed_source
    holdout_config = deepcopy(config)
    baseline_window = dict(holdout_config.get("baseline") or {})
    holdout_config["start"] = str(baseline_window["holdout_start"])
    holdout_config["end"] = str(baseline_window["holdout_end"])
    holdout_config["use_full_available_window"] = True
    holdout_config["fixed_candidate_source"] = fixed_source

    train_plugin = KALCBFixedTradePlanOptimizationPlugin(train_config, output_dir=ROUND_DIR, max_workers=2)
    write_progress(started, "train_plugin_ready", 0, 0)
    holdout_plugin = KALCBFixedTradePlanOptimizationPlugin(holdout_config, output_dir=ROUND_DIR / "holdout_mutation_attribution", max_workers=2)
    write_progress(started, "holdout_plugin_ready", 0, 0)

    baseline_train = compact(train_plugin.evaluate_mutations(round5_baseline))
    baseline_holdout = compact(holdout_plugin.evaluate_mutations(round5_baseline))
    candidates = ordered_candidates(collect_candidates(state))
    total = len(candidates)
    rows: list[dict[str, Any]] = []
    write_payload(started, rows, candidates, baseline_train, baseline_holdout, round5_baseline, r4_optimized)

    for index, item in enumerate(candidates, start=1):
        phase = item["phase"]
        candidate: Experiment = item["candidate"]
        write_progress(started, f"evaluating_phase_{phase}_{candidate.name}", index - 1, total)
        mutations = dict(round5_baseline)
        mutations.update(candidate.mutations)
        for window, plugin, compare in (
            ("train", train_plugin, baseline_train),
            ("holdout", holdout_plugin, baseline_holdout),
        ):
            row = evaluate_one(phase, candidate, window, plugin, mutations, compare)
            rows.append(row)
        write_payload(started, rows, candidates, baseline_train, baseline_holdout, round5_baseline, r4_optimized)
        write_progress(started, f"completed_phase_{phase}_{candidate.name}", index, total)

    print(json.dumps(summary_from_rows(rows), indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")


def collect_candidates(state: dict[str, Any]) -> dict[int, list[Experiment]]:
    out: dict[int, list[Experiment]] = {}
    for phase in range(1, 12):
        items = list(get_phase_candidates(phase))
        result = (state.get("phase_results") or {}).get(str(phase)) or {}
        for item in result.get("suggested_experiments") or []:
            name = item.get("name")
            mutations = item.get("mutations")
            if name and isinstance(mutations, dict):
                items.append(Experiment(str(name), dict(mutations)))
        seen: set[str] = set()
        deduped: list[Experiment] = []
        for item in items:
            if item.name in seen:
                continue
            seen.add(item.name)
            deduped.append(item)
        out[phase] = deduped
    return out


def ordered_candidates(candidates_by_phase: dict[int, list[Experiment]]) -> list[dict[str, Any]]:
    cheap: list[dict[str, Any]] = []
    source: list[dict[str, Any]] = []
    for phase, candidates in candidates_by_phase.items():
        for candidate in candidates:
            row = {"phase": phase, "candidate": candidate}
            if phase == 1 and any(key in candidate.mutations for key in (SOURCE_SECTION_MUTATION, SOURCE_RANK_MUTATION, SOURCE_PATH_MUTATION)):
                source.append(row)
            else:
                cheap.append(row)
    return cheap + source


def evaluate_one(
    phase: int,
    candidate: Experiment,
    window: str,
    plugin: KALCBFixedTradePlanOptimizationPlugin,
    mutations: dict[str, Any],
    compare: dict[str, Any],
) -> dict[str, Any]:
    try:
        metrics = compact(plugin.evaluate_mutations(mutations))
        delta = metric_delta(metrics, compare)
        return {
            "phase": phase,
            "candidate": candidate.name,
            "window": window,
            "mutation_hash": stable_signature(candidate.mutations),
            "mutations": dict(candidate.mutations),
            "metrics": metrics,
            "delta": delta,
            "beneficial_quality": is_beneficial_quality(delta),
            "beneficial_frequency_aware": is_beneficial_frequency_aware(delta, compare),
        }
    except Exception as exc:
        return {
            "phase": phase,
            "candidate": candidate.name,
            "window": window,
            "mutation_hash": stable_signature(candidate.mutations),
            "mutations": dict(candidate.mutations),
            "invalid": True,
            "error": f"{type(exc).__name__}: {exc}",
            "beneficial_quality": False,
            "beneficial_frequency_aware": False,
        }


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}
    out["immutable_score"] = metrics.get("immutable_score", score_fixed(metrics))
    return out


def metric_delta(metrics: dict[str, Any], compare: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and isinstance(compare.get(key), (int, float)):
            out[key] = float(value) - float(compare[key])
    return out


def is_beneficial_quality(delta: dict[str, Any]) -> bool:
    return (
        float(delta.get("broker_net_return_pct", 0.0) or 0.0) > 0.0
        and float(delta.get("avg_trade_net_pct", 0.0) or 0.0) >= -0.0005
        and float(delta.get("broker_max_drawdown_pct", 0.0) or 0.0) <= 0.005
        and float(delta.get("same_bar_fill_count", 0.0) or 0.0) <= 0.0
        and float(delta.get("end_open_position_count", 0.0) or 0.0) <= 0.0
    )


def is_beneficial_frequency_aware(delta: dict[str, Any], compare: dict[str, Any]) -> bool:
    base_trades = max(float(compare.get("trade_count", 0.0) or 0.0), 1.0)
    return is_beneficial_quality(delta) and float(delta.get("trade_count", 0.0) or 0.0) >= -0.10 * base_trades


def write_payload(
    started: float,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    baseline_train: dict[str, Any],
    baseline_holdout: dict[str, Any],
    round5_baseline: dict[str, Any],
    r4_optimized: dict[str, Any],
) -> None:
    payload = {
        "elapsed_seconds": time.time() - started,
        "strategy": "kalcb",
        "round": 5,
        "scope": "all current round_5 phase candidate definitions evaluated standalone against the clean round_5 baseline",
        "total_candidate_definitions": len(candidates),
        "completed_candidate_windows": len(rows),
        "completed_candidate_definitions": len({(row.get("phase"), row.get("candidate"), row.get("mutation_hash")) for row in rows}),
        "baseline_identity": {
            "round5_phase1_base_equals_round4_optimized": round5_baseline == r4_optimized,
            "round4_mutation_count": len(r4_optimized),
            "round5_phase1_base_mutation_count": len(round5_baseline),
        },
        "baseline_train": baseline_train,
        "baseline_holdout": baseline_holdout,
        "rows": rows,
        "summary": summary_from_rows(rows),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_csv(rows)


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "window",
        "candidate",
        "beneficial_quality",
        "beneficial_frequency_aware",
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
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") or {}
            delta = row.get("delta") or {}
            writer.writerow(
                {
                    "phase": row.get("phase"),
                    "window": row.get("window"),
                    "candidate": row.get("candidate"),
                    "beneficial_quality": row.get("beneficial_quality"),
                    "beneficial_frequency_aware": row.get("beneficial_frequency_aware"),
                    "net": metrics.get("broker_net_return_pct"),
                    "delta_net": delta.get("broker_net_return_pct"),
                    "dd": metrics.get("broker_max_drawdown_pct"),
                    "delta_dd": delta.get("broker_max_drawdown_pct"),
                    "trades": metrics.get("trade_count"),
                    "delta_trades": delta.get("trade_count"),
                    "avg_trade": metrics.get("avg_trade_net_pct"),
                    "delta_avg_trade": delta.get("avg_trade_net_pct"),
                    "capture": metrics.get("avg_mfe_capture"),
                    "delta_capture": delta.get("avg_mfe_capture"),
                    "score": metrics.get("immutable_score"),
                    "delta_score": delta.get("immutable_score"),
                    "mutation_hash": row.get("mutation_hash"),
                    "mutations_json": json.dumps(row.get("mutations") or {}, sort_keys=True, default=str),
                    "invalid": row.get("invalid", False),
                    "error": row.get("error", ""),
                }
            )


def summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row.get("phase", 0) or 0), str(row.get("candidate")), str(row.get("mutation_hash")))
        grouped.setdefault(key, {"phase": key[0], "candidate": key[1], "mutation_hash": key[2]})[str(row.get("window"))] = row
    paired = []
    for item in grouped.values():
        train = item.get("train") or {}
        holdout = item.get("holdout") or {}
        if not train or not holdout:
            continue
        paired.append(
            {
                "phase": item["phase"],
                "candidate": item["candidate"],
                "mutation_hash": item["mutation_hash"],
                "positive_quality_both": bool(train.get("beneficial_quality")) and bool(holdout.get("beneficial_quality")),
                "positive_frequency_aware_both": bool(train.get("beneficial_frequency_aware")) and bool(holdout.get("beneficial_frequency_aware")),
                "train_delta_net": (train.get("delta") or {}).get("broker_net_return_pct"),
                "holdout_delta_net": (holdout.get("delta") or {}).get("broker_net_return_pct"),
                "train_delta_dd": (train.get("delta") or {}).get("broker_max_drawdown_pct"),
                "holdout_delta_dd": (holdout.get("delta") or {}).get("broker_max_drawdown_pct"),
                "train_delta_trades": (train.get("delta") or {}).get("trade_count"),
                "holdout_delta_trades": (holdout.get("delta") or {}).get("trade_count"),
                "train_delta_avg_trade": (train.get("delta") or {}).get("avg_trade_net_pct"),
                "holdout_delta_avg_trade": (holdout.get("delta") or {}).get("avg_trade_net_pct"),
                "train_delta_capture": (train.get("delta") or {}).get("avg_mfe_capture"),
                "holdout_delta_capture": (holdout.get("delta") or {}).get("avg_mfe_capture"),
            }
        )
    return {
        "row_count": len(rows),
        "paired_candidate_count": len(paired),
        "positive_quality_both_count": sum(1 for row in paired if row["positive_quality_both"]),
        "positive_frequency_aware_both_count": sum(1 for row in paired if row["positive_frequency_aware_both"]),
        "positive_quality_both": sorted([row for row in paired if row["positive_quality_both"]], key=lambda row: (row["phase"], row["candidate"])),
        "positive_frequency_aware_both": sorted([row for row in paired if row["positive_frequency_aware_both"]], key=lambda row: (row["phase"], row["candidate"])),
        "top_holdout_net_positive": sorted(paired, key=lambda row: float(row.get("holdout_delta_net") or -999.0), reverse=True)[:20],
        "top_train_net_positive": sorted(paired, key=lambda row: float(row.get("train_delta_net") or -999.0), reverse=True)[:20],
    }


def write_progress(started: float, stage: str, done: int, total: int) -> None:
    PROGRESS.write_text(
        json.dumps(
            {
                "stage": stage,
                "done_candidate_definitions": done,
                "total_candidate_definitions": total,
                "elapsed_seconds": time.time() - started,
                "updated_at_epoch": time.time(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
