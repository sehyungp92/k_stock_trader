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
OUT_JSON = OUT_DIR / "round5_all_phase_candidate_train_holdout_audit.json"
OUT_CSV = OUT_DIR / "round5_all_phase_candidate_train_holdout_audit.csv"
PROGRESS = OUT_DIR / "progress.json"


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
    r4_optimized = json.loads((ROOT / "data/backtests/output/kalcb/round_4/optimized_config.json").read_text(encoding="utf-8"))["mutations"]
    round5_baseline = dict(state["phase_results"]["1"]["base_mutations"])

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

    # Reuse the replay/cache locations already built during round_5 and its
    # holdout-attribution pass. Fresh output dirs force expensive context rebuilds.
    train_plugin = KALCBFixedTradePlanOptimizationPlugin(train_config, output_dir=ROUND_DIR, max_workers=2)
    holdout_plugin = KALCBFixedTradePlanOptimizationPlugin(holdout_config, output_dir=ROUND_DIR / "holdout_mutation_attribution", max_workers=2)

    candidates_by_phase = collect_candidates(state)
    total_candidates = sum(len(items) for items in candidates_by_phase.values())
    rows: list[dict[str, Any]] = []

    write_progress(started, "plugins_ready", 0, total_candidates)
    baseline_train = compact(train_plugin.evaluate_mutations(round5_baseline))
    write_progress(started, "baseline_train_done", 0, total_candidates)
    baseline_holdout = compact(holdout_plugin.evaluate_mutations(round5_baseline))
    write_progress(started, "baseline_holdout_done", 0, total_candidates)
    phase_bases: dict[int, dict[str, Any]] = {}
    phase_base_train: dict[int, dict[str, Any]] = {}
    phase_base_holdout: dict[int, dict[str, Any]] = {}
    for phase in range(1, 12):
        phase_bases[phase] = dict((state.get("phase_results", {}).get(str(phase)) or {}).get("base_mutations") or round5_baseline)
        phase_base_train[phase] = compact(train_plugin.evaluate_mutations(phase_bases[phase]))
        phase_base_holdout[phase] = compact(holdout_plugin.evaluate_mutations(phase_bases[phase]))
        write_progress(started, f"phase_{phase}_base_metrics_done", 0, total_candidates)

    write_progress(started, "baselines_done", 0, total_candidates)
    for phase in range(1, 12):
        candidates = candidates_by_phase.get(phase, [])
        if not candidates:
            continue
        for context_name, base_mutations, compare_train, compare_holdout in (
            ("round5_baseline_standalone", round5_baseline, baseline_train, baseline_holdout),
            ("actual_phase_context", phase_bases[phase], phase_base_train[phase], phase_base_holdout[phase]),
        ):
            rows.extend(
                evaluate_context(
                    phase=phase,
                    context_name=context_name,
                    candidates=candidates,
                    base_mutations=base_mutations,
                    train_plugin=train_plugin,
                    holdout_plugin=holdout_plugin,
                    compare_train=compare_train,
                    compare_holdout=compare_holdout,
                )
            )
            write_partial(
                started=started,
                state=state,
                r4_optimized=r4_optimized,
                round5_baseline=round5_baseline,
                candidates_by_phase=candidates_by_phase,
                baseline_train=baseline_train,
                baseline_holdout=baseline_holdout,
                phase_base_train=phase_base_train,
                phase_base_holdout=phase_base_holdout,
                rows=rows,
            )
            write_progress(started, f"phase_{phase}_{context_name}_done", sum(len(items) for p, items in candidates_by_phase.items() if p <= phase), total_candidates)

    write_partial(
        started=started,
        state=state,
        r4_optimized=r4_optimized,
        round5_baseline=round5_baseline,
        candidates_by_phase=candidates_by_phase,
        baseline_train=baseline_train,
        baseline_holdout=baseline_holdout,
        phase_base_train=phase_base_train,
        phase_base_holdout=phase_base_holdout,
        rows=rows,
    )
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
            key = item.name
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        out[phase] = deduped
    return out


def evaluate_context(
    *,
    phase: int,
    context_name: str,
    candidates: list[Experiment],
    base_mutations: dict[str, Any],
    train_plugin: KALCBFixedTradePlanOptimizationPlugin,
    holdout_plugin: KALCBFixedTradePlanOptimizationPlugin,
    compare_train: dict[str, Any],
    compare_holdout: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plugin_name, plugin, compare in (
        ("train", train_plugin, compare_train),
        ("holdout", holdout_plugin, compare_holdout),
    ):
        evaluator = plugin.create_evaluate_batch(phase, base_mutations)
        scored = evaluator(candidates, base_mutations)
        by_name = {item.name: item for item in scored}
        close = getattr(evaluator, "close", None)
        if callable(close):
            close()
        for candidate in candidates:
            item = by_name.get(candidate.name)
            if item is None:
                rows.append(
                    {
                        "phase": phase,
                        "context": context_name,
                        "window": plugin_name,
                        "candidate": candidate.name,
                        "mutation_hash": stable_signature(candidate.mutations),
                        "mutations": dict(candidate.mutations),
                        "invalid": True,
                        "error": "candidate_not_returned_by_evaluator",
                    }
                )
                continue
            metrics = compact(item.metrics)
            deltas = metric_delta(metrics, compare)
            rows.append(
                {
                    "phase": phase,
                    "context": context_name,
                    "window": plugin_name,
                    "candidate": candidate.name,
                    "mutation_hash": stable_signature(candidate.mutations),
                    "mutations": dict(candidate.mutations),
                    "rejected": bool(item.rejected),
                    "reject_reason": item.reject_reason,
                    "score": item.score,
                    "metrics": metrics,
                    "compare": compare,
                    "delta": deltas,
                    "beneficial_quality": is_beneficial_quality(deltas),
                    "beneficial_frequency_aware": is_beneficial_frequency_aware(deltas, compare),
                }
            )
    return rows


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


def write_partial(
    *,
    started: float,
    state: dict[str, Any],
    r4_optimized: dict[str, Any],
    round5_baseline: dict[str, Any],
    candidates_by_phase: dict[int, list[Experiment]],
    baseline_train: dict[str, Any],
    baseline_holdout: dict[str, Any],
    phase_base_train: dict[int, dict[str, Any]],
    phase_base_holdout: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "generated_at_epoch": time.time(),
        "elapsed_seconds": time.time() - started,
        "strategy": "kalcb",
        "round": 5,
        "candidate_count_by_phase": {str(phase): len(items) for phase, items in candidates_by_phase.items()},
        "total_candidate_definitions": sum(len(items) for items in candidates_by_phase.values()),
        "baseline_identity": {
            "round5_phase1_base_equals_round4_optimized": round5_baseline == r4_optimized,
            "round4_mutation_count": len(r4_optimized),
            "round5_phase1_base_mutation_count": len(round5_baseline),
        },
        "baseline_train": baseline_train,
        "baseline_holdout": baseline_holdout,
        "phase_base_train": {str(k): v for k, v in phase_base_train.items()},
        "phase_base_holdout": {str(k): v for k, v in phase_base_holdout.items()},
        "rows": rows,
        "summary": summary_from_rows(rows),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_csv(rows)


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "context",
        "window",
        "candidate",
        "rejected",
        "reject_reason",
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
                    "context": row.get("context"),
                    "window": row.get("window"),
                    "candidate": row.get("candidate"),
                    "rejected": row.get("rejected"),
                    "reject_reason": row.get("reject_reason"),
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
                    "score": metrics.get("immutable_score", row.get("score")),
                    "delta_score": delta.get("immutable_score"),
                    "mutation_hash": row.get("mutation_hash"),
                    "mutations_json": json.dumps(row.get("mutations") or {}, sort_keys=True, default=str),
                }
            )


def summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        by_key[(int(row.get("phase", 0) or 0), str(row.get("context")), str(row.get("candidate")) + "::" + str(row.get("mutation_hash")) + "::" + str(row.get("window")))] = row

    paired: list[dict[str, Any]] = []
    group: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row.get("phase", 0) or 0), str(row.get("context")), str(row.get("candidate")), str(row.get("mutation_hash")))
        item = group.setdefault(key, {"phase": key[0], "context": key[1], "candidate": key[2], "mutation_hash": key[3]})
        item[str(row.get("window"))] = row
    for item in group.values():
        train = item.get("train") or {}
        holdout = item.get("holdout") or {}
        if not train or not holdout:
            continue
        paired.append(
            {
                "phase": item["phase"],
                "context": item["context"],
                "candidate": item["candidate"],
                "mutation_hash": item["mutation_hash"],
                "train_quality": bool(train.get("beneficial_quality")),
                "holdout_quality": bool(holdout.get("beneficial_quality")),
                "train_frequency_aware": bool(train.get("beneficial_frequency_aware")),
                "holdout_frequency_aware": bool(holdout.get("beneficial_frequency_aware")),
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
                "train_rejected": bool(train.get("rejected")),
                "holdout_rejected": bool(holdout.get("rejected")),
            }
        )
    positive_quality = [row for row in paired if row["positive_quality_both"]]
    positive_frequency = [row for row in paired if row["positive_frequency_aware_both"]]
    return {
        "row_count": len(rows),
        "paired_candidate_context_count": len(paired),
        "positive_quality_both_count": len(positive_quality),
        "positive_frequency_aware_both_count": len(positive_frequency),
        "positive_quality_both": sorted(positive_quality, key=lambda row: (row["context"], row["phase"], row["candidate"])),
        "positive_frequency_aware_both": sorted(positive_frequency, key=lambda row: (row["context"], row["phase"], row["candidate"])),
        "top_holdout_net_positive": sorted(
            paired,
            key=lambda row: float(row.get("holdout_delta_net") or -999.0),
            reverse=True,
        )[:20],
        "top_train_net_positive": sorted(
            paired,
            key=lambda row: float(row.get("train_delta_net") or -999.0),
            reverse=True,
        )[:20],
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
