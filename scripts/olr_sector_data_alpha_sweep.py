from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "olr_round2_oos_deep_ablation.py"
ROUND4_CONFIG = ROOT / "data" / "backtests" / "output" / "olr" / "round_4" / "optimized_config.json"
DEFAULT_OUTPUT = ROOT / "tmp" / "olr_sector_data_alpha_sweep"
LOW_TAIL_MAX = 300.0
HIGH_TAIL_MIN = 650.0
BASELINE_CURRENT = "current_round4_hard_sector_filter"
BASELINE_NO_FILTER = "no_sector_filter_allow_400_500"
BASELINE_REJECT300 = "reject300_no_mid_band"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether OLR sector-daily/intraday data can replace hard sector filters.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--holdout-days", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--train-top", type=int, default=110, help="Train-confirm this many OOS-ranked candidates, plus mandatory baselines/family leaders. Use 0 for all.")
    parser.add_argument("--family-leaders", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--stage2-timeout-seconds", type=float, default=0.0, help="Optional timeout for one streaming stage-2 source; 0 disables.")
    parser.add_argument("--stage1-cache-limit", type=int, default=4, help="Maximum unique stage-1 snapshot/context caches to hold in memory during streaming.")
    parser.add_argument("--focused", action="store_true", help="Run the first-pass curated sector-data hypothesis set.")
    parser.add_argument("--combo-focused", action="store_true", help="Run the derived daily/intraday sector interaction hypothesis set.")
    parser.add_argument("--labels", default="", help="Optional comma-separated exact candidate labels to evaluate, plus mandatory baselines.")
    parser.add_argument("--legacy-batch-compile", action="store_true", help="Use the older opaque batch compiler instead of the streaming source compiler.")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    helper = load_helper()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / "evaluations.jsonl"
    progress_path = output_dir / "progress.jsonl"
    if args.fresh:
        for path in (eval_path, progress_path):
            if path.exists():
                path.unlink()

    config = helper.normalize_runtime_config("olr", helper.load_yaml_config(str(ROOT / "config" / "optimization" / "olr.yaml")))
    config["capability_level"] = "real_replay"
    config["holdout_days"] = int(args.holdout_days)
    config["use_full_available_window"] = True

    round4 = read_json(ROUND4_CONFIG)
    base = copy.deepcopy(round4["mutations"])
    candidates = build_candidates(helper, base)
    if args.focused:
        candidates = focused_candidates(helper, candidates)
    if args.combo_focused:
        candidates = combo_focused_candidates(helper, candidates)
    labels = {safe_label(label.strip()) for label in str(args.labels or "").split(",") if label.strip()}
    if labels:
        mandatory = {BASELINE_CURRENT, BASELINE_NO_FILTER, BASELINE_REJECT300}
        candidates = helper.dedupe_candidates([candidate for candidate in candidates if candidate.label in labels or candidate.label in mandatory])
    if int(args.max_candidates) > 0:
        mandatory = {BASELINE_CURRENT, BASELINE_NO_FILTER, BASELINE_REJECT300}
        limited = []
        for candidate in candidates:
            if candidate.label in mandatory or len(limited) < int(args.max_candidates):
                limited.append(candidate)
        candidates = helper.dedupe_candidates(limited)

    write_json(
        output_dir / "candidate_plan.json",
        {
            "generated_at_utc": utc_now(),
            "candidate_count": len(candidates),
            "kind_counts": dict(Counter(candidate.kind for candidate in candidates)),
            "candidates": [
                {
                    "label": candidate.label,
                    "kind": candidate.kind,
                    "reason": candidate.reason,
                    "changed_from_round4": changed_from_base(base, candidate.mutations),
                    "score_band_rules": candidate.mutations.get("olr.afternoon.score_band_rules", []),
                }
                for candidate in candidates
            ],
        },
    )
    status(progress_path, "candidate_plan", total=len(candidates), counts=dict(Counter(candidate.kind for candidate in candidates)))

    cached = {
        key: row
        for key, row in helper.load_cached_evaluations(eval_path).items()
        if row.get("metrics") and not row.get("error")
    }
    evaluator = helper.evaluate_candidates if args.legacy_batch_compile else streaming_evaluate_candidates
    evaluator_extra = (
        {}
        if args.legacy_batch_compile
        else {
            "stage2_timeout_seconds": float(args.stage2_timeout_seconds),
            "stage1_cache_limit": max(1, int(args.stage1_cache_limit)),
        }
    )
    oos_rows = evaluator(
        config,
        candidates,
        "oos",
        output_dir,
        eval_path,
        progress_path,
        cached,
        holdout_days=int(args.holdout_days),
        batch_size=max(1, int(args.batch_size)),
        **evaluator_extra,
    )
    oos_by_label = {row["label"]: row for row in oos_rows}
    train_labels = select_train_labels(
        helper,
        candidates,
        oos_rows,
        train_top=int(args.train_top),
        family_leaders=max(0, int(args.family_leaders)),
    )
    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    train_candidates = [candidate_by_label[label] for label in train_labels if label in candidate_by_label]
    status(progress_path, "train_confirm_plan", total=len(train_candidates), labels=train_labels)
    train_rows = evaluator(
        config,
        train_candidates,
        "train",
        output_dir,
        eval_path,
        progress_path,
        cached,
        holdout_days=int(args.holdout_days),
        batch_size=max(1, int(args.batch_size)),
        **evaluator_extra,
    )
    train_by_label = {row["label"]: row for row in train_rows}

    payload = summarize(helper, candidates, oos_by_label, train_by_label, started, holdout_days=int(args.holdout_days))
    write_json(output_dir / "sector_data_alpha_sweep.json", payload)
    write_text(output_dir / "sector_data_alpha_sweep.md", render_markdown(helper, payload))
    if payload.get("best_data_alpha"):
        write_json(output_dir / "best_data_alpha_mutations.json", payload["best_data_alpha"].get("mutations", {}))
    status(
        progress_path,
        "complete",
        elapsed_seconds=payload["elapsed_seconds"],
        result_path=str(output_dir / "sector_data_alpha_sweep.json"),
        summary_path=str(output_dir / "sector_data_alpha_sweep.md"),
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True), flush=True)
    return 0


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("olr_round2_oos_deep_ablation", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_candidates(helper: Any, base: dict[str, Any]) -> list[Any]:
    out: list[Any] = []

    def add(label: str, kind: str, mutations: dict[str, Any], reason: str) -> None:
        out.append(helper.Candidate(safe_label(label), kind, copy.deepcopy(mutations), reason))

    current = copy.deepcopy(base)
    add(BASELINE_CURRENT, "baseline", current, "Current round-4 conditional score-band rule with a hard sector allow-list.")

    no_filter = with_rules(base, [low_rule(), mid_rule(400.0, 500.0), high_rule()])
    add(BASELINE_NO_FILTER, "baseline", no_filter, "Round-4 400-500 middle band with the hard sector allow-list removed.")

    reject300 = with_rules(base, [low_rule(), high_rule()])
    add(BASELINE_REJECT300, "baseline", reject300, "Round-3-style reject 300-650 notch expressed through score-band rules.")

    for floor in (45.0, 50.0, 55.0, 60.0):
        add(
            f"hard_filter_daily_floor_{num(floor)}",
            "hard_filter_control",
            with_mutations(current, {"olr.research.min_sector_daily_score_pct": floor}),
            "Control: keep the hard sector list and add a daily sector-score research floor.",
        )
    for weight in (0.001, 0.002):
        add(
            f"hard_filter_intraday_weight_{wlabel(weight)}",
            "hard_filter_control",
            with_mutations(current, {"olr.afternoon.weight_intraday_sector": weight}),
            "Control: keep the hard sector list and add a small 14:30 intraday-sector rerank weight.",
        )
    for floor in (0.0, 45.0, 50.0, 55.0, 60.0):
        for regime_weight in (0.07, 0.09, 0.11):
            extra: dict[str, Any] = {
                "olr.research.use_sector_daily_regime_score": True,
                "olr.research.weights.sector_regime": regime_weight,
            }
            if floor > 0.0:
                extra["olr.research.min_sector_daily_score_pct"] = floor
            add(
                f"hard_filter_dailyregw_{wlabel(regime_weight)}_daily{num(floor)}",
                "hard_filter_control",
                with_mutations(current, extra),
                "Control: keep the hard sector list and explicitly opt in to the daily sector-score research channel.",
            )

    daily_floors = (0.0, 45.0, 50.0, 55.0, 60.0)
    intraday_weights = (0.0, 0.001, 0.002)
    middle_bands = (
        (350.0, 500.0),
        (375.0, 500.0),
        (400.0, 500.0),
        (425.0, 500.0),
        (450.0, 500.0),
        (400.0, 475.0),
        (425.0, 475.0),
        (450.0, 475.0),
        (475.0, 500.0),
        (500.0, 650.0),
    )
    for lo, hi in middle_bands:
        for floor in daily_floors:
            for weight in intraday_weights:
                if floor == 0.0 and weight == 0.0 and (lo, hi) == (400.0, 500.0):
                    continue
                mutations = with_rules(base, [low_rule(), mid_rule(lo, hi), high_rule()])
                extra: dict[str, Any] = {}
                if floor > 0.0:
                    extra["olr.research.min_sector_daily_score_pct"] = floor
                if weight > 0.0:
                    extra["olr.afternoon.weight_intraday_sector"] = weight
                mutations = with_mutations(mutations, extra)
                add(
                    f"data_band_{num(lo)}_{num(hi)}_daily{num(floor)}_iw{wlabel(weight)}",
                    "sector_data_band",
                    mutations,
                    "No hard sector list: daily sector-score floor chooses the pool and 14:30 sector score softly reranks.",
                )

    for floor in (0.0, 45.0, 50.0, 55.0):
        for confirm in (45.0, 50.0, 55.0):
            for weight in (0.0, 0.001):
                mutations = with_rules(base, [low_rule(), mid_rule(400.0, 500.0, min_intraday_sector_score_pct=confirm), high_rule()])
                extra = {"olr.research.min_sector_daily_score_pct": floor} if floor > 0.0 else {}
                if weight > 0.0:
                    extra["olr.afternoon.weight_intraday_sector"] = weight
                mutations = with_mutations(mutations, extra)
                add(
                    f"data_midconfirm_{num(confirm)}_daily{num(floor)}_iw{wlabel(weight)}",
                    "sector_data_mid_confirm",
                    mutations,
                    "No hard sector list: only the recovered 400-500 score band needs 14:30 sector confirmation.",
                )

    for floor in (0.0, 45.0, 50.0, 55.0):
        for confirm in (45.0, 50.0, 55.0):
            mutations = with_mutations(no_filter, {"olr.afternoon.min_intraday_sector_score_pct": confirm})
            if floor > 0.0:
                mutations["olr.research.min_sector_daily_score_pct"] = floor
            add(
                f"data_global_intraday_floor_{num(confirm)}_daily{num(floor)}",
                "sector_data_global_confirm",
                mutations,
                "No hard sector list: mild global 14:30 sector confirmation floor, validated but treated cautiously.",
            )

    for regime_weight in (0.09, 0.11, 0.13):
        for floor in (0.0, 45.0, 50.0, 55.0):
            for weight in (0.0, 0.001):
                mutations = copy.deepcopy(no_filter)
                mutations["olr.research.use_sector_daily_regime_score"] = True
                mutations["olr.research.weights.sector_regime"] = regime_weight
                if floor > 0.0:
                    mutations["olr.research.min_sector_daily_score_pct"] = floor
                if weight > 0.0:
                    mutations["olr.afternoon.weight_intraday_sector"] = weight
                add(
                    f"data_sectorregw_{wlabel(regime_weight)}_daily{num(floor)}_iw{wlabel(weight)}",
                    "sector_data_research_weight",
                    mutations,
                    "No hard sector list: increase the research weight on sector_daily_score_pct while keeping total score normalized.",
                )

    for regime_weight, participation_weight in ((0.09, 0.05), (0.11, 0.05), (0.11, 0.03), (0.13, 0.03)):
        for floor in (45.0, 50.0, 55.0):
            mutations = copy.deepcopy(no_filter)
            mutations["olr.research.use_sector_daily_regime_score"] = True
            mutations["olr.research.use_sector_daily_participation"] = True
            mutations["olr.research.weights.sector_regime"] = regime_weight
            mutations["olr.research.weights.sector_participation"] = participation_weight
            mutations["olr.research.min_sector_daily_score_pct"] = floor
            mutations["olr.afternoon.weight_intraday_sector"] = 0.001
            add(
                f"data_sectorregw_{wlabel(regime_weight)}_partw_{wlabel(participation_weight)}_daily{num(floor)}_iw001",
                "sector_data_research_weight",
                mutations,
                "No hard sector list: shift research emphasis from participation toward sector_daily_score_pct.",
            )

    for mode in ("score", "campaign", "hybrid"):
        for floor in (0.0, 45.0, 50.0, 55.0):
            for weight in (0.0, 0.001):
                mutations = copy.deepcopy(no_filter)
                mutations["olr.research.use_sector_daily_regime_score"] = True
                mutations["olr.frontier.active_selection_mode"] = mode
                if floor > 0.0:
                    mutations["olr.research.min_sector_daily_score_pct"] = floor
                if weight > 0.0:
                    mutations["olr.afternoon.weight_intraday_sector"] = weight
                add(
                    f"data_frontier_{mode}_daily{num(floor)}_iw{wlabel(weight)}",
                    "sector_data_pool_mode",
                    mutations,
                    "No hard sector list: test whether research-score-led pool construction lets sector_daily_score_pct matter more.",
                )

    for top_long in (25, 30, 40):
        for floor in (45.0, 50.0, 55.0):
            mutations = copy.deepcopy(no_filter)
            mutations["olr.research.use_sector_daily_regime_score"] = True
            mutations["olr.research.top_long_count"] = top_long
            mutations["olr.research.min_sector_daily_score_pct"] = floor
            mutations["olr.afternoon.weight_intraday_sector"] = 0.001
            add(
                f"data_pool_top{top_long}_daily{num(floor)}_iw001",
                "sector_data_pool_size",
                mutations,
                "No hard sector list: widen the daily pool only when it passes a daily sector-score floor, then softly rerank at 14:30.",
            )

    add_combo_candidates(add, base, current)
    add_soft_combo_candidates(add, base, current, no_filter)

    return helper.dedupe_candidates(out)


def add_combo_candidates(add: Any, base: dict[str, Any], current: dict[str, Any]) -> None:
    """Derived sector rules: daily sector sets context, intraday/stock-relative action confirms."""

    hard_allowed = round4_mid_allowed_sectors(current)
    bands = ((400.0, 500.0), (425.0, 500.0), (400.0, 475.0))
    confirm_profiles = (
        (
            "confirm55_lead0",
            {
                "min_sector_confirm_mean_score_pct": 55.0,
                "min_sector_confirm_min_score_pct": 42.5,
                "min_stock_intraday_sector_ret_gap_pct": 0.0,
            },
            "Daily and 14:30 sector scores agree; stock is not lagging its sector intraday.",
        ),
        (
            "confirm60_lead1",
            {
                "min_sector_confirm_mean_score_pct": 60.0,
                "min_sector_confirm_min_score_pct": 45.0,
                "min_stock_intraday_sector_ret_gap_pct": 1.0,
            },
            "Stronger daily/intraday sector agreement plus stock intraday leadership.",
        ),
        (
            "daily50_intraday45_stocklead",
            {
                "min_sector_daily_score_pct": 50.0,
                "min_sector_intraday_score_pct": 45.0,
                "min_stock_sector_daily_ret20_gap_pct": 0.0,
                "min_stock_intraday_sector_ret_gap_pct": 0.5,
            },
            "Sector is acceptable on daily and intraday data while the stock leads peers.",
        ),
        (
            "daily45_intraday50_stocklead",
            {
                "min_sector_daily_score_pct": 45.0,
                "min_sector_intraday_score_pct": 50.0,
                "min_stock_sector_daily_ret20_gap_pct": -2.0,
                "min_stock_intraday_sector_ret_gap_pct": 1.0,
            },
            "Intraday sector confirmation can rescue only mildly weaker daily sector context.",
        ),
        (
            "rotation10_lead1",
            {
                "min_sector_daily_score_pct": 35.0,
                "min_sector_rotation_score": 10.0,
                "min_stock_intraday_leadership_score": 1.0,
            },
            "Emerging sector rotation: intraday sector improvement plus stock-relative leadership.",
        ),
        (
            "rotation20_lead3",
            {
                "min_sector_daily_score_pct": 35.0,
                "min_sector_rotation_score": 20.0,
                "min_stock_intraday_leadership_score": 3.0,
            },
            "Stronger emerging rotation profile; useful if hard sectors miss fresh leadership.",
        ),
        (
            "quality40_flow",
            {
                "min_sector_confirm_quality_score": 40.0,
                "min_sector_daily_flow_agreement_5d": -0.10,
                "min_stock_intraday_sector_ret_gap_pct": 0.0,
            },
            "Composite sector quality plus non-hostile daily sector flow agreement.",
        ),
        (
            "nofade_daily60_intraday40",
            {
                "min_sector_daily_score_pct": 60.0,
                "min_sector_intraday_score_pct": 40.0,
                "min_sector_intraday_daily_score_delta": -20.0,
                "min_stock_intraday_sector_ret_gap_pct": 0.0,
            },
            "Allow strong daily sector context only if 14:30 sector action has not badly faded.",
        ),
    )

    for lo, hi in bands:
        for name, conditions, reason in confirm_profiles:
            no_filter = with_rules(base, [low_rule(), mid_rule(lo, hi, **conditions), high_rule()])
            add(
                f"combo_nofilter_{name}_{num(lo)}_{num(hi)}",
                "sector_data_combo",
                no_filter,
                f"No hard sector list: {reason}",
            )
            if hard_allowed and (lo, hi) == (400.0, 500.0):
                hard_conditions = {**conditions, "allowed_sectors": hard_allowed}
                hard_filter = with_rules(base, [low_rule(), mid_rule(lo, hi, **hard_conditions), high_rule()])
                add(
                    f"combo_hardfilter_{name}_{num(lo)}_{num(hi)}",
                    "sector_data_combo_control",
                    hard_filter,
                    f"Keep round-4 hard sectors but require derived sector/stock confirmation: {reason}",
                )

    for floor in (35.0, 40.0, 45.0, 50.0):
        for rotation in (10.0, 15.0, 20.0):
            conditions = {
                "min_sector_daily_score_pct": floor,
                "max_sector_daily_score_pct": 60.0,
                "min_sector_rotation_score": rotation,
                "min_stock_intraday_sector_ret_gap_pct": 1.0,
            }
            mutations = with_rules(base, [low_rule(), mid_rule(400.0, 500.0, **conditions), high_rule()])
            add(
                f"combo_nofilter_weakdaily_rotation{num(rotation)}_floor{num(floor)}",
                "sector_data_combo",
                mutations,
                "No hard sector list: explicitly target weak-to-mid daily sectors that rotate positively by 14:30.",
            )


def round4_mid_allowed_sectors(mutations: dict[str, Any]) -> list[str]:
    rules = mutations.get("olr.afternoon.score_band_rules") or ()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if float(rule.get("min_score", -1.0) or -1.0) <= 400.0 and float(rule.get("max_score", 0.0) or 0.0) >= 500.0:
            values = rule.get("allowed_sectors")
            if isinstance(values, str):
                return [item for item in values.replace(";", ",").split(",") if item]
            if isinstance(values, (list, tuple)):
                return [str(item) for item in values if str(item)]
    return []


def add_soft_combo_candidates(add: Any, base: dict[str, Any], current: dict[str, Any], no_filter: dict[str, Any]) -> None:
    profiles = (
        ("q0005", {"olr.afternoon.weight_sector_confirm_quality": 0.0005}, "Small soft rerank toward daily/intraday sector confirmation quality."),
        ("q001", {"olr.afternoon.weight_sector_confirm_quality": 0.001}, "Soft rerank toward daily/intraday sector confirmation quality."),
        ("lead001", {"olr.afternoon.weight_stock_sector_leadership": 0.001}, "Soft rerank toward stocks leading their sector intraday."),
        ("lead002", {"olr.afternoon.weight_stock_sector_leadership": 0.002}, "Stronger stock-vs-sector intraday leadership rerank."),
        ("rot001", {"olr.afternoon.weight_sector_rotation": 0.001}, "Soft rerank toward sectors improving intraday versus daily context."),
        (
            "q001_lead001",
            {"olr.afternoon.weight_sector_confirm_quality": 0.001, "olr.afternoon.weight_stock_sector_leadership": 0.001},
            "Daily/intraday sector confirmation plus stock-relative leadership rerank.",
        ),
        (
            "q0005_lead001",
            {"olr.afternoon.weight_sector_confirm_quality": 0.0005, "olr.afternoon.weight_stock_sector_leadership": 0.001},
            "Milder sector confirmation plus stock-relative leadership rerank.",
        ),
        (
            "rot0005_lead001",
            {"olr.afternoon.weight_sector_rotation": 0.0005, "olr.afternoon.weight_stock_sector_leadership": 0.001},
            "Emerging sector rotation plus stock-relative leadership rerank.",
        ),
        (
            "intraday001_q0005_lead001",
            {
                "olr.afternoon.weight_intraday_sector": 0.001,
                "olr.afternoon.weight_sector_confirm_quality": 0.0005,
                "olr.afternoon.weight_stock_sector_leadership": 0.001,
            },
            "Existing tiny intraday-sector rerank plus derived confirmation and leadership.",
        ),
        (
            "intraday001_rot0005_lead001",
            {
                "olr.afternoon.weight_intraday_sector": 0.001,
                "olr.afternoon.weight_sector_rotation": 0.0005,
                "olr.afternoon.weight_stock_sector_leadership": 0.001,
            },
            "Existing tiny intraday-sector rerank plus rotation and stock leadership.",
        ),
    )
    for name, extra, reason in profiles:
        add(
            f"soft_hardfilter_{name}",
            "sector_data_soft_control",
            with_mutations(current, extra),
            f"Keep round-4 sector list: {reason}",
        )
        add(
            f"soft_nofilter_{name}",
            "sector_data_soft",
            with_mutations(no_filter, extra),
            f"No hard sector list: {reason}",
        )


def focused_candidates(helper: Any, candidates: Sequence[Any]) -> list[Any]:
    mandatory = {
        BASELINE_CURRENT,
        BASELINE_NO_FILTER,
        BASELINE_REJECT300,
        "hard_filter_daily_floor_45",
        "hard_filter_daily_floor_50",
        "hard_filter_daily_floor_55",
        "hard_filter_daily_floor_60",
        "hard_filter_intraday_weight_0p001",
        "hard_filter_intraday_weight_0p002",
    }
    labels = set(mandatory)
    for floor in (0, 45, 50, 55, 60):
        for regime_weight in ("0p07", "0p09", "0p11"):
            labels.add(f"hard_filter_dailyregw_{regime_weight}_daily{floor}")
        for weight in ("0", "0p001", "0p002"):
            labels.add(f"data_band_400_500_daily{floor}_iw{weight}")
    for lo, hi in ((425, 500), (450, 500), (400, 475), (425, 475), (475, 500)):
        for floor in (45, 50, 55):
            for weight in ("0", "0p001"):
                labels.add(f"data_band_{lo}_{hi}_daily{floor}_iw{weight}")
    for confirm in (45, 50, 55):
        for floor in (0, 45, 50, 55):
            for weight in ("0", "0p001"):
                labels.add(f"data_midconfirm_{confirm}_daily{floor}_iw{weight}")
    for confirm in (45, 50, 55):
        for floor in (0, 45, 50, 55):
            labels.add(f"data_global_intraday_floor_{confirm}_daily{floor}")
    for regime_weight in ("0p09", "0p11", "0p13"):
        for floor in (0, 45, 50, 55):
            for weight in ("0", "0p001"):
                labels.add(f"data_sectorregw_{regime_weight}_daily{floor}_iw{weight}")
    for mode in ("score", "campaign", "hybrid"):
        for floor in (0, 45, 50, 55):
            for weight in ("0", "0p001"):
                labels.add(f"data_frontier_{mode}_daily{floor}_iw{weight}")
    for top_long in (25, 30, 40):
        for floor in (45, 50, 55):
            labels.add(f"data_pool_top{top_long}_daily{floor}_iw001")
    selected = [candidate for candidate in candidates if candidate.label in labels]
    return helper.dedupe_candidates(selected)


def combo_focused_candidates(helper: Any, candidates: Sequence[Any]) -> list[Any]:
    mandatory = {
        BASELINE_CURRENT,
        BASELINE_NO_FILTER,
        BASELINE_REJECT300,
        "hard_filter_intraday_weight_0p001",
        "hard_filter_intraday_weight_0p002",
    }
    selected = [
        candidate
        for candidate in candidates
        if candidate.label in mandatory
        or str(candidate.kind).startswith("sector_data_combo")
        or str(candidate.kind).startswith("sector_data_soft")
    ]
    return helper.dedupe_candidates(selected)


def select_train_labels(helper: Any, candidates: Sequence[Any], oos_rows: Sequence[dict[str, Any]], *, train_top: int, family_leaders: int) -> list[str]:
    candidate_labels = {candidate.label for candidate in candidates}
    mandatory = {BASELINE_CURRENT, BASELINE_NO_FILTER, BASELINE_REJECT300}
    ranked = sorted(oos_rows, key=lambda row: oos_score(helper, row.get("metrics") or {}), reverse=True)
    labels: list[str] = []
    seen: set[str] = set()

    def add(label: str) -> None:
        if label in candidate_labels and label not in seen:
            labels.append(label)
            seen.add(label)

    for label in sorted(mandatory):
        add(label)
    if train_top <= 0:
        for candidate in candidates:
            add(candidate.label)
    else:
        for row in ranked[:train_top]:
            add(row["label"])
        for row in ranked:
            metrics = row.get("metrics") or {}
            if helper.metric_net(metrics) >= helper.metric_net((ranked[0].get("metrics") if ranked else {}) or {}) * 0.80 and helper.metric_trades(metrics) >= 25:
                add(row["label"])
        if family_leaders > 0:
            by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in ranked:
                by_family[family(row["label"])].append(row)
            for rows in by_family.values():
                for row in rows[:family_leaders]:
                    add(row["label"])
    return labels


def streaming_evaluate_candidates(
    config: dict[str, Any],
    candidates: Sequence[Any],
    window: str,
    output_dir: Path,
    eval_path: Path,
    progress_path: Path,
    cached: dict[tuple[str, str], dict[str, Any]],
    *,
    holdout_days: int,
    batch_size: int,
    stage2_timeout_seconds: float = 0.0,
    stage1_cache_limit: int = 4,
) -> list[dict[str, Any]]:
    helper = load_helper()
    requested = list(candidates)
    rows: list[dict[str, Any]] = []
    missing: list[Any] = []
    for candidate in requested:
        key = (window, candidate.label)
        if key in cached:
            rows.append(cached[key])
        else:
            missing.append(candidate)
    if not missing:
        return rows

    from backtests.auto.shared.cache_keys import stable_signature
    from backtests.strategies.olr import trade_plan_sweep as tps
    from backtests.strategies.olr.allocation_holdout_eval import _bundle_for_source
    from backtests.strategies.olr.runner import run_olr_backtest

    window_cfg = helper.window_config(config, window, holdout_days)
    include_holdout = window == "oos"
    status(
        progress_path,
        f"{window}_stream_prepare_dataset_start",
        candidates=len(missing),
        include_holdout=include_holdout,
    )
    prepare_started = time.monotonic()
    dataset = tps.prepare_research_sweep_dataset(
        window_cfg,
        holdout_days=holdout_days,
        expected_universe_size=tps.DEFAULT_EXPECTED_UNIVERSE_SIZE,
        include_holdout=include_holdout,
    )
    eligible_dates, next_by_date = tps._eligible_execution_dates(dataset)
    if window == "oos":
        dates = tuple(day for day in eligible_dates if day >= dataset.holdout_start)
    else:
        dates = tuple(day for day in eligible_dates if day < dataset.holdout_start)
    if not dates:
        raise RuntimeError(f"No {window} dates resolved for OLR streaming evaluation")
    status(
        progress_path,
        f"{window}_stream_prepare_dataset_done",
        elapsed_seconds=round(time.monotonic() - prepare_started, 3),
        dataset_dates=len(dataset.trading_dates),
        eval_dates=len(dates),
        date_start=dates[0].isoformat(),
        date_end=dates[-1].isoformat(),
    )

    research_started = time.monotonic()
    status(progress_path, f"{window}_stream_research_cache_start", dataset_dates=len(dataset.trading_dates))
    research_payload = {"base_mutations": {}, "selected_stage1_seed": {"mutations": {}}}
    base = {}
    research_cache = tps.research_snapshots_for_dataset(dataset, base)
    status(
        progress_path,
        f"{window}_stream_research_cache_done",
        elapsed_seconds=round(time.monotonic() - research_started, 3),
        snapshots=len(research_cache),
    )

    source_map, sources = helper.build_sources(missing)
    source_by_signature = {signature: source for signature, source in source_map.items()}
    needed_source_signatures = []
    for candidate in missing:
        signature = stable_signature(helper.selection_mutations(candidate.mutations))
        if signature not in needed_source_signatures:
            needed_source_signatures.append(signature)
    stage1_signature_by_label: dict[str, str] = {}
    for candidate in missing:
        signature = stable_signature(helper.selection_mutations(candidate.mutations))
        source = source_by_signature[signature]
        stage1_signature_by_label[candidate.label] = stable_signature(dict(base) | dict(source.stage1_mutations or {}))
    execution_candidates = sorted(
        missing,
        key=lambda candidate: (
            stage1_signature_by_label[candidate.label],
            str(candidate.kind),
            str(candidate.label),
        ),
    )
    stage1_cache: dict[str, tuple[dict[Any, Any], dict[Any, dict[str, Any]]]] = {}
    stage1_cache_order: list[str] = []
    bundle_cache: dict[str, Any] = {}

    def compile_source(source: Any) -> Any:
        cached_bundle = bundle_cache.get(source.name)
        if cached_bundle is not None:
            return cached_bundle
        source_stage1 = dict(source.stage1_mutations or {})
        stage1_mutations = dict(base)
        stage1_mutations.update(source_stage1)
        stage1_key = stable_signature(stage1_mutations)
        if stage1_key not in stage1_cache:
            stage_started = time.monotonic()
            status(
                progress_path,
                f"{window}_stream_stage1_start",
                source=source.name,
                stage1_key=stage1_key[:16],
                unique_stage1_done=len(stage1_cache),
                unique_stage1_total=len({stable_signature(dict(base) | dict(src.stage1_mutations or {})) for src in sources}),
                stage1_mutation_count=len(stage1_mutations),
            )
            stage1_snapshots = tps.snapshots_for_experiment(dataset, stage1_mutations, research_snapshots=research_cache)
            stage1_cfg = tps.OLRConfig.from_mapping(dataset.config, stage1_mutations)
            contexts = tps.afternoon_contexts_for_snapshots(dataset, stage1_snapshots, stage1_cfg)
            stage1_cache[stage1_key] = (stage1_snapshots, contexts)
            stage1_cache_order.append(stage1_key)
            while len(stage1_cache_order) > max(1, int(stage1_cache_limit)):
                evict_key = stage1_cache_order.pop(0)
                if evict_key != stage1_key:
                    stage1_cache.pop(evict_key, None)
            status(
                progress_path,
                f"{window}_stream_stage1_done",
                source=source.name,
                stage1_key=stage1_key[:16],
                elapsed_seconds=round(time.monotonic() - stage_started, 3),
                snapshots=len(stage1_snapshots),
                contexts=len(contexts),
            )
        stage1_snapshots, contexts = stage1_cache[stage1_key]
        stage2_started = time.monotonic()
        status(progress_path, f"{window}_stream_stage2_start", source=source.name, stage2_mutation_count=len(source.stage2_mutations or {}))
        cfg = tps.OLRConfig.from_mapping(dataset.config, source.mutations)
        selected_by_day: dict[Any, Any] = {}
        selections = []
        counts = {day: 0 for day in eligible_dates}
        last_stage2_heartbeat = stage2_started
        for day_index, day in enumerate(eligible_dates, start=1):
            base_snapshot = stage1_snapshots.get(day)
            if base_snapshot is None:
                continue
            selected = tps.afternoon_selection_from_contexts(base_snapshot, contexts.get(day, {}), cfg)
            selected_by_day[day] = selected
            for candidate in selected.candidates[: max(1, int(cfg.overnight_slot_count))]:
                next_day = next_by_date.get(day)
                if next_day is None:
                    continue
                if not dataset.bars_by_key.get((day, candidate.symbol)) or not dataset.bars_by_key.get((next_day, candidate.symbol)):
                    continue
                selections.append(tps.FixedSelection(day, candidate.symbol, source.name, candidate))
                counts[day] = counts.get(day, 0) + 1
            elapsed_stage2 = time.monotonic() - stage2_started
            if elapsed_stage2 >= 120.0 and (time.monotonic() - last_stage2_heartbeat >= 120.0 or day_index == len(eligible_dates)):
                status(
                    progress_path,
                    f"{window}_stream_stage2_progress",
                    source=source.name,
                    elapsed_seconds=round(elapsed_stage2, 3),
                    day_index=day_index,
                    eligible_dates=len(eligible_dates),
                    selected_days=len(selected_by_day),
                    fixed_selections=len(selections),
                    current_day=day.isoformat(),
                )
                last_stage2_heartbeat = time.monotonic()
            if stage2_timeout_seconds and elapsed_stage2 > float(stage2_timeout_seconds):
                raise TimeoutError(
                    f"stage2 timeout after {elapsed_stage2:.1f}s for {source.name} "
                    f"at day {day_index}/{len(eligible_dates)}"
                )
        source_hash = tps._snapshot_hash(selected_by_day)
        source_payload = {"name": source.name, "hash": source_hash, "count": len(selections)}
        candidate_hash = stable_signature([source_payload])
        compiled = tps.CompiledExecutionSet(
            dataset=dataset,
            sources=(source,),
            snapshots_by_source={source.name: selected_by_day},
            selections_by_source={source.name: tuple(sorted(selections, key=lambda item: (item.trade_date, item.symbol)))},
            selection_counts_by_source={source.name: counts},
            eligible_dates=eligible_dates,
            next_session_by_date=next_by_date,
            source_fingerprint=stable_signature([dataset.source_fingerprint, candidate_hash, True, include_holdout]),
            candidate_artifact_hash=candidate_hash,
            fast_cache_enabled=True,
        )
        bundle = _bundle_for_source(compiled, source, dates, candidate_only=False)
        bundle_cache[source.name] = bundle
        status(
            progress_path,
            f"{window}_stream_stage2_done",
            source=source.name,
            elapsed_seconds=round(time.monotonic() - stage2_started, 3),
            selected_days=len(selected_by_day),
            fixed_selections=len(selections),
        )
        return bundle

    status(
        progress_path,
        f"{window}_stream_sources_start",
        candidates=len(execution_candidates),
        unique_sources=len(needed_source_signatures),
        unique_stage1=len({stable_signature(dict(base) | dict(source_by_signature[sig].stage1_mutations or {})) for sig in needed_source_signatures}),
        batch_size=batch_size,
    )
    for index, candidate in enumerate(execution_candidates, start=1):
        signature = stable_signature(helper.selection_mutations(candidate.mutations))
        source = source_by_signature[signature]
        status(progress_path, f"{window}_candidate_start", label=candidate.label, kind=candidate.kind, index=index, batch_total=len(execution_candidates), source=source.name)
        try:
            started = time.monotonic()
            bundle = compile_source(source)
            result = run_olr_backtest({**window_cfg, "capability_level": "compiled"}, candidate.mutations, replay_bundle=bundle)
            row = {
                "label": candidate.label,
                "kind": candidate.kind,
                "reason": candidate.reason,
                "window": window,
                "mutations": candidate.mutations,
                "metrics": dict(result.metrics),
                "source": {
                    "source_fingerprint": result.source_fingerprint,
                    "candidate_snapshot_hash": result.candidate_snapshot_hash,
                    "feature_bundle_hash": result.feature_bundle_hash,
                    "capability_level": result.capability_level,
                    "date_start": dates[0].isoformat(),
                    "date_end": dates[-1].isoformat(),
                    "date_count": len(dates),
                    "streaming_compile": True,
                },
                "trade_rows": helper.trade_rows(result.trades),
                "decision_summary": helper.decision_summary(result.decisions),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            row = {
                "label": candidate.label,
                "kind": candidate.kind,
                "reason": candidate.reason,
                "window": window,
                "mutations": candidate.mutations,
                "metrics": {},
                "source": {"error": repr(exc), "streaming_compile": True},
                "trade_rows": [],
                "decision_summary": {},
                "elapsed_seconds": 0.0,
                "error": repr(exc),
            }
        append_jsonl(eval_path, row)
        cached[(window, candidate.label)] = row
        rows.append(row)
        bundle_cache.pop(source.name, None)
        status(
            progress_path,
            f"{window}_candidate_done",
            label=candidate.label,
            net_return_pct=helper.metric_net(row["metrics"]),
            trades=helper.metric_trades(row["metrics"]),
            win_rate=helper.metric_win(row["metrics"]),
            drawdown=helper.metric_dd(row["metrics"]),
            elapsed_seconds=row.get("elapsed_seconds", 0.0),
            error=row.get("error", ""),
        )
    by_label = {(row["window"], row["label"]): row for row in rows}
    return [by_label[(window, candidate.label)] for candidate in requested if (window, candidate.label) in by_label]


def summarize(
    helper: Any,
    candidates: Sequence[Any],
    oos_by_label: dict[str, dict[str, Any]],
    train_by_label: dict[str, dict[str, Any]],
    started: float,
    *,
    holdout_days: int,
) -> dict[str, Any]:
    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    rows = []
    for label in sorted(oos_by_label):
        if label not in candidate_by_label:
            continue
        row = combined_row(helper, candidate_by_label[label], oos_by_label.get(label, {}), train_by_label.get(label, {}))
        row["oos_score"] = oos_score(helper, row.get("oos", {}).get("metrics", {}))
        row["balanced_score"] = balanced_score(helper, row, oos_by_label, train_by_label)
        row["delta_vs_current"] = metric_deltas(helper, row, BASELINE_CURRENT, oos_by_label, train_by_label)
        row["delta_vs_no_filter"] = metric_deltas(helper, row, BASELINE_NO_FILTER, oos_by_label, train_by_label)
        rows.append(row)

    confirmed = [row for row in rows if row.get("train", {}).get("metrics")]
    confirmed.sort(key=lambda row: row["balanced_score"], reverse=True)
    oos_ranked = sorted(rows, key=lambda row: row["oos_score"], reverse=True)
    data_confirmed = [row for row in confirmed if is_sector_data_candidate(row)]
    strict_vs_current = [
        row
        for row in data_confirmed
        if helper.metric_net(row["oos"]["metrics"]) > helper.metric_net(oos_by_label[BASELINE_CURRENT]["metrics"])
        and helper.metric_trades(row["oos"]["metrics"]) >= helper.metric_trades(oos_by_label[BASELINE_CURRENT]["metrics"])
        and helper.metric_net(row["train"]["metrics"]) >= 0.98 * helper.metric_net(train_by_label[BASELINE_CURRENT]["metrics"])
        and helper.metric_dd(row["train"]["metrics"]) <= helper.metric_dd(train_by_label[BASELINE_CURRENT]["metrics"]) + 0.025
    ]
    strict_vs_no_filter = [
        row
        for row in data_confirmed
        if helper.metric_net(row["oos"]["metrics"]) > helper.metric_net(oos_by_label[BASELINE_NO_FILTER]["metrics"])
        and helper.metric_net(row["train"]["metrics"]) > helper.metric_net(train_by_label[BASELINE_NO_FILTER]["metrics"])
        and helper.metric_trades(row["oos"]["metrics"]) >= helper.metric_trades(oos_by_label[BASELINE_NO_FILTER]["metrics"]) * 0.85
    ]
    best_data = strict_vs_current[0] if strict_vs_current else (strict_vs_no_filter[0] if strict_vs_no_filter else (data_confirmed[0] if data_confirmed else {}))
    baselines = {
        label: combined_row(helper, candidate_by_label[label], oos_by_label.get(label, {}), train_by_label.get(label, {}))
        for label in (BASELINE_CURRENT, BASELINE_NO_FILTER, BASELINE_REJECT300)
        if label in candidate_by_label
    }
    trade_diffs = {}
    if best_data:
        trade_diffs = {
            "best_vs_current_oos": trade_delta_summary(best_data.get("oos_trade_rows", []), oos_by_label[BASELINE_CURRENT].get("trade_rows", [])),
            "best_vs_no_filter_oos": trade_delta_summary(best_data.get("oos_trade_rows", []), oos_by_label[BASELINE_NO_FILTER].get("trade_rows", [])),
            "best_vs_current_train": trade_delta_summary(best_data.get("train_trade_rows", []), train_by_label.get(BASELINE_CURRENT, {}).get("trade_rows", [])),
            "best_vs_no_filter_train": trade_delta_summary(best_data.get("train_trade_rows", []), train_by_label.get(BASELINE_NO_FILTER, {}).get("trade_rows", [])),
        }

    return {
        "generated_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "strategy": "olr",
        "source_round": 4,
        "holdout_days": holdout_days,
        "candidate_count": len(candidates),
        "oos_evaluated": len(oos_by_label),
        "train_confirmed": len(train_by_label),
        "kind_counts": dict(Counter(candidate.kind for candidate in candidates)),
        "baselines": baselines,
        "best_data_alpha": drop_trade_rows(best_data),
        "strict_data_beats_current": [drop_trade_rows(row) for row in strict_vs_current[:20]],
        "strict_data_beats_no_filter": [drop_trade_rows(row) for row in strict_vs_no_filter[:20]],
        "confirmed_ranked": [drop_trade_rows(row) for row in confirmed[:80]],
        "oos_ranked": [drop_trade_rows(row) for row in oos_ranked[:80]],
        "family_leaders": family_leaders([drop_trade_rows(row) for row in confirmed]),
        "trade_diffs": trade_diffs,
    }


def combined_row(helper: Any, candidate: Any, oos: dict[str, Any], train: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": candidate.label,
        "kind": candidate.kind,
        "reason": candidate.reason,
        "mutations": copy.deepcopy(candidate.mutations),
        "changed_from_round4": changed_from_base(read_json(ROUND4_CONFIG)["mutations"], candidate.mutations),
        "score_band_rules": candidate.mutations.get("olr.afternoon.score_band_rules", []),
        "oos": compact_eval(helper, oos),
        "train": compact_eval(helper, train),
        "oos_trade_rows": oos.get("trade_rows", []),
        "train_trade_rows": train.get("trade_rows", []),
    }


def compact_eval(helper: Any, row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "metrics": helper.compact_eval(row).get("metrics", {}),
        "source": row.get("source", {}),
        "decision_summary": row.get("decision_summary", {}),
        "sector_rejections": {
            "sector_daily_score_below_floor": reason_count(row, "sector_daily_score_below_floor"),
            "intraday_sector_score_below_floor": reason_count(row, "intraday_sector_score_below_floor"),
            "afternoon_score_band_rule_miss": reason_count(row, "afternoon_score_band_rule_miss"),
        },
        "elapsed_seconds": row.get("elapsed_seconds", 0.0),
        "error": row.get("error", ""),
    }


def balanced_score(helper: Any, row: dict[str, Any], oos_by_label: dict[str, dict[str, Any]], train_by_label: dict[str, dict[str, Any]]) -> float:
    oos = row.get("oos", {}).get("metrics", {})
    train = row.get("train", {}).get("metrics", {})
    if not train:
        return -1e9 + oos_score(helper, oos)
    current_oos = oos_by_label.get(BASELINE_CURRENT, {}).get("metrics", {})
    current_train = train_by_label.get(BASELINE_CURRENT, {}).get("metrics", {})
    no_filter_oos = oos_by_label.get(BASELINE_NO_FILTER, {}).get("metrics", {})
    no_filter_train = train_by_label.get(BASELINE_NO_FILTER, {}).get("metrics", {})
    score = 950.0 * helper.metric_net(oos)
    score += 280.0 * helper.metric_net(train)
    score += 2.0 * helper.metric_trades(oos)
    score += 0.20 * helper.metric_trades(train)
    score += 55.0 * helper.metric_win(oos)
    score -= 260.0 * max(0.0, helper.metric_dd(oos) - max(0.065, helper.metric_dd(current_oos)))
    score -= 180.0 * max(0.0, helper.metric_dd(train) - max(0.10, helper.metric_dd(current_train)))
    if is_sector_data_candidate(row):
        score += 80.0 * max(0.0, helper.metric_net(oos) - helper.metric_net(no_filter_oos))
        score += 50.0 * max(0.0, helper.metric_net(train) - helper.metric_net(no_filter_train))
        score -= 360.0 * max(0.0, 0.97 * helper.metric_net(current_train) - helper.metric_net(train))
    return score


def oos_score(helper: Any, metrics: dict[str, Any]) -> float:
    net = helper.metric_net(metrics)
    trades = helper.metric_trades(metrics)
    win = helper.metric_win(metrics)
    dd = helper.metric_dd(metrics)
    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    return 1000.0 * net + 2.2 * trades + 65.0 * win + 8.0 * math.tanh((pf - 1.5) / 1.0) - 220.0 * max(0.0, dd - 0.065)


def metric_deltas(
    helper: Any,
    row: dict[str, Any],
    baseline_label: str,
    oos_by_label: dict[str, dict[str, Any]],
    train_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_oos = (oos_by_label.get(baseline_label) or {}).get("metrics", {})
    base_train = (train_by_label.get(baseline_label) or {}).get("metrics", {})
    oos = row.get("oos", {}).get("metrics", {})
    train = row.get("train", {}).get("metrics", {})
    return {
        "oos_net_delta": helper.metric_net(oos) - helper.metric_net(base_oos),
        "oos_trade_delta": helper.metric_trades(oos) - helper.metric_trades(base_oos),
        "oos_win_delta": helper.metric_win(oos) - helper.metric_win(base_oos),
        "oos_dd_delta": helper.metric_dd(oos) - helper.metric_dd(base_oos),
        "train_net_delta": helper.metric_net(train) - helper.metric_net(base_train) if train else None,
        "train_trade_delta": helper.metric_trades(train) - helper.metric_trades(base_train) if train else None,
        "train_win_delta": helper.metric_win(train) - helper.metric_win(base_train) if train else None,
        "train_dd_delta": helper.metric_dd(train) - helper.metric_dd(base_train) if train else None,
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
        "added_by_sector": group_trades(added, "candidate_sector"),
        "removed_by_sector": group_trades(removed, "candidate_sector"),
        "added_score_bins": score_bins(added),
        "removed_score_bins": score_bins(removed),
        "worst_added": [compact_trade(row) for row in sorted(added, key=lambda item: fnum(item.get("net_pnl")))[:10]],
        "best_added": [compact_trade(row) for row in sorted(added, key=lambda item: fnum(item.get("net_pnl")), reverse=True)[:10]],
        "worst_removed": [compact_trade(row) for row in sorted(removed, key=lambda item: fnum(item.get("net_pnl")))[:10]],
        "best_removed": [compact_trade(row) for row in sorted(removed, key=lambda item: fnum(item.get("net_pnl")), reverse=True)[:10]],
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
        "avg_r": statistics.fmean(fnum(row.get("r")) for row in rows),
        "avg_score": statistics.fmean(fnum(row.get("candidate_score")) for row in rows),
    }


def group_trades(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "UNKNOWN")].append(row)
    out = []
    for name, group in groups.items():
        payload = summarize_trades(group)
        payload[key] = name
        out.append(payload)
    out.sort(key=lambda item: (item["net_pnl"], -item["count"]))
    return out[:12]


def score_bins(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [(0, 300), (300, 350), (350, 400), (400, 425), (425, 450), (450, 475), (475, 500), (500, 550), (550, 600), (600, 650), (650, 100000)]
    out = []
    for lo, hi in bins:
        group = [row for row in rows if lo <= fnum(row.get("candidate_score")) < hi]
        if group:
            payload = summarize_trades(group)
            payload["score_bin"] = f"{lo}-{hi}"
            out.append(payload)
    return out


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


def family_leaders(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[family(row["label"])].append(row)
    leaders = []
    for name, group in groups.items():
        group.sort(key=lambda row: row.get("balanced_score", -1e9), reverse=True)
        leader = copy.deepcopy(group[0])
        leader["family"] = name
        leaders.append(leader)
    leaders.sort(key=lambda row: row.get("balanced_score", -1e9), reverse=True)
    return leaders[:40]


def family(label: str) -> str:
    if label.startswith("data_band_"):
        return "_".join(label.split("_")[:4])
    if label.startswith("data_midconfirm_"):
        return "data_midconfirm"
    if label.startswith("data_global_intraday_floor_"):
        return "data_global_intraday_floor"
    if label.startswith("data_sectorregw_"):
        return "data_sectorregw"
    if label.startswith("data_frontier_"):
        return "_".join(label.split("_")[:3])
    if label.startswith("data_pool_top"):
        return "data_pool_size"
    if label.startswith("combo_nofilter_"):
        return "_".join(label.split("_")[:3])
    if label.startswith("combo_hardfilter_"):
        return "_".join(label.split("_")[:3])
    if label.startswith("soft_hardfilter_"):
        return "_".join(label.split("_")[:3])
    if label.startswith("soft_nofilter_"):
        return "_".join(label.split("_")[:3])
    return label.split("_daily")[0]


def render_markdown(helper: Any, payload: dict[str, Any]) -> str:
    lines = [
        "# OLR Sector-Data Alpha Sweep",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Candidates: {payload['candidate_count']} OOS evaluated, {payload['train_confirmed']} train-confirmed.",
        f"- Holdout: {payload['holdout_days']} days.",
        "",
        "## Baselines",
        "| Label | OOS net | OOS trades | OOS win | OOS DD | Train net | Train trades | Train win | Train DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in (BASELINE_CURRENT, BASELINE_NO_FILTER, BASELINE_REJECT300):
        row = payload["baselines"].get(label, {})
        lines.append(metrics_row(helper, label, row))
    lines.extend(["", "## Best Sector-Data Candidate"])
    best = payload.get("best_data_alpha") or {}
    if best:
        lines.append(f"- Label: `{best['label']}`")
        lines.append(f"- Kind: `{best['kind']}`")
        lines.append(f"- Reason: {best['reason']}")
        lines.append(
            "- OOS: "
            f"{pct(helper.metric_net(best['oos']['metrics']))} net, "
            f"{helper.metric_trades(best['oos']['metrics']):.0f} trades, "
            f"{pct(helper.metric_win(best['oos']['metrics']))} win, "
            f"{pct(helper.metric_dd(best['oos']['metrics']))} DD."
        )
        lines.append(
            "- Train: "
            f"{pct(helper.metric_net(best['train']['metrics']))} net, "
            f"{helper.metric_trades(best['train']['metrics']):.0f} trades, "
            f"{pct(helper.metric_win(best['train']['metrics']))} win, "
            f"{pct(helper.metric_dd(best['train']['metrics']))} DD."
        )
        lines.append(f"- Delta vs current: {format_delta(best.get('delta_vs_current', {}))}")
        lines.append(f"- Delta vs no-filter: {format_delta(best.get('delta_vs_no_filter', {}))}")
        lines.append("")
        lines.append("Changed mutations:")
        for key, value in best.get("changed_from_round4", {}).items():
            lines.append(f"- `{key}`: `{value['from']}` -> `{value['to']}`")
    else:
        lines.append("- No train-confirmed sector-data candidate was available.")

    lines.extend(["", "## Strict Sector-Data Candidates"])
    if payload.get("strict_data_beats_current"):
        lines.append("- At least one sector-data candidate beat current round 4 on OOS, OOS trades, and retained at least 98% of train net.")
    else:
        lines.append("- No sector-data candidate beat the current hard-sector round-4 baseline under the strict robustness criteria.")
    lines.append(f"- Candidates beating the no-filter baseline on both OOS and train: {len(payload.get('strict_data_beats_no_filter', []))}.")

    lines.extend(["", "## Top Confirmed", "| Label | Kind | OOS net | OOS trades | OOS win | OOS DD | Train net | Train trades | Train win | Train DD |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in payload.get("confirmed_ranked", [])[:25]:
        lines.append(metrics_row(helper, row["label"], row, include_kind=True))

    diffs = payload.get("trade_diffs", {})
    if diffs:
        lines.extend(["", "## Trade Deltas"])
        for name in ("best_vs_current_oos", "best_vs_no_filter_oos", "best_vs_current_train", "best_vs_no_filter_train"):
            diff = diffs.get(name, {})
            if not diff:
                continue
            lines.append(f"- `{name}`: added {diff['added_count']} trades ({money(diff['added']['net_pnl'])}, win {pct(diff['added']['win_rate'])}); removed {diff['removed_count']} trades ({money(diff['removed']['net_pnl'])}, win {pct(diff['removed']['win_rate'])}); common PnL delta {money(diff['common_net_pnl_delta'])}.")
            worst = diff.get("worst_added", [])[:3]
            if worst:
                pretty = "; ".join(f"{item['entry_date']} {item['symbol']} {item['sector']} score {fnum(item['score']):.1f} {money(item['net_pnl'])}" for item in worst)
                lines.append(f"  Worst added: {pretty}.")
    lines.append("")
    return "\n".join(lines)


def metrics_row(helper: Any, label: str, row: dict[str, Any], *, include_kind: bool = False) -> str:
    oos = row.get("oos", {}).get("metrics", {})
    train = row.get("train", {}).get("metrics", {})
    cells = [
        f"`{label}`",
    ]
    if include_kind:
        cells.append(f"`{row.get('kind', '')}`")
    cells.extend(
        [
            pct(helper.metric_net(oos)),
            f"{helper.metric_trades(oos):.0f}",
            pct(helper.metric_win(oos)),
            pct(helper.metric_dd(oos)),
            pct(helper.metric_net(train)),
            f"{helper.metric_trades(train):.0f}",
            pct(helper.metric_win(train)),
            pct(helper.metric_dd(train)),
        ]
    )
    return "| " + " | ".join(cells) + " |"


def format_delta(delta: dict[str, Any]) -> str:
    if not delta:
        return "n/a"
    return (
        f"OOS net {pct(delta.get('oos_net_delta', 0.0), signed=True)}, "
        f"OOS trades {delta.get('oos_trade_delta', 0.0):+.0f}, "
        f"train net {pct(delta.get('train_net_delta', 0.0), signed=True)}, "
        f"train trades {delta.get('train_trade_delta', 0.0):+.0f}"
    )


def drop_trade_rows(row: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out.pop("oos_trade_rows", None)
    out.pop("train_trade_rows", None)
    return out


def is_sector_data_candidate(row: dict[str, Any]) -> bool:
    return str(row.get("kind", "")).startswith("sector_data") or str(row.get("label", "")).startswith("data_")


def with_rules(base: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
    mutations = copy.deepcopy(base)
    mutations["olr.afternoon.reject_score_min"] = 0.0
    mutations["olr.afternoon.reject_score_max"] = 0.0
    mutations["olr.afternoon.score_band_rules"] = copy.deepcopy(rules)
    return mutations


def with_mutations(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    mutations = copy.deepcopy(base)
    for key, value in extra.items():
        if value is not None:
            mutations[key] = copy.deepcopy(value)
    return mutations


def low_rule() -> dict[str, Any]:
    return {"name": "base_low_lt300", "max_score": LOW_TAIL_MAX}


def high_rule() -> dict[str, Any]:
    return {"name": "base_high_gt650", "min_score": HIGH_TAIL_MIN}


def mid_rule(lo: float, hi: float, **conditions: Any) -> dict[str, Any]:
    return {"name": f"mid_{num(lo)}_{num(hi)}", "min_score": float(lo), "max_score": float(hi), **copy.deepcopy(conditions)}


def changed_from_base(base: dict[str, Any], mutations: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    keys = sorted(set(base) | set(mutations))
    for key in keys:
        if base.get(key) != mutations.get(key):
            out[key] = {"from": base.get(key), "to": mutations.get(key)}
    return out


def reason_count(row: dict[str, Any], reason: str) -> int:
    for key, value in row.get("decision_summary", {}).get("reason_counts", []):
        if key == reason:
            return int(value)
    return 0


def num(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def wlabel(value: float) -> str:
    raw = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return raw.replace(".", "p")


def fnum(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def pct(value: Any, *, signed: bool = False) -> str:
    val = fnum(value) * 100.0
    return f"{val:+.2f}%" if signed else f"{val:.2f}%"


def money(value: Any) -> str:
    return f"{fnum(value):,.0f}"


def safe_label(label: str) -> str:
    out = []
    for char in label:
        if char.isalnum() or char in {"_", "-", "."}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_")[:160]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def status(path: Path, stage: str, **extra: Any) -> None:
    payload = {"timestamp_utc": utc_now(), "stage": stage, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
