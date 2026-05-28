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
ROUND3_CONFIG = ROOT / "data" / "backtests" / "output" / "olr" / "round_3" / "optimized_config.json"
DEFAULT_OUTPUT = ROOT / "tmp" / "olr_nonmonotone_score_band_search"
HOLDOUT_DAYS = 42
LOW_TAIL_MAX = 300.0
HIGH_TAIL_MIN = 650.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Search conditional/non-monotone OLR afternoon score-band rules.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--fresh-train", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--train-top", type=int, default=70)
    parser.add_argument("--train-min-oos-net", type=float, default=0.17)
    parser.add_argument("--train-min-trades", type=float, default=30.0)
    parser.add_argument("--family-leaders", type=int, default=1)
    parser.add_argument("--max-train-labels", type=int, default=90)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
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

    round3 = read_json(ROUND3_CONFIG)
    base = copy.deepcopy(round3["mutations"])
    candidates = build_candidates(helper, base)
    if int(args.max_candidates) > 0:
        mandatory = {candidate.label for candidate in candidates if candidate.kind == "baseline" or candidate.label.startswith("rule_equiv_")}
        limited: list[Any] = []
        for candidate in candidates:
            if candidate.label in mandatory or len(limited) < int(args.max_candidates):
                limited.append(candidate)
        candidates = dedupe_by_signature(helper, limited)

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
                    "score_band_rules": candidate.mutations.get("olr.afternoon.score_band_rules", []),
                    "mutations_delta": changed_from_base(base, candidate.mutations),
                }
                for candidate in candidates
            ],
        },
    )
    status(progress_path, "candidate_plan", total=len(candidates), counts=dict(Counter(candidate.kind for candidate in candidates)))

    cached = helper.load_cached_evaluations(eval_path)
    if args.fresh_train:
        removed_train = sum(1 for key in cached if key[0] == "train")
        cached = {key: row for key, row in cached.items() if key[0] != "train"}
        status(progress_path, "fresh_train_cache_ignored", removed_train=removed_train)
    oos_rows = helper.evaluate_candidates(
        config,
        candidates,
        "oos",
        output_dir,
        eval_path,
        progress_path,
        cached,
        holdout_days=int(args.holdout_days),
        batch_size=max(1, int(args.batch_size)),
    )
    oos_by_label = {row["label"]: row for row in oos_rows}
    train_labels = select_train_labels(
        oos_rows,
        top_n=max(1, int(args.train_top)),
        min_oos_net=float(args.train_min_oos_net),
        min_oos_trades=float(args.train_min_trades),
        family_leaders=max(0, int(args.family_leaders)),
        max_labels=max(0, int(args.max_train_labels)),
    )
    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    train_candidates = [candidate_by_label[label] for label in train_labels if label in candidate_by_label]
    status(progress_path, "train_confirm_plan", total=len(train_candidates), labels=train_labels)
    train_rows = helper.evaluate_candidates(
        config,
        train_candidates,
        "train",
        output_dir,
        eval_path,
        progress_path,
        cached,
        holdout_days=int(args.holdout_days),
        batch_size=max(1, int(args.batch_size)),
    )
    train_by_label = {row["label"]: row for row in train_rows}
    payload = summarize(helper, candidates, oos_by_label, train_by_label, started)
    write_json(output_dir / "nonmonotone_score_band_search.json", payload)
    write_text(output_dir / "nonmonotone_score_band_search.md", render_markdown(payload))
    if payload.get("best_balanced"):
        write_json(output_dir / "best_balanced_mutations.json", payload["best_balanced"].get("mutations", {}))
    if payload.get("best_oos_first"):
        write_json(output_dir / "best_oos_first_mutations.json", payload["best_oos_first"].get("mutations", {}))
    status(
        progress_path,
        "complete",
        elapsed_seconds=payload["elapsed_seconds"],
        result_path=str(output_dir / "nonmonotone_score_band_search.json"),
        summary_path=str(output_dir / "nonmonotone_score_band_search.md"),
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

    def add_baseline(label: str, reject_min: float, reason: str) -> None:
        mutations = copy.deepcopy(base)
        mutations.pop("olr.afternoon.score_band_rules", None)
        mutations["olr.afternoon.reject_score_min"] = float(reject_min)
        mutations["olr.afternoon.reject_score_max"] = HIGH_TAIL_MIN
        add(label, "baseline", mutations, reason)

    add_baseline("baseline_reject300_round3", 300.0, "Current round-3 optimized score notch.")
    add_baseline("baseline_reject400_round2", 400.0, "Round-2 style notch for comparison.")
    add_baseline("baseline_reject500_oos_branch", 500.0, "OOS-first reject-score branch from the prior sweep.")
    add_rule(helper, out, base, "rule_equiv_reject300", [], "Rule-engine equivalence check for the current round-3 notch.")
    add_rule(helper, out, base, "rule_equiv_reject500", [rule(300.0, 500.0, "mid_300_500")], "Rule-engine equivalence check for reject_score_min=500.")

    boundaries = [float(value) for value in range(300, 651, 25)]
    allowed_widths = {25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 350.0}
    for lo_index, lo in enumerate(boundaries[:-1]):
        for hi in boundaries[lo_index + 1 :]:
            if hi - lo in allowed_widths:
                add_rule(
                    helper,
                    out,
                    base,
                    f"allow_{num(lo)}_{num(hi)}",
                    [rule(lo, hi, f"mid_{num(lo)}_{num(hi)}")],
                    "Single added score interval inside the current 300-650 reject notch.",
                )

    for lo in [float(value) for value in range(380, 491, 10)]:
        for width in (20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0):
            hi = min(520.0, lo + width)
            if hi > lo and hi <= HIGH_TAIL_MIN:
                add_rule(
                    helper,
                    out,
                    base,
                    f"fine_allow_{num(lo)}_{num(hi)}",
                    [rule(lo, hi, f"fine_{num(lo)}_{num(hi)}")],
                    "Fine-grained interval around the 400-500 area that drove the reject500 OOS uplift.",
                )

    pair_specs = [
        ((300, 350), (425, 475)),
        ((300, 375), (425, 475)),
        ((300, 400), (450, 500)),
        ((350, 400), (450, 500)),
        ((375, 425), (450, 500)),
        ((400, 425), (450, 475)),
        ((400, 450), (475, 500)),
        ((425, 450), (475, 500)),
        ((425, 475), (500, 550)),
        ((450, 500), (550, 600)),
        ((450, 475), (600, 650)),
        ((475, 500), (600, 650)),
    ]
    for first, second in pair_specs:
        add_rule(
            helper,
            out,
            base,
            f"pair_{num(first[0])}_{num(first[1])}_{num(second[0])}_{num(second[1])}",
            [rule(*first, name=f"mid_{num(first[0])}_{num(first[1])}"), rule(*second, name=f"mid_{num(second[0])}_{num(second[1])}")],
            "Two disjoint added intervals to test genuinely non-monotone score acceptance.",
        )

    conditional_bands = [
        (300.0, 500.0),
        (350.0, 500.0),
        (400.0, 500.0),
        (425.0, 500.0),
        (450.0, 500.0),
        (400.0, 475.0),
        (425.0, 475.0),
        (450.0, 475.0),
        (475.0, 500.0),
        (450.0, 650.0),
    ]
    conditions = [
        ("rank1", {"max_rank": 1}),
        ("rank2", {"max_rank": 2}),
        ("rank3", {"max_rank": 3}),
        ("gap05", {"max_gap": 0.05}),
        ("gap08", {"max_gap": 0.08}),
        ("gap12", {"max_gap": 0.12}),
        ("close45", {"min_close_location": 0.45}),
        ("close55", {"min_close_location": 0.55}),
        ("close65", {"min_close_location": 0.65}),
        ("vwap0", {"min_vwap_ret": 0.0}),
        ("vwap005", {"min_vwap_ret": 0.005}),
        ("relvol075", {"min_rel_volume": 0.75}),
        ("relvol1", {"min_rel_volume": 1.0}),
        ("dd05", {"max_open_drawdown": 0.05}),
        ("dd10", {"max_open_drawdown": 0.10}),
        ("block_battery", {"blocked_sectors": ["BATTERY"]}),
        ("block_battery_steel", {"blocked_sectors": ["BATTERY", "STEEL"]}),
        ("block_battery_steel_auto", {"blocked_sectors": ["BATTERY", "STEEL", "AUTOMOTIVE"]}),
        ("allow_oos_positive_sectors", {"allowed_sectors": ["CHEMICALS", "SEMICONDUCTORS", "DEFENSE"]}),
        (
            "allow_broad_positive_sectors",
            {"allowed_sectors": ["CHEMICALS", "SEMICONDUCTORS", "DEFENSE", "ELECTRONICS", "IT", "SHIPBUILDING", "FINANCIAL"]},
        ),
    ]
    for lo, hi in conditional_bands:
        for condition_label, condition in conditions:
            add_rule(
                helper,
                out,
                base,
                f"cond_{num(lo)}_{num(hi)}_{condition_label}",
                [rule(lo, hi, f"mid_{num(lo)}_{num(hi)}", **condition)],
                "Band-specific condition: keep current tails and admit this middle interval only when the condition matches.",
            )

    combo_conditions = [
        ("rank2_gap05", {"max_rank": 2, "max_gap": 0.05}),
        ("rank2_gap08", {"max_rank": 2, "max_gap": 0.08}),
        ("rank2_vwap0", {"max_rank": 2, "min_vwap_ret": 0.0}),
        ("rank2_close55", {"max_rank": 2, "min_close_location": 0.55}),
        ("gap05_block_battery_steel", {"max_gap": 0.05, "blocked_sectors": ["BATTERY", "STEEL"]}),
        ("gap08_block_battery_steel_auto", {"max_gap": 0.08, "blocked_sectors": ["BATTERY", "STEEL", "AUTOMOTIVE"]}),
        ("vwap0_close55", {"min_vwap_ret": 0.0, "min_close_location": 0.55}),
        ("vwap0_relvol075", {"min_vwap_ret": 0.0, "min_rel_volume": 0.75}),
        ("dd05_gap08", {"max_open_drawdown": 0.05, "max_gap": 0.08}),
    ]
    for lo, hi in [(300.0, 500.0), (400.0, 500.0), (425.0, 500.0), (450.0, 500.0), (450.0, 475.0)]:
        for condition_label, condition in combo_conditions:
            add_rule(
                helper,
                out,
                base,
                f"cond2_{num(lo)}_{num(hi)}_{condition_label}",
                [rule(lo, hi, f"mid_{num(lo)}_{num(hi)}", **condition)],
                "Two-condition band-specific rule aimed at keeping reject500 OOS winners while filtering weak train reshuffles.",
            )

    return dedupe_by_signature(helper, out)


def add_rule(helper: Any, out: list[Any], base: dict[str, Any], label: str, middle_rules: list[dict[str, Any]], reason: str) -> None:
    mutations = copy.deepcopy(base)
    mutations["olr.afternoon.reject_score_min"] = 0.0
    mutations["olr.afternoon.reject_score_max"] = 0.0
    mutations["olr.afternoon.score_band_rules"] = [
        {"name": "base_low_lt300", "max_score": LOW_TAIL_MAX},
        *copy.deepcopy(middle_rules),
        {"name": "base_high_gt650", "min_score": HIGH_TAIL_MIN},
    ]
    out.append(helper.Candidate(safe_label(label), "score_band_rule", mutations, reason))


def rule(lo: float, hi: float, name: str, **conditions: Any) -> dict[str, Any]:
    return {"name": name, "min_score": float(lo), "max_score": float(hi), **copy.deepcopy(conditions)}


def select_train_labels(
    oos_rows: Sequence[dict[str, Any]],
    *,
    top_n: int,
    min_oos_net: float,
    min_oos_trades: float,
    family_leaders: int,
    max_labels: int,
) -> list[str]:
    ranked = sorted(oos_rows, key=lambda row: (oos_score(row.get("metrics") or {}), row["label"]), reverse=True)
    rows_by_label = {row["label"]: row for row in oos_rows}
    labels: list[str] = []
    seen: set[str] = set()
    mandatory = {
        "baseline_reject300_round3",
        "baseline_reject400_round2",
        "baseline_reject500_oos_branch",
        "rule_equiv_reject300",
        "rule_equiv_reject500",
    }

    def add(label: str) -> None:
        if label in seen or label not in rows_by_label:
            return
        seen.add(label)
        labels.append(label)

    for label in (
        "baseline_reject300_round3",
        "baseline_reject400_round2",
        "baseline_reject500_oos_branch",
        "rule_equiv_reject300",
        "rule_equiv_reject500",
    ):
        add(label)

    for row in ranked[:top_n]:
        add(row["label"])

    for row in ranked:
        metrics = row.get("metrics") or {}
        if metric_net(metrics) >= min_oos_net and metric_trades(metrics) >= min_oos_trades:
            add(row["label"])

    if family_leaders > 0:
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in oos_rows:
            by_family[family_name(row["label"])].append(row)
        leaders: list[dict[str, Any]] = []
        for rows in by_family.values():
            rows.sort(key=lambda row: (oos_score(row.get("metrics") or {}), row["label"]), reverse=True)
            leaders.extend(rows[:family_leaders])
        leaders.sort(key=lambda row: (oos_score(row.get("metrics") or {}), row["label"]), reverse=True)
        for row in leaders:
            add(row["label"])

    for label in focused_probe_labels():
        add(label)

    if max_labels > 0 and len(labels) > max_labels:
        kept: list[str] = []
        kept_seen: set[str] = set()

        def keep(label: str) -> None:
            if label in kept_seen:
                return
            kept_seen.add(label)
            kept.append(label)

        for label in labels:
            if label in mandatory:
                keep(label)
        for label in labels:
            if len(kept) >= max_labels:
                break
            keep(label)
        return kept
    return labels


def focused_probe_labels() -> tuple[str, ...]:
    return (
        "allow_350_475",
        "allow_400_475",
        "allow_425_475",
        "allow_450_475",
        "fine_allow_440_490",
        "fine_allow_450_480",
        "fine_allow_460_480",
        "fine_allow_460_490",
        "fine_allow_470_490",
        "cond_400_475_gap05",
        "cond_425_475_gap05",
        "cond_425_500_gap05",
        "cond_425_500_block_battery_steel",
        "cond_425_500_allow_oos_positive_sectors",
        "cond_450_475_gap05",
        "cond_450_500_gap05",
        "cond2_400_500_gap05_block_battery_steel",
        "cond2_425_500_gap05_block_battery_steel",
        "cond2_450_475_gap05_block_battery_steel",
        "cond2_450_500_gap05_block_battery_steel",
        "pair_300_350_425_475",
        "pair_400_425_450_475",
        "pair_425_475_500_550",
    )


def summarize(
    helper: Any,
    candidates: Sequence[Any],
    oos_by_label: dict[str, dict[str, Any]],
    train_by_label: dict[str, dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    baselines = {
        label: combined_row(label, candidate_by_label, oos_by_label, train_by_label)
        for label in (
            "baseline_reject300_round3",
            "baseline_reject400_round2",
            "baseline_reject500_oos_branch",
            "rule_equiv_reject300",
            "rule_equiv_reject500",
        )
    }
    baseline300 = baselines["baseline_reject300_round3"]
    baseline400 = baselines["baseline_reject400_round2"]
    baseline500 = baselines["baseline_reject500_oos_branch"]
    rows = [
        score_combined_row(combined_row(label, candidate_by_label, oos_by_label, train_by_label), baseline300, baseline400, baseline500)
        for label in sorted(set(oos_by_label) | set(train_by_label))
        if label in candidate_by_label
    ]
    confirmed = [row for row in rows if row.get("train")]
    confirmed.sort(key=lambda row: row["balanced_score"], reverse=True)
    oos_ranked = sorted(rows, key=lambda row: row["oos_score"], reverse=True)
    balanced_candidates = [
        row
        for row in confirmed
        if metric_net(row["oos"]["metrics"]) > metric_net(baseline300["oos"]["metrics"])
        and metric_trades(row["oos"]["metrics"]) >= metric_trades(baseline300["oos"]["metrics"])
        and metric_net(row["train"]["metrics"]) >= metric_net(baseline400["train"]["metrics"])
        and metric_dd(row["train"]["metrics"]) <= max(metric_dd(baseline300["train"]["metrics"]) + 0.025, metric_dd(baseline500["train"]["metrics"]))
    ]
    strict_candidates = [
        row
        for row in confirmed
        if metric_net(row["oos"]["metrics"]) > metric_net(baseline300["oos"]["metrics"])
        and metric_net(row["train"]["metrics"]) >= metric_net(baseline300["train"]["metrics"]) * 0.98
    ]
    oos_first_candidates = [
        row
        for row in confirmed
        if metric_trades(row["oos"]["metrics"]) >= metric_trades(baseline300["oos"]["metrics"])
        and metric_net(row["oos"]["metrics"]) > metric_net(baseline300["oos"]["metrics"])
    ]
    oos_first_candidates.sort(key=lambda row: (metric_net(row["oos"]["metrics"]), -metric_dd(row["oos"]["metrics"])), reverse=True)
    best_balanced = balanced_candidates[0] if balanced_candidates else (confirmed[0] if confirmed else {})
    best_oos_first = oos_first_candidates[0] if oos_first_candidates else (oos_ranked[0] if oos_ranked else {})
    payload = {
        "generated_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "candidate_count": len(candidates),
        "oos_evaluated": len(oos_by_label),
        "train_confirmed": len(train_by_label),
        "baselines": baselines,
        "equivalence_checks": equivalence_checks(baselines),
        "best_balanced": best_balanced,
        "best_oos_first": best_oos_first,
        "strict_no_material_train_deterioration": strict_candidates[:10],
        "balanced_ranked": balanced_candidates[:25],
        "confirmed_ranked": confirmed[:50],
        "oos_ranked": oos_ranked[:50],
        "family_leaders": family_leaders(confirmed),
        "best_trade_diffs": best_trade_diffs(best_balanced, baselines),
    }
    return payload


def combined_row(
    label: str,
    candidate_by_label: dict[str, Any],
    oos_by_label: dict[str, dict[str, Any]],
    train_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate = candidate_by_label.get(label)
    mutations = copy.deepcopy(candidate.mutations) if candidate else {}
    return {
        "label": label,
        "kind": candidate.kind if candidate else "",
        "reason": candidate.reason if candidate else "",
        "score_band_rules": mutations.get("olr.afternoon.score_band_rules", []),
        "mutations": mutations,
        "oos": compact_eval(oos_by_label.get(label, {})),
        "train": compact_eval(train_by_label.get(label, {})),
        "errors": [row.get("error") for row in (oos_by_label.get(label, {}), train_by_label.get(label, {})) if row.get("error")],
    }


def score_combined_row(row: dict[str, Any], baseline300: dict[str, Any], baseline400: dict[str, Any], baseline500: dict[str, Any]) -> dict[str, Any]:
    oos = row.get("oos", {}).get("metrics", {})
    train = row.get("train", {}).get("metrics", {})
    row["oos_score"] = oos_score(oos)
    row["balanced_score"] = balanced_score(oos, train, baseline300, baseline400, baseline500)
    row["delta_vs_round3"] = {
        "oos_net": metric_net(oos) - metric_net(baseline300["oos"]["metrics"]),
        "oos_trades": metric_trades(oos) - metric_trades(baseline300["oos"]["metrics"]),
        "train_net": metric_net(train) - metric_net(baseline300["train"]["metrics"]),
        "train_trades": metric_trades(train) - metric_trades(baseline300["train"]["metrics"]),
    }
    row["delta_vs_reject500"] = {
        "oos_net": metric_net(oos) - metric_net(baseline500["oos"]["metrics"]),
        "oos_trades": metric_trades(oos) - metric_trades(baseline500["oos"]["metrics"]),
        "train_net": metric_net(train) - metric_net(baseline500["train"]["metrics"]),
        "train_trades": metric_trades(train) - metric_trades(baseline500["train"]["metrics"]),
    }
    return row


def oos_score(metrics: dict[str, Any]) -> float:
    net = metric_net(metrics)
    trades = metric_trades(metrics)
    dd = metric_dd(metrics)
    win = metric_win(metrics)
    return 1000.0 * net + 2.0 * trades + 60.0 * win - 180.0 * max(0.0, dd - 0.06)


def balanced_score(oos: dict[str, Any], train: dict[str, Any], baseline300: dict[str, Any], baseline400: dict[str, Any], baseline500: dict[str, Any]) -> float:
    if not train:
        return -1e9 + oos_score(oos)
    score = 900.0 * metric_net(oos)
    score += 280.0 * metric_net(train)
    score += 1.5 * metric_trades(oos)
    score += 0.25 * metric_trades(train)
    score -= 500.0 * max(0.0, metric_net(baseline300["train"]["metrics"]) - metric_net(train))
    score -= 300.0 * max(0.0, metric_net(baseline400["train"]["metrics"]) - metric_net(train))
    score -= 250.0 * max(0.0, metric_dd(oos) - metric_dd(baseline500["oos"]["metrics"]))
    score -= 180.0 * max(0.0, metric_dd(train) - metric_dd(baseline300["train"]["metrics"]))
    return score


def equivalence_checks(baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "rule_equiv_reject300_minus_baseline": metric_deltas(baselines["rule_equiv_reject300"], baselines["baseline_reject300_round3"]),
        "rule_equiv_reject500_minus_baseline": metric_deltas(baselines["rule_equiv_reject500"], baselines["baseline_reject500_oos_branch"]),
    }


def metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "oos_net": metric_net(left["oos"]["metrics"]) - metric_net(right["oos"]["metrics"]),
        "oos_trades": metric_trades(left["oos"]["metrics"]) - metric_trades(right["oos"]["metrics"]),
        "train_net": metric_net(left["train"]["metrics"]) - metric_net(right["train"]["metrics"]),
        "train_trades": metric_trades(left["train"]["metrics"]) - metric_trades(right["train"]["metrics"]),
    }


def family_leaders(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[family_name(row["label"])].append(row)
    leaders = []
    for family, family_rows in groups.items():
        family_rows.sort(key=lambda row: row["balanced_score"], reverse=True)
        leader = copy.deepcopy(family_rows[0])
        leader["family"] = family
        leaders.append(leader)
    leaders.sort(key=lambda row: row["balanced_score"], reverse=True)
    return leaders[:30]


def best_trade_diffs(best: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not best:
        return {}
    out = {}
    for window in ("train", "oos"):
        best_trades = keyed_trades(best.get(window, {}).get("trade_rows", []))
        for label in ("baseline_reject300_round3", "baseline_reject500_oos_branch"):
            base_trades = keyed_trades(baselines[label].get(window, {}).get("trade_rows", []))
            out[f"{window}_best_vs_{label}"] = trade_delta_summary(best_trades, base_trades)
    return out


def keyed_trades(trades: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in trades:
        key = "|".join(str(row.get(field, "")) for field in ("entry_date", "symbol", "entry_fill_time", "exit_fill_time"))
        out[key] = row
    return out


def trade_delta_summary(current: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current_keys = set(current)
    base_keys = set(base)
    added = [current[key] for key in sorted(current_keys - base_keys)]
    removed = [base[key] for key in sorted(base_keys - current_keys)]
    common = sorted(current_keys & base_keys)
    return {
        "added_count": len(added),
        "removed_count": len(removed),
        "common_count": len(common),
        "added": summarize_trades(added),
        "removed": summarize_trades(removed),
        "common_net_pnl_delta": sum(fnum(current[key].get("net_pnl")) - fnum(base[key].get("net_pnl")) for key in common),
        "added_score_bins": score_bins(added),
        "worst_added": sorted([compact_trade(row) for row in added], key=lambda row: row["net_pnl"])[:10],
        "best_added": sorted([compact_trade(row) for row in added], key=lambda row: row["net_pnl"], reverse=True)[:10],
    }


def summarize_trades(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "net_pnl": 0.0, "avg_r": 0.0, "win_rate": 0.0, "avg_score": 0.0, "avg_rank": 0.0}
    return {
        "count": len(rows),
        "net_pnl": sum(fnum(row.get("net_pnl")) for row in rows),
        "avg_r": statistics.fmean(fnum(row.get("r")) for row in rows),
        "win_rate": sum(1 for row in rows if fnum(row.get("net_pnl")) > 0.0) / len(rows),
        "avg_score": statistics.fmean(fnum(row.get("candidate_score")) for row in rows),
        "avg_rank": statistics.fmean(fnum(row.get("candidate_rank")) for row in rows),
    }


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


def compact_eval(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    keys = (
        "official_mtm_net_return_pct",
        "net_return_pct",
        "total_trades",
        "entry_fill_count",
        "win_rate",
        "profit_factor",
        "official_mtm_max_drawdown_pct",
        "max_drawdown_pct",
        "expected_total_r",
        "entry_level_expected_total_r",
        "same_bar_fill_count",
        "forced_replay_close_count",
        "rejected_order_count",
        "end_open_position_count",
    )
    return {
        "label": row.get("label", ""),
        "metrics": {key: row.get("metrics", {}).get(key) for key in keys if key in row.get("metrics", {})},
        "source": row.get("source", {}),
        "trade_rows": row.get("trade_rows", []),
        "decision_summary": row.get("decision_summary", {}),
        "elapsed_seconds": row.get("elapsed_seconds", 0.0),
        "error": row.get("error", ""),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OLR Conditional Non-Monotone Score-Band Search",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Candidates evaluated OOS: {payload['oos_evaluated']}",
        f"- Candidates train-confirmed: {payload['train_confirmed']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']}",
        "",
        "## Baselines",
    ]
    for label, row in payload["baselines"].items():
        lines.append(format_row(row))
    lines.extend(
        [
            "",
            "## Equivalence Checks",
            f"- rule_equiv_reject300 minus baseline: {payload['equivalence_checks']['rule_equiv_reject300_minus_baseline']}",
            f"- rule_equiv_reject500 minus baseline: {payload['equivalence_checks']['rule_equiv_reject500_minus_baseline']}",
            "",
            "## Best Balanced Candidate",
            format_row(payload.get("best_balanced", {})),
            "",
            "## Best OOS-First Candidate",
            format_row(payload.get("best_oos_first", {})),
            "",
            "## Strict No-Material-Train-Deterioration Candidates",
        ]
    )
    if payload.get("strict_no_material_train_deterioration"):
        for row in payload["strict_no_material_train_deterioration"][:10]:
            lines.append(format_row(row))
    else:
        lines.append("- None.")
    lines.extend(["", "## Balanced Ranked"])
    for row in payload.get("balanced_ranked", [])[:20]:
        lines.append(format_row(row))
    lines.extend(["", "## Confirmed Ranked"])
    for row in payload.get("confirmed_ranked", [])[:25]:
        lines.append(format_row(row))
    return "\n".join(lines) + "\n"


def format_row(row: dict[str, Any]) -> str:
    if not row:
        return "- None."
    oos = row.get("oos", {}).get("metrics", {})
    train = row.get("train", {}).get("metrics", {})
    rules = row.get("score_band_rules") or []
    return (
        f"- {row.get('label', '')}: OOS {pct(metric_net(oos))}, trades {metric_trades(oos):.0f}, "
        f"win {pct(metric_win(oos))}, DD {pct(metric_dd(oos))}; "
        f"train {pct(metric_net(train))}, trades {metric_trades(train):.0f}, win {pct(metric_win(train))}, DD {pct(metric_dd(train))}; "
        f"rules={rules}"
    )


def changed_from_base(base: dict[str, Any], mutations: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sorted(mutations.items()) if base.get(key) != value}


def dedupe_by_signature(helper: Any, candidates: Sequence[Any]) -> list[Any]:
    out = []
    seen = set()
    for candidate in candidates:
        signature = helper.stable_signature(candidate.mutations)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(candidate)
    return out


def family_name(label: str) -> str:
    parts = label.split("_")
    if label.startswith("cond2_"):
        return "_".join(parts[:4])
    if label.startswith("cond_"):
        return "_".join(parts[:3])
    if label.startswith("fine_allow_"):
        return "fine_allow"
    if label.startswith("allow_"):
        return "allow"
    if label.startswith("pair_"):
        return "pair"
    return parts[0] if parts else label


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


def num(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def status(path: Path, stage: str, **payload: Any) -> None:
    row = {"ts": utc_now(), "stage": stage, **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    print(json.dumps(row, sort_keys=True, default=str), flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
