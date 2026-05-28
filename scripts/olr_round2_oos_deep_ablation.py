from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.auto.shared.cache_keys import stable_signature
from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.strategies.olr.allocation_holdout_eval import _bundle_for_source
from backtests.strategies.olr.phase_candidates import get_phase_candidates
from backtests.strategies.olr.runner import run_olr_backtest
from backtests.strategies.olr.trade_plan_sweep import CandidateSource, build_compiled_execution_set


DEFAULT_OUTPUT = ROOT / "tmp" / "olr_round2_oos_deep_ablation"
STAGE1_PREFIXES = (
    "olr.universe.",
    "olr.frontier.",
    "olr.discovery.",
    "olr.premarket.",
    "olr.research.",
    "olr.signal.",
)
SNAPSHOT_PREFIXES = (*STAGE1_PREFIXES, "olr.afternoon.")
EXECUTION_PREFIXES = (
    "olr.trade_plan.",
    "olr.execution.",
    "olr.allocation.",
    "olr.cost.",
    "olr.robustness.",
    "olr.overnight.",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    kind: str
    mutations: dict[str, Any]
    reason: str


@dataclass(frozen=True, slots=True)
class AcceptedStep:
    label: str
    group: str
    key: str
    value: Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep OLR round-2 OOS ablation and targeted repair sweep.")
    parser.add_argument("--config", default=str(ROOT / "config" / "optimization" / "olr.yaml"))
    parser.add_argument("--round-root", default=str(ROOT / "data" / "backtests" / "output" / "olr"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--holdout-days", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--max-oos", type=int, default=0)
    parser.add_argument("--top-train", type=int, default=85)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    eval_path = output_dir / "evaluations.jsonl"
    if args.fresh:
        for path in (progress_path, eval_path):
            if path.exists():
                path.unlink()

    config = normalize_runtime_config("olr", load_yaml_config(args.config))
    config["capability_level"] = "real_replay"
    config["holdout_days"] = int(args.holdout_days)
    config["use_full_available_window"] = True
    round_root = Path(args.round_root)
    round1 = read_json(round_root / "round_1" / "optimized_config.json")
    round2 = read_json(round_root / "round_2" / "optimized_config.json")
    round2_state = read_json(round_root / "round_2" / "phase_state.json")
    initial = dict(config.get("initial_mutations") or {})
    round1_final = dict(round1.get("mutations") or {})
    round2_final = dict(round2.get("mutations") or {})

    status(progress_path, "build_candidate_plan_start")
    steps = accepted_steps(initial, round1, round2_state)
    base_candidates = build_base_candidates(initial, round1_final, round2_final, steps)
    if args.quick:
        phase_specs = []
        perturbations = build_curated_perturbations(initial, round1_final, round2_final, limit_grid=True)
    else:
        phase_specs = build_phase_spec_candidates(round2_final)
        perturbations = build_curated_perturbations(initial, round1_final, round2_final, limit_grid=False)
    candidates = dedupe_candidates([*base_candidates, *phase_specs, *perturbations])
    if int(args.max_oos) > 0:
        mandatory = {
            "initial_config",
            "round_1_final",
            "round_2_final",
        }
        kept = []
        for candidate in candidates:
            if len(kept) < int(args.max_oos) or candidate.label in mandatory or candidate.kind.endswith("ablation"):
                kept.append(candidate)
        candidates = dedupe_candidates(kept)
    write_json(
        output_dir / "candidate_plan.json",
        {
            "generated_at_utc": utc_now(),
            "candidate_count": len(candidates),
            "counts": dict(Counter(item.kind for item in candidates)),
            "accepted_steps": [asdict(step) for step in steps],
            "candidates": [candidate_payload(item, include_mutations=True) for item in candidates],
        },
    )
    status(progress_path, "candidate_plan", total=len(candidates), counts=dict(Counter(item.kind for item in candidates)))

    cached = load_cached_evaluations(eval_path)
    final_candidate = next(item for item in candidates if item.label == "round_2_final")
    final_oos = evaluate_candidates(
        config,
        [final_candidate],
        "oos",
        output_dir,
        eval_path,
        progress_path,
        cached,
        holdout_days=int(args.holdout_days),
        batch_size=1,
    )[0]
    edge = edge_case_diagnostics(final_oos)
    write_json(output_dir / "round2_final_oos_edge_diagnostics.json", edge)
    targeted = build_targeted_repair_candidates(round2_final, edge, final_oos)
    new_targeted = [item for item in dedupe_candidates(targeted) if stable_signature(item.mutations) not in {stable_signature(c.mutations) for c in candidates}]
    candidates = dedupe_candidates([*candidates, *new_targeted])
    status(progress_path, "targeted_candidate_plan", total=len(new_targeted), counts=dict(Counter(item.reason for item in new_targeted)))

    oos_results = evaluate_candidates(
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
    oos_by_label = {row["label"]: row for row in oos_results}
    oos_ranked = sorted(oos_results, key=lambda row: oos_score(row["metrics"]), reverse=True)
    mandatory_train = {
        "initial_config",
        "round_1_final",
        "round_2_final",
        *(row["label"] for row in oos_results if row.get("kind") in {"accepted_step_ablation", "phase_ablation", "round_ablation"}),
    }
    train_labels = set(row["label"] for row in oos_ranked[: max(0, int(args.top_train))])
    train_labels.update(mandatory_train)
    train_candidates = [item for item in candidates if item.label in train_labels]
    status(progress_path, "train_confirm_plan", total=len(train_candidates))
    train_results = evaluate_candidates(
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
    train_by_label = {row["label"]: row for row in train_results}

    final_train = train_by_label.get("round_2_final")
    if final_train is None:
        final_train = evaluate_candidates(
            config,
            [final_candidate],
            "train",
            output_dir,
            eval_path,
            progress_path,
            cached,
            holdout_days=int(args.holdout_days),
            batch_size=1,
        )[0]
        train_by_label["round_2_final"] = final_train

    confirmed = []
    for label, oos in oos_by_label.items():
        train = train_by_label.get(label)
        if not train:
            continue
        confirmed.append(
            {
                "label": label,
                "kind": oos.get("kind", ""),
                "reason": oos.get("reason", ""),
                "combined_score": combined_score(oos["metrics"], train["metrics"], final_train["metrics"], oos_by_label["round_2_final"]["metrics"]),
                "oos": compact_eval(oos),
                "train": compact_eval(train),
            }
        )
    confirmed.sort(key=lambda row: row["combined_score"], reverse=True)

    payload = {
        "generated_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "strategy": "olr",
        "target_round": 2,
        "holdout_days": int(args.holdout_days),
        "candidate_count": len(candidates),
        "counts": {
            "oos_evaluated": len(oos_results),
            "train_confirmed": len(train_results),
            "targeted_candidates": len(new_targeted),
        },
        "candidate_kind_counts": dict(Counter(item.kind for item in candidates)),
        "target_baseline": {
            "oos": compact_eval(oos_by_label["round_2_final"]),
            "train": compact_eval(final_train),
        },
        "round1_baseline": compact_eval(oos_by_label.get("round_1_final", {})),
        "initial_baseline": compact_eval(oos_by_label.get("initial_config", {})),
        "edge_case_diagnostics": edge,
        "oos_ranked": [compact_eval(row) | {"oos_score": oos_score(row["metrics"])} for row in oos_ranked],
        "confirmed_train_ranked": confirmed,
        "ablation_impacts": build_ablation_impacts(oos_by_label, train_by_label, "round_2_final"),
        "top_recommendation": confirmed[0] if confirmed else {},
    }
    write_json(output_dir / "oos_ablation_results.json", payload)
    write_json(output_dir / "recommended_mutations.json", payload.get("top_recommendation", {}))
    (output_dir / "oos_ablation_summary.md").write_text(render_markdown(payload), encoding="utf-8")
    status(
        progress_path,
        "complete",
        result_path=str(output_dir / "oos_ablation_results.json"),
        summary_path=str(output_dir / "oos_ablation_summary.md"),
        elapsed_seconds=payload["elapsed_seconds"],
    )
    return 0


def accepted_steps(initial: dict[str, Any], round1: dict[str, Any], round2_state: dict[str, Any]) -> list[AcceptedStep]:
    final1 = dict(round1.get("mutations") or {})
    source = dict(round1.get("candidate_source") or {})
    steps: list[AcceptedStep] = []
    included: set[str] = set()

    def add_group(group: str, mutations: dict[str, Any]) -> None:
        for key in sorted(mutations):
            if final1.get(key, mutations[key]) != mutations[key]:
                continue
            if key in included:
                continue
            included.add(key)
            steps.append(AcceptedStep(f"round1_{group}_{safe_label(key)}", f"round1_{group}", key, deepcopy(mutations[key])))

    add_group("stage1", dict(source.get("stage1_mutations") or {}))
    add_group("stage2", dict(source.get("stage2_mutations") or {}))
    allocation = allocation_to_mutations(dict(round1.get("allocation") or {}))
    add_group("allocation", allocation)
    for key in ("olr.trade_plan.entry", "olr.trade_plan.exit"):
        if key in final1 and key not in included:
            included.add(key)
            steps.append(AcceptedStep(f"round1_execution_{safe_label(key)}", "round1_execution", key, deepcopy(final1[key])))
    for key in sorted(final1):
        if key.startswith("_"):
            continue
        if initial.get(key, object()) != final1[key] and key not in included:
            included.add(key)
            steps.append(AcceptedStep(f"round1_remaining_{safe_label(key)}", "round1_remaining", key, deepcopy(final1[key])))

    phase_results = dict(round2_state.get("phase_results") or {})
    for phase_key in sorted(phase_results, key=lambda item: int(item) if str(item).isdigit() else 999):
        result = phase_results.get(phase_key) or {}
        mutations = dict(result.get("new_mutations") or {})
        for key in sorted(mutations):
            steps.append(AcceptedStep(f"round2_phase{phase_key}_{safe_label(key)}", f"round2_phase_{phase_key}", key, deepcopy(mutations[key])))
    return steps


def build_base_candidates(
    initial: dict[str, Any],
    round1_final: dict[str, Any],
    round2_final: dict[str, Any],
    steps: Sequence[AcceptedStep],
) -> list[Candidate]:
    candidates = [
        Candidate("initial_config", "round_final", dict(initial), "Config initial OLR mutation set before round-1 promotion."),
        Candidate("round_1_final", "round_final", dict(round1_final), "Round 1 cumulative mutation set."),
        Candidate("round_2_final", "round_final", dict(round2_final), "Round 2 final cumulative mutation set."),
    ]
    cumulative = dict(initial)
    for index, step in enumerate(steps, start=1):
        cumulative[step.key] = deepcopy(step.value)
        candidates.append(
            Candidate(
                f"cumulative_step_{index:02d}_{safe_label(step.label)}",
                "accepted_step_cumulative",
                dict(cumulative),
                f"Cumulative accepted mutations through {step.label}.",
            )
        )
    group_to_steps: dict[str, list[AcceptedStep]] = defaultdict(list)
    for step in steps:
        group_to_steps[step.group].append(step)
    for group in sorted(group_to_steps):
        mutated = dict(initial)
        for step in steps:
            if step.group == group:
                continue
            mutated[step.key] = deepcopy(step.value)
        candidates.append(
            Candidate(
                f"drop_group_{safe_label(group)}",
                "phase_ablation",
                mutated,
                f"Round-2 final replayed after removing accepted group {group}.",
            )
        )
    for index, drop_step in enumerate(steps, start=1):
        mutated = dict(initial)
        for step in steps:
            if step is drop_step:
                continue
            mutated[step.key] = deepcopy(step.value)
        candidates.append(
            Candidate(
                f"drop_step_{index:02d}_{safe_label(drop_step.label)}",
                "accepted_step_ablation",
                mutated,
                f"Round-2 final replayed after removing accepted step {drop_step.label}.",
            )
        )
    for key in sorted(round2_final):
        if key.startswith("_"):
            continue
        if initial.get(key, object()) != round2_final[key]:
            mutated = dict(round2_final)
            if key in initial:
                mutated[key] = deepcopy(initial[key])
                reason = f"Revert {key} to initial config value."
            else:
                mutated.pop(key, None)
                reason = f"Remove {key}; it was absent from the initial config."
            candidates.append(Candidate(f"revert_to_initial_{safe_label(key)}", "key_ablation", mutated, reason))
        if round1_final.get(key, object()) != round2_final[key]:
            mutated = dict(round2_final)
            if key in round1_final:
                mutated[key] = deepcopy(round1_final[key])
                reason = f"Revert {key} to round-1 value."
            else:
                mutated.pop(key, None)
                reason = f"Remove {key}; it was absent from round 1."
            candidates.append(Candidate(f"revert_to_round1_{safe_label(key)}", "key_ablation", mutated, reason))
    return candidates


def build_phase_spec_candidates(final: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    for phase in range(1, 7):
        for experiment in get_phase_candidates(phase):
            mutated = dict(final)
            mutated.update(dict(experiment.mutations or {}))
            out.append(
                Candidate(
                    f"phase_spec_{phase}_{safe_label(experiment.name)}",
                    "phase_spec_candidate",
                    mutated,
                    f"Official phase-{phase} candidate applied to round-2 final.",
                )
            )
    return out


def build_curated_perturbations(
    initial: dict[str, Any],
    round1_final: dict[str, Any],
    final: dict[str, Any],
    *,
    limit_grid: bool,
) -> list[Candidate]:
    out: list[Candidate] = []

    def add(label: str, kind: str, updates: dict[str, Any], reason: str) -> None:
        mutated = dict(final)
        for key, value in updates.items():
            if value is None:
                mutated.pop(key, None)
            else:
                mutated[key] = deepcopy(value)
        out.append(Candidate(label, kind, mutated, reason))

    scalar_grid = {
        "olr.research.top_long_count": [15, 20, 25, 30, 40],
        "olr.overnight.slot_count": [3, 4, 5, 6],
        "olr.allocation.target_gross_exposure": [0.80, 1.00, 1.10, 1.20, 1.35],
        "olr.allocation.max_position_pct": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "olr.allocation.rank_decay": [1.00, 1.25, 1.50, 2.00],
        "olr.afternoon.reject_score_min": [0.0, 300.0, 350.0, 400.0, 450.0, 500.0],
        "olr.afternoon.reject_score_max": [0.0, 600.0, 650.0, 700.0, 750.0],
        "olr.afternoon.min_rel_volume": [0.0, 0.50, 0.75, 1.00],
        "olr.afternoon.min_close_location": [0.0, 0.45, 0.55, 0.65],
        "olr.afternoon.min_vwap_ret": [-0.02, -0.005, 0.0, 0.002],
        "olr.afternoon.min_flow_5d": [-9.99, 0.0],
        "olr.afternoon.min_sector_flow": [-9.99, 0.0],
    }
    for key, values in scalar_grid.items():
        for value in values:
            if final.get(key) == value:
                continue
            add(f"perturb_{safe_label(key)}_{safe_label(value)}", "key_perturbation", {key: value}, f"Curated scalar perturbation of {key}.")

    for mode in ("momentum", "hybrid", "daily_plus_intraday", "vwap_strength", "gap_hold", "flow_confirmed"):
        if final.get("olr.afternoon.score_mode") != mode:
            add(f"perturb_score_mode_{mode}", "key_perturbation", {"olr.afternoon.score_mode": mode}, "Rerank afternoon candidates using an alternate score mode.")
    for mode in ("hot", "hybrid", "score"):
        if final.get("olr.frontier.active_selection_mode") != mode:
            add(f"perturb_frontier_mode_{mode}", "key_perturbation", {"olr.frontier.active_selection_mode": mode}, "Rerank stage-1 candidates using an alternate frontier mode.")
    for key in ("olr.afternoon.require_close_above_prev", "olr.afternoon.use_lagged_flow_score"):
        if key in final:
            add(f"flip_{safe_label(key)}", "key_perturbation", {key: not bool(final[key])}, f"Boolean flip of {key}.")

    allocation_grid = [
        (1.00, 0.50, 1.00),
        (1.00, 0.55, 1.25),
        (1.10, 0.55, 1.50),
        (1.10, 0.60, 1.50),
        (1.20, 0.60, 1.25),
        (1.20, 0.65, 1.50),
        (1.35, 0.55, 2.00),
    ]
    if limit_grid:
        allocation_grid = allocation_grid[:4]
    for gross, cap, decay in allocation_grid:
        add(
            f"alloc_combo_g{safe_label(gross)}_cap{safe_label(cap)}_d{safe_label(decay)}",
            "allocation_combo",
            {
                "olr.allocation.mode": "rank_weighted",
                "olr.allocation.target_gross_exposure": gross,
                "olr.allocation.max_position_pct": cap,
                "olr.allocation.rank_decay": decay,
            },
            "Joint allocation perturbation to separate leverage from rank concentration.",
        )

    notch_grid = [
        (None, None),
        (300.0, 650.0),
        (350.0, 650.0),
        (400.0, 600.0),
        (400.0, 650.0),
        (450.0, 700.0),
        (500.0, 700.0),
    ]
    if limit_grid:
        notch_grid = notch_grid[:4]
    for lo, hi in notch_grid:
        updates = {}
        suffix = "none" if lo is None else f"{safe_label(lo)}_{safe_label(hi)}"
        if lo is None:
            updates = {"olr.afternoon.reject_score_min": 0.0, "olr.afternoon.reject_score_max": 0.0}
        else:
            updates = {"olr.afternoon.reject_score_min": lo, "olr.afternoon.reject_score_max": hi}
        add(f"score_notch_{suffix}", "score_notch_perturbation", updates, "Perturb the accepted score rejection notch.")

    for label, entry in entry_variants(round1_final, final, limit_grid=limit_grid):
        add(f"entry_{safe_label(label)}", "entry_perturbation", {"olr.trade_plan.entry": entry}, "Entry route perturbation.")
    for label, exit_plan in exit_variants(round1_final, final, limit_grid=limit_grid):
        add(f"exit_{safe_label(label)}", "exit_perturbation", {"olr.trade_plan.exit": exit_plan}, "Exit route perturbation.")

    sector_tests = [
        ("block_auto_ent_consumer", ["AUTOMOTIVE", "ENTERTAINMENT", "CONSUMER"], ()),
        ("block_def_consumer", ["DEFENSE", "CONSUMER"], ()),
        ("allow_semis_chem_elec", (), ["SEMICONDUCTORS", "CHEMICALS", "ELECTRONICS", "TELECOM", "HEAVY INDUSTRY"]),
    ]
    for label, blocked, allowed in sector_tests:
        add(
            f"sector_{label}",
            "selector_perturbation",
            {"olr.afternoon.blocked_sectors": list(blocked), "olr.afternoon.allowed_sectors": list(allowed)},
            "Sector filter perturbation from OOS weakness hypotheses.",
        )

    return out


def build_targeted_repair_candidates(final: dict[str, Any], edge: dict[str, Any], final_oos: dict[str, Any]) -> list[Candidate]:
    metrics = dict(final_oos.get("metrics") or {})
    win_rate = metric_win(metrics)
    net = metric_net(metrics)
    drawdown = metric_dd(metrics)
    trades = metric_trades(metrics)
    worst3_share = float(edge.get("worst_3_loss_share_of_all_losses") or 0.0)
    out: list[Candidate] = []

    def add(label: str, updates: dict[str, Any], reason: str) -> None:
        mutated = dict(final)
        mutated.update(deepcopy(updates))
        out.append(Candidate(f"targeted_{safe_label(label)}", "targeted_oos_repair", mutated, reason))

    if net < 0.0 or drawdown > 0.06 or worst3_share > 0.35:
        for gross in (0.70, 0.80, 0.90, 1.00):
            add(f"risk_gross_{gross}", {"olr.allocation.target_gross_exposure": gross}, "OOS loss/drawdown concentration: reduce gross exposure.")
        for cap in (0.35, 0.40, 0.45, 0.50):
            add(f"risk_cap_{cap}", {"olr.allocation.max_position_pct": cap}, "OOS loss/drawdown concentration: reduce single-name cap.")
        for label, exit_plan in exit_variants({}, final, limit_grid=False, risk_first=True):
            add(f"risk_exit_{label}", {"olr.trade_plan.exit": exit_plan}, "OOS loss/drawdown concentration: test tighter managed exits.")
    if win_rate < 0.45:
        add("quality_close_relvol", {"olr.afternoon.min_close_location": 0.55, "olr.afternoon.min_rel_volume": 0.75}, "Low OOS win rate: require stronger 14:30 close quality and volume.")
        add("quality_vwap_close", {"olr.afternoon.min_vwap_ret": 0.0, "olr.afternoon.min_close_location": 0.50}, "Low OOS win rate: require VWAP reclaim/hold.")
        add("quality_flow_confirmed", {"olr.afternoon.score_mode": "flow_confirmed", "olr.afternoon.min_flow_5d": 0.0, "olr.afternoon.min_sector_flow": 0.0}, "Low OOS win rate: shift ranking toward confirmed lagged flow.")
        add("quality_require_prev_close", {"olr.afternoon.require_close_above_prev": True}, "Low OOS win rate: reject names failing prior-close confirmation.")
    if trades < 25:
        add("frequency_slot6_g100", {"olr.overnight.slot_count": 6, "olr.allocation.target_gross_exposure": 1.0, "olr.allocation.max_position_pct": 0.40}, "Low OOS frequency: increase slots while lowering per-name cap.")
        add("frequency_no_notch_slot5", {"olr.afternoon.reject_score_min": 0.0, "olr.afternoon.reject_score_max": 0.0, "olr.overnight.slot_count": 5}, "Low OOS frequency: remove score notch but preserve slot expansion.")
    return out


def evaluate_candidates(
    config: dict[str, Any],
    candidates: Sequence[Candidate],
    window: str,
    output_dir: Path,
    eval_path: Path,
    progress_path: Path,
    cached: dict[tuple[str, str], dict[str, Any]],
    *,
    holdout_days: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    requested = list(candidates)
    rows: list[dict[str, Any]] = []
    missing: list[Candidate] = []
    for candidate in requested:
        key = (window, candidate.label)
        if key in cached:
            rows.append(cached[key])
        else:
            missing.append(candidate)
    if not missing:
        return rows

    for batch_index in range(0, len(missing), batch_size):
        batch = missing[batch_index : batch_index + batch_size]
        status(progress_path, f"{window}_batch_start", index=batch_index // batch_size + 1, total_batches=math.ceil(len(missing) / batch_size), candidates=len(batch))
        source_map, sources = build_sources(batch)
        compiled = build_compiled_execution_set(
            window_config(config, window, holdout_days),
            {"base_mutations": {}, "selected_stage1_seed": {"mutations": {}}},
            sources,
            holdout_days=holdout_days,
            use_fast_cache=True,
            include_holdout=(window == "oos"),
        )
        if window == "oos":
            dates = tuple(day for day in compiled.eligible_dates if day >= compiled.dataset.holdout_start)
        else:
            dates = tuple(day for day in compiled.eligible_dates if day < compiled.dataset.holdout_start)
        if not dates:
            raise RuntimeError(f"No {window} dates resolved for OLR evaluation")
        bundle_cache = {}
        for index, candidate in enumerate(batch, start=1):
            status(progress_path, f"{window}_candidate_start", label=candidate.label, kind=candidate.kind, index=index, batch_total=len(batch))
            source = source_map[stable_signature(selection_mutations(candidate.mutations))]
            bundle = bundle_cache.get(source.name)
            if bundle is None:
                bundle = _bundle_for_source(compiled, source, dates, candidate_only=False)
                bundle_cache[source.name] = bundle
            try:
                started = time.monotonic()
                result = run_olr_backtest({**window_config(config, window, holdout_days), "capability_level": "compiled"}, candidate.mutations, replay_bundle=bundle)
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
                    },
                    "trade_rows": trade_rows(result.trades),
                    "decision_summary": decision_summary(result.decisions),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            except Exception as exc:  # keep the long sweep alive and inspect failures later
                row = {
                    "label": candidate.label,
                    "kind": candidate.kind,
                    "reason": candidate.reason,
                    "window": window,
                    "mutations": candidate.mutations,
                    "metrics": {},
                    "source": {"error": repr(exc)},
                    "trade_rows": [],
                    "decision_summary": {},
                    "elapsed_seconds": 0.0,
                    "error": repr(exc),
                }
            append_jsonl(eval_path, row)
            cached[(window, candidate.label)] = row
            rows.append(row)
            status(
                progress_path,
                f"{window}_candidate_done",
                label=candidate.label,
                net_return_pct=metric_net(row["metrics"]),
                trades=metric_trades(row["metrics"]),
                win_rate=metric_win(row["metrics"]),
                drawdown=metric_dd(row["metrics"]),
                elapsed_seconds=row.get("elapsed_seconds", 0.0),
                error=row.get("error", ""),
            )
    by_label = {(row["window"], row["label"]): row for row in rows}
    return [by_label[(window, candidate.label)] for candidate in requested if (window, candidate.label) in by_label]


def build_sources(candidates: Sequence[Candidate]) -> tuple[dict[str, CandidateSource], list[CandidateSource]]:
    source_map: dict[str, CandidateSource] = {}
    for index, candidate in enumerate(candidates, start=1):
        selection = selection_mutations(candidate.mutations)
        signature = stable_signature(selection)
        if signature in source_map:
            continue
        source_map[signature] = CandidateSource(
            rank=len(source_map) + 1,
            name=f"src_{len(source_map) + 1:04d}_{safe_label(candidate.label)[:80]}",
            stage1_name=f"stage1_{len(source_map) + 1:04d}",
            stage2_name=f"stage2_{len(source_map) + 1:04d}",
            score=0.0,
            mutations=dict(candidate.mutations),
            stage1_mutations=stage1_mutations(candidate.mutations),
            stage2_mutations=selection,
        )
    return source_map, list(source_map.values())


def selection_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in sorted(mutations.items()) if key.startswith(SNAPSHOT_PREFIXES)}


def stage1_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in sorted(mutations.items()) if key.startswith(STAGE1_PREFIXES)}


def window_config(config: dict[str, Any], window: str, holdout_days: int) -> dict[str, Any]:
    out = dict(config)
    out["holdout_days"] = int(holdout_days)
    out["use_full_available_window"] = True
    if window == "train":
        out["use_full_available_window"] = False
    return out


def edge_case_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    rows = list(result.get("trade_rows") or [])
    losses = [row for row in rows if float(row.get("net_pnl", 0.0) or 0.0) < 0.0]
    losses.sort(key=lambda row: float(row.get("net_pnl", 0.0) or 0.0))
    total_loss = -sum(float(row.get("net_pnl", 0.0) or 0.0) for row in losses)
    worst_loss = -sum(float(row.get("net_pnl", 0.0) or 0.0) for row in losses[:3])
    net_profit = sum(float(row.get("net_pnl", 0.0) or 0.0) for row in rows)
    initial_equity = 10_000_000.0
    metrics = dict(result.get("metrics") or {})
    final_equity = float(metrics.get("final_equity", 0.0) or 0.0)
    if final_equity:
        initial_equity = max(final_equity - net_profit, 1.0)
    remove = []
    for k in (1, 2, 3, 5, 8):
        removed = sum(float(row.get("net_pnl", 0.0) or 0.0) for row in losses[:k])
        adjusted = net_profit - removed
        remove.append({"remove_worst_k": k, "adjusted_net_return_pct": adjusted / initial_equity, "remaining_trades": max(0, len(rows) - min(k, len(losses)))})
    return {
        "trade_count": len(rows),
        "loss_count": len(losses),
        "total_loss": total_loss,
        "worst_3_loss": worst_loss,
        "worst_3_loss_share_of_all_losses": worst_loss / total_loss if total_loss > 0 else 0.0,
        "remove_worst_impacts": remove,
        "worst_trades": losses[:12],
        "losses_by_symbol": group_rows(losses, "symbol", top_n=12),
        "losses_by_entry_date": group_rows(losses, "entry_date", top_n=12),
        "losses_by_exit_reason": group_rows(losses, "exit_reason", top_n=12),
        "all_trades_by_symbol": group_rows(rows, "symbol", top_n=12),
    }


def build_ablation_impacts(oos_by_label: dict[str, dict[str, Any]], train_by_label: dict[str, dict[str, Any]], target_label: str) -> dict[str, Any]:
    target_oos = dict(oos_by_label[target_label]["metrics"])
    target_train = dict(train_by_label.get(target_label, {}).get("metrics") or {})
    out = {}
    for label, row in sorted(oos_by_label.items()):
        kind = row.get("kind", "")
        if "ablation" not in kind and kind not in {"accepted_step_cumulative", "round_final"}:
            continue
        train = train_by_label.get(label)
        out[label] = {
            "kind": kind,
            "reason": row.get("reason", ""),
            "oos": metric_delta(row.get("metrics", {}), target_oos),
            "train": metric_delta((train or {}).get("metrics", {}), target_train) if train else {},
        }
    return out


def metric_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "net_delta_pct_points": 100.0 * (metric_net(metrics) - metric_net(baseline)),
        "trade_delta": metric_trades(metrics) - metric_trades(baseline),
        "win_delta_pct_points": 100.0 * (metric_win(metrics) - metric_win(baseline)),
        "drawdown_delta_pct_points": 100.0 * (metric_dd(metrics) - metric_dd(baseline)),
    }


def oos_score(metrics: dict[str, Any]) -> float:
    net = metric_net(metrics)
    trades = metric_trades(metrics)
    win = metric_win(metrics)
    dd = metric_dd(metrics)
    avg_trade = float(metrics.get("avg_trade_net_pct", 0.0) or 0.0)
    return 1000.0 * net + 80.0 * math.tanh((trades - 20.0) / 20.0) + 45.0 * (win - 0.45) + 90.0 * math.tanh(avg_trade / 0.006) - 600.0 * dd


def combined_score(oos: dict[str, Any], train: dict[str, Any], final_train: dict[str, Any], final_oos: dict[str, Any]) -> float:
    score = oos_score(oos)
    train_net = metric_net(train)
    final_train_net = max(metric_net(final_train), 0.01)
    train_ratio = train_net / final_train_net
    if train_ratio < 0.75:
        score -= 260.0 * (0.75 - train_ratio)
    if train_ratio > 1.0:
        score += 20.0 * min(train_ratio - 1.0, 0.50)
    final_oos_trades = max(metric_trades(final_oos), 1.0)
    trade_ratio = metric_trades(oos) / final_oos_trades
    if trade_ratio < 0.60:
        score -= 80.0 * (0.60 - trade_ratio)
    return score


def render_markdown(payload: dict[str, Any]) -> str:
    target_oos = payload["target_baseline"]["oos"]["metrics"]
    target_train = payload["target_baseline"]["train"]["metrics"]
    edge = payload["edge_case_diagnostics"]
    lines = [
        "# OLR Round 2 OOS Deep Ablation",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- OOS evaluated: {payload['counts']['oos_evaluated']}",
        f"- Train-confirmed: {payload['counts']['train_confirmed']}",
        f"- Target train: {pct(metric_net(target_train))} net, {metric_trades(target_train):.0f} trades, {pct(metric_win(target_train))} win, {pct(metric_dd(target_train))} DD.",
        f"- Target OOS: {pct(metric_net(target_oos))} net, {metric_trades(target_oos):.0f} trades, {pct(metric_win(target_oos))} win, {pct(metric_dd(target_oos))} DD.",
        "",
        "## Edge-Case Check",
        f"- Worst 3 OOS losses explain {pct(edge.get('worst_3_loss_share_of_all_losses', 0.0))} of all losing PnL.",
    ]
    for row in edge.get("remove_worst_impacts", [])[:5]:
        lines.append(f"- Remove worst {row['remove_worst_k']}: adjusted OOS net {pct(row['adjusted_net_return_pct'])} over {row['remaining_trades']} trades.")
    lines.extend(["", "## Top Train-Confirmed Candidates"])
    lines.extend(
        markdown_table(
            [
                {
                    "label": row["label"],
                    "kind": row["kind"],
                    "oos_net": 100.0 * metric_net(row["oos"]["metrics"]),
                    "oos_trades": metric_trades(row["oos"]["metrics"]),
                    "oos_win": 100.0 * metric_win(row["oos"]["metrics"]),
                    "train_net": 100.0 * metric_net(row["train"]["metrics"]),
                    "train_trades": metric_trades(row["train"]["metrics"]),
                    "score": row["combined_score"],
                }
                for row in payload.get("confirmed_train_ranked", [])[:30]
            ],
            [
                ("Label", "label"),
                ("Kind", "kind"),
                ("OOS Net %", "oos_net"),
                ("OOS Trades", "oos_trades"),
                ("OOS Win %", "oos_win"),
                ("Train Net %", "train_net"),
                ("Train Trades", "train_trades"),
                ("Score", "score"),
            ],
        )
    )
    lines.extend(["", "## Best Ablation Uplifts"])
    impacts = []
    for label, row in payload.get("ablation_impacts", {}).items():
        impacts.append(
            {
                "label": label,
                "kind": row.get("kind", ""),
                "oos_net_delta": row.get("oos", {}).get("net_delta_pct_points", 0.0),
                "oos_trade_delta": row.get("oos", {}).get("trade_delta", 0.0),
                "train_net_delta": row.get("train", {}).get("net_delta_pct_points", 0.0),
            }
        )
    impacts.sort(key=lambda item: item["oos_net_delta"], reverse=True)
    lines.extend(markdown_table(impacts[:30], [("Label", "label"), ("Kind", "kind"), ("OOS Net Delta pp", "oos_net_delta"), ("OOS Trade Delta", "oos_trade_delta"), ("Train Net Delta pp", "train_net_delta")]))
    return "\n".join(lines) + "\n"


def entry_variants(round1_final: dict[str, Any], final: dict[str, Any], *, limit_grid: bool) -> list[tuple[str, dict[str, Any]]]:
    variants = [
        ("round2_final", dict(final.get("olr.trade_plan.entry") or {})),
        ("round1_original", dict(round1_final.get("olr.trade_plan.entry") or {})),
        ("close_auction", {"name": "close_auction", "mode": "close_auction"}),
        ("confirm_b2_ret0_vwm002_cl50", {"name": "confirm_b2_ret0_vwm002_cl50", "mode": "confirm_next_bar", "max_signal_bars": 2, "min_bar_ret": 0.0, "min_vwap_ret": -0.002, "min_close_location": 0.50, "max_vwap_extension_pct": 9.99}),
        ("confirm_b4_ret0_vw0_cl50", {"name": "confirm_b4_ret0_vw0_cl50", "mode": "confirm_next_bar", "max_signal_bars": 4, "min_bar_ret": 0.0, "min_vwap_ret": 0.0, "min_close_location": 0.50, "max_vwap_extension_pct": 0.025}),
        ("confirm_b6_vw0_cl60_cap20", {"name": "confirm_b6_vw0_cl60_cap20", "mode": "confirm_next_bar", "max_signal_bars": 6, "min_bar_ret": 0.0, "min_vwap_ret": 0.0, "min_close_location": 0.60, "max_vwap_extension_pct": 0.020}),
        ("late_cont_b4_bo05", {"name": "late_cont_b4_bo05", "mode": "late_continuation", "after_bar": 1, "max_signal_bars": 4, "min_breakout_pct": 0.0005, "min_vwap_ret": 0.0, "min_close_location": 0.50}),
        ("decision_high_b4_bo10", {"name": "decision_high_b4_bo10", "mode": "decision_high_breakout", "max_signal_bars": 4, "min_breakout_pct": 0.001, "min_close_location": 0.60}),
        ("vwap_reclaim_b6_pb4", {"name": "vwap_reclaim_b6_pb4", "mode": "vwap_reclaim", "max_signal_bars": 6, "max_pullback_from_vwap_pct": 0.004, "min_reclaim_ret": 0.0, "min_vwap_ret": 0.0, "min_close_location": 0.50}),
    ]
    return dedupe_named_dicts(variants[:5] if limit_grid else variants)


def exit_variants(round1_final: dict[str, Any], final: dict[str, Any], *, limit_grid: bool = False, risk_first: bool = False) -> list[tuple[str, dict[str, Any]]]:
    variants = [
        ("round2_final", dict(final.get("olr.trade_plan.exit") or {})),
        ("round1_next_close", dict(round1_final.get("olr.trade_plan.exit") or {})),
        ("next_close", {"name": "next_close", "mode": "next_close", "hard_stop_enabled": False}),
        ("fade1_g075", {"name": "fade1_g075", "mode": "managed", "hard_stop_enabled": False, "mfe_fade_start_r": 1.0, "mfe_fade_gap_r": 0.75, "mfe_fade_floor_r": 0.0}),
        ("fade125_g100", {"name": "fade125_g100", "mode": "managed", "hard_stop_enabled": False, "mfe_fade_start_r": 1.25, "mfe_fade_gap_r": 1.0, "mfe_fade_floor_r": 0.0}),
        ("fade2_g125", {"name": "fade2_g125", "mode": "managed", "hard_stop_enabled": False, "mfe_fade_start_r": 2.0, "mfe_fade_gap_r": 1.25, "mfe_fade_floor_r": 0.0}),
        ("target150", {"name": "target150", "mode": "managed", "hard_stop_enabled": False, "target_r": 1.50}),
        ("target250", {"name": "target250", "mode": "managed", "hard_stop_enabled": False, "target_r": 2.50}),
        ("hard_decision_low_target150", {"name": "hard_decision_low_target150", "mode": "managed", "stop_mode": "decision_low", "hard_stop_enabled": True, "target_r": 1.50}),
        ("partial050_be_target150", {"name": "partial050_be_target150", "mode": "managed", "stop_mode": "decision_low", "hard_stop_enabled": True, "partial_trigger_r": 0.50, "partial_fraction": 0.50, "partial_stop_r": 0.0, "target_r": 1.50}),
        ("vwap_fail2_target100", {"name": "vwap_fail2_target100", "mode": "managed", "stop_mode": "vwap", "hard_stop_enabled": True, "vwap_fail_bars": 2, "vwap_fail_pct": 0.001, "target_r": 1.0}),
        ("max_hold8", {"name": "max_hold8", "mode": "managed", "hard_stop_enabled": False, "max_hold_bars": 8}),
    ]
    if risk_first:
        variants = variants[3:]
    if limit_grid:
        variants = variants[:6]
    return dedupe_named_dicts(variants)


def dedupe_named_dicts(items: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    seen = set()
    for label, value in items:
        if not value:
            continue
        key = stable_signature(value)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, value))
    return out


def allocation_to_mutations(allocation: dict[str, Any]) -> dict[str, Any]:
    if not allocation:
        return {}
    return {
        "olr.allocation.mode": allocation.get("mode", "rank_weighted"),
        "olr.allocation.target_gross_exposure": allocation.get("target_gross_exposure", 1.0),
        "olr.allocation.max_position_pct": allocation.get("max_position_pct", 0.5),
        "olr.allocation.min_selected": allocation.get("min_selected", 1),
        "olr.allocation.rank_decay": allocation.get("rank_decay", 1.0),
    }


def trade_rows(trades: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for trade in trades:
        payload = trade.to_json_dict() if hasattr(trade, "to_json_dict") else {}
        route = dict(payload.get("route_metadata") or {})
        cohort = dict(payload.get("cohort_metadata") or {})
        entry_price = float(payload.get("entry_price") or 0.0)
        qty = max(int(payload.get("qty") or 1), 1)
        notional = max(entry_price * qty, 1.0)
        rows.append(
            {
                "symbol": str(payload.get("symbol") or ""),
                "qty": qty,
                "entry_date": str(payload.get("entry_fill_time") or "")[:10],
                "entry_fill_time": payload.get("entry_fill_time"),
                "exit_fill_time": payload.get("exit_fill_time"),
                "entry_price": entry_price,
                "exit_price": payload.get("exit_price"),
                "gross_pnl": payload.get("gross_pnl", 0.0),
                "net_pnl": payload.get("net_pnl", 0.0),
                "net_return_pct": float(payload.get("net_pnl", 0.0) or 0.0) / notional,
                "r": payload.get("r_multiple", 0.0),
                "mfe": payload.get("mfe", 0.0),
                "mae": payload.get("mae", 0.0),
                "exit_reason": payload.get("exit_reason", ""),
                "candidate_rank": route.get("candidate_rank") or cohort.get("candidate_rank"),
                "candidate_score": route.get("candidate_score") or cohort.get("candidate_score"),
                "candidate_sector": route.get("sector") or route.get("candidate_sector") or cohort.get("sector") or cohort.get("candidate_sector"),
                "route_metadata": route,
                "cohort_metadata": cohort,
            }
        )
    return rows


def decision_summary(decisions: Iterable[Any]) -> dict[str, Any]:
    codes = Counter(str(getattr(item, "decision_code", "") or "") for item in decisions)
    reasons = Counter(str(getattr(item, "reason", "") or "") for item in decisions if getattr(item, "reason", ""))
    return {"decision_code_counts": codes.most_common(20), "reason_counts": reasons.most_common(20)}


def group_rows(rows: Sequence[dict[str, Any]], key: str, *, top_n: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "UNKNOWN")].append(row)
    out = []
    for name, group in groups.items():
        out.append(
            {
                key: name,
                "trades": len(group),
                "net_pnl": sum(float(item.get("net_pnl", 0.0) or 0.0) for item in group),
                "win_rate": sum(1 for item in group if float(item.get("net_pnl", 0.0) or 0.0) > 0.0) / max(len(group), 1),
                "avg_r": statistics.fmean(float(item.get("r", 0.0) or 0.0) for item in group) if group else 0.0,
            }
        )
    out.sort(key=lambda item: (item["net_pnl"], -item["trades"]))
    return out[:top_n]


def compact_eval(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    keys = (
        "official_mtm_net_return_pct",
        "net_return_pct",
        "total_trades",
        "trade_count",
        "entry_fill_count",
        "win_rate",
        "net_win_share",
        "profit_factor",
        "max_drawdown_pct",
        "official_mtm_max_drawdown_pct",
        "avg_trade_net_pct",
        "entry_level_expected_total_r",
        "expected_total_r",
        "mfe_capture",
        "olr_alpha_capture",
        "same_bar_fill_count",
        "forced_replay_close_count",
        "rejected_order_count",
        "end_open_position_count",
    )
    metrics = {key: row.get("metrics", {}).get(key) for key in keys if key in row.get("metrics", {})}
    return {
        "label": row.get("label", ""),
        "kind": row.get("kind", ""),
        "reason": row.get("reason", ""),
        "metrics": metrics,
        "source": row.get("source", {}),
        "mutations": row.get("mutations", {}),
        "elapsed_seconds": row.get("elapsed_seconds", 0.0),
        "error": row.get("error", ""),
    }


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


def candidate_payload(candidate: Candidate, *, include_mutations: bool) -> dict[str, Any]:
    payload = {"label": candidate.label, "kind": candidate.kind, "reason": candidate.reason}
    if include_mutations:
        payload["mutations"] = candidate.mutations
    return payload


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = stable_signature(candidate.mutations)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def load_cached_evaluations(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(str(row.get("window")), str(row.get("label")))] = row
    return out


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def status(path: Path, stage: str, **extra: Any) -> None:
    payload = {"ts": utc_now(), "stage": stage, **extra}
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)
    append_jsonl(path, payload)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    out = ["| " + " | ".join(label for label, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def safe_label(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    compact = "_".join(part for part in "".join(out).split("_") if part)
    return compact[:120] or "value"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
