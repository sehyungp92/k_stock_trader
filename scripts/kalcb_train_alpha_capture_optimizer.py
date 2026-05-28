from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SHARED_PATH = REPO_ROOT / "scripts" / "kalcb_r_capture_shared_core_replay.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "train_alpha_capture_optimizer"
SEED_PATH = (
    ROUND_DIR
    / "local_minimum_recovery"
    / "07_alpha_conversion_next_round"
    / "next_round_seed_auto_pullback_q85_rank8_r0p015.json"
)

from backtests.strategies.kalcb.candidate_surfacing_recovery import PoolVariant, evaluate_compiled_candidate_pool  # noqa: E402
from backtests.strategies.kalcb.fixed_trade_plan_phase import _configured_entry_routes, _route_candidate_passes  # noqa: E402
from backtests.strategies.kalcb.shadow_ledger_reranker import write_jsonl  # noqa: E402


def _load_shared_module():
    spec = importlib.util.spec_from_file_location("kalcb_r_capture_shared_core_replay_module", SHARED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared replay module: {SHARED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared_module()
opt = shared.opt


@dataclass(frozen=True)
class SelectionPolicy:
    name: str
    label: str
    scope: str
    budget: str
    active_count: int
    pool_size: int
    source: str


@dataclass(frozen=True)
class MutationVariant:
    name: str
    first30_risk_mult: float | None = None
    min_bar_ret: float | None = None
    min_vwap_ret: float | None = None
    quality_min_bar_ret: float | None = None
    relax_delayed_relvol_to: float | None = None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def log(event: str, **extra: Any) -> None:
    payload = {"ts": now_iso(), "event": event, **extra}
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)
    append_jsonl(OUT_DIR / "progress.jsonl", payload)


def num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def pct(value: Any) -> str:
    return f"{100.0 * num(value):.2f}%"


def load_seed_payload() -> dict[str, Any]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


def load_seed_mutations() -> dict[str, Any]:
    mutations = load_seed_payload().get("mutations")
    if not isinstance(mutations, dict):
        raise ValueError(f"Missing mutations in {SEED_PATH}")
    return copy.deepcopy(mutations)


def fit_train_only_models(train: Any) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    models: dict[str, dict[str, Any]] = {}
    train_scores: dict[str, np.ndarray] = {}
    for spec in opt.model_specs():
        features = [col for col in opt.FEATURE_GROUPS[spec.feature_group] if col in train.columns]
        train_mask = opt.scope_mask(train, spec.train_scope)
        x_train = train.loc[train_mask, features]
        y_train = opt.target_values(train.loc[train_mask], spec.target)
        model = opt.make_model(spec.model_kind)
        log("model_fit_start", name=spec.name, rows=len(x_train), features=len(features), target=spec.target)
        model.fit(x_train, y_train)
        models[spec.name] = {"spec": spec, "features": features, "model": model}
        train_scores[spec.name] = np.asarray(model.predict(train[features]), dtype=float)
        log("model_fit_done", name=spec.name)
    return models, train_scores


def policy_from(label: str, scope: str, budget: str, source: str = "train_proxy_shortlist") -> SelectionPolicy:
    k = int(budget.replace("top", "")) if budget.startswith("top") else 8
    clean = f"{label}_{scope}_{budget}".replace("__", "_")
    return SelectionPolicy(clean, label, scope, budget, k, k, source)


def summarize_proxy_selection(train: Any, scores: dict[str, np.ndarray]) -> tuple[list[SelectionPolicy], list[dict[str, Any]]]:
    labels = [
        "surfaced_all_context_hgb_quality",
        "surfaced_all_context_trees_quality",
        "surfaced_all_context_ridge_quality",
        "dataset_all_context_hgb_mfe",
        "dataset_all_context_hgb_quality",
        "dataset_all_context_trees_quality",
        "dataset_all_context_ridge_quality",
        "surfaced_first30_only_hgb_mfe",
        "dataset_first30_only_hgb_mfe",
        "dataset_first30_sector_hgb_mfe",
    ]
    budgets = ["top4", "top8", "top12", "top16"]
    rows: list[dict[str, Any]] = []
    current_r = float(train.loc[train["selected_first30"].eq(1.0), "positive_r"].sum())
    dataset_r = float(train["positive_r"].sum())
    surfaced_r = float(train.loc[train["surfaced_candidate"].eq(1.0), "positive_r"].sum())
    for label in labels:
        if label not in scores:
            continue
        for scope in ("dataset", "surfaced"):
            for budget in budgets:
                selected = opt.select_by_budget(train, scores[label], scope, budget, {})
                if selected.empty:
                    continue
                summary = opt.summarize_selection(
                    train,
                    selected,
                    window="train",
                    scope=scope,
                    budget=budget,
                    label=label,
                    current_selected_r=current_r,
                    dataset_available_r=dataset_r,
                    surfaced_available_r=surfaced_r,
                )
                proxy_r = num(summary.get("selected_positive_r"))
                proxy_net = num(summary.get("selected_net_eod_r"))
                rows_count = num(summary.get("selected_rows"))
                hit_share = num(summary.get("positive_r_hit_share"))
                # Train-only shortlist score: expected opportunity plus enough frequency, with a penalty for bad proxy EOD.
                summary["train_proxy_alpha_score"] = (
                    proxy_r
                    + 0.35 * max(proxy_net, 0.0)
                    - 0.20 * max(-proxy_net, 0.0)
                    + 0.65 * rows_count
                    + 100.0 * hit_share
                )
                summary["policy_name"] = policy_from(label, scope, budget).name
                rows.append(summary)
    ranked = sorted(rows, key=lambda row: num(row.get("train_proxy_alpha_score")), reverse=True)
    policies: list[SelectionPolicy] = []
    seen: set[str] = set()
    forced = [
        policy_from("surfaced_all_context_hgb_quality", "surfaced", "top8", "forced_stage2_control"),
        policy_from("surfaced_all_context_ridge_quality", "dataset", "top8", "forced_all_context_challenger"),
        policy_from("dataset_all_context_trees_quality", "dataset", "top4", "forced_balanced_probe"),
        policy_from("dataset_all_context_trees_quality", "dataset", "top8", "forced_dataset_all_context"),
    ]
    for policy in forced:
        if policy.name not in seen:
            policies.append(policy)
            seen.add(policy.name)
    for row in ranked:
        policy = policy_from(str(row["label"]), str(row["scope"]), str(row["budget"]))
        if policy.name in seen:
            continue
        policies.append(policy)
        seen.add(policy.name)
        if len(policies) >= 12:
            break
    return policies, ranked


def selected_pool_rows(part: Any, score: np.ndarray, policy: SelectionPolicy, feature_by_key: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    selected = opt.select_by_budget(part, score, policy.scope, policy.budget, {})
    candidate = shared.ReplayCandidate(policy.name, policy.label, policy.scope, policy.budget, policy.active_count, policy.pool_size, "train_only", policy.source)
    return shared.selected_pool_rows(selected=selected, feature_by_key=feature_by_key, candidate=candidate)


def apply_mutation(seed: dict[str, Any], variant: MutationVariant) -> dict[str, Any]:
    out = copy.deepcopy(seed)
    routes = [dict(route) for route in out.get("kalcb.entry.routes") or [] if isinstance(route, dict)]
    for route in routes:
        mode = str(route.get("mode") or "")
        if variant.min_bar_ret is not None:
            route["min_bar_ret"] = float(variant.min_bar_ret)
        if variant.min_vwap_ret is not None:
            route["min_vwap_ret"] = float(variant.min_vwap_ret)
        if variant.quality_min_bar_ret is not None:
            route["quality_min_bar_ret"] = float(variant.quality_min_bar_ret)
        if mode != "first30_open" and variant.relax_delayed_relvol_to is not None:
            context_min = dict(route.get("context_min") or {})
            context_min["first30_rel_volume"] = float(variant.relax_delayed_relvol_to)
            route["context_min"] = context_min
        if mode == "first30_open" and variant.first30_risk_mult is not None:
            route["risk_mult"] = float(variant.first30_risk_mult)
            route["notional_mult"] = float(variant.first30_risk_mult)
    out["kalcb.entry.routes"] = routes
    if variant.quality_min_bar_ret is not None:
        out["kalcb.entry.quality_min_bar_ret"] = float(variant.quality_min_bar_ret)
    return out


def mutation_variants() -> list[MutationVariant]:
    return [
        MutationVariant("seed_risk99"),
        MutationVariant("seed_risk80", first30_risk_mult=0.80),
        MutationVariant("seed_risk65", first30_risk_mult=0.65),
        MutationVariant("seed_risk50", first30_risk_mult=0.50),
        MutationVariant("seed_risk40", first30_risk_mult=0.40),
        MutationVariant("seed_relaxed_risk50", first30_risk_mult=0.50, min_bar_ret=-0.03, min_vwap_ret=-0.02, quality_min_bar_ret=-9.99, relax_delayed_relvol_to=3.0),
    ]


def static_route_eligibility(pool_rows: list[dict[str, Any]], mutations: dict[str, Any]) -> dict[str, Any]:
    routes = _configured_entry_routes(mutations)
    eligible = 0
    mode_counts: dict[str, int] = {}
    blockers: dict[str, int] = {}
    for row in pool_rows:
        passed_any = False
        first_reason = ""
        for route in routes:
            passed, reason = _route_candidate_passes(route, mutations, row)
            mode = str(route.get("mode") or route.get("name") or "route")
            if passed:
                passed_any = True
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            elif not first_reason:
                first_reason = reason
        if passed_any:
            eligible += 1
        else:
            blockers[first_reason or "not_eligible"] = blockers.get(first_reason or "not_eligible", 0) + 1
    return {
        "static_route_eligible_count": eligible,
        "static_route_eligible_share": eligible / max(len(pool_rows), 1),
        "static_route_eligible_by_mode": mode_counts,
        "top_static_blockers": sorted(blockers.items(), key=lambda item: item[1], reverse=True)[:8],
    }


def train_alpha_score(metrics: dict[str, Any], proxy_positive_r: float) -> dict[str, Any]:
    net = num(metrics.get("broker_net_return_pct"))
    dd = num(metrics.get("broker_max_drawdown_pct"))
    trades = num(metrics.get("trade_count"))
    capture = num(metrics.get("avg_mfe_capture"))
    worst_fold = num(metrics.get("worst_fold_net"))
    same_bar = num(metrics.get("same_bar_fill_count"))
    open_positions = num(metrics.get("end_open_position_count"))
    dd_excess = max(dd - 0.08, 0.0)
    frequency_component = 14.0 * min(trades / 150.0, 1.5)
    score = (
        100.0 * net
        + frequency_component
        + 9.0 * max(capture, 0.0)
        + 16.0 * max(worst_fold, 0.0)
        + 0.0005 * max(proxy_positive_r, 0.0)
        - 260.0 * dd_excess
        - 20.0 * same_bar
        - 20.0 * open_positions
    )
    pass_hygiene = (
        net > 0.0
        and dd <= 0.08
        and trades >= 80.0
        and worst_fold > 0.0
        and same_bar == 0.0
        and open_positions == 0.0
        and capture >= 0.20
    )
    return {
        "train_alpha_score": score,
        "train_pass_hygiene": pass_hygiene,
        "train_frequency_component": frequency_component,
        "train_dd_excess": dd_excess,
    }


def seed_benchmark() -> dict[str, Any]:
    payload = load_seed_payload()
    source = dict(payload.get("source_row") or {})
    train = dict(source.get("train") or {})
    score = train_alpha_score(train, proxy_positive_r=0.0)
    return {
        "name": str(payload.get("name") or "seed"),
        "role": "existing_seed_benchmark_not_part_of_new_pool_search",
        "train": train,
        **score,
    }


def run_train_optimizer() -> dict[str, Any]:
    seed = load_seed_mutations()
    train_config, holdout_config = shared.load_base_config()
    df = opt.read_pipeline()
    train = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    models, train_scores = fit_train_only_models(train)
    policies, proxy_ranked = summarize_proxy_selection(train, train_scores)
    log("policy_shortlist_done", policies=len(policies), top_proxy_policy=(proxy_ranked[0] or {}).get("policy_name") if proxy_ranked else "")
    feature_rows_train = shared.load_feature_rows("train")
    feature_rows_holdout = shared.load_feature_rows("holdout")

    train_pools: dict[str, list[dict[str, Any]]] = {}
    for policy in policies:
        rows = selected_pool_rows(train, train_scores[policy.label], policy, feature_rows_train)
        train_pools[policy.name] = rows
        write_jsonl(OUT_DIR / f"pool_rows_train_{policy.name}.jsonl", rows)
        log(
            "train_pool_prepared",
            policy=policy.name,
            rows=len(rows),
            proxy_positive_r=sum(num(row.get("optimizer_positive_r_proxy")) for row in rows),
            proxy_net_eod_r=sum(num(row.get("optimizer_net_eod_r_proxy")) for row in rows),
        )

    log("train_context_build_start")
    train_bundle = shared.build_window_bundle(train_config)
    log("train_context_build_done", sessions=len(train_bundle["dataset"].trading_dates), contexts=len(train_bundle["context_by_key"]))

    rows: list[dict[str, Any]] = []
    variants = mutation_variants()
    for policy in policies:
        pool_rows = train_pools[policy.name]
        proxy_positive_r = sum(num(row.get("optimizer_positive_r_proxy")) for row in pool_rows)
        proxy_net_eod_r = sum(num(row.get("optimizer_net_eod_r_proxy")) for row in pool_rows)
        for variant in variants:
            mutations = apply_mutation(seed, variant)
            eligibility = static_route_eligibility(pool_rows, mutations)
            if num(eligibility.get("static_route_eligible_count")) < 20.0:
                row = {
                    "policy": policy.__dict__,
                    "variant": variant.__dict__,
                    "window": "train",
                    "proxy_positive_r": proxy_positive_r,
                    "proxy_net_eod_r": proxy_net_eod_r,
                    "metrics": {"trade_count": 0.0},
                    "compiled_replay": eligibility,
                    "skipped": True,
                    "skip_reason": "static_route_eligible_count_lt_20",
                    **train_alpha_score({"trade_count": 0.0}, proxy_positive_r),
                }
                rows.append(row)
                log("train_replay_skipped", policy=policy.name, variant=variant.name, eligible=eligibility.get("static_route_eligible_count"))
                continue
            log("train_replay_start", policy=policy.name, variant=variant.name, eligible=eligibility.get("static_route_eligible_count"))
            result = evaluate_compiled_candidate_pool(
                window="train",
                variant=PoolVariant(f"{policy.name}_{variant.name}", policy.pool_size, active_count=policy.active_count),
                config=train_config,
                dataset=train_bundle["dataset"],
                context_by_key=train_bundle["context_by_key"],
                pool_rows=pool_rows,
                seed_mutations=mutations,
                output_dir=OUT_DIR,
                replay_name=f"{policy.name}_{variant.name}",
            )
            metrics = dict(result.get("metrics") or {})
            score = train_alpha_score(metrics, proxy_positive_r)
            row = {
                "policy": policy.__dict__,
                "variant": variant.__dict__,
                "window": "train",
                "proxy_positive_r": proxy_positive_r,
                "proxy_net_eod_r": proxy_net_eod_r,
                "metrics": metrics,
                "compiled_replay": result.get("compiled_replay") or {},
                "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
                "trade_rows_path": result.get("trade_rows_path"),
                "skipped": False,
                **score,
            }
            rows.append(row)
            log(
                "train_replay_done",
                policy=policy.name,
                variant=variant.name,
                score=score["train_alpha_score"],
                pass_hygiene=score["train_pass_hygiene"],
                trades=metrics.get("trade_count"),
                net=metrics.get("broker_net_return_pct"),
                dd=metrics.get("broker_max_drawdown_pct"),
                worst_fold=metrics.get("worst_fold_net"),
            )

    eligible_rows = [row for row in rows if not row.get("skipped")]
    pass_rows = [row for row in eligible_rows if row.get("train_pass_hygiene")]
    train_champion = max(pass_rows or eligible_rows, key=lambda row: num(row.get("train_alpha_score"))) if (pass_rows or eligible_rows) else {}
    raw_train_winner = max(eligible_rows, key=lambda row: num(row.get("train_alpha_score"))) if eligible_rows else {}
    locked_audits: list[dict[str, Any]] = []
    audit_targets: list[dict[str, Any]] = []
    for target in (train_champion, raw_train_winner):
        if target and all((target.get("policy") or {}).get("name") != (item.get("policy") or {}).get("name") or (target.get("variant") or {}).get("name") != (item.get("variant") or {}).get("name") for item in audit_targets):
            audit_targets.append(target)

    if audit_targets:
        log("holdout_context_build_start", audit_targets=len(audit_targets))
        holdout_bundle = shared.build_window_bundle(holdout_config)
        log("holdout_context_build_done", sessions=len(holdout_bundle["dataset"].trading_dates), contexts=len(holdout_bundle["context_by_key"]))
        for target in audit_targets:
            policy = SelectionPolicy(**dict(target["policy"]))
            variant = MutationVariant(**dict(target["variant"]))
            model_info = models[policy.label]
            holdout_score = np.asarray(model_info["model"].predict(holdout[model_info["features"]]), dtype=float)
            holdout_pool = selected_pool_rows(holdout, holdout_score, policy, feature_rows_holdout)
            write_jsonl(OUT_DIR / f"pool_rows_locked_holdout_{policy.name}_{variant.name}.jsonl", holdout_pool)
            mutations = apply_mutation(seed, variant)
            log("locked_holdout_replay_start", policy=policy.name, variant=variant.name)
            result = evaluate_compiled_candidate_pool(
                window="holdout_locked_audit",
                variant=PoolVariant(f"{policy.name}_{variant.name}_locked_holdout", policy.pool_size, active_count=policy.active_count),
                config=holdout_config,
                dataset=holdout_bundle["dataset"],
                context_by_key=holdout_bundle["context_by_key"],
                pool_rows=holdout_pool,
                seed_mutations=mutations,
                output_dir=OUT_DIR,
                replay_name=f"{policy.name}_{variant.name}_locked_holdout",
            )
            audit = {
                "policy": policy.__dict__,
                "variant": variant.__dict__,
                "selection_basis": "frozen_train_selected_policy_only_no_holdout_optimization",
                "proxy_positive_r": sum(num(row.get("optimizer_positive_r_proxy")) for row in holdout_pool),
                "proxy_net_eod_r": sum(num(row.get("optimizer_net_eod_r_proxy")) for row in holdout_pool),
                "metrics": dict(result.get("metrics") or {}),
                "compiled_replay": result.get("compiled_replay") or {},
                "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
                "trade_rows_path": result.get("trade_rows_path"),
            }
            locked_audits.append(audit)
            log(
                "locked_holdout_replay_done",
                policy=policy.name,
                variant=variant.name,
                trades=audit["metrics"].get("trade_count"),
                net=audit["metrics"].get("broker_net_return_pct"),
                dd=audit["metrics"].get("broker_max_drawdown_pct"),
            )

    return {
        "created_at_utc": now_iso(),
        "usage_contract": "train_only_alpha_capture_optimization_holdout_locked_audit_after_train_selection",
        "objective": {
            "train_alpha_score": "100*net + frequency bonus + capture + worst_fold + proxy_R - DD/hygiene penalties",
            "hygiene_pass": "net>0, DD<=8%, trades>=80, worst_fold>0, no same-bar/open positions, capture>=20%",
            "holdout_policy": "not used for shortlist, scoring, ranking, or champion selection",
        },
        "seed_benchmark": seed_benchmark(),
        "proxy_ranked_train": proxy_ranked[:30],
        "policies": [policy.__dict__ for policy in policies],
        "train_rows": rows,
        "train_champion": train_champion,
        "raw_train_winner": raw_train_winner,
        "locked_holdout_audits": locked_audits,
    }


def route_mix(row: dict[str, Any]) -> str:
    summary = dict(row.get("entry_route_mode_summary") or {})
    return "; ".join(f"{mode}:{int(num((data or {}).get('trades')))}" for mode, data in summary.items())


def render_report(summary: dict[str, Any]) -> str:
    train_rows = [row for row in summary.get("train_rows", []) if not row.get("skipped")]
    ranked = sorted(train_rows, key=lambda row: num(row.get("train_alpha_score")), reverse=True)
    pass_ranked = [row for row in ranked if row.get("train_pass_hygiene")]
    champion = dict(summary.get("train_champion") or {})
    seed = dict(summary.get("seed_benchmark") or {})
    seed_train = dict(seed.get("train") or {})
    lines = [
        "# KALCB Train-Only Alpha-Capture Optimizer",
        "",
        "This run optimizes only on the training set. Holdout appears only as a locked audit after the train champion is frozen.",
        "",
        "## Train Champion",
        "",
    ]
    if champion:
        m = dict(champion.get("metrics") or {})
        p = dict(champion.get("policy") or {})
        v = dict(champion.get("variant") or {})
        lines.extend(
            [
                f"- Policy: `{p.get('name')}`",
                f"- Variant: `{v.get('name')}`",
                f"- Train score: {num(champion.get('train_alpha_score')):.2f}; hygiene pass: {champion.get('train_pass_hygiene')}",
                f"- Train net/DD/trades: {pct(m.get('broker_net_return_pct'))} / {pct(m.get('broker_max_drawdown_pct'))} / {num(m.get('trade_count')):.0f}",
                f"- Train capture/worst fold/proxy R: {pct(m.get('avg_mfe_capture'))} / {pct(m.get('worst_fold_net'))} / {num(champion.get('proxy_positive_r')):.1f}R",
                f"- Route mix: {route_mix(champion)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Existing Seed Benchmark",
            "",
            f"- Seed train net/DD/trades: {pct(seed_train.get('broker_net_return_pct'))} / {pct(seed_train.get('broker_max_drawdown_pct'))} / {num(seed_train.get('trade_count')):.0f}",
            f"- Seed train capture: {pct(seed_train.get('avg_mfe_capture'))}; score under this objective: {num(seed.get('train_alpha_score')):.2f}",
            "",
            "## Top Train Replays",
            "",
            "| rank | pass | policy | variant | score | net | DD | trades | worst fold | capture | proxy R | routes |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(ranked[:18], start=1):
        p = dict(row.get("policy") or {})
        v = dict(row.get("variant") or {})
        m = dict(row.get("metrics") or {})
        lines.append(
            f"| {rank} | {row.get('train_pass_hygiene')} | `{p.get('name')}` | `{v.get('name')}` | "
            f"{num(row.get('train_alpha_score')):.2f} | {pct(m.get('broker_net_return_pct'))} | "
            f"{pct(m.get('broker_max_drawdown_pct'))} | {num(m.get('trade_count')):.0f} | "
            f"{pct(m.get('worst_fold_net'))} | {pct(m.get('avg_mfe_capture'))} | "
            f"{num(row.get('proxy_positive_r')):.1f} | {route_mix(row)} |"
        )
    lines.extend(["", "## Top Hygiene-Passing Train Replays", ""])
    if pass_ranked:
        for row in pass_ranked[:8]:
            p = dict(row.get("policy") or {})
            v = dict(row.get("variant") or {})
            m = dict(row.get("metrics") or {})
            lines.append(
                f"- `{p.get('name')}` + `{v.get('name')}`: score={num(row.get('train_alpha_score')):.2f}, "
                f"net={pct(m.get('broker_net_return_pct'))}, DD={pct(m.get('broker_max_drawdown_pct'))}, "
                f"trades={num(m.get('trade_count')):.0f}, worst_fold={pct(m.get('worst_fold_net'))}"
            )
    else:
        lines.append("- No replay passed the training hygiene contract.")
    lines.extend(["", "## Locked Holdout Audit", ""])
    audits = list(summary.get("locked_holdout_audits") or [])
    if audits:
        lines.append("| train-selected policy | variant | proxy R | trades | net | DD | capture | routes |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for row in audits:
            p = dict(row.get("policy") or {})
            v = dict(row.get("variant") or {})
            m = dict(row.get("metrics") or {})
            lines.append(
                f"| `{p.get('name')}` | `{v.get('name')}` | {num(row.get('proxy_positive_r')):.1f} | "
                f"{num(m.get('trade_count')):.0f} | {pct(m.get('broker_net_return_pct'))} | "
                f"{pct(m.get('broker_max_drawdown_pct'))} | {pct(m.get('avg_mfe_capture'))} | {route_mix(row)} |"
            )
        lines.append("")
        lines.append("Locked audit note: these rows were not used to choose the champion.")
    else:
        lines.append("- No holdout audit was run because no train replay completed.")
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- If the train champion does not beat the existing seed benchmark on train score, the next train-only work should optimize route/capture on the seed rather than promote a lower-return selector.",
            "- If a higher-frequency policy wins train but has weak worst-fold, treat frequency as real but unstable and split by route family before promotion.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("start")
    summary = run_train_optimizer()
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary_path = OUT_DIR / "kalcb_train_alpha_capture_optimizer_results.json"
    report_path = OUT_DIR / "kalcb_train_alpha_capture_optimizer_report.md"
    write_json(summary_path, summary)
    report_path.write_text(render_report(summary), encoding="utf-8")
    log("complete", summary=str(summary_path), report=str(report_path), elapsed_seconds=summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
