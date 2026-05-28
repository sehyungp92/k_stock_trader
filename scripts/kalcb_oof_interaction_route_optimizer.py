from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_SCRIPT = REPO_ROOT / "scripts" / "kalcb_interaction_route_conversion_optimizer.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "oof_interaction_route_optimizer"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("kalcb_interaction_route_conversion_optimizer_module", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load interaction optimizer module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT_DIR = OUT_DIR
    return module


base = _load_base_module()
shared = base.shared
opt = base.opt
train_opt = base.train_opt


@dataclass(frozen=True)
class OOFRankerSpec:
    name: str
    feature_set: str
    model_kind: str
    target: str
    scope: str
    budget: str
    active_count: int
    pool_size: int


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def log(event: str, **extra: Any) -> None:
    payload = {"ts": now_iso(), "event": event, **extra}
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)
    append_jsonl(OUT_DIR / "progress.jsonl", payload)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def pct(value: Any) -> str:
    return f"{100.0 * num(value):.2f}%"


def oof_ranker_specs() -> list[OOFRankerSpec]:
    return [
        OOFRankerSpec("oof_base_hgb_quality_top16", "base_all_context", "hgb", "alpha_quality", "dataset", "top16", 16, 16),
        OOFRankerSpec("oof_interaction_hgb_quality_top16", "interaction_full", "hgb", "alpha_quality", "dataset", "top16", 16, 16),
        OOFRankerSpec("oof_interaction_hgb_routeable_top16", "interaction_route", "hgb", "routeable_alpha", "dataset", "top16", 16, 16),
        OOFRankerSpec("oof_interaction_trees_quality_top16", "interaction_full", "trees", "alpha_quality", "dataset", "top16", 16, 16),
        OOFRankerSpec("oof_interaction_trees_quality_top24a16", "interaction_full", "trees", "alpha_quality", "dataset", "top24", 16, 24),
    ]


def chronological_folds(train: pd.DataFrame, folds: int = 5) -> list[np.ndarray]:
    dates = np.array(sorted(str(day)[:10] for day in train["trade_date"].dropna().unique()))
    out: list[np.ndarray] = []
    for fold in range(folds):
        lo = int(fold * len(dates) / folds)
        hi = int((fold + 1) * len(dates) / folds)
        fold_dates = set(dates[lo:hi])
        out.append(train["trade_date"].astype(str).str[:10].isin(fold_dates).to_numpy())
    return out


def fit_oof_rankers(train: pd.DataFrame, holdout: pd.DataFrame, folds: int = 5) -> dict[str, dict[str, Any]]:
    fitted: dict[str, dict[str, Any]] = {}
    fold_masks = chronological_folds(train, folds)
    for spec in oof_ranker_specs():
        features = base.feature_columns(train, spec.feature_set)
        scope_mask = opt.scope_mask(train, spec.scope)
        oof = np.full(len(train), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        for fold_index, valid_mask in enumerate(fold_masks):
            train_mask = scope_mask & ~valid_mask
            valid_scope = scope_mask & valid_mask
            model = base.make_model(spec.model_kind)
            log(
                "oof_fold_fit_start",
                name=spec.name,
                fold=fold_index,
                train_rows=int(train_mask.sum()),
                valid_rows=int(valid_scope.sum()),
                features=len(features),
            )
            model.fit(train.loc[train_mask, features], base.target_values(train.loc[train_mask], spec.target))
            preds = np.asarray(model.predict(train.loc[valid_mask, features]), dtype=float)
            oof[valid_mask] = preds
            valid_part = train.loc[valid_mask].copy().reset_index(drop=True)
            valid_part["_score"] = preds
            scoped_valid = valid_part.loc[opt.scope_mask(valid_part, spec.scope)].copy()
            selected_parts: list[pd.DataFrame] = []
            fixed_k = int(spec.budget.replace("top", "")) if spec.budget.startswith("top") else spec.pool_size
            for _, group in scoped_valid.groupby("trade_date", sort=True):
                selected_parts.append(group.sort_values("_score", ascending=False).head(fixed_k))
            selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else scoped_valid.head(0)
            fold_rows.append(
                {
                    "fold": fold_index,
                    "valid_rows": int(valid_mask.sum()),
                    "valid_scope_rows": int(valid_scope.sum()),
                    "selected_rows": int(len(selected)),
                    "selected_positive_r": float(selected["positive_r"].sum()) if len(selected) else 0.0,
                    "selected_net_eod_r": float(selected["net_eod_r_proxy"].sum()) if len(selected) else 0.0,
                }
            )
            log("oof_fold_fit_done", name=spec.name, fold=fold_index, selected_r=fold_rows[-1]["selected_positive_r"])
        if np.isnan(oof).any():
            fallback = np.nanmedian(oof)
            oof = np.nan_to_num(oof, nan=float(fallback if math.isfinite(fallback) else 0.0))
        full_model = base.make_model(spec.model_kind)
        full_model.fit(train.loc[scope_mask, features], base.target_values(train.loc[scope_mask], spec.target))
        fitted[spec.name] = {
            "spec": spec,
            "features": features,
            "model": full_model,
            "train_score": oof,
            "holdout_score": np.asarray(full_model.predict(holdout[features]), dtype=float),
            "fold_proxy_rows": fold_rows,
        }
        log("oof_ranker_done", name=spec.name, fold_selected_r=sum(num(row["selected_positive_r"]) for row in fold_rows))
    return fitted


def pool_rows_for_score(part: pd.DataFrame, score: np.ndarray, spec: OOFRankerSpec, feature_by_key: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    selected = opt.select_by_budget(part, score, spec.scope, spec.budget, {})
    return shared.selected_pool_rows(
        selected=selected,
        feature_by_key=feature_by_key,
        candidate=shared.ReplayCandidate(spec.name, spec.name, spec.scope, spec.budget, spec.active_count, spec.pool_size, "train_oof", "oof_interaction_route_train_only"),
    )


def run_optimizer(max_replays: int = 14) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = train_opt.load_seed_mutations()
    train_config, holdout_config = shared.load_base_config()
    df = opt.read_pipeline()
    train_raw = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout_raw = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    feature_rows_train = shared.load_feature_rows("train")
    feature_rows_holdout = shared.load_feature_rows("holdout")
    train = base.merge_causal_features(train_raw, feature_rows_train)
    holdout = base.merge_causal_features(holdout_raw, feature_rows_holdout)
    fitted = fit_oof_rankers(train, holdout)

    pool_map: dict[str, list[dict[str, Any]]] = {}
    pool_summaries: list[dict[str, Any]] = []
    for name, info in fitted.items():
        spec = info["spec"]
        pool_rows = pool_rows_for_score(train, info["train_score"], spec, feature_rows_train)
        pool_map[name] = pool_rows
        base.write_jsonl(OUT_DIR / f"pool_rows_train_{name}.jsonl", pool_rows)
        active = [row for row in pool_rows if bool(row.get("pool_active"))]
        pool_summaries.append(
            {
                "ranker": spec.__dict__,
                "fold_proxy_rows": info.get("fold_proxy_rows") or [],
                "pool_rows": len(pool_rows),
                "active_rows": len(active),
                "active_proxy_positive_r": sum(num(row.get("optimizer_positive_r_proxy")) for row in active),
                "active_proxy_net_eod_r": sum(num(row.get("optimizer_net_eod_r_proxy")) for row in active),
                "active_days": len({str(row.get("trade_date") or "")[:10] for row in active}),
            }
        )
    log("oof_pool_build_done", pools=len(pool_map))

    route_screen_rows: list[dict[str, Any]] = []
    routes = base.route_specs()
    for pool_name, pool_rows in pool_map.items():
        for route in routes:
            proxy = base.route_proxy_summary(pool_rows, route.build(seed))
            route_screen_rows.append(
                {
                    "ranker_name": pool_name,
                    "ranker": fitted[pool_name]["spec"].__dict__,
                    "route": {"name": route.name, "description": route.description},
                    "proxy": proxy,
                    "route_screen_score": base.route_screen_score(proxy),
                }
            )
    route_screen_ranked = sorted(route_screen_rows, key=lambda row: num(row.get("route_screen_score")), reverse=True)
    forced = {
        ("oof_base_hgb_quality_top16", "seed_risk99"),
        ("oof_interaction_trees_quality_top16", "first30_soft_quality4"),
        ("oof_interaction_trees_quality_top16", "delayed_rank16_r15_soft"),
        ("oof_interaction_hgb_routeable_top16", "first30_soft_quality5"),
    }
    replay_keys: set[tuple[str, str]] = set(forced)
    for row in route_screen_ranked:
        replay_keys.add((str(row["ranker_name"]), str(row["route"]["name"])))
        if len(replay_keys) >= max_replays:
            break
    write_json(OUT_DIR / "route_proxy_screen.json", route_screen_ranked)
    log("oof_route_screen_done", screened=len(route_screen_ranked), selected_replays=len(replay_keys))

    log("train_context_build_start")
    train_bundle = shared.build_window_bundle(train_config)
    train_dates = list(train_bundle["dataset"].trading_dates)
    initial_equity = float((train_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    log("train_context_build_done", sessions=len(train_dates), contexts=len(train_bundle["context_by_key"]))

    route_by_name = {route.name: route for route in routes}
    train_rows: list[dict[str, Any]] = []
    for ranker_name, route_name in sorted(replay_keys):
        spec = fitted[ranker_name]["spec"]
        route = route_by_name[route_name]
        pool_rows = pool_map[ranker_name]
        mutations = route.build(seed)
        proxy = base.route_proxy_summary(pool_rows, mutations)
        if num(proxy.get("route_eligible_count")) < 45.0:
            train_rows.append(
                {
                    "ranker_name": ranker_name,
                    "ranker": spec.__dict__,
                    "route": {"name": route.name, "description": route.description},
                    "proxy": proxy,
                    "metrics": {"trade_count": 0.0},
                    "stability": {},
                    "skipped": True,
                    "skip_reason": "route_eligible_count_lt_45",
                    "train_alpha_conversion_score": -999.0,
                    "train_alpha_conversion_pass": False,
                }
            )
            log("train_replay_skipped", ranker=ranker_name, route=route.name, eligible=proxy.get("route_eligible_count"))
            continue
        replay_name = f"{ranker_name}_{route.name}"
        log("train_replay_start", ranker=ranker_name, route=route.name, eligible=proxy.get("route_eligible_count"), proxy_r=proxy.get("route_eligible_proxy_positive_r"))
        result = base.evaluate_compiled_candidate_pool(
            window="train",
            variant=base.PoolVariant(replay_name, spec.pool_size, active_count=spec.active_count),
            config=train_config,
            dataset=train_bundle["dataset"],
            context_by_key=train_bundle["context_by_key"],
            pool_rows=pool_rows,
            seed_mutations=mutations,
            output_dir=OUT_DIR,
            replay_name=replay_name,
        )
        metrics = dict(result.get("metrics") or {})
        trades = base.read_trade_rows(result.get("trade_rows_path"))
        stability = base.stability_metrics(trades, train_dates, initial_equity)
        score = base.alpha_conversion_score(metrics, stability, proxy)
        row = {
            "ranker_name": ranker_name,
            "ranker": spec.__dict__,
            "route": {"name": route.name, "description": route.description},
            "proxy": proxy,
            "metrics": metrics,
            "stability": stability,
            "compiled_replay": result.get("compiled_replay") or {},
            "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
            "trade_rows_path": result.get("trade_rows_path"),
            "skipped": False,
            **score,
        }
        train_rows.append(row)
        log(
            "train_replay_done",
            ranker=ranker_name,
            route=route.name,
            score=score["train_alpha_conversion_score"],
            pass_hygiene=score["train_alpha_conversion_pass"],
            trades=metrics.get("trade_count"),
            net=metrics.get("broker_net_return_pct"),
            dd=metrics.get("broker_max_drawdown_pct"),
            worst5=stability.get("five_fold_worst_net"),
        )

    replayed = [row for row in train_rows if not row.get("skipped")]
    pass_rows = [row for row in replayed if row.get("train_alpha_conversion_pass")]
    champion = max(pass_rows or replayed, key=lambda row: num(row.get("train_alpha_conversion_score"))) if (pass_rows or replayed) else {}

    locked_holdout: dict[str, Any] = {}
    if champion:
        ranker_name = str(champion["ranker_name"])
        route_name = str((champion.get("route") or {}).get("name") or "")
        spec = fitted[ranker_name]["spec"]
        route = route_by_name[route_name]
        log("locked_holdout_context_build_start", ranker=ranker_name, route=route_name)
        holdout_bundle = shared.build_window_bundle(holdout_config)
        holdout_dates = list(holdout_bundle["dataset"].trading_dates)
        holdout_equity = float((holdout_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
        holdout_pool = pool_rows_for_score(holdout, fitted[ranker_name]["holdout_score"], spec, feature_rows_holdout)
        base.write_jsonl(OUT_DIR / f"pool_rows_locked_holdout_{ranker_name}_{route_name}.jsonl", holdout_pool)
        proxy = base.route_proxy_summary(holdout_pool, route.build(seed))
        log("locked_holdout_replay_start", ranker=ranker_name, route=route_name, rows=len(holdout_pool), eligible=proxy.get("route_eligible_count"))
        result = base.evaluate_compiled_candidate_pool(
            window="holdout_locked_audit",
            variant=base.PoolVariant(f"{ranker_name}_{route_name}_locked_holdout", spec.pool_size, active_count=spec.active_count),
            config=holdout_config,
            dataset=holdout_bundle["dataset"],
            context_by_key=holdout_bundle["context_by_key"],
            pool_rows=holdout_pool,
            seed_mutations=route.build(seed),
            output_dir=OUT_DIR,
            replay_name=f"{ranker_name}_{route_name}_locked_holdout",
        )
        metrics = dict(result.get("metrics") or {})
        trades = base.read_trade_rows(result.get("trade_rows_path"))
        locked_holdout = {
            "selection_basis": "locked_train_oof_selected_ranker_route_only_no_holdout_optimization",
            "ranker": spec.__dict__,
            "route": {"name": route.name, "description": route.description},
            "proxy": proxy,
            "metrics": metrics,
            "stability": base.stability_metrics(trades, holdout_dates, holdout_equity),
            "compiled_replay": result.get("compiled_replay") or {},
            "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
            "trade_rows_path": result.get("trade_rows_path"),
        }
        log("locked_holdout_replay_done", ranker=ranker_name, route=route_name, trades=metrics.get("trade_count"), net=metrics.get("broker_net_return_pct"), dd=metrics.get("broker_max_drawdown_pct"))

    return {
        "created_at_utc": now_iso(),
        "usage_contract": "train_only_oof_interaction_route_optimizer_holdout_locked_after_train_selection",
        "objective": {
            "selection_data": "train only, chronological out-of-fold train scores for ranker selection",
            "score": "shared-core train replay score with frequency/capture/stability and drawdown penalties",
            "holdout_policy": "locked audit after train OOF ranker/route champion is selected",
        },
        "ranker_pool_summaries": sorted(pool_summaries, key=lambda row: num(row.get("active_proxy_positive_r")), reverse=True),
        "route_proxy_screen": route_screen_ranked,
        "selected_replay_keys": sorted([{"ranker": a, "route": b} for a, b in replay_keys], key=lambda x: (x["ranker"], x["route"])),
        "train_rows": train_rows,
        "train_champion": champion,
        "locked_holdout_audit": locked_holdout,
        "max_replays": max_replays,
    }


def route_mix(row: dict[str, Any]) -> str:
    return base.route_mix(row)


def render_report(summary: dict[str, Any]) -> str:
    rows = [row for row in summary.get("train_rows", []) if not row.get("skipped")]
    ranked = sorted(rows, key=lambda row: num(row.get("train_alpha_conversion_score")), reverse=True)
    champion = dict(summary.get("train_champion") or {})
    lines = [
        "# KALCB OOF Interaction + Route Optimizer",
        "",
        "Train-only chronological out-of-fold ranker sweep with shared-core route conversion replays. Holdout is locked until after train selection.",
        "",
        "## Train OOF Champion",
        "",
    ]
    if champion:
        ranker = dict(champion.get("ranker") or {})
        route = dict(champion.get("route") or {})
        metrics = dict(champion.get("metrics") or {})
        stability = dict(champion.get("stability") or {})
        proxy = dict(champion.get("proxy") or {})
        lines.extend(
            [
                f"- Ranker: `{ranker.get('name')}` ({ranker.get('feature_set')}, {ranker.get('model_kind')}, {ranker.get('target')}, {ranker.get('budget')})",
                f"- Route: `{route.get('name')}` ({route.get('description')})",
                f"- Train score: {num(champion.get('train_alpha_conversion_score')):.2f}; pass: {champion.get('train_alpha_conversion_pass')}",
                f"- Train net/DD/trades: {pct(metrics.get('broker_net_return_pct'))} / {pct(metrics.get('broker_max_drawdown_pct'))} / {num(metrics.get('trade_count')):.0f}",
                f"- Train capture/five-fold worst/eligible proxy R: {pct(metrics.get('avg_mfe_capture'))} / {pct(stability.get('five_fold_worst_net'))} / {num(proxy.get('route_eligible_proxy_positive_r')):.1f}R",
                f"- Route mix: {route_mix(champion)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Train Replays",
            "",
            "| rank | pass | ranker | route | score | net | DD | trades | 5-fold worst | capture | eligible R | routes |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(ranked[:18], start=1):
        ranker = dict(row.get("ranker") or {})
        route = dict(row.get("route") or {})
        metrics = dict(row.get("metrics") or {})
        stability = dict(row.get("stability") or {})
        proxy = dict(row.get("proxy") or {})
        lines.append(
            f"| {rank} | {row.get('train_alpha_conversion_pass')} | `{ranker.get('name')}` | `{route.get('name')}` | "
            f"{num(row.get('train_alpha_conversion_score')):.2f} | {pct(metrics.get('broker_net_return_pct'))} | "
            f"{pct(metrics.get('broker_max_drawdown_pct'))} | {num(metrics.get('trade_count')):.0f} | "
            f"{pct(stability.get('five_fold_worst_net'))} | {pct(metrics.get('avg_mfe_capture'))} | "
            f"{num(proxy.get('route_eligible_proxy_positive_r')):.1f} | {route_mix(row)} |"
        )
    lines.extend(["", "## OOF Pool Proxy", ""])
    lines.append("| rank | ranker | active R | active EOD R | fold selected R | active rows |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for rank, row in enumerate(list(summary.get("ranker_pool_summaries") or [])[:8], start=1):
        ranker = dict(row.get("ranker") or {})
        fold_r = sum(num(item.get("selected_positive_r")) for item in row.get("fold_proxy_rows") or [])
        lines.append(
            f"| {rank} | `{ranker.get('name')}` | {num(row.get('active_proxy_positive_r')):.1f} | "
            f"{num(row.get('active_proxy_net_eod_r')):.1f} | {fold_r:.1f} | {num(row.get('active_rows')):.0f} |"
        )
    holdout = dict(summary.get("locked_holdout_audit") or {})
    if holdout:
        metrics = dict(holdout.get("metrics") or {})
        stability = dict(holdout.get("stability") or {})
        proxy = dict(holdout.get("proxy") or {})
        lines.extend(
            [
                "",
                "## Locked Holdout Audit",
                "",
                f"- Train-selected ranker/route: `{(holdout.get('ranker') or {}).get('name')}` + `{(holdout.get('route') or {}).get('name')}`",
                f"- Holdout net/DD/trades: {pct(metrics.get('broker_net_return_pct'))} / {pct(metrics.get('broker_max_drawdown_pct'))} / {num(metrics.get('trade_count')):.0f}",
                f"- Holdout capture/five-fold worst/eligible proxy R: {pct(metrics.get('avg_mfe_capture'))} / {pct(stability.get('five_fold_worst_net'))} / {num(proxy.get('route_eligible_proxy_positive_r')):.1f}R",
                f"- Route mix: {route_mix(holdout)}",
                "- Locked audit note: this was not used to choose the train champion.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    max_replays = int(os.environ.get("KALCB_MAX_REPLAYS", "14") or "14")
    log("start", out_dir=str(OUT_DIR), max_replays=max_replays)
    summary = run_optimizer(max_replays=max_replays)
    summary["elapsed_seconds"] = round(time.time() - start, 3)
    write_json(OUT_DIR / "kalcb_oof_interaction_route_results.json", summary)
    report = render_report(summary)
    (OUT_DIR / "kalcb_oof_interaction_route_report.md").write_text(report, encoding="utf-8")
    log("complete", elapsed_seconds=summary["elapsed_seconds"], summary=str(OUT_DIR / "kalcb_oof_interaction_route_results.json"), report=str(OUT_DIR / "kalcb_oof_interaction_route_report.md"))


if __name__ == "__main__":
    main()
