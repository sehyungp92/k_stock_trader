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

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OPTIMIZER_PATH = REPO_ROOT / "scripts" / "kalcb_r_capture_optimizer.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "shared_core_replay"
CONFIG_PATH = REPO_ROOT / "config" / "optimization" / "kalcb.yaml"
SEED_PATH = (
    ROUND_DIR
    / "local_minimum_recovery"
    / "07_alpha_conversion_next_round"
    / "next_round_seed_auto_pullback_q85_rank8_r0p015.json"
)
FEATURE_DIR = ROUND_DIR / "local_minimum_recovery" / "08_candidate_surfacing_recovery"

from backtests.strategies.kalcb.candidate_surfacing_recovery import (  # noqa: E402
    PoolVariant,
    build_conservative_route_family_mutations,
    evaluate_compiled_candidate_pool,
)
from backtests.strategies.kalcb.first30_signal_sweep import build_contexts, prepare_first30_dataset  # noqa: E402
from backtests.strategies.kalcb.shadow_ledger_reranker import read_jsonl, write_jsonl  # noqa: E402


def _load_optimizer_module():
    spec = importlib.util.spec_from_file_location("kalcb_r_capture_optimizer_module", OPTIMIZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load optimizer module: {OPTIMIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


opt = _load_optimizer_module()


@dataclass(frozen=True)
class ReplayCandidate:
    name: str
    label: str
    scope: str
    budget: str
    active_count: int
    pool_size: int
    execution: str
    purpose: str


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


def symbol_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def load_base_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    config["workers"] = 2
    config["validation_gate_enabled"] = False
    config["skip_initial_baseline_eval"] = True
    train_config = copy.deepcopy(config)
    holdout_config = copy.deepcopy(config)
    holdout = dict(holdout_config.get("baseline") or {})
    holdout_config["start"] = str(holdout["holdout_start"])
    holdout_config["end"] = str(holdout["holdout_end"])
    holdout_config["use_full_available_window"] = True
    return train_config, holdout_config


def load_seed_mutations() -> dict[str, Any]:
    payload = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    mutations = payload.get("mutations")
    if not isinstance(mutations, dict):
        raise ValueError(f"Missing mutations in {SEED_PATH}")
    return copy.deepcopy(mutations)


def build_window_bundle(config: dict[str, Any]) -> dict[str, Any]:
    dataset = prepare_first30_dataset(config)
    contexts = build_contexts(dataset)
    context_by_key = {(day, ctx.symbol): ctx for day, items in contexts.items() for ctx in items}
    return {"dataset": dataset, "contexts": contexts, "context_by_key": context_by_key}


def load_feature_rows(window: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = FEATURE_DIR / f"candidate_surfacing_{window}_features.jsonl"
    rows = read_jsonl(path)
    return {
        (str(row.get("trade_date") or "")[:10], symbol_key(row.get("symbol"))): dict(row)
        for row in rows
        if row.get("trade_date") and row.get("symbol")
    }


def selected_pool_rows(
    *,
    selected: Any,
    feature_by_key: dict[tuple[str, str], dict[str, Any]],
    candidate: ReplayCandidate,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing = 0
    for day, group in selected.groupby("trade_date", sort=True):
        ordered = group.sort_values("_score", ascending=False).head(candidate.pool_size)
        for rank, (_, source) in enumerate(ordered.iterrows(), start=1):
            day_label = str(day)[:10]
            symbol = symbol_key(source.get("symbol"))
            row = dict(feature_by_key.get((day_label, symbol)) or {})
            if not row:
                missing += 1
                row = {
                    "trade_date": day_label,
                    "symbol": symbol,
                    "sector": str(source.get("sector") or "UNKNOWN"),
                    "first30_ret": num(source.get("first30_ret")),
                    "first30_vwap_ret": num(source.get("first30_vwap_ret")),
                    "first30_gap": num(source.get("first30_gap")),
                    "first30_rel_volume": num(source.get("first30_rel_volume")),
                    "first30_signal_bar_cpr": num(source.get("first30_close_location")),
                    "first30_range_close_location": num(source.get("first30_close_location")),
                    "first30_open_drawdown": num(source.get("first30_open_drawdown")),
                    "first30_low_vs_prev_close": num(source.get("first30_low_vs_prev_close")),
                    "first30_range_atr": num(source.get("first30_range_atr")),
                }
            row.update(
                {
                    "trade_date": day_label,
                    "symbol": symbol,
                    "window": str(source.get("window") or row.get("window") or ""),
                    "pool_variant": candidate.name,
                    "pool_size": candidate.pool_size,
                    "pool_rank": rank,
                    "pool_active": rank <= candidate.active_count,
                    "frontier_role_for_replay": "initial_active" if rank <= candidate.active_count else "frontier_shadow",
                    "causal_ranker_score": num(source.get("_score")),
                    "optimizer_label": candidate.label,
                    "optimizer_scope": candidate.scope,
                    "optimizer_budget": candidate.budget,
                    "optimizer_positive_r_proxy": num(source.get("positive_r")),
                    "optimizer_net_eod_r_proxy": num(source.get("net_eod_r_proxy")),
                    "optimizer_mae_r_proxy": num(source.get("mae_r_proxy")),
                    "optimizer_surfaced_candidate": bool(num(source.get("surfaced_candidate"))),
                    "optimizer_selected_first30": bool(num(source.get("selected_first30"))),
                }
            )
            rows.append(row)
    if missing:
        log("pool_feature_fallback_rows", candidate=candidate.name, missing=missing)
    return rows


def prepare_ranker_pools(candidates: list[ReplayCandidate]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    df = opt.read_pipeline()
    train = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    specs = opt.model_specs()
    scores = opt.fit_scores(train, holdout, specs)
    feature_rows = {"train": load_feature_rows("train"), "holdout": load_feature_rows("holdout")}
    parts = {"train": train, "holdout": holdout}
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for candidate in candidates:
        pools[candidate.name] = {}
        for window, part in parts.items():
            selected = opt.select_by_budget(
                part,
                scores[candidate.label][window],
                candidate.scope,
                candidate.budget,
                opt.current_selected_counts(part) if candidate.budget == "current_count" else {},
            )
            pool_rows = selected_pool_rows(selected=selected, feature_by_key=feature_rows[window], candidate=candidate)
            pools[candidate.name][window] = pool_rows
            write_jsonl(OUT_DIR / f"pool_rows_{window}_{candidate.name}.jsonl", pool_rows)
            log(
                "pool_prepared",
                candidate=candidate.name,
                window=window,
                rows=len(pool_rows),
                active_rows=sum(1 for row in pool_rows if row.get("pool_active")),
                proxy_positive_r=sum(num(row.get("optimizer_positive_r_proxy")) for row in pool_rows),
                proxy_net_eod_r=sum(num(row.get("optimizer_net_eod_r_proxy")) for row in pool_rows),
            )
    return pools


def replay_candidates(candidates: list[ReplayCandidate], pools: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    train_config, holdout_config = load_base_config()
    seed = load_seed_mutations()
    execution_mutations = {
        "seed_execution": seed,
        "delayed_route_family": build_conservative_route_family_mutations(seed),
    }
    bundles: dict[str, dict[str, Any]] = {}
    for window, config in (("train", train_config), ("holdout", holdout_config)):
        log("context_build_start", window=window)
        bundles[window] = build_window_bundle(config)
        log(
            "context_build_done",
            window=window,
            sessions=len(bundles[window]["dataset"].trading_dates),
            symbols=len(bundles[window]["dataset"].symbols),
            contexts=len(bundles[window]["context_by_key"]),
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        mutations = execution_mutations[candidate.execution]
        variant = PoolVariant(candidate.name, candidate.pool_size, active_count=candidate.active_count)
        for window, config in (("train", train_config), ("holdout", holdout_config)):
            log("replay_start", candidate=candidate.name, execution=candidate.execution, window=window)
            result = evaluate_compiled_candidate_pool(
                window=window,
                variant=variant,
                config=config,
                dataset=bundles[window]["dataset"],
                context_by_key=bundles[window]["context_by_key"],
                pool_rows=pools[candidate.name][window],
                seed_mutations=mutations,
                output_dir=OUT_DIR,
                replay_name=candidate.name,
            )
            metrics = dict(result.get("metrics") or {})
            proxy_rows = pools[candidate.name][window]
            row = {
                "candidate": candidate.name,
                "label": candidate.label,
                "scope": candidate.scope,
                "budget": candidate.budget,
                "execution": candidate.execution,
                "purpose": candidate.purpose,
                "window": window,
                "proxy_positive_r": sum(num(item.get("optimizer_positive_r_proxy")) for item in proxy_rows),
                "proxy_net_eod_r": sum(num(item.get("optimizer_net_eod_r_proxy")) for item in proxy_rows),
                "proxy_rows": len(proxy_rows),
                "metrics": metrics,
                "compiled_replay": result.get("compiled_replay") or {},
                "trade_rows_path": result.get("trade_rows_path"),
                "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
                "replay_digest": result.get("replay_digest") or {},
            }
            rows.append(row)
            log(
                "replay_done",
                candidate=candidate.name,
                execution=candidate.execution,
                window=window,
                trades=metrics.get("trade_count"),
                net=metrics.get("broker_net_return_pct"),
                dd=metrics.get("broker_max_drawdown_pct"),
                capture=metrics.get("avg_mfe_capture"),
            )
    return {
        "created_at_utc": now_iso(),
        "usage_contract": "research_only_train_frozen_ranker_replayed_through_shared_kalcb_core",
        "rows": rows,
    }


def render_report(summary: dict[str, Any]) -> str:
    rows = list(summary.get("rows") or [])
    holdout = [row for row in rows if row.get("window") == "holdout"]
    holdout_sorted = sorted(
        holdout,
        key=lambda row: (
            num((row.get("metrics") or {}).get("broker_net_return_pct")),
            -num((row.get("metrics") or {}).get("broker_max_drawdown_pct")),
            num((row.get("metrics") or {}).get("trade_count")),
        ),
        reverse=True,
    )
    lines = [
        "# KALCB R-Capture Shared-Core Replay",
        "",
        "Train-frozen optimizer pools replayed through `KALCBReplayAdapter -> shared core -> SimBroker`.",
        "",
        "## Holdout Ranking",
        "",
        "| rank | candidate | execution | proxy R | proxy EOD R | trades | net | DD | MFE capture | static eligible |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(holdout_sorted, start=1):
        metrics = dict(row.get("metrics") or {})
        compiled = dict(row.get("compiled_replay") or {})
        lines.append(
            f"| {rank} | `{row.get('candidate')}` | `{row.get('execution')}` | "
            f"{num(row.get('proxy_positive_r')):.1f} | {num(row.get('proxy_net_eod_r')):.1f} | "
            f"{num(metrics.get('trade_count')):.0f} | {pct(metrics.get('broker_net_return_pct'))} | "
            f"{pct(metrics.get('broker_max_drawdown_pct'))} | {pct(metrics.get('avg_mfe_capture'))} | "
            f"{num(compiled.get('static_route_eligible_count')):.0f} |"
        )
    lines.extend(["", "## Window Details", ""])
    lines.append("| candidate | window | execution | trades | net | DD | avg MFE R | avg MAE R | capture | routes |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: (str(item.get("candidate")), str(item.get("window")))):
        metrics = dict(row.get("metrics") or {})
        route_summary = dict(row.get("entry_route_mode_summary") or {})
        route_bits = []
        for route, data in route_summary.items():
            route_bits.append(f"{route}:{int(num((data or {}).get('trades')))}")
        lines.append(
            f"| `{row.get('candidate')}` | {row.get('window')} | `{row.get('execution')}` | "
            f"{num(metrics.get('trade_count')):.0f} | {pct(metrics.get('broker_net_return_pct'))} | "
            f"{pct(metrics.get('broker_max_drawdown_pct'))} | {num(metrics.get('avg_mfe_r')):.2f} | "
            f"{num(metrics.get('avg_mae_r')):.2f} | {pct(metrics.get('avg_mfe_capture'))} | "
            f"{'; '.join(route_bits)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Proxy-R selection improves dramatically before execution, but shared-core conversion is still governed by route eligibility, same-day path, and exit capture.",
            "- A replay candidate is promotable only if the shared-core holdout net/drawdown/capture confirms the proxy R gain; high proxy R alone is not enough.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        ReplayCandidate(
            name="primary_dataset_all_context_trees_quality_top4_seed",
            label="dataset_all_context_trees_quality",
            scope="dataset",
            budget="top4",
            active_count=4,
            pool_size=4,
            execution="seed_execution",
            purpose="Balanced all-context first replay candidate from optimizer queue.",
        ),
        ReplayCandidate(
            name="primary_dataset_all_context_trees_quality_top4_delayed",
            label="dataset_all_context_trees_quality",
            scope="dataset",
            budget="top4",
            active_count=4,
            pool_size=4,
            execution="delayed_route_family",
            purpose="Same selected names with conservative delayed route-family execution.",
        ),
        ReplayCandidate(
            name="challenger_surfaced_all_context_ridge_quality_top8_seed",
            label="surfaced_all_context_ridge_quality",
            scope="dataset",
            budget="top8",
            active_count=8,
            pool_size=8,
            execution="seed_execution",
            purpose="Max all-context R-capture challenger with positive proxy EOD.",
        ),
        ReplayCandidate(
            name="stage2_current_surfaced_all_context_hgb_quality_top8_seed",
            label="surfaced_all_context_hgb_quality",
            scope="surfaced",
            budget="top8",
            active_count=8,
            pool_size=8,
            execution="seed_execution",
            purpose="Stage 2 current-surfaced selector diagnostic.",
        ),
    ]
    log("start", candidates=len(candidates))
    pools = prepare_ranker_pools(candidates)
    summary = replay_candidates(candidates, pools)
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary["candidate_plan"] = [candidate.__dict__ for candidate in candidates]
    summary_path = OUT_DIR / "kalcb_r_capture_shared_core_replay_results.json"
    report_path = OUT_DIR / "kalcb_r_capture_shared_core_replay_report.md"
    write_json(summary_path, summary)
    report_path.write_text(render_report(summary), encoding="utf-8")
    log("complete", summary=str(summary_path), report=str(report_path), elapsed_seconds=summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
