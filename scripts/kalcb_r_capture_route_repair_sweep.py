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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SHARED_PATH = REPO_ROOT / "scripts" / "kalcb_r_capture_shared_core_replay.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
BASE_REPLAY_DIR = ROUND_DIR / "r_capture_optimizer" / "shared_core_replay"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "route_repair_sweep"

from backtests.strategies.kalcb.candidate_surfacing_recovery import (  # noqa: E402
    PoolVariant,
    build_conservative_route_family_mutations,
    evaluate_compiled_candidate_pool,
)
from backtests.strategies.kalcb.shadow_ledger_reranker import read_jsonl  # noqa: E402


def _load_shared_module():
    spec = importlib.util.spec_from_file_location("kalcb_r_capture_shared_core_replay_module", SHARED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared replay module: {SHARED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared_module()


@dataclass(frozen=True)
class SweepSpec:
    name: str
    pool_stem: str
    active_count: int
    pool_size: int
    mutation_family: str
    purpose: str
    min_bar_ret: float | None = None
    min_vwap_ret: float | None = None
    quality_min_bar_ret: float | None = None
    delayed_relvol_context_min: float | None = None
    remove_delayed_relvol_context: bool = False
    first30_risk_mult: float | None = None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


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


def load_pool(window: str, stem: str) -> list[dict[str, Any]]:
    path = BASE_REPLAY_DIR / f"pool_rows_{window}_{stem}.jsonl"
    return read_jsonl(path)


def apply_mutation_spec(seed: dict[str, Any], spec: SweepSpec) -> dict[str, Any]:
    if spec.mutation_family == "seed":
        out = copy.deepcopy(seed)
    elif spec.mutation_family == "delayed":
        out = build_conservative_route_family_mutations(seed)
    else:
        raise ValueError(f"Unknown mutation family: {spec.mutation_family}")

    routes = [dict(route) for route in out.get("kalcb.entry.routes") or [] if isinstance(route, dict)]
    for route in routes:
        mode = str(route.get("mode") or "")
        if spec.min_bar_ret is not None:
            route["min_bar_ret"] = float(spec.min_bar_ret)
        if spec.min_vwap_ret is not None:
            route["min_vwap_ret"] = float(spec.min_vwap_ret)
        if spec.quality_min_bar_ret is not None:
            route["quality_min_bar_ret"] = float(spec.quality_min_bar_ret)
        if mode != "first30_open":
            context_min = dict(route.get("context_min") or {})
            if spec.remove_delayed_relvol_context:
                context_min.pop("first30_rel_volume", None)
            elif spec.delayed_relvol_context_min is not None:
                context_min["first30_rel_volume"] = float(spec.delayed_relvol_context_min)
            route["context_min"] = context_min
        if mode == "first30_open" and spec.first30_risk_mult is not None:
            route["risk_mult"] = float(spec.first30_risk_mult)
            route["notional_mult"] = float(spec.first30_risk_mult)
    out["kalcb.entry.routes"] = routes
    if spec.quality_min_bar_ret is not None:
        out["kalcb.entry.quality_min_bar_ret"] = float(spec.quality_min_bar_ret)
    return out


def specs() -> list[SweepSpec]:
    primary = "primary_dataset_all_context_trees_quality_top4_seed"
    challenger = "challenger_surfaced_all_context_ridge_quality_top8_seed"
    stage2 = "stage2_current_surfaced_all_context_hgb_quality_top8_seed"
    return [
        SweepSpec(
            name="primary_top4_delayed_relaxed_ret_vwap_keep_q85",
            pool_stem=primary,
            active_count=4,
            pool_size=4,
            mutation_family="delayed",
            min_bar_ret=-0.03,
            min_vwap_ret=-0.02,
            quality_min_bar_ret=-9.99,
            purpose="Allow later-reversal top4 names past static first30/VWAP gates while keeping q85 RVOL context.",
        ),
        SweepSpec(
            name="primary_top4_delayed_relaxed_ret_vwap_relvol3",
            pool_stem=primary,
            active_count=4,
            pool_size=4,
            mutation_family="delayed",
            min_bar_ret=-0.03,
            min_vwap_ret=-0.02,
            quality_min_bar_ret=-9.99,
            delayed_relvol_context_min=3.0,
            purpose="Same as q85 repair, but lowers delayed-route RVOL context to test whether q85 is choking conversion.",
        ),
        SweepSpec(
            name="primary_top4_seed_relaxed_open_risk50",
            pool_stem=primary,
            active_count=4,
            pool_size=4,
            mutation_family="seed",
            min_bar_ret=-0.03,
            min_vwap_ret=-0.02,
            quality_min_bar_ret=-9.99,
            first30_risk_mult=0.50,
            purpose="Let top4 later-reversal names use incumbent first30 route at half first30 risk.",
        ),
        SweepSpec(
            name="challenger_top8_seed_first30_risk65",
            pool_stem=challenger,
            active_count=8,
            pool_size=8,
            mutation_family="seed",
            first30_risk_mult=0.65,
            purpose="Reduce drawdown on the all-context top8 challenger while preserving route gates.",
        ),
        SweepSpec(
            name="challenger_top8_seed_first30_risk50",
            pool_stem=challenger,
            active_count=8,
            pool_size=8,
            mutation_family="seed",
            first30_risk_mult=0.50,
            purpose="Lower-risk all-context top8 challenger for train DD hygiene.",
        ),
        SweepSpec(
            name="stage2_top8_seed_first30_risk50",
            pool_stem=stage2,
            active_count=8,
            pool_size=8,
            mutation_family="seed",
            first30_risk_mult=0.50,
            purpose="Risk-scaled current-surfaced Stage 2 diagnostic.",
        ),
    ]


def replay_sweep() -> dict[str, Any]:
    train_config, holdout_config = shared.load_base_config()
    seed = shared.load_seed_mutations()
    bundles: dict[str, dict[str, Any]] = {}
    for window, config in (("train", train_config), ("holdout", holdout_config)):
        log("context_build_start", window=window)
        bundles[window] = shared.build_window_bundle(config)
        log(
            "context_build_done",
            window=window,
            sessions=len(bundles[window]["dataset"].trading_dates),
            contexts=len(bundles[window]["context_by_key"]),
        )

    rows: list[dict[str, Any]] = []
    for spec in specs():
        mutations = apply_mutation_spec(seed, spec)
        variant = PoolVariant(spec.name, spec.pool_size, active_count=spec.active_count)
        for window, config in (("train", train_config), ("holdout", holdout_config)):
            pool_rows = load_pool(window, spec.pool_stem)
            log("replay_start", spec=spec.name, window=window)
            result = evaluate_compiled_candidate_pool(
                window=window,
                variant=variant,
                config=config,
                dataset=bundles[window]["dataset"],
                context_by_key=bundles[window]["context_by_key"],
                pool_rows=pool_rows,
                seed_mutations=mutations,
                output_dir=OUT_DIR,
                replay_name=spec.name,
            )
            metrics = dict(result.get("metrics") or {})
            row = {
                **spec.__dict__,
                "window": window,
                "proxy_positive_r": sum(num(item.get("optimizer_positive_r_proxy")) for item in pool_rows),
                "proxy_net_eod_r": sum(num(item.get("optimizer_net_eod_r_proxy")) for item in pool_rows),
                "metrics": metrics,
                "compiled_replay": result.get("compiled_replay") or {},
                "trade_rows_path": result.get("trade_rows_path"),
                "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
            }
            rows.append(row)
            log(
                "replay_done",
                spec=spec.name,
                window=window,
                trades=metrics.get("trade_count"),
                net=metrics.get("broker_net_return_pct"),
                dd=metrics.get("broker_max_drawdown_pct"),
                capture=metrics.get("avg_mfe_capture"),
                eligible=(result.get("compiled_replay") or {}).get("static_route_eligible_count"),
            )
    return {
        "created_at_utc": now_iso(),
        "usage_contract": "research_only_route_and_risk_repair_sweep_after_shared_core_r_capture_replay",
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
        "# KALCB R-Capture Route Repair Sweep",
        "",
        "Small shared-core sweep targeted at the measured route blockers from the R-capture replay.",
        "",
        "## Holdout Ranking",
        "",
        "| rank | spec | proxy R | trades | net | DD | capture | eligible | purpose |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(holdout_sorted, start=1):
        metrics = dict(row.get("metrics") or {})
        compiled = dict(row.get("compiled_replay") or {})
        lines.append(
            f"| {rank} | `{row.get('name')}` | {num(row.get('proxy_positive_r')):.1f} | "
            f"{num(metrics.get('trade_count')):.0f} | {pct(metrics.get('broker_net_return_pct'))} | "
            f"{pct(metrics.get('broker_max_drawdown_pct'))} | {pct(metrics.get('avg_mfe_capture'))} | "
            f"{num(compiled.get('static_route_eligible_count')):.0f} | {row.get('purpose')} |"
        )
    lines.extend(["", "## Train/Holdout Details", ""])
    lines.append("| spec | window | trades | net | DD | avg MFE R | avg MAE R | capture | routes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: (str(item.get("name")), str(item.get("window")))):
        metrics = dict(row.get("metrics") or {})
        route_summary = dict(row.get("entry_route_mode_summary") or {})
        routes = "; ".join(f"{mode}:{int(num((data or {}).get('trades')))}" for mode, data in route_summary.items())
        lines.append(
            f"| `{row.get('name')}` | {row.get('window')} | {num(metrics.get('trade_count')):.0f} | "
            f"{pct(metrics.get('broker_net_return_pct'))} | {pct(metrics.get('broker_max_drawdown_pct'))} | "
            f"{num(metrics.get('avg_mfe_r')):.2f} | {num(metrics.get('avg_mae_r')):.2f} | "
            f"{pct(metrics.get('avg_mfe_capture'))} | {routes} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Risk-scaled all-context top8 is the first promotable-looking branch if it preserves train drawdown below the 8% hygiene line.",
            "- Relaxing static first30/VWAP gates should only advance if it creates real shared-core holdout trades without train drawdown blowout.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("start", specs=len(specs()))
    summary = replay_sweep()
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary_path = OUT_DIR / "kalcb_r_capture_route_repair_sweep_results.json"
    report_path = OUT_DIR / "kalcb_r_capture_route_repair_sweep_report.md"
    write_json(summary_path, summary)
    report_path.write_text(render_report(summary), encoding="utf-8")
    log("complete", summary=str(summary_path), report=str(report_path), elapsed_seconds=summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
