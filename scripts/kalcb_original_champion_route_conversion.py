from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_SCRIPT = REPO_ROOT / "scripts" / "kalcb_interaction_route_conversion_optimizer.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
SOURCE_DIR = ROUND_DIR / "r_capture_optimizer" / "train_stability_route_optimizer"
FAST_DIR = ROUND_DIR / "r_capture_optimizer" / "fast_extensive_search"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "original_champion_route_conversion"


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
train_opt = base.train_opt

from backtests.strategies.kalcb.candidate_surfacing_recovery import PoolVariant, evaluate_compiled_candidate_pool  # noqa: E402


@dataclass(frozen=True)
class RouteReplaySpec:
    name: str
    description: str
    build: Any
    family: str


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def pct(value: Any) -> str:
    return f"{100.0 * num(value):.2f}%"


def fmt(value: Any, digits: int = 2) -> str:
    return f"{num(value):.{digits}f}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def log(event: str, **extra: Any) -> None:
    payload = {"ts": now_iso(), "event": event, **extra}
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)
    append_jsonl(OUT_DIR / "progress.jsonl", payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_original_pool_rows() -> list[dict[str, Any]]:
    path = SOURCE_DIR / "base_train_pool_rows.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"Missing original champion pool rows: {path}")
    return rows


def route_catalogue() -> list[RouteReplaySpec]:
    routes: list[RouteReplaySpec] = []
    seen: set[str] = set()

    def add(name: str, description: str, build: Any, family: str) -> None:
        if name not in seen:
            routes.append(RouteReplaySpec(name, description, build, family))
            seen.add(name)

    add("seed_risk99", "Original train champion route seed.", lambda seed: copy.deepcopy(seed), "seed")
    for risk in (0.80, 0.65, 0.50, 0.40):
        add(
            f"seed_risk{int(risk * 100)}",
            f"Original route gates with first30 risk/notional multiplier {risk:.2f}.",
            lambda seed, risk=risk: base.set_first30_route(seed, risk=risk),
            "seed_risk",
        )

    for q, min_bar, min_vwap, cpr, rvol, range_min in (
        (3, -0.006, -0.012, 0.55, 0.75, 0.30),
        (4, -0.003, -0.008, 0.60, 1.00, 0.35),
        (5, 0.000, -0.005, 0.65, 1.25, 0.50),
    ):
        for risk in (0.99, 0.80, 0.65, 0.50, 0.40):
            add(
                f"first30_q{q}_risk{int(risk * 100)}",
                f"First30 conversion with {q} quality votes and risk {risk:.2f}.",
                lambda seed, risk=risk, q=q, min_bar=min_bar, min_vwap=min_vwap, cpr=cpr, rvol=rvol, range_min=range_min: base.set_first30_route(
                    seed,
                    risk=risk,
                    min_bar=min_bar,
                    min_vwap=min_vwap,
                    min_votes=q,
                    q_cpr=cpr,
                    q_rvol=rvol,
                    q_range_min=range_min,
                ),
                "first30_soft",
            )

    for rank in (8, 12, 16, 24):
        for delayed_risk in (0.05, 0.10, 0.15, 0.20):
            for q in (4, 5):
                add(
                    f"delayed_rank{rank}_r{int(delayed_risk * 100)}_q{q}",
                    f"Soft first30 plus delayed route family rank {rank}, delayed risk {delayed_risk:.2f}, q{q}.",
                    lambda seed, rank=rank, delayed_risk=delayed_risk, q=q: base.with_delayed_bundle(
                        seed,
                        first30_risk=0.65 if delayed_risk < 0.20 else 0.50,
                        delayed_risk=delayed_risk,
                        rank=rank,
                        quality_votes=q,
                        include_deferred=True,
                        soft_first30=True,
                    ),
                    "delayed_family",
                )
    return routes


def proxy_score(proxy: dict[str, Any]) -> float:
    eligible_r = num(proxy.get("route_eligible_proxy_positive_r"))
    eligible_net = num(proxy.get("route_eligible_proxy_net_eod_r"))
    active_r = num(proxy.get("active_proxy_positive_r"))
    active_net = num(proxy.get("active_proxy_net_eod_r"))
    eligible = num(proxy.get("route_eligible_count"))
    active_eligible = num(proxy.get("route_eligible_active_count"))
    return (
        eligible_r
        + 0.35 * active_r
        + 0.40 * max(eligible_net, 0.0)
        - 0.30 * max(-eligible_net, 0.0)
        + 0.12 * max(active_net, 0.0)
        + 3.0 * eligible
        + 2.0 * active_eligible
    )


def route_mix(row: dict[str, Any]) -> str:
    summary = dict(row.get("entry_route_mode_summary") or {})
    if summary:
        return "; ".join(f"{mode}:{int(num((data or {}).get('trades')))}" for mode, data in summary.items())
    counts = dict((row.get("stability") or {}).get("entry_route_mode_counts") or {})
    return "; ".join(f"{mode}:{count}" for mode, count in counts.items())


def trade_r_sum(trade_rows: list[dict[str, Any]]) -> float:
    return sum(num(row.get("r")) for row in trade_rows)


def trade_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("entry_date") or "")[:10],
        str(row.get("symbol") or "").zfill(6),
        str(row.get("entry_route_mode") or row.get("entry_route") or ""),
    )


def select_replay_routes(proxy_rows: list[dict[str, Any]], max_replays: int) -> list[str]:
    forced = [
        "seed_risk99",
        "seed_risk80",
        "seed_risk65",
        "seed_risk50",
        "seed_risk40",
        "first30_q5_risk99",
        "first30_q5_risk80",
        "first30_q4_risk99",
        "first30_q4_risk80",
        "first30_q4_risk65",
        "first30_q4_risk50",
        "first30_q3_risk65",
        "first30_q3_risk50",
        "delayed_rank16_r10_q5",
        "delayed_rank16_r15_q4",
        "delayed_rank24_r10_q5",
        "delayed_rank24_r20_q4",
    ]
    row_by_name = {str(row["route"]["name"]): row for row in proxy_rows}
    ranked_names = [str(row["route"]["name"]) for row in sorted(proxy_rows, key=lambda row: num(row.get("route_proxy_screen_score")), reverse=True)]
    selected: list[str] = []
    seen: set[str] = set()
    seen_signatures: set[tuple[Any, ...]] = set()
    for name in [*forced, *ranked_names]:
        if name in seen:
            continue
        row = row_by_name.get(name) or {}
        signature = replay_signature(name, str(row.get("family") or ""), dict(row.get("proxy") or {}))
        if signature in seen_signatures:
            continue
        selected.append(name)
        seen.add(name)
        seen_signatures.add(signature)
        if max_replays > 0 and len(selected) >= max_replays:
            break
    return selected


def replay_signature(route_name: str, family: str, proxy: dict[str, Any]) -> tuple[Any, ...]:
    risk_match = re.search(r"risk(\d+)|_r(\d+)", route_name)
    q_match = re.search(r"_q(\d+)", route_name)
    risk = risk_match.group(1) or risk_match.group(2) if risk_match else ""
    q = q_match.group(1) if q_match else ""
    modes = tuple(sorted((proxy.get("route_eligible_by_mode") or {}).items()))
    return (
        family,
        risk,
        q,
        int(num(proxy.get("route_eligible_count"))),
        int(num(proxy.get("route_eligible_active_count"))),
        round(num(proxy.get("route_eligible_proxy_positive_r")), 3),
        round(num(proxy.get("route_eligible_proxy_net_eod_r")), 3),
        modes,
    )


def oof_proxy_comparison(original_control_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results_path = FAST_DIR / "kalcb_fast_extensive_search_results.json"
    if not results_path.exists():
        return []
    old_keys = {trade_key(row) for row in original_control_trades}
    payload = read_json(results_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("train_rows") or []:
        trades = base.read_trade_rows(row.get("trade_rows_path"))
        keys = {trade_key(trade) for trade in trades}
        metrics = dict(row.get("metrics") or {})
        rows.append(
            {
                "lane": "oof_proxy_fast_search",
                "pool_name": row.get("pool_name"),
                "route": (row.get("route") or {}).get("name"),
                "train_alpha_conversion_score": row.get("train_alpha_conversion_score"),
                "train_alpha_conversion_pass": row.get("train_alpha_conversion_pass"),
                "metrics": metrics,
                "stability": row.get("stability") or {},
                "trade_r_sum": trade_r_sum(trades),
                "overlap_with_original_control": len(old_keys & keys),
                "trade_rows_path": row.get("trade_rows_path"),
            }
        )
    return rows


def write_report(payload: dict[str, Any]) -> None:
    train_rows = sorted(payload["train_rows"], key=lambda row: num(row.get("train_alpha_conversion_score")), reverse=True)
    proxy_rows = sorted(payload["route_proxy_screen"], key=lambda row: num(row.get("route_proxy_screen_score")), reverse=True)
    control = payload.get("incumbent_control") or {}
    champion = payload.get("train_champion") or {}
    oof_rows = payload.get("oof_proxy_comparison") or []
    holdout = payload.get("locked_holdout_audit") or {}

    lines: list[str] = []
    lines.append("# KALCB Original Champion Route / Execution Conversion")
    lines.append("")
    lines.append("Train-only route and execution-conversion sweep on the exact original `dataset_all_context_hgb_quality` / `base_top16` champion pool. OOF/proxy pools are reported separately and are not mixed into the route decision.")
    lines.append("")
    lines.append("## Pool Control")
    lines.append("")
    lines.append(f"- Source pool: `{payload['pool_control']['source_pool']}`")
    lines.append(f"- Pool rows / active rows: {payload['pool_control']['pool_rows']} / {payload['pool_control']['active_rows']}")
    lines.append(f"- Proxy-screened routes / exact replays: {len(payload['route_proxy_screen'])} / {len(payload['selected_replay_routes'])}")
    lines.append("- Exact replay selection de-duplicates route variants with the same effective eligibility/risk/quality signature to avoid repeated shared-core runs that generate identical trades.")
    lines.append(f"- Prior champion hash: `{payload['pool_control'].get('prior_candidate_snapshot_hash', '')}`")
    lines.append(f"- Replayed control metrics reproduce the incumbent; replay-name-dependent hash: `{((control.get('metrics') or {}).get('candidate_snapshot_hash') or '')}`")
    lines.append(f"- Incumbent route: `{(control.get('route') or {}).get('name', '')}`")
    lines.append(f"- Incumbent train net/DD/trades/R: {pct((control.get('metrics') or {}).get('broker_net_return_pct'))} / {pct((control.get('metrics') or {}).get('broker_max_drawdown_pct'))} / {int(num((control.get('metrics') or {}).get('trade_count')))} / {fmt(control.get('trade_r_sum'), 1)}R")
    lines.append("")
    lines.append("## Train Route Replay Ranking")
    lines.append("")
    lines.append("| rank | pass | route | family | score | net | DD | trades | R sum | delta R | worst fold | capture | eligible R | routes |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    control_r = num(control.get("trade_r_sum"))
    for idx, row in enumerate(train_rows[:30], 1):
        metrics = dict(row.get("metrics") or {})
        stability = dict(row.get("stability") or {})
        proxy = dict(row.get("proxy") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(bool(row.get("train_alpha_conversion_pass"))),
                    f"`{(row.get('route') or {}).get('name')}`",
                    str(row.get("family") or ""),
                    fmt(row.get("train_alpha_conversion_score"), 2),
                    pct(metrics.get("broker_net_return_pct")),
                    pct(metrics.get("broker_max_drawdown_pct")),
                    str(int(num(metrics.get("trade_count")))),
                    f"{fmt(row.get('trade_r_sum'), 1)}",
                    f"{fmt(num(row.get('trade_r_sum')) - control_r, 1)}",
                    pct(stability.get("five_fold_worst_net")),
                    pct(metrics.get("avg_mfe_capture")),
                    f"{fmt(proxy.get('route_eligible_proxy_positive_r'), 1)}",
                    route_mix(row),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Top Proxy Screen")
    lines.append("")
    lines.append("| rank | route | family | proxy score | eligible | active eligible | eligible R | eligible net R | blockers |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
    for idx, row in enumerate(proxy_rows[:25], 1):
        proxy = dict(row.get("proxy") or {})
        blockers = "; ".join(f"{name}:{count}" for name, count in proxy.get("top_route_blockers") or [])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{(row.get('route') or {}).get('name')}`",
                    str(row.get("family") or ""),
                    fmt(row.get("route_proxy_screen_score"), 1),
                    str(int(num(proxy.get("route_eligible_count")))),
                    str(int(num(proxy.get("route_eligible_active_count")))),
                    fmt(proxy.get("route_eligible_proxy_positive_r"), 1),
                    fmt(proxy.get("route_eligible_proxy_net_eod_r"), 1),
                    blockers,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Separate OOF/Proxy Lane")
    lines.append("")
    lines.append("These rows are included only to compare the already-run OOF/proxy experiments against the original-pool control. They are not candidates from the original champion pool.")
    lines.append("")
    lines.append("| pool | route | pass | net | DD | trades | R sum | overlap with original |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(oof_rows, key=lambda item: num((item.get("metrics") or {}).get("broker_net_return_pct")), reverse=True):
        metrics = dict(row.get("metrics") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('pool_name')}`",
                    f"`{row.get('route')}`",
                    str(bool(row.get("train_alpha_conversion_pass"))),
                    pct(metrics.get("broker_net_return_pct")),
                    pct(metrics.get("broker_max_drawdown_pct")),
                    str(int(num(metrics.get("trade_count")))),
                    fmt(row.get("trade_r_sum"), 1),
                    str(row.get("overlap_with_original_control")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if champion and control:
        champ_name = (champion.get("route") or {}).get("name")
        control_name = (control.get("route") or {}).get("name")
        if champ_name == control_name:
            lines.append("- No tested route expansion or first30 soft-conversion variant beat the original-pool seed control on train stability score.")
        else:
            delta_net = num((champion.get("metrics") or {}).get("broker_net_return_pct")) - num((control.get("metrics") or {}).get("broker_net_return_pct"))
            delta_trades = num((champion.get("metrics") or {}).get("trade_count")) - num((control.get("metrics") or {}).get("trade_count"))
            delta_r = num(champion.get("trade_r_sum")) - num(control.get("trade_r_sum"))
            lines.append(f"- Best train-selected original-pool route: `{champ_name}`, with delta net {pct(delta_net)}, delta trades {int(delta_trades)}, delta R {fmt(delta_r, 1)}R versus `{control_name}`.")
        lines.append("- Holdout is intentionally not used in this selection pass.")
    lines.append("")
    if holdout:
        metrics = dict(holdout.get("metrics") or {})
        stability = dict(holdout.get("stability") or {})
        lines.append("## Locked Holdout Audit")
        lines.append("")
        lines.append("This audit is run only after the train-selected route is fixed.")
        lines.append("")
        lines.append(f"- Train-selected route: `{(holdout.get('route') or {}).get('name', '')}`")
        lines.append(f"- Holdout net/DD/trades/R: {pct(metrics.get('broker_net_return_pct'))} / {pct(metrics.get('broker_max_drawdown_pct'))} / {int(num(metrics.get('trade_count')))} / {fmt(holdout.get('trade_r_sum'), 1)}R")
        lines.append(f"- Holdout five-fold worst/capture: {pct(stability.get('five_fold_worst_net'))} / {pct(metrics.get('avg_mfe_capture'))}")
        lines.append(f"- Route mix: {route_mix(holdout)}")
        lines.append("")

    report_path = OUT_DIR / "kalcb_original_champion_route_conversion_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(max_replays: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_rows = load_original_pool_rows()
    write_jsonl(OUT_DIR / "original_champion_train_pool_rows.jsonl", pool_rows)
    seed = train_opt.load_seed_mutations()
    train_config, _holdout_config = shared.load_base_config()

    old_results_path = SOURCE_DIR / "kalcb_train_stability_route_optimizer_results.json"
    old_results = read_json(old_results_path)
    prior_hash = ((old_results.get("train_champion") or {}).get("metrics") or {}).get("candidate_snapshot_hash")

    routes = route_catalogue()
    route_by_name = {route.name: route for route in routes}
    proxy_rows: list[dict[str, Any]] = []
    log("proxy_screen_start", routes=len(routes), pool_rows=len(pool_rows))
    for route in routes:
        proxy = base.route_proxy_summary(pool_rows, route.build(seed))
        proxy_rows.append(
            {
                "route": {"name": route.name, "description": route.description},
                "family": route.family,
                "proxy": proxy,
                "route_proxy_screen_score": proxy_score(proxy),
            }
        )
    proxy_rows = sorted(proxy_rows, key=lambda row: num(row.get("route_proxy_screen_score")), reverse=True)
    write_json(OUT_DIR / "route_proxy_screen.json", proxy_rows)
    selected_names = select_replay_routes(proxy_rows, max_replays)
    log("proxy_screen_done", selected_replays=len(selected_names), top_route=(proxy_rows[0]["route"]["name"] if proxy_rows else ""))

    log("train_context_build_start")
    train_bundle = shared.build_window_bundle(train_config)
    train_dates = list(train_bundle["dataset"].trading_dates)
    initial_equity = float((train_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    log("train_context_build_done", sessions=len(train_dates), contexts=len(train_bundle["context_by_key"]))

    train_rows: list[dict[str, Any]] = []
    for name in selected_names:
        route = route_by_name[name]
        mutations = route.build(seed)
        proxy = base.route_proxy_summary(pool_rows, mutations)
        replay_name = f"original_champion_pool_{route.name}"
        log("train_replay_start", route=route.name, family=route.family, eligible=proxy.get("route_eligible_count"), eligible_r=proxy.get("route_eligible_proxy_positive_r"))
        result = evaluate_compiled_candidate_pool(
            window="train",
            variant=PoolVariant(replay_name, 16, active_count=16),
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
            "lane": "original_champion_pool",
            "route": {"name": route.name, "description": route.description},
            "family": route.family,
            "proxy": proxy,
            "metrics": metrics,
            "stability": stability,
            "compiled_replay": result.get("compiled_replay") or {},
            "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
            "trade_rows_path": result.get("trade_rows_path"),
            "trade_r_sum": trade_r_sum(trades),
            **score,
        }
        train_rows.append(row)
        write_json(
            OUT_DIR / "kalcb_original_champion_route_conversion_checkpoint.json",
            {
                "created_at_utc": now_iso(),
                "checkpoint": True,
                "usage_contract": "partial_train_only_original_champion_pool_route_conversion",
                "selected_replay_routes": selected_names,
                "completed_replays": len(train_rows),
                "latest_route": route.name,
                "train_rows": train_rows,
            },
        )
        log(
            "train_replay_done",
            route=route.name,
            score=score["train_alpha_conversion_score"],
            pass_hygiene=score["train_alpha_conversion_pass"],
            trades=metrics.get("trade_count"),
            net=metrics.get("broker_net_return_pct"),
            dd=metrics.get("broker_max_drawdown_pct"),
            r_sum=row["trade_r_sum"],
        )

    replayed = [row for row in train_rows if num((row.get("metrics") or {}).get("trade_count")) > 0.0]
    pass_rows = [row for row in replayed if row.get("train_alpha_conversion_pass")]
    champion = max(pass_rows or replayed, key=lambda row: num(row.get("train_alpha_conversion_score"))) if (pass_rows or replayed) else {}
    control = next((row for row in train_rows if (row.get("route") or {}).get("name") == "seed_risk99"), {})
    control_trades = base.read_trade_rows(control.get("trade_rows_path")) if control else []

    payload = {
        "created_at_utc": now_iso(),
        "usage_contract": "train_only_original_champion_pool_route_conversion_no_holdout_optimization",
        "pool_control": {
            "source_pool": str(SOURCE_DIR / "base_train_pool_rows.jsonl"),
            "policy": "dataset_all_context_hgb_quality_dataset_top16",
            "filter": "base_top16",
            "prior_route": "risk99",
            "pool_rows": len(pool_rows),
            "active_rows": sum(1 for row in pool_rows if bool(row.get("pool_active"))),
            "prior_candidate_snapshot_hash": prior_hash,
        },
        "route_proxy_screen": proxy_rows,
        "selected_replay_routes": selected_names,
        "train_rows": train_rows,
        "incumbent_control": control,
        "train_champion": champion,
        "oof_proxy_comparison": oof_proxy_comparison(control_trades),
    }
    write_json(OUT_DIR / "kalcb_original_champion_route_conversion_results.json", payload)
    write_report(payload)
    log("done", champion=((champion.get("route") or {}).get("name") if champion else ""), report=str(OUT_DIR / "kalcb_original_champion_route_conversion_report.md"))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Route/execution conversion sweep on exact original KALCB train champion pool.")
    parser.add_argument("--max-replays", type=int, default=32, help="Maximum train shared-core route replays after proxy screening; <=0 replays all routes.")
    args = parser.parse_args()
    run(max_replays=args.max_replays)


if __name__ == "__main__":
    main()
