from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROUTE_SCRIPT = REPO_ROOT / "scripts" / "kalcb_original_champion_route_conversion.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
SOURCE_DIR = ROUND_DIR / "r_capture_optimizer" / "original_champion_route_conversion"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "q3_incremental_guard_search"


def _load_route_module():
    spec = importlib.util.spec_from_file_location("kalcb_original_champion_route_conversion_module", ROUTE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load route module: {ROUTE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT_DIR = OUT_DIR
    return module


route_mod = _load_route_module()
base = route_mod.base
shared = route_mod.shared
train_opt = route_mod.train_opt

from backtests.strategies.kalcb.candidate_surfacing_recovery import PoolVariant, evaluate_compiled_candidate_pool  # noqa: E402


TRAIN_ROUTE = "first30_q3_risk50"
SEED_ROUTE = "seed_risk99"


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float | str


@dataclass(frozen=True)
class GuardSpec:
    name: str
    description: str
    conditions: tuple[Condition, ...]
    source: str


NUMERIC_FEATURES = [
    "pool_rank",
    "candidate_rank",
    "causal_rank_in_day",
    "frontier_rank",
    "first30_ret",
    "first30_vwap_ret",
    "first30_rel_volume",
    "first30_range_close_location",
    "first30_signal_bar_cpr",
    "cpr",
    "first30_range_atr",
    "first30_gap_retention_ratio",
    "first30_gap",
    "first30_open_drawdown",
    "first30_low_vs_prev_close",
    "first30_sector_leadership_pct",
    "first30_sector_relvol_ratio",
    "first30_sector_ret_spread",
    "first30_quality_pct",
    "sector_intraday_score_pct",
    "sector_intraday_breadth",
    "sector_intraday_participation",
    "sector_intraday_ret",
    "sector_intraday_rel_volume",
    "sector_daily_score_pct",
    "sector_daily_breadth_20d",
    "sector_daily_participation",
    "sector_daily_ret_5d",
    "sector_daily_ret_20d",
    "daily_close20_loc",
    "daily_close60_loc",
    "daily_momentum_pct",
    "daily_return_5d",
    "daily_return_20d",
    "daily_return_60d",
    "daily_volume_ratio_20d",
    "daily_adv20_krw_log",
    "daily_acceleration_5v20",
    "daily_sector_alignment_pct",
    "flow_score",
    "accumulation_score",
    "momentum_score",
    "continuation_joint_quality_pct",
    "bar_rvol",
    "stock_sector_daily_ret5_spread",
    "stock_sector_daily_ret20_spread",
]

DERIVED_FEATURES = [
    "ix_intraday_confirm",
    "ix_trend_confirm",
    "ix_sector_confirm",
    "ix_gap_retention_quality",
    "ix_route_efficiency",
    "ix_liquidity_momentum",
    "ix_extension_risk",
    "ix_q3_vote_count",
]


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path | str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def trade_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("entry_date") or row.get("trade_date") or "")[:10],
        str(row.get("symbol") or "").zfill(6),
        str(row.get("entry_route_mode") or row.get("entry_route") or ""),
    )


def date_symbol_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("entry_date") or row.get("trade_date") or "")[:10],
        str(row.get("symbol") or "").zfill(6),
    )


def parse_day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def clean_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out[:96] or "guard"


def clipped(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def augment_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    first30_ret = num(out.get("first30_ret"))
    vwap = num(out.get("first30_vwap_ret"))
    cpr = num(out.get("first30_signal_bar_cpr") if out.get("first30_signal_bar_cpr") is not None else out.get("cpr"))
    rvol = num(out.get("first30_rel_volume") if out.get("first30_rel_volume") is not None else out.get("bar_rvol"))
    close_loc = num(out.get("first30_range_close_location"))
    range_atr = num(out.get("first30_range_atr"))
    sector_intraday = num(out.get("sector_intraday_score_pct")) / 100.0
    sector_daily = num(out.get("sector_daily_score_pct")) / 100.0
    daily_close20 = num(out.get("daily_close20_loc"))
    daily_mom = num(out.get("daily_momentum_pct")) / 100.0
    daily_ret20 = num(out.get("daily_return_20d"))
    daily_ret60 = num(out.get("daily_return_60d"))
    volume20 = math.log1p(max(num(out.get("daily_volume_ratio_20d")), 0.0))
    gap_retention = num(out.get("first30_gap_retention_ratio"))
    low_vs_prev = num(out.get("first30_low_vs_prev_close"))
    flow = num(out.get("flow_score"))
    accumulation = num(out.get("accumulation_score"))
    continuation = num(out.get("continuation_joint_quality_pct")) / 100.0

    out["ix_q3_vote_count"] = float(
        sum(
            [
                first30_ret >= -0.006,
                vwap >= -0.012,
                cpr >= 0.55,
                rvol >= 0.75,
                range_atr >= 0.30,
            ]
        )
    )
    out["ix_intraday_confirm"] = (
        clipped(first30_ret / 0.04, -1.0, 1.0)
        + clipped(vwap / 0.015, -1.0, 1.0)
        + clipped(cpr, 0.0, 1.0)
        + clipped(math.log1p(max(rvol, 0.0)) / math.log(8.0), 0.0, 1.5)
        + clipped(close_loc, 0.0, 1.0)
        + clipped(sector_intraday, 0.0, 1.0)
    )
    out["ix_trend_confirm"] = (
        clipped(daily_close20, 0.0, 1.0)
        + clipped(daily_mom, 0.0, 1.2)
        + clipped(daily_ret20 / 0.20, -1.0, 1.5)
        + clipped(daily_ret60 / 0.40, -1.0, 1.5)
        + clipped(volume20 / math.log(5.0), 0.0, 1.5)
    )
    out["ix_sector_confirm"] = (
        clipped(sector_intraday, 0.0, 1.0)
        + clipped(sector_daily, 0.0, 1.0)
        + clipped(num(out.get("sector_intraday_breadth")), 0.0, 1.0)
        + clipped(num(out.get("sector_daily_breadth_20d")), 0.0, 1.0)
    )
    out["ix_gap_retention_quality"] = (
        clipped(gap_retention, 0.0, 1.5)
        + clipped(low_vs_prev / 0.03, -1.0, 1.0)
        + clipped(vwap / 0.015, -1.0, 1.0)
        - clipped(-num(out.get("first30_open_drawdown")) / 0.02, 0.0, 2.0)
    )
    out["ix_route_efficiency"] = clipped(close_loc, 0.0, 1.0) - 0.18 * max(range_atr - 1.2, 0.0) + clipped(vwap / 0.015, -1.0, 1.0)
    out["ix_liquidity_momentum"] = clipped(math.log1p(max(rvol, 0.0)) / math.log(8.0), 0.0, 1.5) + clipped(volume20 / math.log(5.0), 0.0, 1.5) + clipped(first30_ret / 0.04, -1.0, 1.0)
    out["ix_extension_risk"] = max(range_atr - 1.6, 0.0) + max(first30_ret - 0.05, 0.0) * 10.0 + max(num(out.get("first30_gap")) - 0.05, 0.0) * 8.0
    out["ix_sector_flow_stack"] = clipped(sector_intraday, 0.0, 1.0) + clipped(sector_daily, 0.0, 1.0) + clipped(flow, -1.0, 1.0) + clipped(accumulation, -1.0, 1.0) + clipped(continuation, 0.0, 1.0)
    return out


def condition_value(row: dict[str, Any], feature: str) -> float | str:
    value = row.get(feature)
    if feature == "sector":
        return str(value or "UNKNOWN")
    return num(value)


def condition_pass(row: dict[str, Any], condition: Condition) -> bool:
    value = condition_value(row, condition.feature)
    if condition.op == ">=":
        return num(value) >= num(condition.threshold)
    if condition.op == "<=":
        return num(value) <= num(condition.threshold)
    if condition.op == "!=":
        return str(value) != str(condition.threshold)
    if condition.op == "==":
        return str(value) == str(condition.threshold)
    raise ValueError(f"Unknown op: {condition.op}")


def guard_pass(row: dict[str, Any], guard: GuardSpec) -> bool:
    augmented = augment_row(row)
    return all(condition_pass(augmented, condition) for condition in guard.conditions)


def guard_label(guard: GuardSpec) -> str:
    parts = []
    for condition in guard.conditions:
        threshold = condition.threshold if isinstance(condition.threshold, str) else f"{condition.threshold:.5g}"
        parts.append(f"{condition.feature}{condition.op}{threshold}")
    return " & ".join(parts) if parts else "no_guard"


def load_route_results() -> dict[str, Any]:
    return read_json(SOURCE_DIR / "kalcb_original_champion_route_conversion_results.json")


def route_row(results: dict[str, Any], route_name: str) -> dict[str, Any]:
    for row in results.get("train_rows") or []:
        if (row.get("route") or {}).get("name") == route_name:
            return row
    raise KeyError(route_name)


def summarize_trade_group(rows: list[dict[str, Any]], session_dates: list[date], initial_equity: float) -> dict[str, Any]:
    stability = base.stability_metrics(rows, session_dates, initial_equity)
    total_net = sum(num(row.get("net_pnl")) for row in rows) / max(initial_equity, 1.0)
    total_r = sum(num(row.get("r")) for row in rows)
    avg_r = total_r / max(len(rows), 1)
    win_share = sum(1 for row in rows if num(row.get("r")) > 0.0) / max(len(rows), 1)
    return {
        "trades": len(rows),
        "net_return_from_rows": total_net,
        "r_sum": total_r,
        "avg_r": avg_r,
        "win_share": win_share,
        "stability": stability,
        "by_sector": aggregate_by(rows, "sector"),
        "by_exit": aggregate_by(rows, "exit_reason"),
        "by_month": aggregate_by(rows, lambda row: str(row.get("entry_date") or "")[:7]),
    }


def aggregate_by(rows: list[dict[str, Any]], key: str | Any) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(key(row) if callable(key) else row.get(key) or "UNKNOWN")
        grouped.setdefault(label, []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for label, items in grouped.items():
        out[label] = {
            "trades": len(items),
            "r_sum": sum(num(row.get("r")) for row in items),
            "avg_r": sum(num(row.get("r")) for row in items) / max(len(items), 1),
            "win_share": sum(1 for row in items if num(row.get("r")) > 0.0) / max(len(items), 1),
        }
    return dict(sorted(out.items(), key=lambda item: item[1]["r_sum"]))


def feature_diagnostics(incremental: list[dict[str, Any]], seed_overlap: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inc_aug = [augment_row(row) for row in incremental]
    seed_aug = [augment_row(row) for row in seed_overlap]
    for feature in [*NUMERIC_FEATURES, *DERIVED_FEATURES]:
        inc_values = np.array([num(row.get(feature)) for row in inc_aug if math.isfinite(num(row.get(feature)))], dtype=float)
        seed_values = np.array([num(row.get(feature)) for row in seed_aug if math.isfinite(num(row.get(feature)))], dtype=float)
        if len(inc_values) < 10:
            continue
        inc_r = np.array([num(row.get("r")) for row in inc_aug], dtype=float)
        values = np.array([num(row.get(feature)) for row in inc_aug], dtype=float)
        if np.nanstd(values) <= 1e-12:
            continue
        corr = float(np.corrcoef(np.nan_to_num(values), np.nan_to_num(inc_r))[0, 1]) if len(values) > 2 else 0.0
        rows.append(
            {
                "feature": feature,
                "incremental_median": float(np.nanmedian(inc_values)),
                "incremental_p25": float(np.nanpercentile(inc_values, 25)),
                "incremental_p75": float(np.nanpercentile(inc_values, 75)),
                "seed_overlap_median": float(np.nanmedian(seed_values)) if len(seed_values) else None,
                "corr_with_incremental_r": corr if math.isfinite(corr) else 0.0,
            }
        )
    return sorted(rows, key=lambda row: abs(num(row.get("corr_with_incremental_r"))), reverse=True)


def proxy_guard_metrics(
    guard: GuardSpec,
    q3_rows: list[dict[str, Any]],
    seed_keys: set[tuple[str, str]],
    session_dates: list[date],
    initial_equity: float,
    seed_r: float,
    q3_r: float,
) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    incremental_kept = 0
    incremental_total = 0
    for row in q3_rows:
        key = trade_key(row)
        if date_symbol_key(row) in seed_keys:
            kept.append(row)
            continue
        incremental_total += 1
        if guard_pass(row, guard):
            kept.append(row)
            incremental_kept += 1
    stability = base.stability_metrics(kept, session_dates, initial_equity)
    r_sum = sum(num(row.get("r")) for row in kept)
    net_from_rows = sum(num(row.get("net_pnl")) for row in kept) / max(initial_equity, 1.0)
    dd_proxy = drawdown_from_trades(kept, initial_equity)
    added_r_total = max(q3_r - seed_r, 1e-9)
    added_r_kept = r_sum - seed_r
    retention = added_r_kept / added_r_total
    removed = [row for row in q3_rows if date_symbol_key(row) not in seed_keys and row not in kept]
    removed_bad_r = -sum(min(num(row.get("r")), 0.0) for row in removed)
    return {
        "guard": guard_to_dict(guard),
        "proxy_trades": len(kept),
        "proxy_incremental_kept": incremental_kept,
        "proxy_incremental_total": incremental_total,
        "proxy_net_from_rows": net_from_rows,
        "proxy_drawdown_from_rows": dd_proxy,
        "proxy_r_sum": r_sum,
        "proxy_added_r_retention": retention,
        "proxy_worst_fold": stability.get("five_fold_worst_net"),
        "proxy_negative_folds": stability.get("five_fold_negative_count"),
        "proxy_avg_r": r_sum / max(len(kept), 1),
        "proxy_removed_bad_r": removed_bad_r,
        "proxy_score": proxy_score(net_from_rows, dd_proxy, len(kept), r_sum, retention, num(stability.get("five_fold_worst_net")), removed_bad_r),
    }


def proxy_score(net: float, dd: float, trades: int, r_sum: float, retention: float, worst_fold: float, removed_bad_r: float) -> float:
    return (
        100.0 * net
        + 0.050 * r_sum
        + 0.08 * min(trades, 240)
        + 18.0 * max(worst_fold, -0.05)
        + 20.0 * min(max(retention, 0.0), 1.25)
        + 0.15 * removed_bad_r
        - 500.0 * max(dd - 0.07, 0.0)
        - 45.0 * max(0.65 - retention, 0.0)
    )


def drawdown_from_trades(rows: list[dict[str, Any]], initial_equity: float) -> float:
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    ordered = sorted(rows, key=lambda row: (str(row.get("entry_date") or ""), str(row.get("entry_time") or ""), str(row.get("symbol") or "")))
    for row in ordered:
        equity += num(row.get("net_pnl"))
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def guard_to_dict(guard: GuardSpec) -> dict[str, Any]:
    return {
        "name": guard.name,
        "description": guard.description,
        "source": guard.source,
        "conditions": [condition.__dict__ for condition in guard.conditions],
    }


def guard_from_dict(row: dict[str, Any]) -> GuardSpec:
    return GuardSpec(
        name=str(row["name"]),
        description=str(row.get("description") or row["name"]),
        source=str(row.get("source") or "unknown"),
        conditions=tuple(Condition(str(item["feature"]), str(item["op"]), item["threshold"]) for item in row.get("conditions") or []),
    )


def candidate_guards(incremental: list[dict[str, Any]]) -> list[GuardSpec]:
    augmented = [augment_row(row) for row in incremental]
    guards: list[GuardSpec] = [GuardSpec("no_guard", "No incremental guard.", tuple(), "baseline")]
    singles: list[GuardSpec] = []
    quantiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for feature in [*NUMERIC_FEATURES, *DERIVED_FEATURES]:
        values = np.array([num(row.get(feature)) for row in augmented], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 12 or np.nanstd(values) <= 1e-12:
            continue
        for threshold in sorted({float(np.nanquantile(values, q)) for q in quantiles}):
            singles.append(GuardSpec(clean_name(f"{feature}_ge_{threshold:.5g}"), f"{feature} >= {threshold:.5g}", (Condition(feature, ">=", threshold),), "single_threshold"))
            singles.append(GuardSpec(clean_name(f"{feature}_le_{threshold:.5g}"), f"{feature} <= {threshold:.5g}", (Condition(feature, "<=", threshold),), "single_threshold"))

    sectors: dict[str, list[dict[str, Any]]] = {}
    for row in incremental:
        sectors.setdefault(str(row.get("sector") or "UNKNOWN"), []).append(row)
    for sector, rows in sectors.items():
        if len(rows) >= 4:
            r_sum = sum(num(row.get("r")) for row in rows)
            if r_sum < 0.0 or r_sum / len(rows) < 2.0:
                singles.append(GuardSpec(clean_name(f"sector_not_{sector}"), f"sector != {sector}", (Condition("sector", "!=", sector),), "sector_exclusion"))

    return [*guards, *singles]


def shortlist_guards(
    q3_rows: list[dict[str, Any]],
    seed_keys: set[tuple[str, str]],
    session_dates: list[date],
    initial_equity: float,
    seed_r: float,
    q3_r: float,
    max_proxy: int,
) -> tuple[list[dict[str, Any]], list[GuardSpec]]:
    incremental = [row for row in q3_rows if date_symbol_key(row) not in seed_keys]
    singles = candidate_guards(incremental)
    proxy_rows: list[dict[str, Any]] = []
    for guard in singles:
        metrics = proxy_guard_metrics(guard, q3_rows, seed_keys, session_dates, initial_equity, seed_r, q3_r)
        if metrics["proxy_incremental_kept"] >= 8 and metrics["proxy_added_r_retention"] >= 0.25:
            proxy_rows.append(metrics)

    ranked_singles = sorted(proxy_rows, key=lambda row: num(row.get("proxy_score")), reverse=True)
    top_single_guards = [guard_from_dict(row["guard"]) for row in ranked_singles[:60] if (row.get("guard") or {}).get("name") != "no_guard"]
    seen: set[str] = {json.dumps(row["guard"]["conditions"], sort_keys=True, default=str) for row in proxy_rows}
    for i, left in enumerate(top_single_guards):
        for right in top_single_guards[i + 1 :]:
            conditions = tuple(dict.fromkeys([*left.conditions, *right.conditions]))
            if len(conditions) != 2:
                continue
            name = clean_name(f"{left.name}_and_{right.name}")
            guard = GuardSpec(name, f"{guard_label(left)} & {guard_label(right)}", conditions, "pairwise_threshold")
            signature = json.dumps([condition.__dict__ for condition in conditions], sort_keys=True, default=str)
            if signature in seen:
                continue
            seen.add(signature)
            metrics = proxy_guard_metrics(guard, q3_rows, seed_keys, session_dates, initial_equity, seed_r, q3_r)
            if metrics["proxy_incremental_kept"] >= 8 and metrics["proxy_added_r_retention"] >= 0.25:
                proxy_rows.append(metrics)

    proxy_rows = sorted(proxy_rows, key=lambda row: num(row.get("proxy_score")), reverse=True)
    finalist_guards: list[GuardSpec] = []
    finalist_seen: set[str] = set()
    forced = ["no_guard"]
    for row in proxy_rows:
        guard = guard_from_dict(row["guard"])
        if guard.name in forced or len(finalist_guards) < max_proxy:
            signature = json.dumps([condition.__dict__ for condition in guard.conditions], sort_keys=True, default=str)
            if signature not in finalist_seen:
                finalist_guards.append(guard)
                finalist_seen.add(signature)
        if len(finalist_guards) >= max_proxy:
            break
    return proxy_rows, finalist_guards


def route_eligibility(row: dict[str, Any], mutations: dict[str, Any]) -> bool:
    meta = base._pool_route_meta(row)
    for route in base._configured_entry_routes(mutations):
        passed, _reason = base._route_candidate_passes(route, mutations, meta)
        if passed:
            return True
    return False


def guarded_pool_rows(pool_rows: list[dict[str, Any]], guard: GuardSpec, seed_mutations: dict[str, Any], q3_mutations: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts = {
        "input_rows": len(pool_rows),
        "seed_eligible_rows": 0,
        "q3_eligible_rows": 0,
        "q3_extra_guard_pass_rows": 0,
        "kept_rows": 0,
    }
    for row in pool_rows:
        seed_ok = route_eligibility(row, seed_mutations)
        q3_ok = route_eligibility(row, q3_mutations)
        if seed_ok:
            counts["seed_eligible_rows"] += 1
        if q3_ok:
            counts["q3_eligible_rows"] += 1
        keep_extra = bool(q3_ok and not seed_ok and guard_pass(row, guard))
        if keep_extra:
            counts["q3_extra_guard_pass_rows"] += 1
        if seed_ok or keep_extra:
            out.append(dict(row))
    counts["kept_rows"] = len(out)
    return out, counts


def guard_replay_score(metrics: dict[str, Any], stability: dict[str, Any], r_sum: float, seed_metrics: dict[str, Any], q3_metrics: dict[str, Any], seed_r: float, q3_r: float) -> dict[str, Any]:
    net = num(metrics.get("broker_net_return_pct"))
    dd = num(metrics.get("broker_max_drawdown_pct"))
    trades = num(metrics.get("trade_count"))
    capture = num(metrics.get("avg_mfe_capture"))
    worst5 = num(stability.get("five_fold_worst_net"))
    neg_folds = num(stability.get("five_fold_negative_count"))
    added_r_total = max(q3_r - seed_r, 1e-9)
    retention = (r_sum - seed_r) / added_r_total
    q3_dd = num(q3_metrics.get("broker_max_drawdown_pct"))
    seed_net = num(seed_metrics.get("broker_net_return_pct"))
    score = (
        100.0 * net
        + 0.035 * r_sum
        + 0.07 * min(trades, 260.0)
        + 20.0 * worst5
        + 12.0 * capture
        + 26.0 * min(max(retention, 0.0), 1.2)
        + 140.0 * max(q3_dd - dd, 0.0)
        - 450.0 * max(dd - 0.07, 0.0)
        - 10.0 * neg_folds
    )
    pass_guard = (
        net > seed_net
        and dd <= 0.0725
        and trades >= 170.0
        and worst5 >= 0.035
        and neg_folds == 0.0
        and retention >= 0.65
        and capture >= 0.40
    )
    return {
        "train_guard_score": score,
        "train_guard_pass": pass_guard,
        "added_r_retention": retention,
        "dd_improvement_vs_q3": q3_dd - dd,
    }


def run_exact_replays(
    finalist_guards: list[GuardSpec],
    pool_rows: list[dict[str, Any]],
    train_config: dict[str, Any],
    seed_mutations: dict[str, Any],
    q3_mutations: dict[str, Any],
    train_dates: list[date],
    initial_equity: float,
    seed_row: dict[str, Any],
    q3_row: dict[str, Any],
    max_exact: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    log("train_context_build_start")
    train_bundle = shared.build_window_bundle(train_config)
    train_dates = list(train_bundle["dataset"].trading_dates)
    initial_equity = float((train_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    log("train_context_build_done", sessions=len(train_dates), contexts=len(train_bundle["context_by_key"]))

    selected = finalist_guards[:max_exact] if max_exact > 0 else finalist_guards
    for guard in selected:
        filtered_rows, guard_counts = guarded_pool_rows(pool_rows, guard, seed_mutations, q3_mutations)
        replay_name = f"q3_guard_{guard.name}"
        write_jsonl(OUT_DIR / f"pool_rows_train_{replay_name}.jsonl", filtered_rows)
        log("train_replay_start", guard=guard.name, rows=len(filtered_rows), q3_extra=guard_counts["q3_extra_guard_pass_rows"])
        result = evaluate_compiled_candidate_pool(
            window="train",
            variant=PoolVariant(replay_name, 16, active_count=16),
            config=train_config,
            dataset=train_bundle["dataset"],
            context_by_key=train_bundle["context_by_key"],
            pool_rows=filtered_rows,
            seed_mutations=q3_mutations,
            output_dir=OUT_DIR,
            replay_name=replay_name,
        )
        metrics = dict(result.get("metrics") or {})
        trades = base.read_trade_rows(result.get("trade_rows_path"))
        stability = base.stability_metrics(trades, train_dates, initial_equity)
        r_sum = sum(num(row.get("r")) for row in trades)
        score = guard_replay_score(metrics, stability, r_sum, seed_row["metrics"], q3_row["metrics"], num(seed_row["trade_r_sum"]), num(q3_row["trade_r_sum"]))
        row = {
            "guard": guard_to_dict(guard),
            "guard_label": guard_label(guard),
            "guard_counts": guard_counts,
            "metrics": metrics,
            "stability": stability,
            "trade_r_sum": r_sum,
            "trade_rows_path": result.get("trade_rows_path"),
            "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
            **score,
        }
        rows.append(row)
        write_json(OUT_DIR / "kalcb_q3_incremental_guard_search_checkpoint.json", {"created_at_utc": now_iso(), "completed_replays": len(rows), "train_rows": rows})
        log(
            "train_replay_done",
            guard=guard.name,
            pass_hygiene=score["train_guard_pass"],
            score=score["train_guard_score"],
            net=metrics.get("broker_net_return_pct"),
            dd=metrics.get("broker_max_drawdown_pct"),
            trades=metrics.get("trade_count"),
            r_sum=r_sum,
            retention=score["added_r_retention"],
        )
    return rows


def load_session_context(config: dict[str, Any]) -> tuple[list[date], float]:
    bundle = shared.build_window_bundle(config)
    dates = list(bundle["dataset"].trading_dates)
    equity = float((bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    return dates, equity


def approximate_session_context_from_trades(rows: list[dict[str, Any]]) -> tuple[list[date], float]:
    dates = sorted({day for row in rows if (day := parse_day(row.get("entry_date"))) is not None})
    return dates, 100_000_000.0


def build_holdout_pool_and_replay(champion: dict[str, Any], seed_mutations: dict[str, Any], q3_mutations: dict[str, Any], holdout_config: dict[str, Any]) -> dict[str, Any]:
    guard = guard_from_dict(champion["guard"])
    log("locked_holdout_audit_start", guard=guard.name)
    df = base.opt.read_pipeline()
    train = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    models, _train_scores = train_opt.fit_train_only_models(train)
    policy = train_opt.SelectionPolicy(
        name="dataset_all_context_hgb_quality_dataset_top16",
        label="dataset_all_context_hgb_quality",
        scope="dataset",
        budget="top16",
        active_count=16,
        pool_size=16,
        source="train_champion_family",
    )
    model_info = models[policy.label]
    holdout_score = np.asarray(model_info["model"].predict(holdout[model_info["features"]]), dtype=float)
    feature_rows_holdout = shared.load_feature_rows("holdout")
    holdout_pool = train_opt.selected_pool_rows(holdout, holdout_score, policy, feature_rows_holdout)
    filtered_rows, guard_counts = guarded_pool_rows(holdout_pool, guard, seed_mutations, q3_mutations)
    write_jsonl(OUT_DIR / f"pool_rows_locked_holdout_{guard.name}.jsonl", filtered_rows)
    log("locked_holdout_context_build_start", rows=len(filtered_rows), q3_extra=guard_counts["q3_extra_guard_pass_rows"])
    holdout_bundle = shared.build_window_bundle(holdout_config)
    holdout_dates = list(holdout_bundle["dataset"].trading_dates)
    holdout_equity = float((holdout_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    replay_name = f"q3_guard_{guard.name}_locked_holdout"
    result = evaluate_compiled_candidate_pool(
        window="holdout_locked_audit",
        variant=PoolVariant(replay_name, 16, active_count=16),
        config=holdout_config,
        dataset=holdout_bundle["dataset"],
        context_by_key=holdout_bundle["context_by_key"],
        pool_rows=filtered_rows,
        seed_mutations=q3_mutations,
        output_dir=OUT_DIR,
        replay_name=replay_name,
    )
    metrics = dict(result.get("metrics") or {})
    trades = base.read_trade_rows(result.get("trade_rows_path"))
    audit = {
        "selection_basis": "locked_after_train_selected_q3_guard_no_holdout_optimization",
        "guard": guard_to_dict(guard),
        "guard_counts": guard_counts,
        "metrics": metrics,
        "stability": base.stability_metrics(trades, holdout_dates, holdout_equity),
        "trade_r_sum": sum(num(row.get("r")) for row in trades),
        "trade_rows_path": result.get("trade_rows_path"),
        "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
    }
    log("locked_holdout_audit_done", guard=guard.name, net=metrics.get("broker_net_return_pct"), dd=metrics.get("broker_max_drawdown_pct"), trades=metrics.get("trade_count"), r_sum=audit["trade_r_sum"])
    return audit


def route_mix(row: dict[str, Any]) -> str:
    summary = dict(row.get("entry_route_mode_summary") or {})
    if summary:
        return "; ".join(f"{mode}:{int(num((data or {}).get('trades')))}" for mode, data in summary.items())
    counts = dict((row.get("stability") or {}).get("entry_route_mode_counts") or {})
    return "; ".join(f"{mode}:{count}" for mode, count in counts.items())


def write_report(payload: dict[str, Any]) -> None:
    seed = payload["controls"]["seed"]
    q3 = payload["controls"]["q3"]
    incremental = payload["incremental_summary"]
    train_rows = sorted(payload.get("train_rows") or [], key=lambda row: num(row.get("train_guard_score")), reverse=True)
    pass_rows = [row for row in train_rows if row.get("train_guard_pass")]
    champion = payload.get("train_champion") or {}
    holdout = payload.get("locked_holdout_audit") or {}

    lines: list[str] = []
    lines.append("# KALCB Q3 Incremental Guard Search")
    lines.append("")
    lines.append("Train-only guard search for the incremental trades added by `first30_q3_risk50` on the exact original champion pool. Seed-eligible candidates stay available; guards only decide which q3-expanded candidates remain available.")
    lines.append("")
    lines.append("## Incremental Trade Profile")
    lines.append("")
    lines.append(f"- Seed control: {pct(seed['metrics'].get('broker_net_return_pct'))} net, {pct(seed['metrics'].get('broker_max_drawdown_pct'))} DD, {int(num(seed['metrics'].get('trade_count')))} trades, {fmt(seed.get('trade_r_sum'), 1)}R.")
    lines.append(f"- Q3 route: {pct(q3['metrics'].get('broker_net_return_pct'))} net, {pct(q3['metrics'].get('broker_max_drawdown_pct'))} DD, {int(num(q3['metrics'].get('trade_count')))} trades, {fmt(q3.get('trade_r_sum'), 1)}R.")
    lines.append(f"- Incremental q3 trades: {incremental['trades']} trades, {fmt(incremental['r_sum'], 1)}R, avg {fmt(incremental['avg_r'], 2)}R/trade, win share {pct(incremental['win_share'])}.")
    lines.append("")
    lines.append("Worst incremental sectors by R:")
    for sector, row in list(incremental["by_sector"].items())[:8]:
        lines.append(f"- `{sector}`: {row['trades']} trades, {fmt(row['r_sum'], 1)}R, avg {fmt(row['avg_r'], 2)}R.")
    lines.append("")
    lines.append("## Exact Train Replays")
    lines.append("")
    lines.append(f"- Proxy guard candidates screened: {len(payload.get('proxy_ranked') or [])}.")
    lines.append(f"- Exact guard replays: {len(payload.get('train_rows') or [])}.")
    lines.append("")
    lines.append("| rank | pass | guard | score | net | DD | trades | R sum | added R kept | worst fold | capture | q3 extra rows | routes |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for idx, row in enumerate(train_rows[:25], 1):
        metrics = dict(row.get("metrics") or {})
        stability = dict(row.get("stability") or {})
        counts = dict(row.get("guard_counts") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(bool(row.get("train_guard_pass"))),
                    f"`{row.get('guard_label')}`",
                    fmt(row.get("train_guard_score"), 2),
                    pct(metrics.get("broker_net_return_pct")),
                    pct(metrics.get("broker_max_drawdown_pct")),
                    str(int(num(metrics.get("trade_count")))),
                    fmt(row.get("trade_r_sum"), 1),
                    pct(row.get("added_r_retention")),
                    pct(stability.get("five_fold_worst_net")),
                    pct(metrics.get("avg_mfe_capture")),
                    str(int(num(counts.get("q3_extra_guard_pass_rows")))),
                    route_mix(row),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Train Selection")
    lines.append("")
    if champion:
        m = dict(champion.get("metrics") or {})
        lines.append(f"- Train-selected guard: `{champion.get('guard_label')}`.")
        lines.append(f"- Train result: {pct(m.get('broker_net_return_pct'))} net, {pct(m.get('broker_max_drawdown_pct'))} DD, {int(num(m.get('trade_count')))} trades, {fmt(champion.get('trade_r_sum'), 1)}R.")
        lines.append(f"- Added-R retention vs q3: {pct(champion.get('added_r_retention'))}; DD improvement vs unguarded q3: {pct(champion.get('dd_improvement_vs_q3'))}.")
        if pass_rows:
            lines.append("- Selection basis: highest train guard score among guards passing the train-only stability contract.")
        else:
            lines.append("- No guard passed the stricter stability contract; champion is the highest-scoring replay for diagnosis only.")
    lines.append("")
    if holdout:
        m = dict(holdout.get("metrics") or {})
        s = dict(holdout.get("stability") or {})
        lines.append("## Locked Holdout Audit")
        lines.append("")
        lines.append("Run only after the train guard was selected.")
        lines.append("")
        lines.append(f"- Holdout result: {pct(m.get('broker_net_return_pct'))} net, {pct(m.get('broker_max_drawdown_pct'))} DD, {int(num(m.get('trade_count')))} trades, {fmt(holdout.get('trade_r_sum'), 1)}R.")
        lines.append(f"- Holdout worst fold/capture: {pct(s.get('five_fold_worst_net'))} / {pct(m.get('avg_mfe_capture'))}.")
        lines.append(f"- Route mix: {route_mix(holdout)}.")
        lines.append("")
    report_path = OUT_DIR / "kalcb_q3_incremental_guard_search_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(max_proxy: int, max_exact: int, locked_holdout: bool) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_route_results()
    seed_row = route_row(results, SEED_ROUTE)
    q3_row = route_row(results, TRAIN_ROUTE)
    seed_trades = read_jsonl(seed_row.get("trade_rows_path"))
    q3_trades = read_jsonl(q3_row.get("trade_rows_path"))
    seed_keys = {date_symbol_key(row) for row in seed_trades}
    q3_keys = {date_symbol_key(row) for row in q3_trades}
    incremental_trades = [row for row in q3_trades if date_symbol_key(row) not in seed_keys]
    seed_overlap = [row for row in q3_trades if date_symbol_key(row) in seed_keys]
    train_config, holdout_config = shared.load_base_config()
    train_dates, initial_equity = approximate_session_context_from_trades(q3_trades)

    incremental_summary = summarize_trade_group(incremental_trades, train_dates, initial_equity)
    diagnostics = feature_diagnostics(incremental_trades, seed_overlap)
    log("incremental_profile_done", seed_trades=len(seed_trades), q3_trades=len(q3_trades), overlap=len(seed_keys & q3_keys), incremental=len(incremental_trades), incremental_r=incremental_summary["r_sum"])

    proxy_ranked, finalists = shortlist_guards(
        q3_trades,
        seed_keys,
        train_dates,
        initial_equity,
        num(seed_row.get("trade_r_sum")),
        num(q3_row.get("trade_r_sum")),
        max_proxy=max_proxy,
    )
    write_json(OUT_DIR / "guard_proxy_ranked.json", proxy_ranked)
    write_json(OUT_DIR / "incremental_feature_diagnostics.json", diagnostics)
    log("guard_proxy_screen_done", proxy_candidates=len(proxy_ranked), finalists=len(finalists), top=(proxy_ranked[0]["guard"]["name"] if proxy_ranked else ""))

    seed_mutations = train_opt.load_seed_mutations()
    q3_route = next(route for route in route_mod.route_catalogue() if route.name == TRAIN_ROUTE)
    q3_mutations = q3_route.build(seed_mutations)
    pool_rows = read_jsonl(SOURCE_DIR / "original_champion_train_pool_rows.jsonl")
    train_rows = run_exact_replays(finalists, pool_rows, train_config, seed_mutations, q3_mutations, train_dates, initial_equity, seed_row, q3_row, max_exact=max_exact)
    pass_rows = [row for row in train_rows if row.get("train_guard_pass")]
    champion = max(pass_rows or train_rows, key=lambda row: num(row.get("train_guard_score"))) if train_rows else {}
    payload = {
        "created_at_utc": now_iso(),
        "usage_contract": "train_only_q3_incremental_stability_guard_search_holdout_locked_after_train_selection",
        "controls": {
            "seed": seed_row,
            "q3": q3_row,
            "seed_trade_count": len(seed_trades),
            "q3_trade_count": len(q3_trades),
            "overlap_trade_count": len(seed_keys & q3_keys),
        },
        "incremental_summary": incremental_summary,
        "incremental_feature_diagnostics": diagnostics,
        "proxy_ranked": proxy_ranked,
        "finalist_guards": [guard_to_dict(guard) for guard in finalists],
        "train_rows": train_rows,
        "train_champion": champion,
    }
    if locked_holdout and champion:
        payload["locked_holdout_audit"] = build_holdout_pool_and_replay(champion, seed_mutations, q3_mutations, holdout_config)
    write_json(OUT_DIR / "kalcb_q3_incremental_guard_search_results.json", payload)
    write_report(payload)
    log("done", champion=(champion.get("guard") or {}).get("name") if champion else "", report=str(OUT_DIR / "kalcb_q3_incremental_guard_search_report.md"))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only guard search for incremental q3 KALCB trades.")
    parser.add_argument("--max-proxy", type=int, default=24)
    parser.add_argument("--max-exact", type=int, default=18)
    parser.add_argument("--no-holdout", action="store_true")
    args = parser.parse_args()
    run(max_proxy=args.max_proxy, max_exact=args.max_exact, locked_holdout=not args.no_holdout)


if __name__ == "__main__":
    main()
