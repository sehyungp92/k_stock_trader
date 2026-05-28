from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "olr_round2_oos_deep_ablation.py"
ROUND_ROOT = ROOT / "data" / "backtests" / "output" / "olr"
ROUND2_CONFIG = ROUND_ROOT / "round_2" / "optimized_config.json"
BASE_SWEEP_EVALS = ROOT / "tmp" / "olr_round2_reject_min_sweep" / "evaluations.jsonl"
DEFAULT_OUTPUT = ROOT / "tmp" / "olr_reject500_filter_analysis"
HOLDOUT_DAYS = 42


def main() -> int:
    parser = argparse.ArgumentParser(description="Test whether reject_score_min=500 OOS benefits can be isolated from bad added train trades.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-label", action="append", default=[])
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

    base_rows = load_rows(BASE_SWEEP_EVALS)
    baselines = baseline_lookup(base_rows)
    trade_diff = build_trade_diff_report(baselines)
    write_json(output_dir / "reject500_trade_diff.json", trade_diff)
    write_text(output_dir / "reject500_trade_diff.md", render_trade_diff(trade_diff))

    config = helper.normalize_runtime_config("olr", helper.load_yaml_config(str(ROOT / "config" / "optimization" / "olr.yaml")))
    config["capability_level"] = "real_replay"
    config["holdout_days"] = HOLDOUT_DAYS
    config["use_full_available_window"] = True

    round2 = read_json(ROUND2_CONFIG)
    base_mutations = copy.deepcopy(round2["mutations"])
    candidates = build_candidates(helper, base_mutations)
    skip_labels = set(args.skip_label or [])
    if skip_labels:
        candidates = [candidate for candidate in candidates if candidate.label not in skip_labels]
    cached = helper.load_cached_evaluations(eval_path)
    train_rows = helper.evaluate_candidates(
        config,
        candidates,
        "train",
        output_dir,
        eval_path,
        progress_path,
        cached,
        holdout_days=HOLDOUT_DAYS,
        batch_size=max(1, int(args.batch_size)),
    )
    oos_rows = helper.evaluate_candidates(
        config,
        candidates,
        "oos",
        output_dir,
        eval_path,
        progress_path,
        cached,
        holdout_days=HOLDOUT_DAYS,
        batch_size=max(1, int(args.batch_size)),
    )

    all_rows = [*base_rows, *train_rows, *oos_rows]
    by_label_window = {(row["window"], row["label"]): row for row in all_rows}
    summary = summarize_candidates(helper, candidates, by_label_window, baselines)
    payload = {
        "generated_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "candidate_count": len(candidates),
        "baselines": compact_baselines(helper, baselines),
        "trade_diff": trade_diff,
        "ranked": summary,
        "shortlist": shortlist(summary),
    }
    write_json(output_dir / "reject500_filter_sweep.json", payload)
    write_text(output_dir / "reject500_filter_summary.md", render_summary(payload))
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True), flush=True)
    return 0


def load_helper():
    spec = importlib.util.spec_from_file_location("olr_round2_oos_deep_ablation", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_candidates(helper: Any, base: dict[str, Any]) -> list[Any]:
    out = []

    def add(label: str, updates: dict[str, Any], reason: str) -> None:
        mutations = copy.deepcopy(base)
        mutations["olr.afternoon.reject_score_min"] = 500.0
        mutations["olr.afternoon.reject_score_max"] = 650.0
        mutations.update(copy.deepcopy(updates))
        out.append(helper.Candidate(f"r500_{label}", "reject500_filter", mutations, reason))

    for floor in (300, 350, 375, 400, 425, 450, 475, 500):
        add(f"score_floor_{floor}", {"olr.afternoon.min_score": float(floor)}, "Allow reject500 only above a minimum score floor.")
    for top_n in (4, 5, 6, 7):
        add(f"top_n_{top_n}", {"olr.afternoon.top_n": top_n}, "Reduce lower-ranked opportunities introduced by the wider allowed score band.")
    for decay in (2.0, 2.5, 3.0, 4.0):
        add(f"rank_decay_{label_num(decay)}", {"olr.allocation.rank_decay": decay}, "De-emphasize lower-ranked incremental trades without removing the band.")
    for gross in (0.9, 1.0, 1.1):
        add(f"gross_{label_num(gross)}", {"olr.allocation.target_gross_exposure": gross}, "Dampen added band risk through gross exposure.")
    for cap in (0.45, 0.50, 0.55, 0.60):
        add(f"cap_{label_num(cap)}", {"olr.allocation.max_position_pct": cap}, "Dampen added band risk through per-name cap.")
    for close in (0.45, 0.50, 0.55, 0.60, 0.65):
        add(f"close_location_{label_num(close)}", {"olr.afternoon.min_close_location": close}, "Require stronger 14:30 bar close quality.")
    for vwap in (-0.005, 0.0, 0.0025, 0.005, 0.010):
        add(f"vwap_{label_num(vwap)}", {"olr.afternoon.min_vwap_ret": vwap}, "Require stronger same-day VWAP confirmation.")
    for rel_volume in (0.50, 0.75, 1.00, 1.25, 1.50):
        add(f"rel_volume_{label_num(rel_volume)}", {"olr.afternoon.min_rel_volume": rel_volume}, "Require stronger relative volume confirmation.")
    add("require_prev_close", {"olr.afternoon.require_close_above_prev": True}, "Require the afternoon close to hold above previous close.")
    for prior5 in (0.0, 0.02, 0.03, 0.05):
        add(f"prior5_{label_num(prior5)}", {"olr.afternoon.min_prior_ret5": prior5}, "Require stronger short-term momentum before allowing the 400-500 band.")
    for max_prior20 in (0.15, 0.20, 0.30, 0.50):
        add(f"max_prior20_{label_num(max_prior20)}", {"olr.afternoon.max_prior_ret20": max_prior20}, "Reject more extended 20-day momentum names in the added band.")
    for max_gap in (0.05, 0.08, 0.12):
        add(f"max_gap_{label_num(max_gap)}", {"olr.afternoon.max_gap": max_gap}, "Reject large gap-up names in the added band.")
    for drawdown in (0.05, 0.10, 0.15):
        add(f"max_open_drawdown_{label_num(drawdown)}", {"olr.afternoon.max_open_drawdown": drawdown}, "Reject intraday breakdowns before the 14:30 decision.")
    for mode in ("vwap_strength", "flow_confirmed", "daily_plus_intraday"):
        add(f"score_mode_{mode}", {"olr.afternoon.score_mode": mode}, "Retest reject500 with alternate score ranking mode.")

    combo_updates = {
        "floor425_close55": {"olr.afternoon.min_score": 425.0, "olr.afternoon.min_close_location": 0.55},
        "floor425_vwap0": {"olr.afternoon.min_score": 425.0, "olr.afternoon.min_vwap_ret": 0.0},
        "floor425_decay25": {"olr.afternoon.min_score": 425.0, "olr.allocation.rank_decay": 2.5},
        "floor450_close55": {"olr.afternoon.min_score": 450.0, "olr.afternoon.min_close_location": 0.55},
        "floor450_vwap0": {"olr.afternoon.min_score": 450.0, "olr.afternoon.min_vwap_ret": 0.0},
        "floor450_decay25": {"olr.afternoon.min_score": 450.0, "olr.allocation.rank_decay": 2.5},
        "floor450_cap50": {"olr.afternoon.min_score": 450.0, "olr.allocation.max_position_pct": 0.50},
        "floor475_decay25": {"olr.afternoon.min_score": 475.0, "olr.allocation.rank_decay": 2.5},
        "close55_vwap0": {"olr.afternoon.min_close_location": 0.55, "olr.afternoon.min_vwap_ret": 0.0},
        "close55_relvol1": {"olr.afternoon.min_close_location": 0.55, "olr.afternoon.min_rel_volume": 1.0},
        "vwap0_relvol1": {"olr.afternoon.min_vwap_ret": 0.0, "olr.afternoon.min_rel_volume": 1.0},
        "prevclose_close55": {"olr.afternoon.require_close_above_prev": True, "olr.afternoon.min_close_location": 0.55},
        "top6_decay25": {"olr.afternoon.top_n": 6, "olr.allocation.rank_decay": 2.5},
        "top6_floor425": {"olr.afternoon.top_n": 6, "olr.afternoon.min_score": 425.0},
        "top6_floor450": {"olr.afternoon.top_n": 6, "olr.afternoon.min_score": 450.0},
    }
    for label, updates in combo_updates.items():
        add(label, updates, "Combination intended to keep the useful reject500 sub-band while filtering weak train additions.")

    deduped = []
    seen = set()
    for candidate in out:
        signature = helper.stable_signature(candidate.mutations)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(candidate)
    return deduped


def baseline_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        label = str(row.get("label", ""))
        if not label.startswith("focused_reject_score_min_"):
            continue
        suffix = label.rsplit("_", 1)[-1]
        out[f"{row['window']}_{suffix}"] = row
    required = {"train_300", "train_400", "train_450", "train_500", "oos_300", "oos_400", "oos_450", "oos_500"}
    missing = sorted(required - set(out))
    if missing:
        raise RuntimeError(f"Missing baseline rows: {missing}")
    return out


def build_trade_diff_report(baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for window in ("train", "oos"):
        row500 = baselines[f"{window}_500"]
        for base in (300, 400, 450):
            row_base = baselines[f"{window}_{base}"]
            report[f"{window}_500_vs_{base}"] = trade_diff(row500, row_base)
    return report


def trade_diff(current: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    current_trades = keyed_trades(current.get("trade_rows") or [])
    base_trades = keyed_trades(base.get("trade_rows") or [])
    current_keys = set(current_trades)
    base_keys = set(base_trades)
    added = [current_trades[key] for key in sorted(current_keys - base_keys)]
    removed = [base_trades[key] for key in sorted(base_keys - current_keys)]
    common = sorted(current_keys & base_keys)
    common_delta = []
    for key in common:
        now = current_trades[key]
        old = base_trades[key]
        common_delta.append(
            {
                "key": key,
                "symbol": now.get("symbol"),
                "entry_date": now.get("entry_date"),
                "net_pnl_delta": fnum(now.get("net_pnl")) - fnum(old.get("net_pnl")),
                "r_delta": fnum(now.get("r")) - fnum(old.get("r")),
                "score_delta": fnum(now.get("candidate_score")) - fnum(old.get("candidate_score")),
            }
        )
    return {
        "current_label": current.get("label"),
        "base_label": base.get("label"),
        "current_metrics": compact_metrics(current.get("metrics") or {}),
        "base_metrics": compact_metrics(base.get("metrics") or {}),
        "metric_delta": metric_delta(current.get("metrics") or {}, base.get("metrics") or {}),
        "added_trade_count": len(added),
        "removed_trade_count": len(removed),
        "common_trade_count": len(common),
        "added_summary": summarize_trades(added),
        "removed_summary": summarize_trades(removed),
        "common_delta_summary": summarize_deltas(common_delta),
        "added_score_bins": score_bins(added),
        "added_by_rank": group_trades(added, "candidate_rank"),
        "added_by_sector": group_trades(added, "candidate_sector"),
        "added_by_month": group_trades(added, lambda row: str(row.get("entry_date", ""))[:7]),
        "worst_added": sorted((compact_trade(row) for row in added), key=lambda row: row["net_pnl"])[:12],
        "best_added": sorted((compact_trade(row) for row in added), key=lambda row: row["net_pnl"], reverse=True)[:12],
    }


def summarize_candidates(helper: Any, candidates: list[Any], by_label_window: dict[tuple[str, str], dict[str, Any]], baselines: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_train_300 = baselines["train_300"]["metrics"]
    baseline_oos_500 = baselines["oos_500"]["metrics"]
    baseline_train_500 = baselines["train_500"]["metrics"]
    baseline_train_400 = baselines["train_400"]["metrics"]
    rows = []
    for candidate in candidates:
        train = by_label_window.get(("train", candidate.label))
        oos = by_label_window.get(("oos", candidate.label))
        if not train or not oos:
            continue
        train_m = train.get("metrics") or {}
        oos_m = oos.get("metrics") or {}
        row = {
            "label": candidate.label,
            "reason": candidate.reason,
            "updates": changed_from_reject500(candidate.mutations),
            "train": compact_metrics(train_m),
            "oos": compact_metrics(oos_m),
            "train_net_delta_vs_300": metric_net(train_m) - metric_net(baseline_train_300),
            "train_net_delta_vs_400": metric_net(train_m) - metric_net(baseline_train_400),
            "train_net_delta_vs_500": metric_net(train_m) - metric_net(baseline_train_500),
            "oos_net_delta_vs_500": metric_net(oos_m) - metric_net(baseline_oos_500),
            "oos_retention_vs_500": metric_net(oos_m) / max(metric_net(baseline_oos_500), 1e-9),
            "train_repair_score": train_repair_score(train_m, oos_m, baseline_train_300, baseline_train_400, baseline_train_500, baseline_oos_500),
            "errors": [value for value in (train.get("error"), oos.get("error")) if value],
        }
        rows.append(row)
    rows.sort(key=lambda row: row["train_repair_score"], reverse=True)
    return rows


def train_repair_score(train: dict[str, Any], oos: dict[str, Any], train300: dict[str, Any], train400: dict[str, Any], train500: dict[str, Any], oos500: dict[str, Any]) -> float:
    train_net = metric_net(train)
    oos_net = metric_net(oos)
    train_floor = metric_net(train400)
    score = 1000.0 * oos_net
    score += 450.0 * max(0.0, train_net - metric_net(train500))
    score += 350.0 * max(0.0, train_net - train_floor)
    score -= 800.0 * max(0.0, metric_net(oos500) - oos_net)
    score -= 60.0 * max(0.0, 30.0 - metric_trades(oos))
    score -= 200.0 * max(0.0, metric_dd(oos) - metric_dd(oos500))
    score -= 120.0 * max(0.0, metric_dd(train) - metric_dd(train300))
    return score


def shortlist(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keep_oos_10 = [row for row in rows if metric_net(row["oos"]) >= 0.10 and metric_trades(row["oos"]) >= 30.0]
    keep_train_400 = [row for row in rows if metric_net(row["train"]) >= 1.3760101761769667]
    balanced = [row for row in rows if metric_net(row["oos"]) >= 0.09 and metric_net(row["train"]) >= 1.3760101761769667]
    return {
        "top_by_repair_score": rows[:15],
        "oos_at_least_10pct_and_30_trades": keep_oos_10[:15],
        "train_at_least_round2_400": keep_train_400[:15],
        "balanced_oos_9pct_train_round2": balanced[:15],
    }


def changed_from_reject500(mutations: dict[str, Any]) -> dict[str, Any]:
    base = read_json(ROUND2_CONFIG)["mutations"]
    base = copy.deepcopy(base)
    base["olr.afternoon.reject_score_min"] = 500.0
    base["olr.afternoon.reject_score_max"] = 650.0
    return {key: value for key, value in sorted(mutations.items()) if base.get(key) != value}


def keyed_trades(trades: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in trades:
        key = trade_key(row)
        if key in out:
            key = f"{key}#{len(out)}"
        out[key] = row
    return out


def trade_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(key, ""))
        for key in ("entry_date", "symbol", "entry_fill_time", "exit_fill_time")
    )


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "count": 0,
            "net_pnl": 0.0,
            "sum_r": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "avg_score": 0.0,
            "avg_rank": 0.0,
        }
    return {
        "count": len(trades),
        "net_pnl": sum(fnum(row.get("net_pnl")) for row in trades),
        "sum_r": sum(fnum(row.get("r")) for row in trades),
        "avg_r": statistics.fmean(fnum(row.get("r")) for row in trades),
        "win_rate": sum(1 for row in trades if fnum(row.get("net_pnl")) > 0.0) / len(trades),
        "avg_score": statistics.fmean(fnum(row.get("candidate_score")) for row in trades),
        "avg_rank": statistics.fmean(fnum(row.get("candidate_rank")) for row in trades),
        "min_score": min(fnum(row.get("candidate_score")) for row in trades),
        "max_score": max(fnum(row.get("candidate_score")) for row in trades),
    }


def summarize_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "net_pnl_delta": 0.0, "avg_r_delta": 0.0}
    return {
        "count": len(rows),
        "net_pnl_delta": sum(fnum(row.get("net_pnl_delta")) for row in rows),
        "avg_r_delta": statistics.fmean(fnum(row.get("r_delta")) for row in rows),
        "score_delta_sum": sum(fnum(row.get("score_delta")) for row in rows),
    }


def score_bins(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [(0, 300), (300, 350), (350, 400), (400, 425), (425, 450), (450, 475), (475, 500), (500, 650), (650, 100000)]
    out = []
    for lo, hi in bins:
        group = [row for row in trades if lo <= fnum(row.get("candidate_score")) < hi]
        if group:
            payload = summarize_trades(group)
            payload["score_bin"] = f"{lo}-{hi}"
            out.append(payload)
    return out


def group_trades(trades: list[dict[str, Any]], key: str | Any) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        if callable(key):
            group_key = str(key(row))
        else:
            group_key = str(row.get(key, "UNKNOWN"))
        groups[group_key].append(row)
    out = []
    for group_key, group in groups.items():
        payload = summarize_trades(group)
        payload["group"] = group_key
        out.append(payload)
    out.sort(key=lambda row: (row["net_pnl"], -row["count"]))
    return out[:20]


def compact_trade(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_date": row.get("entry_date"),
        "symbol": row.get("symbol"),
        "sector": row.get("candidate_sector"),
        "rank": row.get("candidate_rank"),
        "score": row.get("candidate_score"),
        "net_pnl": row.get("net_pnl"),
        "r": row.get("r"),
        "net_return_pct": row.get("net_return_pct"),
    }


def compact_baselines(helper: Any, baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {key: helper.compact_eval(row) for key, row in sorted(baselines.items())}


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "official_mtm_net_return_pct",
        "net_return_pct",
        "total_trades",
        "entry_fill_count",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "official_mtm_max_drawdown_pct",
        "expected_total_r",
        "entry_level_expected_total_r",
        "same_bar_fill_count",
        "forced_replay_close_count",
        "rejected_order_count",
        "end_open_position_count",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def metric_delta(current: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {
        "net_delta": metric_net(current) - metric_net(base),
        "trade_delta": metric_trades(current) - metric_trades(base),
        "win_delta": metric_win(current) - metric_win(base),
        "drawdown_delta": metric_dd(current) - metric_dd(base),
        "profit_factor_delta": fnum(current.get("profit_factor")) - fnum(base.get("profit_factor")),
    }


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


def render_trade_diff(report: dict[str, Any]) -> str:
    lines = ["# reject_score_min=500 Trade Difference Analysis", ""]
    for name, item in report.items():
        lines.extend(
            [
                f"## {name}",
                f"- Metric net delta: {pct_points(item['metric_delta']['net_delta'])}",
                f"- Metric trade delta: {item['metric_delta']['trade_delta']:+.0f}",
                f"- Added closed trades: {item['added_trade_count']}",
                f"- Removed closed trades: {item['removed_trade_count']}",
                f"- Added net PnL: {money(item['added_summary']['net_pnl'])}",
                f"- Added avg R: {item['added_summary']['avg_r']:.3f}",
                f"- Added win rate: {pct(item['added_summary']['win_rate'])}",
                f"- Added score range: {item['added_summary'].get('min_score', 0.0):.2f} to {item['added_summary'].get('max_score', 0.0):.2f}",
                "",
                "Score bins:",
            ]
        )
        for row in item["added_score_bins"]:
            lines.append(f"- {row['score_bin']}: {row['count']} trades, net {money(row['net_pnl'])}, avg R {row['avg_r']:.3f}, win {pct(row['win_rate'])}")
        lines.append("")
    return "\n".join(lines)


def render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# reject_score_min=500 Filter Sweep",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Candidates tested: {payload['candidate_count']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']}",
        "",
        "## Baseline Reminder",
    ]
    for key in ("train_300", "train_400", "train_500", "oos_300", "oos_400", "oos_500"):
        metrics = payload["baselines"][key]["metrics"]
        lines.append(f"- {key}: net {pct(metric_net(metrics))}, trades {metric_trades(metrics):.0f}, win {pct(metric_win(metrics))}, DD {pct(metric_dd(metrics))}")
    lines.extend(["", "## Top Repair Candidates"])
    for row in payload["shortlist"]["top_by_repair_score"][:15]:
        lines.append(format_candidate_row(row))
    lines.extend(["", "## Candidates With OOS >= 10% And >= 30 Trades"])
    if payload["shortlist"]["oos_at_least_10pct_and_30_trades"]:
        for row in payload["shortlist"]["oos_at_least_10pct_and_30_trades"][:15]:
            lines.append(format_candidate_row(row))
    else:
        lines.append("- None.")
    lines.extend(["", "## Balanced Candidates: OOS >= 9% And Train >= Round2/400"])
    if payload["shortlist"]["balanced_oos_9pct_train_round2"]:
        for row in payload["shortlist"]["balanced_oos_9pct_train_round2"][:15]:
            lines.append(format_candidate_row(row))
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def format_candidate_row(row: dict[str, Any]) -> str:
    return (
        f"- {row['label']}: train {pct(metric_net(row['train']))}/{metric_trades(row['train']):.0f} trades, "
        f"OOS {pct(metric_net(row['oos']))}/{metric_trades(row['oos']):.0f} trades, "
        f"train vs 500 {pct_points(row['train_net_delta_vs_500'])}, OOS vs 500 {pct_points(row['oos_net_delta_vs_500'])}, "
        f"updates={row['updates']}"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def pct_points(value: Any) -> str:
    return f"{100.0 * fnum(value):+.2f} pct-pts"


def money(value: Any) -> str:
    return f"{fnum(value):,.0f}"


def label_num(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
