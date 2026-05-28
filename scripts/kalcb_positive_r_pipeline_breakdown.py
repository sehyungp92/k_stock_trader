from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtests.engine.replay import run_replay
from backtests.engine.sim_broker import BrokerCosts
from backtests.strategies.kalcb.first30_signal_sweep import (
    First30Context,
    First30Spec,
    Selection,
    build_contexts,
    evaluate_selections,
    prepare_first30_dataset,
    score_candidate,
)
from backtests.strategies.kalcb.fixed_trade_plan_phase import (
    KALCBFixedTradePlanOptimizationPlugin,
    SOURCE_PATH_MUTATION,
    SOURCE_RANK_MUTATION,
    SOURCE_SECTION_MUTATION,
    _broker_trade_rows,
    _mutation_key,
)
from backtests.strategies.kalcb.premarket_frontier_sweep import (
    FrontierSpec,
    PremarketFeature,
    build_premarket_features,
    score_frontier,
)
from backtests.strategies.kalcb.runner import KALCBReplayAdapter, _collapse_exit_legs
from backtests.strategies.kalcb.trade_plan_sweep import (
    _clone_snapshots_for_replay,
    _training_only_config,
    load_fixed_candidate_source,
)
from strategy_kalcb.config import KALCBConfig


ROOT = Path(".")
ROUND_DIR = ROOT / "data/backtests/output/kalcb/round_5"
OUT_DIR = ROUND_DIR / "positive_r_pipeline_breakdown"
OUT_JSON = OUT_DIR / "kalcb_positive_r_pipeline_breakdown.json"
OUT_MD = OUT_DIR / "kalcb_positive_r_pipeline_breakdown.md"
OUT_CSV = OUT_DIR / "kalcb_positive_r_pipeline_rows.csv"
OUT_TRADE_CSV = OUT_DIR / "kalcb_positive_r_entered_trades.csv"


FEATURE_KEYS = (
    "daily_return_5d",
    "daily_return_20d",
    "daily_return_60d",
    "daily_volume_ratio_20d",
    "daily_close20_loc",
    "daily_close60_loc",
    "daily_atr_pct",
    "daily_adv20_krw",
    "flow_combined_5d",
    "flow_combined_20d",
    "flow_positive_days_5d",
    "flow_acceleration",
    "flow_z_score",
    "foreign_5d",
    "foreign_z",
    "foreign_acceleration",
    "inst_5d",
    "inst_z",
    "inst_acceleration",
    "flow_agreement_5d",
    "flow_divergence_5d",
    "sector_flow_5d",
    "sector_participation",
    "market_score",
    "market_kospi_ret_5d",
    "market_kosdaq_ret_5d",
    "first30_ret",
    "first30_vwap_ret",
    "first30_gap",
    "first30_rel_volume",
    "first30_close_location",
    "first30_open_drawdown",
    "first30_low_vs_prev_close",
    "first30_range_atr",
    "sector_daily_score_pct",
    "sector_daily_participation",
    "sector_daily_ret_5d",
    "sector_daily_ret_20d",
    "sector_intraday_score_pct",
    "sector_intraday_ret",
    "sector_intraday_breadth",
    "sector_intraday_participation",
)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress("loading_config")
    config = yaml.safe_load((ROOT / "config/optimization/kalcb.yaml").read_text(encoding="utf-8")) or {}
    config["workers"] = 2
    config["validation_gate_enabled"] = False
    config["skip_initial_baseline_eval"] = True

    optimized = read_json(ROUND_DIR / "optimized_config.json")
    mutations = dict(optimized["mutations"])
    source_ref = {
        "path": mutations[SOURCE_PATH_MUTATION],
        "section": mutations[SOURCE_SECTION_MUTATION],
        "rank": int(mutations[SOURCE_RANK_MUTATION]),
    }
    candidate_source = load_fixed_candidate_source(
        source_ref["path"],
        section=source_ref["section"],
        rank=source_ref["rank"],
        strict_expected=False,
    )

    train_config = deepcopy(config)
    train_config["fixed_candidate_source"] = source_ref
    holdout_config = deepcopy(config)
    holdout = dict(holdout_config.get("baseline") or {})
    holdout_config["start"] = str(holdout["holdout_start"])
    holdout_config["end"] = str(holdout["holdout_end"])
    holdout_config["use_full_available_window"] = True
    holdout_config["fixed_candidate_source"] = source_ref

    progress("initialising_plugins")
    train_plugin = KALCBFixedTradePlanOptimizationPlugin(train_config, output_dir=ROUND_DIR, max_workers=2)
    holdout_plugin = KALCBFixedTradePlanOptimizationPlugin(
        holdout_config,
        output_dir=ROUND_DIR / "holdout_mutation_attribution",
        max_workers=2,
    )

    payload: dict[str, Any] = {
        "strategy": "kalcb",
        "generated_at_epoch": time.time(),
        "source_ref": source_ref,
        "positive_r_definition": {
            "dataset_to_selected": "sum(max(mfe_r, 0)) from the causal 09:30-to-flatten first30 opportunity proxy",
            "entered": "sum(max(mfe_r, 0)) from shared-core broker trade paths",
            "captured": "sum(max(realized_net_r, 0)) from closed broker trades",
            "note": "The dataset/candidate/selected stages use a common opportunity proxy. The entered/captured stages use actual shared-core trade R.",
        },
        "windows": {},
    }
    all_rows: list[dict[str, Any]] = []
    all_trade_rows: list[dict[str, Any]] = []
    for window_name, plugin, window_config in (
        ("train", train_plugin, train_config),
        ("holdout", holdout_plugin, holdout_config),
    ):
        progress(f"{window_name}:building_dataset_contexts")
        window_payload, window_rows, window_trade_rows = analyse_window(
            window_name=window_name,
            plugin=plugin,
            base_config=window_config,
            mutations=mutations,
            candidate_source=candidate_source,
        )
        payload["windows"][window_name] = window_payload
        all_rows.extend(window_rows)
        all_trade_rows.extend(window_trade_rows)

    payload["elapsed_seconds"] = round(time.time() - started, 3)
    payload["cross_window_interpretation"] = interpretation(payload)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_rows_csv(all_rows)
    write_trades_csv(all_trade_rows)
    OUT_MD.write_text(markdown_report(payload), encoding="utf-8")
    progress("done")
    print(json.dumps(payload["cross_window_interpretation"], indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_TRADE_CSV}")


def analyse_window(
    *,
    window_name: str,
    plugin: KALCBFixedTradePlanOptimizationPlugin,
    base_config: dict[str, Any],
    mutations: dict[str, Any],
    candidate_source: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    training_config = _training_only_config(dict(base_config), train_only=True)
    dataset = prepare_first30_dataset(training_config)
    contexts = build_contexts(dataset)
    context_by_key = {(day, ctx.symbol): ctx for day, items in contexts.items() for ctx in items}
    cfg = KALCBConfig.from_mapping(training_config, mutations)

    all_selections = [
        Selection(day, ctx.symbol, 0.0, "all_context")
        for day, items in contexts.items()
        for ctx in items
    ]
    opportunity_rows = evaluate_selections(dataset, all_selections, cfg)
    opportunity_by_key = {(row.trade_date, row.symbol): row for row in opportunity_rows}

    features_by_day = build_premarket_features(contexts)
    premarket_by_key = {
        (day, feature.symbol): feature
        for day, features in features_by_day.items()
        for feature in features
    }

    snapshots = dict(plugin.context.compiled_replay.snapshots)
    surfaced_keys = {
        (day, candidate.symbol)
        for day, snapshot in snapshots.items()
        for candidate in snapshot.candidates
    }
    selected_keys = {
        (day, str(symbol))
        for day, snapshot in snapshots.items()
        for symbol in ((snapshot.metadata or {}).get("active_symbols") or [])
    }
    surfaced_contexts_by_day: dict[Any, list[First30Context]] = defaultdict(list)
    for key in surfaced_keys:
        ctx = context_by_key.get(key)
        if ctx is not None:
            surfaced_contexts_by_day[key[0]].append(ctx)

    progress(f"{window_name}:running_unsuppressed_replay")
    trade_rows, decisions = run_replay_details(plugin, mutations)
    entered_keys = {
        (str(row.get("entry_date") or "")[:10], str(row.get("symbol") or ""))
        for row in trade_rows
    }
    entered_date_symbol = {
        (row.get("entry_date"), row.get("symbol")): dict(row)
        for row in trade_rows
    }
    rejection_by_key = rejection_reasons_by_key(decisions)

    analysis_rows: list[dict[str, Any]] = []
    for row in opportunity_rows:
        key = (row.trade_date, row.symbol)
        key_label = (row.trade_date.isoformat(), row.symbol)
        ctx = context_by_key.get(key)
        if ctx is None:
            continue
        surfaced = key in surfaced_keys
        selected = key in selected_keys
        entered = key_label in entered_keys
        analysis = {
            "window": window_name,
            "trade_date": row.trade_date.isoformat(),
            "symbol": row.symbol,
            "sector": ctx.sector,
            "dataset_available": True,
            "surfaced_candidate": surfaced,
            "selected_first30": selected,
            "entered_shared_core": entered,
            "positive_mfe_r_proxy": max(num(row.mfe_r), 0.0),
            "mfe_r_proxy": num(row.mfe_r),
            "mae_r_proxy": num(row.mae_r),
            "net_eod_r_proxy": safe_div(num(row.net_eod_pct), num(row.risk_pct)),
            "gross_eod_r_proxy": safe_div(num(row.gross_eod_pct), num(row.risk_pct)),
            "frontier_loss_reason": "" if surfaced else frontier_loss_reason(candidate_source.frontier, premarket_by_key.get(key)),
            "first30_loss_reason": "" if (not surfaced or selected) else first30_loss_reason(candidate_source.first30, ctx, surfaced_contexts_by_day.get(row.trade_date, ())),
            "entry_loss_reason": "" if (not selected or entered) else rejection_by_key.get(key_label, "no_entry_decision_or_post_signal_constraint"),
        }
        analysis.update(context_features(ctx))
        analysis_rows.append(analysis)

    trade_csv_rows = []
    for row in trade_rows:
        out = dict(row)
        out["window"] = window_name
        trade_csv_rows.append(out)

    payload = {
        "date_window": {
            "start": dataset.trading_dates[0].isoformat() if dataset.trading_dates else "",
            "end": dataset.trading_dates[-1].isoformat() if dataset.trading_dates else "",
            "sessions": len(dataset.trading_dates),
        },
        "context_count": len(analysis_rows),
        "pipeline": pipeline_summary(analysis_rows, trade_rows),
        "stage_losses": stage_losses(analysis_rows, trade_rows),
        "top_blockers": top_blockers(analysis_rows, trade_rows),
        "feature_retention": feature_retention(analysis_rows),
        "entered_trade_path": entered_trade_path(trade_rows),
        "decision_summary": decision_summary(decisions),
        "top_lost_rows": top_lost_rows(analysis_rows),
    }
    return payload, analysis_rows, trade_csv_rows


def run_replay_details(plugin: KALCBFixedTradePlanOptimizationPlugin, mutations: dict[str, Any]) -> tuple[tuple[dict[str, Any], ...], list[Any]]:
    run_mutations = dict(mutations)
    run_mutations["kalcb.entry.fast_replay_suppress_rejections"] = False
    key = _mutation_key(run_mutations)
    cached = plugin._evaluation_details.get(key)
    if cached is not None:
        # The cached evaluation has trade rows and a decision summary, but not raw
        # decisions. Fall through to replay again so entry-stage R loss can be
        # attributed at symbol/day level.
        pass
    context = plugin._context_for_mutations(run_mutations)
    initial_equity = float(context.compiled_replay.initial_equity)
    plan_cfg = plugin._config_for_mutations(run_mutations)
    costs = BrokerCosts(
        commission_bps=plan_cfg.commission_bps,
        tax_bps_on_sell=plan_cfg.tax_bps_on_sell,
        slippage_bps=plan_cfg.slippage_bps,
    )
    adapter = KALCBReplayAdapter(
        plan_cfg,
        _clone_snapshots_for_replay(context.compiled_replay.snapshots),
        initial_equity=initial_equity,
        costs=costs,
    )
    replay = run_replay(
        context.compiled_replay.bars,
        adapter,
        initial_equity=initial_equity,
        costs=costs,
        close_open_positions=False,
        bars_are_ordered=True,
        buying_power_leverage=max(float(plan_cfg.intraday_leverage), 1.0),
    )
    replay.decisions.extend(adapter._sync_new_fills(replay.broker))
    adapter.finalize_frontier_shadow(context.compiled_replay.bars[-1] if context.compiled_replay.bars else None)
    trades = _collapse_exit_legs(replay.trades)
    return _broker_trade_rows(trades), list(replay.decisions)


def pipeline_summary(rows: list[dict[str, Any]], trade_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    trades = [dict(row) for row in trade_rows]
    entered_keys = {(str(row.get("entry_date") or "")[:10], str(row.get("symbol") or "")) for row in trades}
    stages = [
        ("dataset_available", [row for row in rows], "positive_mfe_r_proxy"),
        ("candidates_surfaced", [row for row in rows if row["surfaced_candidate"]], "positive_mfe_r_proxy"),
        ("selected_first30", [row for row in rows if row["selected_first30"]], "positive_mfe_r_proxy"),
        (
            "entered_symbols_proxy",
            [row for row in rows if (str(row.get("trade_date") or "")[:10], str(row.get("symbol") or "")) in entered_keys],
            "positive_mfe_r_proxy",
        ),
    ]
    out: list[dict[str, Any]] = []
    dataset_total = sum(num(row["positive_mfe_r_proxy"]) for row in rows)
    previous_total = None
    for name, items, key in stages:
        total = sum(num(row[key]) for row in items)
        out.append(
            {
                "stage": name,
                "basis": "first30_opportunity_proxy",
                "count": len(items),
                "active_days": len({row["trade_date"] for row in items}),
                "total_positive_r": total,
                "avg_positive_r_per_row": total / max(len(items), 1),
                "retained_vs_dataset": safe_div(total, dataset_total),
                "retained_vs_previous": safe_div(total, previous_total) if previous_total is not None else 1.0,
            }
        )
        previous_total = total

    entered_positive_mfe = sum(max(num(row.get("mfe_r")), 0.0) for row in trades)
    captured_positive = sum(max(num(row.get("r")), 0.0) for row in trades)
    net_r = sum(num(row.get("r")) for row in trades)
    selected_total = previous_total or 0.0
    out.append(
        {
            "stage": "entered_shared_core",
            "basis": "actual_shared_core_trade_mfe",
            "basis_transition": "proxy_to_actual_live_stop_risk",
            "count": len(trades),
            "active_days": len({str(row.get("entry_date") or "") for row in trades}),
            "total_positive_r": entered_positive_mfe,
            "total_net_realized_r": net_r,
            "retained_vs_dataset": safe_div(entered_positive_mfe, dataset_total),
            "retained_vs_previous": safe_div(entered_positive_mfe, selected_total),
        }
    )
    out.append(
        {
            "stage": "captured_realized_positive",
        "basis": "actual_shared_core_realized_net_r",
            "count": sum(1 for row in trades if num(row.get("r")) > 0.0),
            "active_days": len({str(row.get("entry_date") or "") for row in trades if num(row.get("r")) > 0.0}),
            "total_positive_r": captured_positive,
            "total_net_realized_r": net_r,
            "retained_vs_dataset": safe_div(captured_positive, dataset_total),
            "retained_vs_previous": safe_div(captured_positive, entered_positive_mfe),
        }
    )
    return out


def stage_losses(rows: list[dict[str, Any]], trade_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    trades = [dict(row) for row in trade_rows]
    selected = [row for row in rows if row["selected_first30"]]
    entered_keys = {(str(row.get("entry_date") or "")[:10], str(row.get("symbol") or "")) for row in trades}
    out = {
        "dataset_to_candidates": loss_summary(
            [row for row in rows if num(row["positive_mfe_r_proxy"]) > 0.0],
            lambda row: bool(row["surfaced_candidate"]),
            "frontier_loss_reason",
        ),
        "candidates_to_selected": loss_summary(
            [row for row in rows if row["surfaced_candidate"] and num(row["positive_mfe_r_proxy"]) > 0.0],
            lambda row: bool(row["selected_first30"]),
            "first30_loss_reason",
        ),
        "selected_to_entered": loss_summary(
            [row for row in selected if num(row["positive_mfe_r_proxy"]) > 0.0],
            lambda row: (row["trade_date"], row["symbol"]) in entered_keys,
            "entry_loss_reason",
        ),
    }
    entered_positive = sum(max(num(row.get("mfe_r")), 0.0) for row in trades)
    captured_positive = sum(max(num(row.get("r")), 0.0) for row in trades)
    giveback_by_reason: dict[str, float] = defaultdict(float)
    count_by_reason: Counter[str] = Counter()
    for row in trades:
        lost = max(num(row.get("mfe_r")), 0.0) - max(num(row.get("r")), 0.0)
        if lost <= 0:
            continue
        reason = str(row.get("exit_reason") or "unknown")
        giveback_by_reason[reason] += lost
        count_by_reason[reason] += 1
    out["entered_to_captured"] = {
        "previous_total_positive_r": entered_positive,
        "retained_total_positive_r": captured_positive,
        "lost_positive_r": entered_positive - captured_positive,
        "retention": safe_div(captured_positive, entered_positive),
        "top_reasons": [
            {"reason": reason, "lost_positive_r": value, "count": count_by_reason[reason]}
            for reason, value in sorted(giveback_by_reason.items(), key=lambda item: item[1], reverse=True)[:12]
        ],
    }
    return out


def loss_summary(rows: list[dict[str, Any]], retained_fn: Any, reason_key: str) -> dict[str, Any]:
    previous_total = sum(num(row["positive_mfe_r_proxy"]) for row in rows)
    retained = [row for row in rows if retained_fn(row)]
    lost = [row for row in rows if not retained_fn(row)]
    retained_total = sum(num(row["positive_mfe_r_proxy"]) for row in retained)
    by_reason: dict[str, dict[str, Any]] = {}
    for row in lost:
        reason = str(row.get(reason_key) or "unknown")
        item = by_reason.setdefault(reason, {"reason": reason, "lost_positive_r": 0.0, "count": 0})
        item["lost_positive_r"] += num(row["positive_mfe_r_proxy"])
        item["count"] += 1
    return {
        "previous_count": len(rows),
        "retained_count": len(retained),
        "lost_count": len(lost),
        "previous_total_positive_r": previous_total,
        "retained_total_positive_r": retained_total,
        "lost_positive_r": previous_total - retained_total,
        "retention": safe_div(retained_total, previous_total),
        "top_reasons": sorted(by_reason.values(), key=lambda item: item["lost_positive_r"], reverse=True)[:12],
    }


def top_blockers(rows: list[dict[str, Any]], trade_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    losses = stage_losses(rows, trade_rows)
    return {
        name: value.get("top_reasons", [])
        for name, value in losses.items()
    }


def feature_retention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    boundaries = {
        "dataset_to_candidates": ("surfaced_candidate", [row for row in rows if num(row["positive_mfe_r_proxy"]) > 0.0]),
        "candidates_to_selected": ("selected_first30", [row for row in rows if row["surfaced_candidate"] and num(row["positive_mfe_r_proxy"]) > 0.0]),
        "selected_to_entered": ("entered_shared_core", [row for row in rows if row["selected_first30"] and num(row["positive_mfe_r_proxy"]) > 0.0]),
    }
    out: dict[str, Any] = {}
    for boundary, (flag, source_rows) in boundaries.items():
        feature_rows = []
        for feature in FEATURE_KEYS:
            quartiles = quartile_retention(source_rows, feature, flag)
            if not quartiles:
                continue
            top_lost = max(quartiles, key=lambda item: item["lost_positive_r"])
            worst_retention = min(quartiles, key=lambda item: item["retention"])
            total_lost = sum(item["lost_positive_r"] for item in quartiles)
            feature_rows.append(
                {
                    "feature": feature,
                    "total_lost_positive_r": total_lost,
                    "top_lost_bucket": top_lost,
                    "worst_retention_bucket": worst_retention,
                    "quartiles": quartiles,
                }
            )
        feature_rows.sort(
            key=lambda item: (
                item["top_lost_bucket"]["lost_positive_r"],
                item["total_lost_positive_r"],
            ),
            reverse=True,
        )
        out[boundary] = feature_rows[:16]
    return out


def quartile_retention(rows: list[dict[str, Any]], feature: str, flag: str) -> list[dict[str, Any]]:
    pairs = [(row, num(row.get(feature))) for row in rows if finite(row.get(feature))]
    if len(pairs) < 8:
        return []
    pairs.sort(key=lambda item: item[1])
    out = []
    for idx in range(4):
        lo = int(len(pairs) * idx / 4)
        hi = int(len(pairs) * (idx + 1) / 4)
        bucket_pairs = pairs[lo:hi]
        bucket_rows = [row for row, _ in bucket_pairs]
        vals = [value for _, value in bucket_pairs]
        total = sum(num(row["positive_mfe_r_proxy"]) for row in bucket_rows)
        retained = sum(num(row["positive_mfe_r_proxy"]) for row in bucket_rows if row.get(flag))
        out.append(
            {
                "bucket": f"q{idx + 1}",
                "feature_min": min(vals),
                "feature_max": max(vals),
                "count": len(bucket_rows),
                "total_positive_r": total,
                "retained_positive_r": retained,
                "lost_positive_r": total - retained,
                "retention": safe_div(retained, total),
            }
        )
    return out


def entered_trade_path(trade_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in trade_rows]
    total_mfe = sum(max(num(row.get("mfe_r")), 0.0) for row in rows)
    captured = sum(max(num(row.get("r")), 0.0) for row in rows)
    by_exit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_entry_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_exit[str(row.get("exit_reason") or "unknown")].append(row)
        by_entry_route[str(row.get("entry_route") or "unknown")].append(row)
    return {
        "count": len(rows),
        "total_mfe_r": total_mfe,
        "captured_positive_r": captured,
        "net_r": sum(num(row.get("r")) for row in rows),
        "mfe_capture": safe_div(captured, total_mfe),
        "by_exit_reason": {reason: r_summary(items) for reason, items in sorted(by_exit.items())},
        "by_entry_route": {reason: r_summary(items) for reason, items in sorted(by_entry_route.items())},
        "top_giveback_trades": sorted(
            [
                {
                    "entry_date": row.get("entry_date"),
                    "symbol": row.get("symbol"),
                    "r": num(row.get("r")),
                    "mfe_r": num(row.get("mfe_r")),
                    "giveback_r": num(row.get("mfe_r")) - num(row.get("r")),
                    "exit_reason": row.get("exit_reason"),
                    "first30_ret": row.get("first30_ret"),
                    "first30_rel_volume": row.get("first30_rel_volume"),
                    "sector": row.get("sector"),
                }
                for row in rows
            ],
            key=lambda item: item["giveback_r"],
            reverse=True,
        )[:15],
    }


def r_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_mfe = sum(max(num(row.get("mfe_r")), 0.0) for row in rows)
    captured = sum(max(num(row.get("r")), 0.0) for row in rows)
    net = sum(num(row.get("r")) for row in rows)
    return {
        "count": len(rows),
        "total_mfe_r": total_mfe,
        "captured_positive_r": captured,
        "net_r": net,
        "avg_r": safe_div(net, len(rows)),
        "win_share": safe_div(sum(1 for row in rows if num(row.get("r")) > 0.0), len(rows)),
        "mfe_capture": safe_div(captured, total_mfe),
        "avg_mfe_r": safe_div(total_mfe, len(rows)),
        "avg_mae_r": safe_div(sum(num(row.get("mae_r")) for row in rows), len(rows)),
    }


def top_lost_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    specs = {
        "dataset_not_surfaced": lambda row: not row["surfaced_candidate"],
        "surfaced_not_selected": lambda row: row["surfaced_candidate"] and not row["selected_first30"],
        "selected_not_entered": lambda row: row["selected_first30"] and not row["entered_shared_core"],
    }
    out = {}
    keys = (
        "trade_date",
        "symbol",
        "sector",
        "positive_mfe_r_proxy",
        "net_eod_r_proxy",
        "frontier_loss_reason",
        "first30_loss_reason",
        "entry_loss_reason",
        "daily_return_20d",
        "daily_close20_loc",
        "flow_combined_5d",
        "flow_z_score",
        "first30_ret",
        "first30_rel_volume",
        "first30_close_location",
        "first30_range_atr",
        "sector_daily_score_pct",
        "sector_intraday_score_pct",
    )
    for name, pred in specs.items():
        items = [row for row in rows if pred(row) and num(row["positive_mfe_r_proxy"]) > 0.0]
        items.sort(key=lambda row: num(row["positive_mfe_r_proxy"]), reverse=True)
        out[name] = [{key: row.get(key) for key in keys} for row in items[:20]]
    return out


def rejection_reasons_by_key(decisions: Iterable[Any]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for decision in decisions:
        code = str(getattr(decision, "decision_code", "") or "")
        if code not in {"entry_rejected", "entry_blocked"}:
            continue
        ts = getattr(decision, "timestamp", None)
        symbol = str(getattr(decision, "symbol", "") or "")
        if ts is None or not symbol:
            continue
        metadata = dict(getattr(decision, "metadata", {}) or {})
        gates = metadata.get("gates") or metadata.get("filter_decisions") or ()
        first_failed = ""
        for gate in gates:
            if isinstance(gate, dict) and bool(gate.get("applicable", True)) and not bool(gate.get("passed", True)):
                first_failed = str(gate.get("filter_name") or "")
                break
        reason = str(getattr(decision, "reason", "") or "unknown")
        label = f"{first_failed or reason}:{reason}"
        out.setdefault((ts.date().isoformat(), symbol), label)
    return out


def decision_summary(decisions: Iterable[Any]) -> dict[str, Any]:
    codes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    failed_gates: Counter[str] = Counter()
    for decision in decisions:
        code = str(getattr(decision, "decision_code", "") or "")
        codes[code] += 1
        if code not in {"entry_rejected", "entry_blocked"}:
            continue
        reason = str(getattr(decision, "reason", "") or "unknown")
        reasons[reason] += 1
        metadata = dict(getattr(decision, "metadata", {}) or {})
        gates = metadata.get("gates") or metadata.get("filter_decisions") or ()
        first_failed = ""
        for gate in gates:
            if isinstance(gate, dict) and bool(gate.get("applicable", True)) and not bool(gate.get("passed", True)):
                first_failed = str(gate.get("filter_name") or "")
                break
        failed_gates[first_failed or reason] += 1
    return {
        "decision_code_counts": codes.most_common(),
        "entry_rejection_reasons": reasons.most_common(20),
        "entry_failed_gates": failed_gates.most_common(20),
    }


def frontier_loss_reason(spec: FrontierSpec, feature: PremarketFeature | None) -> str:
    if feature is None:
        return "missing_premarket_feature"
    checks = (
        ("frontier_min_ret5", feature.ret5 >= spec.min_ret5),
        ("frontier_min_ret20", feature.ret20 >= spec.min_ret20),
        ("frontier_max_ret20", feature.ret20 <= spec.max_ret20),
        ("frontier_min_ret60", feature.ret60 >= spec.min_ret60),
        ("frontier_min_close20_loc", feature.close20_loc >= spec.min_close20_loc),
        ("frontier_min_adv20_krw", feature.adv20_krw >= spec.min_adv20_krw),
        ("frontier_max_atr_pct", feature.atr_pct <= spec.max_atr_pct),
        ("frontier_min_volume_surge", feature.volume_surge >= spec.min_volume_surge),
        ("frontier_min_flow_5d", feature.flow_5d >= spec.min_flow_5d),
        ("frontier_min_flow_z", feature.flow_z >= spec.min_flow_z),
        ("frontier_min_flow_acceleration", feature.flow_acceleration >= spec.min_flow_acceleration),
        ("frontier_min_foreign_flow_5d", feature.foreign_5d >= spec.min_foreign_flow_5d),
        ("frontier_min_inst_flow_5d", feature.inst_5d >= spec.min_inst_flow_5d),
        ("frontier_min_foreign_z", feature.foreign_z >= spec.min_foreign_z),
        ("frontier_min_inst_z", feature.inst_z >= spec.min_inst_z),
        ("frontier_min_flow_agreement", feature.flow_agreement_5d >= spec.min_flow_agreement),
        ("frontier_max_flow_divergence", feature.flow_divergence_5d <= spec.max_flow_divergence),
        ("frontier_min_sector_flow", feature.sector_flow_5d >= spec.min_sector_flow),
        ("frontier_min_sector_participation", feature.sector_participation >= spec.min_sector_participation),
        ("frontier_min_market_score", feature.market_score >= spec.min_market_score),
        ("frontier_require_above_sma20", (not spec.require_above_sma20) or feature.above_sma20),
        ("frontier_require_above_sma60", (not spec.require_above_sma60) or feature.above_sma60),
        ("frontier_require_flow_available", (not spec.require_flow_available) or feature.flow_available),
    )
    for name, passed in checks:
        if not passed:
            return name
    return "frontier_top_n_rank_cutoff" if score_frontier(spec, feature) is not None else "frontier_no_score"


def first30_loss_reason(spec: First30Spec, ctx: First30Context, day_contexts: Iterable[First30Context]) -> str:
    checks = (
        ("first30_min_ret", ctx.first30_ret >= spec.min_first30_ret),
        ("first30_min_vwap_ret", ctx.vwap_ret >= spec.min_vwap_ret),
        ("first30_min_gap", ctx.gap >= spec.min_gap),
        ("first30_max_gap", ctx.gap <= spec.max_gap),
        ("first30_min_rel_volume", ctx.rel_volume >= spec.min_rel_volume),
        ("first30_min_close_location", ctx.close_location >= spec.min_close_location),
        ("first30_max_open_drawdown", abs(min(ctx.open_drawdown, 0.0)) <= spec.max_open_drawdown),
        ("first30_max_range_atr", ctx.range_atr <= spec.max_range_atr),
        ("first30_min_prior_ret5", ctx.daily.return_5d >= spec.min_prior_ret5),
        ("first30_min_prior_ret20", ctx.daily.return_20d >= spec.min_prior_ret20),
        ("first30_max_prior_ret20", ctx.daily.return_20d <= spec.max_prior_ret20),
        ("first30_min_prior_ret60", ctx.daily.return_60d >= spec.min_prior_ret60),
        ("first30_min_low_vs_prev_close", ctx.low_vs_prev_close >= spec.min_low_vs_prev_close),
        ("first30_min_flow_5d", ctx.flow.combined_5d >= spec.min_flow_5d),
        ("first30_min_foreign_flow_5d", ctx.flow.foreign_5d >= spec.min_foreign_flow_5d),
        ("first30_min_inst_flow_5d", ctx.flow.inst_5d >= spec.min_inst_flow_5d),
        ("first30_min_flow_z", ctx.flow.z_score >= spec.min_flow_z),
        ("first30_min_flow_agreement", ctx.flow.agreement_5d >= spec.min_flow_agreement),
        ("first30_max_flow_divergence", ctx.flow.divergence_5d <= spec.max_flow_divergence),
        ("first30_min_sector_flow", ctx.flow.sector_flow_5d >= spec.min_sector_flow),
        ("first30_min_market_score", ctx.market.score >= spec.min_market_score),
        ("first30_require_close_above_prev", (not spec.require_close_above_prev) or ctx.intraday.close > ctx.daily.prev_close),
        ("first30_momentum_positive_ret", spec.score_mode not in {"momentum", "hybrid", "efficient", "flow_confirmed"} or ctx.first30_ret >= 0.0),
        ("first30_vwap_strength_positive", spec.score_mode not in {"vwap_strength", "flow_confirmed"} or ctx.vwap_ret >= 0.0),
        ("first30_gap_hold", spec.score_mode != "gap_hold" or (ctx.gap >= 0.002 and ctx.first30_ret >= 0.0 and ctx.low_vs_prev_close >= -0.02)),
        (
            "first30_flow_confirmed",
            spec.score_mode != "flow_confirmed"
            or (ctx.flow.combined_5d > 0.0 or ctx.flow.z_score > 0.0 or ctx.flow.sector_flow_5d > 0.0),
        ),
    )
    for name, passed in checks:
        if not passed:
            return name
    scored = [
        (score_candidate(spec, item), item.symbol)
        for item in day_contexts
        if first30_base_passes(spec, item)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = {symbol for _, symbol in scored[: max(1, int(spec.top_n))]}
    return "first30_top_n_rank_cutoff" if ctx.symbol not in selected else "selected"


def first30_base_passes(spec: First30Spec, ctx: First30Context) -> bool:
    return first30_loss_reason_without_rank(spec, ctx) == ""


def first30_loss_reason_without_rank(spec: First30Spec, ctx: First30Context) -> str:
    # Kept separate to avoid recursive top-N checks in first30_loss_reason.
    if ctx.first30_ret < spec.min_first30_ret:
        return "first30_min_ret"
    if ctx.vwap_ret < spec.min_vwap_ret:
        return "first30_min_vwap_ret"
    if ctx.gap < spec.min_gap:
        return "first30_min_gap"
    if ctx.gap > spec.max_gap:
        return "first30_max_gap"
    if ctx.rel_volume < spec.min_rel_volume:
        return "first30_min_rel_volume"
    if ctx.close_location < spec.min_close_location:
        return "first30_min_close_location"
    if abs(min(ctx.open_drawdown, 0.0)) > spec.max_open_drawdown:
        return "first30_max_open_drawdown"
    if ctx.range_atr > spec.max_range_atr:
        return "first30_max_range_atr"
    if ctx.daily.return_5d < spec.min_prior_ret5:
        return "first30_min_prior_ret5"
    if ctx.daily.return_20d < spec.min_prior_ret20:
        return "first30_min_prior_ret20"
    if ctx.daily.return_20d > spec.max_prior_ret20:
        return "first30_max_prior_ret20"
    if ctx.daily.return_60d < spec.min_prior_ret60:
        return "first30_min_prior_ret60"
    if ctx.low_vs_prev_close < spec.min_low_vs_prev_close:
        return "first30_min_low_vs_prev_close"
    if ctx.flow.combined_5d < spec.min_flow_5d:
        return "first30_min_flow_5d"
    if ctx.flow.foreign_5d < spec.min_foreign_flow_5d:
        return "first30_min_foreign_flow_5d"
    if ctx.flow.inst_5d < spec.min_inst_flow_5d:
        return "first30_min_inst_flow_5d"
    if ctx.flow.z_score < spec.min_flow_z:
        return "first30_min_flow_z"
    if ctx.flow.agreement_5d < spec.min_flow_agreement:
        return "first30_min_flow_agreement"
    if ctx.flow.divergence_5d > spec.max_flow_divergence:
        return "first30_max_flow_divergence"
    if ctx.flow.sector_flow_5d < spec.min_sector_flow:
        return "first30_min_sector_flow"
    if ctx.market.score < spec.min_market_score:
        return "first30_min_market_score"
    if spec.require_close_above_prev and ctx.intraday.close <= ctx.daily.prev_close:
        return "first30_require_close_above_prev"
    if spec.score_mode in {"momentum", "hybrid", "efficient", "flow_confirmed"} and ctx.first30_ret < 0.0:
        return "first30_momentum_positive_ret"
    if spec.score_mode in {"vwap_strength", "flow_confirmed"} and ctx.vwap_ret < 0.0:
        return "first30_vwap_strength_positive"
    if spec.score_mode == "gap_hold" and (ctx.gap < 0.002 or ctx.first30_ret < 0.0 or ctx.low_vs_prev_close < -0.02):
        return "first30_gap_hold"
    if spec.score_mode == "flow_confirmed" and not (ctx.flow.combined_5d > 0.0 or ctx.flow.z_score > 0.0 or ctx.flow.sector_flow_5d > 0.0):
        return "first30_flow_confirmed"
    return ""


def context_features(ctx: First30Context) -> dict[str, Any]:
    daily = ctx.daily
    flow = ctx.flow
    market = ctx.market
    sd = ctx.sector_daily
    si = ctx.sector_intraday
    return {
        "daily_return_5d": daily.return_5d,
        "daily_return_20d": daily.return_20d,
        "daily_return_60d": daily.return_60d,
        "daily_volume_ratio_20d": daily.volume_ratio_20d,
        "daily_close20_loc": daily.close20_loc,
        "daily_close60_loc": daily.close60_loc,
        "daily_atr_pct": safe_div(daily.atr14, daily.prev_close),
        "daily_adv20_krw": daily.adv20_krw,
        "daily_above_sma20": daily.above_sma20,
        "daily_above_sma60": daily.above_sma60,
        "flow_available": flow.available,
        "flow_combined_1d": flow.combined_1d,
        "flow_combined_3d": flow.combined_3d,
        "flow_combined_5d": flow.combined_5d,
        "flow_combined_20d": flow.combined_20d,
        "flow_positive_days_5d": flow.positive_days_5d,
        "flow_acceleration": flow.acceleration,
        "flow_z_score": flow.z_score,
        "foreign_5d": flow.foreign_5d,
        "foreign_z": flow.foreign_z,
        "foreign_acceleration": flow.foreign_acceleration,
        "inst_5d": flow.inst_5d,
        "inst_z": flow.inst_z,
        "inst_acceleration": flow.inst_acceleration,
        "flow_agreement_5d": flow.agreement_5d,
        "flow_divergence_5d": flow.divergence_5d,
        "sector_flow_5d": flow.sector_flow_5d,
        "sector_participation": flow.sector_participation,
        "market_score": market.score,
        "market_kospi_ret_5d": market.kospi_ret_5d,
        "market_kosdaq_ret_5d": market.kosdaq_ret_5d,
        "first30_ret": ctx.first30_ret,
        "first30_vwap_ret": ctx.vwap_ret,
        "first30_gap": ctx.gap,
        "first30_rel_volume": ctx.rel_volume,
        "first30_close_location": ctx.close_location,
        "first30_open_drawdown": ctx.open_drawdown,
        "first30_low_vs_prev_close": ctx.low_vs_prev_close,
        "first30_range_atr": ctx.range_atr,
        "sector_daily_score_pct": getattr(sd, "score_pct", None),
        "sector_daily_participation": getattr(sd, "participation", None),
        "sector_daily_ret_5d": getattr(sd, "ret_5d", None),
        "sector_daily_ret_20d": getattr(sd, "ret_20d", None),
        "sector_intraday_score_pct": getattr(si, "score_pct", None),
        "sector_intraday_ret": getattr(si, "ret", None),
        "sector_intraday_breadth": getattr(si, "breadth", None),
        "sector_intraday_participation": getattr(si, "participation", None),
    }


def interpretation(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"summary": [], "recommended_experiments": []}
    for window, data in payload.get("windows", {}).items():
        pipeline = {row["stage"]: row for row in data.get("pipeline", [])}
        dataset_r = num(pipeline.get("dataset_available", {}).get("total_positive_r"))
        surfaced_r = num(pipeline.get("candidates_surfaced", {}).get("total_positive_r"))
        selected_r = num(pipeline.get("selected_first30", {}).get("total_positive_r"))
        entered_proxy_r = num(pipeline.get("entered_symbols_proxy", {}).get("total_positive_r"))
        entered_r = num(pipeline.get("entered_shared_core", {}).get("total_positive_r"))
        captured_r = num(pipeline.get("captured_realized_positive", {}).get("total_positive_r"))
        out["summary"].append(
            {
                "window": window,
                "dataset_positive_r": dataset_r,
                "surfaced_positive_r": surfaced_r,
                "selected_positive_r": selected_r,
                "entered_proxy_positive_r": entered_proxy_r,
                "entered_positive_mfe_r": entered_r,
                "captured_positive_r": captured_r,
                "dataset_to_surfaced_retention": safe_div(surfaced_r, dataset_r),
                "surfaced_to_selected_retention": safe_div(selected_r, surfaced_r),
                "selected_to_entered_proxy_retention": safe_div(entered_proxy_r, selected_r),
                "entered_to_captured_retention": safe_div(captured_r, entered_r),
            }
        )
    out["recommended_experiments"] = [
        "Run a premarket frontier branch that relaxes strict flow-available and rank cutoffs only for top-quartile first30_rel_volume / first30_ret / sector_intraday_score cohorts, then cap per-sector exposure.",
        "Run a first30 selector variant that keeps top-N=1 for normal days but adds a second slot when first30_rel_volume and first30_range_atr are both in the top quartile and the sector intraday score is confirmed.",
        "Run an entry audit variant with require_initial_active=False but gated by frontier_rank <= 8, first30_rel_volume >= top-quartile, first30_signal_bar_cpr >= 0.75, and max positions/sector caps unchanged.",
        "Run exit variants by cohort rather than global target/giveback: MFE floor or late giveback only for high-MFE/high-giveback cohorts identified in entered_trade_path.top_giveback_trades.",
    ]
    return out


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB positive-R pipeline breakdown",
        "",
        f"Generated in {payload.get('elapsed_seconds', 0.0):.1f}s.",
        "",
        "Positive R before entry is `max(MFE R, 0)` from the causal 09:30-to-flatten first30 opportunity proxy. Entered and captured use actual shared-core broker trade R.",
        "",
    ]
    for window, data in payload.get("windows", {}).items():
        lines.extend([f"## {window.title()} window", ""])
        lines.extend([
            "| Stage | Basis | Count | Total +R | Retained vs prior |",
            "|---|---|---:|---:|---:|",
        ])
        for row in data.get("pipeline", []):
            retained = "n/a" if row.get("basis_transition") else pct(row.get("retained_vs_previous"))
            lines.append(
                f"| {row['stage']} | {row.get('basis', '')} | {int(row.get('count', 0))} | {num(row.get('total_positive_r')):.1f} | {retained} |"
            )
        lines.append("")
        lines.extend(["### R leakage by component", ""])
        lines.extend(["| Boundary | Lost +R | Retention | Top blocker |", "|---|---:|---:|---|"])
        for boundary, loss in data.get("stage_losses", {}).items():
            reasons = loss.get("top_reasons") or []
            top = reasons[0] if reasons else {}
            top_label = top.get("reason", "")
            if top:
                top_label = f"{top_label} ({num(top.get('lost_positive_r')):.1f}R)"
            lines.append(
                f"| {boundary} | {num(loss.get('lost_positive_r')):.1f} | {pct(loss.get('retention'))} | {top_label} |"
            )
        lines.append("")
        lines.extend(["### Highest-signal feature cuts", ""])
        lines.extend(["| Boundary | Feature | Worst bucket | Lost +R in bucket | Retention |", "|---|---|---|---:|---:|"])
        for boundary, feature_rows in data.get("feature_retention", {}).items():
            for feature in feature_rows[:5]:
                bucket = feature.get("top_lost_bucket", {})
                label = f"{bucket.get('bucket')} [{num(bucket.get('feature_min')):.4g}, {num(bucket.get('feature_max')):.4g}]"
                lines.append(
                    f"| {boundary} | {feature.get('feature')} | {label} | {num(bucket.get('lost_positive_r')):.1f} | {pct(bucket.get('retention'))} |"
                )
        lines.append("")
        path = data.get("entered_trade_path", {})
        lines.extend(
            [
                "### Entered path",
                "",
                f"Entered positive MFE R: {num(path.get('total_mfe_r')):.1f}; captured positive R: {num(path.get('captured_positive_r')):.1f}; net R: {num(path.get('net_r')):.1f}; MFE capture: {pct(path.get('mfe_capture'))}.",
                "",
            ]
        )
    lines.extend(["## Recommended quantitative next steps", ""])
    for item in payload.get("cross_window_interpretation", {}).get("recommended_experiments", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Row-level audit: `{OUT_CSV}`")
    lines.append(f"Entered trades: `{OUT_TRADE_CSV}`")
    lines.append(f"JSON: `{OUT_JSON}`")
    return "\n".join(lines)


def write_rows_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_trades_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with OUT_TRADE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def progress(stage: str) -> None:
    print(json.dumps({"stage": stage, "ts": round(time.time(), 3)}, sort_keys=True), flush=True)


def num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def safe_div(a: Any, b: Any) -> float:
    denom = num(b)
    if abs(denom) < 1e-12:
        return 0.0
    return num(a) / denom


def pct(value: Any) -> str:
    return f"{100.0 * num(value):.1f}%"


if __name__ == "__main__":
    main()
