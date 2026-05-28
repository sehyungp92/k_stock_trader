from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_OPT_PATH = REPO_ROOT / "scripts" / "kalcb_train_alpha_capture_optimizer.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "train_stability_route_optimizer"

from backtests.strategies.kalcb.candidate_surfacing_recovery import PoolVariant, evaluate_compiled_candidate_pool  # noqa: E402
from backtests.strategies.kalcb.fixed_trade_plan_phase import _configured_entry_routes, _route_candidate_passes  # noqa: E402
from backtests.strategies.kalcb.shadow_ledger_reranker import read_jsonl, write_jsonl  # noqa: E402


def _load_train_opt_module():
    spec = importlib.util.spec_from_file_location("kalcb_train_alpha_capture_optimizer_module", TRAIN_OPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load train optimizer module: {TRAIN_OPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_opt = _load_train_opt_module()
shared = train_opt.shared
opt = train_opt.opt


@dataclass(frozen=True)
class FilterSpec:
    name: str
    description: str
    rules: tuple[tuple[str, str, float], ...]


@dataclass(frozen=True)
class RiskSpec:
    name: str
    first30_risk_mult: float | None


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


def parse_date(value: Any) -> date | None:
    text = str(value or "")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def filter_specs() -> list[FilterSpec]:
    return [
        FilterSpec("base_top16", "No extra filter; train-selected top16 policy.", ()),
        FilterSpec("cpr_ge_055", "Require first30 signal CPR >= 0.55.", (("first30_signal_bar_cpr", ">=", 0.55),)),
        FilterSpec("cpr_ge_065", "Require first30 signal CPR >= 0.65.", (("first30_signal_bar_cpr", ">=", 0.65),)),
        FilterSpec("cpr_ge_075", "Require first30 signal CPR >= 0.75.", (("first30_signal_bar_cpr", ">=", 0.75),)),
        FilterSpec("quality_ge_75", "Require first30 quality >= 75.", (("first30_quality_pct", ">=", 75.0),)),
        FilterSpec("quality_ge_80", "Require first30 quality >= 80.", (("first30_quality_pct", ">=", 80.0),)),
        FilterSpec("sector_intraday_ge_50", "Require sector intraday score >= 50.", (("sector_intraday_score_pct", ">=", 50.0),)),
        FilterSpec("sector_intraday_ge_625", "Require sector intraday score >= 62.5.", (("sector_intraday_score_pct", ">=", 62.5),)),
        FilterSpec("sector_daily_ge_50", "Require sector daily score >= 50.", (("sector_daily_score_pct", ">=", 50.0),)),
        FilterSpec("daily_close20_ge_60", "Require daily close20 location >= 0.60.", (("daily_close20_loc", ">=", 0.60),)),
        FilterSpec("range_atr_lte_2", "Cap first30 range/ATR <= 2.0.", (("first30_range_atr", "<=", 2.0),)),
        FilterSpec("range_atr_lte_16", "Cap first30 range/ATR <= 1.6.", (("first30_range_atr", "<=", 1.6),)),
        FilterSpec("gap_retention_ge_50", "Require first30 gap retention >= 0.50.", (("first30_gap_retention_ratio", ">=", 0.50),)),
        FilterSpec("rvol_ge_3", "Require first30 RVOL >= 3.", (("first30_rel_volume", ">=", 3.0),)),
        FilterSpec("rvol_ge_5", "Require first30 RVOL >= 5.", (("first30_rel_volume", ">=", 5.0),)),
        FilterSpec(
            "quality75_range2",
            "Require first30 quality >=75 and range/ATR <=2.",
            (("first30_quality_pct", ">=", 75.0), ("first30_range_atr", "<=", 2.0)),
        ),
        FilterSpec(
            "cpr65_sector50",
            "Require CPR >=0.65 and sector intraday score >=50.",
            (("first30_signal_bar_cpr", ">=", 0.65), ("sector_intraday_score_pct", ">=", 50.0)),
        ),
        FilterSpec(
            "sector50_gapret50",
            "Require sector intraday score >=50 and gap retention >=0.50.",
            (("sector_intraday_score_pct", ">=", 50.0), ("first30_gap_retention_ratio", ">=", 0.50)),
        ),
        FilterSpec(
            "daily60_sector50",
            "Require daily close20 >=0.60 and sector intraday score >=50.",
            (("daily_close20_loc", ">=", 0.60), ("sector_intraday_score_pct", ">=", 50.0)),
        ),
        FilterSpec(
            "rvol3_cpr65",
            "Require RVOL >=3 and CPR >=0.65.",
            (("first30_rel_volume", ">=", 3.0), ("first30_signal_bar_cpr", ">=", 0.65)),
        ),
    ]


def risk_specs() -> list[RiskSpec]:
    return [
        RiskSpec("risk99", None),
        RiskSpec("risk80", 0.80),
        RiskSpec("risk65", 0.65),
        RiskSpec("risk50", 0.50),
    ]


def passes_rules(row: dict[str, Any], spec: FilterSpec) -> bool:
    for key, op, threshold in spec.rules:
        value = num(row.get(key))
        if op == ">=" and value < threshold:
            return False
        if op == "<=" and value > threshold:
            return False
    return True


def apply_filter(pool_rows: list[dict[str, Any]], spec: FilterSpec, *, active_count: int) -> list[dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = {}
    for row in pool_rows:
        if passes_rules(row, spec):
            by_day.setdefault(str(row.get("trade_date") or "")[:10], []).append(dict(row))
    out: list[dict[str, Any]] = []
    for day, rows in sorted(by_day.items()):
        ordered = sorted(rows, key=lambda row: (int(row.get("pool_rank") or 999), str(row.get("symbol") or "")))
        for rank, row in enumerate(ordered, start=1):
            row["pool_variant"] = spec.name
            row["pool_rank"] = rank
            row["pool_active"] = rank <= active_count
            row["frontier_role_for_replay"] = "initial_active" if rank <= active_count else "frontier_shadow"
            out.append(row)
    return out


def mutate_risk(seed: dict[str, Any], risk: RiskSpec) -> dict[str, Any]:
    out = copy.deepcopy(seed)
    if risk.first30_risk_mult is None:
        return out
    routes = [dict(route) for route in out.get("kalcb.entry.routes") or [] if isinstance(route, dict)]
    for route in routes:
        if str(route.get("mode") or "") == "first30_open":
            route["risk_mult"] = float(risk.first30_risk_mult)
            route["notional_mult"] = float(risk.first30_risk_mult)
    out["kalcb.entry.routes"] = routes
    return out


def static_route_eligibility(pool_rows: list[dict[str, Any]], mutations: dict[str, Any]) -> dict[str, Any]:
    routes = _configured_entry_routes(mutations)
    eligible = 0
    blockers: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    for row in pool_rows:
        passed_any = False
        first_reason = ""
        for route in routes:
            passed, reason = _route_candidate_passes(route, mutations, row)
            mode = str(route.get("mode") or route.get("name") or "route")
            if passed:
                passed_any = True
                by_mode[mode] = by_mode.get(mode, 0) + 1
            elif not first_reason:
                first_reason = reason
        if passed_any:
            eligible += 1
        else:
            blockers[first_reason or "not_eligible"] = blockers.get(first_reason or "not_eligible", 0) + 1
    return {
        "static_route_eligible_count": eligible,
        "static_route_eligible_share": eligible / max(len(pool_rows), 1),
        "static_route_eligible_by_mode": by_mode,
        "top_static_blockers": sorted(blockers.items(), key=lambda item: item[1], reverse=True)[:8],
    }


def proxy_summary(pool_rows: list[dict[str, Any]], mutations: dict[str, Any]) -> dict[str, Any]:
    active_rows = [row for row in pool_rows if bool(row.get("pool_active"))]
    eligibility = static_route_eligibility(pool_rows, mutations)
    return {
        "pool_rows": len(pool_rows),
        "active_rows": len(active_rows),
        "active_days": len({str(row.get("trade_date") or "")[:10] for row in active_rows}),
        "proxy_positive_r": sum(num(row.get("optimizer_positive_r_proxy")) for row in active_rows),
        "proxy_net_eod_r": sum(num(row.get("optimizer_net_eod_r_proxy")) for row in active_rows),
        **eligibility,
    }


def read_trade_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    return read_jsonl(source)


def fold_date_map(dates: list[date], folds: int = 5) -> dict[date, int]:
    if not dates:
        return {}
    ordered = sorted(dates)
    out: dict[date, int] = {}
    for index, day in enumerate(ordered):
        fold = min(folds - 1, int(index * folds / max(len(ordered), 1)))
        out[day] = fold
    return out


def stability_metrics(trade_rows: list[dict[str, Any]], session_dates: list[date], initial_equity: float) -> dict[str, Any]:
    date_to_fold = fold_date_map(session_dates, 5)
    fold_net = [0.0 for _ in range(5)]
    fold_trades = [0 for _ in range(5)]
    sector_net: dict[str, float] = {}
    sector_trades: dict[str, int] = {}
    by_exit: dict[str, int] = {}
    by_route: dict[str, int] = {}
    total_trades = len(trade_rows)
    total_net_pnl = 0.0
    total_mfe = 0.0
    total_r = 0.0
    total_giveback = 0.0
    loser_with_mfe_gt1 = 0
    for row in trade_rows:
        pnl = num(row.get("net_pnl"))
        total_net_pnl += pnl
        day = parse_date(row.get("entry_date"))
        if day is not None and day in date_to_fold:
            fold = date_to_fold[day]
            fold_net[fold] += pnl / max(initial_equity, 1.0)
            fold_trades[fold] += 1
        sector = str(row.get("sector") or "UNKNOWN")
        sector_net[sector] = sector_net.get(sector, 0.0) + pnl / max(initial_equity, 1.0)
        sector_trades[sector] = sector_trades.get(sector, 0) + 1
        by_exit[str(row.get("exit_reason") or "unknown")] = by_exit.get(str(row.get("exit_reason") or "unknown"), 0) + 1
        by_route[str(row.get("entry_route_mode") or row.get("entry_route") or "unknown")] = by_route.get(str(row.get("entry_route_mode") or row.get("entry_route") or "unknown"), 0) + 1
        mfe_r = num(row.get("mfe_r"))
        r = num(row.get("r"))
        total_mfe += mfe_r
        total_r += r
        total_giveback += max(mfe_r - r, 0.0)
        if r < 0.0 and mfe_r > 1.0:
            loser_with_mfe_gt1 += 1
    negative_folds = sum(1 for value in fold_net if value < 0.0)
    worst_sector = min(sector_net.values(), default=0.0)
    negative_sectors = sum(1 for value in sector_net.values() if value < 0.0)
    largest_sector_share = max(sector_trades.values(), default=0) / max(total_trades, 1)
    return {
        "five_fold_net": fold_net,
        "five_fold_trades": fold_trades,
        "five_fold_worst_net": min(fold_net) if fold_net else 0.0,
        "five_fold_median_net": float(np.median(fold_net)) if fold_net else 0.0,
        "five_fold_negative_count": negative_folds,
        "sector_net": dict(sorted(sector_net.items(), key=lambda item: item[1])),
        "sector_trades": dict(sorted(sector_trades.items(), key=lambda item: item[1], reverse=True)),
        "worst_sector_net": worst_sector,
        "negative_sector_count": negative_sectors,
        "largest_sector_trade_share": largest_sector_share,
        "exit_reason_counts": dict(sorted(by_exit.items(), key=lambda item: item[1], reverse=True)),
        "entry_route_mode_counts": dict(sorted(by_route.items(), key=lambda item: item[1], reverse=True)),
        "avg_trade_r": total_r / max(total_trades, 1),
        "avg_mfe_r_from_trades": total_mfe / max(total_trades, 1),
        "avg_giveback_r": total_giveback / max(total_trades, 1),
        "loser_with_mfe_gt1_share": loser_with_mfe_gt1 / max(total_trades, 1),
        "trade_net_return_pct_from_rows": total_net_pnl / max(initial_equity, 1.0),
    }


def stability_score(metrics: dict[str, Any], stability: dict[str, Any], proxy: dict[str, Any]) -> dict[str, Any]:
    net = num(metrics.get("broker_net_return_pct"))
    dd = num(metrics.get("broker_max_drawdown_pct"))
    trades = num(metrics.get("trade_count"))
    capture = num(metrics.get("avg_mfe_capture"))
    worst5 = num(stability.get("five_fold_worst_net"))
    neg_folds = num(stability.get("five_fold_negative_count"))
    worst_sector = num(stability.get("worst_sector_net"))
    neg_sectors = num(stability.get("negative_sector_count"))
    largest_sector_share = num(stability.get("largest_sector_trade_share"))
    proxy_r = num(proxy.get("proxy_positive_r"))
    dd_excess = max(dd - 0.08, 0.0)
    freq_bonus = 18.0 * min(trades / 140.0, 1.5)
    score = (
        100.0 * net
        + freq_bonus
        + 12.0 * max(capture, 0.0)
        + 20.0 * max(worst5, 0.0)
        + 0.00045 * proxy_r
        - 180.0 * max(-worst5, 0.0)
        - 260.0 * dd_excess
        - 8.0 * neg_folds
        - 35.0 * max(-worst_sector, 0.0)
        - 1.5 * neg_sectors
        - 8.0 * max(largest_sector_share - 0.35, 0.0)
    )
    pass_hygiene = (
        net > 0.0
        and dd <= 0.08
        and trades >= 80.0
        and worst5 >= 0.0
        and neg_folds == 0
        and capture >= 0.25
        and num(metrics.get("same_bar_fill_count")) == 0.0
        and num(metrics.get("end_open_position_count")) == 0.0
    )
    return {
        "train_stability_score": score,
        "train_stability_pass": pass_hygiene,
        "frequency_bonus": freq_bonus,
        "dd_excess": dd_excess,
    }


def train_candidate_pool() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    df = opt.read_pipeline()
    train = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    models, train_scores = train_opt.fit_train_only_models(train)
    policy = train_opt.SelectionPolicy(
        name="dataset_all_context_hgb_quality_dataset_top16",
        label="dataset_all_context_hgb_quality",
        scope="dataset",
        budget="top16",
        active_count=16,
        pool_size=16,
        source="train_champion_family",
    )
    feature_rows_train = shared.load_feature_rows("train")
    pool_rows = train_opt.selected_pool_rows(train, train_scores[policy.label], policy, feature_rows_train)
    return pool_rows, {"policy": policy.__dict__, "models": models, "train": train, "holdout": holdout}


def shortlist_filters(pool_rows: list[dict[str, Any]], seed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[FilterSpec]]:
    rows: list[dict[str, Any]] = []
    for spec in filter_specs():
        filtered = apply_filter(pool_rows, spec, active_count=16)
        proxy = proxy_summary(filtered, seed)
        proxy_score = (
            num(proxy.get("proxy_positive_r"))
            + 0.45 * max(num(proxy.get("proxy_net_eod_r")), 0.0)
            - 0.20 * max(-num(proxy.get("proxy_net_eod_r")), 0.0)
            + 5.0 * num(proxy.get("static_route_eligible_count"))
            + 1.5 * num(proxy.get("active_days"))
        )
        rows.append({"filter": spec.__dict__, "proxy": proxy, "proxy_screen_score": proxy_score})
    ranked = sorted(rows, key=lambda row: num(row.get("proxy_screen_score")), reverse=True)
    selected_names: set[str] = {"base_top16", "cpr_ge_065", "quality_ge_75", "sector_intraday_ge_50", "range_atr_lte_2"}
    for row in ranked:
        selected_names.add(str((row.get("filter") or {}).get("name") or ""))
        if len(selected_names) >= 12:
            break
    selected = [spec for spec in filter_specs() if spec.name in selected_names and num(next((row["proxy"]["static_route_eligible_count"] for row in rows if row["filter"]["name"] == spec.name), 0)) >= 40.0]
    return ranked, selected


def run_optimizer() -> dict[str, Any]:
    seed = train_opt.load_seed_mutations()
    train_config, holdout_config = shared.load_base_config()
    pool_rows, context = train_candidate_pool()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "base_train_pool_rows.jsonl", pool_rows)
    proxy_ranked, selected_filters = shortlist_filters(pool_rows, seed)
    log("filter_shortlist_done", filters=len(selected_filters), top_filter=(proxy_ranked[0]["filter"]["name"] if proxy_ranked else ""))

    log("train_context_build_start")
    train_bundle = shared.build_window_bundle(train_config)
    train_dates = list(train_bundle["dataset"].trading_dates)
    initial_equity = float((train_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    log("train_context_build_done", sessions=len(train_dates), contexts=len(train_bundle["context_by_key"]))

    rows: list[dict[str, Any]] = []
    for filter_spec in selected_filters:
        filtered_rows = apply_filter(pool_rows, filter_spec, active_count=16)
        write_jsonl(OUT_DIR / f"pool_rows_train_{filter_spec.name}.jsonl", filtered_rows)
        for risk in risk_specs():
            mutations = mutate_risk(seed, risk)
            proxy = proxy_summary(filtered_rows, mutations)
            if num(proxy.get("static_route_eligible_count")) < 40.0:
                rows.append(
                    {
                        "filter": filter_spec.__dict__,
                        "risk": risk.__dict__,
                        "window": "train",
                        "proxy": proxy,
                        "metrics": {"trade_count": 0.0},
                        "stability": {},
                        "skipped": True,
                        "skip_reason": "static_route_eligible_count_lt_40",
                        "train_stability_score": -999.0,
                        "train_stability_pass": False,
                    }
                )
                log("train_replay_skipped", filter=filter_spec.name, risk=risk.name, eligible=proxy.get("static_route_eligible_count"))
                continue
            log("train_replay_start", filter=filter_spec.name, risk=risk.name, eligible=proxy.get("static_route_eligible_count"))
            replay_name = f"{filter_spec.name}_{risk.name}"
            result = evaluate_compiled_candidate_pool(
                window="train",
                variant=PoolVariant(replay_name, 16, active_count=16),
                config=train_config,
                dataset=train_bundle["dataset"],
                context_by_key=train_bundle["context_by_key"],
                pool_rows=filtered_rows,
                seed_mutations=mutations,
                output_dir=OUT_DIR,
                replay_name=replay_name,
            )
            metrics = dict(result.get("metrics") or {})
            trades = read_trade_rows(result.get("trade_rows_path"))
            stability = stability_metrics(trades, train_dates, initial_equity)
            score = stability_score(metrics, stability, proxy)
            row = {
                "filter": filter_spec.__dict__,
                "risk": risk.__dict__,
                "window": "train",
                "proxy": proxy,
                "metrics": metrics,
                "stability": stability,
                "compiled_replay": result.get("compiled_replay") or {},
                "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
                "trade_rows_path": result.get("trade_rows_path"),
                "skipped": False,
                **score,
            }
            rows.append(row)
            log(
                "train_replay_done",
                filter=filter_spec.name,
                risk=risk.name,
                score=score["train_stability_score"],
                pass_hygiene=score["train_stability_pass"],
                trades=metrics.get("trade_count"),
                net=metrics.get("broker_net_return_pct"),
                dd=metrics.get("broker_max_drawdown_pct"),
                worst5=stability.get("five_fold_worst_net"),
            )

    replayed = [row for row in rows if not row.get("skipped")]
    pass_rows = [row for row in replayed if row.get("train_stability_pass")]
    champion = max(pass_rows or replayed, key=lambda row: num(row.get("train_stability_score"))) if (pass_rows or replayed) else {}
    locked_holdout: dict[str, Any] = {}
    if champion:
        filter_spec = FilterSpec(**dict(champion["filter"]))
        risk = RiskSpec(**dict(champion["risk"]))
        log("locked_holdout_context_build_start", filter=filter_spec.name, risk=risk.name)
        holdout_bundle = shared.build_window_bundle(holdout_config)
        holdout_dates = list(holdout_bundle["dataset"].trading_dates)
        holdout_equity = float((holdout_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
        models = context["models"]
        holdout = context["holdout"]
        policy = train_opt.SelectionPolicy(**dict(context["policy"]))
        model_info = models[policy.label]
        holdout_score = np.asarray(model_info["model"].predict(holdout[model_info["features"]]), dtype=float)
        feature_rows_holdout = shared.load_feature_rows("holdout")
        holdout_pool = train_opt.selected_pool_rows(holdout, holdout_score, policy, feature_rows_holdout)
        holdout_filtered = apply_filter(holdout_pool, filter_spec, active_count=16)
        write_jsonl(OUT_DIR / f"pool_rows_locked_holdout_{filter_spec.name}_{risk.name}.jsonl", holdout_filtered)
        mutations = mutate_risk(seed, risk)
        proxy = proxy_summary(holdout_filtered, mutations)
        log("locked_holdout_replay_start", filter=filter_spec.name, risk=risk.name, rows=len(holdout_filtered), eligible=proxy.get("static_route_eligible_count"))
        result = evaluate_compiled_candidate_pool(
            window="holdout_locked_audit",
            variant=PoolVariant(f"{filter_spec.name}_{risk.name}_locked_holdout", 16, active_count=16),
            config=holdout_config,
            dataset=holdout_bundle["dataset"],
            context_by_key=holdout_bundle["context_by_key"],
            pool_rows=holdout_filtered,
            seed_mutations=mutations,
            output_dir=OUT_DIR,
            replay_name=f"{filter_spec.name}_{risk.name}_locked_holdout",
        )
        metrics = dict(result.get("metrics") or {})
        trades = read_trade_rows(result.get("trade_rows_path"))
        locked_holdout = {
            "selection_basis": "locked_train_selected_filter_and_risk_only_no_holdout_optimization",
            "filter": filter_spec.__dict__,
            "risk": risk.__dict__,
            "proxy": proxy,
            "metrics": metrics,
            "stability": stability_metrics(trades, holdout_dates, holdout_equity),
            "compiled_replay": result.get("compiled_replay") or {},
            "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
            "trade_rows_path": result.get("trade_rows_path"),
        }
        log(
            "locked_holdout_replay_done",
            filter=filter_spec.name,
            risk=risk.name,
            trades=metrics.get("trade_count"),
            net=metrics.get("broker_net_return_pct"),
            dd=metrics.get("broker_max_drawdown_pct"),
        )

    return {
        "created_at_utc": now_iso(),
        "usage_contract": "train_only_filter_risk_stability_optimizer_holdout_locked_after_train_selection",
        "base_policy": context["policy"],
        "objective": {
            "selection_data": "train only",
            "score": "100*train_net + frequency + capture + five_fold_worst + proxy_R - DD/subfold/sector penalties",
            "holdout_policy": "locked audit after train filter/risk champion is selected",
        },
        "proxy_filter_screen": proxy_ranked,
        "selected_filters": [spec.__dict__ for spec in selected_filters],
        "train_rows": rows,
        "train_champion": champion,
        "locked_holdout_audit": locked_holdout,
    }


def route_mix(row: dict[str, Any]) -> str:
    summary = dict(row.get("entry_route_mode_summary") or {})
    return "; ".join(f"{mode}:{int(num((data or {}).get('trades')))}" for mode, data in summary.items())


def render_report(summary: dict[str, Any]) -> str:
    rows = [row for row in summary.get("train_rows", []) if not row.get("skipped")]
    ranked = sorted(rows, key=lambda row: num(row.get("train_stability_score")), reverse=True)
    champion = dict(summary.get("train_champion") or {})
    lines = [
        "# KALCB Train Stability / Route Optimizer",
        "",
        "Train-only targeted filter and risk sweep for the `dataset_all_context_hgb_quality top16` family. Holdout is locked until after train selection.",
        "",
        "## Train Champion",
        "",
    ]
    if champion:
        filt = dict(champion.get("filter") or {})
        risk = dict(champion.get("risk") or {})
        metrics = dict(champion.get("metrics") or {})
        stability = dict(champion.get("stability") or {})
        proxy = dict(champion.get("proxy") or {})
        lines.extend(
            [
                f"- Filter: `{filt.get('name')}` ({filt.get('description')})",
                f"- Risk: `{risk.get('name')}`",
                f"- Train stability score: {num(champion.get('train_stability_score')):.2f}; pass: {champion.get('train_stability_pass')}",
                f"- Train net/DD/trades: {pct(metrics.get('broker_net_return_pct'))} / {pct(metrics.get('broker_max_drawdown_pct'))} / {num(metrics.get('trade_count')):.0f}",
                f"- Train capture/five-fold worst/proxy R: {pct(metrics.get('avg_mfe_capture'))} / {pct(stability.get('five_fold_worst_net'))} / {num(proxy.get('proxy_positive_r')):.1f}R",
                f"- Sector worst/largest sector share: {pct(stability.get('worst_sector_net'))} / {pct(stability.get('largest_sector_trade_share'))}",
                f"- Route mix: {route_mix(champion)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Train Replays",
            "",
            "| rank | pass | filter | risk | score | net | DD | trades | 5-fold worst | neg folds | capture | proxy R | worst sector | routes |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(ranked[:18], start=1):
        filt = dict(row.get("filter") or {})
        risk = dict(row.get("risk") or {})
        metrics = dict(row.get("metrics") or {})
        stability = dict(row.get("stability") or {})
        proxy = dict(row.get("proxy") or {})
        lines.append(
            f"| {rank} | {row.get('train_stability_pass')} | `{filt.get('name')}` | `{risk.get('name')}` | "
            f"{num(row.get('train_stability_score')):.2f} | {pct(metrics.get('broker_net_return_pct'))} | "
            f"{pct(metrics.get('broker_max_drawdown_pct'))} | {num(metrics.get('trade_count')):.0f} | "
            f"{pct(stability.get('five_fold_worst_net'))} | {num(stability.get('five_fold_negative_count')):.0f} | "
            f"{pct(metrics.get('avg_mfe_capture'))} | {num(proxy.get('proxy_positive_r')):.1f} | "
            f"{pct(stability.get('worst_sector_net'))} | {route_mix(row)} |"
        )
    lines.extend(["", "## Proxy Filter Screen", ""])
    lines.append("| rank | filter | score | active rows | eligible | proxy R | proxy EOD R |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(list(summary.get("proxy_filter_screen") or [])[:12], start=1):
        filt = dict(row.get("filter") or {})
        proxy = dict(row.get("proxy") or {})
        lines.append(
            f"| {rank} | `{filt.get('name')}` | {num(row.get('proxy_screen_score')):.1f} | "
            f"{num(proxy.get('active_rows')):.0f} | {num(proxy.get('static_route_eligible_count')):.0f} | "
            f"{num(proxy.get('proxy_positive_r')):.1f} | {num(proxy.get('proxy_net_eod_r')):.1f} |"
        )
    lines.extend(["", "## Locked Holdout Audit", ""])
    audit = dict(summary.get("locked_holdout_audit") or {})
    if audit:
        filt = dict(audit.get("filter") or {})
        risk = dict(audit.get("risk") or {})
        metrics = dict(audit.get("metrics") or {})
        stability = dict(audit.get("stability") or {})
        proxy = dict(audit.get("proxy") or {})
        lines.extend(
            [
                f"- Train-selected filter/risk: `{filt.get('name')}` + `{risk.get('name')}`",
                f"- Holdout net/DD/trades: {pct(metrics.get('broker_net_return_pct'))} / {pct(metrics.get('broker_max_drawdown_pct'))} / {num(metrics.get('trade_count')):.0f}",
                f"- Holdout capture/five-fold worst/proxy R: {pct(metrics.get('avg_mfe_capture'))} / {pct(stability.get('five_fold_worst_net'))} / {num(proxy.get('proxy_positive_r')):.1f}R",
                f"- Route mix: {route_mix(audit)}",
                "- Locked audit note: this was not used to choose the train champion.",
            ]
        )
    else:
        lines.append("- No locked holdout audit was run.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Promotion direction should follow the train champion only if it improves the train stability score versus the unfiltered base.",
            "- Filters that improve proxy R but reduce replay frequency or five-fold stability are rejected quantitatively here, not by hypothesis.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("start")
    summary = run_optimizer()
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary_path = OUT_DIR / "kalcb_train_stability_route_optimizer_results.json"
    report_path = OUT_DIR / "kalcb_train_stability_route_optimizer_report.md"
    write_json(summary_path, summary)
    report_path.write_text(render_report(summary), encoding="utf-8")
    log("complete", summary=str(summary_path), report=str(report_path), elapsed_seconds=summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
