from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_SCRIPT = REPO_ROOT / "scripts" / "kalcb_interaction_route_conversion_optimizer.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "fast_extensive_search"


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
class ScoreSpec:
    name: str
    kind: str
    feature_set: str = ""
    model_kind: str = ""
    target: str = ""


@dataclass(frozen=True)
class PoolSpec:
    name: str
    score_name: str
    budget: str
    pool_size: int
    active_count: int
    score_kind: str


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


def z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return values * 0.0
    return (values - float(values.mean())) / std


def day_rank(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).groupby(df["trade_date"]).rank(pct=True, method="average")


def formula_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    s: dict[str, pd.Series] = {}
    def g(key: str) -> pd.Series:
        if key in df.columns:
            return pd.to_numeric(df[key], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index, dtype=float)
    s["f_intraday_context"] = z(g("ix_intraday_quality")) + 0.85 * z(g("ix_continuation_context")) + 0.25 * z(g("ix_flow_confirm"))
    s["f_routeable_context"] = z(g("ix_routeability_proxy")) + 0.75 * z(g("ix_continuation_context")) + 0.35 * z(g("ix_sector_confirm"))
    s["f_gap_sector_rvol"] = z(g("ix_gap_quality")) + 0.70 * z(g("ix_sector_intraday_rvol")) + 0.30 * z(g("first30_gap_retention_sector_breadth"))
    s["f_close_vwap_pressure"] = z(g("ix_close_position_pressure")) + 0.75 * z(g("ix_vwap_cpr_rvol")) + 0.30 * z(g("first30_vwap_ret"))
    s["f_trend_flow_sector"] = z(g("ix_trend_flow_sector")) + 0.50 * z(g("ix_daily_strength")) + 0.40 * z(g("sector_daily_score_pct"))
    s["f_leadership_continuation"] = z(g("ix_sector_leadership_stack")) + 0.60 * z(g("continuation_joint_quality_pct")) + 0.35 * z(g("ix_market_adjusted_momentum"))
    s["f_low_extension_quality"] = z(g("ix_intraday_quality")) + 0.60 * z(g("ix_routeability_proxy")) - 0.45 * z(g("ix_breakout_extension_pressure")) - 0.30 * z(g("ix_gap_reversal_risk"))
    s["f_rvol_cpr_range"] = z(g("first30_rel_volume")) + 0.70 * z(g("first30_signal_bar_cpr")) - 0.45 * z(g("first30_range_atr")) + 0.35 * z(g("first30_ret"))
    s["f_sector_confirmed_pullback"] = z(g("sector_intraday_score_pct")) + 0.55 * z(g("first30_gap_retention_ratio")) - 0.25 * z(g("first30_ret")) + 0.20 * z(g("daily_close20_loc"))
    s["f_daily_breakout_context"] = z(g("daily_momentum_pct")) + 0.60 * z(g("daily_close20_loc")) + 0.40 * z(g("stock_sector_daily_ret20_spread")) + 0.30 * z(g("first30_sector_ret_spread"))
    s["f_liquidity_momentum"] = z(g("ix_liquidity_momentum")) + 0.55 * z(g("daily_adv20_krw_log")) + 0.35 * z(g("first30_rel_volume"))
    s["f_balanced_alpha"] = 0.55 * s["f_intraday_context"] + 0.45 * s["f_routeable_context"] + 0.35 * s["f_trend_flow_sector"] - 0.25 * z(g("negative_mae_abs"))
    s["f_balanced_no_mae"] = 0.55 * s["f_intraday_context"] + 0.45 * s["f_routeable_context"] + 0.35 * s["f_trend_flow_sector"]
    s["f_dayrank_routeable"] = day_rank(g("ix_routeability_proxy"), df) + 0.75 * day_rank(g("ix_continuation_context"), df) + 0.35 * day_rank(g("ix_gap_quality"), df)
    s["f_dayrank_momentum"] = day_rank(g("ix_intraday_quality"), df) + 0.60 * day_rank(g("first30_rel_volume"), df) + 0.45 * day_rank(g("sector_intraday_score_pct"), df)
    s["f_dayrank_low_extension"] = day_rank(g("ix_intraday_quality"), df) + 0.50 * day_rank(g("ix_routeability_proxy"), df) - 0.45 * day_rank(g("ix_breakout_extension_pressure"), df)
    return {name: values.to_numpy(dtype=float) for name, values in s.items()}


def score_specs() -> list[ScoreSpec]:
    specs = [ScoreSpec(name, "formula") for name in formula_scores(pd.DataFrame({"trade_date": []})).keys()]
    specs.extend(
        [
            ScoreSpec("ml_hgb_base_quality", "ml", "base_all_context", "hgb", "alpha_quality"),
            ScoreSpec("ml_hgb_interaction_quality", "ml", "interaction_full", "hgb", "alpha_quality"),
            ScoreSpec("ml_ridge_interaction_quality", "ml", "interaction_full", "ridge", "alpha_quality"),
        ]
    )
    if os.environ.get("KALCB_INCLUDE_ROUTEABLE_ML", "0") == "1":
        specs.append(ScoreSpec("ml_hgb_interaction_routeable", "ml", "interaction_route", "hgb", "routeable_alpha"))
    if os.environ.get("KALCB_INCLUDE_TREE_ML", "0") == "1":
        specs.append(ScoreSpec("ml_trees_interaction_quality", "ml", "interaction_full", "trees", "alpha_quality"))
    return specs


def chronological_folds(df: pd.DataFrame, folds: int = 5) -> list[np.ndarray]:
    dates = np.array(sorted(str(day)[:10] for day in df["trade_date"].dropna().unique()))
    masks: list[np.ndarray] = []
    for fold in range(folds):
        lo = int(fold * len(dates) / folds)
        hi = int((fold + 1) * len(dates) / folds)
        fold_dates = set(dates[lo:hi])
        masks.append(df["trade_date"].astype(str).str[:10].isin(fold_dates).to_numpy())
    return masks


def fit_oof_score(train: pd.DataFrame, holdout: pd.DataFrame, spec: ScoreSpec, folds: int) -> dict[str, Any]:
    features = base.feature_columns(train, spec.feature_set)
    scope_mask = opt.scope_mask(train, "dataset")
    scores = np.full(len(train), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, valid_mask in enumerate(chronological_folds(train, folds)):
        train_mask = scope_mask & ~valid_mask
        model = base.make_model(spec.model_kind)
        log("ml_oof_fit_start", name=spec.name, fold=fold_index, rows=int(train_mask.sum()), features=len(features))
        model.fit(train.loc[train_mask, features], base.target_values(train.loc[train_mask], spec.target))
        preds = np.asarray(model.predict(train.loc[valid_mask, features]), dtype=float)
        scores[valid_mask] = preds
        fold_rows.append({"fold": fold_index, "rows": int(valid_mask.sum())})
        log("ml_oof_fit_done", name=spec.name, fold=fold_index)
    scores = np.nan_to_num(scores, nan=float(np.nanmedian(scores) if np.isfinite(np.nanmedian(scores)) else 0.0))
    full_model = base.make_model(spec.model_kind)
    full_model.fit(train.loc[scope_mask, features], base.target_values(train.loc[scope_mask], spec.target))
    return {
        "train_score": scores,
        "holdout_score": np.asarray(full_model.predict(holdout[features]), dtype=float),
        "features": features,
        "fold_rows": fold_rows,
        "model": full_model,
    }


def select_by_score(part: pd.DataFrame, score: np.ndarray, budget: str) -> pd.DataFrame:
    work = part.copy()
    work["_score"] = np.asarray(score, dtype=float)
    k = int(budget.replace("top", "")) if budget.startswith("top") else 16
    pieces = [group.sort_values("_score", ascending=False).head(k) for _, group in work.groupby("trade_date", sort=True)]
    return pd.concat(pieces, ignore_index=False) if pieces else work.head(0)


def build_pool_rows(
    part: pd.DataFrame,
    score: np.ndarray,
    score_name: str,
    score_kind: str,
    budget: str,
    feature_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[PoolSpec, list[dict[str, Any]]]:
    selected = select_by_score(part, score, budget)
    pool_size = int(budget.replace("top", "")) if budget.startswith("top") else 16
    active_count = min(16, pool_size)
    name = f"{score_name}_{budget}_a{active_count}"
    spec = PoolSpec(name, score_name, budget, pool_size, active_count, score_kind)
    rows = shared.selected_pool_rows(
        selected=selected,
        feature_by_key=feature_by_key,
        candidate=shared.ReplayCandidate(name, score_name, "dataset", budget, active_count, pool_size, "train_only", "fast_extensive_search"),
    )
    return spec, rows


def pool_signature(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        if bool(row.get("pool_active")):
            parts.append(f"{str(row.get('trade_date'))[:10]}:{str(row.get('symbol')).zfill(6)}:{int(num(row.get('pool_rank')))}")
    return "|".join(parts)


def fold_proxy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if bool(row.get("pool_active"))]
    dates = sorted({str(row.get("trade_date") or "")[:10] for row in active})
    fold_by_day: dict[str, int] = {}
    for index, day in enumerate(dates):
        fold_by_day[day] = min(4, int(index * 5 / max(len(dates), 1)))
    fold_r = [0.0] * 5
    fold_net = [0.0] * 5
    for row in active:
        fold = fold_by_day.get(str(row.get("trade_date") or "")[:10], 0)
        fold_r[fold] += num(row.get("optimizer_positive_r_proxy"))
        fold_net[fold] += num(row.get("optimizer_net_eod_r_proxy"))
    return {
        "fold_active_positive_r": fold_r,
        "fold_active_net_eod_r": fold_net,
        "fold_active_positive_r_min": min(fold_r) if fold_r else 0.0,
        "fold_active_net_eod_r_min": min(fold_net) if fold_net else 0.0,
    }


def rows_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    keys = [
        "trade_date",
        "symbol",
        "pool_active",
        "pool_rank",
        "optimizer_positive_r_proxy",
        "optimizer_net_eod_r_proxy",
        "first30_ret",
        "first30_vwap_ret",
        "first30_signal_bar_cpr",
        "first30_range_close_location",
        "first30_rel_volume",
        "first30_range_atr",
        "first30_open_drawdown",
        "first30_low_vs_prev_close",
        "first30_gap_retention_ratio",
        "sector_intraday_score_pct",
        "daily_close20_loc",
    ]
    frame = pd.DataFrame([{key: row.get(key) for key in keys} for row in rows])
    for key in keys:
        if key not in {"trade_date", "symbol", "pool_active"} and key in frame.columns:
            frame[key] = pd.to_numeric(frame[key], errors="coerce").fillna(0.0)
    frame["pool_active"] = frame.get("pool_active", False).astype(bool)
    return frame


def _vote_mask(frame: pd.DataFrame, *, q: int, min_bar: float, min_vwap: float, cpr: float, rvol: float, max_range: float = 99.0) -> pd.Series:
    votes = (
        (frame["first30_ret"] >= min_bar).astype(int)
        + (frame["first30_vwap_ret"] >= min_vwap).astype(int)
        + (frame["first30_signal_bar_cpr"] >= cpr).astype(int)
        + (frame["first30_rel_volume"] >= rvol).astype(int)
        + (frame["first30_range_atr"] <= max_range).astype(int)
        + (frame["sector_intraday_score_pct"] >= 50.0).astype(int)
    )
    return votes >= int(q)


def fast_route_proxy_summary(rows: list[dict[str, Any]], route_name: str) -> dict[str, Any]:
    frame = rows_frame(rows)
    if frame.empty:
        return {
            "pool_rows": 0,
            "active_rows": 0,
            "active_days": 0,
            "active_proxy_positive_r": 0.0,
            "active_proxy_net_eod_r": 0.0,
            "route_eligible_count": 0,
            "route_eligible_active_count": 0,
            "route_eligible_proxy_positive_r": 0.0,
            "route_eligible_active_proxy_positive_r": 0.0,
            "route_eligible_proxy_net_eod_r": 0.0,
            "route_eligible_share": 0.0,
            "route_eligible_by_mode": {},
            "top_route_blockers": [],
            "proxy_mode": "fast_vectorized",
        }
    active = frame["pool_active"]
    rank = frame["pool_rank"].astype(float)

    if route_name == "seed_risk99":
        eligible = active & (frame["first30_ret"] >= 0.01) & (frame["first30_vwap_ret"] >= 0.0)
        mode = "first30_open"
    elif route_name.startswith("first30_soft_quality5") or route_name.startswith("first30_soft_risk80"):
        eligible = active & _vote_mask(frame, q=5, min_bar=0.0, min_vwap=-0.005, cpr=0.65, rvol=1.25, max_range=99.0)
        mode = "first30_open"
    elif route_name.startswith("first30_soft_quality4"):
        eligible = active & _vote_mask(frame, q=4, min_bar=-0.003, min_vwap=-0.008, cpr=0.60, rvol=1.00, max_range=99.0)
        mode = "first30_open"
    elif route_name.startswith("first30_q"):
        match = re.search(r"first30_q(\d+)_risk", route_name)
        q = int(match.group(1)) if match else 4
        if q <= 3:
            eligible = active & _vote_mask(frame, q=3, min_bar=-0.006, min_vwap=-0.012, cpr=0.55, rvol=0.75)
        elif q == 4:
            eligible = active & _vote_mask(frame, q=4, min_bar=-0.003, min_vwap=-0.008, cpr=0.60, rvol=1.00)
        else:
            eligible = active & _vote_mask(frame, q=5, min_bar=0.0, min_vwap=-0.005, cpr=0.65, rvol=1.25)
        mode = "first30_open"
    elif route_name.startswith("delayed"):
        rank_match = re.search(r"rank(\d+)", route_name)
        q_match = re.search(r"_q(\d+)", route_name)
        max_rank = int(rank_match.group(1)) if rank_match else 16
        q = int(q_match.group(1)) if q_match else (4 if "q4" in route_name else 5)
        delayed = (rank > 0) & (rank <= max_rank) & _vote_mask(frame, q=q, min_bar=-0.005, min_vwap=-0.010, cpr=0.60 if q <= 4 else 0.65, rvol=1.00 if q <= 4 else 1.25)
        first30 = active & _vote_mask(frame, q=5, min_bar=0.0, min_vwap=-0.005, cpr=0.65, rvol=1.25)
        eligible = delayed | first30
        mode = "delayed_family"
    else:
        eligible = active & (frame["first30_ret"] >= 0.0) & (frame["first30_vwap_ret"] >= -0.005)
        mode = "approx"

    eligible_active = eligible & active
    blockers = int((~eligible).sum())
    return {
        "pool_rows": int(len(frame)),
        "active_rows": int(active.sum()),
        "active_days": int(frame.loc[active, "trade_date"].astype(str).str[:10].nunique()),
        "active_proxy_positive_r": float(frame.loc[active, "optimizer_positive_r_proxy"].sum()),
        "active_proxy_net_eod_r": float(frame.loc[active, "optimizer_net_eod_r_proxy"].sum()),
        "route_eligible_count": int(eligible.sum()),
        "route_eligible_active_count": int(eligible_active.sum()),
        "route_eligible_proxy_positive_r": float(frame.loc[eligible, "optimizer_positive_r_proxy"].sum()),
        "route_eligible_active_proxy_positive_r": float(frame.loc[eligible_active, "optimizer_positive_r_proxy"].sum()),
        "route_eligible_proxy_net_eod_r": float(frame.loc[eligible, "optimizer_net_eod_r_proxy"].sum()),
        "route_eligible_share": float(eligible.mean()),
        "route_eligible_by_mode": {mode: int(eligible.sum())},
        "top_route_blockers": [["fast_proxy_not_eligible", blockers]],
        "proxy_mode": "fast_vectorized",
    }


def custom_route_specs() -> list[Any]:
    routes = list(base.route_specs())
    seen = {route.name for route in routes}

    def add(route: Any) -> None:
        if route.name not in seen:
            routes.append(route)
            seen.add(route.name)

    for q, min_bar, min_vwap, cpr, rvol in (
        (3, -0.006, -0.012, 0.55, 0.75),
        (4, -0.003, -0.008, 0.60, 1.00),
        (5, 0.000, -0.005, 0.65, 1.25),
    ):
        for risk in (0.50, 0.65, 0.80, 0.99):
            name = f"first30_q{q}_risk{int(risk * 100)}"
            add(
                base.RouteSpec(
                    name,
                    f"First30 soft quality {q}, risk {risk:.2f}.",
                    lambda seed, risk=risk, q=q, min_bar=min_bar, min_vwap=min_vwap, cpr=cpr, rvol=rvol: base.set_first30_route(
                        seed,
                        risk=risk,
                        min_bar=min_bar,
                        min_vwap=min_vwap,
                        min_votes=q,
                        q_cpr=cpr,
                        q_rvol=rvol,
                        q_range_min=0.35,
                    ),
                )
            )
    for rank in (8, 12, 16, 24):
        for delayed_risk in (0.05, 0.10, 0.15):
            for q in (4, 5):
                name = f"delayed_rank{rank}_r{int(delayed_risk * 100)}_q{q}"
                add(
                    base.RouteSpec(
                        name,
                        f"Soft first30 plus delayed route family rank {rank}, delayed risk {delayed_risk:.2f}, q{q}.",
                        lambda seed, rank=rank, delayed_risk=delayed_risk, q=q: base.with_delayed_bundle(
                            seed,
                            first30_risk=0.65,
                            delayed_risk=delayed_risk,
                            rank=rank,
                            quality_votes=q,
                            include_deferred=True,
                            soft_first30=True,
                        ),
                    )
                )
    return routes


def proxy_search_score(row: dict[str, Any]) -> float:
    proxy = dict(row.get("proxy") or {})
    fold = dict(row.get("fold_proxy") or {})
    eligible_r = num(proxy.get("route_eligible_proxy_positive_r"))
    eligible_net = num(proxy.get("route_eligible_proxy_net_eod_r"))
    active_r = num(proxy.get("active_proxy_positive_r"))
    active_net = num(proxy.get("active_proxy_net_eod_r"))
    eligible = num(proxy.get("route_eligible_count"))
    active_eligible = num(proxy.get("route_eligible_active_count"))
    fold_min_r = num(fold.get("fold_active_positive_r_min"))
    fold_min_net = num(fold.get("fold_active_net_eod_r_min"))
    return (
        eligible_r
        + 0.30 * active_r
        + 0.35 * max(eligible_net, 0.0)
        - 0.25 * max(-eligible_net, 0.0)
        + 0.10 * max(active_net, 0.0)
        - 0.08 * max(-active_net, 0.0)
        + 5.0 * eligible
        + 2.5 * active_eligible
        + 0.20 * fold_min_r
        + 0.10 * fold_min_net
    )


def ranker_and_proxy_stage(train: pd.DataFrame, holdout: pd.DataFrame, feature_rows_train: dict[tuple[str, str], dict[str, Any]], *, folds: int) -> dict[str, Any]:
    formula = formula_scores(train)
    formula_holdout = formula_scores(holdout)
    score_payloads: dict[str, dict[str, Any]] = {
        name: {"spec": ScoreSpec(name, "formula").__dict__, "train_score": score, "holdout_score": formula_holdout.get(name), "features": [], "fold_rows": []}
        for name, score in formula.items()
    }
    for spec in [item for item in score_specs() if item.kind == "ml"]:
        score_payloads[spec.name] = {"spec": spec.__dict__, **fit_oof_score(train, holdout, spec, folds)}

    budgets = ["top8", "top12", "top16", "top24"]
    pool_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    pool_specs: dict[str, dict[str, Any]] = {}
    signatures: dict[str, str] = {}
    for score_name, payload in score_payloads.items():
        for budget in budgets:
            pool_spec, rows = build_pool_rows(train, payload["train_score"], score_name, str(payload["spec"]["kind"]), budget, feature_rows_train)
            signature = pool_signature(rows)
            if signature in signatures.values():
                continue
            pool_rows_by_name[pool_spec.name] = rows
            pool_specs[pool_spec.name] = pool_spec.__dict__
            signatures[pool_spec.name] = signature
            if os.environ.get("KALCB_WRITE_ALL_POOLS", "0") == "1":
                base.write_jsonl(OUT_DIR / f"pool_rows_train_{pool_spec.name}.jsonl", rows)
    log("pool_stage_done", scores=len(score_payloads), unique_pools=len(pool_rows_by_name))

    seed = train_opt.load_seed_mutations()
    routes = custom_route_specs()
    route_rows: list[dict[str, Any]] = []
    for pool_name, rows in pool_rows_by_name.items():
        fold = fold_proxy(rows)
        for route in routes:
            proxy = fast_route_proxy_summary(rows, route.name)
            row = {
                "pool_name": pool_name,
                "pool_spec": pool_specs[pool_name],
                "route": {"name": route.name, "description": route.description},
                "proxy": proxy,
                "fold_proxy": fold,
            }
            row["proxy_search_score"] = proxy_search_score(row)
            route_rows.append(row)
    ranked = sorted(route_rows, key=lambda row: num(row.get("proxy_search_score")), reverse=True)
    write_json(OUT_DIR / "fast_extensive_proxy_screen.json", ranked)
    log("proxy_screen_done", routes=len(routes), combos=len(ranked), top_score=ranked[0]["proxy_search_score"] if ranked else 0.0)
    return {
        "score_payloads": score_payloads,
        "pool_rows_by_name": pool_rows_by_name,
        "pool_specs": pool_specs,
        "route_specs": {route.name: route for route in routes},
        "proxy_ranked": ranked,
    }


def selected_replay_pairs(ranked: list[dict[str, Any]], replay_top: int) -> list[tuple[str, str]]:
    max_eligible = int(os.environ.get("KALCB_MAX_REPLAY_ELIGIBLE", "1200") or "1200")
    route_filter = os.environ.get("KALCB_ROUTE_FILTER", "").strip()
    forced = [("ml_hgb_base_quality_top16_a16", "seed_risk99")]
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in forced:
        out.append(pair)
        seen.add(pair)
    used_pools: set[str] = set()
    for row in ranked:
        pool = str(row.get("pool_name") or "")
        route = str((row.get("route") or {}).get("name") or "")
        proxy = dict(row.get("proxy") or {})
        if not pool or not route or (pool, route) in seen:
            continue
        if route_filter and route_filter not in route:
            continue
        eligible = num(proxy.get("route_eligible_count"))
        if eligible > max_eligible:
            continue
        if pool in used_pools and len(out) < max(3, replay_top - 1):
            continue
        out.append((pool, route))
        seen.add((pool, route))
        used_pools.add(pool)
        if len(out) >= replay_top:
            break
    return out[:replay_top]


def run_replays(stage: dict[str, Any], train_config: dict[str, Any], holdout_config: dict[str, Any], feature_rows_holdout: dict[tuple[str, str], dict[str, Any]], holdout: pd.DataFrame, *, replay_top: int, locked_holdout: bool) -> dict[str, Any]:
    if replay_top <= 0:
        return {"train_rows": [], "train_champion": {}, "locked_holdout_audit": {}}
    seed = train_opt.load_seed_mutations()
    pairs = selected_replay_pairs(stage["proxy_ranked"], replay_top)
    log("replay_selection_done", pairs=len(pairs), selected=pairs)
    log("train_context_build_start")
    train_bundle = shared.build_window_bundle(train_config)
    train_dates = list(train_bundle["dataset"].trading_dates)
    initial_equity = float((train_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    log("train_context_build_done", sessions=len(train_dates), contexts=len(train_bundle["context_by_key"]))

    rows: list[dict[str, Any]] = []
    route_specs = stage["route_specs"]
    max_eligible = int(os.environ.get("KALCB_MAX_REPLAY_ELIGIBLE", "1200") or "1200")
    for pool_name, route_name in pairs:
        pool_rows = stage["pool_rows_by_name"].get(pool_name) or []
        route = route_specs[route_name]
        proxy = base.route_proxy_summary(pool_rows, route.build(seed))
        if num(proxy.get("route_eligible_count")) > max_eligible:
            log("train_replay_skipped_eligible_cap", pool=pool_name, route=route_name, eligible=proxy.get("route_eligible_count"), cap=max_eligible)
            continue
        replay_name = f"{pool_name}_{route_name}"
        log("train_replay_start", pool=pool_name, route=route_name, eligible=proxy.get("route_eligible_count"), proxy_r=proxy.get("route_eligible_proxy_positive_r"))
        result = base.evaluate_compiled_candidate_pool(
            window="train",
            variant=base.PoolVariant(replay_name, int((stage["pool_specs"].get(pool_name) or {}).get("pool_size", 16)), active_count=int((stage["pool_specs"].get(pool_name) or {}).get("active_count", 16))),
            config=train_config,
            dataset=train_bundle["dataset"],
            context_by_key=train_bundle["context_by_key"],
            pool_rows=pool_rows,
            seed_mutations=route.build(seed),
            output_dir=OUT_DIR,
            replay_name=replay_name,
        )
        metrics = dict(result.get("metrics") or {})
        trades = base.read_trade_rows(result.get("trade_rows_path"))
        stability = base.stability_metrics(trades, train_dates, initial_equity)
        score = base.alpha_conversion_score(metrics, stability, proxy)
        row = {
            "pool_name": pool_name,
            "pool_spec": stage["pool_specs"].get(pool_name) or {},
            "route": {"name": route.name, "description": route.description},
            "proxy": proxy,
            "metrics": metrics,
            "stability": stability,
            "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
            "trade_rows_path": result.get("trade_rows_path"),
            **score,
        }
        rows.append(row)
        log("train_replay_done", pool=pool_name, route=route_name, score=score["train_alpha_conversion_score"], pass_hygiene=score["train_alpha_conversion_pass"], trades=metrics.get("trade_count"), net=metrics.get("broker_net_return_pct"), dd=metrics.get("broker_max_drawdown_pct"))

    pass_rows = [row for row in rows if row.get("train_alpha_conversion_pass")]
    champion = max(pass_rows or rows, key=lambda row: num(row.get("train_alpha_conversion_score"))) if rows else {}
    audit: dict[str, Any] = {}
    if locked_holdout and champion:
        pool_spec = stage["pool_specs"].get(str(champion.get("pool_name"))) or {}
        score_name = str(pool_spec.get("score_name") or "")
        route = route_specs[str((champion.get("route") or {}).get("name") or "")]
        score_payload = stage["score_payloads"][score_name]
        if score_payload.get("holdout_score") is not None:
            holdout_pool_spec, holdout_pool = build_pool_rows(
                holdout,
                score_payload["holdout_score"],
                score_name,
                str(pool_spec.get("score_kind") or "unknown"),
                str(pool_spec.get("budget") or "top16"),
                feature_rows_holdout,
            )
            base.write_jsonl(OUT_DIR / f"pool_rows_locked_holdout_{holdout_pool_spec.name}_{route.name}.jsonl", holdout_pool)
            log("locked_holdout_context_build_start", pool=holdout_pool_spec.name, route=route.name)
            holdout_bundle = shared.build_window_bundle(holdout_config)
            holdout_dates = list(holdout_bundle["dataset"].trading_dates)
            holdout_equity = float((holdout_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
            proxy = base.route_proxy_summary(holdout_pool, route.build(seed))
            log("locked_holdout_replay_start", pool=holdout_pool_spec.name, route=route.name, eligible=proxy.get("route_eligible_count"))
            result = base.evaluate_compiled_candidate_pool(
                window="holdout_locked_audit",
                variant=base.PoolVariant(f"{holdout_pool_spec.name}_{route.name}_locked_holdout", holdout_pool_spec.pool_size, active_count=holdout_pool_spec.active_count),
                config=holdout_config,
                dataset=holdout_bundle["dataset"],
                context_by_key=holdout_bundle["context_by_key"],
                pool_rows=holdout_pool,
                seed_mutations=route.build(seed),
                output_dir=OUT_DIR,
                replay_name=f"{holdout_pool_spec.name}_{route.name}_locked_holdout",
            )
            metrics = dict(result.get("metrics") or {})
            trades = base.read_trade_rows(result.get("trade_rows_path"))
            audit = {
                "selection_basis": "locked_train_selected_fast_extensive_search_no_holdout_optimization",
                "pool_spec": holdout_pool_spec.__dict__,
                "route": {"name": route.name, "description": route.description},
                "proxy": proxy,
                "metrics": metrics,
                "stability": base.stability_metrics(trades, holdout_dates, holdout_equity),
                "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
                "trade_rows_path": result.get("trade_rows_path"),
            }
            log("locked_holdout_replay_done", pool=holdout_pool_spec.name, route=route.name, trades=metrics.get("trade_count"), net=metrics.get("broker_net_return_pct"), dd=metrics.get("broker_max_drawdown_pct"))
    return {"train_rows": rows, "train_champion": champion, "locked_holdout_audit": audit}


def route_mix(row: dict[str, Any]) -> str:
    return base.route_mix(row)


def render_report(summary: dict[str, Any]) -> str:
    ranked_proxy = list(summary.get("proxy_ranked") or [])
    replayed = sorted(list(summary.get("train_rows") or []), key=lambda row: num(row.get("train_alpha_conversion_score")), reverse=True)
    champion = dict(summary.get("train_champion") or {})
    lines = [
        "# KALCB Fast Extensive Search",
        "",
        "Train-only broad interaction/route proxy search with a small shared-core replay finalist set.",
        "",
        "## Runtime Fix",
        "",
        "- Full shared-core replay is no longer the search primitive; it is reserved for final train-screened candidates.",
        "- The tree model is single-process to avoid CPU oversubscription/orphan workers.",
        f"- Proxy combos screened: {len(ranked_proxy)}; shared-core train replays: {len(replayed)}.",
        "",
        "## Train Replay Champion",
        "",
    ]
    if champion:
        pool = dict(champion.get("pool_spec") or {})
        route = dict(champion.get("route") or {})
        metrics = dict(champion.get("metrics") or {})
        stability = dict(champion.get("stability") or {})
        proxy = dict(champion.get("proxy") or {})
        lines.extend(
            [
                f"- Pool: `{pool.get('name')}` ({pool.get('score_kind')}, {pool.get('budget')})",
                f"- Route: `{route.get('name')}`",
                f"- Train score/pass: {num(champion.get('train_alpha_conversion_score')):.2f} / {champion.get('train_alpha_conversion_pass')}",
                f"- Train net/DD/trades: {pct(metrics.get('broker_net_return_pct'))} / {pct(metrics.get('broker_max_drawdown_pct'))} / {num(metrics.get('trade_count')):.0f}",
                f"- Train capture/five-fold worst/eligible proxy R: {pct(metrics.get('avg_mfe_capture'))} / {pct(stability.get('five_fold_worst_net'))} / {num(proxy.get('route_eligible_proxy_positive_r')):.1f}R",
                f"- Route mix: {route_mix(champion)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Train Replay Finalists",
            "",
            "| rank | pass | pool | route | score | net | DD | trades | 5-fold worst | capture | eligible R | routes |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(replayed, start=1):
        pool = dict(row.get("pool_spec") or {})
        route = dict(row.get("route") or {})
        metrics = dict(row.get("metrics") or {})
        stability = dict(row.get("stability") or {})
        proxy = dict(row.get("proxy") or {})
        lines.append(
            f"| {rank} | {row.get('train_alpha_conversion_pass')} | `{pool.get('name')}` | `{route.get('name')}` | "
            f"{num(row.get('train_alpha_conversion_score')):.2f} | {pct(metrics.get('broker_net_return_pct'))} | "
            f"{pct(metrics.get('broker_max_drawdown_pct'))} | {num(metrics.get('trade_count')):.0f} | "
            f"{pct(stability.get('five_fold_worst_net'))} | {pct(metrics.get('avg_mfe_capture'))} | "
            f"{num(proxy.get('route_eligible_proxy_positive_r')):.1f} | {route_mix(row)} |"
        )
    lines.extend(["", "## Top Proxy Search", ""])
    lines.append("| rank | pool | route | score | eligible | active elig | eligible R | active R | fold min R |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(ranked_proxy[:25], start=1):
        proxy = dict(row.get("proxy") or {})
        fold = dict(row.get("fold_proxy") or {})
        lines.append(
            f"| {rank} | `{row.get('pool_name')}` | `{(row.get('route') or {}).get('name')}` | "
            f"{num(row.get('proxy_search_score')):.1f} | {num(proxy.get('route_eligible_count')):.0f} | "
            f"{num(proxy.get('route_eligible_active_count')):.0f} | {num(proxy.get('route_eligible_proxy_positive_r')):.1f} | "
            f"{num(proxy.get('active_proxy_positive_r')):.1f} | {num(fold.get('fold_active_positive_r_min')):.1f} |"
        )
    audit = dict(summary.get("locked_holdout_audit") or {})
    if audit:
        metrics = dict(audit.get("metrics") or {})
        stability = dict(audit.get("stability") or {})
        proxy = dict(audit.get("proxy") or {})
        lines.extend(
            [
                "",
                "## Locked Holdout Audit",
                "",
                f"- Train-selected pool/route: `{(audit.get('pool_spec') or {}).get('name')}` + `{(audit.get('route') or {}).get('name')}`",
                f"- Holdout net/DD/trades: {pct(metrics.get('broker_net_return_pct'))} / {pct(metrics.get('broker_max_drawdown_pct'))} / {num(metrics.get('trade_count')):.0f}",
                f"- Holdout capture/five-fold worst/eligible proxy R: {pct(metrics.get('avg_mfe_capture'))} / {pct(stability.get('five_fold_worst_net'))} / {num(proxy.get('route_eligible_proxy_positive_r')):.1f}R",
                f"- Route mix: {route_mix(audit)}",
                "- Locked audit note: this was not used to choose the train champion.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    start = time.time()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = int(os.environ.get("KALCB_FAST_FOLDS", "5") or "5")
    replay_top = int(os.environ.get("KALCB_REPLAY_TOP", "4") or "4")
    locked_holdout = os.environ.get("KALCB_LOCKED_HOLDOUT", "1") != "0"
    log("start", folds=folds, replay_top=replay_top, locked_holdout=locked_holdout)
    train_config, holdout_config = shared.load_base_config()
    df = opt.read_pipeline()
    train_raw = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout_raw = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    feature_rows_train = shared.load_feature_rows("train")
    feature_rows_holdout = shared.load_feature_rows("holdout")
    train = base.merge_causal_features(train_raw, feature_rows_train)
    holdout = base.merge_causal_features(holdout_raw, feature_rows_holdout)
    stage = ranker_and_proxy_stage(train, holdout, feature_rows_train, folds=folds)
    replay = run_replays(stage, train_config, holdout_config, feature_rows_holdout, holdout, replay_top=replay_top, locked_holdout=locked_holdout)
    summary = {
        "created_at_utc": now_iso(),
        "usage_contract": "train_only_fast_extensive_interaction_route_search_holdout_locked_after_train_selection",
        "runtime_diagnosis": {
            "primary_issue": "full shared-core replay was being used too early and too often",
            "secondary_issue": "tree model parallelism caused CPU contention and orphan-prone worker processes",
            "fix": "broad proxy and OOF screen first, dedupe pools, replay only finalists, tree n_jobs=1",
        },
        "proxy_ranked": stage["proxy_ranked"],
        "pool_specs": stage["pool_specs"],
        "selected_replay_top": replay_top,
        **replay,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    write_json(OUT_DIR / "kalcb_fast_extensive_search_results.json", summary)
    (OUT_DIR / "kalcb_fast_extensive_search_report.md").write_text(render_report(summary), encoding="utf-8")
    log(
        "complete",
        elapsed_seconds=summary["elapsed_seconds"],
        summary=str(OUT_DIR / "kalcb_fast_extensive_search_results.json"),
        report=str(OUT_DIR / "kalcb_fast_extensive_search_report.md"),
    )


if __name__ == "__main__":
    main()
