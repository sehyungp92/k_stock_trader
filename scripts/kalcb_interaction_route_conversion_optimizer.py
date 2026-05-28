from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_OPT_PATH = REPO_ROOT / "scripts" / "kalcb_train_alpha_capture_optimizer.py"
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
OUT_DIR = ROUND_DIR / "r_capture_optimizer" / "interaction_route_conversion_optimizer"

from backtests.strategies.kalcb.candidate_surfacing_recovery import (  # noqa: E402
    CAUSAL_FEATURE_KEYS,
    LEAKAGE_FEATURE_BLOCKLIST,
    PoolVariant,
    _pool_route_meta,
    evaluate_compiled_candidate_pool,
)
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
class RankerSpec:
    name: str
    feature_set: str
    model_kind: str
    target: str
    scope: str
    budget: str
    active_count: int
    pool_size: int


@dataclass(frozen=True)
class RouteSpec:
    name: str
    description: str
    build: Callable[[dict[str, Any]], dict[str, Any]]


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


def read_trade_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    return read_jsonl(source)


def causal_feature_frame(feature_by_key: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    allowed = set(CAUSAL_FEATURE_KEYS) - set(LEAKAGE_FEATURE_BLOCKLIST)
    for (day, symbol), row in feature_by_key.items():
        out: dict[str, Any] = {"trade_date": str(day)[:10], "symbol": str(symbol).zfill(6)}
        for key in allowed:
            if key in row:
                out[key] = row.get(key)
        rows.append(out)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "symbol"])
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    return frame.drop_duplicates(["trade_date", "symbol"], keep="last")


def merge_causal_features(part: pd.DataFrame, feature_by_key: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    work = part.copy()
    work["symbol"] = work["symbol"].astype(str).str.zfill(6)
    features = causal_feature_frame(feature_by_key)
    if features.empty:
        return engineer_enhanced_features(work)
    merged = work.merge(features, on=["trade_date", "symbol"], how="left", suffixes=("", "__causal"))
    for key in CAUSAL_FEATURE_KEYS:
        causal = f"{key}__causal"
        if causal not in merged.columns:
            continue
        if key in merged.columns:
            merged[key] = merged[key].where(merged[key].notna(), merged[causal])
            merged = merged.drop(columns=[causal])
        else:
            merged = merged.rename(columns={causal: key})
    return engineer_enhanced_features(merged)


def _series(df: pd.DataFrame, key: str, default: float = 0.0) -> pd.Series:
    if key in df.columns:
        return pd.to_numeric(df[key], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _safe_ratio(a: pd.Series, b: pd.Series, scale: float = 1.0) -> pd.Series:
    den = b.replace(0.0, np.nan)
    return (a / den * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_candidates = set(CAUSAL_FEATURE_KEYS) | {col for group in opt.FEATURE_GROUPS.values() for col in group}
    numeric_candidates |= {
        "positive_r",
        "positive_net_eod_r",
        "negative_mae_abs",
        "net_eod_r_proxy",
        "surfaced_candidate",
        "entered_shared_core",
        "selected_first30",
    }
    for key in numeric_candidates:
        if key in out.columns:
            if key == "leading_sector_cluster":
                out[key] = out[key].astype(str).str.lower().isin({"true", "1", "yes"}).astype(float)
            else:
                out[key] = pd.to_numeric(out[key], errors="coerce")

    rvol_log = np.log1p(_series(out, "first30_rel_volume").clip(lower=0.0))
    out["ix_rvol_log"] = rvol_log
    out["ix_daily_strength"] = (
        0.45 * _series(out, "daily_close20_loc")
        + 0.25 * _series(out, "daily_close60_loc")
        + 0.20 * _series(out, "daily_momentum_pct") / 100.0
        + 0.10 * _series(out, "daily_volume_ratio_20d")
    )
    out["ix_sector_confirm"] = (
        (_series(out, "sector_daily_score_pct") / 100.0)
        * (_series(out, "sector_intraday_score_pct") / 100.0)
        * (1.0 + _series(out, "sector_intraday_breadth"))
    )
    out["ix_flow_confirm"] = (
        _series(out, "flow_combined_5d")
        + 0.5 * _series(out, "flow_acceleration")
        + 0.25 * _series(out, "flow_agreement_5d")
        - 0.25 * _series(out, "flow_divergence_5d")
    )
    out["ix_intraday_quality"] = (
        0.35 * _series(out, "first30_ret")
        + 0.25 * _series(out, "first30_vwap_ret")
        + 0.20 * _series(out, "first30_signal_bar_cpr", _series(out, "first30_close_location").fillna(0.0))
        + 0.20 * rvol_log
    )
    out["ix_gap_quality"] = (
        _series(out, "first30_gap_retention_ratio")
        * _series(out, "first30_signal_bar_cpr", _series(out, "first30_close_location").fillna(0.0))
        * rvol_log
    )
    out["ix_range_efficiency"] = _safe_ratio(_series(out, "first30_ret"), _series(out, "first30_range_atr").abs() + 0.25)
    out["ix_vwap_cpr_rvol"] = _series(out, "first30_vwap_ret") * _series(out, "first30_signal_bar_cpr", _series(out, "first30_close_location").fillna(0.0)) * rvol_log
    out["ix_sector_intraday_rvol"] = (_series(out, "sector_intraday_score_pct") / 100.0) * rvol_log
    out["ix_sector_leadership_stack"] = (
        _series(out, "first30_sector_leadership_pct") / 100.0
        + _series(out, "first30_sector_relvol_ratio")
        + _series(out, "sector_intraday_rel_volume")
    )
    out["ix_trend_flow_sector"] = out["ix_daily_strength"] * out["ix_flow_confirm"] * (1.0 + out["ix_sector_confirm"])
    out["ix_market_adjusted_momentum"] = _series(out, "first30_ret") - 0.35 * _series(out, "sector_intraday_ret") + 0.15 * _series(out, "stock_sector_daily_ret5_spread")
    out["ix_liquidity_momentum"] = np.log1p(_series(out, "daily_adv20_krw").clip(lower=0.0)) * _series(out, "first30_ret")
    out["ix_close_position_pressure"] = _series(out, "first30_signal_bar_cpr", _series(out, "first30_close_location").fillna(0.0)) - _series(out, "first30_open_drawdown").abs()
    out["ix_gap_reversal_risk"] = _series(out, "first30_gap").clip(lower=0.0) * (1.0 - _series(out, "first30_gap_retention_ratio").clip(lower=0.0, upper=1.5))
    out["ix_breakout_extension_pressure"] = _series(out, "first30_ret").clip(lower=0.0) * _safe_ratio(_series(out, "first30_range_atr"), _series(out, "daily_atr_pct").abs() + 0.01)
    out["ix_continuation_context"] = (
        _series(out, "continuation_joint_quality_pct") / 100.0
        + out["ix_sector_confirm"]
        + 0.5 * out["ix_gap_quality"]
        + 0.25 * out["ix_trend_flow_sector"]
    )
    out["ix_routeability_proxy"] = (
        (_series(out, "first30_ret") >= 0.0).astype(float)
        + (_series(out, "first30_vwap_ret") >= -0.003).astype(float)
        + (_series(out, "first30_signal_bar_cpr", _series(out, "first30_close_location").fillna(0.0)) >= 0.55).astype(float)
        + (_series(out, "first30_rel_volume") >= 1.25).astype(float)
        + (_series(out, "first30_range_atr") <= 2.25).astype(float)
        + (_series(out, "sector_intraday_score_pct") >= 50.0).astype(float)
    )

    rank_cols = [
        "ix_intraday_quality",
        "ix_continuation_context",
        "ix_gap_quality",
        "ix_routeability_proxy",
        "first30_rel_volume",
        "sector_intraday_score_pct",
    ]
    for key in rank_cols:
        if key in out.columns:
            out[f"{key}_day_pct"] = out.groupby("trade_date")[key].rank(pct=True, method="average")
    return out


DERIVED_FEATURES = [
    "ix_rvol_log",
    "ix_daily_strength",
    "ix_sector_confirm",
    "ix_flow_confirm",
    "ix_intraday_quality",
    "ix_gap_quality",
    "ix_range_efficiency",
    "ix_vwap_cpr_rvol",
    "ix_sector_intraday_rvol",
    "ix_sector_leadership_stack",
    "ix_trend_flow_sector",
    "ix_market_adjusted_momentum",
    "ix_liquidity_momentum",
    "ix_close_position_pressure",
    "ix_gap_reversal_risk",
    "ix_breakout_extension_pressure",
    "ix_continuation_context",
    "ix_routeability_proxy",
    "ix_intraday_quality_day_pct",
    "ix_continuation_context_day_pct",
    "ix_gap_quality_day_pct",
    "ix_routeability_proxy_day_pct",
    "first30_rel_volume_day_pct",
    "sector_intraday_score_pct_day_pct",
]


def feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    base = list(dict.fromkeys(opt.FEATURE_GROUPS["all_context"]))
    causal = [key for key in CAUSAL_FEATURE_KEYS if key in df.columns and key not in LEAKAGE_FEATURE_BLOCKLIST]
    if feature_set == "base_all_context":
        wanted = base
    elif feature_set == "causal_full":
        wanted = base + causal
    elif feature_set == "interaction_full":
        wanted = base + causal + DERIVED_FEATURES
    elif feature_set == "interaction_route":
        wanted = base + causal + DERIVED_FEATURES + [
            "surfaced_candidate",
            "entered_shared_core",
            "selected_first30",
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    out: list[str] = []
    seen: set[str] = set()
    for key in wanted:
        if key in df.columns and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def target_values(df: pd.DataFrame, target: str) -> np.ndarray:
    pos_r = _series(df, "positive_r").clip(lower=0.0)
    pos_net = _series(df, "positive_net_eod_r").clip(lower=0.0)
    mae = _series(df, "negative_mae_abs").clip(lower=0.0)
    if target == "alpha_quality":
        y = np.log1p(pos_r) + 0.45 * np.log1p(pos_net) - 0.35 * np.log1p(mae)
    elif target == "mfe_capture":
        y = np.log1p(pos_r) - 0.25 * np.log1p(mae)
    elif target == "routeable_alpha":
        routeable = 0.30 * _series(df, "entered_shared_core").fillna(0.0) + 0.10 * _series(df, "surfaced_candidate").fillna(0.0)
        y = np.log1p(pos_r) + 0.35 * np.log1p(pos_net) - 0.30 * np.log1p(mae) + routeable
    else:
        raise ValueError(f"Unknown target: {target}")
    return np.asarray(y, dtype=float)


def make_model(kind: str):
    if kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=260, learning_rate=0.04, max_leaf_nodes=28, l2_regularization=0.06, random_state=31),
        )
    if kind == "trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=180, min_samples_leaf=6, max_features=0.65, random_state=31, n_jobs=1),
        )
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=12.0))
    raise ValueError(f"Unknown model kind: {kind}")


def ranker_specs() -> list[RankerSpec]:
    specs: list[RankerSpec] = [
        RankerSpec("base_hgb_quality_top16", "base_all_context", "hgb", "alpha_quality", "dataset", "top16", 16, 16),
        RankerSpec("causal_hgb_quality_top16", "causal_full", "hgb", "alpha_quality", "dataset", "top16", 16, 16),
        RankerSpec("interaction_hgb_quality_top16", "interaction_full", "hgb", "alpha_quality", "dataset", "top16", 16, 16),
        RankerSpec("interaction_hgb_routeable_top16", "interaction_route", "hgb", "routeable_alpha", "dataset", "top16", 16, 16),
        RankerSpec("interaction_trees_quality_top16", "interaction_full", "trees", "alpha_quality", "dataset", "top16", 16, 16),
        RankerSpec("interaction_ridge_quality_top16", "interaction_full", "ridge", "alpha_quality", "dataset", "top16", 16, 16),
        RankerSpec("interaction_hgb_quality_top24a16", "interaction_full", "hgb", "alpha_quality", "dataset", "top24", 16, 24),
        RankerSpec("interaction_hgb_routeable_top24a16", "interaction_route", "hgb", "routeable_alpha", "dataset", "top24", 16, 24),
        RankerSpec("interaction_trees_quality_top24a16", "interaction_full", "trees", "alpha_quality", "dataset", "top24", 16, 24),
    ]
    return specs


def fit_rankers(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, dict[str, Any]]:
    fitted: dict[str, dict[str, Any]] = {}
    for spec in ranker_specs():
        features = feature_columns(train, spec.feature_set)
        mask = opt.scope_mask(train, spec.scope)
        model = make_model(spec.model_kind)
        log("ranker_fit_start", name=spec.name, rows=int(mask.sum()), features=len(features), target=spec.target)
        model.fit(train.loc[mask, features], target_values(train.loc[mask], spec.target))
        fitted[spec.name] = {
            "spec": spec,
            "features": features,
            "model": model,
            "train_score": np.asarray(model.predict(train[features]), dtype=float),
            "holdout_score": np.asarray(model.predict(holdout[features]), dtype=float),
        }
        log("ranker_fit_done", name=spec.name)
    return fitted


def pool_rows_for_score(
    part: pd.DataFrame,
    score: np.ndarray,
    spec: RankerSpec,
    feature_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = opt.select_by_budget(part, score, spec.scope, spec.budget, {})
    policy = train_opt.SelectionPolicy(
        name=spec.name,
        label=spec.name,
        scope=spec.scope,
        budget=spec.budget,
        active_count=spec.active_count,
        pool_size=spec.pool_size,
        source="interaction_route_train_only",
    )
    return train_opt.selected_pool_rows(part, score, policy, feature_by_key) if False else shared.selected_pool_rows(
        selected=selected,
        feature_by_key=feature_by_key,
        candidate=shared.ReplayCandidate(spec.name, spec.name, spec.scope, spec.budget, spec.active_count, spec.pool_size, "train_only", "interaction_route_train_only"),
    )


def set_first30_route(seed: dict[str, Any], *, risk: float | None = None, min_bar: float | None = None, min_vwap: float | None = None, min_votes: int | None = None, q_cpr: float | None = None, q_rvol: float | None = None, q_range_min: float | None = None) -> dict[str, Any]:
    out = copy.deepcopy(seed)
    routes = [dict(route) for route in out.get("kalcb.entry.routes") or [] if isinstance(route, dict)]
    for route in routes:
        if str(route.get("mode") or "") != "first30_open":
            continue
        if risk is not None:
            route["risk_mult"] = float(risk)
            route["notional_mult"] = float(risk)
        if min_bar is not None:
            route["min_bar_ret"] = float(min_bar)
            route["quality_min_bar_ret"] = min(float(min_bar), 0.0)
        if min_vwap is not None:
            route["min_vwap_ret"] = float(min_vwap)
        if min_votes is not None:
            route["min_quality_votes"] = int(min_votes)
        if q_cpr is not None:
            route["quality_min_first30_signal_cpr"] = float(q_cpr)
        if q_rvol is not None:
            route["quality_min_first30_rel_volume"] = float(q_rvol)
        if q_range_min is not None:
            route["quality_min_first30_range_atr"] = float(q_range_min)
    out["kalcb.entry.routes"] = routes
    return out


def delayed_route(
    mode: str,
    *,
    rank: int,
    risk: float,
    priority: int,
    name: str,
    max_signal_bars: int = 24,
    min_votes: int = 5,
    min_bar: float = -0.005,
    min_vwap: float = -0.01,
    q_cpr: float = 0.65,
    q_rvol: float = 1.25,
    pullback: float = 0.012,
    reclaim: float = 0.0,
    after_bar: int = 1,
) -> dict[str, Any]:
    route = {
        "name": name,
        "mode": mode,
        "priority": priority,
        "after_bar": after_bar,
        "max_signal_bars": max_signal_bars,
        "require_initial_active": False,
        "max_frontier_rank": rank,
        "max_session_trades": 1,
        "min_bar_ret": min_bar,
        "min_vwap_ret": min_vwap,
        "min_quality_votes": min_votes,
        "quality_min_bar_ret": min_bar,
        "quality_min_first30_signal_cpr": q_cpr,
        "quality_min_first30_rel_volume": q_rvol,
        "quality_min_accumulation_score": -0.10,
        "risk_mult": risk,
        "notional_mult": risk,
    }
    if mode in {"pullback_acceptance", "avwap_reclaim", "or_high_reclaim", "or_mid_reclaim"}:
        route["max_pullback_from_vwap_pct"] = pullback
        route["min_reclaim_ret"] = reclaim
    if mode == "deferred_continuation":
        route["after_bar"] = max(after_bar, 5)
        route["min_breakout_pct"] = 0.0
        route["min_close_location"] = 0.50
    return route


def with_delayed_bundle(
    seed: dict[str, Any],
    *,
    first30_risk: float,
    delayed_risk: float,
    rank: int,
    quality_votes: int,
    include_deferred: bool,
    soft_first30: bool,
) -> dict[str, Any]:
    out = set_first30_route(
        seed,
        risk=first30_risk,
        min_bar=0.0 if soft_first30 else None,
        min_vwap=-0.005 if soft_first30 else None,
        min_votes=5 if soft_first30 else None,
        q_cpr=0.65 if soft_first30 else None,
        q_rvol=1.25 if soft_first30 else None,
        q_range_min=0.50 if soft_first30 else None,
    )
    routes = [dict(route) for route in out.get("kalcb.entry.routes") or [] if isinstance(route, dict) and str(route.get("mode") or "") == "first30_open"]
    routes.extend(
        [
            delayed_route("pullback_acceptance", rank=rank, risk=delayed_risk, priority=5, name=f"pullback_rank{rank}_r{delayed_risk:g}", min_votes=quality_votes),
            delayed_route("avwap_reclaim", rank=rank, risk=delayed_risk, priority=6, name=f"avwap_rank{rank}_r{delayed_risk:g}", min_votes=quality_votes),
            delayed_route("or_high_reclaim", rank=rank, risk=delayed_risk, priority=7, name=f"orhigh_rank{rank}_r{delayed_risk:g}", min_votes=quality_votes, pullback=0.004),
        ]
    )
    if include_deferred:
        routes.append(
            delayed_route(
                "deferred_continuation",
                rank=rank,
                risk=delayed_risk,
                priority=8,
                name=f"deferred_rank{rank}_r{delayed_risk:g}",
                min_votes=max(4, quality_votes - 1),
                max_signal_bars=30,
                after_bar=5,
            )
        )
    out["kalcb.entry.routes"] = routes
    out["kalcb.entry.frontier_branch_universe"] = True
    out["kalcb.frontier.shadow_enabled"] = False
    return out


def route_specs() -> list[RouteSpec]:
    return [
        RouteSpec("seed_risk99", "Incumbent train champion route seed.", lambda seed: copy.deepcopy(seed)),
        RouteSpec("first30_soft_quality5", "Soften first30 bar/vwap gates and require 5 quality votes.", lambda seed: set_first30_route(seed, min_bar=0.0, min_vwap=-0.005, min_votes=5, q_cpr=0.65, q_rvol=1.25, q_range_min=0.50)),
        RouteSpec("first30_soft_quality4", "More permissive first30 conversion with 4 quality votes.", lambda seed: set_first30_route(seed, min_bar=-0.003, min_vwap=-0.008, min_votes=4, q_cpr=0.60, q_rvol=1.0, q_range_min=0.35)),
        RouteSpec("first30_soft_risk80", "Soft first30 route with 0.80 risk multiplier.", lambda seed: set_first30_route(seed, risk=0.80, min_bar=0.0, min_vwap=-0.005, min_votes=5, q_cpr=0.65, q_rvol=1.25, q_range_min=0.50)),
        RouteSpec("delayed_rank16_r10", "First30 0.80 plus pullback/AVWAP/OR-high rank16 at 0.10 risk.", lambda seed: with_delayed_bundle(seed, first30_risk=0.80, delayed_risk=0.10, rank=16, quality_votes=5, include_deferred=False, soft_first30=False)),
        RouteSpec("delayed_rank16_r15_soft", "Soft first30 0.65 plus pullback/AVWAP/OR-high/deferred rank16 at 0.15 risk.", lambda seed: with_delayed_bundle(seed, first30_risk=0.65, delayed_risk=0.15, rank=16, quality_votes=5, include_deferred=True, soft_first30=True)),
        RouteSpec("delayed_rank24_r10_soft", "Soft first30 0.65 plus delayed route family rank24 at 0.10 risk.", lambda seed: with_delayed_bundle(seed, first30_risk=0.65, delayed_risk=0.10, rank=24, quality_votes=5, include_deferred=True, soft_first30=True)),
        RouteSpec("delayed_rank24_r20_soft_q4", "Soft first30 0.50 plus delayed route family rank24 at 0.20 risk and 4 votes.", lambda seed: with_delayed_bundle(seed, first30_risk=0.50, delayed_risk=0.20, rank=24, quality_votes=4, include_deferred=True, soft_first30=True)),
    ]


def route_proxy_summary(pool_rows: list[dict[str, Any]], mutations: dict[str, Any]) -> dict[str, Any]:
    routes = _configured_entry_routes(mutations)
    active_rows = [row for row in pool_rows if bool(row.get("pool_active"))]
    eligible_rows: list[dict[str, Any]] = []
    eligible_active_rows: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    for row in pool_rows:
        meta = _pool_route_meta(row)
        passed_modes: list[str] = []
        first_reason = ""
        for route in routes:
            passed, reason = _route_candidate_passes(route, mutations, meta)
            mode = str(route.get("mode") or route.get("name") or "route")
            if passed:
                passed_modes.append(mode)
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            elif not first_reason:
                first_reason = reason
        if passed_modes:
            eligible_rows.append(row)
            if bool(row.get("pool_active")):
                eligible_active_rows.append(row)
        else:
            blockers[first_reason or "not_eligible"] = blockers.get(first_reason or "not_eligible", 0) + 1
    return {
        "pool_rows": len(pool_rows),
        "active_rows": len(active_rows),
        "active_days": len({str(row.get("trade_date") or "")[:10] for row in active_rows}),
        "active_proxy_positive_r": sum(num(row.get("optimizer_positive_r_proxy")) for row in active_rows),
        "active_proxy_net_eod_r": sum(num(row.get("optimizer_net_eod_r_proxy")) for row in active_rows),
        "route_eligible_count": len(eligible_rows),
        "route_eligible_active_count": len(eligible_active_rows),
        "route_eligible_proxy_positive_r": sum(num(row.get("optimizer_positive_r_proxy")) for row in eligible_rows),
        "route_eligible_active_proxy_positive_r": sum(num(row.get("optimizer_positive_r_proxy")) for row in eligible_active_rows),
        "route_eligible_proxy_net_eod_r": sum(num(row.get("optimizer_net_eod_r_proxy")) for row in eligible_rows),
        "route_eligible_share": len(eligible_rows) / max(len(pool_rows), 1),
        "route_eligible_by_mode": dict(sorted(mode_counts.items(), key=lambda item: item[1], reverse=True)),
        "top_route_blockers": sorted(blockers.items(), key=lambda item: item[1], reverse=True)[:8],
    }


def route_screen_score(proxy: dict[str, Any]) -> float:
    return (
        num(proxy.get("route_eligible_proxy_positive_r"))
        + 0.35 * max(num(proxy.get("route_eligible_proxy_net_eod_r")), 0.0)
        - 0.20 * max(-num(proxy.get("route_eligible_proxy_net_eod_r")), 0.0)
        + 0.35 * num(proxy.get("active_proxy_positive_r"))
        + 4.0 * num(proxy.get("route_eligible_count"))
        + 1.5 * num(proxy.get("active_days"))
    )


def fold_date_map(dates: list[date], folds: int = 5) -> dict[date, int]:
    ordered = sorted(dates)
    out: dict[date, int] = {}
    for index, day in enumerate(ordered):
        out[day] = min(folds - 1, int(index * folds / max(len(ordered), 1)))
    return out


def stability_metrics(trade_rows: list[dict[str, Any]], session_dates: list[date], initial_equity: float) -> dict[str, Any]:
    date_to_fold = fold_date_map(session_dates, 5)
    fold_net = [0.0 for _ in range(5)]
    fold_trades = [0 for _ in range(5)]
    sector_net: dict[str, float] = {}
    sector_trades: dict[str, int] = {}
    by_route: dict[str, int] = {}
    by_exit: dict[str, int] = {}
    total_r = 0.0
    total_mfe = 0.0
    total_giveback = 0.0
    for row in trade_rows:
        pnl = num(row.get("net_pnl"))
        day = parse_date(row.get("entry_date"))
        if day is not None and day in date_to_fold:
            fold = date_to_fold[day]
            fold_net[fold] += pnl / max(initial_equity, 1.0)
            fold_trades[fold] += 1
        sector = str(row.get("sector") or "UNKNOWN")
        sector_net[sector] = sector_net.get(sector, 0.0) + pnl / max(initial_equity, 1.0)
        sector_trades[sector] = sector_trades.get(sector, 0) + 1
        route = str(row.get("entry_route_mode") or row.get("entry_route") or "unknown")
        by_route[route] = by_route.get(route, 0) + 1
        exit_reason = str(row.get("exit_reason") or "unknown")
        by_exit[exit_reason] = by_exit.get(exit_reason, 0) + 1
        r = num(row.get("r"))
        mfe_r = num(row.get("mfe_r"))
        total_r += r
        total_mfe += mfe_r
        total_giveback += max(mfe_r - r, 0.0)
    trades = len(trade_rows)
    worst_sector = min(sector_net.values(), default=0.0)
    return {
        "five_fold_net": fold_net,
        "five_fold_trades": fold_trades,
        "five_fold_worst_net": min(fold_net) if fold_net else 0.0,
        "five_fold_negative_count": sum(1 for value in fold_net if value < 0.0),
        "sector_net": dict(sorted(sector_net.items(), key=lambda item: item[1])),
        "sector_trades": dict(sorted(sector_trades.items(), key=lambda item: item[1], reverse=True)),
        "worst_sector_net": worst_sector,
        "negative_sector_count": sum(1 for value in sector_net.values() if value < 0.0),
        "largest_sector_trade_share": max(sector_trades.values(), default=0) / max(trades, 1),
        "entry_route_mode_counts": dict(sorted(by_route.items(), key=lambda item: item[1], reverse=True)),
        "exit_reason_counts": dict(sorted(by_exit.items(), key=lambda item: item[1], reverse=True)),
        "avg_trade_r": total_r / max(trades, 1),
        "avg_mfe_r_from_trades": total_mfe / max(trades, 1),
        "avg_giveback_r": total_giveback / max(trades, 1),
    }


def alpha_conversion_score(metrics: dict[str, Any], stability: dict[str, Any], proxy: dict[str, Any]) -> dict[str, Any]:
    net = num(metrics.get("broker_net_return_pct"))
    dd = num(metrics.get("broker_max_drawdown_pct"))
    trades = num(metrics.get("trade_count"))
    capture = num(metrics.get("avg_mfe_capture"))
    worst5 = num(stability.get("five_fold_worst_net"))
    neg_folds = num(stability.get("five_fold_negative_count"))
    worst_sector = num(stability.get("worst_sector_net"))
    neg_sectors = num(stability.get("negative_sector_count"))
    largest_sector_share = num(stability.get("largest_sector_trade_share"))
    route_proxy_r = num(proxy.get("route_eligible_proxy_positive_r"))
    dd_excess = max(dd - 0.08, 0.0)
    freq_bonus = 24.0 * min(trades / 170.0, 1.6)
    score = (
        100.0 * net
        + freq_bonus
        + 12.0 * max(capture, 0.0)
        + 22.0 * max(worst5, 0.0)
        + 0.00030 * route_proxy_r
        - 190.0 * max(-worst5, 0.0)
        - 275.0 * dd_excess
        - 8.0 * neg_folds
        - 32.0 * max(-worst_sector, 0.0)
        - 1.25 * neg_sectors
        - 8.0 * max(largest_sector_share - 0.35, 0.0)
    )
    pass_hygiene = (
        net > 0.0
        and dd <= 0.08
        and trades >= 90.0
        and worst5 >= 0.0
        and neg_folds == 0.0
        and capture >= 0.25
        and num(metrics.get("same_bar_fill_count")) == 0.0
        and num(metrics.get("end_open_position_count")) == 0.0
    )
    return {
        "train_alpha_conversion_score": score,
        "train_alpha_conversion_pass": pass_hygiene,
        "frequency_bonus": freq_bonus,
        "dd_excess": dd_excess,
    }


def route_mix(row: dict[str, Any]) -> str:
    summary = dict(row.get("entry_route_mode_summary") or {})
    if summary:
        return "; ".join(f"{mode}:{int(num((data or {}).get('trades')))}" for mode, data in summary.items())
    counts = dict((row.get("stability") or {}).get("entry_route_mode_counts") or {})
    return "; ".join(f"{mode}:{count}" for mode, count in counts.items())


def run_optimizer(max_replays: int = 30) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = train_opt.load_seed_mutations()
    train_config, holdout_config = shared.load_base_config()
    df = opt.read_pipeline()
    train_raw = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout_raw = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    feature_rows_train = shared.load_feature_rows("train")
    feature_rows_holdout = shared.load_feature_rows("holdout")
    train = merge_causal_features(train_raw, feature_rows_train)
    holdout = merge_causal_features(holdout_raw, feature_rows_holdout)
    write_json(OUT_DIR / "feature_inventory.json", {"train_columns": list(train.columns), "derived_features": DERIVED_FEATURES})

    fitted = fit_rankers(train, holdout)
    pool_map: dict[str, list[dict[str, Any]]] = {}
    pool_summaries: list[dict[str, Any]] = []
    for name, info in fitted.items():
        spec = info["spec"]
        pool_rows = pool_rows_for_score(train, info["train_score"], spec, feature_rows_train)
        pool_map[name] = pool_rows
        write_jsonl(OUT_DIR / f"pool_rows_train_{name}.jsonl", pool_rows)
        active = [row for row in pool_rows if bool(row.get("pool_active"))]
        pool_summaries.append(
            {
                "ranker": spec.__dict__,
                "pool_rows": len(pool_rows),
                "active_rows": len(active),
                "active_proxy_positive_r": sum(num(row.get("optimizer_positive_r_proxy")) for row in active),
                "active_proxy_net_eod_r": sum(num(row.get("optimizer_net_eod_r_proxy")) for row in active),
                "active_days": len({str(row.get("trade_date") or "")[:10] for row in active}),
            }
        )
    log("pool_build_done", pools=len(pool_map))

    route_screen_rows: list[dict[str, Any]] = []
    routes = route_specs()
    for pool_name, pool_rows in pool_map.items():
        for route in routes:
            mutations = route.build(seed)
            proxy = route_proxy_summary(pool_rows, mutations)
            row = {
                "ranker_name": pool_name,
                "ranker": fitted[pool_name]["spec"].__dict__,
                "route": {"name": route.name, "description": route.description},
                "proxy": proxy,
                "route_screen_score": route_screen_score(proxy),
            }
            route_screen_rows.append(row)
    route_screen_ranked = sorted(route_screen_rows, key=lambda row: num(row.get("route_screen_score")), reverse=True)

    forced_pairs = {
        ("base_hgb_quality_top16", "seed_risk99"),
        ("interaction_hgb_quality_top16", "seed_risk99"),
        ("interaction_hgb_routeable_top16", "first30_soft_quality5"),
        ("interaction_hgb_quality_top24a16", "delayed_rank24_r10_soft"),
    }
    replay_keys: set[tuple[str, str]] = set(forced_pairs)
    for row in route_screen_ranked:
        replay_keys.add((str(row["ranker_name"]), str(row["route"]["name"])))
        if len(replay_keys) >= max_replays:
            break
    write_json(OUT_DIR / "route_proxy_screen.json", route_screen_ranked)
    log("route_screen_done", screened=len(route_screen_ranked), selected_replays=len(replay_keys), top_score=route_screen_ranked[0]["route_screen_score"] if route_screen_ranked else 0.0)

    log("train_context_build_start")
    train_bundle = shared.build_window_bundle(train_config)
    train_dates = list(train_bundle["dataset"].trading_dates)
    initial_equity = float((train_bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    log("train_context_build_done", sessions=len(train_dates), contexts=len(train_bundle["context_by_key"]))

    train_rows: list[dict[str, Any]] = []
    route_by_name = {route.name: route for route in routes}
    for ranker_name, route_name in sorted(replay_keys):
        pool_rows = pool_map.get(ranker_name) or []
        route = route_by_name[route_name]
        mutations = route.build(seed)
        proxy = route_proxy_summary(pool_rows, mutations)
        if num(proxy.get("route_eligible_count")) < 45.0:
            row = {
                "ranker_name": ranker_name,
                "ranker": fitted[ranker_name]["spec"].__dict__,
                "route": {"name": route.name, "description": route.description},
                "proxy": proxy,
                "metrics": {"trade_count": 0.0},
                "stability": {},
                "skipped": True,
                "skip_reason": "route_eligible_count_lt_45",
                "train_alpha_conversion_score": -999.0,
                "train_alpha_conversion_pass": False,
            }
            train_rows.append(row)
            log("train_replay_skipped", ranker=ranker_name, route=route.name, eligible=proxy.get("route_eligible_count"))
            continue
        replay_name = f"{ranker_name}_{route.name}"
        log("train_replay_start", ranker=ranker_name, route=route.name, eligible=proxy.get("route_eligible_count"), proxy_r=proxy.get("route_eligible_proxy_positive_r"))
        result = evaluate_compiled_candidate_pool(
            window="train",
            variant=PoolVariant(replay_name, fitted[ranker_name]["spec"].pool_size, active_count=fitted[ranker_name]["spec"].active_count),
            config=train_config,
            dataset=train_bundle["dataset"],
            context_by_key=train_bundle["context_by_key"],
            pool_rows=pool_rows,
            seed_mutations=mutations,
            output_dir=OUT_DIR,
            replay_name=replay_name,
        )
        metrics = dict(result.get("metrics") or {})
        trades = read_trade_rows(result.get("trade_rows_path"))
        stability = stability_metrics(trades, train_dates, initial_equity)
        score = alpha_conversion_score(metrics, stability, proxy)
        row = {
            "ranker_name": ranker_name,
            "ranker": fitted[ranker_name]["spec"].__dict__,
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
        write_jsonl(OUT_DIR / f"pool_rows_locked_holdout_{ranker_name}_{route_name}.jsonl", holdout_pool)
        mutations = route.build(seed)
        proxy = route_proxy_summary(holdout_pool, mutations)
        log("locked_holdout_replay_start", ranker=ranker_name, route=route_name, rows=len(holdout_pool), eligible=proxy.get("route_eligible_count"))
        result = evaluate_compiled_candidate_pool(
            window="holdout_locked_audit",
            variant=PoolVariant(f"{ranker_name}_{route_name}_locked_holdout", spec.pool_size, active_count=spec.active_count),
            config=holdout_config,
            dataset=holdout_bundle["dataset"],
            context_by_key=holdout_bundle["context_by_key"],
            pool_rows=holdout_pool,
            seed_mutations=mutations,
            output_dir=OUT_DIR,
            replay_name=f"{ranker_name}_{route_name}_locked_holdout",
        )
        metrics = dict(result.get("metrics") or {})
        trades = read_trade_rows(result.get("trade_rows_path"))
        locked_holdout = {
            "selection_basis": "locked_train_selected_ranker_route_only_no_holdout_optimization",
            "ranker": spec.__dict__,
            "route": {"name": route.name, "description": route.description},
            "proxy": proxy,
            "metrics": metrics,
            "stability": stability_metrics(trades, holdout_dates, holdout_equity),
            "compiled_replay": result.get("compiled_replay") or {},
            "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
            "trade_rows_path": result.get("trade_rows_path"),
        }
        log(
            "locked_holdout_replay_done",
            ranker=ranker_name,
            route=route_name,
            trades=metrics.get("trade_count"),
            net=metrics.get("broker_net_return_pct"),
            dd=metrics.get("broker_max_drawdown_pct"),
        )

    return {
        "created_at_utc": now_iso(),
        "usage_contract": "train_only_interaction_ranker_and_route_conversion_optimizer_holdout_locked_after_train_selection",
        "objective": {
            "selection_data": "train only",
            "score": "100*train_net + frequency + capture + five_fold_worst + route_proxy_R - DD/subfold/sector penalties",
            "holdout_policy": "locked audit after train ranker/route champion is selected",
        },
        "ranker_pool_summaries": sorted(pool_summaries, key=lambda row: num(row.get("active_proxy_positive_r")), reverse=True),
        "route_proxy_screen": route_screen_ranked,
        "selected_replay_keys": sorted([{"ranker": a, "route": b} for a, b in replay_keys], key=lambda x: (x["ranker"], x["route"])),
        "train_rows": train_rows,
        "train_champion": champion,
        "locked_holdout_audit": locked_holdout,
    }


def render_report(summary: dict[str, Any]) -> str:
    rows = [row for row in summary.get("train_rows", []) if not row.get("skipped")]
    ranked = sorted(rows, key=lambda row: num(row.get("train_alpha_conversion_score")), reverse=True)
    champion = dict(summary.get("train_champion") or {})
    lines = [
        "# KALCB Interaction + Route Conversion Optimizer",
        "",
        "Train-only interaction ranker and route-expansion sweep. Holdout is locked until after train champion selection.",
        "",
        "## Train Champion",
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
                f"- Train alpha-conversion score: {num(champion.get('train_alpha_conversion_score')):.2f}; pass: {champion.get('train_alpha_conversion_pass')}",
                f"- Train net/DD/trades: {pct(metrics.get('broker_net_return_pct'))} / {pct(metrics.get('broker_max_drawdown_pct'))} / {num(metrics.get('trade_count')):.0f}",
                f"- Train capture/five-fold worst/eligible proxy R: {pct(metrics.get('avg_mfe_capture'))} / {pct(stability.get('five_fold_worst_net'))} / {num(proxy.get('route_eligible_proxy_positive_r')):.1f}R",
                f"- Sector worst/largest sector share: {pct(stability.get('worst_sector_net'))} / {pct(stability.get('largest_sector_trade_share'))}",
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
    for rank, row in enumerate(ranked[:22], start=1):
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
    lines.extend(["", "## Route Proxy Screen", ""])
    lines.append("| rank | ranker | route | score | eligible | active eligible | eligible R | active R | blockers |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
    for rank, row in enumerate(list(summary.get("route_proxy_screen") or [])[:18], start=1):
        proxy = dict(row.get("proxy") or {})
        blockers = "; ".join(f"{key}:{value}" for key, value in (proxy.get("top_route_blockers") or [])[:3])
        lines.append(
            f"| {rank} | `{row.get('ranker_name')}` | `{(row.get('route') or {}).get('name')}` | "
            f"{num(row.get('route_screen_score')):.1f} | {num(proxy.get('route_eligible_count')):.0f} | "
            f"{num(proxy.get('route_eligible_active_count')):.0f} | {num(proxy.get('route_eligible_proxy_positive_r')):.1f} | "
            f"{num(proxy.get('active_proxy_positive_r')):.1f} | {blockers} |"
        )
    lines.extend(["", "## Ranker Pool Proxy", ""])
    lines.append("| rank | ranker | rows | active | days | active R | active EOD R |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(list(summary.get("ranker_pool_summaries") or [])[:12], start=1):
        ranker = dict(row.get("ranker") or {})
        lines.append(
            f"| {rank} | `{ranker.get('name')}` | {num(row.get('pool_rows')):.0f} | "
            f"{num(row.get('active_rows')):.0f} | {num(row.get('active_days')):.0f} | "
            f"{num(row.get('active_proxy_positive_r')):.1f} | {num(row.get('active_proxy_net_eod_r')):.1f} |"
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This run tests daily/sector/RVOL/range/gap context through interaction rankers and derived scores, not one-dimensional hard gates.",
            "- Route expansion is evaluated through the shared KALCB core and SimBroker; proxy screens only decide what gets train replayed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("start", out_dir=str(OUT_DIR))
    max_replays = int(os.environ.get("KALCB_MAX_REPLAYS", "18") or "18")
    summary = run_optimizer(max_replays=max_replays)
    summary["max_replays"] = max_replays
    summary["elapsed_seconds"] = round(time.time() - start, 3)
    write_json(OUT_DIR / "kalcb_interaction_route_conversion_results.json", summary)
    report = render_report(summary)
    (OUT_DIR / "kalcb_interaction_route_conversion_report.md").write_text(report, encoding="utf-8")
    log(
        "complete",
        elapsed_seconds=summary["elapsed_seconds"],
        summary=str(OUT_DIR / "kalcb_interaction_route_conversion_results.json"),
        report=str(OUT_DIR / "kalcb_interaction_route_conversion_report.md"),
    )


if __name__ == "__main__":
    main()
