from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND_DIR = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
PIPELINE_DIR = ROUND_DIR / "positive_r_pipeline_breakdown"
ORACLE_DIR = ROUND_DIR / "local_minimum_recovery" / "07_alpha_conversion_next_round"
OUT_DIR = ROUND_DIR / "r_capture_optimizer"

PIPELINE_ROWS = PIPELINE_DIR / "kalcb_positive_r_pipeline_rows.csv"
PIPELINE_JSON = PIPELINE_DIR / "kalcb_positive_r_pipeline_breakdown.json"
ORACLE_TRAIN = ORACLE_DIR / "full_universe_missed_opportunities_train.jsonl"
ORACLE_HOLDOUT = ORACLE_DIR / "full_universe_missed_opportunities_holdout.jsonl"


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


def safe_div(a: Any, b: Any) -> float:
    den = num(b)
    return num(a) / den if den else 0.0


def bool_to_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    lowered = series.astype(str).str.lower()
    if lowered.isin(["true", "false"]).all():
        return lowered.eq("true").astype(float)
    return pd.to_numeric(series, errors="coerce")


def read_pipeline() -> pd.DataFrame:
    log("pipeline_read_start", path=str(PIPELINE_ROWS))
    df = pd.read_csv(PIPELINE_ROWS, dtype={"symbol": str})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    for col in ("dataset_available", "surfaced_candidate", "selected_first30", "entered_shared_core", "daily_above_sma20", "daily_above_sma60", "flow_available"):
        if col in df.columns:
            df[col] = bool_to_float(df[col]).fillna(0.0)
    numeric_cols = [col for col in df.columns if col not in {"window", "trade_date", "symbol", "sector", "frontier_loss_reason", "first30_loss_reason", "entry_loss_reason"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["positive_r"] = df["positive_mfe_r_proxy"].clip(lower=0.0)
    df["positive_net_eod_r"] = df["net_eod_r_proxy"].clip(lower=0.0)
    df["negative_mae_abs"] = (-df["mae_r_proxy"].clip(upper=0.0)).fillna(0.0)
    df = engineer_pipeline_features(df)
    log("pipeline_read_done", rows=len(df), cols=len(df.columns), train_rows=int((df["window"] == "train").sum()), holdout_rows=int((df["window"] == "holdout").sum()))
    return df


def engineer_pipeline_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["daily_adv20_krw_log"] = np.log1p(out.get("daily_adv20_krw", 0.0).clip(lower=0.0))
    out["first30_rel_volume_log"] = np.log1p(out.get("first30_rel_volume", 0.0).clip(lower=0.0))
    gap = out.get("first30_gap", pd.Series(0.0, index=out.index)).replace(0.0, np.nan)
    out["first30_gap_retention_ratio"] = (out.get("first30_low_vs_prev_close", 0.0) / gap.abs()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["first30_ret_x_rvol"] = out.get("first30_ret", 0.0) * out["first30_rel_volume_log"]
    out["first30_cpr_x_rvol"] = out.get("first30_close_location", 0.0) * out["first30_rel_volume_log"]
    out["first30_vwap_x_cpr"] = out.get("first30_vwap_ret", 0.0) * out.get("first30_close_location", 0.0)
    out["sector_daily_intraday_blend"] = 0.5 * out.get("sector_daily_score_pct", 0.0) + 0.5 * out.get("sector_intraday_score_pct", 0.0)
    out["sector_intraday_x_first30_ret"] = out.get("sector_intraday_score_pct", 0.0) * out.get("first30_ret", 0.0)
    out["sector_intraday_x_rvol"] = out.get("sector_intraday_score_pct", 0.0) * out["first30_rel_volume_log"]
    out["trend_flow_alignment"] = out.get("daily_return_20d", 0.0) * out.get("flow_combined_5d", 0.0)
    out["flow_agreement_minus_divergence"] = out.get("flow_agreement_5d", 0.0) - out.get("flow_divergence_5d", 0.0)
    out["daily_trend_stack"] = (
        out.get("daily_return_5d", 0.0)
        + 0.5 * out.get("daily_return_20d", 0.0)
        + 0.25 * out.get("daily_return_60d", 0.0)
        + 0.02 * out.get("daily_volume_ratio_20d", 0.0)
    )
    out["market_adjusted_first30_ret"] = out.get("first30_ret", 0.0) - 0.5 * out.get("sector_intraday_ret", 0.0)
    return out


FIRST30_FEATURES = [
    "first30_ret",
    "first30_vwap_ret",
    "first30_gap",
    "first30_rel_volume",
    "first30_rel_volume_log",
    "first30_close_location",
    "first30_open_drawdown",
    "first30_low_vs_prev_close",
    "first30_range_atr",
    "first30_gap_retention_ratio",
    "first30_ret_x_rvol",
    "first30_cpr_x_rvol",
    "first30_vwap_x_cpr",
]

DAILY_CONTEXT_FEATURES = [
    "daily_return_5d",
    "daily_return_20d",
    "daily_return_60d",
    "daily_volume_ratio_20d",
    "daily_close20_loc",
    "daily_close60_loc",
    "daily_atr_pct",
    "daily_adv20_krw_log",
    "daily_above_sma20",
    "daily_above_sma60",
    "daily_trend_stack",
]

FLOW_CONTEXT_FEATURES = [
    "flow_available",
    "flow_combined_1d",
    "flow_combined_3d",
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
    "flow_agreement_minus_divergence",
    "sector_flow_5d",
    "trend_flow_alignment",
]

SECTOR_MARKET_FEATURES = [
    "sector_participation",
    "market_score",
    "market_kospi_ret_5d",
    "market_kosdaq_ret_5d",
    "sector_daily_score_pct",
    "sector_daily_participation",
    "sector_daily_ret_5d",
    "sector_daily_ret_20d",
    "sector_intraday_score_pct",
    "sector_intraday_ret",
    "sector_intraday_breadth",
    "sector_intraday_participation",
    "sector_daily_intraday_blend",
    "sector_intraday_x_first30_ret",
    "sector_intraday_x_rvol",
    "market_adjusted_first30_ret",
]

FEATURE_GROUPS = {
    "first30_only": FIRST30_FEATURES,
    "first30_sector": FIRST30_FEATURES + SECTOR_MARKET_FEATURES,
    "context_only": DAILY_CONTEXT_FEATURES + FLOW_CONTEXT_FEATURES + SECTOR_MARKET_FEATURES,
    "all_context": FIRST30_FEATURES + DAILY_CONTEXT_FEATURES + FLOW_CONTEXT_FEATURES + SECTOR_MARKET_FEATURES,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_group: str
    model_kind: str
    target: str
    train_scope: str


def target_values(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "mfe_log":
        return np.log1p(df["positive_r"].clip(lower=0.0).to_numpy())
    if target == "eod_quality":
        y = (
            np.log1p(df["positive_r"].clip(lower=0.0).to_numpy())
            + 0.35 * np.log1p(df["positive_net_eod_r"].clip(lower=0.0).to_numpy())
            - 0.25 * np.log1p(df["negative_mae_abs"].clip(lower=0.0).to_numpy())
        )
        return y
    if target == "binary_big_r":
        thresh = float(df["positive_r"].quantile(0.85))
        return (df["positive_r"].to_numpy() >= thresh).astype(float)
    raise ValueError(f"Unknown target: {target}")


def make_model(kind: str):
    if kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=220, learning_rate=0.045, max_leaf_nodes=24, l2_regularization=0.05, random_state=7),
        )
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=12.0))
    if kind == "trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=220, max_depth=8, min_samples_leaf=12, random_state=11, n_jobs=-1),
        )
    raise ValueError(f"Unknown model kind: {kind}")


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "surfaced":
        return df["surfaced_candidate"].eq(1.0)
    if scope == "dataset":
        return df["dataset_available"].eq(1.0)
    if scope == "selected":
        return df["selected_first30"].eq(1.0)
    raise ValueError(f"Unknown scope: {scope}")


def model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for scope in ("surfaced", "dataset"):
        for group in ("first30_only", "first30_sector", "context_only", "all_context"):
            specs.append(ModelSpec(f"{scope}_{group}_hgb_mfe", group, "hgb", "mfe_log", scope))
        specs.append(ModelSpec(f"{scope}_all_context_hgb_quality", "all_context", "hgb", "eod_quality", scope))
        specs.append(ModelSpec(f"{scope}_all_context_trees_quality", "all_context", "trees", "eod_quality", scope))
        specs.append(ModelSpec(f"{scope}_all_context_ridge_quality", "all_context", "ridge", "eod_quality", scope))
    return specs


def fit_scores(train: pd.DataFrame, holdout: pd.DataFrame, specs: list[ModelSpec]) -> dict[str, dict[str, np.ndarray]]:
    scores: dict[str, dict[str, np.ndarray]] = {}
    for spec in specs:
        features = [col for col in FEATURE_GROUPS[spec.feature_group] if col in train.columns]
        train_mask = scope_mask(train, spec.train_scope)
        x_train = train.loc[train_mask, features]
        y_train = target_values(train.loc[train_mask], spec.target)
        model = make_model(spec.model_kind)
        log("model_fit_start", name=spec.name, rows=len(x_train), features=len(features), target=spec.target)
        model.fit(x_train, y_train)
        scores[spec.name] = {
            "train": np.asarray(model.predict(train[features]), dtype=float),
            "holdout": np.asarray(model.predict(holdout[features]), dtype=float),
        }
        log("model_fit_done", name=spec.name)
    return scores


def current_selected_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df[df["selected_first30"].eq(1.0)].groupby("trade_date").size().to_dict()
    return {str(day): int(count) for day, count in counts.items()}


def select_by_budget(df: pd.DataFrame, score: np.ndarray, scope: str, budget: str, selected_counts: dict[str, int]) -> pd.DataFrame:
    work = df.loc[scope_mask(df, scope)].copy()
    work["_score"] = np.asarray(score, dtype=float)[work.index.to_numpy()]
    selected: list[pd.DataFrame] = []
    fixed_k = None
    if budget.startswith("top"):
        fixed_k = int(budget.replace("top", ""))
    for day, group in work.groupby("trade_date", sort=True):
        if fixed_k is None:
            k = int(selected_counts.get(str(day), 0))
        else:
            k = fixed_k
        if k <= 0:
            continue
        selected.append(group.sort_values("_score", ascending=False).head(k))
    if not selected:
        return work.head(0)
    return pd.concat(selected, ignore_index=False)


def summarize_selection(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    window: str,
    scope: str,
    budget: str,
    label: str,
    current_selected_r: float,
    dataset_available_r: float,
    surfaced_available_r: float,
) -> dict[str, Any]:
    total_r = float(selected["positive_r"].sum())
    net_eod = float(selected["net_eod_r_proxy"].sum())
    positive_net_eod = float(selected["positive_net_eod_r"].sum())
    mae = float(selected["negative_mae_abs"].mean()) if len(selected) else 0.0
    hit_share = float((selected["positive_r"] > 0.0).mean()) if len(selected) else 0.0
    active_days = int(selected["trade_date"].nunique()) if len(selected) else 0
    return {
        "window": window,
        "label": label,
        "scope": scope,
        "budget": budget,
        "selected_rows": int(len(selected)),
        "active_days": active_days,
        "selected_positive_r": total_r,
        "selected_net_eod_r": net_eod,
        "selected_positive_net_eod_r": positive_net_eod,
        "avg_abs_mae_r": mae,
        "positive_r_hit_share": hit_share,
        "dataset_r_retention": safe_div(total_r, dataset_available_r),
        "surfaced_r_retention": safe_div(total_r, surfaced_available_r),
        "lift_vs_current_selected_r": safe_div(total_r, current_selected_r),
        "selected_symbols": int(selected[["trade_date", "symbol"]].drop_duplicates().shape[0]) if len(selected) else 0,
    }


def evaluate_pipeline_rankers(df: pd.DataFrame) -> dict[str, Any]:
    train = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    specs = model_specs()
    scores = fit_scores(train, holdout, specs)
    budgets = ["current_count", "top1", "top2", "top4", "top8"]
    rows: list[dict[str, Any]] = []
    selected_examples: dict[str, list[dict[str, Any]]] = {}

    baselines: dict[str, dict[str, Any]] = {}
    for window, part in (("train", train), ("holdout", holdout)):
        dataset_r = float(part["positive_r"].sum())
        surfaced_r = float(part.loc[part["surfaced_candidate"].eq(1.0), "positive_r"].sum())
        current_sel = part[part["selected_first30"].eq(1.0)].copy()
        current_r = float(current_sel["positive_r"].sum())
        baselines[window] = {
            "dataset_available_positive_r": dataset_r,
            "surfaced_positive_r": surfaced_r,
            "current_selected_positive_r": current_r,
            "current_selected_rows": int(len(current_sel)),
            "current_selected_days": int(current_sel["trade_date"].nunique()),
            "current_selected_dataset_retention": safe_div(current_r, dataset_r),
            "current_selected_surfaced_retention": safe_div(current_r, surfaced_r),
        }
        rows.append(
            summarize_selection(
                part,
                current_sel,
                window=window,
                scope="incumbent",
                budget="actual",
                label="current_selected_first30",
                current_selected_r=current_r,
                dataset_available_r=dataset_r,
                surfaced_available_r=surfaced_r,
            )
        )
        surfaced_all = part[part["surfaced_candidate"].eq(1.0)].copy()
        rows.append(
            summarize_selection(
                part,
                surfaced_all,
                window=window,
                scope="surfaced",
                budget="all",
                label="current_surfaced_all_ceiling",
                current_selected_r=current_r,
                dataset_available_r=dataset_r,
                surfaced_available_r=surfaced_r,
            )
        )
        for scope in ("surfaced", "dataset"):
            for budget in budgets:
                counts = current_selected_counts(part) if budget == "current_count" else {}
                scoped = part.loc[scope_mask(part, scope)].copy()
                oracle_selected = select_by_budget(part, scoped["positive_r"].to_numpy() if False else np.where(scope_mask(part, scope), part["positive_r"], -1e9), scope, budget, counts)
                rows.append(
                    summarize_selection(
                        part,
                        oracle_selected,
                        window=window,
                        scope=scope,
                        budget=budget,
                        label=f"oracle_positive_r_{scope}_{budget}",
                        current_selected_r=current_r,
                        dataset_available_r=dataset_r,
                        surfaced_available_r=surfaced_r,
                    )
                )

    for spec in specs:
        for window, part in (("train", train), ("holdout", holdout)):
            dataset_r = baselines[window]["dataset_available_positive_r"]
            surfaced_r = baselines[window]["surfaced_positive_r"]
            current_r = baselines[window]["current_selected_positive_r"]
            part_scores = scores[spec.name][window]
            for eval_scope in ("surfaced", "dataset"):
                for budget in budgets:
                    counts = current_selected_counts(part) if budget == "current_count" else {}
                    selected = select_by_budget(part, part_scores, eval_scope, budget, counts)
                    row = summarize_selection(
                        part,
                        selected,
                        window=window,
                        scope=eval_scope,
                        budget=budget,
                        label=spec.name,
                        current_selected_r=current_r,
                        dataset_available_r=dataset_r,
                        surfaced_available_r=surfaced_r,
                    )
                    row.update({"train_scope": spec.train_scope, "feature_group": spec.feature_group, "model": spec.model_kind, "target": spec.target})
                    rows.append(row)
                    if window == "holdout" and eval_scope == "dataset" and budget in {"top2", "top4"}:
                        key = f"{spec.name}_{eval_scope}_{budget}"
                        selected_examples[key] = (
                            selected.sort_values(["trade_date", "_score"], ascending=[True, False])
                            .head(60)
                            [["trade_date", "symbol", "sector", "positive_r", "net_eod_r_proxy", "mae_r_proxy", "surfaced_candidate", "selected_first30", "_score"]]
                            .to_dict("records")
                        )
        log("model_eval_done", name=spec.name)

    table = pd.DataFrame(rows)
    holdout_ranked = (
        table[table["window"].eq("holdout")]
        .sort_values(["selected_positive_r", "selected_net_eod_r", "positive_r_hit_share"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return {
        "baseline": baselines,
        "rows": table.to_dict("records"),
        "holdout_ranked": holdout_ranked.to_dict("records"),
        "selected_examples": selected_examples,
        "top_feature_correlations": top_feature_correlations(train),
    }


def top_feature_correlations(train: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [col for group in FEATURE_GROUPS.values() for col in group if col in train.columns]
    cols = sorted(set(cols))
    y = train["positive_r"]
    out = []
    for col in cols:
        corr = train[col].corr(y, method="spearman")
        if pd.notna(corr):
            out.append({"feature": col, "spearman_to_positive_r": float(corr), "abs_corr": abs(float(corr))})
    return sorted(out, key=lambda row: row["abs_corr"], reverse=True)[:30]


ORACLE_FEATURES = [
    "first30_ret",
    "first30_vwap_ret",
    "first30_rel_volume",
    "first30_signal_bar_cpr",
    "first30_range_atr",
    "first30_sector_ret_spread",
    "sector_daily_score_pct",
    "sector_daily_participation",
    "sector_daily_breadth_20d",
    "stock_sector_daily_ret5_spread",
    "stock_sector_daily_ret20_spread",
    "sector_intraday_score_pct",
    "sector_intraday_ret",
    "sector_intraday_breadth",
    "sector_intraday_participation",
    "leading_sector_cluster",
]


def read_oracle(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    for col in ORACLE_FEATURES + ["mfe_r", "net_r", "mae_r", "mfe_capture"]:
        if col in df.columns:
            if col == "leading_sector_cluster":
                df[col] = bool_to_float(df[col]).fillna(0.0)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    df["positive_mfe_r"] = df["mfe_r"].clip(lower=0.0)
    df["positive_net_r"] = df["net_r"].clip(lower=0.0)
    df["abs_mae_r"] = (-df["mae_r"].clip(upper=0.0)).fillna(0.0)
    df["first30_rel_volume_log"] = np.log1p(df["first30_rel_volume"].clip(lower=0.0))
    df["first30_ret_x_rvol"] = df["first30_ret"] * df["first30_rel_volume_log"]
    df["sector_intraday_x_first30_ret"] = df["sector_intraday_score_pct"] * df["first30_ret"]
    return df


def evaluate_oracle_ranker() -> dict[str, Any]:
    if not ORACLE_TRAIN.exists() or not ORACLE_HOLDOUT.exists():
        return {"status": "missing_oracle_files"}
    log("oracle_read_start")
    train = read_oracle(ORACLE_TRAIN).reset_index(drop=True)
    holdout = read_oracle(ORACLE_HOLDOUT).reset_index(drop=True)
    features = [col for col in ORACLE_FEATURES + ["first30_rel_volume_log", "first30_ret_x_rvol", "sector_intraday_x_first30_ret"] if col in train.columns]
    y = np.log1p(train["positive_net_r"].clip(lower=0.0)) + 0.35 * np.log1p(train["positive_mfe_r"].clip(lower=0.0)) - 0.20 * np.log1p(train["abs_mae_r"].clip(lower=0.0))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=260, learning_rate=0.04, max_leaf_nodes=24, l2_regularization=0.05, random_state=21),
    )
    log("oracle_model_fit_start", train_rows=len(train), holdout_rows=len(holdout), features=len(features))
    model.fit(train[features], y)
    train["_score"] = model.predict(train[features])
    holdout["_score"] = model.predict(holdout[features])
    rows = []
    for window, part in (("train", train), ("holdout", holdout)):
        available_mfe = float(part["positive_mfe_r"].sum())
        available_net = float(part["positive_net_r"].sum())
        for budget in (1, 2, 4, 8, 16):
            selected = part.sort_values(["trade_date", "_score"], ascending=[True, False]).groupby("trade_date", group_keys=False).head(budget)
            rows.append(
                {
                    "window": window,
                    "budget_topk_per_day": budget,
                    "selected_rows": int(len(selected)),
                    "active_days": int(selected["trade_date"].nunique()),
                    "selected_mfe_r": float(selected["positive_mfe_r"].sum()),
                    "selected_positive_net_r": float(selected["positive_net_r"].sum()),
                    "selected_total_net_r": float(selected["net_r"].sum()),
                    "selected_avg_capture": float(selected["mfe_capture"].mean()) if len(selected) else 0.0,
                    "positive_net_over_mfe": safe_div(selected["positive_net_r"].sum(), selected["positive_mfe_r"].sum()),
                    "total_net_over_mfe": safe_div(selected["net_r"].sum(), selected["positive_mfe_r"].sum()),
                    "selected_avg_abs_mae_r": float(selected["abs_mae_r"].mean()) if len(selected) else 0.0,
                    "mfe_retention": safe_div(selected["positive_mfe_r"].sum(), available_mfe),
                    "positive_net_retention": safe_div(selected["positive_net_r"].sum(), available_net),
                    "route_family_counts": {str(k): int(v) for k, v in selected["route_family"].value_counts().to_dict().items()},
                }
            )
        for budget in (1, 2, 4, 8, 16):
            selected = part.sort_values(["trade_date", "positive_net_r"], ascending=[True, False]).groupby("trade_date", group_keys=False).head(budget)
            rows.append(
                {
                    "window": window,
                    "budget_topk_per_day": budget,
                    "selected_rows": int(len(selected)),
                    "active_days": int(selected["trade_date"].nunique()),
                    "selected_mfe_r": float(selected["positive_mfe_r"].sum()),
                    "selected_positive_net_r": float(selected["positive_net_r"].sum()),
                    "selected_total_net_r": float(selected["net_r"].sum()),
                    "selected_avg_capture": float(selected["mfe_capture"].mean()) if len(selected) else 0.0,
                    "positive_net_over_mfe": safe_div(selected["positive_net_r"].sum(), selected["positive_mfe_r"].sum()),
                    "total_net_over_mfe": safe_div(selected["net_r"].sum(), selected["positive_mfe_r"].sum()),
                    "selected_avg_abs_mae_r": float(selected["abs_mae_r"].mean()) if len(selected) else 0.0,
                    "mfe_retention": safe_div(selected["positive_mfe_r"].sum(), available_mfe),
                    "positive_net_retention": safe_div(selected["positive_net_r"].sum(), available_net),
                    "route_family_counts": {str(k): int(v) for k, v in selected["route_family"].value_counts().to_dict().items()},
                    "oracle_ceiling": True,
                }
            )
    log("oracle_model_eval_done")
    return {
        "status": "ok",
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "features": features,
        "available": {
            "train_mfe_r": float(train["positive_mfe_r"].sum()),
            "train_positive_net_r": float(train["positive_net_r"].sum()),
            "holdout_mfe_r": float(holdout["positive_mfe_r"].sum()),
            "holdout_positive_net_r": float(holdout["positive_net_r"].sum()),
        },
        "rows": rows,
    }


def load_pipeline_summary() -> dict[str, Any]:
    if not PIPELINE_JSON.exists():
        return {}
    return json.loads(PIPELINE_JSON.read_text(encoding="utf-8"))


def render_report(payload: dict[str, Any]) -> str:
    pipe = payload.get("pipeline_stage_optimizer", {})
    baseline = pipe.get("baseline", {})
    holdout_rows = [row for row in pipe.get("holdout_ranked", []) if row.get("label") != "current_surfaced_all_ceiling"]
    primary = [
        row
        for row in holdout_rows
        if row.get("scope") == "dataset"
        and row.get("budget") in {"top2", "top4", "top8"}
        and not str(row.get("label", "")).startswith("oracle_")
    ][:12]
    stage2 = [
        row
        for row in holdout_rows
        if row.get("scope") == "surfaced"
        and row.get("budget") in {"top1", "top2", "top4", "top8", "current_count"}
        and not str(row.get("label", "")).startswith("oracle_")
    ][:12]
    constrained_full = [
        row
        for row in holdout_rows
        if row.get("scope") == "dataset"
        and not str(row.get("label", "")).startswith("oracle_")
        and row.get("feature_group") == "all_context"
        and num(row.get("selected_net_eod_r")) > 0.0
        and row.get("budget") in {"top1", "top2", "top4", "top8"}
    ][:8]
    balanced_full = [
        row
        for row in constrained_full
        if num(row.get("selected_net_eod_r")) >= 20.0 and num(row.get("lift_vs_current_selected_r")) >= 3.0
    ]
    high_r_surfaced = [
        row
        for row in holdout_rows
        if row.get("scope") == "surfaced"
        and not str(row.get("label", "")).startswith("oracle_")
        and row.get("feature_group") == "all_context"
        and row.get("budget") in {"top4", "top8"}
    ][:8]
    oracle_rows = payload.get("oracle_route_capture_optimizer", {}).get("rows", [])
    oracle_holdout = [row for row in oracle_rows if row.get("window") == "holdout" and not row.get("oracle_ceiling")]
    oracle_ceiling = [row for row in oracle_rows if row.get("window") == "holdout" and row.get("oracle_ceiling")]
    b_hold = baseline.get("holdout", {})
    lines = [
        "# KALCB R-Capture Optimizer",
        "",
        "Research-only train-frozen ranking study. Ex-post R/path outcomes are labels, not live features.",
        "",
        "## Funnel Baseline",
        "",
        f"- Holdout dataset available positive R: {num(b_hold.get('dataset_available_positive_r')):.1f}R",
        f"- Holdout currently surfaced positive R: {num(b_hold.get('surfaced_positive_r')):.1f}R ({pct(b_hold.get('surfaced_positive_r', 0) / max(num(b_hold.get('dataset_available_positive_r')), 1))})",
        f"- Holdout currently selected positive R: {num(b_hold.get('current_selected_positive_r')):.1f}R ({pct(b_hold.get('current_selected_dataset_retention'))} of dataset, {pct(b_hold.get('current_selected_surfaced_retention'))} of surfaced)",
        "",
        "## Constrained Implementation Queue",
        "",
    ]
    if balanced_full:
        row = balanced_full[0]
        lines.append(
            f"1. Replay `{row.get('label')}` `{row.get('budget')}` through shared core first: "
            f"{num(row.get('selected_positive_r')):.1f}R selected, {pct(row.get('dataset_r_retention'))} dataset retention, "
            f"{num(row.get('selected_net_eod_r')):.1f} proxy net EOD R, {num(row.get('lift_vs_current_selected_r')):.1f}x current selected R."
        )
    if constrained_full:
        row = constrained_full[0]
        lines.append(
            f"2. Carry `{row.get('label')}` `{row.get('budget')}` as the max all-context R-capture challenger: "
            f"{num(row.get('selected_positive_r')):.1f}R selected and {num(row.get('selected_net_eod_r')):.1f} proxy net EOD R."
        )
    if high_r_surfaced:
        row = high_r_surfaced[0]
        lines.append(
            f"3. Use current-surfaced `{row.get('label')}` `{row.get('budget')}` to isolate Stage 2 selection: "
            f"{num(row.get('selected_positive_r')):.1f}R selected ({pct(row.get('surfaced_r_retention'))} of surfaced), "
            f"but require route/exit repair because proxy net EOD is {num(row.get('selected_net_eod_r')):.1f}R."
        )
    if primary:
        row = primary[0]
        if row.get("feature_group") != "all_context":
            lines.append(
                f"4. Keep `{row.get('label')}` `{row.get('budget')}` as the max-R non-context challenger: "
                f"{num(row.get('selected_positive_r')):.1f}R selected, but do not promote without context and execution proof."
            )
    lines.extend(
        [
            "",
            "Progress threshold for the next shared-core replay: beat current holdout selected R by at least 3x and keep holdout net/drawdown hygiene positive after execution.",
            "",
            "## Best Full-Dataset Selection Probes",
            "",
            "| rank | label | budget | selected R | dataset retention | rows | hit share | net EOD R | lift vs current |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(primary[:10], start=1):
        lines.append(
            f"| {rank} | `{row.get('label')}` | {row.get('budget')} | {num(row.get('selected_positive_r')):.1f} | {pct(row.get('dataset_r_retention'))} | {int(num(row.get('selected_rows')))} | {pct(row.get('positive_r_hit_share'))} | {num(row.get('selected_net_eod_r')):.1f} | {num(row.get('lift_vs_current_selected_r')):.1f}x |"
        )
    lines.extend(
        [
            "",
            "## Best Surfaced-Only Stage 2 Probes",
            "",
            "| rank | label | budget | selected R | surfaced retention | rows | hit share | net EOD R | lift vs current |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(stage2[:10], start=1):
        lines.append(
            f"| {rank} | `{row.get('label')}` | {row.get('budget')} | {num(row.get('selected_positive_r')):.1f} | {pct(row.get('surfaced_r_retention'))} | {int(num(row.get('selected_rows')))} | {pct(row.get('positive_r_hit_share'))} | {num(row.get('selected_net_eod_r')):.1f} | {num(row.get('lift_vs_current_selected_r')):.1f}x |"
        )
    lines.extend(
        [
            "",
            "## Oracle Route/Capture Sanity",
            "",
            "| model | topK/day | MFE R | +Net R | MFE retention | +Net/MFE | route mix |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in oracle_holdout[:5]:
        lines.append(
            f"| train-frozen | {row.get('budget_topk_per_day')} | {num(row.get('selected_mfe_r')):.1f} | {num(row.get('selected_positive_net_r')):.1f} | {pct(row.get('mfe_retention'))} | {pct(row.get('positive_net_over_mfe'))} | `{row.get('route_family_counts')}` |"
        )
    for row in oracle_ceiling[:3]:
        lines.append(
            f"| oracle ceiling | {row.get('budget_topk_per_day')} | {num(row.get('selected_mfe_r')):.1f} | {num(row.get('selected_positive_net_r')):.1f} | {pct(row.get('mfe_retention'))} | {pct(row.get('positive_net_over_mfe'))} | `{row.get('route_family_counts')}` |"
        )
    lines.extend(
        [
            "",
            "## Top Train Correlations",
            "",
        ]
    )
    for row in pipe.get("top_feature_correlations", [])[:12]:
        lines.append(f"- `{row.get('feature')}`: Spearman {num(row.get('spearman_to_positive_r')):.3f}")
    lines.extend(
        [
            "",
            "## Progress Test",
            "",
            "- Right direction: selected positive R and dataset retention increase on holdout using train-frozen causal features.",
            "- Wrong direction: only current candidate pool improves while full dataset remains weak, or net EOD proxy collapses.",
            "- Next implementation should start with the highest holdout R-capture survivor, then replay it through shared core before any promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pipeline = read_pipeline()
    pipeline_result = evaluate_pipeline_rankers(pipeline)
    write_json(OUT_DIR / "pipeline_stage_optimizer_partial.json", pipeline_result)
    oracle_result = evaluate_oracle_ranker()
    payload = {
        "created_at_utc": now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "strategy": "kalcb",
        "round": 5,
        "profile": "r_capture_optimizer_train_frozen",
        "pipeline_summary": load_pipeline_summary().get("cross_window_interpretation", {}),
        "pipeline_stage_optimizer": pipeline_result,
        "oracle_route_capture_optimizer": oracle_result,
        "usage_contract": "research_only_ex_post_labels_for_training_eval_not_live_features",
    }
    write_json(OUT_DIR / "kalcb_r_capture_optimizer_results.json", payload)
    (OUT_DIR / "kalcb_r_capture_optimizer_report.md").write_text(render_report(payload), encoding="utf-8")
    log("complete", result_json=str(OUT_DIR / "kalcb_r_capture_optimizer_results.json"), report=str(OUT_DIR / "kalcb_r_capture_optimizer_report.md"))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
