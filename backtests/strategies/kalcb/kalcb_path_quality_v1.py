from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Any, Iterable

from strategy_common.clock import KST
from strategy_common.market import MarketBar


PATH_QUALITY_MODEL_VERSION = "kalcb-path-quality-v1"
PATH_CALIBRATION_SCORE_VERSION = "kalcb-path-risk-score-v1"
PATH_QUALITY_USAGE_CONTRACT = "research_source_calibration_only_not_live_entry_or_exit_rule"
INTERACTION_REGIME_MODEL_VERSION = "kalcb-interaction-regime-v1"
PATH_HORIZONS = (1, 3, 6, 12)

PATH_SCORE_WEIGHTS = {
    "broker_net_return_pct": 1000.0,
    "worst_fold_net": 160.0,
    "avg_mfe_capture": 80.0,
    "trade_count_frequency": 24.0,
    "broker_max_drawdown_pct": -700.0,
    "mae_tail_loss": -35.0,
    "giveback_loss": -4.0,
}

JOINT_CONTEXT_FEATURES = (
    "daily_return_5d",
    "daily_return_20d",
    "daily_return_60d",
    "daily_volume_ratio_20d",
    "daily_close20_loc",
    "daily_close60_loc",
    "daily_acceleration_5v20",
    "daily_momentum_pct",
    "sector_flow_participation",
    "sector_participation",
    "sector_daily_score_pct",
    "sector_daily_participation",
    "sector_daily_ret_5d",
    "sector_daily_ret_20d",
    "sector_daily_ret_60d",
    "sector_intraday_score_pct",
    "sector_intraday_ret",
    "sector_intraday_rel_volume",
    "sector_intraday_breadth",
    "sector_intraday_effective_count",
    "session_sector_intraday_score_pct_mean",
    "session_sector_intraday_positive_share",
    "session_sector_intraday_effective_count_mean",
    "daily_sector_alignment_pct",
    "stock_sector_daily_ret20_spread",
    "stock_sector_daily_ret5_spread",
    "first30_quality_pct",
    "first30_sector_ret_spread",
    "first30_sector_relvol_ratio",
    "first30_sector_leadership_pct",
    "first30_gap_relvol_sector_breadth",
    "first30_gap_retention_sector_breadth",
    "continuation_joint_quality_pct",
)

INTERACTION_REGIME_PAIRS = (
    ("first30_sector_leadership_pct", "daily_acceleration_5v20"),
    ("first30_gap_relvol", "daily_acceleration_5v20"),
    ("first30_quality_pct", "daily_acceleration_5v20"),
    ("continuation_joint_quality_pct", "daily_acceleration_5v20"),
    ("first30_gap_retention_ratio", "daily_acceleration_5v20"),
    ("first30_sector_leadership_pct", "daily_momentum_pct"),
    ("first30_sector_leadership_pct", "daily_sector_alignment_pct"),
    ("first30_sector_leadership_pct", "stock_sector_daily_ret20_spread"),
    ("first30_sector_leadership_pct", "session_sector_intraday_positive_share"),
    ("first30_sector_leadership_pct", "sector_intraday_score_pct"),
)


@dataclass(frozen=True, slots=True)
class PathObservation:
    trade_date: date
    symbol: str
    features: dict[str, float]
    labels: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def build_path_quality_observations(
    outcomes: Iterable[Any],
    compiled_replay: Any,
    context_by_key: dict[tuple[date, str], Any],
    bars_by_key: dict[tuple[date, str], tuple[MarketBar, ...]],
    *,
    horizons: tuple[int, ...] = PATH_HORIZONS,
) -> list[PathObservation]:
    candidates = _candidates_by_key(compiled_replay)
    observations: list[PathObservation] = []
    for outcome in sorted(outcomes, key=lambda item: (item.trade_date, item.symbol, str(item.entry_time))):
        trade_date = outcome.trade_date
        symbol = str(outcome.symbol)
        ctx = context_by_key.get((trade_date, symbol))
        bars = tuple(sorted(bars_by_key.get((trade_date, symbol), ()), key=lambda item: item.timestamp))
        entry_index = _entry_bar_index(bars, outcome.entry_time)
        if ctx is None or entry_index is None:
            continue
        candidate = candidates.get((trade_date, symbol))
        features = _base_features(outcome, ctx, candidate)
        for horizon in horizons:
            features.update(_horizon_features(bars, entry_index, outcome, ctx, int(horizon)))
        labels = _labels(outcome)
        observations.append(
            PathObservation(
                trade_date=trade_date,
                symbol=symbol,
                features=features,
                labels=labels,
                metadata={
                    "entry_time": str(outcome.entry_time),
                    "entry_type": str(getattr(outcome, "entry_type", "") or ""),
                    "frontier_role": str(getattr(outcome, "frontier_role", "") or ""),
                    "candidate_rank": int(getattr(outcome, "candidate_rank", 0) or 0),
                    "frontier_rank": int(getattr(outcome, "frontier_rank", 0) or 0),
                },
            )
        )
    return observations


def summarize_path_risk(
    observations: Iterable[PathObservation],
    *,
    selected_count: float = 0.0,
) -> dict[str, float]:
    rows = list(observations)
    losers = [row for row in rows if row.labels.get("loser", 0.0) > 0.0]
    selected = float(selected_count or 0.0)
    return {
        "observation_count": float(len(rows)),
        "trade_count": float(len(rows)),
        "selected_count": selected,
        "conversion": float(len(rows)) / selected if selected > 0.0 else 0.0,
        "avg_mfe_r": _avg(row.labels.get("max_mfe_r", 0.0) for row in rows),
        "median_mfe_r": _median(row.labels.get("max_mfe_r", 0.0) for row in rows),
        "avg_mfe_capture": _avg(row.labels.get("mfe_capture", 0.0) for row in rows),
        "median_mfe_capture": _median(row.labels.get("mfe_capture", 0.0) for row in rows),
        "avg_mae_r": _avg(row.labels.get("max_mae_r", 0.0) for row in rows),
        "avg_loser_mae_r": _avg(row.labels.get("max_mae_r", 0.0) for row in losers),
        "mae_le_neg_1_share": _share(row.labels.get("max_mae_r", 0.0) <= -1.0 for row in rows),
        "avg_giveback_r": _avg(row.labels.get("giveback_r", 0.0) for row in rows),
        "avg_loser_giveback_r": _avg(row.labels.get("giveback_r", 0.0) for row in losers),
        "loser_share": _share(row.labels.get("loser", 0.0) > 0.0 for row in rows),
        "tail_loser_share": _share(row.labels.get("tail_loser", 0.0) > 0.0 for row in rows),
    }


def fold_path_risk_metrics(
    observations: Iterable[PathObservation],
    folds: Iterable[tuple[date, date]],
) -> tuple[dict[str, Any], ...]:
    rows = list(observations)
    out = []
    for index, (start, end) in enumerate(folds, start=1):
        fold_rows = [row for row in rows if start <= row.trade_date <= end]
        out.append(
            {
                "fold": index,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "path_risk_metrics": summarize_path_risk(fold_rows),
            }
        )
    return tuple(out)


def fit_path_quality_model(
    observations: Iterable[PathObservation],
    folds: Iterable[tuple[date, date]],
) -> dict[str, Any]:
    rows = list(observations)
    fold_ranges = list(folds)
    if len(rows) < 4 or len(fold_ranges) < 2:
        return _rejected_model("too_few_observations_or_folds", rows, fold_ranges)

    candidates = []
    for feature_name, direction, quantiles in _rule_candidates(rows):
        for quantile in quantiles:
            fold_results = []
            valid_rule = True
            for fold_index, (start, end) in enumerate(fold_ranges, start=1):
                train = [row for row in rows if not (start <= row.trade_date <= end)]
                valid = [row for row in rows if start <= row.trade_date <= end]
                values = [row.features[feature_name] for row in train if feature_name in row.features]
                if not train or not valid or not values:
                    valid_rule = False
                    break
                threshold = _quantile(values, quantile)
                selected = [row for row in valid if _passes_rule(row, feature_name, direction, threshold)]
                min_count = max(1, round(0.10 * len(valid)))
                if len(selected) < min_count:
                    valid_rule = False
                    break
                result = _fold_lift(fold_index, valid, selected, feature_name, direction, threshold)
                if (
                    result["lift_r"] <= 0.0
                    or result["capture_lift"] < -0.05
                    or result["tail_loser_share_delta"] > 0.15
                ):
                    valid_rule = False
                    break
                fold_results.append(result)
            if not valid_rule or len(fold_results) != len(fold_ranges):
                continue
            median_lift = _median(item["lift_r"] for item in fold_results)
            worst_lift = min(item["lift_r"] for item in fold_results)
            score = median_lift + 0.35 * worst_lift + 0.25 * _median(item["capture_lift"] for item in fold_results)
            candidates.append(
                {
                    "score": float(score),
                    "feature": feature_name,
                    "direction": direction,
                    "threshold_quantile": float(quantile),
                    "median_fold_lift_r": float(median_lift),
                    "worst_fold_lift_r": float(worst_lift),
                    "folds": fold_results,
                }
            )
    if not candidates:
        return _rejected_model("no_fold_stable_positive_lift_rule", rows, fold_ranges)
    candidates.sort(key=lambda item: (-float(item["score"]), item["feature"], item["direction"], float(item["threshold_quantile"])))
    best = candidates[0]
    return {
        "version": PATH_QUALITY_MODEL_VERSION,
        "usage_contract": PATH_QUALITY_USAGE_CONTRACT,
        "accepted": True,
        "reject_reason": "",
        "observation_count": float(len(rows)),
        "fold_count": float(len(fold_ranges)),
        "rule": {
            "feature": best["feature"],
            "direction": best["direction"],
            "threshold_quantile": best["threshold_quantile"],
        },
        "median_fold_lift_r": best["median_fold_lift_r"],
        "worst_fold_lift_r": best["worst_fold_lift_r"],
        "folds": best["folds"],
        "candidate_rule_count": float(len(candidates)),
    }


def fit_interaction_regime_model(
    observations: Iterable[PathObservation],
    folds: Iterable[tuple[date, date]],
    *,
    min_fold_selected_share: float = 0.05,
) -> dict[str, Any]:
    rows = list(observations)
    fold_ranges = list(folds)
    if len(rows) < 4 or len(fold_ranges) < 2:
        return _rejected_interaction_model("too_few_observations_or_folds", rows, fold_ranges)

    candidates = []
    for pair in _interaction_regime_pairs(rows):
        feature_a, feature_b = pair
        for quantile_a, quantile_b in _interaction_quantile_grid(feature_a, feature_b):
            fold_results = []
            valid_rule = True
            for fold_index, (start, end) in enumerate(fold_ranges, start=1):
                train = [row for row in rows if not (start <= row.trade_date <= end)]
                valid = [row for row in rows if start <= row.trade_date <= end]
                values_a = [_feature_float(row, feature_a) for row in train]
                values_b = [_feature_float(row, feature_b) for row in train]
                values_a = [value for value in values_a if value is not None]
                values_b = [value for value in values_b if value is not None]
                if not train or not valid or not values_a or not values_b:
                    valid_rule = False
                    break
                direction_a = _feature_direction(feature_a)
                direction_b = _feature_direction(feature_b)
                threshold_a = _quantile(values_a, quantile_a)
                threshold_b = _quantile(values_b, quantile_b)
                conditions = (
                    (feature_a, direction_a, threshold_a),
                    (feature_b, direction_b, threshold_b),
                )
                selected = [row for row in valid if _passes_interaction_rule(row, conditions)]
                min_count = max(1, round(max(float(min_fold_selected_share), 0.0) * len(valid)))
                if len(selected) < min_count:
                    valid_rule = False
                    break
                result = _fold_interaction_lift(
                    fold_index,
                    valid,
                    selected,
                    (
                        (feature_a, direction_a, threshold_a, quantile_a),
                        (feature_b, direction_b, threshold_b, quantile_b),
                    ),
                )
                if (
                    result["lift_r"] <= 0.0
                    or result["capture_lift"] < -0.05
                    or result["tail_loser_share_delta"] > 0.15
                ):
                    valid_rule = False
                    break
                fold_results.append(result)
            if not valid_rule or len(fold_results) != len(fold_ranges):
                continue

            median_lift = _median(item["lift_r"] for item in fold_results)
            worst_lift = min(item["lift_r"] for item in fold_results)
            median_capture_lift = _median(item["capture_lift"] for item in fold_results)
            median_tail_delta = _median(item["tail_loser_share_delta"] for item in fold_results)
            median_share = _median(item["selected_share"] for item in fold_results)
            score = (
                median_lift
                + 0.45 * worst_lift
                + 0.20 * median_capture_lift
                - 0.50 * max(0.0, median_tail_delta)
                + 0.05 * median_share
            )
            candidates.append(
                {
                    "score": float(score),
                    "features": pair,
                    "pair_priority": _pair_priority(pair),
                    "directions": (_feature_direction(feature_a), _feature_direction(feature_b)),
                    "threshold_quantiles": (float(quantile_a), float(quantile_b)),
                    "median_fold_lift_r": float(median_lift),
                    "worst_fold_lift_r": float(worst_lift),
                    "median_capture_lift": float(median_capture_lift),
                    "median_tail_loser_share_delta": float(median_tail_delta),
                    "median_selected_share": float(median_share),
                    "folds": fold_results,
                }
            )
    if not candidates:
        return _rejected_interaction_model("no_fold_stable_stock_leadership_daily_acceleration_rule", rows, fold_ranges)

    candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item.get("pair_priority", 9999)),
            tuple(str(feature) for feature in item["features"]),
            tuple(float(value) for value in item["threshold_quantiles"]),
        )
    )
    best = candidates[0]
    rule_conditions = []
    for feature, direction, quantile in zip(best["features"], best["directions"], best["threshold_quantiles"], strict=True):
        all_values = [_feature_float(row, str(feature)) for row in rows]
        all_values = [value for value in all_values if value is not None]
        rule_conditions.append(
            {
                "feature": str(feature),
                "direction": str(direction),
                "threshold_quantile": float(quantile),
                "threshold": float(_quantile(all_values, float(quantile))) if all_values else 0.0,
            }
        )
    return {
        "version": INTERACTION_REGIME_MODEL_VERSION,
        "usage_contract": PATH_QUALITY_USAGE_CONTRACT,
        "model_type": "two_feature_and_chronological_oof_rule",
        "activation_label": "stock_leadership_plus_daily_acceleration",
        "accepted": True,
        "reject_reason": "",
        "observation_count": float(len(rows)),
        "fold_count": float(len(fold_ranges)),
        "rule": {
            "join": "and",
            "conditions": tuple(rule_conditions),
        },
        "median_fold_lift_r": best["median_fold_lift_r"],
        "worst_fold_lift_r": best["worst_fold_lift_r"],
        "median_capture_lift": best["median_capture_lift"],
        "median_tail_loser_share_delta": best["median_tail_loser_share_delta"],
        "median_selected_share": best["median_selected_share"],
        "folds": best["folds"],
        "candidate_rule_count": float(len(candidates)),
    }


def score_interaction_regime(features: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    if not model.get("accepted"):
        return {"active": False, "score": 0.0, "passed_count": 0, "condition_results": ()}
    conditions = tuple(dict(item) for item in dict(model.get("rule") or {}).get("conditions", ()) or ())
    results = []
    for condition in conditions:
        feature = str(condition.get("feature") or "")
        value = _coerce_float(features.get(feature))
        threshold = _coerce_float(condition.get("threshold"))
        direction = str(condition.get("direction") or "gte")
        passed = bool(value is not None and threshold is not None and _passes_value(value, direction, threshold))
        results.append(
            {
                "feature": feature,
                "direction": direction,
                "threshold": float(threshold or 0.0),
                "value": float(value or 0.0),
                "passed": passed,
            }
        )
    active = bool(results) and all(bool(item["passed"]) for item in results)
    return {
        "active": active,
        "score": float(model.get("median_fold_lift_r", 0.0) or 0.0) if active else 0.0,
        "passed_count": sum(1 for item in results if item["passed"]),
        "condition_results": tuple(results),
    }


def score_path_calibrated_row(
    metrics: dict[str, Any],
    path_risk_metrics: dict[str, Any],
    fold_rows: Iterable[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    fold_nets = [_fold_net(row) for row in fold_rows]
    broker_net = float(metrics.get("broker_net_return_pct", metrics.get("portfolio_equivalent_net_return_pct", 0.0)) or 0.0)
    drawdown = abs(float(metrics.get("broker_max_drawdown_pct", metrics.get("portfolio_equivalent_max_drawdown_pct", 0.0)) or 0.0))
    trade_count = float(metrics.get("trade_count", path_risk_metrics.get("trade_count", 0.0)) or 0.0)
    components = {
        "broker_net_return_pct": PATH_SCORE_WEIGHTS["broker_net_return_pct"] * broker_net,
        "worst_fold_net": PATH_SCORE_WEIGHTS["worst_fold_net"] * (min(fold_nets) if fold_nets else broker_net),
        "avg_mfe_capture": PATH_SCORE_WEIGHTS["avg_mfe_capture"] * float(path_risk_metrics.get("avg_mfe_capture", metrics.get("avg_mfe_capture", 0.0)) or 0.0),
        "trade_count_frequency": PATH_SCORE_WEIGHTS["trade_count_frequency"] * min(max(trade_count / 100.0, 0.0), 1.25),
        "broker_max_drawdown_pct": PATH_SCORE_WEIGHTS["broker_max_drawdown_pct"] * drawdown,
        "mae_tail_loss": PATH_SCORE_WEIGHTS["mae_tail_loss"] * float(path_risk_metrics.get("mae_le_neg_1_share", metrics.get("mae_le_neg_1_share", 0.0)) or 0.0),
        "giveback_loss": PATH_SCORE_WEIGHTS["giveback_loss"] * float(path_risk_metrics.get("avg_giveback_r", 0.0) or 0.0),
    }
    return float(sum(components.values())), components


def _base_features(outcome: Any, ctx: Any, candidate: Any | None) -> dict[str, float]:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    gap = float(getattr(ctx, "gap", 0.0) or 0.0)
    rel_volume = float(getattr(ctx, "rel_volume", 0.0) or 0.0)
    low_vs_prev = float(getattr(ctx, "low_vs_prev_close", 0.0) or 0.0)
    gap_retention_ratio = low_vs_prev / max(abs(gap), 1e-6) if gap > 0.0 else 0.0
    rel_volume_log = _log1p_nonnegative(rel_volume)
    features = {
        "candidate_rank": float(getattr(outcome, "candidate_rank", 0) or metadata.get("candidate_rank", 0) or 0),
        "frontier_rank": float(getattr(outcome, "frontier_rank", 0) or metadata.get("frontier_rank", 0) or 0),
        "frontier_selection_score": float(metadata.get("frontier_selection_score", 0.0) or 0.0),
        "first30_score": float(metadata.get("first30_score", 0.0) or 0.0),
        "first30_ret": float(getattr(ctx, "first30_ret", 0.0) or 0.0),
        "first30_vwap_ret": float(getattr(ctx, "vwap_ret", 0.0) or 0.0),
        "first30_gap": gap,
        "first30_rel_volume": rel_volume,
        "first30_cpr": float(getattr(ctx, "close_location", 0.0) or 0.0),
        "first30_open_drawdown": float(getattr(ctx, "open_drawdown", 0.0) or 0.0),
        "first30_low_vs_prev_close": low_vs_prev,
        "first30_range_atr": float(getattr(ctx, "range_atr", 0.0) or 0.0),
        "first30_gap_retention_ratio": float(gap_retention_ratio),
        "first30_gap_relvol": float(gap * rel_volume_log),
        "first30_low_vs_prev_relvol": float(low_vs_prev * rel_volume_log),
    }
    _add_context_defaults(features, ctx, gap_retention_ratio, rel_volume_log)
    for key in JOINT_CONTEXT_FEATURES:
        value = _coerce_float(metadata.get(key))
        if value is not None:
            features[key] = value
    return features


def _add_context_defaults(features: dict[str, float], ctx: Any, gap_retention_ratio: float, rel_volume_log: float) -> None:
    daily = getattr(ctx, "daily", None)
    if daily is not None:
        ret5 = float(getattr(daily, "return_5d", 0.0) or 0.0)
        ret20 = float(getattr(daily, "return_20d", 0.0) or 0.0)
        ret60 = float(getattr(daily, "return_60d", 0.0) or 0.0)
        volume_ratio = float(getattr(daily, "volume_ratio_20d", 0.0) or 0.0)
        close20 = float(getattr(daily, "close20_loc", 0.0) or 0.0)
        close60 = float(getattr(daily, "close60_loc", 0.0) or 0.0)
        acceleration = ret5 - 0.25 * ret20
        features.setdefault("daily_return_5d", ret5)
        features.setdefault("daily_return_20d", ret20)
        features.setdefault("daily_return_60d", ret60)
        features.setdefault("daily_volume_ratio_20d", volume_ratio)
        features.setdefault("daily_close20_loc", close20)
        features.setdefault("daily_close60_loc", close60)
        features.setdefault("daily_acceleration_5v20", acceleration)
        features.setdefault(
            "daily_momentum_pct",
            100.0
            * (
                0.30 * _bounded(0.5 + ret20 / 0.40, 0.0, 1.0)
                + 0.18 * _bounded(0.5 + ret60 / 0.70, 0.0, 1.0)
                + 0.17 * _bounded(close20, 0.0, 1.0)
                + 0.12 * _bounded(close60, 0.0, 1.0)
                + 0.13 * _bounded(0.5 + math.log(max(volume_ratio, 0.1)) / 4.0, 0.0, 1.0)
                + 0.10 * _bounded(0.5 + acceleration / 0.12, 0.0, 1.0)
            ),
        )

    sector_daily = getattr(ctx, "sector_daily", None)
    sector_intraday = getattr(ctx, "sector_intraday", None)
    sector_daily_metadata = dict(sector_daily.metadata()) if sector_daily is not None else {}
    sector_intraday_metadata = dict(sector_intraday.metadata()) if sector_intraday is not None else {}
    for key, value in {**sector_daily_metadata, **sector_intraday_metadata}.items():
        coerced = _coerce_float(value)
        if coerced is not None:
            features.setdefault(key, coerced)

    flow = getattr(ctx, "flow", None)
    features.setdefault("sector_flow_participation", float(getattr(flow, "sector_participation", 0.0) or 0.0))
    sector_daily_ret_5d = float(features.get("sector_daily_ret_5d", 0.0) or 0.0)
    sector_daily_ret_20d = float(features.get("sector_daily_ret_20d", 0.0) or 0.0)
    sector_intraday_ret = float(features.get("sector_intraday_ret", 0.0) or 0.0)
    sector_intraday_rel_volume = max(float(features.get("sector_intraday_rel_volume", 1.0) or 1.0), 1e-6)
    sector_intraday_breadth = float(features.get("sector_intraday_breadth", 0.5) or 0.5)
    first30_sector_ret_spread = float(features.get("first30_ret", 0.0) or 0.0) - sector_intraday_ret
    first30_sector_relvol_ratio = float(features.get("first30_rel_volume", 0.0) or 0.0) / sector_intraday_rel_volume
    stock_sector_daily_ret20_spread = float(features.get("daily_return_20d", 0.0) or 0.0) - sector_daily_ret_20d
    stock_sector_daily_ret5_spread = float(features.get("daily_return_5d", 0.0) or 0.0) - sector_daily_ret_5d
    first30_quality_pct = 100.0 * (
        0.24 * _bounded(0.5 + float(features.get("first30_ret", 0.0) or 0.0) / 0.06, 0.0, 1.0)
        + 0.18 * _bounded(0.5 + float(features.get("first30_vwap_ret", 0.0) or 0.0) / 0.04, 0.0, 1.0)
        + 0.18 * _bounded(float(features.get("first30_cpr", 0.0) or 0.0), 0.0, 1.0)
        + 0.16 * _bounded(rel_volume_log / math.log(21.0), 0.0, 1.0)
        + 0.14 * _bounded(float(gap_retention_ratio), 0.0, 1.25) / 1.25
        + 0.10 * _bounded(0.5 + float(features.get("first30_low_vs_prev_close", 0.0) or 0.0) / 0.08, 0.0, 1.0)
    )
    first30_sector_leadership_pct = 100.0 * (
        0.45 * _bounded(0.5 + first30_sector_ret_spread / 0.06, 0.0, 1.0)
        + 0.25 * _bounded(math.log(max(first30_sector_relvol_ratio, 0.1)) / 3.0 + 0.5, 0.0, 1.0)
        + 0.20 * _bounded(float(features.get("first30_cpr", 0.0) or 0.0), 0.0, 1.0)
        + 0.10 * _bounded(float(gap_retention_ratio), 0.0, 1.25) / 1.25
    )
    daily_sector_alignment_pct = 100.0 * (
        0.45 * _bounded(0.5 + stock_sector_daily_ret20_spread / 0.40, 0.0, 1.0)
        + 0.25 * _bounded(0.5 + stock_sector_daily_ret5_spread / 0.16, 0.0, 1.0)
        + 0.20 * _bounded(float(features.get("sector_daily_score_pct", 50.0) or 50.0) / 100.0, 0.0, 1.0)
        + 0.10 * _bounded(float(features.get("sector_daily_participation", 0.0) or 0.0), 0.0, 1.0)
    )
    features.setdefault("sector_participation", float(features.get("sector_daily_participation", features.get("sector_flow_participation", 0.0)) or 0.0))
    features.setdefault("first30_quality_pct", first30_quality_pct)
    features.setdefault("first30_sector_ret_spread", first30_sector_ret_spread)
    features.setdefault("first30_sector_relvol_ratio", first30_sector_relvol_ratio)
    features.setdefault("first30_sector_leadership_pct", first30_sector_leadership_pct)
    features.setdefault("stock_sector_daily_ret20_spread", stock_sector_daily_ret20_spread)
    features.setdefault("stock_sector_daily_ret5_spread", stock_sector_daily_ret5_spread)
    features.setdefault("daily_sector_alignment_pct", daily_sector_alignment_pct)
    features.setdefault("first30_gap_relvol_sector_breadth", float(features.get("first30_gap_relvol", 0.0) or 0.0) * sector_intraday_breadth)
    features.setdefault("first30_gap_retention_sector_breadth", float(gap_retention_ratio) * sector_intraday_breadth)
    features.setdefault(
        "continuation_joint_quality_pct",
        100.0
        * (
            0.34 * first30_quality_pct / 100.0
            + 0.23 * float(features.get("daily_momentum_pct", 0.0) or 0.0) / 100.0
            + 0.18 * first30_sector_leadership_pct / 100.0
            + 0.15 * daily_sector_alignment_pct / 100.0
            + 0.10 * _bounded(sector_intraday_breadth, 0.0, 1.0)
        ),
    )


def _horizon_features(
    bars: tuple[MarketBar, ...],
    entry_index: int,
    outcome: Any,
    ctx: Any,
    horizon: int,
) -> dict[str, float]:
    stop = min(len(bars), entry_index + max(1, int(horizon)))
    window = bars[entry_index:stop]
    if not window:
        return {}
    entry = max(float(getattr(outcome, "entry_price", 0.0) or 0.0), 1e-9)
    risk = max(float(getattr(outcome, "risk_per_share", 0.0) or 0.0), 1e-9)
    high = max(float(bar.high) for bar in window)
    low = min(float(bar.low) for bar in window)
    last = window[-1]
    current_r = (float(last.close) - entry) / risk
    mfe_r = max(0.0, (high - entry) / risk)
    mae_r = (low - entry) / risk
    vwap = _running_vwap(bars[:stop])
    or_high, or_low = _opening_range(ctx, bars)
    prefix = f"h{int(horizon)}"
    return {
        f"{prefix}_current_r": float(current_r),
        f"{prefix}_mfe_r": float(mfe_r),
        f"{prefix}_mae_r": float(mae_r),
        f"{prefix}_giveback_r": float(max(0.0, mfe_r - current_r)),
        f"{prefix}_vwap_ret": float(float(last.close) / max(vwap, 1e-9) - 1.0),
        f"{prefix}_or_position": float((float(last.close) - or_low) / max(or_high - or_low, 1e-9)),
        f"{prefix}_close_location": float((float(last.close) - float(last.low)) / max(float(last.high) - float(last.low), 1e-9)),
        f"{prefix}_recent_return": float(float(last.close) / entry - 1.0),
        f"{prefix}_down_streak": float(_down_streak(window)),
        f"{prefix}_below_entry_streak": float(_threshold_streak(window, entry, "close_lt")),
        f"{prefix}_below_vwap_streak": float(_threshold_streak_against_vwap(bars[:stop])),
    }


def _labels(outcome: Any) -> dict[str, float]:
    entry = max(float(getattr(outcome, "entry_price", 0.0) or 0.0), 1e-9)
    risk = max(float(getattr(outcome, "risk_per_share", 0.0) or 0.0), 1e-9)
    net_pct = float(getattr(outcome, "net_return_pct", 0.0) or 0.0)
    final_net_r = net_pct * entry / risk
    mfe_r = float(getattr(outcome, "mfe_r", 0.0) or 0.0)
    mae_r = float(getattr(outcome, "mae_r", 0.0) or 0.0)
    loser = 1.0 if net_pct < 0.0 else 0.0
    return {
        "final_net_pct": net_pct,
        "final_net_r": float(final_net_r),
        "mfe_capture": float(getattr(outcome, "mfe_capture", 0.0) or 0.0),
        "max_mfe_r": mfe_r,
        "max_mae_r": mae_r,
        "giveback_r": float(max(0.0, mfe_r - final_net_r)),
        "loser": loser,
        "tail_loser": 1.0 if loser and mae_r <= -1.0 else 0.0,
    }


def _rule_candidates(rows: list[PathObservation]) -> list[tuple[str, str, tuple[float, ...]]]:
    available = {key for row in rows for key in row.features}
    high_features = (
        "first30_gap",
        "first30_low_vs_prev_close",
        "first30_gap_retention_ratio",
        "first30_gap_relvol",
        "first30_low_vs_prev_relvol",
        "first30_rel_volume",
        "first30_ret",
        "first30_vwap_ret",
        "first30_cpr",
        "h1_current_r",
        "h3_current_r",
        "h6_current_r",
        "h12_current_r",
        "h3_mfe_r",
        "h6_mfe_r",
        "h12_mfe_r",
        "h3_mae_r",
        "h6_mae_r",
        "h3_close_location",
        "h6_close_location",
        "h3_vwap_ret",
        "h6_vwap_ret",
        "h3_or_position",
        "h6_or_position",
    )
    low_features = (
        "frontier_rank",
        "h3_giveback_r",
        "h6_giveback_r",
        "h12_giveback_r",
        "h3_down_streak",
        "h6_down_streak",
        "h3_below_vwap_streak",
        "h6_below_vwap_streak",
    )
    out = [(name, "gte", (0.50, 0.60, 0.70)) for name in high_features if name in available]
    out.extend((name, "lte", (0.30, 0.40, 0.50)) for name in low_features if name in available)
    return out


def _interaction_regime_pairs(rows: list[PathObservation]) -> list[tuple[str, str]]:
    available = {key for row in rows for key in row.features}
    pairs = [pair for pair in INTERACTION_REGIME_PAIRS if pair[0] in available and pair[1] in available]
    if pairs:
        return pairs
    daily_features = [feature for feature in available if feature.startswith("daily_") or feature.startswith("stock_sector_daily_")]
    leadership_features = [
        feature
        for feature in available
        if feature.startswith("first30_") or feature in {"continuation_joint_quality_pct", "sector_intraday_score_pct"}
    ]
    out: list[tuple[str, str]] = []
    for leader in sorted(leadership_features):
        for daily in sorted(daily_features):
            if leader != daily:
                out.append((leader, daily))
    return out[:30]


def _interaction_quantile_grid(feature_a: str, feature_b: str) -> tuple[tuple[float, float], ...]:
    quantiles_a = _candidate_quantiles_for_direction(_feature_direction(feature_a))
    quantiles_b = _candidate_quantiles_for_direction(_feature_direction(feature_b))
    return tuple((a, b) for a in quantiles_a for b in quantiles_b)


def _pair_priority(pair: tuple[str, str]) -> int:
    try:
        return INTERACTION_REGIME_PAIRS.index(pair)
    except ValueError:
        return 9999


def _candidate_quantiles_for_direction(direction: str) -> tuple[float, ...]:
    return (0.25, 0.35, 0.45) if direction == "lte" else (0.55, 0.65, 0.75, 0.85)


def _feature_direction(feature: str) -> str:
    if feature in {"frontier_rank", "candidate_rank"} or feature.endswith("_drawdown"):
        return "lte"
    return "gte"


def _feature_float(row: PathObservation, feature: str) -> float | None:
    return _coerce_float(row.features.get(feature))


def _passes_interaction_rule(row: PathObservation, conditions: tuple[tuple[str, str, float], ...]) -> bool:
    for feature, direction, threshold in conditions:
        value = _feature_float(row, feature)
        if value is None or not _passes_value(value, direction, threshold):
            return False
    return True


def _fold_interaction_lift(
    fold_index: int,
    valid: list[PathObservation],
    selected: list[PathObservation],
    conditions: tuple[tuple[str, str, float, float], ...],
) -> dict[str, Any]:
    base_r = _avg(row.labels.get("final_net_r", 0.0) for row in valid)
    sel_r = _avg(row.labels.get("final_net_r", 0.0) for row in selected)
    base_capture = _avg(row.labels.get("mfe_capture", 0.0) for row in valid)
    sel_capture = _avg(row.labels.get("mfe_capture", 0.0) for row in selected)
    base_tail = _share(row.labels.get("tail_loser", 0.0) > 0.0 for row in valid)
    sel_tail = _share(row.labels.get("tail_loser", 0.0) > 0.0 for row in selected)
    base_giveback = _avg(row.labels.get("giveback_r", 0.0) for row in valid)
    sel_giveback = _avg(row.labels.get("giveback_r", 0.0) for row in selected)
    return {
        "fold": int(fold_index),
        "conditions": tuple(
            {
                "feature": feature,
                "direction": direction,
                "threshold": float(threshold),
                "threshold_quantile": float(quantile),
            }
            for feature, direction, threshold, quantile in conditions
        ),
        "validation_count": int(len(valid)),
        "selected_count": int(len(selected)),
        "selected_share": float(len(selected)) / max(float(len(valid)), 1.0),
        "base_avg_r": float(base_r),
        "selected_avg_r": float(sel_r),
        "lift_r": float(sel_r - base_r),
        "capture_lift": float(sel_capture - base_capture),
        "tail_loser_share_delta": float(sel_tail - base_tail),
        "giveback_delta_r": float(sel_giveback - base_giveback),
    }


def _fold_lift(
    fold_index: int,
    valid: list[PathObservation],
    selected: list[PathObservation],
    feature: str,
    direction: str,
    threshold: float,
) -> dict[str, float | int | str]:
    base_r = _avg(row.labels.get("final_net_r", 0.0) for row in valid)
    sel_r = _avg(row.labels.get("final_net_r", 0.0) for row in selected)
    base_capture = _avg(row.labels.get("mfe_capture", 0.0) for row in valid)
    sel_capture = _avg(row.labels.get("mfe_capture", 0.0) for row in selected)
    base_tail = _share(row.labels.get("tail_loser", 0.0) > 0.0 for row in valid)
    sel_tail = _share(row.labels.get("tail_loser", 0.0) > 0.0 for row in selected)
    return {
        "fold": int(fold_index),
        "feature": feature,
        "direction": direction,
        "threshold": float(threshold),
        "validation_count": int(len(valid)),
        "selected_count": int(len(selected)),
        "selected_share": float(len(selected)) / max(float(len(valid)), 1.0),
        "lift_r": float(sel_r - base_r),
        "capture_lift": float(sel_capture - base_capture),
        "tail_loser_share_delta": float(sel_tail - base_tail),
    }


def _rejected_model(reason: str, rows: list[PathObservation], folds: list[tuple[date, date]]) -> dict[str, Any]:
    return {
        "version": PATH_QUALITY_MODEL_VERSION,
        "usage_contract": PATH_QUALITY_USAGE_CONTRACT,
        "accepted": False,
        "reject_reason": reason,
        "observation_count": float(len(rows)),
        "fold_count": float(len(folds)),
        "rule": {},
        "median_fold_lift_r": 0.0,
        "worst_fold_lift_r": 0.0,
        "folds": [],
        "candidate_rule_count": 0.0,
    }


def _rejected_interaction_model(reason: str, rows: list[PathObservation], folds: list[tuple[date, date]]) -> dict[str, Any]:
    return {
        "version": INTERACTION_REGIME_MODEL_VERSION,
        "usage_contract": PATH_QUALITY_USAGE_CONTRACT,
        "model_type": "two_feature_and_chronological_oof_rule",
        "activation_label": "stock_leadership_plus_daily_acceleration",
        "accepted": False,
        "reject_reason": reason,
        "observation_count": float(len(rows)),
        "fold_count": float(len(folds)),
        "rule": {},
        "median_fold_lift_r": 0.0,
        "worst_fold_lift_r": 0.0,
        "median_capture_lift": 0.0,
        "median_tail_loser_share_delta": 0.0,
        "median_selected_share": 0.0,
        "folds": [],
        "candidate_rule_count": 0.0,
    }


def _candidates_by_key(compiled_replay: Any) -> dict[tuple[date, str], Any]:
    out: dict[tuple[date, str], Any] = {}
    for day, snapshot in dict(getattr(compiled_replay, "snapshots", {}) or {}).items():
        for candidate in getattr(snapshot, "candidates", ()) or ():
            out[(day, str(candidate.symbol))] = candidate
    return out


def _entry_bar_index(bars: tuple[MarketBar, ...], entry_time: Any) -> int | None:
    if not bars:
        return None
    try:
        target = entry_time.astimezone(KST)
    except AttributeError:
        return None
    for index, bar in enumerate(bars):
        ts = bar.timestamp.astimezone(KST)
        if ts == target:
            return index
    for index, bar in enumerate(bars):
        if bar.timestamp.astimezone(KST) >= target:
            return index
    return None


def _opening_range(ctx: Any, bars: tuple[MarketBar, ...]) -> tuple[float, float]:
    source = tuple(getattr(ctx, "bars", ()) or ())[:6] or bars[:6] or bars[:1]
    return max(float(bar.high) for bar in source), min(float(bar.low) for bar in source)


def _running_vwap(bars: tuple[MarketBar, ...]) -> float:
    volume = sum(max(float(bar.volume), 0.0) for bar in bars)
    if volume <= 0:
        return max(float(bars[-1].close), 1e-9) if bars else 1e-9
    value = sum(((float(bar.high) + float(bar.low) + float(bar.close)) / 3.0) * max(float(bar.volume), 0.0) for bar in bars)
    return max(value / volume, 1e-9)


def _log1p_nonnegative(value: float) -> float:
    return math.log1p(max(float(value or 0.0), 0.0))


def _down_streak(window: tuple[MarketBar, ...]) -> int:
    streak = 0
    previous_close: float | None = None
    for bar in window:
        close = float(bar.close)
        if previous_close is not None and close < previous_close:
            streak += 1
        else:
            streak = 0
        previous_close = close
    return streak


def _threshold_streak(window: tuple[MarketBar, ...], threshold: float, mode: str) -> int:
    streak = 0
    for bar in window:
        value = float(bar.close)
        ok = value < threshold if mode == "close_lt" else value > threshold
        streak = streak + 1 if ok else 0
    return streak


def _threshold_streak_against_vwap(bars: tuple[MarketBar, ...]) -> int:
    streak = 0
    for index, bar in enumerate(bars):
        vwap = _running_vwap(bars[: index + 1])
        streak = streak + 1 if float(bar.close) < vwap else 0
    return streak


def _passes_rule(row: PathObservation, feature: str, direction: str, threshold: float) -> bool:
    value = row.features.get(feature)
    if value is None:
        return False
    return _passes_value(float(value), direction, float(threshold))


def _passes_value(value: float, direction: str, threshold: float) -> bool:
    if direction == "lte":
        return float(value) <= float(threshold)
    return float(value) >= float(threshold)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _quantile(values: Iterable[float], q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    index = int(round((len(rows) - 1) * max(0.0, min(float(q), 1.0))))
    return rows[index]


def _fold_net(row: dict[str, Any]) -> float:
    metrics = dict(row.get("metrics") or row.get("calibration_metrics") or {})
    for key in ("broker_net_return_pct", "portfolio_equivalent_net_return_pct", "slot_cumulative_net_return_pct"):
        if key in metrics:
            return float(metrics.get(key, 0.0) or 0.0)
    return 0.0


def _avg(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / max(float(len(rows)), 1.0)


def _median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(median(rows)) if rows else 0.0


def _share(values: Iterable[bool]) -> float:
    rows = [bool(value) for value in values]
    return sum(1 for value in rows if value) / max(float(len(rows)), 1.0)


def _bounded(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))
