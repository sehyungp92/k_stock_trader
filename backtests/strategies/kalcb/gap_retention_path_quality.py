from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Any, Callable, Iterable, Mapping

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.first30 import FIRST30_END

from .kalcb_path_quality_v1 import JOINT_CONTEXT_FEATURES


GAP_RETENTION_REPORT_VERSION = "kalcb-gap-retention-path-quality-v1"
PATH_HORIZONS = (1, 3, 6, 12)


@dataclass(frozen=True, slots=True)
class CandidatePathRow:
    window: str
    trade_date: date
    symbol: str
    frontier_role: str
    frontier_rank: int
    candidate_rank: int
    features: dict[str, float]
    labels: dict[str, float]


def build_candidate_path_rows(
    context: Any,
    *,
    window: str,
    stop_pct: float,
    round_trip_cost_pct: float = 0.0028,
    horizons: tuple[int, ...] = PATH_HORIZONS,
) -> list[CandidatePathRow]:
    rows: list[CandidatePathRow] = []
    dates = tuple(getattr(context, "train_dates", ()) or getattr(context, "compiled_replay", None).session_dates)
    snapshots = dict(getattr(context.compiled_replay, "snapshots", {}) or {})
    bars_by_key = dict(getattr(context.dataset, "bars_by_key", {}) or {})
    context_by_key = dict(getattr(context, "context_by_key", {}) or {})
    for day in sorted(dates):
        snapshot = snapshots.get(day)
        if snapshot is None:
            continue
        for candidate in tuple(getattr(snapshot, "candidates", ()) or ()):
            symbol = str(candidate.symbol)
            bars = tuple(sorted(bars_by_key.get((day, symbol), ()), key=lambda item: item.timestamp))
            entry_index = _first30_entry_index(bars)
            ctx = context_by_key.get((day, symbol))
            if ctx is None or entry_index is None:
                continue
            row = _candidate_path_row(
                window=window,
                day=day,
                candidate=candidate,
                ctx=ctx,
                bars=bars,
                entry_index=entry_index,
                stop_pct=stop_pct,
                round_trip_cost_pct=round_trip_cost_pct,
                horizons=horizons,
            )
            if row is not None:
                rows.append(row)
    return rows


def build_gap_retention_report(
    train_rows: Iterable[CandidatePathRow],
    holdout_rows: Iterable[CandidatePathRow],
    *,
    min_holdout_rows: int = 8,
) -> dict[str, Any]:
    train = list(train_rows)
    holdout = list(holdout_rows)
    thresholds = _train_thresholds(train)
    rules = _rule_specs(thresholds)
    role_values = ["all", "initial_active", "frontier_shadow"]
    rule_rows: list[dict[str, Any]] = []
    for role in role_values:
        train_role = _filter_role(train, role)
        holdout_role = _filter_role(holdout, role)
        base_train = _summarize_rows(train_role)
        base_holdout = _summarize_rows(holdout_role)
        for name, description, predicate in rules:
            train_selected = [row for row in train_role if predicate(row)]
            holdout_selected = [row for row in holdout_role if predicate(row)]
            train_summary = _summarize_rows(train_selected, baseline=base_train)
            holdout_summary = _summarize_rows(holdout_selected, baseline=base_holdout)
            stable_positive = (
                train_summary["count"] >= 20
                and holdout_summary["count"] >= min_holdout_rows
                and train_summary["avg_net_eod_r_lift"] > 0.0
                and holdout_summary["avg_net_eod_r_lift"] > 0.0
                and train_summary["avg_net_eod_r"] > 0.0
                and holdout_summary["avg_net_eod_r"] > 0.0
            )
            rule_rows.append(
                {
                    "rule": name,
                    "role": role,
                    "description": description,
                    "stable_positive": bool(stable_positive),
                    "train": train_summary,
                    "holdout": holdout_summary,
                }
            )
    return {
        "version": GAP_RETENTION_REPORT_VERSION,
        "usage_contract": "research_only_candidate_path_feature_discovery_not_live_routing",
        "threshold_source": "train_only",
        "thresholds": thresholds,
        "summary": {
            "train": _role_summaries(train),
            "holdout": _role_summaries(holdout),
        },
        "rules": rule_rows,
        "stable_positive_rules": [row for row in rule_rows if row["stable_positive"]],
    }


def compact_report_markdown(report: Mapping[str, Any], *, title: str = "KALCB Gap-Retention Path-Quality Report") -> str:
    lines = [
        f"# {title}",
        "",
        f"Version: `{report.get('version')}`",
        f"Usage: `{report.get('usage_contract')}`",
        "Thresholds are trained on the training window only, then applied unchanged to holdout.",
        "",
        "## Role Summary",
    ]
    for window in ("train", "holdout"):
        lines.append(f"### {window}")
        for row in list((report.get("summary") or {}).get(window, ()) or ()):
            lines.append(
                "- {role}: n={count:.0f}, avgR={avg_net_eod_r:.2f}, win={win_share:.1%}, "
                "MFE={avg_mfe_r:.2f}, MAE={avg_mae_r:.2f}, capture={avg_mfe_capture:.1%}".format(**row)
            )
    lines.extend(["", "## Stable Positive Rules"])
    stable = list(report.get("stable_positive_rules") or ())
    if not stable:
        lines.append("No rule was train/holdout positive with enough holdout observations.")
    for row in stable:
        train = row["train"]
        holdout = row["holdout"]
        lines.append(
            "- {rule} [{role}]: train n={tn:.0f}, lift={tlift:+.2f}R, avg={tavg:.2f}R; "
            "holdout n={hn:.0f}, lift={hlift:+.2f}R, avg={havg:.2f}R".format(
                rule=row["rule"],
                role=row["role"],
                tn=train["count"],
                tlift=train["avg_net_eod_r_lift"],
                tavg=train["avg_net_eod_r"],
                hn=holdout["count"],
                hlift=holdout["avg_net_eod_r_lift"],
                havg=holdout["avg_net_eod_r"],
            )
        )
    lines.extend(["", "## Top Rule Table"])
    ranked = sorted(
        list(report.get("rules") or ()),
        key=lambda row: (
            not bool(row.get("stable_positive")),
            -float(row["train"].get("avg_net_eod_r_lift", 0.0) or 0.0),
            -float(row["holdout"].get("avg_net_eod_r_lift", 0.0) or 0.0),
        ),
    )[:18]
    for row in ranked:
        train = row["train"]
        holdout = row["holdout"]
        lines.append(
            "- {rule} [{role}]: train {tn:.0f}/{ts:.1%}, lift={tlift:+.2f}R; "
            "holdout {hn:.0f}/{hs:.1%}, lift={hlift:+.2f}R; stable={stable}".format(
                rule=row["rule"],
                role=row["role"],
                tn=train["count"],
                ts=train["selected_share"],
                tlift=train["avg_net_eod_r_lift"],
                hn=holdout["count"],
                hs=holdout["selected_share"],
                hlift=holdout["avg_net_eod_r_lift"],
                stable=row["stable_positive"],
            )
        )
    return "\n".join(lines) + "\n"


def _candidate_path_row(
    *,
    window: str,
    day: date,
    candidate: Any,
    ctx: Any,
    bars: tuple[MarketBar, ...],
    entry_index: int,
    stop_pct: float,
    round_trip_cost_pct: float,
    horizons: tuple[int, ...],
) -> CandidatePathRow | None:
    post = bars[entry_index:]
    if not post:
        return None
    entry_bar = post[0]
    entry = max(float(entry_bar.open), 1e-9)
    risk = max(entry * max(float(stop_pct), 1e-6), 1.0)
    high = max(float(bar.high) for bar in post)
    low = min(float(bar.low) for bar in post)
    exit_price = float(post[-1].close)
    gross_r = (exit_price - entry) / risk
    cost_r = float(round_trip_cost_pct or 0.0) * entry / risk
    net_r = gross_r - cost_r
    mfe_r = max(0.0, (high - entry) / risk)
    mae_r = (low - entry) / risk
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    features = _base_features(candidate, ctx, metadata)
    for horizon in horizons:
        features.update(_horizon_features(post, entry, risk, int(horizon)))
    labels = {
        "gross_eod_r": float(gross_r),
        "net_eod_r": float(net_r),
        "net_eod_pct": float(exit_price / entry - 1.0 - float(round_trip_cost_pct or 0.0)),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "mfe_capture": float(net_r / mfe_r) if mfe_r > 0 else 0.0,
        "giveback_r": float(max(0.0, mfe_r - net_r)),
        "loser": 1.0 if net_r < 0.0 else 0.0,
        "mae_le_neg_1": 1.0 if mae_r <= -1.0 else 0.0,
    }
    return CandidatePathRow(
        window=window,
        trade_date=day,
        symbol=str(candidate.symbol),
        frontier_role=str(metadata.get("frontier_role") or "initial_active"),
        frontier_rank=int(metadata.get("frontier_rank") or 0),
        candidate_rank=int(metadata.get("candidate_rank") or 0),
        features=features,
        labels=labels,
    )


def _base_features(candidate: Any, ctx: Any, metadata: Mapping[str, Any]) -> dict[str, float]:
    gap = float(getattr(ctx, "gap", metadata.get("first30_gap", 0.0)) or 0.0)
    rel_volume = float(getattr(ctx, "rel_volume", metadata.get("first30_rel_volume", 0.0)) or 0.0)
    low_vs_prev = float(getattr(ctx, "low_vs_prev_close", metadata.get("first30_low_vs_prev_close", 0.0)) or 0.0)
    rel_volume_log = math.log1p(max(rel_volume, 0.0))
    gap_retention_ratio = low_vs_prev / max(abs(gap), 1e-6) if gap > 0.0 else 0.0
    features = {
        "candidate_rank": float(metadata.get("candidate_rank") or 0),
        "frontier_rank": float(metadata.get("frontier_rank") or 0),
        "frontier_selection_score": float(metadata.get("frontier_selection_score") or 0.0),
        "first30_score": float(metadata.get("first30_score") or 0.0),
        "first30_gap": gap,
        "first30_ret": float(getattr(ctx, "first30_ret", metadata.get("first30_ret", 0.0)) or 0.0),
        "first30_vwap_ret": float(getattr(ctx, "vwap_ret", metadata.get("first30_vwap_ret", 0.0)) or 0.0),
        "first30_rel_volume": rel_volume,
        "first30_cpr": float(getattr(ctx, "close_location", metadata.get("first30_close_location", 0.0)) or 0.0),
        "first30_open_drawdown": float(getattr(ctx, "open_drawdown", metadata.get("first30_open_drawdown", 0.0)) or 0.0),
        "first30_low_vs_prev_close": low_vs_prev,
        "first30_range_atr": float(getattr(ctx, "range_atr", metadata.get("first30_range_atr", 0.0)) or 0.0),
        "first30_gap_retention_ratio": float(gap_retention_ratio),
        "first30_gap_relvol": float(gap * rel_volume_log),
        "first30_low_vs_prev_relvol": float(low_vs_prev * rel_volume_log),
        "daily_atr": float(getattr(candidate, "daily_atr", 0.0) or 0.0),
        "flow_score": float(getattr(candidate, "flow_score", 0.0) or 0.0),
        "accumulation_score": float(getattr(candidate, "accumulation_score", 0.0) or 0.0),
    }
    for key in JOINT_CONTEXT_FEATURES:
        value = _coerce_float(metadata.get(key))
        if value is not None:
            features[key] = value
    return features


def _horizon_features(post_bars: tuple[MarketBar, ...], entry: float, risk: float, horizon: int) -> dict[str, float]:
    window = post_bars[: max(1, horizon)]
    if not window:
        return {}
    last = window[-1]
    high = max(float(bar.high) for bar in window)
    low = min(float(bar.low) for bar in window)
    current_r = (float(last.close) - entry) / risk
    mfe_r = max(0.0, (high - entry) / risk)
    mae_r = (low - entry) / risk
    prefix = f"h{horizon}"
    return {
        f"{prefix}_current_r": float(current_r),
        f"{prefix}_mfe_r": float(mfe_r),
        f"{prefix}_mae_r": float(mae_r),
        f"{prefix}_giveback_r": float(max(0.0, mfe_r - current_r)),
        f"{prefix}_close_location": float((float(last.close) - float(last.low)) / max(float(last.high) - float(last.low), 1e-9)),
        f"{prefix}_recent_return": float(float(last.close) / entry - 1.0),
        f"{prefix}_down_streak": float(_down_streak(window)),
        f"{prefix}_below_entry_streak": float(_below_entry_streak(window, entry)),
    }


def _first30_entry_index(bars: tuple[MarketBar, ...]) -> int | None:
    for index, bar in enumerate(bars):
        if bar.timestamp.astimezone(KST).time() >= FIRST30_END:
            return index
    return None


def _train_thresholds(rows: list[CandidatePathRow]) -> dict[str, float]:
    features = (
        "first30_gap",
        "first30_low_vs_prev_close",
        "first30_gap_retention_ratio",
        "first30_gap_relvol",
        "first30_low_vs_prev_relvol",
        "first30_rel_volume",
        "first30_cpr",
        "h3_current_r",
        "h6_current_r",
        "h3_mae_r",
        "h6_mae_r",
    )
    out: dict[str, float] = {}
    for feature in features:
        values = [row.features[feature] for row in rows if feature in row.features]
        for quantile in (0.25, 0.50, 0.60, 0.70, 0.75, 0.80):
            out[f"{feature}_q{int(quantile * 100)}"] = _quantile(values, quantile)
    return out


def _rule_specs(thresholds: Mapping[str, float]) -> list[tuple[str, str, Callable[[CandidatePathRow], bool]]]:
    t = thresholds
    return [
        (
            "gap_q75",
            "first30 gap is in the top training quartile",
            lambda row: row.features.get("first30_gap", 0.0) >= t.get("first30_gap_q75", 0.0),
        ),
        (
            "gap_retention_q75",
            "first30 low-vs-prev-close is in the top training quartile",
            lambda row: row.features.get("first30_low_vs_prev_close", 0.0) >= t.get("first30_low_vs_prev_close_q75", 0.0),
        ),
        (
            "gap_relvol_q75",
            "gap times log rel-volume is in the top training quartile",
            lambda row: row.features.get("first30_gap_relvol", 0.0) >= t.get("first30_gap_relvol_q75", 0.0),
        ),
        (
            "gap_and_relvol_q60",
            "gap and first30 rel-volume are both above their training q60",
            lambda row: row.features.get("first30_gap", 0.0) >= t.get("first30_gap_q60", 0.0)
            and row.features.get("first30_rel_volume", 0.0) >= t.get("first30_rel_volume_q60", 0.0),
        ),
        (
            "gap_retention_and_relvol_q60",
            "low-vs-prev-close and first30 rel-volume are both above training q60",
            lambda row: row.features.get("first30_low_vs_prev_close", 0.0) >= t.get("first30_low_vs_prev_close_q60", 0.0)
            and row.features.get("first30_rel_volume", 0.0) >= t.get("first30_rel_volume_q60", 0.0),
        ),
        (
            "gap_q60_rank8",
            "gap above q60 and frontier rank at most 8",
            lambda row: row.features.get("first30_gap", 0.0) >= t.get("first30_gap_q60", 0.0) and row.frontier_rank <= 8,
        ),
        (
            "gap_retention_q60_rank8",
            "low-vs-prev-close above q60 and frontier rank at most 8",
            lambda row: row.features.get("first30_low_vs_prev_close", 0.0) >= t.get("first30_low_vs_prev_close_q60", 0.0) and row.frontier_rank <= 8,
        ),
        (
            "gap_relvol_q70_rank12",
            "gap-relvol above q70 and frontier rank at most 12",
            lambda row: row.features.get("first30_gap_relvol", 0.0) >= t.get("first30_gap_relvol_q70", 0.0) and row.frontier_rank <= 12,
        ),
        (
            "gap_path_h3_q60",
            "gap above q60 and first three post-entry bars positive enough",
            lambda row: row.features.get("first30_gap", 0.0) >= t.get("first30_gap_q60", 0.0)
            and row.features.get("h3_current_r", 0.0) >= t.get("h3_current_r_q60", 0.0),
        ),
        (
            "retention_path_h3_q60",
            "low-vs-prev-close above q60 and first three post-entry bars positive enough",
            lambda row: row.features.get("first30_low_vs_prev_close", 0.0) >= t.get("first30_low_vs_prev_close_q60", 0.0)
            and row.features.get("h3_current_r", 0.0) >= t.get("h3_current_r_q60", 0.0),
        ),
        (
            "gap_path_h6_q60",
            "gap above q60 and six-bar current R above q60",
            lambda row: row.features.get("first30_gap", 0.0) >= t.get("first30_gap_q60", 0.0)
            and row.features.get("h6_current_r", 0.0) >= t.get("h6_current_r_q60", 0.0),
        ),
        (
            "bad_gap_avoid_q25",
            "gap is in the bottom training quartile",
            lambda row: row.features.get("first30_gap", 0.0) <= t.get("first30_gap_q25", 0.0),
        ),
        (
            "bad_retention_avoid_q25",
            "low-vs-prev-close is in the bottom training quartile",
            lambda row: row.features.get("first30_low_vs_prev_close", 0.0) <= t.get("first30_low_vs_prev_close_q25", 0.0),
        ),
    ]


def _summarize_rows(rows: list[CandidatePathRow], baseline: Mapping[str, float] | None = None) -> dict[str, float]:
    base_avg = float((baseline or {}).get("avg_net_eod_r", 0.0) or 0.0)
    count = float(len(rows))
    avg_r = _avg(row.labels["net_eod_r"] for row in rows)
    return {
        "count": count,
        "selected_share": count / max(float((baseline or {}).get("count", count) or count), 1.0),
        "avg_net_eod_r": avg_r,
        "median_net_eod_r": _median(row.labels["net_eod_r"] for row in rows),
        "avg_net_eod_r_lift": avg_r - base_avg,
        "win_share": _share(row.labels["net_eod_r"] > 0.0 for row in rows),
        "avg_mfe_r": _avg(row.labels["mfe_r"] for row in rows),
        "avg_mae_r": _avg(row.labels["mae_r"] for row in rows),
        "mae_le_neg_1_share": _share(row.labels["mae_le_neg_1"] > 0.0 for row in rows),
        "avg_mfe_capture": _avg(row.labels["mfe_capture"] for row in rows),
        "avg_giveback_r": _avg(row.labels["giveback_r"] for row in rows),
    }


def _role_summaries(rows: list[CandidatePathRow]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for role in ("all", "initial_active", "frontier_shadow"):
        summary = _summarize_rows(_filter_role(rows, role))
        out.append({"role": role, **summary})
    return out


def _filter_role(rows: list[CandidatePathRow], role: str) -> list[CandidatePathRow]:
    if role == "all":
        return rows
    return [row for row in rows if row.frontier_role == role]


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


def _below_entry_streak(window: tuple[MarketBar, ...], entry: float) -> int:
    streak = 0
    for bar in window:
        if float(bar.close) < entry:
            streak += 1
        else:
            streak = 0
    return streak


def _avg(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _median(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(median(vals)) if vals else 0.0


def _share(values: Iterable[bool]) -> float:
    vals = [bool(value) for value in values]
    return sum(1 for value in vals if value) / len(vals) if vals else 0.0


def _quantile(values: Iterable[float], quantile: float) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    index = (len(vals) - 1) * min(max(float(quantile), 0.0), 1.0)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return vals[int(index)]
    weight = index - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


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
