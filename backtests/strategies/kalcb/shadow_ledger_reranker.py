from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

from backtests.auto.shared.phase_state import _utc_now_iso


SHADOW_LEDGER_RERANKER_VERSION = "kalcb-shadow-ledger-same-day-reranker-v1"
SHADOW_LEDGER_RERANKER_USAGE_CONTRACT = (
    "research_only_ex_post_shadow_labels_not_live_entry_or_promotion_rule"
)

ROUTE_FAMILIES = (
    "first30_open",
    "pullback_acceptance",
    "avwap_reclaim",
    "or_high_reclaim",
)
DELAYED_ROUTE_FAMILIES = (
    "pullback_acceptance",
    "avwap_reclaim",
    "or_high_reclaim",
)

FIRST30_FEATURE_KEYS = (
    "first30_ret",
    "first30_vwap_ret",
    "first30_gap",
    "first30_rel_volume",
    "first30_signal_bar_cpr",
    "first30_range_close_location",
    "first30_low_vs_prev_close",
    "first30_range_atr",
    "first30_gap_retention_ratio",
    "first30_gap_relvol",
    "first30_low_vs_prev_relvol",
    "first30_quality_pct",
)

DAILY_SECTOR_FEATURE_KEYS = (
    "daily_return_5d",
    "daily_return_20d",
    "daily_return_60d",
    "daily_volume_ratio_20d",
    "daily_acceleration_5v20",
    "daily_momentum_pct",
    "sector_daily_score_pct",
    "sector_daily_participation",
    "sector_daily_breadth_20d",
    "sector_daily_ret_5d",
    "sector_daily_ret_20d",
    "sector_daily_ret_60d",
    "sector_intraday_score_pct",
    "sector_intraday_ret",
    "sector_intraday_breadth",
    "sector_intraday_participation",
    "sector_intraday_rel_volume",
    "stock_sector_daily_ret5_spread",
    "stock_sector_daily_ret20_spread",
    "daily_sector_alignment_pct",
    "first30_sector_ret_spread",
    "first30_sector_relvol_ratio",
    "first30_sector_leadership_pct",
    "first30_gap_relvol_sector_breadth",
    "first30_gap_retention_sector_breadth",
    "continuation_joint_quality_pct",
)

PATH_FEATURE_KEYS = (
    "h1_current_r",
    "h1_mfe_r",
    "h1_mae_r",
    "h3_current_r",
    "h3_mfe_r",
    "h3_mae_r",
    "h6_current_r",
    "h6_mfe_r",
    "h6_mae_r",
    "h12_current_r",
    "h12_mfe_r",
    "h12_mae_r",
)

LEDGER_CONTEXT_FEATURE_KEYS = (
    *FIRST30_FEATURE_KEYS,
    *DAILY_SECTOR_FEATURE_KEYS,
    *PATH_FEATURE_KEYS,
    "frontier_rank",
    "candidate_rank",
    "frontier_selection_score",
    "flow_score",
    "accumulation_score",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return rows
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def aggregate_route_shadow_rows(
    rows: Iterable[dict[str, Any]],
    *,
    route_family: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source_row in rows:
        row = dict(source_row)
        day = str(row.get("entry_date") or row.get("trade_date") or str(row.get("entry_time") or "")[:10])[:10]
        symbol = str(row.get("symbol") or "")
        if not day or not symbol:
            continue
        grouped[(day, symbol)].append(row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        r_values = [_num(item.get("r")) for item in items]
        mfe_values = [_num(item.get("mfe_r")) for item in items]
        mae_values = [_num(item.get("mae_r")) for item in items]
        giveback_values = [_num(item.get("giveback_r")) for item in items]
        captures = [
            max(0.0, min(_num(item.get("r")) / max(_num(item.get("mfe_r")), 1e-9), 2.0))
            for item in items
            if _num(item.get("mfe_r")) > 0.0
        ]
        exits = Counter(str(item.get("exit_reason") or "unknown") for item in items)
        modes = Counter(str(item.get("entry_route_mode") or item.get("entry_type") or route_family or "unknown") for item in items)
        out[key] = {
            "route_family": str(route_family or modes.most_common(1)[0][0]),
            "shadow_trade_count": len(items),
            "shadow_total_r": float(sum(r_values)),
            "shadow_avg_r": float(sum(r_values) / len(r_values)) if r_values else 0.0,
            "shadow_max_mfe_r": max(mfe_values, default=0.0),
            "shadow_min_mae_r": min(mae_values, default=0.0),
            "shadow_avg_mfe_capture": float(fmean(captures)) if captures else 0.0,
            "shadow_avg_giveback_r": float(fmean(giveback_values)) if giveback_values else 0.0,
            "shadow_exit_reasons": dict(exits),
            "shadow_entry_route_modes": dict(modes),
        }
        for field in PATH_FEATURE_KEYS:
            values = [_num(item.get(field)) for item in items if _optional_num(item.get(field)) is not None]
            if values:
                out[key][f"shadow_{field}"] = float(fmean(values))
    return out


def prepare_shadow_ledger_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_per_sector: int = 8,
) -> list[dict[str, Any]]:
    prepared = [dict(row) for row in rows]
    _attach_day_candidate_context(prepared, max_per_sector=max_per_sector)
    for row in prepared:
        _attach_best_route_and_labels(row)
    prepared.sort(
        key=lambda item: (
            str(item.get("trade_date") or ""),
            int(_num(item.get("candidate_rank"))),
            str(item.get("symbol") or ""),
        )
    )
    return prepared


def build_same_day_reranker_artifacts(
    train_rows: Iterable[dict[str, Any]],
    holdout_rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    max_per_sector: int = 8,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = prepare_shadow_ledger_rows(train_rows, max_per_sector=max_per_sector)
    holdout = prepare_shadow_ledger_rows(holdout_rows, max_per_sector=max_per_sector)
    profile = fit_reranker_profile(train)
    scored_train = score_shadow_ledger_rows(train, profile)
    scored_holdout = score_shadow_ledger_rows(holdout, profile)
    summary = summarize_reranker(
        scored_train,
        scored_holdout,
        profile=profile,
        max_per_sector=max_per_sector,
    )

    train_path = out / "shadow_same_day_reranker_train.jsonl"
    holdout_path = out / "shadow_same_day_reranker_holdout.jsonl"
    summary_path = out / "shadow_same_day_reranker_summary.json"
    report_path = out / "shadow_same_day_reranker_report.md"
    write_jsonl(train_path, scored_train)
    write_jsonl(holdout_path, scored_holdout)
    summary["artifact_paths"] = {
        "train_jsonl": str(train_path),
        "holdout_jsonl": str(holdout_path),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report_path.write_text(render_reranker_report(summary), encoding="utf-8")
    return summary


def fit_reranker_profile(train_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in train_rows]
    values_by_field = {
        key: [_num(row.get(key)) for row in rows if _optional_num(row.get(key)) is not None]
        for key in (
            "same_day_replacement_value_r",
            "marginal_slot_replacement_value_r",
            "best_route_shadow_total_r",
            "best_route_shadow_max_mfe_r",
            "first30_rel_volume",
            "first30_signal_bar_cpr",
            "sector_daily_score_pct",
            "sector_intraday_score_pct",
        )
    }
    sector_stats: dict[str, dict[str, Any]] = {}
    by_sector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sector[str(row.get("sector") or "UNKNOWN")].append(row)
    for sector, sector_rows in by_sector.items():
        outcome_rows = [row for row in sector_rows if row.get("route_outcome_available")]
        sector_stats[sector] = {
            "n": len(sector_rows),
            "outcome_n": len(outcome_rows),
            "avg_replacement_value_r": _avg(row.get("same_day_replacement_value_r") for row in outcome_rows),
            "avg_shadow_total_r": _avg(row.get("best_route_shadow_total_r") for row in outcome_rows),
            "avg_mfe_r": _avg(row.get("best_route_shadow_max_mfe_r") for row in outcome_rows),
            "avg_mae_r": _avg(row.get("best_route_shadow_min_mae_r") for row in outcome_rows),
            "avg_mfe_capture": _avg(row.get("best_route_shadow_avg_mfe_capture") for row in outcome_rows),
            "avg_giveback_r": _avg(row.get("best_route_shadow_avg_giveback_r") for row in outcome_rows),
            "positive_replacement_share": _share(
                _num(row.get("same_day_replacement_value_r")) > 0.0 for row in outcome_rows
            ),
        }
    return {
        "version": SHADOW_LEDGER_RERANKER_VERSION,
        "usage_contract": SHADOW_LEDGER_RERANKER_USAGE_CONTRACT,
        "source_window": "train",
        "created_at": _utc_now_iso(),
        "row_count": len(rows),
        "field_centers": {
            key: {
                "median": _median(values),
                "iqr": max(_quantile(values, 0.75) - _quantile(values, 0.25), 1e-6),
            }
            for key, values in values_by_field.items()
        },
        "sector_priors": sector_stats,
    }


def score_shadow_ledger_rows(
    rows: Iterable[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    scored = []
    sector_priors = dict(profile.get("sector_priors") or {})
    for source_row in rows:
        row = dict(source_row)
        components = _score_components(row, sector_priors)
        penalties = _score_penalties(row, sector_priors)
        score = sum(components.values()) - sum(penalties.values())
        row["reranker_version"] = SHADOW_LEDGER_RERANKER_VERSION
        row["reranker_usage_contract"] = SHADOW_LEDGER_RERANKER_USAGE_CONTRACT
        row["reranker_score"] = float(score)
        row["reranker_components"] = components
        row["reranker_penalties"] = penalties
        row["reranker_score_uses_ex_post_labels"] = True
        scored.append(row)
    _attach_ranks(scored)
    return scored


def summarize_reranker(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    max_per_sector: int,
) -> dict[str, Any]:
    summary = {
        "strategy": "kalcb",
        "reranker_version": SHADOW_LEDGER_RERANKER_VERSION,
        "usage_contract": SHADOW_LEDGER_RERANKER_USAGE_CONTRACT,
        "created_at": _utc_now_iso(),
        "max_per_sector": int(max_per_sector),
        "profile": profile,
        "train": _window_summary(train_rows),
        "holdout": _window_summary(holdout_rows),
        "feature_coverage": {
            "train": feature_coverage(train_rows, LEDGER_CONTEXT_FEATURE_KEYS),
            "holdout": feature_coverage(holdout_rows, LEDGER_CONTEXT_FEATURE_KEYS),
        },
        "route_family_outcome_coverage": {
            "train": route_family_outcome_coverage(train_rows),
            "holdout": route_family_outcome_coverage(holdout_rows),
        },
        "top_candidates": {
            "train": top_rows(train_rows, limit=20),
            "holdout": top_rows(holdout_rows, limit=20),
        },
        "missed_best_candidate_days": {
            "train": missed_best_candidate_days(train_rows),
            "holdout": missed_best_candidate_days(holdout_rows),
        },
        "sector_diagnostics": {
            "train": sector_diagnostics(train_rows),
            "holdout": sector_diagnostics(holdout_rows),
        },
        "holdout_sector_validation": holdout_sector_validation(train_rows, holdout_rows),
        "root_cause_layer_attribution": root_cause_layer_attribution(train_rows, holdout_rows),
    }
    return summary


def render_reranker_report(summary: dict[str, Any]) -> str:
    train = dict(summary.get("train") or {})
    holdout = dict(summary.get("holdout") or {})
    root = dict(summary.get("root_cause_layer_attribution") or {})
    lines = [
        "# KALCB Shadow-Ledger Same-Day Reranker",
        "",
        f"Version: `{summary.get('reranker_version')}`",
        f"Usage contract: `{summary.get('usage_contract')}`",
        "",
        "## Summary",
        "",
        f"- Train: rows={train.get('row_count', 0)}, scored outcomes={train.get('route_outcome_count', 0)}, positive replacement days={train.get('positive_top_replacement_days', 0)}, top replacement total={_signed(train.get('top_ranked_same_day_replacement_total_r'))}R",
        f"- Holdout: rows={holdout.get('row_count', 0)}, scored outcomes={holdout.get('route_outcome_count', 0)}, positive replacement days={holdout.get('positive_top_replacement_days', 0)}, top replacement total={_signed(holdout.get('top_ranked_same_day_replacement_total_r'))}R",
        f"- Main bottleneck: {root.get('primary_bottleneck', 'review')}",
        "",
        "## Feature Coverage",
        "",
    ]
    feature_keys = (
        "first30_rel_volume",
        "first30_signal_bar_cpr",
        "frontier_rank",
        "sector_daily_score_pct",
        "sector_daily_participation",
        "sector_daily_breadth_20d",
        "stock_sector_daily_ret5_spread",
        "stock_sector_daily_ret20_spread",
        "sector_intraday_score_pct",
        "sector_intraday_ret",
        "sector_intraday_breadth",
        "sector_intraday_participation",
        "daily_sector_alignment_pct",
        "first30_sector_leadership_pct",
        "continuation_joint_quality_pct",
    )
    coverage = summary.get("feature_coverage") or {}
    for window in ("train", "holdout"):
        window_coverage = dict(coverage.get(window) or {})
        present = []
        for key in feature_keys:
            row = dict(window_coverage.get(key) or {})
            present.append(f"`{key}`={100.0 * _num(row.get('coverage')):.1f}%")
        lines.append(f"- {window}: " + "; ".join(present))
    lines.extend(
        [
            "",
            "## Selected-Vs-Shadow Candidate Value",
            "",
            f"- Train top-ranked shadow replacement total: {_signed(train.get('top_ranked_same_day_replacement_total_r'))}R; marginal-slot replacement total: {_signed(train.get('top_ranked_marginal_slot_replacement_total_r'))}R; avg top MFE/MAE/capture: {_num_label(train.get('avg_top_ranked_mfe_r'))}R / {_num_label(train.get('avg_top_ranked_mae_r'))}R / {100.0 * _num(train.get('avg_top_ranked_mfe_capture')):.1f}%.",
            f"- Holdout top-ranked shadow replacement total: {_signed(holdout.get('top_ranked_same_day_replacement_total_r'))}R; marginal-slot replacement total: {_signed(holdout.get('top_ranked_marginal_slot_replacement_total_r'))}R; avg top MFE/MAE/capture: {_num_label(holdout.get('avg_top_ranked_mfe_r'))}R / {_num_label(holdout.get('avg_top_ranked_mae_r'))}R / {100.0 * _num(holdout.get('avg_top_ranked_mfe_capture')):.1f}%.",
            "",
        ]
    )
    lines.extend(
        [
            "## Route-Family Outcome Coverage",
            "",
        ]
    )
    route_cov = summary.get("route_family_outcome_coverage") or {}
    for window in ("train", "holdout"):
        lines.append(f"### {window.title()}")
        for family, row in sorted(dict(route_cov.get(window) or {}).items()):
            lines.append(
                f"- `{family}`: candidates={row.get('candidate_count', 0)}, trades={row.get('shadow_trade_count', 0)}, totalR={_signed(row.get('shadow_total_r'))}, avgMFE={_num_label(row.get('avg_mfe_r'))}R, avgMAE={_num_label(row.get('avg_mae_r'))}R, avgCapture={100.0 * _num(row.get('avg_mfe_capture')):.1f}%"
            )
        lines.append("")
    lines.extend(["## Top Same-Day Replacement Candidates", ""])
    for window in ("train", "holdout"):
        lines.append(f"### {window.title()}")
        for row in list((summary.get("top_candidates") or {}).get(window) or [])[:10]:
            lines.append(
                f"- {row.get('trade_date')} `{row.get('symbol')}` {row.get('sector')}: score={_num_label(row.get('reranker_score'))}, route={row.get('best_route_family')}, replacement={_signed(row.get('same_day_replacement_value_r'))}R, MFE={_num_label(row.get('best_route_shadow_max_mfe_r'))}R, MAE={_num_label(row.get('best_route_shadow_min_mae_r'))}R"
            )
        lines.append("")
    lines.extend(["## Missed Best-Candidate Days", ""])
    for window in ("train", "holdout"):
        missed = list((summary.get("missed_best_candidate_days") or {}).get(window) or [])
        lines.append(f"- {window}: {len(missed)} candidate-days where the top reranked shadow candidate had positive same-day replacement value.")
    lines.extend(["", "## Sector Crowding And Leading-Sector Cluster Behavior", ""])
    sector = summary.get("sector_diagnostics") or {}
    for window in ("train", "holdout"):
        window_sector = dict(sector.get(window) or {})
        lines.append(f"### {window.title()}")
        for row in list(window_sector.get("top_positive_sectors") or [])[:5]:
            lines.append(
                f"- Positive `{row.get('sector')}`: candidates={row.get('candidate_count', 0)}, outcomes={row.get('outcome_count', 0)}, avgReplacement={_signed(row.get('avg_replacement_value_r'))}R, leadingCluster={100.0 * _num(row.get('leading_cluster_share')):.1f}%, avgDayShare={100.0 * _num(row.get('avg_sector_day_candidate_share')):.1f}%, candidateCrowding={_num_label(row.get('avg_candidate_sector_crowding_pressure'))}, maxSectorPressure={_num_label(row.get('avg_max_per_sector_pressure'))}."
            )
        for row in list(window_sector.get("top_negative_sectors") or [])[:3]:
            lines.append(
                f"- Negative `{row.get('sector')}`: candidates={row.get('candidate_count', 0)}, outcomes={row.get('outcome_count', 0)}, avgReplacement={_signed(row.get('avg_replacement_value_r'))}R, leadingCluster={100.0 * _num(row.get('leading_cluster_share')):.1f}%, avgDayShare={100.0 * _num(row.get('avg_sector_day_candidate_share')):.1f}%, candidateCrowding={_num_label(row.get('avg_candidate_sector_crowding_pressure'))}, maxSectorPressure={_num_label(row.get('avg_max_per_sector_pressure'))}."
            )
        lines.append("")
    holdout_validation = dict(summary.get("holdout_sector_validation") or {})
    veto_rows = list(holdout_validation.get("veto_sectors") or [])
    lines.extend(["## Holdout Sector Veto Diagnostics", ""])
    if veto_rows:
        for row in veto_rows[:8]:
            lines.append(
                f"- `{row.get('sector')}`: holdoutOutcomes={row.get('holdout_outcome_count', 0)}, holdoutAvgReplacement={_signed(row.get('holdout_avg_replacement_value_r'))}R, holdoutAvgMAE={_num_label(row.get('holdout_avg_mae_r'))}R, trainAvgReplacement={_signed(row.get('train_avg_replacement_value_r'))}R, reason={row.get('veto_reason')}."
            )
    else:
        lines.append("- No holdout sector vetoes with enough route outcomes.")
    lines.append("")
    lines.extend(
        [
            "## Train/Holdout Stability",
            "",
            f"- Route-outcome density: train={100.0 * _num(train.get('route_outcome_count')) / max(_num(train.get('row_count')), 1.0):.1f}%, holdout={100.0 * _num(holdout.get('route_outcome_count')) / max(_num(holdout.get('row_count')), 1.0):.1f}%.",
            f"- Positive top replacement days: train={train.get('positive_top_replacement_days', 0)}/{train.get('top_ranked_day_count', 0)}, holdout={holdout.get('positive_top_replacement_days', 0)}/{holdout.get('top_ranked_day_count', 0)}.",
            f"- Holdout note: {root.get('holdout_validation', 'review')}",
            "",
            "## Layer Attribution",
            "",
        ]
    )
    for key, value in root.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def feature_coverage(rows: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    total = len(rows)
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        non_null = sum(1 for row in rows if _optional_num(row.get(key)) is not None or row.get(key) not in (None, ""))
        out[str(key)] = {
            "non_null": non_null,
            "coverage": non_null / total if total else 0.0,
        }
    return out


def route_family_outcome_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for family in ROUTE_FAMILIES:
        outcome_rows = [row for row in rows if family in dict(row.get("route_outcomes") or {})]
        outcomes = [dict(row.get("route_outcomes") or {}).get(family) or {} for row in outcome_rows]
        out[family] = {
            "candidate_count": len(outcome_rows),
            "shadow_trade_count": sum(int(_num(item.get("shadow_trade_count"))) for item in outcomes),
            "shadow_total_r": float(sum(_num(item.get("shadow_total_r")) for item in outcomes)),
            "avg_mfe_r": _avg(item.get("shadow_max_mfe_r") for item in outcomes),
            "avg_mae_r": _avg(item.get("shadow_min_mae_r") for item in outcomes),
            "avg_mfe_capture": _avg(item.get("shadow_avg_mfe_capture") for item in outcomes),
        }
    return out


def top_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    keys = (
        "window",
        "trade_date",
        "symbol",
        "sector",
        "frontier_role",
        "frontier_rank",
        "candidate_rank",
        "best_route_family",
        "best_route_shadow_total_r",
        "best_route_shadow_max_mfe_r",
        "best_route_shadow_min_mae_r",
        "same_day_actual_total_r",
        "same_day_replacement_value_r",
        "marginal_slot_replacement_value_r",
        "reranker_score",
        "reranker_rank_in_day",
        "leading_sector_cluster",
        "sector_day_candidate_count",
        "sector_day_candidate_share",
        "candidate_sector_crowding_pressure",
        "max_per_sector_pressure",
    )
    outcome_rows = [row for row in rows if row.get("route_outcome_available")]
    ranked = sorted(outcome_rows, key=lambda row: _num(row.get("reranker_score")), reverse=True)
    return [{key: row.get(key) for key in keys if key in row} for row in ranked[:limit]]


def missed_best_candidate_days(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[str(row.get("trade_date") or "")].append(row)
    missed: list[dict[str, Any]] = []
    for day, day_rows in by_day.items():
        outcome_rows = [row for row in day_rows if row.get("route_outcome_available")]
        if not outcome_rows:
            continue
        best = max(outcome_rows, key=lambda row: _num(row.get("same_day_replacement_value_r")))
        if _num(best.get("same_day_replacement_value_r")) <= 0.0:
            continue
        if bool(best.get("current_realized")):
            continue
        missed.append(
            {
                "trade_date": day,
                "symbol": best.get("symbol"),
                "sector": best.get("sector"),
                "best_route_family": best.get("best_route_family"),
                "same_day_replacement_value_r": best.get("same_day_replacement_value_r"),
                "marginal_slot_replacement_value_r": best.get("marginal_slot_replacement_value_r"),
                "reranker_score": best.get("reranker_score"),
                "same_day_actual_total_r": best.get("same_day_actual_total_r"),
            }
        )
    missed.sort(key=lambda row: _num(row.get("same_day_replacement_value_r")), reverse=True)
    return missed[:20]


def sector_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sector[str(row.get("sector") or "UNKNOWN")].append(row)
    sector_rows = []
    for sector, items in by_sector.items():
        outcome = [row for row in items if row.get("route_outcome_available")]
        sector_rows.append(
            {
                "sector": sector,
                "candidate_count": len(items),
                "outcome_count": len(outcome),
                "avg_replacement_value_r": _avg(row.get("same_day_replacement_value_r") for row in outcome),
                "avg_shadow_total_r": _avg(row.get("best_route_shadow_total_r") for row in outcome),
                "avg_sector_daily_score_pct": _avg(row.get("sector_daily_score_pct") for row in items),
                "avg_sector_intraday_score_pct": _avg(row.get("sector_intraday_score_pct") for row in items),
                "leading_cluster_share": _share(bool(row.get("leading_sector_cluster")) for row in items),
                "avg_sector_day_candidate_share": _avg(row.get("sector_day_candidate_share") for row in items),
                "avg_candidate_sector_crowding_pressure": _avg(row.get("candidate_sector_crowding_pressure") for row in items),
                "avg_max_per_sector_pressure": _avg(row.get("max_per_sector_pressure") for row in items),
            }
        )
    sector_rows.sort(key=lambda row: (_num(row.get("avg_replacement_value_r")), _num(row.get("outcome_count"))), reverse=True)
    return {
        "top_positive_sectors": sector_rows[:10],
        "top_negative_sectors": sorted(sector_rows, key=lambda row: _num(row.get("avg_replacement_value_r")))[:10],
    }


def holdout_sector_validation(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train = sector_diagnostics(train_rows)
    train_by_sector = {str(row.get("sector") or "UNKNOWN"): row for row in train.get("top_positive_sectors", []) + train.get("top_negative_sectors", [])}
    holdout_by_sector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in holdout_rows:
        if row.get("route_outcome_available"):
            holdout_by_sector[str(row.get("sector") or "UNKNOWN")].append(row)

    vetoes: list[dict[str, Any]] = []
    for sector, rows in holdout_by_sector.items():
        avg_replacement = _avg(row.get("same_day_replacement_value_r") for row in rows)
        avg_mae = _avg(row.get("best_route_shadow_min_mae_r") for row in rows)
        avg_capture = _avg(row.get("best_route_shadow_avg_mfe_capture") for row in rows)
        reasons = []
        if avg_replacement < 0.0:
            reasons.append("negative_holdout_replacement")
        if avg_mae < -4.0:
            reasons.append("large_holdout_mae")
        if avg_capture < 0.25:
            reasons.append("poor_holdout_capture")
        if not reasons:
            continue
        train_row = dict(train_by_sector.get(sector) or {})
        vetoes.append(
            {
                "sector": sector,
                "holdout_outcome_count": len(rows),
                "holdout_avg_replacement_value_r": avg_replacement,
                "holdout_avg_mae_r": avg_mae,
                "holdout_avg_mfe_capture": avg_capture,
                "train_avg_replacement_value_r": train_row.get("avg_replacement_value_r", 0.0),
                "veto_reason": ",".join(reasons),
            }
        )
    vetoes.sort(
        key=lambda row: (
            _num(row.get("holdout_avg_replacement_value_r")),
            _num(row.get("holdout_avg_mae_r")),
        )
    )
    return {
        "usage": "validation_veto_not_train_score_feature",
        "veto_sector_count": len(vetoes),
        "veto_sectors": vetoes,
    }


def root_cause_layer_attribution(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train_summary = _window_summary(train_rows)
    holdout_summary = _window_summary(holdout_rows)
    route_coverage = route_family_outcome_coverage(train_rows)
    total_route_trades = sum(_num(item.get("shadow_trade_count")) for item in route_coverage.values())
    positive_days = int(train_summary.get("positive_top_replacement_days", 0) or 0)
    route_bottleneck = "route outcomes are sparse" if total_route_trades < 0.08 * max(len(train_rows), 1) else "route outcomes exist but need selection"
    selection_bottleneck = (
        "same-day selection is leaving replacement value"
        if positive_days > 0 and _num(train_summary.get("top_ranked_same_day_replacement_total_r")) > 0.0
        else "same-day replacement value is not consistently positive"
    )
    path_bottleneck = (
        "path quality remains weak: high MFE arrives with severe MAE/giveback tails"
        if _num(train_summary.get("avg_top_ranked_mae_r")) < -1.0 or _num(train_summary.get("avg_top_ranked_mfe_capture")) < 0.35
        else "path quality is not the leading reranker weakness"
    )
    holdout_note = (
        "holdout evidence is sparse; use it as a veto and stability check"
        if int(holdout_summary.get("route_outcome_count", 0) or 0) < 25
        else "holdout has enough route outcomes for stability review"
    )
    primary = selection_bottleneck
    if "sparse" in route_bottleneck:
        primary = route_bottleneck
    if "weak" in path_bottleneck and positive_days > 0:
        primary = f"{selection_bottleneck}; {path_bottleneck}"
    return {
        "primary_bottleneck": primary,
        "candidate_surfacing": "frontier contains shadow opportunities" if positive_days > 0 else "frontier shadow opportunities are not yet proven",
        "candidate_selection": selection_bottleneck,
        "entry_route": route_bottleneck,
        "exit_path_management": path_bottleneck,
        "holdout_validation": holdout_note,
    }


def _attach_day_candidate_context(rows: list[dict[str, Any]], *, max_per_sector: int) -> None:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[str(row.get("trade_date") or "")].append(row)
    for day_rows in by_day.values():
        sector_counts = Counter(str(row.get("sector") or "UNKNOWN") for row in day_rows)
        sector_scores: dict[str, list[float]] = defaultdict(list)
        for row in day_rows:
            score = _optional_num(row.get("sector_daily_score_pct"))
            if score is None:
                score = _optional_num(row.get("first30_sector_leadership_pct"))
            if score is not None:
                sector_scores[str(row.get("sector") or "UNKNOWN")].append(score)
        sector_means = {sector: fmean(values) for sector, values in sector_scores.items() if values}
        ordered = sorted(sector_means.items(), key=lambda item: (item[1], item[0]), reverse=True)
        leading = {sector for sector, value in ordered[: max(1, min(2, len(ordered)))] if value >= 50.0}
        if not leading and ordered:
            leading = {ordered[0][0]}
        for row in day_rows:
            sector = str(row.get("sector") or "UNKNOWN")
            candidate_count = int(sector_counts.get(sector, 0))
            actual_counts = dict(row.get("same_day_actual_sector_counts") or {})
            actual_count = int(actual_counts.get(sector, row.get("same_day_candidate_sector_actual_count") or 0) or 0)
            projected_count = actual_count if row.get("current_realized") else actual_count + 1
            row["sector_day_candidate_count"] = candidate_count
            row["sector_day_candidate_share"] = candidate_count / len(day_rows) if day_rows else 0.0
            row["same_day_candidate_sector_actual_count"] = actual_count
            row["candidate_sector_crowding_pressure"] = max(0.0, candidate_count / max(float(max_per_sector), 1.0) - 1.0)
            row["max_per_sector_pressure"] = max(0.0, projected_count / max(float(max_per_sector), 1.0) - 1.0)
            row["leading_sector_cluster"] = sector in leading


def _attach_best_route_and_labels(row: dict[str, Any]) -> None:
    outcomes = {str(key): dict(value) for key, value in dict(row.get("route_outcomes") or {}).items() if isinstance(value, dict)}
    if not outcomes:
        row["route_outcome_available"] = False
        row["best_route_family"] = ""
        row["best_route_shadow_total_r"] = 0.0
        row["same_day_replacement_value_r"] = None
        row["marginal_slot_replacement_value_r"] = None
        return
    best_family, best = max(
        outcomes.items(),
        key=lambda item: (
            _num(item[1].get("shadow_total_r")),
            _num(item[1].get("shadow_max_mfe_r")),
            -abs(_num(item[1].get("shadow_min_mae_r"))),
        ),
    )
    total = _num(best.get("shadow_total_r"))
    mfe = _num(best.get("shadow_max_mfe_r"))
    mae = _num(best.get("shadow_min_mae_r"))
    capture = _num(best.get("shadow_avg_mfe_capture"))
    giveback = _num(best.get("shadow_avg_giveback_r"))
    actual_total = _num(row.get("same_day_actual_total_r"))
    weakest = _optional_num(row.get("same_day_weakest_actual_r"))
    row["route_outcome_available"] = True
    row["best_route_family"] = best_family
    row["best_route_shadow_total_r"] = total
    row["best_route_shadow_max_mfe_r"] = mfe
    row["best_route_shadow_min_mae_r"] = mae
    row["best_route_shadow_avg_mfe_capture"] = capture
    row["best_route_shadow_avg_giveback_r"] = giveback
    row["same_day_replacement_value_r"] = total - actual_total
    row["marginal_slot_replacement_value_r"] = total - (weakest if weakest is not None else 0.0)
    row["early_mae_path_risk_r"] = min(
        _num(row.get("h3_mae_r")),
        _num(row.get("h6_mae_r")),
        mae,
    )
    row["poor_mfe_capture"] = bool(mfe > 0.0 and capture < 0.25)


def _score_components(row: dict[str, Any], sector_priors: dict[str, Any]) -> dict[str, float]:
    sector = str(row.get("sector") or "UNKNOWN")
    prior = dict(sector_priors.get(sector) or {})
    replacement = _squash(_num(row.get("same_day_replacement_value_r")), 8.0)
    marginal = _squash(_num(row.get("marginal_slot_replacement_value_r")), 6.0)
    route_quality = 0.0
    if row.get("route_outcome_available"):
        route_quality = 0.45 * _squash(_num(row.get("best_route_shadow_total_r")), 5.0)
        route_quality += 0.35 * _squash(_num(row.get("best_route_shadow_max_mfe_r")), 8.0)
        route_quality += 0.20 * max(0.0, min(_num(row.get("best_route_shadow_avg_mfe_capture")), 1.0))
    relvol = max(_num(row.get("first30_rel_volume")), 0.0)
    cpr = max(0.0, min(_num(row.get("first30_signal_bar_cpr")), 1.0))
    first30_quality = 0.50 * min(math.log1p(relvol) / math.log1p(12.0), 1.25) + 0.50 * cpr
    rank = int(_num(row.get("frontier_rank")))
    rank_quality = 0.0 if rank <= 0 else max(0.0, (9.0 - min(rank, 12)) / 8.0)
    spread_quality = 0.50 * _squash(_num(row.get("stock_sector_daily_ret5_spread")), 0.08)
    spread_quality += 0.50 * _squash(_num(row.get("stock_sector_daily_ret20_spread")), 0.15)
    sector_quality = (
        0.20 * max(0.0, min(_num(row.get("sector_daily_score_pct")) / 100.0, 1.0))
        + 0.12 * max(0.0, min(_num(row.get("sector_daily_participation")), 1.0))
        + 0.10 * max(0.0, min(_num(row.get("sector_daily_breadth_20d")), 1.0))
        + 0.18 * max(0.0, min(_num(row.get("sector_intraday_score_pct")) / 100.0, 1.0))
        + 0.12 * max(0.0, min(_num(row.get("sector_intraday_breadth")), 1.0))
        + 0.10 * max(0.0, min(_num(row.get("sector_intraday_participation")), 1.0))
        + 0.10 * spread_quality
        + 0.08 * (1.0 if row.get("leading_sector_cluster") else 0.0)
    )
    sector_history = _sector_history_quality(prior)
    return {
        "replacement_value": 35.0 * replacement,
        "marginal_slot_value": 18.0 * marginal,
        "delayed_route_eligibility": 9.0 * _delayed_route_eligibility_quality(row),
        "route_path_quality": 16.0 * route_quality,
        "first30_quality": 10.0 * first30_quality,
        "frontier_rank_quality": 8.0 * rank_quality,
        "sector_context": 8.0 * sector_quality,
        "train_sector_history": 5.0 * sector_history,
    }


def _score_penalties(row: dict[str, Any], sector_priors: dict[str, Any]) -> dict[str, float]:
    sector = str(row.get("sector") or "UNKNOWN")
    prior = dict(sector_priors.get(sector) or {})
    mae = abs(min(_num(row.get("best_route_shadow_min_mae_r")), _num(row.get("early_mae_path_risk_r"))))
    capture = _num(row.get("best_route_shadow_avg_mfe_capture"))
    route_failure = 0.0 if row.get("route_outcome_available") else 1.0
    over_concentration = max(
        _num(row.get("max_per_sector_pressure")),
        0.50 * _num(row.get("candidate_sector_crowding_pressure")),
        0.0,
    )
    sector_prior_bad = 1.0 if int(prior.get("outcome_n", 0) or 0) >= 3 and _num(prior.get("avg_replacement_value_r")) < 0.0 else 0.0
    return {
        "large_early_mae": 14.0 * min(mae / 4.0, 2.0),
        "poor_mfe_capture": 8.0 * (1.0 - max(0.0, min(capture, 1.0))) if row.get("route_outcome_available") else 0.0,
        "route_failure": 12.0 * route_failure,
        "over_concentration": 8.0 * min(over_concentration, 2.0),
        "bad_train_sector_history": 5.0 * sector_prior_bad,
    }


def _delayed_route_eligibility_quality(row: dict[str, Any]) -> float:
    static = {str(item) for item in row.get("route_family_static_eligible_modes") or ()}
    outcomes = {str(key) for key in dict(row.get("route_outcomes") or {})}
    delayed = set(DELAYED_ROUTE_FAMILIES)
    static_share = len(static & delayed) / len(delayed)
    outcome_share = len(outcomes & delayed) / len(delayed)
    best_bonus = 1.0 if str(row.get("best_route_family") or "") in delayed else 0.0
    return 0.35 * static_share + 0.45 * outcome_share + 0.20 * best_bonus


def _sector_history_quality(prior: dict[str, Any]) -> float:
    if not prior or int(prior.get("outcome_n", 0) or 0) <= 0:
        return 0.5
    replacement = _squash(_num(prior.get("avg_replacement_value_r")), 4.0)
    mfe = _squash(_num(prior.get("avg_mfe_r")), 8.0)
    mae_control = 1.0 - min(abs(_num(prior.get("avg_mae_r"))) / 8.0, 1.0)
    capture = max(0.0, min(_num(prior.get("avg_mfe_capture")), 1.0))
    positive = max(0.0, min(_num(prior.get("positive_replacement_share")), 1.0))
    return 0.35 * replacement + 0.20 * mfe + 0.15 * mae_control + 0.15 * capture + 0.15 * positive


def _attach_ranks(rows: list[dict[str, Any]]) -> None:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[str(row.get("trade_date") or "")].append(row)
    for day_rows in by_day.values():
        day_rows.sort(
            key=lambda row: (
                _num(row.get("reranker_score")),
                _num(row.get("same_day_replacement_value_r")),
                _num(row.get("best_route_shadow_total_r")),
            ),
            reverse=True,
        )
        for index, row in enumerate(day_rows, start=1):
            row["reranker_rank_in_day"] = index


def _window_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_rows = [row for row in rows if row.get("route_outcome_available")]
    top_by_day = [row for row in rows if int(row.get("reranker_rank_in_day") or 0) == 1]
    outcome_top = [row for row in top_by_day if row.get("route_outcome_available")]
    positive_top = [row for row in outcome_top if _num(row.get("same_day_replacement_value_r")) > 0.0]
    route_counts = Counter(str(row.get("best_route_family") or "none") for row in outcome_top)
    return {
        "row_count": len(rows),
        "day_count": len({str(row.get("trade_date") or "") for row in rows}),
        "route_outcome_count": len(outcome_rows),
        "top_ranked_day_count": len(top_by_day),
        "positive_top_replacement_days": len(positive_top),
        "top_ranked_same_day_replacement_total_r": float(sum(_num(row.get("same_day_replacement_value_r")) for row in outcome_top)),
        "top_ranked_marginal_slot_replacement_total_r": float(sum(_num(row.get("marginal_slot_replacement_value_r")) for row in outcome_top)),
        "avg_top_ranked_mfe_r": _avg(row.get("best_route_shadow_max_mfe_r") for row in outcome_top),
        "avg_top_ranked_mae_r": _avg(row.get("best_route_shadow_min_mae_r") for row in outcome_top),
        "avg_top_ranked_mfe_capture": _avg(row.get("best_route_shadow_avg_mfe_capture") for row in outcome_top),
        "top_ranked_route_family_counts": dict(route_counts),
        "avg_score": _avg(row.get("reranker_score") for row in rows),
    }


def _avg(values: Iterable[Any]) -> float:
    nums = [_num(value) for value in values if _optional_num(value) is not None]
    return float(fmean(nums)) if nums else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * float(q))), 0), len(ordered) - 1)
    return float(ordered[index])


def _share(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(1 for item in items if item) / len(items) if items else 0.0


def _optional_num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _num(value: Any) -> float:
    out = _optional_num(value)
    return 0.0 if out is None else out


def _squash(value: float, scale: float) -> float:
    return 0.5 + 0.5 * math.tanh(float(value) / max(float(scale), 1e-9))


def _num_label(value: Any) -> str:
    return f"{_num(value):.2f}"


def _signed(value: Any) -> str:
    return f"{_num(value):+.2f}"
