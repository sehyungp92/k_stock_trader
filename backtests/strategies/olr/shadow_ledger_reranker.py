from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.strategies.olr.runner import compile_olr_replay_bundle, run_olr_backtest
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_olr.config import OLRConfig
from strategy_olr.execution import OLREntryPlan, OLRExitPlan, OLRTradeOutcome, simulate_olr_trade
from strategy_olr.models import OLRAfternoonContext, OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.research import (
    _afternoon_reject_reasons,
    _afternoon_rule_feature_values,
    _afternoon_score_band_rules,
    _afternoon_score_details,
    _matching_afternoon_score_band_rule,
    afternoon_selection_from_contexts,
)


SHADOW_LEDGER_VERSION = "olr-shadow-opportunity-ledger-v1"
SAME_DAY_RERANKER_VERSION = "olr-shadow-same-day-reranker-v1"
DEFAULT_OUTPUT_DIR = Path("data/backtests/output/olr/shadow_ledger")
DEFAULT_FEATURE_KEYS = (
    "daily_candidate_score",
    "daily_candidate_rank",
    "daily_rank_pct",
    "daily_signal_score",
    "relative_strength_pct",
    "accumulation_score",
    "flow_score",
    "foreign_flow_5d",
    "institutional_flow_5d",
    "flow_agreement_5d",
    "prior_return_5d",
    "prior_return_20d",
    "prior_return_60d",
    "afternoon_score",
    "afternoon_score_raw",
    "afternoon_exhaustion_score",
    "afternoon_ret",
    "vwap_ret",
    "gap",
    "rel_volume",
    "close_location",
    "open_drawdown",
    "high_from_open",
    "low_vs_prev_close",
    "range_atr",
    "lagged_flow_5d",
    "lagged_foreign_flow_5d",
    "lagged_institutional_flow_5d",
    "lagged_flow_z",
    "lagged_foreign_z",
    "lagged_institutional_z",
    "lagged_flow_agreement_5d",
    "lagged_flow_divergence_5d",
    "lagged_sector_flow_5d",
    "lagged_sector_foreign_flow_5d",
    "lagged_sector_institutional_flow_5d",
    "sector_strength_pct",
    "sector_participation",
    "sector_daily_score_pct",
    "sector_daily_ret_5d_pct",
    "sector_daily_ret_20d_pct",
    "sector_daily_breadth_20d",
    "sector_daily_participation",
    "sector_daily_rel_volume",
    "sector_daily_flow_5d",
    "sector_intraday_score_pct",
    "sector_intraday_ret_pct",
    "sector_intraday_breadth",
    "sector_intraday_rel_volume",
    "sector_intraday_participation",
    "sector_intraday_daily_score_delta",
    "sector_confirm_min_score_pct",
    "sector_confirm_quality_score",
    "sector_rotation_score",
    "stock_sector_daily_ret5_gap_pct",
    "stock_sector_daily_ret20_gap_pct",
    "stock_intraday_sector_ret_gap_pct",
    "stock_intraday_leadership_score",
    "market_score",
)


def build_shadow_opportunity_ledger(
    snapshots: Mapping[date, OLRDailySnapshot],
    contexts_by_day: Mapping[date, Mapping[str, OLRAfternoonContext]],
    bars_by_key: Mapping[tuple[date, str], Sequence[MarketBar]],
    next_session_by_date: Mapping[date, date],
    config: OLRConfig,
    dates: Sequence[date] | None = None,
    *,
    window: str = "train",
    source_label: str = "olr",
    max_shadow_candidates_per_day: int = 0,
) -> list[dict[str, Any]]:
    """Build a route-aware opportunity ledger from the same 14:30 candidate pool.

    The ledger is deliberately broader than the live selected list: it keeps the
    actual OLR selected slots and same-day shadow candidates, including names
    blocked by the current hard sector score-band rule. Rejected status is kept
    as an audit label so a reranker can learn from daily/intraday context
    without making the hardcoded sector list the objective.
    """

    selected_dates = tuple(dates or sorted(snapshots))
    rows: list[dict[str, Any]] = []
    for day in selected_dates:
        snapshot = snapshots.get(day)
        if snapshot is None:
            continue
        day_rows = build_shadow_opportunity_ledger_for_day(
            snapshot,
            contexts_by_day.get(day, {}),
            bars_by_key,
            next_session_by_date,
            config,
            window=window,
            source_label=source_label,
            max_shadow_candidates=max_shadow_candidates_per_day,
        )
        rows.extend(day_rows)
    return rows


def build_shadow_opportunity_ledger_for_day(
    snapshot: OLRDailySnapshot,
    contexts: Mapping[str, OLRAfternoonContext],
    bars_by_key: Mapping[tuple[date, str], Sequence[MarketBar]],
    next_session_by_date: Mapping[date, date],
    config: OLRConfig,
    *,
    window: str = "train",
    source_label: str = "olr",
    max_shadow_candidates: int = 0,
) -> list[dict[str, Any]]:
    actual_snapshot = afternoon_selection_from_contexts(snapshot, contexts, config)
    actual_rank_by_symbol = {candidate.symbol: index for index, candidate in enumerate(actual_snapshot.candidates, start=1)}
    actual_trade_slots = {
        candidate.symbol
        for candidate in actual_snapshot.candidates[: max(1, int(config.overnight_slot_count))]
    }
    next_day = next_session_by_date.get(snapshot.trade_date)
    all_rows: list[dict[str, Any]] = []
    outcome_by_symbol: dict[str, OLRTradeOutcome | None] = {}
    for rank, candidate in enumerate(_scored_pool_candidates(snapshot, contexts, config), start=1):
        symbol = str(candidate["symbol"]).zfill(6)
        outcome = None
        if next_day is not None:
            outcome = simulate_olr_trade(
                snapshot.trade_date,
                symbol,
                bars_by_key.get((snapshot.trade_date, symbol), ()),
                bars_by_key.get((next_day, symbol), ()),
                candidate["candidate"],
                _entry_plan_from_config(config),
                _exit_plan_from_config(config),
                config,
            )
        outcome_by_symbol[symbol] = outcome
        row = _ledger_row(
            snapshot,
            candidate,
            outcome,
            window=window,
            source_label=source_label,
            pool_rank=rank,
            actual_afternoon_rank=actual_rank_by_symbol.get(symbol, 0),
            actual_trade_slot=symbol in actual_trade_slots,
            next_day=next_day,
        )
        all_rows.append(row)

    selected_values = [
        _outcome_net_r(outcome_by_symbol.get(symbol))
        for symbol in actual_trade_slots
    ]
    weakest_selected_r = min(selected_values) if selected_values else 0.0
    selected_total_r = sum(selected_values)
    selected_count = len(actual_trade_slots)
    for row in all_rows:
        net_r = _num(row.get("route_net_r"))
        row["same_day_selected_count"] = selected_count
        row["same_day_actual_total_r"] = selected_total_r
        row["same_day_weakest_selected_net_r"] = weakest_selected_r
        row["same_day_replacement_value_r"] = net_r - weakest_selected_r
        row["marginal_slot_replacement_value_r"] = 0.0 if row["actual_trade_slot"] else net_r - weakest_selected_r
        row["same_day_actual_sector_counts"] = _sector_counts(item for item in all_rows if item.get("actual_trade_slot"))

    if max_shadow_candidates and max_shadow_candidates > 0:
        actual = [row for row in all_rows if row.get("actual_afternoon_rank")]
        shadow = [row for row in all_rows if not row.get("actual_afternoon_rank")]
        shadow.sort(key=lambda row: (int(row.get("pool_rank", 999)), str(row.get("symbol", ""))))
        keep_keys = {(row["trade_date"], row["symbol"]) for row in actual + shadow[: max(0, int(max_shadow_candidates))]}
        all_rows = [row for row in all_rows if (row["trade_date"], row["symbol"]) in keep_keys]
    return all_rows


def fit_same_day_reranker_profile(
    train_rows: Sequence[dict[str, Any]],
    *,
    feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
    target_key: str = "same_day_replacement_value_r",
    min_feature_observations: int = 8,
    min_abs_correlation: float = 0.025,
    max_abs_weight: float = 0.75,
    sector_prior_strength: float = 8.0,
    sector_prior_weight: float = 0.35,
    score_clip: float = 6.0,
    allow_slot_expansion: bool = False,
    max_replacements_per_day: int = 1,
    replacement_margin: float = 0.0,
) -> dict[str, Any]:
    rows = [row for row in train_rows if str(row.get("window") or "train") == "train"]
    targets = [_num(row.get(target_key)) for row in rows]
    target_mean = mean(targets) if targets else 0.0
    target_std = _std(targets)
    feature_stats: dict[str, dict[str, float]] = {}
    weights: dict[str, float] = {}
    for key in feature_keys:
        pairs = [(_num_or_none(row.get(key)), _num(row.get(target_key))) for row in rows]
        pairs = [(x, y) for x, y in pairs if x is not None]
        if len(pairs) < max(2, int(min_feature_observations)):
            continue
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        x_mean = mean(xs)
        x_std = _std(xs)
        corr = _correlation(xs, ys)
        coverage = len(pairs) / max(float(len(rows)), 1.0)
        feature_stats[key] = {
            "mean": x_mean,
            "std": x_std,
            "coverage": coverage,
            "correlation": corr,
            "observations": float(len(pairs)),
        }
        if x_std > 0.0 and abs(corr) >= float(min_abs_correlation):
            weights[key] = max(-float(max_abs_weight), min(float(max_abs_weight), corr * coverage))
    sector_priors = _sector_priors(rows, target_key=target_key, prior_strength=sector_prior_strength)
    profile = {
        "version": SAME_DAY_RERANKER_VERSION,
        "created_at_utc": _utc_now_iso(),
        "source_window": "train",
        "target_key": target_key,
        "slot_policy": "overlay_replace_weakest_preserve_rank",
        "allow_new_trade_days": bool(allow_slot_expansion),
        "allow_slot_expansion": bool(allow_slot_expansion),
        "max_replacements_per_day": int(max_replacements_per_day),
        "replacement_margin": float(replacement_margin),
        "row_count": len(rows),
        "feature_keys": list(feature_keys),
        "feature_stats": feature_stats,
        "weights": dict(sorted(weights.items())),
        "target_mean": target_mean,
        "target_std": target_std,
        "sector_priors": sector_priors,
        "sector_prior_weight": float(sector_prior_weight),
        "score_clip": float(score_clip),
        "causality": {
            "fit_uses": "train ledger route-aware labels only",
            "score_uses": "same-day 14:30 daily/intraday/sector features only",
            "holdout_labels_used_for_fit": False,
        },
    }
    profile["profile_hash"] = stable_signature(profile)
    return profile


def score_shadow_ledger_rows(rows: Sequence[dict[str, Any]], profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    stats = dict(profile.get("feature_stats") or {})
    weights = dict(profile.get("weights") or {})
    sector_priors = dict(profile.get("sector_priors") or {})
    sector_weight = float(profile.get("sector_prior_weight", 0.35) or 0.0)
    clip = max(float(profile.get("score_clip", 6.0) or 6.0), 0.1)
    scored: list[dict[str, Any]] = []
    for row in rows:
        components: dict[str, float] = {}
        score = 0.0
        for key, weight in weights.items():
            stat = stats.get(key) or {}
            value = _num_or_none(row.get(key))
            std = float(stat.get("std", 0.0) or 0.0)
            if value is None or std <= 0.0:
                continue
            z = max(-clip, min(clip, (value - float(stat.get("mean", 0.0) or 0.0)) / std))
            component = float(weight) * z
            components[key] = component
            score += component
        sector = str(row.get("sector") or "UNKNOWN").upper()
        sector_prior = _num(sector_priors.get(sector, 0.0))
        sector_component = sector_weight * sector_prior
        score += sector_component
        out = dict(row)
        out["reranker_score"] = score
        out["reranker_components"] = components
        out["reranker_sector_prior_component"] = sector_component
        out["reranker_profile_hash"] = str(profile.get("profile_hash") or stable_signature(profile))
        scored.append(out)
    scored.sort(
        key=lambda row: (
            str(row.get("trade_date", "")),
            -float(row.get("reranker_score", 0.0) or 0.0),
            int(row.get("pool_rank", 999) or 999),
            str(row.get("symbol", "")),
        )
    )
    current_day = None
    rank = 0
    for row in scored:
        day = row.get("trade_date")
        if day != current_day:
            current_day = day
            rank = 0
        rank += 1
        row["reranker_rank_in_day"] = rank
    return scored


def _reranker_row_is_selectable(row: Mapping[str, Any], *, replace_score_band_rules: bool) -> bool:
    reasons = {str(reason) for reason in (row.get("hard_filter_reject_reasons") or [])}
    if not reasons:
        return True
    if replace_score_band_rules and reasons == {"afternoon_score_band_rule_miss"}:
        return True
    return False


def snapshots_from_reranked_rows(
    rows: Sequence[dict[str, Any]],
    profile: Mapping[str, Any],
    *,
    top_n: int,
    source_fingerprint: str = "",
    replace_score_band_rules: bool = True,
) -> dict[date, OLRDailySnapshot]:
    scored = [
        row
        for row in score_shadow_ledger_rows(rows, profile)
        if _reranker_row_is_selectable(row, replace_score_band_rules=replace_score_band_rules)
    ]
    by_day: dict[date, list[dict[str, Any]]] = {}
    for row in scored:
        day = _parse_date(row.get("trade_date"))
        if day is not None:
            by_day.setdefault(day, []).append(row)
    out: dict[date, OLRDailySnapshot] = {}
    profile_hash = str(profile.get("profile_hash") or stable_signature(profile))
    for day, day_rows in sorted(by_day.items()):
        selected = []
        limit = max(1, int(top_n))
        if not bool(profile.get("allow_slot_expansion", False)):
            observed_slots = max(int(row.get("same_day_selected_count", 0) or 0) for row in day_rows)
            limit = min(limit, observed_slots)
        if limit <= 0:
            continue
        day_rows = _overlay_replace_weakest_preserve_rank(day_rows, limit, profile)
        total = min(limit, len(day_rows))
        for index, row in enumerate(day_rows[:limit], start=1):
            candidate = OLRDailyCandidate.from_json_dict(dict(row["candidate_payload"]))
            metadata = {
                **dict(candidate.metadata),
                "daily_candidate_score": float(candidate.metadata.get("daily_candidate_score", candidate.selection_score) or 0.0),
                "daily_candidate_rank": int(candidate.metadata.get("daily_candidate_rank", candidate.rank) or 0),
                "source": "olr_shadow_same_day_reranker",
                "shadow_ledger_version": SHADOW_LEDGER_VERSION,
                "shadow_reranker_version": SAME_DAY_RERANKER_VERSION,
                "shadow_reranker_profile_hash": profile_hash,
                "shadow_reranker_score": float(row.get("reranker_score", 0.0) or 0.0),
                "shadow_reranker_rank": index,
                "hard_filter_reject_reasons": list(row.get("hard_filter_reject_reasons") or []),
            }
            selected.append(
                replace(
                    candidate,
                    selection_score=float(row.get("reranker_score", 0.0) or 0.0),
                    rank=index,
                    rank_pct=(index / max(float(total), 1.0)) * 100.0,
                    metadata=metadata,
                )
            )
        out[day] = OLRDailySnapshot(
            trade_date=day,
            candidates=tuple(selected),
            source_fingerprint=source_fingerprint or stable_signature([profile_hash, day.isoformat(), [row["symbol"] for row in day_rows[:limit]]]),
            generated_at=datetime.combine(day, time(14, 30), tzinfo=KST),
            metadata={
                "source": "olr_shadow_same_day_reranker",
                "shadow_ledger_version": SHADOW_LEDGER_VERSION,
                "shadow_reranker_version": SAME_DAY_RERANKER_VERSION,
                "shadow_reranker_profile_hash": profile_hash,
                "intraday_selection_cutoff": "timestamp < 14:30 KST",
                "same_day_flow_used": False,
                "official_performance": False,
                "selected_symbols": [candidate.symbol for candidate in selected],
                "selected_symbol_count": len(selected),
            },
        )
    return out


def _overlay_replace_weakest_preserve_rank(
    day_rows: Sequence[dict[str, Any]],
    limit: int,
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if str(profile.get("slot_policy") or "") == "free_rerank":
        return list(day_rows[:limit])
    baseline = [
        row
        for row in day_rows
        if bool(row.get("actual_trade_slot"))
    ]
    baseline.sort(key=lambda row: (int(row.get("actual_afternoon_rank", 999) or 999), str(row.get("symbol", ""))))
    selected = list(baseline[:limit])
    if not selected:
        return []
    selected_symbols = {str(row.get("symbol") or "") for row in selected}
    shadows = [row for row in day_rows if str(row.get("symbol") or "") not in selected_symbols]
    max_replacements = max(0, int(profile.get("max_replacements_per_day", 1) or 1))
    margin = float(profile.get("replacement_margin", 0.0) or 0.0)
    replacements = 0
    for shadow in shadows:
        if replacements >= max_replacements:
            break
        weakest_index, weakest = min(
            enumerate(selected),
            key=lambda item: (
                float(item[1].get("reranker_score", 0.0) or 0.0),
                -int(item[1].get("actual_afternoon_rank", 999) or 999),
                str(item[1].get("symbol", "")),
            ),
        )
        if float(shadow.get("reranker_score", 0.0) or 0.0) <= float(weakest.get("reranker_score", 0.0) or 0.0) + margin:
            break
        selected[weakest_index] = shadow
        selected_symbols = {str(row.get("symbol") or "") for row in selected}
        replacements += 1
    return selected


def actual_snapshots_from_ledger_rows(rows: Sequence[dict[str, Any]], *, source_fingerprint: str = "") -> dict[date, OLRDailySnapshot]:
    by_day: dict[date, list[dict[str, Any]]] = {}
    for row in rows:
        rank = int(row.get("actual_afternoon_rank", 0) or 0)
        day = _parse_date(row.get("trade_date"))
        if rank > 0 and day is not None:
            by_day.setdefault(day, []).append(row)
    out: dict[date, OLRDailySnapshot] = {}
    for day, day_rows in sorted(by_day.items()):
        day_rows.sort(key=lambda row: (int(row.get("actual_afternoon_rank", 999) or 999), str(row.get("symbol", ""))))
        total = max(float(len(day_rows)), 1.0)
        candidates = []
        for row in day_rows:
            candidate = OLRDailyCandidate.from_json_dict(dict(row["candidate_payload"]))
            rank = int(row.get("actual_afternoon_rank", 999) or 999)
            candidates.append(
                replace(candidate, rank=rank, rank_pct=(rank / total) * 100.0)
            )
        out[day] = OLRDailySnapshot(
            trade_date=day,
            candidates=tuple(candidates),
            source_fingerprint=source_fingerprint or stable_signature([day.isoformat(), [row["symbol"] for row in day_rows]]),
            generated_at=datetime.combine(day, time(14, 30), tzinfo=KST),
            metadata={
                "source": "olr_shadow_ledger_actual_reconstruction",
                "shadow_ledger_version": SHADOW_LEDGER_VERSION,
                "selected_symbols": [candidate.symbol for candidate in candidates],
                "selected_symbol_count": len(candidates),
                "official_performance": False,
            },
        )
    return out


def evaluate_same_day_reranker_with_replay(
    train_rows: Sequence[dict[str, Any]],
    oos_rows: Sequence[dict[str, Any]],
    bars_by_key: Mapping[tuple[date, str], Sequence[MarketBar]],
    config: OLRConfig,
    mutations: Mapping[str, Any] | None = None,
    *,
    runtime_config: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    initial_equity: float = 10_000_000.0,
) -> dict[str, Any]:
    profile = dict(profile or fit_same_day_reranker_profile(train_rows))
    raw_config = dict(runtime_config or {})
    raw_config["capability_level"] = "compiled"
    raw_config["initial_equity"] = initial_equity
    raw_mutations = dict(mutations or {})
    train_actual = _run_snapshot_replay(actual_snapshots_from_ledger_rows(train_rows), bars_by_key, raw_config, raw_mutations, "train_actual")
    train_reranked = _run_snapshot_replay(
        snapshots_from_reranked_rows(train_rows, profile, top_n=config.afternoon_top_n),
        bars_by_key,
        raw_config,
        raw_mutations,
        "train_reranked",
    )
    oos_actual = _run_snapshot_replay(actual_snapshots_from_ledger_rows(oos_rows), bars_by_key, raw_config, raw_mutations, "oos_actual")
    oos_reranked = _run_snapshot_replay(
        snapshots_from_reranked_rows(oos_rows, profile, top_n=config.afternoon_top_n),
        bars_by_key,
        raw_config,
        raw_mutations,
        "oos_reranked",
    )
    train_delta = _metric_delta(train_reranked["metrics"], train_actual["metrics"])
    oos_delta = _metric_delta(oos_reranked["metrics"], oos_actual["metrics"])
    promotion_pass = (
        _metric_net(oos_reranked["metrics"]) > _metric_net(oos_actual["metrics"])
        and _metric_trades(oos_reranked["metrics"]) >= _metric_trades(oos_actual["metrics"])
        and _metric_net(train_reranked["metrics"]) >= _metric_net(train_actual["metrics"]) - 0.02
    )
    return {
        "strategy": "olr",
        "shadow_ledger_version": SHADOW_LEDGER_VERSION,
        "same_day_reranker_version": SAME_DAY_RERANKER_VERSION,
        "created_at_utc": _utc_now_iso(),
        "profile": profile,
        "promotion_pass": promotion_pass,
        "promotion_policy": "promote_only_if_oos_official_mtm_improves_and_train_materially_preserved",
        "train": {"actual": train_actual, "reranked": train_reranked, "delta": train_delta},
        "oos": {"actual": oos_actual, "reranked": oos_reranked, "delta": oos_delta},
        "candidate_mutation": {
            "olr.shadow_reranker.enabled": True,
            "olr.shadow_reranker.profile": profile,
            "olr.shadow_reranker.replace_score_band_rules": True,
            "olr.afternoon.score_band_rules": [],
        }
        if promotion_pass
        else {},
    }


def write_shadow_reranker_artifacts(
    train_rows: Sequence[dict[str, Any]],
    oos_rows: Sequence[dict[str, Any]],
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "shadow_ledger_train.jsonl"
    oos_path = out / "shadow_ledger_oos.jsonl"
    summary_path = out / "shadow_reranker_summary.json"
    report_path = out / "shadow_reranker_report.md"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(oos_path, oos_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return {
        "train_jsonl": str(train_path),
        "oos_jsonl": str(oos_path),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
    }


def feature_coverage(rows: Sequence[dict[str, Any]], feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    total = max(float(len(rows)), 1.0)
    for key in feature_keys:
        values = [_num_or_none(row.get(key)) for row in rows]
        present = [value for value in values if value is not None]
        out[key] = {
            "coverage": len(present) / total,
            "observations": float(len(present)),
            "mean": mean(present) if present else 0.0,
        }
    return out


def _scored_pool_candidates(
    snapshot: OLRDailySnapshot,
    contexts: Mapping[str, OLRAfternoonContext],
    config: OLRConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rules = _afternoon_score_band_rules(config)
    for candidate in snapshot.candidates:
        symbol = str(candidate.symbol).zfill(6)
        ctx = contexts.get(symbol)
        if ctx is None:
            rows.append(
                {
                    "symbol": symbol,
                    "candidate": candidate,
                    "context": None,
                    "score": float(candidate.selection_score),
                    "raw_score": float(candidate.selection_score),
                    "exhaustion_score": 0.0,
                    "rule_features": {},
                    "reject_reasons": ("missing_completed_afternoon_bars",),
                    "matched_score_band_rule": "",
                }
            )
            continue
        score, raw_score, exhaustion = _afternoon_score_details(ctx, config)
        matched_rule = _matching_afternoon_score_band_rule(score, ctx, rules, exhaustion) if rules else ""
        reasons = list(_afternoon_reject_reasons(ctx, config))
        if score < float(config.afternoon_min_score):
            reasons.append("afternoon_score_below_floor")
        if score > float(config.afternoon_max_score):
            reasons.append("afternoon_score_above_cap")
        if (
            float(config.afternoon_reject_score_max) > float(config.afternoon_reject_score_min)
            and float(config.afternoon_reject_score_min) <= score <= float(config.afternoon_reject_score_max)
        ):
            reasons.append("afternoon_score_in_reject_band")
        if rules and not matched_rule:
            reasons.append("afternoon_score_band_rule_miss")
        rule_features = _afternoon_rule_feature_values(ctx, score, exhaustion)
        candidate_payload = replace(
            candidate,
            selection_score=score,
            metadata={
                **dict(candidate.metadata),
                "daily_candidate_score": float(candidate.selection_score or 0.0),
                "daily_candidate_rank": int(candidate.rank or 0),
                "afternoon_score": score,
                "afternoon_score_raw": raw_score,
                "afternoon_exhaustion_score": exhaustion,
                "afternoon_score_band_rule": matched_rule,
                "afternoon_features": _feature_values(candidate, ctx, score, raw_score, exhaustion, rule_features),
            },
        )
        rows.append(
            {
                "symbol": symbol,
                "candidate": candidate_payload,
                "context": ctx,
                "score": score,
                "raw_score": raw_score,
                "exhaustion_score": exhaustion,
                "rule_features": rule_features,
                "reject_reasons": tuple(dict.fromkeys(reasons)),
                "matched_score_band_rule": matched_rule,
            }
        )
    rows.sort(key=lambda row: (-float(row.get("score", 0.0) or 0.0), int(row["candidate"].rank or 999), str(row["symbol"])))
    return rows


def _ledger_row(
    snapshot: OLRDailySnapshot,
    candidate_row: Mapping[str, Any],
    outcome: OLRTradeOutcome | None,
    *,
    window: str,
    source_label: str,
    pool_rank: int,
    actual_afternoon_rank: int,
    actual_trade_slot: bool,
    next_day: date | None,
) -> dict[str, Any]:
    candidate: OLRDailyCandidate = candidate_row["candidate"]
    ctx: OLRAfternoonContext | None = candidate_row.get("context")
    rule_features = dict(candidate_row.get("rule_features") or {})
    features = _feature_values(candidate, ctx, float(candidate_row.get("score", 0.0) or 0.0), float(candidate_row.get("raw_score", 0.0) or 0.0), float(candidate_row.get("exhaustion_score", 0.0) or 0.0), rule_features)
    row = {
        "window": window,
        "source_label": source_label,
        "trade_date": snapshot.trade_date.isoformat(),
        "next_trade_date": next_day.isoformat() if next_day else "",
        "symbol": candidate.symbol,
        "sector": str(candidate.sector or "UNKNOWN").upper(),
        "pool_rank": pool_rank,
        "actual_afternoon_rank": actual_afternoon_rank,
        "actual_trade_slot": bool(actual_trade_slot),
        "frontier_role": "actual" if actual_trade_slot else "shadow",
        "hard_filter_reject_reasons": list(candidate_row.get("reject_reasons") or ()),
        "matched_score_band_rule": str(candidate_row.get("matched_score_band_rule") or ""),
        "fill_feasible": outcome is not None,
        "route_net_r": _outcome_net_r(outcome),
        "route_net_return_pct": float(outcome.net_return_pct) if outcome else 0.0,
        "route_mfe_r": float(outcome.mfe_r) if outcome else 0.0,
        "route_mae_r": float(outcome.mae_r) if outcome else 0.0,
        "route_mfe_capture": float(outcome.mfe_capture) if outcome else 0.0,
        "route_giveback_r": max(0.0, (float(outcome.mfe_r) - _outcome_net_r(outcome))) if outcome else 0.0,
        "entry_reason": str(outcome.entry_reason) if outcome else "",
        "exit_reason": str(outcome.exit_reason) if outcome else "",
        "bars_held": int(outcome.bars_held) if outcome else 0,
        "candidate_payload": candidate.to_json_dict(),
    }
    row.update(features)
    return row


def _feature_values(
    candidate: OLRDailyCandidate,
    ctx: OLRAfternoonContext | None,
    score: float,
    raw_score: float,
    exhaustion: float,
    rule_features: Mapping[str, Any],
) -> dict[str, float]:
    metadata = dict(candidate.metadata or {})
    values = {
        "daily_candidate_score": float(metadata.get("daily_candidate_score", candidate.selection_score) or 0.0),
        "daily_candidate_rank": float(candidate.rank or 999),
        "daily_rank_pct": float(candidate.rank_pct or 0.0),
        "daily_signal_score": float(candidate.daily_signal_score or metadata.get("daily_signal_score", 0.0) or 0.0),
        "relative_strength_pct": float(candidate.rs_percentile or metadata.get("rs_percentile", 0.0) or 0.0),
        "accumulation_score": float(candidate.accumulation_score or 0.0),
        "flow_score": float(candidate.flow_score or 0.0),
        "foreign_flow_5d": float(candidate.foreign_flow_5d or metadata.get("lagged_foreign_flow_5d", 0.0) or 0.0),
        "institutional_flow_5d": float(candidate.institutional_flow_5d or metadata.get("lagged_institutional_flow_5d", 0.0) or 0.0),
        "flow_agreement_5d": float(candidate.flow_agreement_5d or metadata.get("lagged_flow_agreement_5d", 0.0) or 0.0),
        "afternoon_score": float(score),
        "afternoon_score_raw": float(raw_score),
        "afternoon_exhaustion_score": float(exhaustion),
        "market_score": float(metadata.get("market_heat_score", 0.0) or 0.0),
    }
    if ctx is not None:
        values.update(
            {
                "prior_return_5d": float(ctx.prior_return_5d),
                "prior_return_20d": float(ctx.prior_return_20d),
                "prior_return_60d": float(ctx.prior_return_60d),
                "afternoon_ret": float(ctx.afternoon_ret),
                "vwap_ret": float(ctx.vwap_ret),
                "gap": float(ctx.gap),
                "rel_volume": float(ctx.rel_volume),
                "close_location": float(ctx.close_location),
                "open_drawdown": float(ctx.open_drawdown),
                "high_from_open": float(ctx.high_from_open),
                "low_vs_prev_close": float(ctx.low_vs_prev_close),
                "range_atr": float(ctx.range_atr),
                "lagged_flow_5d": float(ctx.lagged_flow_5d),
                "lagged_foreign_flow_5d": float(ctx.lagged_foreign_flow_5d),
                "lagged_institutional_flow_5d": float(ctx.lagged_institutional_flow_5d),
                "lagged_flow_z": float(ctx.lagged_flow_z),
                "lagged_foreign_z": float(ctx.lagged_foreign_z),
                "lagged_institutional_z": float(ctx.lagged_institutional_z),
                "lagged_flow_agreement_5d": float(ctx.lagged_flow_agreement_5d),
                "lagged_flow_divergence_5d": float(ctx.lagged_flow_divergence_5d),
                "lagged_sector_flow_5d": float(ctx.lagged_sector_flow_5d),
                "lagged_sector_foreign_flow_5d": float(ctx.lagged_sector_foreign_flow_5d),
                "lagged_sector_institutional_flow_5d": float(ctx.lagged_sector_institutional_flow_5d),
            }
        )
    for key in DEFAULT_FEATURE_KEYS:
        if key in rule_features:
            values[key] = _num(rule_features[key])
    return values


def _entry_plan_from_config(config: OLRConfig) -> OLREntryPlan:
    payload = dict(config.trade_entry_plan or {})
    if payload:
        allowed = set(OLREntryPlan.__dataclass_fields__)
        return OLREntryPlan(**{key: value for key, value in payload.items() if key in allowed})
    return OLREntryPlan("", config.entry_mode)


def _exit_plan_from_config(config: OLRConfig) -> OLRExitPlan:
    payload = dict(config.trade_exit_plan or {})
    if payload:
        allowed = set(OLRExitPlan.__dataclass_fields__)
        return OLRExitPlan(**{key: value for key, value in payload.items() if key in allowed})
    return OLRExitPlan("", mode=config.exit_mode)


def _outcome_net_r(outcome: OLRTradeOutcome | None) -> float:
    if outcome is None:
        return 0.0
    return float(outcome.net_return_pct) * max(float(outcome.entry_price), 1e-9) / max(float(outcome.risk_per_share), 1e-9)


def _sector_priors(rows: Sequence[dict[str, Any]], *, target_key: str, prior_strength: float) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sector") or "UNKNOWN").upper(), []).append(_num(row.get(target_key)))
    global_mean = mean([_num(row.get(target_key)) for row in rows]) if rows else 0.0
    out = {}
    for sector, values in grouped.items():
        n = len(values)
        raw = mean(values) if values else 0.0
        shrink = n / max(n + float(prior_strength), 1.0)
        out[sector] = shrink * raw + (1.0 - shrink) * global_mean
    return dict(sorted(out.items()))


def _run_snapshot_replay(
    snapshots: Mapping[date, OLRDailySnapshot],
    bars_by_key: Mapping[tuple[date, str], Sequence[MarketBar]],
    config: Mapping[str, Any],
    mutations: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not snapshots:
        return _empty_snapshot_replay(label)
    dates = set(snapshots)
    ordered_dates = sorted(dates)
    if ordered_dates:
        all_dates = sorted({day for day, _ in bars_by_key} | dates)
        next_by_date = {day: all_dates[index + 1] for index, day in enumerate(all_dates[:-1])}
        dates.update(next_by_date.get(day) for day in ordered_dates if next_by_date.get(day) is not None)
    bars = [bar for (day, _), day_bars in bars_by_key.items() if day in dates for bar in day_bars]
    bundle = compile_olr_replay_bundle(
        bars=bars,
        snapshots=dict(snapshots),
        source_fingerprint=stable_signature([label, [day.isoformat() for day in sorted(snapshots)], [snapshot.artifact_hash for snapshot in snapshots.values()]]),
    )
    result = run_olr_backtest(dict(config), dict(mutations), replay_bundle=bundle)
    return {
        "label": label,
        "metrics": dict(result.metrics),
        "source_fingerprint": result.source_fingerprint,
        "candidate_snapshot_hash": result.candidate_snapshot_hash,
        "feature_bundle_hash": result.feature_bundle_hash,
    }


def _empty_snapshot_replay(label: str) -> dict[str, Any]:
    signature = stable_signature([label, "empty"])
    metrics = {
        "total_trades": 0.0,
        "entry_fill_count": 0.0,
        "win_rate": 0.0,
        "entry_level_win_rate": 0.0,
        "net_return_pct": 0.0,
        "official_mtm_net_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "official_mtm_max_drawdown_pct": 0.0,
        "candidate_snapshot_count": 0.0,
        "replay_event_count": 0.0,
        "official_performance": False,
    }
    return {
        "label": label,
        "metrics": metrics,
        "source_fingerprint": signature,
        "candidate_snapshot_hash": signature,
        "feature_bundle_hash": signature,
    }


def _metric_delta(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, float]:
    return {
        "official_mtm_net_return_pct": _metric_net(new) - _metric_net(old),
        "official_mtm_max_drawdown_pct": _metric_dd(new) - _metric_dd(old),
        "entry_fill_count": _metric_trades(new) - _metric_trades(old),
        "win_rate": _metric_win(new) - _metric_win(old),
    }


def _metric_net(metrics: Mapping[str, Any]) -> float:
    return _num(metrics.get("official_mtm_net_return_pct", metrics.get("net_return_pct", 0.0)))


def _metric_dd(metrics: Mapping[str, Any]) -> float:
    return abs(_num(metrics.get("official_mtm_max_drawdown_pct", metrics.get("max_drawdown_pct", 0.0))))


def _metric_trades(metrics: Mapping[str, Any]) -> float:
    return _num(metrics.get("entry_fill_count", metrics.get("total_trades", 0.0)))


def _metric_win(metrics: Mapping[str, Any]) -> float:
    return _num(metrics.get("win_rate", metrics.get("entry_level_win_rate", 0.0)))


def _sector_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        sector = str(row.get("sector") or "UNKNOWN").upper()
        counts[sector] = counts.get(sector, 0) + 1
    return dict(sorted(counts.items()))


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def _render_report(summary: Mapping[str, Any]) -> str:
    train = summary.get("train", {})
    oos = summary.get("oos", {})
    best_variant = summary.get("best_variant") or {}
    lines = [
        "# OLR Shadow Opportunity Ledger Reranker",
        "",
        f"- Ledger version: `{summary.get('shadow_ledger_version', SHADOW_LEDGER_VERSION)}`",
        f"- Reranker version: `{summary.get('same_day_reranker_version', SAME_DAY_RERANKER_VERSION)}`",
        f"- Promotion pass: `{bool(summary.get('promotion_pass'))}`",
        "- Fit policy: train-only route-aware shadow labels; OOS is validation only.",
        "",
        "## Replay Validation",
        _report_metric_line("Train actual", (train.get("actual") or {}).get("metrics", {})),
        _report_metric_line("Train reranked", (train.get("reranked") or {}).get("metrics", {})),
        _report_metric_line("OOS actual", (oos.get("actual") or {}).get("metrics", {})),
        _report_metric_line("OOS reranked", (oos.get("reranked") or {}).get("metrics", {})),
        "",
        "## Best Variant",
        _variant_report_line(best_variant) if best_variant else "- None",
        "",
        "## Variant Sweep",
        "| Variant | Margin | Train MTM | Train Trades | OOS MTM | OOS Trades | OOS Win | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *[_variant_table_line(row) for row in sorted(summary.get("variant_sweep") or [], key=lambda item: _metric_net(((item.get("oos") or {}).get("reranked_metrics") or {})), reverse=True)],
        "",
        "## Candidate Mutation",
        "```json",
        json.dumps(summary.get("candidate_mutation") or {}, indent=2, sort_keys=True, default=str),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _variant_report_line(row: Mapping[str, Any]) -> str:
    if not row:
        return "- None"
    return (
        f"- `{row.get('name', '')}`: "
        f"Train MTM {_metric_net(((row.get('train') or {}).get('reranked_metrics') or {})) * 100:.2f}%, "
        f"OOS MTM {_metric_net(((row.get('oos') or {}).get('reranked_metrics') or {})) * 100:.2f}%, "
        f"promotion pass `{bool(row.get('promotion_pass'))}`"
    )


def _variant_table_line(row: Mapping[str, Any]) -> str:
    train_metrics = (row.get("train") or {}).get("reranked_metrics") or {}
    oos_metrics = (row.get("oos") or {}).get("reranked_metrics") or {}
    return (
        f"| `{row.get('name', '')}` "
        f"| {float(row.get('replacement_margin', 0.0) or 0.0):.2f} "
        f"| {_metric_net(train_metrics) * 100:.2f}% "
        f"| {_metric_trades(train_metrics):.0f} "
        f"| {_metric_net(oos_metrics) * 100:.2f}% "
        f"| {_metric_trades(oos_metrics):.0f} "
        f"| {_metric_win(oos_metrics) * 100:.2f}% "
        f"| `{bool(row.get('promotion_pass'))}` |"
    )


def _report_metric_line(label: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"- {label}: MTM {100.0 * _metric_net(metrics):.2f}%, "
        f"trades {_metric_trades(metrics):.0f}, win {100.0 * _metric_win(metrics):.2f}%, "
        f"DD {100.0 * _metric_dd(metrics):.2f}%"
    )


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _num_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _num(value: Any) -> float:
    result = _num_or_none(value)
    return 0.0 if result is None else result


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    var = sum((value - avg) ** 2 for value in values) / max(float(len(values) - 1), 1.0)
    return math.sqrt(max(var, 0.0))


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_std = _std(xs)
    y_std = _std(ys)
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / max(float(len(xs) - 1), 1.0)
    corr = cov / (x_std * y_std)
    if math.isnan(corr) or math.isinf(corr):
        return 0.0
    return max(-1.0, min(1.0, corr))
