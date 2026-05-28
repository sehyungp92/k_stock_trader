from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Iterable

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.strategies.kalcb.first30_signal_sweep import First30Context, KALCBFirst30Dataset, Selection, build_contexts, daily_feature, shared_first30_feature
from backtests.strategies.kalcb.trade_plan_sweep import EntrySpec, ExitSpec, TradePlanSpec, _core_outcomes_metrics_digest, compile_core_replay, find_entry_signal, simulate_trade
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.research import build_research_snapshot, daily_selection_from_snapshot


STRUCTURAL_CAMPAIGN_SURFACING_VERSION = "kalcb-stage09-structural-campaign-surfacing-v1"
STRUCTURAL_CAMPAIGN_USAGE_CONTRACT = "research_only_shared_daily_selection_source_no_oracle_live_features"
ALCB_DELTA_DIAGNOSTICS_VERSION = "kalcb-stage09-alcb-delta-diagnostics-v1"
ALCB_BREAKOUT_REPLAY_VERSION = "kalcb-stage09-alcb-breakout-replay-v1"
ALCB_FAITHFULNESS_FUNNEL_VERSION = "kalcb-stage09-alcb-faithfulness-funnel-v1"
RECALL_KS = (8, 16, 32, 64)
OPTIMIZER_SHORTLIST_SIZE = 5
OPTIMIZER_SOURCE_SHORTLIST_SIZE = 60
ALCB_BREAKOUT_REPLAY_SHORTLIST_SIZE = 5
ALCB_FAITHFULNESS_SHORTLIST_SIZE = 8
ALCB_FAITHFULNESS_TOP_N_GRID = (8, 12, 16, 24, 32, 40, 64)
ALCB_FAITHFULNESS_FILTER_GRID = (
    {"min_first30_rel_volume": 0.0, "min_first30_signal_cpr": 0.45, "require_first30_campaign_breakout_acceptance": False},
    {"min_first30_rel_volume": 1.5, "min_first30_signal_cpr": 0.55, "require_first30_campaign_breakout_acceptance": False},
    {"min_first30_rel_volume": 2.0, "min_first30_signal_cpr": 0.60, "require_first30_campaign_breakout_acceptance": False},
    {"min_first30_rel_volume": 2.0, "min_first30_signal_cpr": 0.60, "require_first30_campaign_breakout_acceptance": True},
)
ALCB_FAITHFULNESS_SELECTOR_MODES = (
    "structural_first30",
    "structural_first30_causal_tiebreak",
    "first30_confirmation",
    "blend_struct60_first3030_causal10",
    "blend_struct70_causal30",
)
ALCB_FAITHFULNESS_TOP1_MODES = ("first30_confirmation", "blend_struct70_causal30")
STRUCTURAL_MIN_SCORE_GRID = (3.0, 4.0, 5.0, 6.0, 7.0)
STRUCTURAL_FRONTIER_GRID = (16, 24, 32, 40, 64)
STRUCTURAL_MIN_RS_GRID = (0.0, 50.0, 60.0, 70.0)
STRUCTURAL_MIN_SECTOR_DAILY_GRID = (0.0, 45.0, 55.0, 65.0)
SELECTOR_VARIANT_SPECS = (
    {
        "name": "structural_first30",
        "description": "Structural score first, first30 confirmation second.",
        "causal_weight": 0.0,
    },
    {
        "name": "structural_first30_causal_tiebreak",
        "description": "Structural score first, first30 confirmation second, causal calibration as a final tie-break.",
        "causal_weight": 0.02,
    },
    {
        "name": "first30_confirmation",
        "description": "First30 confirmation rank inside the structurally valid source pool.",
        "causal_weight": 0.0,
    },
    {
        "name": "blend_struct70_first3020_causal10",
        "description": "Weighted selector: 70% structural, 20% first30, 10% causal calibration.",
        "structural_weight": 0.70,
        "first30_weight": 0.20,
        "causal_weight": 0.10,
    },
    {
        "name": "blend_struct60_first3030_causal10",
        "description": "Weighted selector: 60% structural, 30% first30, 10% causal calibration.",
        "structural_weight": 0.60,
        "first30_weight": 0.30,
        "causal_weight": 0.10,
    },
    {
        "name": "blend_struct70_causal30",
        "description": "Diagnostic weighted selector: 70% structural, 30% causal calibration.",
        "structural_weight": 0.70,
        "first30_weight": 0.0,
        "causal_weight": 0.30,
        "diagnostic": True,
    },
)
POOL_VARIANT_SPECS = (
    {"name": "structural_active_budget", "frontier_size": None},
    {"name": "structural_frontier24", "frontier_size": 24},
    {"name": "structural_frontier32", "frontier_size": 32},
    {"name": "structural_frontier40", "frontier_size": 40},
)
ORACLE_SCORE_KEYS = (
    "label_composite_oracle_recall",
    "oracle_score",
    "best_route_shadow_r",
    "same_day_replacement_value_r",
    "net_r",
    "mfe_r",
)
CAMPAIGN_ROUTE_SPECS = (
    {"name": "pullback_box_high", "mode": "pullback_acceptance", "level_source": "campaign_box_high", "min_reclaim_closes": 1},
    {"name": "pullback_campaign_avwap", "mode": "pullback_acceptance", "level_source": "campaign_avwap", "min_reclaim_closes": 1},
    {"name": "avwap_reclaim_campaign_avwap", "mode": "avwap_reclaim", "level_source": "campaign_avwap", "min_reclaim_closes": 1},
    {"name": "or_high_breakout_acceptance", "mode": "or_high_reclaim", "level_source": "campaign_breakout_level", "min_reclaim_closes": 2},
    {"name": "pullback_breakout_acceptance", "mode": "pullback_acceptance", "level_source": "campaign_breakout_level", "min_reclaim_closes": 2},
)
POOL_ARTIFACT_KEYS = (
    "window",
    "trade_date",
    "symbol",
    "sector",
    "structural_source_rank",
    "structural_source_role",
    "frontier_rank",
    "frontier_role",
    "selection_score",
    "relative_strength_pct",
    "stock_vs_universe_strength",
    "sector_daily_score_pct",
    "sector_participation",
    "structural_campaign_score",
    "first30_confirmation_score",
    "campaign_state",
    "campaign_box_high",
    "campaign_box_low",
    "campaign_box_mid",
    "campaign_box_range_pct",
    "campaign_box_containment",
    "campaign_box_atr_ratio",
    "campaign_box_squeeze_pct",
    "campaign_box_tier",
    "campaign_avwap",
    "campaign_avwap_anchor_available",
    "campaign_breakout_level",
    "campaign_breakout_displacement",
    "first30_rel_volume",
    "first30_signal_cpr",
    "first30_vwap_ret",
    "first30_gap",
    "first30_low_vs_prev_close",
    "first30_breakout_confirmation",
    "first30_breakout_acceptance",
    "first30_breakout_acceptance_closes",
    "first30_above_campaign_avwap",
    "campaign_avwap_distance_pct",
    "campaign_box_high_distance_pct",
    "causal_calibration_score",
    "causal_rank_in_day",
    "causal_score_source",
    "causal_score_uses_ex_post_labels",
    "selector_variant",
    "active_rank_score",
    "score_uses_ex_post_labels",
    "pool_variant",
    "pool_rank",
    "pool_size",
    "pool_active",
    "active_budget_source",
    "active_budget_for_day",
    "frontier_role_for_replay",
)


def compute_campaign_avwap(
    bars: Iterable[MarketBar],
    anchor_date: str | date | None,
    cutoff_ts: datetime | None,
) -> float | None:
    parsed_anchor = _parse_date(anchor_date)
    cum_pv = 0.0
    cum_vol = 0.0
    for bar in sorted(bars, key=lambda item: item.timestamp):
        if parsed_anchor is not None and bar.timestamp.date() < parsed_anchor:
            continue
        if cutoff_ts is not None and bar.timestamp > cutoff_ts:
            continue
        volume = max(float(bar.volume), 0.0)
        if volume <= 0.0:
            continue
        typical = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        cum_pv += typical * volume
        cum_vol += volume
    if cum_vol <= 0.0:
        return None
    return float(cum_pv / cum_vol)


def attach_first30_confirmation(
    context: First30Context,
    campaign_metadata: dict[str, Any],
    cfg: KALCBConfig,
    *,
    bars: Iterable[MarketBar] | None = None,
) -> dict[str, Any]:
    del cfg
    metadata = dict(campaign_metadata or {})
    campaign = metadata.get("structural_campaign") if isinstance(metadata.get("structural_campaign"), dict) else {}
    box = campaign.get("compression_box") if isinstance(campaign, dict) else None
    anchor = (box or {}).get("start_date") if isinstance(box, dict) else None
    first30_bars = list(getattr(context, "bars", ()) or ())
    cutoff = max((bar.timestamp for bar in first30_bars), default=None)
    avwap = compute_campaign_avwap(bars or first30_bars, anchor, cutoff)
    breakout = qualify_first30_campaign_breakout(context, metadata)
    breakout_level = float(metadata.get("campaign_breakout_level") or metadata.get("campaign_box_high") or 0.0)
    available_bars = sorted(list(bars or first30_bars), key=lambda item: item.timestamp)
    parsed_anchor = _parse_date(anchor) if anchor else None
    acceptance_closes = _consecutive_closes_above_level(available_bars, breakout_level)
    breakout_acceptance = breakout_level > 0.0 and acceptance_closes >= 2
    above_campaign_avwap = bool(avwap and float(context.intraday.close) > float(avwap))
    relvol_score = 2.0 if context.rel_volume >= 3.0 else 1.0 if context.rel_volume >= 1.5 else 0.0
    cpr_score = 1.0 if context.close_location >= 0.70 else 0.5 if context.close_location >= 0.55 else 0.0
    gap_retention = context.low_vs_prev_close / max(abs(context.gap), 1e-6) if context.gap > 0 else 0.0
    gap_score = 1.0 if gap_retention >= 0.75 else 0.5 if gap_retention >= 0.35 else 0.0
    vwap_score = 1.0 if context.vwap_ret >= 0.0 else 0.0
    acceptance_score = 1.0 if breakout_acceptance else 0.0
    score = relvol_score + cpr_score + gap_score + vwap_score + acceptance_score
    out = {
        **metadata,
        "campaign_avwap": float(avwap or 0.0),
        "campaign_avwap_source": "anchor_to_first30_cutoff" if avwap else "unavailable",
        "campaign_avwap_anchor_date": str(anchor or ""),
        "campaign_avwap_cutoff_ts": cutoff.isoformat() if cutoff else "",
        "campaign_avwap_anchor_available": bool(parsed_anchor and available_bars and available_bars[0].timestamp.date() <= parsed_anchor),
        "campaign_avwap_distance_pct": float(context.intraday.close) / max(float(avwap or 0.0), 1e-9) - 1.0 if avwap else 0.0,
        "campaign_box_high_distance_pct": float(context.intraday.close) / max(float(metadata.get("campaign_box_high") or 0.0), 1e-9) - 1.0 if float(metadata.get("campaign_box_high") or 0.0) > 0 else 0.0,
        "first30_confirmation_score": float(score),
        "first30_rel_volume": float(context.rel_volume),
        "first30_signal_cpr": float(context.close_location),
        "first30_vwap_ret": float(context.vwap_ret),
        "first30_gap": float(context.gap),
        "first30_low_vs_prev_close": float(context.low_vs_prev_close),
        "first30_above_campaign_avwap": above_campaign_avwap,
        "first30_breakout_confirmation": bool(breakout["first30_breakout_confirmation"]),
        "first30_breakout_acceptance": bool(breakout_acceptance),
        "first30_breakout_acceptance_closes": int(acceptance_closes),
        "campaign_breakout_displacement": float(breakout["campaign_breakout_displacement"]),
        "campaign_state": "first30_confirmed" if breakout["first30_breakout_confirmation"] or breakout_acceptance else str(metadata.get("campaign_state") or "none"),
    }
    out["campaign_state_score"] = 4.0 if out["campaign_state"] == "first30_confirmed" else float(metadata.get("campaign_state_score") or 0.0)
    return out


def qualify_first30_campaign_breakout(context: First30Context, campaign_metadata: dict[str, Any]) -> dict[str, Any]:
    level = float(campaign_metadata.get("campaign_breakout_level") or campaign_metadata.get("campaign_box_high") or 0.0)
    close = float(context.intraday.close)
    high = float(context.intraday.high)
    atr = max(float(context.daily.atr14 or 0.0), 1e-9)
    displacement = (close - level) / atr if level > 0.0 else 0.0
    confirmed = level > 0.0 and close >= level and high >= level and context.rel_volume >= 1.0
    return {
        "first30_breakout_confirmation": bool(confirmed),
        "campaign_breakout_displacement": float(displacement),
    }


def _campaign_compression_box(metadata: dict[str, Any]) -> dict[str, Any]:
    box = metadata.get("compression_box")
    if isinstance(box, dict):
        return box
    campaign = metadata.get("structural_campaign")
    if isinstance(campaign, dict) and isinstance(campaign.get("compression_box"), dict):
        return dict(campaign["compression_box"])
    return {}


def _consecutive_closes_above_level(bars: Iterable[MarketBar], level: float) -> int:
    if level <= 0.0:
        return 0
    count = 0
    for bar in reversed(sorted(bars, key=lambda item: item.timestamp)):
        if float(bar.close) <= level:
            break
        count += 1
    return count


def build_structural_campaign_artifact_rows(
    dataset: KALCBFirst30Dataset,
    contexts: dict[date, tuple[First30Context, ...]] | None,
    cfg: KALCBConfig,
    *,
    window: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    context_by_key = {(day, ctx.symbol): ctx for day, items in (contexts or {}).items() for ctx in items}
    for trade_day in dataset.trading_dates:
        research = build_research_snapshot(
            dataset.daily_by_symbol,
            trade_day,
            cfg,
            sector_map=dataset.sector_map,
            daily_flow_by_symbol=dataset.flow_by_symbol,
            daily_foreign_flow_by_symbol=dataset.foreign_flow_by_symbol,
            daily_institutional_flow_by_symbol=dataset.institutional_flow_by_symbol,
            source_fingerprint=f"{dataset.daily_source_fingerprint}:{trade_day.isoformat()}",
            compute_source_fingerprint=False,
            metadata={"research_causal_source_fingerprint_policy": "stage09_prior_daily_dataset_hash"},
        )
        snapshot = daily_selection_from_snapshot(research, cfg)
        active_symbols = set(snapshot.metadata.get("active_symbols") or ())
        for rank, candidate in enumerate(snapshot.candidates, start=1):
            ctx = context_by_key.get((trade_day, candidate.symbol)) or _lazy_first30_context(dataset, trade_day, candidate.symbol)
            base_meta = dict(candidate.metadata or {})
            overlay = (
                attach_first30_confirmation(ctx, base_meta, cfg, bars=dataset.bars_by_key.get((trade_day, candidate.symbol), ()))
                if ctx is not None
                else dict(base_meta)
            )
            compression_box = _campaign_compression_box(overlay)
            rows.append(
                {
                    "window": window,
                    "trade_date": trade_day.isoformat(),
                    "symbol": candidate.symbol,
                    "sector": candidate.sector,
                    "structural_source_rank": rank,
                    "structural_source_role": "active" if candidate.symbol in active_symbols else "overflow",
                    "frontier_rank": rank,
                    "frontier_role": "initial_active" if candidate.symbol in active_symbols else "frontier_shadow",
                    "selection_score": candidate.selection_score,
                    "relative_strength_pct": float(overlay.get("relative_strength_pct") or overlay.get("rs_percentile") or candidate.rs_percentile or 0.0),
                    "stock_vs_universe_strength": float(overlay.get("stock_vs_universe_strength") or 0.0),
                    "sector_daily_score_pct": float(overlay.get("sector_daily_score_pct") or 0.0),
                    "sector_participation": float(overlay.get("sector_participation") or overlay.get("sector_daily_participation") or 0.0),
                    "structural_campaign_score": float(overlay.get("structural_campaign_score") or 0.0),
                    "first30_confirmation_score": float(overlay.get("first30_confirmation_score") or 0.0),
                    "campaign_state": str(overlay.get("campaign_state") or ""),
                    "campaign_box_high": float(overlay.get("campaign_box_high") or 0.0),
                    "campaign_box_low": float(overlay.get("campaign_box_low") or 0.0),
                    "campaign_box_mid": float(overlay.get("campaign_box_mid") or 0.0),
                    "campaign_box_range_pct": float(overlay.get("campaign_box_range_pct") or 0.0),
                    "campaign_box_containment": float(overlay.get("campaign_box_containment") or 0.0),
                    "campaign_box_atr_ratio": float(compression_box.get("atr_ratio") or 0.0),
                    "campaign_box_squeeze_pct": float(compression_box.get("squeeze_pct") or 0.0),
                    "campaign_box_tier": str(compression_box.get("tier") or ""),
                    "campaign_avwap": float(overlay.get("campaign_avwap") or 0.0),
                    "campaign_avwap_anchor_available": bool(overlay.get("campaign_avwap_anchor_available")),
                    "campaign_breakout_level": float(overlay.get("campaign_breakout_level") or 0.0),
                    "campaign_breakout_displacement": float(overlay.get("campaign_breakout_displacement") or 0.0),
                    "first30_rel_volume": float(overlay.get("first30_rel_volume") or 0.0),
                    "first30_signal_cpr": float(overlay.get("first30_signal_cpr") or 0.0),
                    "first30_vwap_ret": float(overlay.get("first30_vwap_ret") or 0.0),
                    "first30_gap": float(overlay.get("first30_gap") or 0.0),
                    "first30_low_vs_prev_close": float(overlay.get("first30_low_vs_prev_close") or 0.0),
                    "first30_breakout_confirmation": bool(overlay.get("first30_breakout_confirmation")),
                    "first30_breakout_acceptance": bool(overlay.get("first30_breakout_acceptance")),
                    "first30_breakout_acceptance_closes": int(overlay.get("first30_breakout_acceptance_closes") or 0),
                    "first30_above_campaign_avwap": bool(overlay.get("first30_above_campaign_avwap")),
                    "campaign_avwap_distance_pct": float(overlay.get("campaign_avwap_distance_pct") or 0.0),
                    "campaign_box_high_distance_pct": float(overlay.get("campaign_box_high_distance_pct") or 0.0),
                    "campaign_metadata": _campaign_metadata_artifact(overlay),
                    "score_uses_ex_post_labels": False,
                }
            )
    rows.sort(key=lambda item: (str(item.get("trade_date") or ""), int(item.get("structural_source_rank") or 0), str(item.get("symbol") or "")))
    return rows


def _lazy_first30_context(dataset: KALCBFirst30Dataset, trade_day: date, symbol: str) -> Any | None:
    daily = daily_feature(dataset, symbol, trade_day)
    bars = tuple(dataset.bars_by_key.get((trade_day, symbol), ()))
    if daily is None or not bars:
        return None
    first30 = shared_first30_feature(daily, bars)
    if first30 is None:
        return None
    intraday = SimpleNamespace(
        open=first30.open,
        high=first30.high,
        low=first30.low,
        close=first30.close,
        vwap=first30.vwap,
        volume=first30.volume,
        expected_30m_volume=first30.expected_30m_volume,
    )
    return SimpleNamespace(
        day=trade_day,
        symbol=str(symbol).zfill(6),
        sector=dataset.sector_map.get(str(symbol).zfill(6), "UNKNOWN"),
        daily=daily,
        intraday=intraday,
        bars=bars,
        rel_volume=first30.rel_volume,
        close_location=first30.range_close_location,
        gap=first30.gap,
        low_vs_prev_close=first30.low_vs_prev_close,
        vwap_ret=first30.vwap_ret,
    )


def attach_causal_calibration_scores(
    feature_rows: Iterable[dict[str, Any]],
    causal_rows: Iterable[dict[str, Any]] | None,
    *,
    score_source: str = "stage08_causal_candidate_ranker",
) -> list[dict[str, Any]]:
    """Attach causal calibration scores without copying ex-post label fields."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for source in causal_rows or ():
        row = dict(source)
        day = str(row.get("trade_date") or row.get("entry_date") or "")[:10]
        symbol = str(row.get("symbol") or "")
        if not day or not symbol:
            continue
        score = _optional_num(row.get("causal_ranker_score"))
        if score is None:
            score = _optional_num(row.get("causal_calibration_score"))
        if score is None:
            continue
        index[(day, symbol)] = {
            "causal_calibration_score": float(score),
            "causal_rank_in_day": int(_num(row.get("causal_rank_in_day"))),
            "causal_score_source": score_source,
            "causal_score_uses_ex_post_labels": False,
        }
    out: list[dict[str, Any]] = []
    for source in feature_rows:
        row = source if isinstance(source, dict) else dict(source)
        key = (str(row.get("trade_date") or "")[:10], str(row.get("symbol") or ""))
        overlay = index.get(key)
        if overlay:
            row.update(overlay)
        else:
            row.setdefault("causal_calibration_score", 0.0)
            row.setdefault("causal_rank_in_day", 0)
            row.setdefault("causal_score_source", "")
            row.setdefault("causal_score_uses_ex_post_labels", False)
        out.append(row)
    return out


def split_structural_campaign_pools(
    rows: Iterable[dict[str, Any]],
    active_budget_by_day: dict[str, int] | None,
    cfg: KALCBConfig,
    *,
    frontier_size: int | None = None,
    variant_name: str = "structural_active_budget",
    selector: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[str(row.get("trade_date") or "")].append(dict(row))
    out: list[dict[str, Any]] = []
    cap = int(frontier_size or cfg.research_structural_frontier_count or max(cfg.frontier_size, cfg.ws_budget, cfg.research_top_long_count))
    selector_spec = dict(selector or _selector_spec("structural_first30_causal_tiebreak"))
    for day, day_rows in sorted(by_day.items()):
        if active_budget_by_day is not None and day in active_budget_by_day:
            active_limit = max(0, int(active_budget_by_day.get(day) or 0))
        else:
            active_limit = max(0, int(cfg.ws_budget or 0))
        ordered = sorted(day_rows, key=lambda row: _selector_rank_key(row, selector_spec))
        active_symbols = {str(row.get("symbol") or "") for row in ordered[:active_limit]}
        for rank, row in enumerate(ordered[:cap], start=1):
            symbol = str(row.get("symbol") or "")
            role = "active" if symbol in active_symbols else "overflow"
            out.append(
                {
                    **row,
                    "pool_variant": variant_name,
                    "pool_rank": rank,
                    "pool_size": cap,
                    "pool_active": role == "active",
                    "selector_variant": str(selector_spec.get("name") or ""),
                    "active_rank_score": float(_selector_score(row, selector_spec)),
                    "active_budget_source": "incumbent_daily_active_count" if active_budget_by_day else "ws_budget",
                    "active_budget_for_day": active_limit,
                    "structural_source_role": role,
                    "frontier_role_for_replay": "initial_active" if role == "active" else "frontier_shadow",
                }
            )
    out.sort(key=lambda item: (str(item.get("trade_date") or ""), int(item.get("pool_rank") or 0), str(item.get("symbol") or "")))
    return out


def compile_structural_campaign_replay(
    pool_rows: Iterable[dict[str, Any]],
    dataset: KALCBFirst30Dataset,
    contexts: dict[date, tuple[First30Context, ...]],
    cfg: KALCBConfig,
):
    context_by_key = {(day, ctx.symbol): ctx for day, items in contexts.items() for ctx in items}
    selections: list[Selection] = []
    frontier_by_day: dict[date, list[str]] = defaultdict(list)
    frontier_scores_by_day: dict[date, dict[str, float]] = defaultdict(dict)
    candidate_metadata_by_key: dict[tuple[date, str], dict[str, Any]] = {}
    active_counts: dict[date, int] = defaultdict(int)
    for row in pool_rows:
        trade_day = date.fromisoformat(str(row.get("trade_date")))
        symbol = str(row.get("symbol") or "")
        score = _feature_num(row, "active_rank_score", default=float(row.get("structural_campaign_score") or 0.0) * 10.0 + float(row.get("first30_confirmation_score") or 0.0))
        frontier_by_day[trade_day].append(symbol)
        frontier_scores_by_day[trade_day][symbol] = score
        candidate_metadata_by_key[(trade_day, symbol)] = dict(row.get("campaign_metadata") or {})
        if bool(row.get("pool_active")):
            active_counts[trade_day] += 1
            selections.append(Selection(trade_day, symbol, score, "structural_campaign_surfacing"))
    return compile_core_replay(
        selections,
        dataset,
        context_by_key,
        dataset.trading_dates,
        {day: max(count, 0) for day, count in active_counts.items()},
        cfg,
        frontier_by_day={day: tuple(symbols) for day, symbols in frontier_by_day.items()},
        frontier_scores_by_day={day: dict(scores) for day, scores in frontier_scores_by_day.items()},
        candidate_metadata_by_key=candidate_metadata_by_key,
        source_calibration_metadata={"structural_campaign_surfacing": STRUCTURAL_CAMPAIGN_SURFACING_VERSION},
    )


def build_alcb_breakout_replay_artifacts(
    train_rows: Iterable[dict[str, Any]],
    holdout_rows: Iterable[dict[str, Any]],
    *,
    train_dataset: KALCBFirst30Dataset,
    holdout_dataset: KALCBFirst30Dataset,
    output_dir: str | Path,
    cfg: KALCBConfig,
    train_active_budget_by_day: dict[str, int] | None = None,
    holdout_active_budget_by_day: dict[str, int] | None = None,
    shortlist_size: int = ALCB_BREAKOUT_REPLAY_SHORTLIST_SIZE,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_features = [dict(row) for row in train_rows]
    holdout_features = [dict(row) for row in holdout_rows]
    train_context_by_key = _full_context_by_key(build_contexts(train_dataset))
    holdout_context_by_key = _full_context_by_key(build_contexts(holdout_dataset))
    variants = _alcb_breakout_replay_variants()
    train_rows_out = _evaluate_alcb_breakout_replay_variants(
        train_features,
        train_dataset,
        train_context_by_key,
        cfg,
        train_active_budget_by_day,
        variants,
        window="train",
        selection_basis="train_sweep",
    )
    shortlist = _select_alcb_breakout_replay_shortlist(train_rows_out, shortlist_size)
    holdout_rows_out = _evaluate_alcb_breakout_replay_variants(
        holdout_features,
        holdout_dataset,
        holdout_context_by_key,
        cfg,
        holdout_active_budget_by_day,
        [dict(row.get("variant_spec") or {}) for row in shortlist],
        window="holdout",
        selection_basis="frozen_train_shortlist_holdout_scored_once",
        frozen_train_rows=shortlist,
    )
    summary = {
        "version": ALCB_BREAKOUT_REPLAY_VERSION,
        "usage_contract": STRUCTURAL_CAMPAIGN_USAGE_CONTRACT,
        "replay_contract": "train_sweep_selects_shortlist_holdout_scored_once_no_holdout_optimization_no_oracle_live_features",
        "candidate_source": "fixed_stage09_structural_campaign_features",
        "objective": "promote_only_disciplined_positive_actual_replay_conversion_from_alcb_like_first30_campaign_breakout_surface",
        "train_variant_count": len(train_rows_out),
        "shortlist_size": len(shortlist),
        "train_rows": train_rows_out,
        "shortlist": shortlist,
        "holdout_scored_once_rows": holdout_rows_out,
        "best_train_variant": shortlist[0] if shortlist else {},
        "holdout_frozen_rank1_variant": holdout_rows_out[0] if holdout_rows_out else {},
        "best_holdout_frozen_variant": holdout_rows_out[0] if holdout_rows_out else {},
        "holdout_selection_basis": "reported_in_frozen_train_rank_order_no_holdout_sort",
    }
    paths = {
        "structural_campaign_alcb_breakout_replay_train_jsonl": out / "structural_campaign_alcb_breakout_replay_train.jsonl",
        "structural_campaign_alcb_breakout_replay_shortlist_json": out / "structural_campaign_alcb_breakout_replay_shortlist.json",
        "structural_campaign_alcb_breakout_replay_holdout_jsonl": out / "structural_campaign_alcb_breakout_replay_holdout.jsonl",
        "structural_campaign_alcb_breakout_replay_summary_json": out / "structural_campaign_alcb_breakout_replay_summary.json",
    }
    write_jsonl(paths["structural_campaign_alcb_breakout_replay_train_jsonl"], train_rows_out)
    paths["structural_campaign_alcb_breakout_replay_shortlist_json"].write_text(json.dumps(shortlist, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_jsonl(paths["structural_campaign_alcb_breakout_replay_holdout_jsonl"], holdout_rows_out)
    paths["structural_campaign_alcb_breakout_replay_summary_json"].write_text(json.dumps({key: value for key, value in summary.items() if key != "train_rows"}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    summary["artifact_paths"] = {key: str(path) for key, path in paths.items()}
    return summary


def build_alcb_faithfulness_funnel_artifacts(
    train_rows: Iterable[dict[str, Any]],
    holdout_rows: Iterable[dict[str, Any]],
    *,
    train_dataset: KALCBFirst30Dataset,
    holdout_dataset: KALCBFirst30Dataset,
    output_dir: str | Path,
    cfg: KALCBConfig,
    train_oracle_rows: Iterable[dict[str, Any]] | None = None,
    holdout_oracle_rows: Iterable[dict[str, Any]] | None = None,
    shortlist_size: int = ALCB_FAITHFULNESS_SHORTLIST_SIZE,
) -> dict[str, Any]:
    """Quantify where ALCB-style structural candidates are lost before PnL."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_features = [dict(row) for row in train_rows]
    holdout_features = [dict(row) for row in holdout_rows]
    train_oracles = [dict(row) for row in train_oracle_rows or ()]
    holdout_oracles = [dict(row) for row in holdout_oracle_rows or ()]
    train_context_by_key = _full_context_by_key(build_contexts(train_dataset))
    holdout_context_by_key = _full_context_by_key(build_contexts(holdout_dataset))
    variants = _alcb_faithfulness_variants()
    train_eval = _evaluate_alcb_faithfulness_funnel(
        train_features,
        train_dataset,
        train_context_by_key,
        cfg,
        train_oracles,
        variants,
        window="train",
        selection_basis="train_sweep",
    )
    shortlist = _select_alcb_faithfulness_shortlist(train_eval, shortlist_size)
    holdout_eval = _evaluate_alcb_faithfulness_funnel(
        holdout_features,
        holdout_dataset,
        holdout_context_by_key,
        cfg,
        holdout_oracles,
        (dict(row.get("variant_spec") or {}) for row in shortlist),
        window="holdout",
        selection_basis="frozen_train_shortlist_holdout_scored_once",
        frozen_train_rows=shortlist,
    )
    best_train = dict(shortlist[0]) if shortlist else {}
    best_holdout = dict(holdout_eval[0]) if holdout_eval else {}
    train_misses = _alcb_faithfulness_top1_misses(
        train_features,
        train_dataset,
        train_context_by_key,
        cfg,
        train_oracles,
        dict(best_train.get("variant_spec") or {}),
        window="train",
    )
    holdout_misses = _alcb_faithfulness_top1_misses(
        holdout_features,
        holdout_dataset,
        holdout_context_by_key,
        cfg,
        holdout_oracles,
        dict(best_train.get("variant_spec") or {}),
        window="holdout",
    )
    summary = {
        "version": ALCB_FAITHFULNESS_FUNNEL_VERSION,
        "usage_contract": STRUCTURAL_CAMPAIGN_USAGE_CONTRACT,
        "funnel_contract": "train_sweep_topN_tradable_pool_route_funnel_holdout_scored_only_from_frozen_train_shortlist",
        "purpose": "test_alcb_faithfulness_topN_structural_pools_or_pdh_combined_breakouts_campaign_levels_and_top1_miss_reasons",
        "route_scope": "lightweight_per_candidate_replay_uses_trade_plan_entry_exit_simulation_without_oracle_live_features",
        "train_variant_count": len(train_eval),
        "shortlist_size": len(shortlist),
        "best_train_variant": best_train,
        "holdout_frozen_rank1_variant": best_holdout,
        "shortlist": shortlist,
        "holdout_scored_once_rows": holdout_eval,
        "train_top1_miss_counts": _counts(row.get("miss_reason") for row in train_misses),
        "holdout_top1_miss_counts": _counts(row.get("miss_reason") for row in holdout_misses),
        "generated_at_utc": _utc_now_iso(),
    }
    paths = {
        "structural_campaign_alcb_faithfulness_funnel_train_jsonl": out / "structural_campaign_alcb_faithfulness_funnel_train.jsonl",
        "structural_campaign_alcb_faithfulness_funnel_shortlist_json": out / "structural_campaign_alcb_faithfulness_funnel_shortlist.json",
        "structural_campaign_alcb_faithfulness_funnel_holdout_jsonl": out / "structural_campaign_alcb_faithfulness_funnel_holdout.jsonl",
        "structural_campaign_alcb_faithfulness_top1_misses_train_jsonl": out / "structural_campaign_alcb_faithfulness_top1_misses_train.jsonl",
        "structural_campaign_alcb_faithfulness_top1_misses_holdout_jsonl": out / "structural_campaign_alcb_faithfulness_top1_misses_holdout.jsonl",
        "structural_campaign_alcb_faithfulness_funnel_summary_json": out / "structural_campaign_alcb_faithfulness_funnel_summary.json",
    }
    summary["artifact_paths"] = {key: str(path) for key, path in paths.items()}
    write_jsonl(paths["structural_campaign_alcb_faithfulness_funnel_train_jsonl"], train_eval)
    paths["structural_campaign_alcb_faithfulness_funnel_shortlist_json"].write_text(json.dumps(shortlist, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_jsonl(paths["structural_campaign_alcb_faithfulness_funnel_holdout_jsonl"], holdout_eval)
    write_jsonl(paths["structural_campaign_alcb_faithfulness_top1_misses_train_jsonl"], train_misses)
    write_jsonl(paths["structural_campaign_alcb_faithfulness_top1_misses_holdout_jsonl"], holdout_misses)
    paths["structural_campaign_alcb_faithfulness_funnel_summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary


def _alcb_faithfulness_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for top_n in ALCB_FAITHFULNESS_TOP_N_GRID:
        for selector_mode in ALCB_FAITHFULNESS_SELECTOR_MODES:
            for route_bundle in _alcb_faithfulness_route_bundles():
                for filters in ALCB_FAITHFULNESS_FILTER_GRID:
                    variants.append(_alcb_faithfulness_variant(top_n, selector_mode, route_bundle, filters))
    return variants


def _alcb_faithfulness_variant(top_n: int, selector_mode: str, route_bundle: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
    spec = {
        "top_n": int(top_n),
        "pool_variant": f"tradable_top{int(top_n)}",
        "selector_mode": str(selector_mode),
        "route_bundle": str(route_bundle["name"]),
        "route_bundle_description": str(route_bundle.get("description") or ""),
        "routes": [dict(route) for route in route_bundle["routes"]],
        "max_signal_bars": int(route_bundle.get("max_signal_bars") or 18),
        "min_first30_rel_volume": float(filters.get("min_first30_rel_volume") or 0.0),
        "min_first30_signal_cpr": float(filters.get("min_first30_signal_cpr") or 0.0),
        "require_first30_campaign_breakout_acceptance": bool(filters.get("require_first30_campaign_breakout_acceptance")),
        "exit": dict(route_bundle.get("exit") or {"name": "eod_atr", "stop_mode": "atr", "hard_stop_enabled": False}),
    }
    spec["variant_id"] = stable_signature(spec)[:20]
    spec["name"] = (
        f"{spec['pool_variant']}__{selector_mode}__{spec['route_bundle']}"
        f"__rv{_alcb_label_num(spec['min_first30_rel_volume'])}"
        f"_cpr{_alcb_label_num(spec['min_first30_signal_cpr'])}"
        f"_accept{int(spec['require_first30_campaign_breakout_acceptance'])}"
    )
    return spec


def _alcb_faithfulness_route_bundles() -> tuple[dict[str, Any], ...]:
    raw_routes = (
        {"name": "combined_breakout", "mode": "combined_breakout", "priority": 0},
        {"name": "or_breakout", "mode": "or_breakout", "priority": 1},
        {"name": "pdh_breakout", "mode": "pdh_breakout", "priority": 2},
    )
    campaign_routes = (
        {"name": "campaign_breakout_two_close", "mode": "or_high_reclaim", "level_source": "campaign_breakout_level", "min_reclaim_closes": 2, "min_reclaim_ret": 0.0, "max_pullback_from_vwap_pct": 0.004},
        {"name": "campaign_box_high_pullback", "mode": "pullback_acceptance", "level_source": "campaign_box_high", "min_reclaim_closes": 1, "min_reclaim_ret": 0.0, "max_pullback_from_vwap_pct": 0.006},
        {"name": "campaign_avwap_retest", "mode": "avwap_reclaim", "level_source": "campaign_avwap", "min_reclaim_closes": 1, "min_reclaim_ret": 0.0, "max_pullback_from_vwap_pct": 0.006},
    )
    return (
        {"name": "raw_or_pdh_combined_18", "description": "OR/PDH/combined breakout through 18 post-first30 bars.", "routes": raw_routes, "max_signal_bars": 18},
        {"name": "raw_or_pdh_combined_36", "description": "ALCB-like longer entry window for OR/PDH/combined breakout.", "routes": raw_routes, "max_signal_bars": 36},
        {"name": "campaign_levels_36", "description": "Campaign AVWAP, box-high, and breakout-level retest/reclaim routes.", "routes": campaign_routes, "max_signal_bars": 36},
        {"name": "full_alcb_stack_36", "description": "Raw breakout routes plus campaign-level reclaim/retest routes.", "routes": (*raw_routes, *campaign_routes), "max_signal_bars": 36},
    )


def _evaluate_alcb_faithfulness_funnel(
    feature_rows: list[dict[str, Any]],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    oracle_rows: list[dict[str, Any]],
    variants: Iterable[dict[str, Any]],
    *,
    window: str,
    selection_basis: str,
    frozen_train_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    oracle = _best_oracle_by_key(oracle_rows)
    best_by_day = _best_oracle_by_day(oracle.values())
    top_decile = _top_decile_oracle_keys(oracle.values())
    diagnostic_top1_by_mode = {
        mode: _diagnostic_top1_by_day_from_rows(feature_rows, mode)
        for mode in ALCB_FAITHFULNESS_TOP1_MODES
    }
    pool_cache: dict[tuple[int, str], list[dict[str, Any]]] = {}
    route_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    frozen_by_id = {str(row.get("variant_id") or ""): row for row in frozen_train_rows or []}
    rows_out: list[dict[str, Any]] = []
    for variant in variants:
        spec = dict(variant)
        pool_key = (int(spec.get("top_n") or 0), str(spec.get("selector_mode") or "structural_first30"))
        if pool_key not in pool_cache:
            pool_cache[pool_key] = _alcb_topn_pool_rows(feature_rows, top_n=pool_key[0], selector_mode=pool_key[1])
        pool = pool_cache[pool_key]
        row = _alcb_faithfulness_result_row(
            spec,
            pool,
            dataset,
            context_by_key,
            cfg,
            oracle,
            best_by_day,
            top_decile,
            diagnostic_top1_by_mode,
            route_cache,
            window=window,
            selection_basis=selection_basis,
        )
        frozen = frozen_by_id.get(str(row.get("variant_id") or ""))
        if frozen:
            row["frozen_train_rank"] = frozen.get("train_rank")
            row["frozen_train_funnel_score"] = frozen.get("funnel_score")
        rows_out.append(row)
    if selection_basis == "train_sweep":
        rows_out.sort(key=_alcb_faithfulness_sort_key)
        for rank, row in enumerate(rows_out, start=1):
            row["train_rank"] = rank
    else:
        rows_out.sort(key=lambda row: (int(row.get("frozen_train_rank") or 9999), str(row.get("variant_id") or "")))
    return rows_out


def _alcb_topn_pool_rows(feature_rows: list[dict[str, Any]], *, top_n: int, selector_mode: str) -> list[dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selector = _selector_spec(selector_mode)
    for row in feature_rows:
        by_day[str(row.get("trade_date") or "")].append(dict(row))
    out: list[dict[str, Any]] = []
    for day, rows in sorted(by_day.items()):
        ordered = sorted(rows, key=lambda item: _selector_rank_key(item, selector))
        for rank, row in enumerate(ordered[: max(0, int(top_n))], start=1):
            out.append(
                {
                    **row,
                    "pool_variant": f"tradable_top{int(top_n)}",
                    "pool_rank": rank,
                    "candidate_rank": rank,
                    "frontier_rank": rank,
                    "frontier_role": "initial_active",
                    "frontier_initial_active": True,
                    "selector_variant": str(selector.get("name") or selector_mode),
                    "active_rank_score": _selector_score(row, selector),
                }
            )
    return out


def _alcb_faithfulness_result_row(
    variant: dict[str, Any],
    pool_rows: list[dict[str, Any]],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    oracle: dict[tuple[str, str], dict[str, Any]],
    best_by_day: dict[str, dict[str, Any]],
    top_decile: set[tuple[str, str]],
    diagnostic_top1_by_mode: dict[str, dict[str, dict[str, Any]]],
    route_cache: dict[tuple[Any, ...], dict[str, Any]],
    *,
    window: str,
    selection_basis: str,
) -> dict[str, Any]:
    selected_keys = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in pool_rows}
    outcomes: list[Any] = []
    result_counts: dict[str, int] = {}
    reject_reasons: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    entry_counts: dict[str, int] = {}
    for row in pool_rows:
        result = _alcb_faithfulness_route_result(row, dataset, context_by_key, cfg, variant, route_cache)
        stage = str(result.get("stage") or "unknown")
        result_counts[stage] = result_counts.get(stage, 0) + 1
        if stage == "traded":
            outcome = result.get("outcome")
            if outcome is not None:
                outcomes.append(outcome)
                route_counts[str(result.get("route_name") or "")] = route_counts.get(str(result.get("route_name") or ""), 0) + 1
                entry_counts[str(result.get("entry_reason") or "")] = entry_counts.get(str(result.get("entry_reason") or ""), 0) + 1
        else:
            reason = str(result.get("reason") or stage)
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    labeled = [oracle[key] for key in selected_keys if key in oracle]
    top1_summary = _alcb_top1_classification_summary(diagnostic_top1_by_mode, pool_rows, dataset, context_by_key, cfg, oracle, variant, route_cache)
    trade_rs = [_outcome_r(outcome) for outcome in outcomes]
    positive = [value for value in trade_rs if value > 0.0]
    negative = [value for value in trade_rs if value <= 0.0]
    row = {
        "version": ALCB_FAITHFULNESS_FUNNEL_VERSION,
        "window": window,
        "selection_basis": selection_basis,
        "variant_id": str(variant.get("variant_id") or ""),
        "variant_name": str(variant.get("name") or ""),
        "variant_spec": variant,
        "pool_variant": str(variant.get("pool_variant") or ""),
        "top_n": int(variant.get("top_n") or 0),
        "selector_mode": str(variant.get("selector_mode") or ""),
        "route_bundle": str(variant.get("route_bundle") or ""),
        "min_first30_rel_volume": float(variant.get("min_first30_rel_volume") or 0.0),
        "min_first30_signal_cpr": float(variant.get("min_first30_signal_cpr") or 0.0),
        "require_first30_campaign_breakout_acceptance": bool(variant.get("require_first30_campaign_breakout_acceptance")),
        "pool_candidate_count": len(pool_rows),
        "pool_day_count": len({str(row.get("trade_date") or "") for row in pool_rows}),
        "avg_pool_size": _avg_day_count(pool_rows),
        "oracle_labeled_count": len(labeled),
        "avg_oracle_score": _avg(_oracle_rank_tuple(row)[0] for row in labeled),
        "avg_oracle_mfe_r": _avg(row.get("mfe_r") for row in labeled),
        "avg_oracle_net_r": _avg(row.get("net_r") for row in labeled),
        "best_oracle_in_pool_share": sum(1 for day, oracle_row in best_by_day.items() if (day, str(oracle_row.get("symbol") or "")) in selected_keys) / max(len(best_by_day), 1),
        "top_decile_oracle_recall": len(top_decile & selected_keys) / max(len(top_decile), 1),
        "trade_count": len(outcomes),
        "positive_trade_count": len(positive),
        "negative_trade_count": len(negative),
        "simulated_total_r": float(sum(trade_rs)),
        "simulated_avg_r": mean(trade_rs) if trade_rs else 0.0,
        "simulated_net_return_sum_pct": float(sum(float(getattr(outcome, "net_return_pct", 0.0)) for outcome in outcomes)),
        "avg_mfe_r": _avg(getattr(outcome, "mfe_r", 0.0) for outcome in outcomes),
        "avg_mfe_capture": _avg(getattr(outcome, "mfe_capture", 0.0) for outcome in outcomes),
        "result_counts": dict(sorted(result_counts.items())),
        "top_route_reject_reasons": [{"reason": key, "count": value} for key, value in sorted(reject_reasons.items(), key=lambda item: (-item[1], item[0]))[:12]],
        "route_trade_counts": dict(sorted(route_counts.items())),
        "entry_reason_counts": dict(sorted(entry_counts.items())),
        "top1_classification": top1_summary,
    }
    row["funnel_score"] = _alcb_faithfulness_score(row)
    row["rejected"] = bool(len(outcomes) < 5 or float(row["simulated_total_r"]) <= 0.0)
    row["reject_reason"] = "too_few_trades" if len(outcomes) < 5 else "non_positive_total_r" if float(row["simulated_total_r"]) <= 0.0 else ""
    return row


def _alcb_faithfulness_route_result(
    row: dict[str, Any],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    variant: dict[str, Any],
    cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    day_label = str(row.get("trade_date") or "")
    symbol = str(row.get("symbol") or "")
    cache_key = (
        day_label,
        symbol,
        str(variant.get("route_bundle") or ""),
        float(variant.get("min_first30_rel_volume") or 0.0),
        float(variant.get("min_first30_signal_cpr") or 0.0),
        bool(variant.get("require_first30_campaign_breakout_acceptance")),
        int(variant.get("max_signal_bars") or 0),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    filter_reason = _alcb_candidate_filter_reason(row, variant)
    if filter_reason:
        result = {"stage": "route_rejected", "reason": filter_reason}
        cache[cache_key] = result
        return result
    if not day_label or not symbol:
        result = {"stage": "route_rejected", "reason": "missing_day_or_symbol"}
        cache[cache_key] = result
        return result
    trade_day = date.fromisoformat(day_label)
    ctx = context_by_key.get((trade_day, symbol))
    bars = dataset.bars_by_key.get((trade_day, symbol), ())
    if ctx is None:
        result = {"stage": "route_rejected", "reason": "missing_first30_context"}
        cache[cache_key] = result
        return result
    ordered_bars = tuple(bar for bar in sorted(bars, key=lambda item: item.timestamp) if bar.timestamp.astimezone(KST).time() <= cfg.flatten_time)
    if not ordered_bars:
        result = {"stage": "route_rejected", "reason": "missing_intraday_bars"}
        cache[cache_key] = result
        return result
    prior_day_high = _dataset_prior_day_high(dataset, symbol, trade_day)
    metadata = _alcb_replay_candidate_metadata(row)
    exit_spec = _alcb_faithfulness_exit_spec(variant)
    no_fill = False
    first_reject = ""
    attempts: list[dict[str, Any]] = []
    for route in _alcb_faithfulness_entry_specs(variant):
        level_source = str(route.reclaim_level_source or "legacy")
        if level_source != "legacy" and _route_level(row, level_source) <= 0.0:
            reason = f"missing_{level_source}"
            first_reject = first_reject or reason
            attempts.append({"route": route.name, "reason": reason})
            continue
        signal = find_entry_signal(ordered_bars, ctx, route, cfg, prior_day_high=prior_day_high, candidate_metadata=metadata)
        if signal is None:
            first_reject = first_reject or "no_entry_signal"
            attempts.append({"route": route.name, "reason": "no_entry_signal"})
            continue
        if signal.fill_index >= len(ordered_bars):
            no_fill = True
            attempts.append({"route": route.name, "reason": "no_next_bar_fill"})
            continue
        outcome = simulate_trade(
            trade_day,
            symbol,
            ordered_bars,
            ctx,
            route,
            exit_spec,
            cfg,
            prior_day_high=prior_day_high,
            candidate_metadata=metadata,
        )
        if outcome is None:
            no_fill = True
            attempts.append({"route": route.name, "reason": "simulate_trade_no_fill"})
            continue
        result = {
            "stage": "traded",
            "reason": "traded",
            "route_name": route.name,
            "entry_reason": signal.reason,
            "outcome": outcome,
            "outcome_r": _outcome_r(outcome),
            "attempts": attempts,
        }
        cache[cache_key] = result
        return result
    result = {"stage": "no_fill" if no_fill else "route_rejected", "reason": "no_next_bar_fill" if no_fill else (first_reject or "no_entry_signal"), "attempts": attempts}
    cache[cache_key] = result
    return result


def _alcb_candidate_filter_reason(row: dict[str, Any], variant: dict[str, Any]) -> str:
    min_rvol = float(variant.get("min_first30_rel_volume") or 0.0)
    min_cpr = float(variant.get("min_first30_signal_cpr") or 0.0)
    if _feature_num(row, "first30_rel_volume") < min_rvol:
        return "first30_rel_volume_below_min"
    if _feature_num(row, "first30_signal_cpr", "first30_signal_bar_cpr") < min_cpr:
        return "first30_signal_cpr_below_min"
    if bool(variant.get("require_first30_campaign_breakout_acceptance")) and not (
        bool(row.get("first30_breakout_acceptance")) or bool(row.get("first30_breakout_confirmation"))
    ):
        return "missing_first30_campaign_breakout_acceptance"
    return ""


def _alcb_faithfulness_entry_specs(variant: dict[str, Any]) -> list[EntrySpec]:
    max_bars = int(variant.get("max_signal_bars") or 18)
    min_cpr = float(variant.get("min_first30_signal_cpr") or 0.0)
    out: list[EntrySpec] = []
    for route in variant.get("routes") or ():
        item = dict(route)
        out.append(
            EntrySpec(
                name=str(item.get("name") or item.get("mode") or ""),
                mode=str(item.get("mode") or "breakout"),
                max_signal_bars=int(item.get("max_signal_bars") or max_bars),
                min_close_location=float(item.get("min_close_location", min_cpr) or 0.0),
                min_breakout_pct=float(item.get("min_breakout_pct") or 0.0),
                max_pullback_from_vwap_pct=float(item.get("max_pullback_from_vwap_pct", 0.01) or 0.0),
                min_reclaim_ret=float(item.get("min_reclaim_ret", -9.99)),
                min_reclaim_closes=int(item.get("min_reclaim_closes") or 1),
                reclaim_level_source=str(item.get("level_source") or item.get("reclaim_level_source") or "legacy"),
            )
        )
    return out


def _alcb_faithfulness_exit_spec(variant: dict[str, Any]) -> ExitSpec:
    exit_spec = dict(variant.get("exit") or {})
    return ExitSpec(
        name=str(exit_spec.get("name") or "eod_atr"),
        stop_mode=str(exit_spec.get("stop_mode") or "atr"),
        hard_stop_enabled=bool(exit_spec.get("hard_stop_enabled", False)),
        stop_atr_mult=float(exit_spec.get("stop_atr_mult", 0.8) or 0.8),
        stop_pct=float(exit_spec.get("stop_pct", 0.006) or 0.006),
    )


def _dataset_prior_day_high(dataset: KALCBFirst30Dataset, symbol: str, trade_day: date) -> float:
    rows = dataset.daily_by_symbol.get(str(symbol), ()) or dataset.daily_by_symbol.get(str(symbol).zfill(6), ())
    prior: list[dict[str, Any]] = []
    for row in rows:
        row_date = _coerce_row_date(row)
        if row_date is not None and row_date < trade_day:
            prior.append(row)
    if not prior:
        return 0.0
    latest = max(prior, key=lambda item: _coerce_row_date(item) or date.min)
    return float(latest.get("high") or 0.0)


def _coerce_row_date(row: dict[str, Any]) -> date | None:
    value = row.get("date") or row.get("trade_date") or row.get("timestamp")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _outcome_r(outcome: Any) -> float:
    entry = max(float(getattr(outcome, "entry_price", 0.0) or 0.0), 1e-9)
    risk = max(float(getattr(outcome, "risk_per_share", 0.0) or 0.0), 1e-9)
    return float(getattr(outcome, "net_return_pct", 0.0) or 0.0) * entry / risk


def _alcb_top1_classification_summary(
    diagnostic_top1_by_mode: dict[str, dict[str, dict[str, Any]]],
    pool_rows: list[dict[str, Any]],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    oracle: dict[tuple[str, str], dict[str, Any]],
    variant: dict[str, Any],
    route_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    selected = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in pool_rows}
    rows_by_key = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")): row for row in pool_rows}
    out: dict[str, Any] = {}
    for mode in ALCB_FAITHFULNESS_TOP1_MODES:
        counts: dict[str, int] = {}
        total_oracle = 0.0
        captured_r = 0.0
        for top in diagnostic_top1_by_mode.get(mode, {}).values():
            key = (str(top.get("trade_date") or ""), str(top.get("symbol") or ""))
            oracle_row = oracle.get(key)
            if oracle_row is not None:
                total_oracle += _num(oracle_row.get("net_r"))
            if key not in selected:
                reason = "not_selected"
            else:
                result = _alcb_faithfulness_route_result(rows_by_key[key], dataset, context_by_key, cfg, variant, route_cache)
                reason = _alcb_top1_reason_from_route_result(result)
                if reason == "captured":
                    captured_r += _num(result.get("outcome_r"))
            counts[reason] = counts.get(reason, 0) + 1
        out[mode] = {
            "day_count": sum(counts.values()),
            "miss_counts": dict(sorted(counts.items())),
            "oracle_top1_net_r_sum": float(total_oracle),
            "captured_top1_r_sum": float(captured_r),
            "captured_vs_oracle_r": captured_r / total_oracle if abs(total_oracle) > 1e-9 else 0.0,
        }
    return out


def _alcb_faithfulness_top1_misses(
    feature_rows: list[dict[str, Any]],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    oracle_rows: list[dict[str, Any]],
    variant: dict[str, Any],
    *,
    window: str,
) -> list[dict[str, Any]]:
    if not variant:
        return []
    oracle = _best_oracle_by_key(oracle_rows)
    selected_pool = _alcb_topn_pool_rows(feature_rows, top_n=int(variant.get("top_n") or 0), selector_mode=str(variant.get("selector_mode") or "structural_first30"))
    selected_by_key = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")): row for row in selected_pool}
    feature_by_key = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")): row for row in feature_rows}
    route_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    misses: list[dict[str, Any]] = []
    for mode in ALCB_FAITHFULNESS_TOP1_MODES:
        for day, top in _diagnostic_top1_by_day_from_rows(feature_rows, mode).items():
            key = (day, str(top.get("symbol") or ""))
            oracle_row = oracle.get(key)
            if key not in selected_by_key:
                reason = "not_selected"
                result: dict[str, Any] = {}
                selected_rank = 0
            else:
                selected_rank = int(_num(selected_by_key[key].get("pool_rank")))
                result = _alcb_faithfulness_route_result(selected_by_key[key], dataset, context_by_key, cfg, variant, route_cache)
                reason = _alcb_top1_reason_from_route_result(result)
            if reason == "captured":
                continue
            feature = feature_by_key.get(key, top)
            misses.append(
                {
                    "version": ALCB_FAITHFULNESS_FUNNEL_VERSION,
                    "window": window,
                    "variant_id": str(variant.get("variant_id") or ""),
                    "variant_name": str(variant.get("name") or ""),
                    "diagnostic_mode": mode,
                    "trade_date": day,
                    "symbol": key[1],
                    "miss_reason": reason,
                    "route_reject_reason": str(result.get("reason") or ""),
                    "selected_rank": selected_rank,
                    "top_n": int(variant.get("top_n") or 0),
                    "selector_mode": str(variant.get("selector_mode") or ""),
                    "route_bundle": str(variant.get("route_bundle") or ""),
                    "oracle_score": _oracle_rank_tuple(oracle_row)[0] if oracle_row else 0.0,
                    "oracle_net_r": _num((oracle_row or {}).get("net_r")),
                    "oracle_mfe_r": _num((oracle_row or {}).get("mfe_r")),
                    "simulated_r": _num(result.get("outcome_r")),
                    "structural_source_rank": int(_num(feature.get("structural_source_rank"))),
                    "structural_campaign_score": _num(feature.get("structural_campaign_score")),
                    "first30_confirmation_score": _num(feature.get("first30_confirmation_score")),
                    "first30_rel_volume": _num(feature.get("first30_rel_volume")),
                    "first30_signal_cpr": _num(feature.get("first30_signal_cpr")),
                    "campaign_state": str(feature.get("campaign_state") or ""),
                    "campaign_avwap": _num(feature.get("campaign_avwap")),
                    "campaign_box_high": _num(feature.get("campaign_box_high")),
                    "campaign_breakout_level": _num(feature.get("campaign_breakout_level")),
                    "route_attempts": result.get("attempts") or [],
                }
            )
    misses.sort(key=lambda row: (str(row.get("diagnostic_mode") or ""), str(row.get("trade_date") or "")))
    return misses


def _diagnostic_top1_by_day_from_rows(rows: list[dict[str, Any]], selector_mode: str) -> dict[str, dict[str, Any]]:
    by_day = _group_rows_by_day(rows)
    selector = _selector_spec(selector_mode)
    return {day: sorted(items, key=lambda row: _selector_rank_key(row, selector))[0] for day, items in by_day.items() if items}


def _group_rows_by_day(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("trade_date") or "")].append(dict(row))
    return out


def _alcb_top1_reason_from_route_result(result: dict[str, Any]) -> str:
    stage = str(result.get("stage") or "")
    if stage == "route_rejected":
        return "route_rejected"
    if stage == "no_fill":
        return "no_fill"
    if stage == "traded":
        return "captured" if _num(result.get("outcome_r")) > 0.0 else "bad_exit"
    return "route_rejected"


def _alcb_faithfulness_score(row: dict[str, Any]) -> float:
    total_r = _num(row.get("simulated_total_r"))
    avg_r = _num(row.get("simulated_avg_r"))
    trade_count = _num(row.get("trade_count"))
    capture = _num(row.get("avg_mfe_capture"))
    best_share = _num(row.get("best_oracle_in_pool_share"))
    top_decile = _num(row.get("top_decile_oracle_recall"))
    top1 = dict(row.get("top1_classification") or {})
    miss_penalty = 0.0
    for payload in top1.values():
        counts = dict(payload.get("miss_counts") or {}) if isinstance(payload, dict) else {}
        miss_penalty += 0.30 * int(counts.get("not_selected", 0))
        miss_penalty += 0.20 * int(counts.get("route_rejected", 0))
        miss_penalty += 0.15 * int(counts.get("no_fill", 0))
        miss_penalty += 0.10 * int(counts.get("bad_exit", 0))
    frequency_penalty = max(0.0, trade_count / max(_num(row.get("pool_day_count")), 1.0) - 4.0) * 4.0
    return total_r + 2.0 * avg_r + 5.0 * capture + 12.0 * best_share + 8.0 * top_decile - miss_penalty - frequency_penalty


def _select_alcb_faithfulness_shortlist(rows: list[dict[str, Any]], shortlist_size: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_alcb_faithfulness_sort_key)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        key = f"{row.get('top_n')}:{row.get('selector_mode')}:{row.get('route_bundle')}"
        if key in seen and len(out) < max(1, int(shortlist_size)) - 1:
            continue
        seen.add(key)
        out.append(dict(row))
        if len(out) >= max(1, int(shortlist_size)):
            break
    return out


def _alcb_faithfulness_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row.get("rejected")),
        -_num(row.get("funnel_score")),
        -_num(row.get("simulated_total_r")),
        -_num(row.get("trade_count")),
        -_num(row.get("best_oracle_in_pool_share")),
        str(row.get("variant_id") or ""),
    )


def _alcb_breakout_replay_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    active_selector_modes = (
        "structural_first30_causal_tiebreak",
        "first30_confirmation",
        "blend_struct70_first3020_causal10",
        "blend_struct60_first3030_causal10",
        "blend_struct70_causal30",
    )
    active_pool = {"pool_variant": "active_only", "frontier_size": None, "frontier_branch": False}
    for min_rvol in (2.0, 3.0):
        for selector_mode in active_selector_modes:
            variants.append(_alcb_breakout_replay_variant(active_pool, route_family="first30_open_proxy", min_rvol=min_rvol, min_cpr=0.60, max_signal_bars=1, selector_mode=selector_mode))
    variants.append(_alcb_breakout_replay_variant(active_pool, route_family="breakout_family", min_rvol=2.0, min_cpr=0.60, max_signal_bars=18))
    variants.append(_alcb_breakout_replay_variant(active_pool, route_family="campaign_or_high_reclaim", min_rvol=2.0, min_cpr=0.60, max_signal_bars=18))
    frontier_pool = {"pool_variant": "frontier40_branch", "frontier_size": 40, "frontier_branch": True}
    for selector_mode in ("first30_confirmation", "blend_struct60_first3030_causal10"):
        variants.append(_alcb_breakout_replay_variant(frontier_pool, route_family="first30_open_proxy", min_rvol=2.0, min_cpr=0.60, max_signal_bars=1, selector_mode=selector_mode))
    return variants


def _alcb_breakout_replay_variant(
    pool: dict[str, Any],
    *,
    route_family: str,
    min_rvol: float,
    min_cpr: float,
    max_signal_bars: int,
    selector_mode: str = "structural_first30_causal_tiebreak",
) -> dict[str, Any]:
    spec = {
        **dict(pool),
        "route_family": route_family,
        "selector_mode": selector_mode,
        "min_first30_rel_volume": float(min_rvol),
        "min_first30_signal_cpr": float(min_cpr),
        "max_signal_bars": int(max_signal_bars),
        "max_session_trades_per_route": 1,
        "require_first30_campaign_breakout_acceptance": True,
        "exit": {"name": "eod_atr", "stop_mode": "atr", "hard_stop_enabled": False},
    }
    spec["variant_id"] = stable_signature(spec)[:20]
    spec["name"] = (
        f"{spec['pool_variant']}__{route_family}"
        f"__{selector_mode}"
        f"__rv{_alcb_label_num(min_rvol)}_cpr{_alcb_label_num(min_cpr)}_bars{int(max_signal_bars)}"
    )
    return spec


def _evaluate_alcb_breakout_replay_variants(
    feature_rows: list[dict[str, Any]],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    active_budget_by_day: dict[str, int] | None,
    variants: Iterable[dict[str, Any]],
    *,
    window: str,
    selection_basis: str,
    frozen_train_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    compiled_cache: dict[tuple[Any, ...], Any] = {}
    frozen_by_id = {str(row.get("variant_id") or ""): row for row in frozen_train_rows or []}
    out: list[dict[str, Any]] = []
    for variant in variants:
        spec = dict(variant)
        if not spec:
            continue
        pool_key = (
            str(spec.get("pool_variant") or ""),
            int(spec.get("frontier_size") or 0),
            bool(spec.get("frontier_branch")),
            float(spec.get("min_first30_rel_volume") or 0.0),
            float(spec.get("min_first30_signal_cpr") or 0.0),
            bool(spec.get("require_first30_campaign_breakout_acceptance", True)),
            str(spec.get("selector_mode") or "structural_first30_causal_tiebreak"),
        )
        if pool_key not in compiled_cache:
            compiled_cache[pool_key] = _compile_alcb_breakout_replay_pool(
                feature_rows,
                dataset,
                context_by_key,
                cfg,
                active_budget_by_day,
                pool_variant=pool_key[0],
                frontier_size=pool_key[1] or None,
                frontier_branch=pool_key[2],
                min_first30_rel_volume=pool_key[3],
                min_first30_signal_cpr=pool_key[4],
                require_first30_campaign_breakout_acceptance=pool_key[5],
                selector_mode=pool_key[6],
            )
        compiled = compiled_cache[pool_key]
        if compiled is None or not compiled.snapshots:
            row = _alcb_replay_empty_row(spec, window, selection_basis, "empty_candidate_pool")
        else:
            trade_spec = _alcb_trade_plan_spec(spec)
            replay_cfg = _alcb_replay_config(cfg, spec)
            outcomes, metrics, digest = _core_outcomes_metrics_digest(
                trade_spec,
                compiled,
                replay_cfg,
                tuple(dataset.trading_dates),
                compiled.selection_counts,
                audit=False,
            )
            row = _alcb_replay_result_row(spec, window, selection_basis, trade_spec, outcomes, metrics, digest, compiled)
        frozen = frozen_by_id.get(str(row.get("variant_id") or ""))
        if frozen:
            row["frozen_train_rank"] = frozen.get("train_rank")
            row["frozen_train_replay_score"] = frozen.get("replay_score")
        out.append(row)
    if selection_basis == "train_sweep":
        out.sort(key=_alcb_replay_sort_key)
        for rank, row in enumerate(out, start=1):
            row["train_rank"] = rank
    else:
        out.sort(key=lambda row: (int(row.get("frozen_train_rank") or 9999), str(row.get("variant_id") or "")))
    return out


def _compile_alcb_breakout_replay_pool(
    feature_rows: list[dict[str, Any]],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    active_budget_by_day: dict[str, int] | None,
    *,
    pool_variant: str,
    frontier_size: int | None,
    frontier_branch: bool,
    min_first30_rel_volume: float,
    min_first30_signal_cpr: float,
    require_first30_campaign_breakout_acceptance: bool,
    selector_mode: str,
):
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_day[str(row.get("trade_date") or "")].append(dict(row))
    pool_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    frontier_by_day: dict[date, tuple[str, ...]] = {}
    frontier_scores_by_day: dict[date, dict[str, float]] = {}
    metadata_by_key: dict[tuple[date, str], dict[str, Any]] = {}
    selector = _selector_spec(selector_mode)
    for day_label, rows in sorted(by_day.items()):
        ordered = sorted(rows, key=lambda row: _selector_rank_key(row, selector))
        active_limit = max(0, int((active_budget_by_day or {}).get(day_label, cfg.ws_budget) if active_budget_by_day is not None else cfg.ws_budget))
        day_active = [
            row
            for row in ordered[:active_limit]
            if _alcb_replay_candidate_passes_proxy(
                row,
                min_first30_rel_volume=min_first30_rel_volume,
                min_first30_signal_cpr=min_first30_signal_cpr,
                require_first30_campaign_breakout_acceptance=require_first30_campaign_breakout_acceptance,
            )
        ]
        raw_pool = ordered[: int(frontier_size or active_limit)] if frontier_branch else ordered[:active_limit]
        day_pool = [
            row
            for row in raw_pool
            if _alcb_replay_candidate_passes_proxy(
                row,
                min_first30_rel_volume=min_first30_rel_volume,
                min_first30_signal_cpr=min_first30_signal_cpr,
                require_first30_campaign_breakout_acceptance=require_first30_campaign_breakout_acceptance,
            )
        ]
        if not day_pool:
            continue
        trade_day = date.fromisoformat(day_label)
        active_symbols = {str(row.get("symbol") or "") for row in day_active}
        for rank, row in enumerate(day_pool, start=1):
            symbol = str(row.get("symbol") or "")
            enriched = {
                **row,
                "pool_variant": pool_variant,
                "pool_rank": rank,
                "pool_active": symbol in active_symbols,
                "candidate_rank": rank,
                "frontier_role": "initial_active" if symbol in active_symbols else "frontier_shadow",
                "frontier_role_for_replay": "initial_active" if symbol in active_symbols else "frontier_shadow",
                "frontier_initial_active": symbol in active_symbols,
                "frontier_rank": rank,
                "selector_variant": str(selector.get("name") or ""),
                "active_rank_score": float(_selector_score(row, selector)),
            }
            pool_rows.append(enriched)
            metadata_by_key[(trade_day, symbol)] = _alcb_replay_candidate_metadata(enriched)
        active_rows.extend(day_active)
        frontier_by_day[trade_day] = tuple(str(row.get("symbol") or "") for row in day_pool)
        frontier_scores_by_day[trade_day] = {
            str(row.get("symbol") or ""): _selector_score(row, selector)
            for row in day_pool
        }
    pool_context_by_key = _context_by_key_for_rows(context_by_key, pool_rows)
    selections = [
        Selection(
            date.fromisoformat(str(row.get("trade_date") or "")),
            str(row.get("symbol") or ""),
            _selector_score(row, selector),
            "stage09_alcb_breakout_active",
        )
        for row in active_rows
        if (date.fromisoformat(str(row.get("trade_date") or "")), str(row.get("symbol") or "")) in pool_context_by_key
    ]
    selection_counts: dict[date, int] = defaultdict(int)
    for selection in selections:
        selection_counts[selection.trade_date] += 1
    return compile_core_replay(
        selections,
        dataset,
        pool_context_by_key,
        dataset.trading_dates,
        dict(selection_counts),
        cfg,
        frontier_by_day=frontier_by_day if frontier_branch else None,
        frontier_scores_by_day=frontier_scores_by_day if frontier_branch else None,
        candidate_metadata_by_key=metadata_by_key,
        source_calibration_metadata={
            "structural_campaign_surfacing": STRUCTURAL_CAMPAIGN_SURFACING_VERSION,
            "alcb_breakout_replay": ALCB_BREAKOUT_REPLAY_VERSION,
            "pool_variant": pool_variant,
            "selector_mode": selector_mode,
        },
    )


def _full_context_by_key(contexts: dict[date, tuple[First30Context, ...]]) -> dict[tuple[date, str], First30Context]:
    return {(day, ctx.symbol): ctx for day, items in contexts.items() for ctx in items}


def _context_by_key_for_rows(
    context_by_key: dict[tuple[date, str], First30Context],
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[date, str], First30Context]:
    out: dict[tuple[date, str], First30Context] = {}
    for row in rows:
        day_label = str(row.get("trade_date") or "")
        symbol = str(row.get("symbol") or "")
        if not day_label or not symbol:
            continue
        trade_day = date.fromisoformat(day_label)
        key = (trade_day, symbol)
        ctx = context_by_key.get(key)
        if ctx is not None:
            out[key] = ctx
    return out


def _alcb_replay_candidate_passes_proxy(
    row: dict[str, Any],
    *,
    min_first30_rel_volume: float,
    min_first30_signal_cpr: float,
    require_first30_campaign_breakout_acceptance: bool,
) -> bool:
    if _feature_num(row, "first30_rel_volume") < float(min_first30_rel_volume):
        return False
    if _feature_num(row, "first30_signal_cpr", "first30_signal_bar_cpr") < float(min_first30_signal_cpr):
        return False
    if require_first30_campaign_breakout_acceptance and not (
        bool(row.get("first30_breakout_acceptance")) or bool(row.get("first30_breakout_confirmation"))
    ):
        return False
    return True


def _alcb_replay_candidate_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("campaign_metadata") or {})
    for key, value in row.items():
        if key != "campaign_metadata":
            metadata[key] = value
    metadata["source"] = "stage09_structural_campaign_alcb_breakout_replay"
    metadata["selector_variant"] = str(row.get("selector_variant") or "")
    metadata["active_rank_score"] = float(_num(row.get("active_rank_score")))
    metadata["causal_calibration_score"] = float(_num(row.get("causal_calibration_score")))
    metadata["causal_score_source"] = str(row.get("causal_score_source") or "")
    metadata["causal_score_uses_ex_post_labels"] = bool(row.get("causal_score_uses_ex_post_labels"))
    metadata["first30_breakout_acceptance"] = bool(row.get("first30_breakout_acceptance"))
    metadata["first30_breakout_confirmation"] = bool(row.get("first30_breakout_confirmation"))
    metadata["first30_breakout_acceptance_closes"] = int(_num(row.get("first30_breakout_acceptance_closes")))
    metadata["frontier_initial_active"] = bool(row.get("frontier_initial_active", row.get("pool_active", True)))
    metadata["frontier_role"] = "initial_active" if metadata["frontier_initial_active"] else "frontier_shadow"
    metadata["candidate_rank"] = int(_num(row.get("candidate_rank") or row.get("pool_rank") or row.get("frontier_rank")))
    return metadata


def _alcb_trade_plan_spec(variant: dict[str, Any]) -> TradePlanSpec:
    exit_spec = dict(variant.get("exit") or {})
    route_family = str(variant.get("route_family") or "")
    entry = EntrySpec(
        name=str(variant.get("name") or ""),
        mode="first30_open" if route_family == "first30_open_proxy" else "breakout" if route_family == "breakout_family" else "or_high_reclaim",
        max_signal_bars=int(variant.get("max_signal_bars") or 18),
        min_close_location=float(variant.get("min_first30_signal_cpr") or 0.0),
    )
    exit_plan = ExitSpec(
        name=str(exit_spec.get("name") or "eod_atr"),
        stop_mode=str(exit_spec.get("stop_mode") or "atr"),
        hard_stop_enabled=bool(exit_spec.get("hard_stop_enabled", False)),
    )
    return TradePlanSpec(str(variant.get("name") or variant.get("variant_id") or ""), entry, exit_plan)


def _alcb_replay_config(cfg: KALCBConfig, variant: dict[str, Any]) -> KALCBConfig:
    frontier_size = int(variant.get("frontier_size") or 0)
    routes = _alcb_replay_routes(variant)
    return cfg.with_mutations(
        {
            "kalcb.entry.frontier_branch_universe": bool(variant.get("frontier_branch")),
            "kalcb.entry.routes": routes,
            "kalcb.entry.max_signal_bars": int(variant.get("max_signal_bars") or 18),
            "kalcb.entry.min_first30_rel_volume": float(variant.get("min_first30_rel_volume") or 0.0),
            "kalcb.entry.min_first30_signal_cpr": float(variant.get("min_first30_signal_cpr") or 0.0),
            "kalcb.entry.max_frontier_rank": frontier_size if bool(variant.get("frontier_branch")) else 0,
            "kalcb.frontier.shadow_enabled": False,
            "kalcb.frontier.rotation_enabled": False,
        }
    )


def _alcb_replay_routes(variant: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "max_signal_bars": int(variant.get("max_signal_bars") or 18),
        "min_first30_rel_volume": float(variant.get("min_first30_rel_volume") or 0.0),
        "min_first30_signal_cpr": float(variant.get("min_first30_signal_cpr") or 0.0),
        "min_close_location": float(variant.get("min_first30_signal_cpr") or 0.0),
        "max_frontier_rank": int(variant.get("frontier_size") or 0) if bool(variant.get("frontier_branch")) else 0,
        "route_max_session_trades": int(variant.get("max_session_trades_per_route") or 1),
        "context_min": {"first30_breakout_acceptance": 1.0} if bool(variant.get("require_first30_campaign_breakout_acceptance", True)) else {},
    }
    if str(variant.get("route_family") or "") == "first30_open_proxy":
        return [
            {
                **base,
                "name": "first30_open_proxy",
                "mode": "first30_open",
                "priority": 0,
            }
        ]
    if str(variant.get("route_family") or "") == "campaign_or_high_reclaim":
        return [
            {
                **base,
                "name": "campaign_breakout_or_high_reclaim",
                "mode": "or_high_reclaim",
                "priority": 0,
                "level_source": "campaign_breakout_level",
                "min_reclaim_closes": 2,
                "min_reclaim_ret": 0.0,
                "max_pullback_from_vwap_pct": 0.004,
            }
        ]
    return [
        {**base, "name": "combined_breakout", "mode": "combined_breakout", "priority": 0},
        {**base, "name": "or_breakout", "mode": "or_breakout", "priority": 1},
        {**base, "name": "pdh_breakout", "mode": "pdh_breakout", "priority": 2},
    ]


def _alcb_replay_result_row(
    variant: dict[str, Any],
    window: str,
    selection_basis: str,
    trade_spec: TradePlanSpec,
    outcomes: list[Any],
    metrics: dict[str, float],
    digest: dict[str, Any],
    compiled: Any,
) -> dict[str, Any]:
    trade_count = int(metrics.get("broker_trade_count", metrics.get("trade_count", 0.0)) or 0)
    days = len(tuple(compiled.session_dates or ()))
    trades_per_day = trade_count / max(days, 1)
    entry_counts = _counts(getattr(outcome, "entry_type", "") for outcome in outcomes)
    frontier_counts = _counts(getattr(outcome, "frontier_role", "") for outcome in outcomes)
    rejected, reject_reason = _alcb_replay_reject_reason(metrics, trade_count, trades_per_day)
    row = {
        "version": ALCB_BREAKOUT_REPLAY_VERSION,
        "window": window,
        "selection_basis": selection_basis,
        "variant_id": str(variant.get("variant_id") or ""),
        "variant_name": str(variant.get("name") or ""),
        "variant_spec": variant,
        "pool_variant": str(variant.get("pool_variant") or ""),
        "selector_mode": str(variant.get("selector_mode") or "structural_first30_causal_tiebreak"),
        "frontier_branch": bool(variant.get("frontier_branch")),
        "frontier_size": int(variant.get("frontier_size") or 0),
        "route_family": str(variant.get("route_family") or ""),
        "min_first30_rel_volume": float(variant.get("min_first30_rel_volume") or 0.0),
        "min_first30_signal_cpr": float(variant.get("min_first30_signal_cpr") or 0.0),
        "max_signal_bars": int(variant.get("max_signal_bars") or 0),
        "trade_plan_spec": _trade_plan_payload(trade_spec),
        "rejected": rejected,
        "reject_reason": reject_reason,
        "replay_score": _alcb_replay_score(metrics, trade_count, trades_per_day, rejected),
        "trade_count": trade_count,
        "trades_per_day": trades_per_day,
        "entry_type_counts": entry_counts,
        "frontier_role_counts": frontier_counts,
        "metrics": metrics,
        "replay_digest": {
            "decision_count": digest.get("decision_count"),
            "entry_rejection_count": digest.get("entry_rejection_count"),
            "top_entry_rejection_reasons": digest.get("top_entry_rejection_reasons"),
            "top_entry_failed_gates": digest.get("top_entry_failed_gates"),
            "same_bar_fill_count": digest.get("same_bar_fill_count"),
        },
        "compiled_replay_fingerprint": str(getattr(compiled, "source_fingerprint", "")),
        "candidate_artifact_hash": str(getattr(compiled, "candidate_artifact_hash", "")),
    }
    for key in (
        "broker_net_return_pct",
        "official_mtm_net_return_pct",
        "broker_net_profit",
        "broker_expected_total_r",
        "broker_avg_r",
        "broker_max_drawdown_pct",
        "broker_mfe_capture",
        "avg_mfe_r",
        "avg_mfe_capture",
        "portfolio_equivalent_net_return_pct",
        "portfolio_equivalent_max_drawdown_pct",
        "candidate_pool_count",
        "avg_candidate_pool_per_day",
        "initial_active_candidate_count",
        "frontier_expansion_candidate_count",
        "candidate_pool_conversion",
    ):
        row[key] = float(metrics.get(key, 0.0) or 0.0)
    return row


def _alcb_replay_empty_row(variant: dict[str, Any], window: str, selection_basis: str, reason: str) -> dict[str, Any]:
    return {
        "version": ALCB_BREAKOUT_REPLAY_VERSION,
        "window": window,
        "selection_basis": selection_basis,
        "variant_id": str(variant.get("variant_id") or ""),
        "variant_name": str(variant.get("name") or ""),
        "variant_spec": variant,
        "pool_variant": str(variant.get("pool_variant") or ""),
        "selector_mode": str(variant.get("selector_mode") or "structural_first30_causal_tiebreak"),
        "frontier_branch": bool(variant.get("frontier_branch")),
        "route_family": str(variant.get("route_family") or ""),
        "rejected": True,
        "reject_reason": reason,
        "replay_score": -1e9,
        "trade_count": 0,
        "trades_per_day": 0.0,
        "metrics": {},
    }


def _alcb_replay_reject_reason(metrics: dict[str, float], trade_count: int, trades_per_day: float) -> tuple[bool, str]:
    if trade_count < 5:
        return True, "too_few_trades"
    if trades_per_day > 3.5:
        return True, "trade_count_explosion"
    if float(metrics.get("broker_net_return_pct", 0.0) or 0.0) <= 0.0:
        return True, "non_positive_broker_net_return"
    if float(metrics.get("broker_expected_total_r", 0.0) or 0.0) <= 0.0:
        return True, "non_positive_expected_total_r"
    if float(metrics.get("broker_max_drawdown_pct", 0.0) or 0.0) > 0.08:
        return True, "drawdown_veto"
    return False, ""


def _alcb_replay_score(metrics: dict[str, float], trade_count: int, trades_per_day: float, rejected: bool) -> float:
    if rejected:
        return -1e6 + float(metrics.get("broker_expected_total_r", 0.0) or 0.0)
    net = float(metrics.get("broker_net_return_pct", 0.0) or 0.0)
    expected_r = float(metrics.get("broker_expected_total_r", 0.0) or 0.0)
    drawdown = float(metrics.get("broker_max_drawdown_pct", 0.0) or 0.0)
    mfe_capture = float(metrics.get("broker_mfe_capture", metrics.get("avg_mfe_capture", 0.0)) or 0.0)
    discipline_penalty = max(0.0, trades_per_day - 2.5) * 8.0 + max(0.0, 0.15 - trades_per_day) * 3.0
    return expected_r + 100.0 * net + 5.0 * mfe_capture - 100.0 * drawdown - discipline_penalty


def _select_alcb_breakout_replay_shortlist(rows: list[dict[str, Any]], shortlist_size: int) -> list[dict[str, Any]]:
    eligible = [dict(row) for row in rows if not bool(row.get("rejected"))]
    if not eligible:
        eligible = [dict(row) for row in rows]
    eligible.sort(key=_alcb_replay_sort_key)
    out: list[dict[str, Any]] = []
    seen_pools: set[str] = set()
    for row in eligible:
        pool = f"{row.get('pool_variant')}:{row.get('route_family')}:{row.get('selector_mode')}:{row.get('min_first30_rel_volume')}"
        if pool in seen_pools and len(out) < max(1, int(shortlist_size)) - 1:
            continue
        seen_pools.add(pool)
        out.append(row)
        if len(out) >= max(1, int(shortlist_size)):
            return out
    for row in eligible:
        if row not in out:
            out.append(row)
        if len(out) >= max(1, int(shortlist_size)):
            break
    return out


def _alcb_replay_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row.get("rejected")),
        -float(row.get("replay_score") or 0.0),
        -float(row.get("broker_expected_total_r") or 0.0),
        float(row.get("broker_max_drawdown_pct") or 0.0),
        float(row.get("trades_per_day") or 0.0),
        str(row.get("variant_id") or ""),
    )


def _trade_plan_payload(spec: TradePlanSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "entry": asdict(spec.entry),
        "exit": asdict(spec.exit),
    }


def _alcb_label_num(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def active_budget_by_day_from_context(existing_context: Any) -> dict[str, int]:
    compiled = getattr(existing_context, "compiled_replay", None)
    snapshots = dict(getattr(compiled, "snapshots", {}) or {})
    out: dict[str, int] = {}
    for day, snapshot in sorted(snapshots.items(), key=lambda item: str(item[0])):
        day_label = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
        metadata = dict(getattr(snapshot, "metadata", {}) or {})
        active_symbols = [str(symbol) for symbol in metadata.get("active_symbols", ()) if str(symbol)]
        if active_symbols:
            out[day_label] = len(active_symbols)
            continue
        inferred = 0
        for candidate in tuple(getattr(snapshot, "candidates", ()) or ()):
            candidate_meta = dict(getattr(candidate, "metadata", {}) or {})
            inferred += int(bool(candidate_meta.get("frontier_initial_active")))
        out[day_label] = inferred
    return out


def optimize_structural_campaign_train(
    train_rows: Iterable[dict[str, Any]],
    holdout_rows: Iterable[dict[str, Any]],
    *,
    cfg: KALCBConfig,
    train_active_budget_by_day: dict[str, int] | None = None,
    holdout_active_budget_by_day: dict[str, int] | None = None,
    train_oracle_rows: Iterable[dict[str, Any]] | None = None,
    holdout_oracle_rows: Iterable[dict[str, Any]] | None = None,
    train_causal_rows: Iterable[dict[str, Any]] | None = None,
    holdout_causal_rows: Iterable[dict[str, Any]] | None = None,
    shortlist_size: int = OPTIMIZER_SHORTLIST_SIZE,
    grid: dict[str, Iterable[Any]] | None = None,
) -> dict[str, Any]:
    train_features = attach_causal_calibration_scores(train_rows, train_causal_rows)
    holdout_features = attach_causal_calibration_scores(holdout_rows, holdout_causal_rows)
    train_oracles = [dict(row) for row in train_oracle_rows or ()]
    holdout_oracles = [dict(row) for row in holdout_oracle_rows or ()]
    specs = _optimizer_specs(cfg, grid=grid)
    train_eval_rows = _evaluate_optimizer_specs_fast(
        train_features,
        specs,
        cfg=cfg,
        active_budget_by_day=train_active_budget_by_day,
        oracle_rows=train_oracles,
        window="train",
    )
    shortlist = _select_optimizer_shortlist(train_eval_rows, shortlist_size)
    holdout_eval_rows: list[dict[str, Any]] = []
    for rank, selected in enumerate(shortlist, start=1):
        source_spec = dict(selected.get("source_spec") or {})
        pool_variant = str(selected.get("pool_variant") or "")
        selector_variant = str(selected.get("selector_variant") or "")
        matches = [
            row
            for row in _evaluate_optimizer_specs_fast(
                holdout_features,
                (source_spec,),
                cfg=cfg,
                active_budget_by_day=holdout_active_budget_by_day,
                oracle_rows=holdout_oracles,
                window="holdout",
            )
            if str(row.get("pool_variant") or "") == pool_variant and str(row.get("selector_variant") or "") == selector_variant
        ]
        if not matches:
            continue
        row = dict(matches[0])
        row["frozen_train_rank"] = rank
        row["frozen_train_variant_id"] = str(selected.get("optimizer_variant_id") or "")
        row["selection_basis"] = "frozen_train_shortlist_holdout_scored_once"
        holdout_eval_rows.append(row)
    return {
        "version": STRUCTURAL_CAMPAIGN_SURFACING_VERSION,
        "usage_contract": STRUCTURAL_CAMPAIGN_USAGE_CONTRACT,
        "optimizer_contract": "train_sweep_selects_shortlist_holdout_scored_once_no_holdout_optimization_causal_tiebreaker_only",
        "grid": {
            "research_min_structural_campaign_score": sorted({float(spec.get("min_structural_campaign_score") or 0.0) for spec in specs}),
            "research_structural_frontier_count": sorted({int(spec.get("frontier_count") or 0) for spec in specs}),
            "research_min_rs_percentile": sorted({float(spec.get("min_rs_percentile") or 0.0) for spec in specs}),
            "research_min_sector_daily_score_pct": sorted({float(spec.get("min_sector_daily_score_pct") or 0.0) for spec in specs}),
            "research_min_sector_participation": sorted({float(spec.get("min_sector_participation") or 0.0) for spec in specs}),
            "research_max_box_range_pct": sorted({float(spec.get("max_box_range_pct") or 0.0) for spec in specs}),
        },
        "selector_variants": [dict(spec) for spec in SELECTOR_VARIANT_SPECS],
        "objective": {
            "source_score": "active_recall + recall_at_32 + top_decile + 0.5*best_pool + 0.2*route_share + train_top1_quality + monotonicity_bonus - overwide/leakage/active_budget_penalties",
            "primary_order": [
                "candidate_surfacing_recall",
                "active_selector_quality",
                "active_budget_discipline",
                "route_static_conversion",
                "causal_ranker_small_tie_breaker",
            ],
        },
        "train_oracle_available": bool(train_oracles),
        "holdout_oracle_available": bool(holdout_oracles),
        "train_source_spec_count": len(specs),
        "train_source_shortlist_limit": OPTIMIZER_SOURCE_SHORTLIST_SIZE,
        "train_variant_count": len(train_eval_rows),
        "train_rows": train_eval_rows,
        "shortlist_size": len(shortlist),
        "shortlist": shortlist,
        "holdout_scored_once_rows": holdout_eval_rows,
        "best_train_variant": shortlist[0] if shortlist else {},
        "holdout_frozen_rank1_variant": holdout_eval_rows[0] if holdout_eval_rows else {},
        "best_holdout_frozen_variant": holdout_eval_rows[0] if holdout_eval_rows else {},
        "holdout_selection_basis": "reported_in_frozen_train_rank_order_no_holdout_sort",
    }


def build_structural_campaign_surfacing_artifacts(
    train_rows: Iterable[dict[str, Any]],
    holdout_rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    cfg: KALCBConfig | None = None,
    active_budget_by_day: dict[str, int] | None = None,
    train_active_budget_by_day: dict[str, int] | None = None,
    holdout_active_budget_by_day: dict[str, int] | None = None,
    train_oracle_rows: Iterable[dict[str, Any]] | None = None,
    holdout_oracle_rows: Iterable[dict[str, Any]] | None = None,
    train_causal_rows: Iterable[dict[str, Any]] | None = None,
    holdout_causal_rows: Iterable[dict[str, Any]] | None = None,
    optimizer_grid: dict[str, Iterable[Any]] | None = None,
) -> dict[str, Any]:
    config = cfg or KALCBConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_features = attach_causal_calibration_scores(train_rows, train_causal_rows)
    holdout_features = attach_causal_calibration_scores(holdout_rows, holdout_causal_rows)
    train_oracles = [dict(row) for row in train_oracle_rows or ()]
    holdout_oracles = [dict(row) for row in holdout_oracle_rows or ()]
    train_active_budget = train_active_budget_by_day if train_active_budget_by_day is not None else active_budget_by_day
    holdout_active_budget = holdout_active_budget_by_day if holdout_active_budget_by_day is not None else active_budget_by_day
    train_pools = _materialize_pool_variants(train_features, train_active_budget, config)
    holdout_pools = _materialize_pool_variants(holdout_features, holdout_active_budget, config)
    recall = {
        "train": summarize_structural_recall(train_features, train_pools, oracle_rows=train_oracles),
        "holdout": summarize_structural_recall(holdout_features, holdout_pools, oracle_rows=holdout_oracles),
    }
    optimizer = optimize_structural_campaign_train(
        train_features,
        holdout_features,
        cfg=config,
        train_active_budget_by_day=train_active_budget,
        holdout_active_budget_by_day=holdout_active_budget,
        train_oracle_rows=train_oracles,
        holdout_oracle_rows=holdout_oracles,
        grid=optimizer_grid,
    )
    alcb_delta = build_alcb_delta_diagnostics(
        train_features,
        holdout_features,
        train_oracle_rows=train_oracles,
        holdout_oracle_rows=holdout_oracles,
    )
    manifest = {
        "version": STRUCTURAL_CAMPAIGN_SURFACING_VERSION,
        "usage_contract": STRUCTURAL_CAMPAIGN_USAGE_CONTRACT,
        "variant_contract": "structural_source_pool_train_only_active_selector_variants_holdout_scored_once",
        "variants": [
            {
                **dict(spec),
                "active_budget_source": "incumbent_daily_active_count" if (train_active_budget or holdout_active_budget) else "ws_budget",
                "materialized": True,
            }
            for spec in POOL_VARIANT_SPECS
        ],
        "selector_variants": [dict(spec) for spec in SELECTOR_VARIANT_SPECS],
    }
    ranker_profile = {
        "version": STRUCTURAL_CAMPAIGN_SURFACING_VERSION,
        "status": "attached_if_stage08_scores_available",
        "role": "train_only_selector_tie_breaker_and_diagnostic_not_primary_source",
        "live_feature_contract": STRUCTURAL_CAMPAIGN_USAGE_CONTRACT,
        "forbidden_live_fields": ["mfe_r", "mae_r", "net_r", "oracle_score", "label_composite_oracle_recall"],
        "causal_score_fields": ["causal_calibration_score", "causal_rank_in_day", "causal_score_source"],
        "train_score_coverage": _causal_score_coverage(train_features),
        "holdout_score_coverage": _causal_score_coverage(holdout_features),
    }
    replay_summary = {
        "version": STRUCTURAL_CAMPAIGN_SURFACING_VERSION,
        "status": "not_run",
        "reason": "stage09 artifact builder materializes structural features, pools, and recall before replay judgment",
        "compile_structural_campaign_replay_available": True,
        "first_required_replay": "structural_active_budget_first30_anchor_sanity",
    }
    paths = {
        "structural_campaign_features_train_jsonl": out / "structural_campaign_features_train.jsonl",
        "structural_campaign_features_holdout_jsonl": out / "structural_campaign_features_holdout.jsonl",
        "structural_campaign_pools_train_jsonl": out / "structural_campaign_pools_train.jsonl",
        "structural_campaign_pools_holdout_jsonl": out / "structural_campaign_pools_holdout.jsonl",
        "structural_campaign_research_snapshots_train_jsonl": out / "structural_campaign_research_snapshots_train.jsonl",
        "structural_campaign_research_snapshots_holdout_jsonl": out / "structural_campaign_research_snapshots_holdout.jsonl",
        "structural_campaign_recall_summary_json": out / "structural_campaign_recall_summary.json",
        "structural_campaign_train_optimizer_jsonl": out / "structural_campaign_train_optimizer.jsonl",
        "structural_campaign_train_optimizer_shortlist_json": out / "structural_campaign_train_optimizer_shortlist.json",
        "structural_campaign_holdout_frozen_shortlist_jsonl": out / "structural_campaign_holdout_frozen_shortlist.jsonl",
        "structural_campaign_optimizer_summary_json": out / "structural_campaign_optimizer_summary.json",
        "structural_campaign_ranker_profile_json": out / "structural_campaign_ranker_profile.json",
        "structural_campaign_replay_summary_json": out / "structural_campaign_replay_summary.json",
        "structural_campaign_alcb_delta_diagnostics_json": out / "structural_campaign_alcb_delta_diagnostics.json",
        "structural_campaign_variant_manifest_json": out / "structural_campaign_variant_manifest.json",
        "diagnostics_summary_json": out / "diagnostics_summary.json",
        "structural_campaign_report_md": out / "structural_campaign_report.md",
        "full_diagnostics_index_json": out / "full_diagnostics_index.json",
    }
    write_jsonl(paths["structural_campaign_features_train_jsonl"], (structural_campaign_feature_artifact_row(row) for row in train_features))
    write_jsonl(paths["structural_campaign_features_holdout_jsonl"], (structural_campaign_feature_artifact_row(row) for row in holdout_features))
    write_jsonl(paths["structural_campaign_pools_train_jsonl"], (_pool_artifact_row(row) for row in train_pools))
    write_jsonl(paths["structural_campaign_pools_holdout_jsonl"], (_pool_artifact_row(row) for row in holdout_pools))
    write_jsonl(paths["structural_campaign_research_snapshots_train_jsonl"], _research_snapshot_index_rows(train_features, train_pools, "train"))
    write_jsonl(paths["structural_campaign_research_snapshots_holdout_jsonl"], _research_snapshot_index_rows(holdout_features, holdout_pools, "holdout"))
    paths["structural_campaign_recall_summary_json"].write_text(json.dumps(recall, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_jsonl(paths["structural_campaign_train_optimizer_jsonl"], optimizer["train_rows"])
    write_jsonl(paths["structural_campaign_holdout_frozen_shortlist_jsonl"], optimizer["holdout_scored_once_rows"])
    paths["structural_campaign_train_optimizer_shortlist_json"].write_text(json.dumps(optimizer["shortlist"], indent=2, sort_keys=True, default=str), encoding="utf-8")
    paths["structural_campaign_optimizer_summary_json"].write_text(json.dumps({key: value for key, value in optimizer.items() if key != "train_rows"}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    paths["structural_campaign_ranker_profile_json"].write_text(json.dumps(ranker_profile, indent=2, sort_keys=True, default=str), encoding="utf-8")
    paths["structural_campaign_replay_summary_json"].write_text(json.dumps(replay_summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    paths["structural_campaign_alcb_delta_diagnostics_json"].write_text(json.dumps(alcb_delta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    paths["structural_campaign_variant_manifest_json"].write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report = render_structural_campaign_report(recall, manifest, optimizer, alcb_delta)
    paths["structural_campaign_report_md"].write_text(report, encoding="utf-8")
    summary = {
        "version": STRUCTURAL_CAMPAIGN_SURFACING_VERSION,
        "usage_contract": STRUCTURAL_CAMPAIGN_USAGE_CONTRACT,
        "created_at": _utc_now_iso(),
        "train": {"feature_row_count": len(train_features), "pool_row_count": len(train_pools), "recall": recall["train"]},
        "holdout": {"feature_row_count": len(holdout_features), "pool_row_count": len(holdout_pools), "recall": recall["holdout"]},
        "optimizer": {key: value for key, value in optimizer.items() if key != "train_rows"},
        "variant_manifest": manifest,
        "ranker_profile": ranker_profile,
        "replay_summary": replay_summary,
        "alcb_delta_diagnostics": alcb_delta,
        "artifact_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["diagnostics_summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    paths["full_diagnostics_index_json"].write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary


def _optimizer_specs(cfg: KALCBConfig, *, grid: dict[str, Iterable[Any]] | None = None) -> list[dict[str, Any]]:
    min_score_values = _grid_values(grid, "research_min_structural_campaign_score", STRUCTURAL_MIN_SCORE_GRID, float)
    frontier_values = _grid_values(grid, "research_structural_frontier_count", STRUCTURAL_FRONTIER_GRID, int)
    min_rs_values = _grid_values(grid, "research_min_rs_percentile", STRUCTURAL_MIN_RS_GRID, float)
    min_sector_values = _grid_values(grid, "research_min_sector_daily_score_pct", STRUCTURAL_MIN_SECTOR_DAILY_GRID, float)
    min_participation_values = _grid_values(grid, "research_min_sector_participation", (cfg.research_min_sector_participation,), float)
    max_box_range_values = _grid_values(grid, "research_max_box_range_pct", (cfg.research_max_box_range_pct,), float)
    specs: list[dict[str, Any]] = []
    for min_score in min_score_values:
        for frontier_count in frontier_values:
            for min_rs in min_rs_values:
                for min_sector_daily in min_sector_values:
                    for min_participation in min_participation_values:
                        for max_box_range in max_box_range_values:
                            mutations = {
                                "kalcb.research.min_structural_campaign_score": float(min_score),
                                "kalcb.research.structural_frontier_count": int(frontier_count),
                                "kalcb.research.min_rs_percentile": float(min_rs),
                                "kalcb.research.min_sector_daily_score_pct": float(min_sector_daily),
                                "kalcb.research.min_sector_participation": float(min_participation),
                                "kalcb.research.max_box_range_pct": float(max_box_range),
                            }
                            spec = {
                                "min_structural_campaign_score": float(min_score),
                                "frontier_count": int(frontier_count),
                                "min_rs_percentile": float(min_rs),
                                "min_sector_daily_score_pct": float(min_sector_daily),
                                "min_sector_participation": float(min_participation),
                                "max_box_range_pct": float(max_box_range),
                                "mutations": mutations,
                            }
                            spec["source_variant_id"] = stable_signature(mutations)[:16]
                            specs.append(spec)
    return specs


def _pool_artifact_row(row: dict[str, Any]) -> dict[str, Any]:
    slim = {key: row.get(key) for key in POOL_ARTIFACT_KEYS if key in row}
    slim["campaign_metadata"] = _campaign_metadata_artifact(row)
    return slim


def structural_campaign_feature_artifact_row(row: dict[str, Any]) -> dict[str, Any]:
    slim_keys = tuple(key for key in POOL_ARTIFACT_KEYS if not key.startswith("pool_") and key not in {"active_budget_source", "active_budget_for_day", "frontier_role_for_replay"})
    slim = {key: row.get(key) for key in slim_keys if key in row}
    slim["campaign_metadata"] = _campaign_metadata_artifact(row)
    return slim


def _campaign_metadata_artifact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "structural_campaign_score",
            "first30_confirmation_score",
            "campaign_state",
            "campaign_state_score",
            "campaign_box_high",
            "campaign_box_low",
            "campaign_box_mid",
            "campaign_box_range_pct",
            "campaign_box_containment",
            "campaign_box_atr_ratio",
            "campaign_box_squeeze_pct",
            "campaign_box_tier",
            "campaign_avwap",
            "campaign_breakout_level",
            "campaign_breakout_displacement",
            "first30_rel_volume",
            "first30_signal_cpr",
            "first30_vwap_ret",
            "first30_gap",
            "first30_low_vs_prev_close",
            "first30_above_campaign_avwap",
            "first30_breakout_confirmation",
            "first30_breakout_acceptance",
            "first30_breakout_acceptance_closes",
            "campaign_avwap_distance_pct",
            "campaign_box_high_distance_pct",
            "causal_calibration_score",
            "causal_rank_in_day",
            "causal_score_source",
            "causal_score_uses_ex_post_labels",
            "selector_variant",
            "active_rank_score",
            "relative_strength_pct",
            "stock_vs_universe_strength",
            "sector_daily_score_pct",
            "sector_participation",
            "score_uses_ex_post_labels",
        )
        if key in row
    }


def _grid_values(
    grid: dict[str, Iterable[Any]] | None,
    key: str,
    default: Iterable[Any],
    cast,
) -> tuple[Any, ...]:
    raw = tuple((grid or {}).get(key, default) or default)
    values = tuple(dict.fromkeys(cast(value) for value in raw))
    return values or tuple(cast(value) for value in default)


def _evaluate_optimizer_specs_fast(
    feature_rows: list[dict[str, Any]],
    specs: Iterable[dict[str, Any]],
    *,
    cfg: KALCBConfig,
    active_budget_by_day: dict[str, int] | None,
    oracle_rows: list[dict[str, Any]],
    window: str,
) -> list[dict[str, Any]]:
    _prepare_optimizer_feature_rows(feature_rows)
    feature_index = _optimizer_feature_index(feature_rows)
    oracle = _best_oracle_by_key(oracle_rows or ())
    best_by_day = _best_oracle_by_day(oracle.values()) if oracle else {}
    top_decile_oracle = _top_decile_oracle_keys(oracle.values()) if oracle else set()
    source_evals: list[dict[str, Any]] = []
    route_cache: dict[tuple[str, str], tuple[list[str], str]] = {}
    prescreen_selector = _selector_spec("structural_first30")
    prescreen_pool = {"name": "source_prescreen_frontier32", "frontier_size": 32}
    for spec in list(specs):
        source_by_day = _optimizer_source_by_day(feature_index, spec)
        source_rows = [row for rows in source_by_day.values() for row in rows]
        recall = _optimizer_source_recall(source_rows, source_by_day, oracle, best_by_day, top_decile_oracle)
        monotonicity = _monotonicity_score(dict(recall.get("structural_score_buckets") or {}))
        leakage = bool(recall.get("score_uses_ex_post_labels")) or any(bool(row.get("causal_score_uses_ex_post_labels")) for row in source_rows)
        variant_cfg = cfg.with_mutations(dict(spec.get("mutations") or {}))
        ranked = _optimizer_ranked_source_by_day(source_by_day, prescreen_selector)
        pool = _optimizer_pool_view_from_ranked(
            ranked,
            cfg=variant_cfg,
            active_budget_by_day=active_budget_by_day,
            frontier_size=prescreen_pool.get("frontier_size"),
            selector=prescreen_selector,
        )
        metrics = _optimizer_pool_metrics(pool, oracle, best_by_day, top_decile_oracle)
        route = _route_conversion_summary(pool["rows"], cache=route_cache)
        budget = {
            "active_budget_source": "incumbent_daily_active_count" if active_budget_by_day is not None else "ws_budget",
            "active_budget_day_count": pool["day_count"],
            "active_budget_violation_count": pool["active_budget_violation_count"],
            "active_budget_violation_share": pool["active_budget_violation_count"] / max(pool["day_count"], 1),
        }
        source_evals.append(
            {
                "spec": spec,
                "source_by_day": source_by_day,
                "source_rows": source_rows,
                "recall": recall,
                "monotonicity": monotonicity,
                "leakage": leakage,
                "variant_cfg": variant_cfg,
                "prescreen_row": _optimizer_result_row(
                    window=window,
                    spec=spec,
                    pool_variant=str(prescreen_pool["name"]),
                    selector=prescreen_selector,
                    feature_count=len(source_rows),
                    pool_rows=pool["rows"],
                    recall=recall,
                    metrics=metrics,
                    route=route,
                    budget=budget,
                    monotonicity=monotonicity,
                    leakage=leakage,
                ),
            }
        )
    eligible_sources = [item for item in source_evals if not bool(item["prescreen_row"].get("score_uses_ex_post_labels"))]
    ranked_sources = sorted(eligible_sources or source_evals, key=lambda item: _optimizer_sort_key(item["prescreen_row"]))
    source_limit = min(len(ranked_sources), max(1, OPTIMIZER_SOURCE_SHORTLIST_SIZE))
    selected_source_ids = {str(item["spec"].get("source_variant_id") or "") for item in ranked_sources[:source_limit]}
    out: list[dict[str, Any]] = []
    for source_eval in source_evals:
        spec = dict(source_eval["spec"])
        if str(spec.get("source_variant_id") or "") not in selected_source_ids:
            continue
        source_by_day = dict(source_eval["source_by_day"])
        source_rows = list(source_eval["source_rows"])
        recall = dict(source_eval["recall"])
        monotonicity = float(source_eval["monotonicity"])
        leakage = bool(source_eval["leakage"])
        variant_cfg = source_eval["variant_cfg"]
        for selector in SELECTOR_VARIANT_SPECS:
            ranked = _optimizer_ranked_source_by_day(source_by_day, selector)
            for pool_spec in POOL_VARIANT_SPECS:
                pool_variant = str(pool_spec.get("name") or "")
                pool = _optimizer_pool_view_from_ranked(
                    ranked,
                    cfg=variant_cfg,
                    active_budget_by_day=active_budget_by_day,
                    frontier_size=pool_spec.get("frontier_size"),
                    selector=selector,
                )
                metrics = _optimizer_pool_metrics(pool, oracle, best_by_day, top_decile_oracle)
                route = _route_conversion_summary(pool["rows"], cache=route_cache)
                budget = {
                    "active_budget_source": "incumbent_daily_active_count" if active_budget_by_day is not None else "ws_budget",
                    "active_budget_day_count": pool["day_count"],
                    "active_budget_violation_count": pool["active_budget_violation_count"],
                    "active_budget_violation_share": pool["active_budget_violation_count"] / max(pool["day_count"], 1),
                }
                out.append(
                    _optimizer_result_row(
                        window=window,
                        spec=spec,
                        pool_variant=pool_variant,
                        selector=selector,
                        feature_count=len(source_rows),
                        pool_rows=pool["rows"],
                        recall=recall,
                        metrics=metrics,
                        route=route,
                        budget=budget,
                        monotonicity=monotonicity,
                        leakage=leakage,
                    )
                )
    return out


def _prepare_optimizer_feature_rows(feature_rows: Iterable[dict[str, Any]]) -> None:
    for row in feature_rows:
        if bool(row.get("_optimizer_prepared")):
            continue
        row["_optimizer_structural_norm"] = max(0.0, min(_feature_num(row, "structural_campaign_score") / 10.0, 1.0))
        confirmation = max(_feature_num(row, "first30_confirmation_score"), 0.0)
        relvol = min(math.log1p(max(_feature_num(row, "first30_rel_volume"), 0.0)) / math.log1p(10.0), 1.0)
        cpr = max(0.0, min(_feature_num(row, "first30_signal_cpr", "first30_signal_bar_cpr"), 1.0))
        row["_optimizer_first30_norm"] = float(0.60 * min(confirmation / 6.0, 1.0) + 0.25 * relvol + 0.15 * cpr)
        row["_optimizer_causal_norm"] = 0.5 + 0.5 * math.tanh(_feature_num(row, "causal_calibration_score") / 1.5)
        row["_optimizer_relvol_norm"] = min(max(_feature_num(row, "first30_rel_volume") / 5.0, 0.0), 2.0)
        row["_optimizer_cpr_norm"] = min(max(_feature_num(row, "first30_signal_cpr", "first30_signal_bar_cpr"), 0.0), 1.0)
        row["_optimizer_prepared"] = True


def _optimizer_feature_index(feature_rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_day[str(row.get("trade_date") or "")].append(row)
    return {
        day: sorted(rows, key=_structural_rank_key)
        for day, rows in by_day.items()
    }


def _optimizer_source_by_day(
    feature_index: dict[str, list[dict[str, Any]]],
    spec: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    cap = max(int(spec.get("frontier_count") or 0), 1)
    return {
        day: filtered[:cap]
        for day, rows in sorted(feature_index.items())
        if (filtered := [row for row in rows if _optimizer_row_passes(row, spec)])
    }


def _optimizer_row_passes(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    if _num(row.get("structural_campaign_score")) < float(spec.get("min_structural_campaign_score") or 0.0):
        return False
    if _feature_num(row, "relative_strength_pct", "rs_percentile") < float(spec.get("min_rs_percentile") or 0.0):
        return False
    if _feature_num(row, "sector_daily_score_pct") < float(spec.get("min_sector_daily_score_pct") or 0.0):
        return False
    if _feature_num(row, "sector_participation", "sector_daily_participation") < float(spec.get("min_sector_participation") or 0.0):
        return False
    max_box_range = float(spec.get("max_box_range_pct") or 0.0)
    return max_box_range <= 0.0 or _feature_num(row, "campaign_box_range_pct", "box_range_pct") <= max_box_range


def _optimizer_source_recall(
    source_rows: list[dict[str, Any]],
    source_by_day: dict[str, list[dict[str, Any]]],
    oracle: dict[tuple[str, str], dict[str, Any]],
    best_by_day: dict[str, dict[str, Any]],
    top_decile_oracle: set[tuple[str, str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "feature_row_count": len(source_rows),
        "score_uses_ex_post_labels": any(bool(row.get("score_uses_ex_post_labels")) for row in source_rows),
        "oracle_label_available": bool(oracle),
        "oracle_row_count": len(oracle),
        "oracle_day_count": len({day for day, _symbol in oracle}),
        "structural_score_buckets": _score_bucket_summary(source_rows, oracle),
    }
    if not oracle:
        return out
    for k in RECALL_KS:
        hits = 0
        days = 0
        for day, oracle_row in best_by_day.items():
            rows = source_by_day.get(day, ())
            if not rows:
                continue
            days += 1
            top_symbols = {str(row.get("symbol") or "") for row in rows[:k]}
            hits += int(str(oracle_row.get("symbol") or "") in top_symbols)
        out[f"recall_at_{k}"] = hits / max(days, 1)
    return out


def _optimizer_pool_view(
    source_by_day: dict[str, list[dict[str, Any]]],
    *,
    cfg: KALCBConfig,
    active_budget_by_day: dict[str, int] | None,
    frontier_size: Any,
    selector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selector_spec = dict(selector or _selector_spec("structural_first30_causal_tiebreak"))
    return _optimizer_pool_view_from_ranked(
        _optimizer_ranked_source_by_day(source_by_day, selector_spec),
        cfg=cfg,
        active_budget_by_day=active_budget_by_day,
        frontier_size=frontier_size,
        selector=selector_spec,
    )


def _optimizer_ranked_source_by_day(
    source_by_day: dict[str, list[dict[str, Any]]],
    selector: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    selector_spec = dict(selector or _selector_spec("structural_first30_causal_tiebreak"))
    return {
        day: sorted(day_rows, key=lambda row: _selector_rank_key(row, selector_spec))
        for day, day_rows in sorted(source_by_day.items())
    }


def _optimizer_pool_view_from_ranked(
    ranked_source_by_day: dict[str, list[dict[str, Any]]],
    *,
    cfg: KALCBConfig,
    active_budget_by_day: dict[str, int] | None,
    frontier_size: Any,
    selector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cap = int(frontier_size or cfg.research_structural_frontier_count or max(cfg.frontier_size, cfg.ws_budget, cfg.research_top_long_count))
    cap = max(cap, 1)
    selector_spec = dict(selector or _selector_spec("structural_first30_causal_tiebreak"))
    rows: list[dict[str, Any]] = []
    ranked_by_day: dict[str, list[dict[str, Any]]] = {}
    pool_keys: set[tuple[str, str]] = set()
    active_keys: set[tuple[str, str]] = set()
    pool_counts: list[int] = []
    active_counts: list[int] = []
    violations = 0
    for day, day_rows in sorted(ranked_source_by_day.items()):
        expected = int((active_budget_by_day or {}).get(day, cfg.ws_budget) if active_budget_by_day is not None else cfg.ws_budget)
        active_limit = max(expected, 0)
        selected = list(day_rows[:cap])
        ranked_by_day[day] = selected
        active_count = min(active_limit, len(selected))
        pool_counts.append(len(selected))
        active_counts.append(active_count)
        violations += int(active_count > max(expected, 0))
        for index, source in enumerate(selected, start=1):
            key = (day, str(source.get("symbol") or ""))
            rows.append(source)
            pool_keys.add(key)
            if index <= active_limit:
                active_keys.add(key)
    return {
        "rows": rows,
        "ranked_by_day": ranked_by_day,
        "pool_keys": pool_keys,
        "active_keys": active_keys,
        "selector_variant": str(selector_spec.get("name") or ""),
        "day_count": len(ranked_source_by_day),
        "pool_row_count": len(rows),
        "active_row_count": len(active_keys),
        "avg_pool_size": mean(pool_counts) if pool_counts else 0.0,
        "avg_active_count": mean(active_counts) if active_counts else 0.0,
        "active_budget_violation_count": violations,
    }


def _optimizer_pool_metrics(
    pool: dict[str, Any],
    oracle: dict[tuple[str, str], dict[str, Any]],
    best_by_day: dict[str, dict[str, Any]],
    top_decile_oracle: set[tuple[str, str]],
) -> dict[str, Any]:
    pool_keys = set(pool.get("pool_keys") or ())
    active_keys = set(pool.get("active_keys") or ())
    out: dict[str, Any] = {
        "oracle_label_available": bool(oracle),
        "pool_row_count": int(pool.get("pool_row_count") or 0),
        "active_row_count": int(pool.get("active_row_count") or 0),
        "avg_pool_size": float(pool.get("avg_pool_size") or 0.0),
        "avg_active_count": float(pool.get("avg_active_count") or 0.0),
    }
    if not oracle:
        out["recall_contract"] = "pool_coverage_without_oracle_labels"
        return out
    oracle_days = sorted(best_by_day)
    out["best_oracle_in_pool_share"] = sum(1 for day in oracle_days if (day, str(best_by_day[day].get("symbol") or "")) in pool_keys) / max(len(oracle_days), 1)
    out["best_oracle_in_active_share"] = sum(1 for day in oracle_days if (day, str(best_by_day[day].get("symbol") or "")) in active_keys) / max(len(oracle_days), 1)
    out["top_decile_oracle_recall"] = len(top_decile_oracle & pool_keys) / max(len(top_decile_oracle), 1)
    ranked_by_day = {str(day): list(rows) for day, rows in dict(pool.get("ranked_by_day") or {}).items()}
    top1_oracle_scores: list[float] = []
    top1_net_values: list[float] = []
    top1_positive = 0
    for k in RECALL_KS:
        hits = 0
        days = 0
        for day, oracle_row in best_by_day.items():
            rows = ranked_by_day.get(day, ())
            if not rows:
                continue
            days += 1
            top_symbols = {str(row.get("symbol") or "") for row in rows[:k]}
            hits += int(str(oracle_row.get("symbol") or "") in top_symbols)
        out[f"recall_at_{k}"] = hits / max(days, 1)
    for day, rows in ranked_by_day.items():
        if not rows:
            continue
        top = rows[0]
        oracle_row = oracle.get((day, str(top.get("symbol") or "")))
        if oracle_row is None:
            continue
        oracle_score = _oracle_rank_tuple(oracle_row)[0]
        net_r = _num(oracle_row.get("net_r"))
        top1_oracle_scores.append(oracle_score)
        top1_net_values.append(net_r)
        top1_positive += int(net_r > 0.0)
    out["avg_top1_oracle_score"] = mean(top1_oracle_scores) if top1_oracle_scores else 0.0
    out["top1_net_r_sum"] = float(sum(top1_net_values))
    out["top1_positive_day_share"] = top1_positive / max(len(top1_net_values), 1)
    out["recall_at_active_budget"] = out["best_oracle_in_active_share"]
    return out


def _evaluate_optimizer_source_spec(
    feature_rows: list[dict[str, Any]],
    *,
    cfg: KALCBConfig,
    spec: dict[str, Any],
    active_budget_by_day: dict[str, int] | None,
    oracle_rows: list[dict[str, Any]],
    window: str,
) -> list[dict[str, Any]]:
    source_features = _source_frontier_rows(_filter_structural_optimizer_rows(feature_rows, spec), int(spec.get("frontier_count") or 0))
    variant_cfg = cfg.with_mutations(dict(spec.get("mutations") or {}))
    pools = _materialize_pool_variants(source_features, active_budget_by_day, variant_cfg)
    recall = summarize_structural_recall(source_features, pools, oracle_rows=oracle_rows)
    monotonicity = _monotonicity_score(dict(recall.get("structural_score_buckets") or {}))
    leakage = bool(recall.get("score_uses_ex_post_labels"))
    rows: list[dict[str, Any]] = []
    for pool_variant, metrics in sorted(dict(recall.get("variants") or {}).items()):
        pool_rows = [row for row in pools if str(row.get("pool_variant") or "") == pool_variant]
        route = _route_conversion_summary(pool_rows)
        budget = _active_budget_summary(pool_rows, active_budget_by_day, cfg)
        row = _optimizer_result_row(
            window=window,
            spec=spec,
            pool_variant=pool_variant,
            selector=_selector_spec("structural_first30_causal_tiebreak"),
            feature_count=len(source_features),
            pool_rows=pool_rows,
            recall=recall,
            metrics=dict(metrics or {}),
            route=route,
            budget=budget,
            monotonicity=monotonicity,
            leakage=leakage,
        )
        rows.append(row)
    return rows


def _optimizer_result_row(
    *,
    window: str,
    spec: dict[str, Any],
    pool_variant: str,
    selector: dict[str, Any] | None = None,
    feature_count: int,
    pool_rows: list[dict[str, Any]],
    recall: dict[str, Any],
    metrics: dict[str, Any],
    route: dict[str, Any],
    budget: dict[str, Any],
    monotonicity: float,
    leakage: bool,
) -> dict[str, Any]:
    active_recall = _num(metrics.get("recall_at_active_budget", metrics.get("best_oracle_in_active_share")))
    recall_at_32 = _num(metrics.get("recall_at_32", recall.get("recall_at_32")))
    top_decile = _num(metrics.get("top_decile_oracle_recall"))
    best_pool = _num(metrics.get("best_oracle_in_pool_share"))
    route_share = _num(route.get("route_eligible_share"))
    avg_pool_size = _num(metrics.get("avg_pool_size"))
    avg_top1_oracle = _num(metrics.get("avg_top1_oracle_score"))
    top1_positive_share = _num(metrics.get("top1_positive_day_share"))
    top1_quality = max(-1.0, min(avg_top1_oracle / 10.0, 2.0))
    overwide_penalty = min(avg_pool_size / 64.0, 2.0) * 0.08 + max(0.0, avg_pool_size - _num(metrics.get("avg_active_count"))) / 64.0 * 0.04
    leakage_penalty = 2.0 if leakage else 0.0
    active_budget_penalty = _num(budget.get("active_budget_violation_share")) * 1.0
    monotonicity_bonus = 0.20 * max(monotonicity, 0.0)
    monotonicity_penalty = 0.20 * max(-monotonicity, 0.0)
    source_score = (
        active_recall
        + recall_at_32
        + top_decile
        + 0.50 * best_pool
        + 0.20 * route_share
        + 0.20 * top1_quality
        + 0.10 * top1_positive_share
        + monotonicity_bonus
        - monotonicity_penalty
        - overwide_penalty
        - leakage_penalty
        - active_budget_penalty
    )
    selector_spec = dict(selector or _selector_spec("structural_first30_causal_tiebreak"))
    selector_name = str(selector_spec.get("name") or "")
    variant_id = stable_signature([spec.get("source_variant_id"), pool_variant, selector_name])[:20]
    return {
        "window": window,
        "optimizer_variant_id": variant_id,
        "source_variant_id": spec.get("source_variant_id"),
        "pool_variant": pool_variant,
        "selector_variant": selector_name,
        "selector_spec": selector_spec,
        "source_spec": dict(spec),
        "mutations": dict(spec.get("mutations") or {}),
        "feature_count": int(feature_count),
        "pool_row_count": len(pool_rows),
        "oracle_label_available": bool(recall.get("oracle_label_available")),
        "score_uses_ex_post_labels": bool(leakage),
        "train_only_selection_metric": "source_score" if window == "train" else "",
        "source_score": float(source_score),
        "recall_at_active_budget": float(active_recall),
        "recall_at_32": float(recall_at_32),
        "top_decile_oracle_recall": float(top_decile),
        "best_oracle_in_pool_share": float(best_pool),
        "best_oracle_in_active_share": float(_num(metrics.get("best_oracle_in_active_share"))),
        "avg_top1_oracle_score": float(avg_top1_oracle),
        "top1_net_r_sum": float(_num(metrics.get("top1_net_r_sum"))),
        "top1_positive_day_share": float(top1_positive_share),
        "top1_quality_component": float(top1_quality),
        "avg_pool_size": float(avg_pool_size),
        "avg_active_count": float(_num(metrics.get("avg_active_count"))),
        "monotonicity_score": float(monotonicity),
        "overwide_pool_penalty": float(overwide_penalty),
        "leakage_penalty": float(leakage_penalty),
        **{f"route_{key}": value for key, value in route.items()},
        **{f"budget_{key}": value for key, value in budget.items()},
    }


def _filter_structural_optimizer_rows(rows: Iterable[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    min_structural = float(spec.get("min_structural_campaign_score") or 0.0)
    min_rs = float(spec.get("min_rs_percentile") or 0.0)
    min_sector_daily = float(spec.get("min_sector_daily_score_pct") or 0.0)
    min_sector_participation = float(spec.get("min_sector_participation") or 0.0)
    max_box_range = float(spec.get("max_box_range_pct") or 0.0)
    for source in rows:
        row = dict(source)
        if _num(row.get("structural_campaign_score")) < min_structural:
            continue
        if _feature_num(row, "relative_strength_pct", "rs_percentile") < min_rs:
            continue
        if _feature_num(row, "sector_daily_score_pct") < min_sector_daily:
            continue
        if _feature_num(row, "sector_participation", "sector_daily_participation") < min_sector_participation:
            continue
        if max_box_range > 0.0 and _feature_num(row, "campaign_box_range_pct", "box_range_pct") > max_box_range:
            continue
        out.append(row)
    return out


def _source_frontier_rows(rows: Iterable[dict[str, Any]], frontier_count: int) -> list[dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[str(row.get("trade_date") or "")].append(dict(row))
    out: list[dict[str, Any]] = []
    cap = max(int(frontier_count or 0), 1)
    for day, items in sorted(by_day.items()):
        for rank, row in enumerate(sorted(items, key=_structural_rank_key)[:cap], start=1):
            out.append({**row, "optimizer_source_rank": rank, "optimizer_source_frontier_count": cap, "optimizer_trade_date": day})
    out.sort(key=lambda item: (str(item.get("trade_date") or ""), int(item.get("optimizer_source_rank") or 0), str(item.get("symbol") or "")))
    return out


def _select_optimizer_shortlist(rows: list[dict[str, Any]], shortlist_size: int) -> list[dict[str, Any]]:
    eligible = [dict(row) for row in rows if not bool(row.get("score_uses_ex_post_labels"))]
    eligible.sort(key=_optimizer_sort_key)
    pareto_candidates = eligible[: min(len(eligible), 750)]
    pareto = _pareto_front(pareto_candidates)
    pareto_ids = {str(item.get("optimizer_variant_id") or "") for item in pareto}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in [*pareto, *eligible]:
        key = str(source.get("optimizer_variant_id") or "")
        if key in seen:
            continue
        seen.add(key)
        row = dict(source)
        row["pareto_selected"] = key in pareto_ids
        row["train_rank"] = len(selected) + 1
        selected.append(row)
        if len(selected) >= max(1, int(shortlist_size or 1)):
            break
    return selected


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    maximize = (
        "recall_at_active_budget",
        "recall_at_32",
        "top_decile_oracle_recall",
        "best_oracle_in_pool_share",
        "avg_top1_oracle_score",
        "top1_positive_day_share",
        "monotonicity_score",
        "route_route_eligible_share",
    )
    minimize = ("avg_pool_size", "budget_active_budget_violation_share")
    front: list[dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if _dominates(other, row, maximize=maximize, minimize=minimize):
                dominated = True
                break
        if not dominated:
            front.append(row)
    front.sort(key=_optimizer_sort_key)
    return front


def _dominates(a: dict[str, Any], b: dict[str, Any], *, maximize: tuple[str, ...], minimize: tuple[str, ...]) -> bool:
    better = False
    for key in maximize:
        av = _num(a.get(key))
        bv = _num(b.get(key))
        if av < bv - 1e-12:
            return False
        better = better or av > bv + 1e-12
    for key in minimize:
        av = _num(a.get(key))
        bv = _num(b.get(key))
        if av > bv + 1e-12:
            return False
        better = better or av < bv - 1e-12
    return better


def _optimizer_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, str]:
    return (
        -_num(row.get("source_score")),
        -_num(row.get("recall_at_active_budget")),
        -_num(row.get("avg_top1_oracle_score")),
        -_num(row.get("recall_at_32")),
        -_num(row.get("top_decile_oracle_recall")),
        -_num(row.get("monotonicity_score")),
        _num(row.get("avg_pool_size")),
        str(row.get("optimizer_variant_id") or ""),
    )


def _route_conversion_summary(
    pool_rows: list[dict[str, Any]],
    *,
    cache: dict[tuple[str, str], tuple[list[str], str]] | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    eligible = 0
    blockers: dict[str, int] = {}
    for row in pool_rows:
        key = (str(row.get("trade_date") or "")[:10], str(row.get("symbol") or ""))
        cached = cache.get(key) if cache is not None else None
        if cached is None:
            passed_modes = []
            first_reason = ""
            for route in CAMPAIGN_ROUTE_SPECS:
                ok, reason = _route_static_eligible(row, route)
                if ok:
                    passed_modes.append(str(route["name"]))
                elif not first_reason:
                    first_reason = reason
            if cache is not None:
                cache[key] = (passed_modes, first_reason)
        else:
            passed_modes, first_reason = cached
        if passed_modes:
            eligible += 1
            for mode in passed_modes:
                counts[mode] = counts.get(mode, 0) + 1
        else:
            blockers[first_reason or "not_route_eligible"] = blockers.get(first_reason or "not_route_eligible", 0) + 1
    return {
        "route_eligible_candidate_count": eligible,
        "route_eligible_share": eligible / max(len(pool_rows), 1),
        "route_eligible_counts_by_mode": counts,
        "top_route_blockers": sorted(blockers.items(), key=lambda item: (-item[1], item[0]))[:8],
    }


def _route_static_eligible(row: dict[str, Any], route: dict[str, Any]) -> tuple[bool, str]:
    source = str(route.get("level_source") or "")
    level = _route_level(row, source)
    if level <= 0.0:
        return False, f"missing_{source}"
    if _num(row.get("first30_confirmation_score")) <= 0.0:
        return False, "missing_first30_confirmation"
    min_closes = int(route.get("min_reclaim_closes") or 1)
    if min_closes >= 2 and int(_num(row.get("first30_breakout_acceptance_closes"))) < min_closes:
        return False, "missing_two_close_breakout_acceptance"
    if source == "campaign_avwap" and not (bool(row.get("first30_above_campaign_avwap")) or _num(row.get("campaign_avwap_distance_pct")) >= -0.015):
        return False, "not_near_campaign_avwap"
    return True, "eligible"


def _route_level(row: dict[str, Any], source: str) -> float:
    return {
        "campaign_avwap": _feature_num(row, "campaign_avwap"),
        "campaign_box_high": _feature_num(row, "campaign_box_high"),
        "campaign_box_mid": _feature_num(row, "campaign_box_mid"),
        "campaign_breakout_level": _feature_num(row, "campaign_breakout_level"),
    }.get(source, 0.0)


def build_alcb_delta_diagnostics(
    train_rows: Iterable[dict[str, Any]],
    holdout_rows: Iterable[dict[str, Any]],
    *,
    train_oracle_rows: Iterable[dict[str, Any]] | None = None,
    holdout_oracle_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version": ALCB_DELTA_DIAGNOSTICS_VERSION,
        "purpose": "quantify_differences_from_successful_alcb_structural_box_and_or_breakout_conversion",
        "oracle_contract": "oracle_rows_are_ex_post_labels_only_never_live_features",
        "reference_deltas_tested": [
            "ALCB strict compression requires containment>=0.80 and box_height/ATR50<=1.10",
            "ALCB winners were mostly OR/PDH/combined breakout conversions after structural surfacing",
            "KALCB active budget may miss names already present in the structural frontier",
        ],
        "train": _alcb_delta_window(train_rows, train_oracle_rows or (), "train"),
        "holdout": _alcb_delta_window(holdout_rows, holdout_oracle_rows or (), "holdout"),
    }


def _alcb_delta_window(
    feature_rows: Iterable[dict[str, Any]],
    oracle_rows: Iterable[dict[str, Any]],
    window: str,
) -> dict[str, Any]:
    features = [dict(row) for row in feature_rows]
    oracle = _best_oracle_by_key(oracle_rows)
    best_by_day = _best_oracle_by_day(oracle.values())
    top_decile = _top_decile_oracle_keys(oracle.values())
    variants = {
        name: _alcb_variant_metrics(features, [row for row in features if predicate(row)], oracle, best_by_day, top_decile, description)
        for name, description, predicate in _alcb_variant_specs()
    }
    return {
        "window": window,
        "feature_row_count": len(features),
        "feature_day_count": len({str(row.get("trade_date") or "") for row in features}),
        "oracle_label_available": bool(oracle),
        "oracle_row_count": len(oracle),
        "best_oracle_day_count": len(best_by_day),
        "oracle_route_family_counts": _counts(row.get("route_family") for row in oracle.values()),
        "best_oracle_route_family_counts": _counts(row.get("route_family") for row in best_by_day.values()),
        "field_coverage": _alcb_field_coverage(features),
        "proxy_variants": variants,
        "active_frontier_miss": _alcb_active_frontier_miss(features, best_by_day),
        "first30_rel_volume_buckets": _alcb_numeric_buckets(
            features,
            oracle,
            "first30_rel_volume",
            (("lt1", None, 1.0), ("1_to_1p5", 1.0, 1.5), ("1p5_to_2", 1.5, 2.0), ("2_to_3", 2.0, 3.0), ("gte3", 3.0, None)),
        ),
        "first30_signal_cpr_buckets": _alcb_numeric_buckets(
            features,
            oracle,
            "first30_signal_cpr",
            (("lt0p4", None, 0.40), ("0p4_to_0p6", 0.40, 0.60), ("0p6_to_0p7", 0.60, 0.70), ("gte0p7", 0.70, None)),
        ),
        "campaign_avwap_distance_buckets": _alcb_numeric_buckets(
            features,
            oracle,
            "campaign_avwap_distance_pct",
            (("below_avwap", None, 0.0), ("0_to_0p5pct", 0.0, 0.005), ("0p5_to_1pct", 0.005, 0.010), ("1_to_2pct", 0.010, 0.020), ("gte2pct", 0.020, None)),
        ),
    }


def _alcb_variant_specs() -> tuple[tuple[str, str, Any], ...]:
    def box_containment(row: dict[str, Any]) -> bool:
        return _feature_num(row, "campaign_box_containment") >= 0.80

    def strict_box(row: dict[str, Any]) -> bool:
        return box_containment(row) and 0.0 < _feature_num(row, "campaign_box_atr_ratio") <= 1.10

    def tight_box(row: dict[str, Any]) -> bool:
        tier = str(row.get("campaign_box_tier") or "").lower()
        return box_containment(row) and (tier in {"tight", "balanced"} or _feature_num(row, "campaign_box_squeeze_pct") <= 0.30)

    def prior_breakout(row: dict[str, Any]) -> bool:
        state = str(row.get("campaign_state") or "")
        return state in {"breakout_watch", "first30_confirmed"} or _feature_num(row, "campaign_breakout_displacement") >= 0.0

    def first30_momentum(row: dict[str, Any]) -> bool:
        return _feature_num(row, "first30_rel_volume") >= 2.0 and _feature_num(row, "first30_signal_cpr") >= 0.60

    def breakout_acceptance(row: dict[str, Any]) -> bool:
        return bool(row.get("first30_breakout_acceptance")) or bool(row.get("first30_breakout_confirmation"))

    return (
        ("all_structural_candidates", "KALCB current structural source universe", lambda row: True),
        ("alcb_box_containment80", "ALCB containment floor without squeeze cap", box_containment),
        ("alcb_strict_box", "ALCB-style containment>=0.80 and box/ATR<=1.10", strict_box),
        ("alcb_tight_or_balanced_box", "KALCB box tier proxy for ALCB tight/balanced compression", tight_box),
        ("prior_daily_breakout_watch", "prior-daily structural breakout watch state", prior_breakout),
        ("first30_rvol2_cpr60", "ALCB momentum quality proxy before campaign-level breakout", first30_momentum),
        ("first30_rvol3_cpr60", "hotter ALCB RVOL proxy before campaign-level breakout", lambda row: first30_momentum(row) and _feature_num(row, "first30_rel_volume") >= 3.0),
        ("or_breakout_proxy", "campaign-level two-close/confirmation breakout plus RVOL>=2 CPR>=0.60", lambda row: first30_momentum(row) and breakout_acceptance(row)),
        ("or_breakout_proxy_rvol3", "campaign breakout proxy with RVOL>=3", lambda row: first30_momentum(row) and _feature_num(row, "first30_rel_volume") >= 3.0 and breakout_acceptance(row)),
        ("or_breakout_avwap_cap_0p5pct", "ALCB combined-breakout style AVWAP proximity cap", lambda row: first30_momentum(row) and breakout_acceptance(row) and 0.0 <= _feature_num(row, "campaign_avwap_distance_pct") <= 0.005),
        ("or_breakout_avwap_cap_1pct", "looser AVWAP proximity cap for a hotter Korean tape", lambda row: first30_momentum(row) and breakout_acceptance(row) and 0.0 <= _feature_num(row, "campaign_avwap_distance_pct") <= 0.010),
        ("strict_box_or_breakout_proxy", "strict ALCB box plus first30 campaign breakout proxy", lambda row: strict_box(row) and first30_momentum(row) and breakout_acceptance(row)),
    )


def _alcb_variant_metrics(
    features: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    oracle_by_key: dict[tuple[str, str], dict[str, Any]],
    best_by_day: dict[str, dict[str, Any]],
    top_decile: set[tuple[str, str]],
    description: str,
) -> dict[str, Any]:
    del features
    keys = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in selected}
    active_keys = {
        (str(row.get("trade_date") or ""), str(row.get("symbol") or ""))
        for row in selected
        if str(row.get("structural_source_role") or "") == "active" or bool(row.get("pool_active"))
    }
    oracle_rows = [oracle_by_key[key] for key in keys if key in oracle_by_key]
    oracle_days = sorted(best_by_day)
    route = _route_conversion_summary(selected)
    return {
        "description": description,
        "row_count": len(selected),
        "day_count": len({str(row.get("trade_date") or "") for row in selected}),
        "avg_rows_per_day": _avg_day_count(selected),
        "active_share": len(active_keys) / max(len(keys), 1),
        "avg_structural_rank": _avg(row.get("structural_source_rank") for row in selected),
        "avg_structural_campaign_score": _avg(row.get("structural_campaign_score") for row in selected),
        "avg_first30_rel_volume": _avg(row.get("first30_rel_volume") for row in selected),
        "avg_first30_signal_cpr": _avg(row.get("first30_signal_cpr") for row in selected),
        "oracle_labeled_count": len(oracle_rows),
        "avg_oracle_score": _avg(_oracle_rank_tuple(row)[0] for row in oracle_rows),
        "avg_mfe_r": _avg(row.get("mfe_r") for row in oracle_rows),
        "avg_net_r": _avg(row.get("net_r") for row in oracle_rows),
        "oracle_route_family_counts": _counts(row.get("route_family") for row in oracle_rows),
        "best_oracle_in_proxy_share": sum(1 for day in oracle_days if (day, str(best_by_day[day].get("symbol") or "")) in keys) / max(len(oracle_days), 1),
        "best_oracle_in_active_proxy_share": sum(1 for day in oracle_days if (day, str(best_by_day[day].get("symbol") or "")) in active_keys) / max(len(oracle_days), 1),
        "top_decile_oracle_recall": len(top_decile & keys) / max(len(top_decile), 1),
        "route_eligible_share": route["route_eligible_share"],
        "route_eligible_counts_by_mode": route["route_eligible_counts_by_mode"],
        "top_route_blockers": route["top_route_blockers"],
    }


def _alcb_field_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for field in (
        "campaign_box_containment",
        "campaign_box_atr_ratio",
        "campaign_box_squeeze_pct",
        "campaign_box_tier",
        "campaign_avwap",
        "campaign_breakout_level",
        "first30_rel_volume",
        "first30_signal_cpr",
        "first30_vwap_ret",
    ):
        present = [row.get(field) for row in rows if row.get(field) not in (None, "")]
        numeric = [_num(value) for value in present if _finite(value)]
        out[field] = {
            "present_count": len(present),
            "coverage": len(present) / max(len(rows), 1),
            "avg": mean(numeric) if numeric else 0.0,
        }
    return out


def _alcb_active_frontier_miss(
    features: list[dict[str, Any]],
    best_by_day: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    feature_by_key = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")): row for row in features}
    active_keys = {
        (str(row.get("trade_date") or ""), str(row.get("symbol") or ""))
        for row in features
        if str(row.get("structural_source_role") or "") == "active" or bool(row.get("pool_active"))
    }
    misses: list[dict[str, Any]] = []
    for day, oracle_row in best_by_day.items():
        key = (day, str(oracle_row.get("symbol") or ""))
        feature = feature_by_key.get(key)
        if feature is None or key in active_keys:
            continue
        misses.append(
            {
                "trade_date": day,
                "symbol": key[1],
                "oracle_score": _oracle_rank_tuple(oracle_row)[0],
                "mfe_r": _num(oracle_row.get("mfe_r")),
                "net_r": _num(oracle_row.get("net_r")),
                "route_family": str(oracle_row.get("route_family") or ""),
                "structural_source_rank": int(_num(feature.get("structural_source_rank"))),
                "structural_campaign_score": _num(feature.get("structural_campaign_score")),
                "first30_rel_volume": _num(feature.get("first30_rel_volume")),
                "first30_signal_cpr": _num(feature.get("first30_signal_cpr")),
                "first30_breakout_acceptance": bool(feature.get("first30_breakout_acceptance")),
                "campaign_box_atr_ratio": _num(feature.get("campaign_box_atr_ratio")),
                "campaign_box_containment": _num(feature.get("campaign_box_containment")),
                "campaign_avwap_distance_pct": _num(feature.get("campaign_avwap_distance_pct")),
            }
        )
    misses.sort(key=lambda row: (_num(row.get("oracle_score")), _num(row.get("mfe_r"))), reverse=True)
    return {
        "best_oracle_day_count": len(best_by_day),
        "missed_best_in_frontier_count": len(misses),
        "missed_best_in_frontier_share": len(misses) / max(len(best_by_day), 1),
        "avg_missed_structural_rank": _avg(row.get("structural_source_rank") for row in misses),
        "missed_rank_buckets": {
            "rank_1_to_8": sum(1 for row in misses if 1 <= int(row.get("structural_source_rank") or 0) <= 8),
            "rank_9_to_16": sum(1 for row in misses if 9 <= int(row.get("structural_source_rank") or 0) <= 16),
            "rank_17_to_32": sum(1 for row in misses if 17 <= int(row.get("structural_source_rank") or 0) <= 32),
            "rank_gt32": sum(1 for row in misses if int(row.get("structural_source_rank") or 0) > 32),
        },
        "sample_top_missed": misses[:12],
    }


def _alcb_numeric_buckets(
    rows: list[dict[str, Any]],
    oracle_by_key: dict[tuple[str, str], dict[str, Any]],
    field: str,
    buckets: Iterable[tuple[str, float | None, float | None]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, lower, upper in buckets:
        selected = [
            row
            for row in rows
            if (lower is None or _feature_num(row, field) >= lower)
            and (upper is None or _feature_num(row, field) < upper)
        ]
        keys = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in selected}
        oracle_rows = [oracle_by_key[key] for key in keys if key in oracle_by_key]
        out[label] = {
            "row_count": len(selected),
            "avg_structural_campaign_score": _avg(row.get("structural_campaign_score") for row in selected),
            "active_share": sum(1 for row in selected if str(row.get("structural_source_role") or "") == "active") / max(len(selected), 1),
            "oracle_labeled_count": len(oracle_rows),
            "avg_oracle_score": _avg(_oracle_rank_tuple(row)[0] for row in oracle_rows),
            "avg_mfe_r": _avg(row.get("mfe_r") for row in oracle_rows),
            "avg_net_r": _avg(row.get("net_r") for row in oracle_rows),
        }
    return out


def _active_budget_summary(pool_rows: list[dict[str, Any]], active_budget_by_day: dict[str, int] | None, cfg: KALCBConfig) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        by_day[str(row.get("trade_date") or "")].append(row)
    violations = 0
    for day, rows in by_day.items():
        expected = int((active_budget_by_day or {}).get(day, cfg.ws_budget) if active_budget_by_day is not None else cfg.ws_budget)
        active = sum(1 for row in rows if bool(row.get("pool_active")))
        violations += int(active > max(expected, 0))
    return {
        "active_budget_source": "incumbent_daily_active_count" if active_budget_by_day is not None else "ws_budget",
        "active_budget_day_count": len(by_day),
        "active_budget_violation_count": violations,
        "active_budget_violation_share": violations / max(len(by_day), 1),
    }


def _monotonicity_score(bucket_summary: dict[str, dict[str, Any]]) -> float:
    observed: list[tuple[str, float]] = []
    for label, row in sorted(bucket_summary.items()):
        if int(row.get("oracle_labeled_count") or 0) <= 0:
            continue
        value = _num(row.get("avg_oracle_score"))
        if value == 0.0:
            value = _num(row.get("avg_mfe_r"))
        observed.append((label, value))
    if len(observed) < 2:
        return 0.0
    good = 0
    bad = 0
    for lower_index in range(len(observed)):
        for upper_index in range(lower_index + 1, len(observed)):
            if observed[upper_index][1] + 1e-12 >= observed[lower_index][1]:
                good += 1
            else:
                bad += 1
    return (good - bad) / max(good + bad, 1)


def _materialize_pool_variants(
    feature_rows: Iterable[dict[str, Any]],
    active_budget_by_day: dict[str, int] | None,
    cfg: KALCBConfig,
) -> list[dict[str, Any]]:
    features = [dict(row) for row in feature_rows]
    rows: list[dict[str, Any]] = []
    for spec in POOL_VARIANT_SPECS:
        rows.extend(
            split_structural_campaign_pools(
                features,
                active_budget_by_day,
                cfg,
                frontier_size=spec.get("frontier_size"),
                variant_name=str(spec.get("name")),
            )
        )
    return rows


def _research_snapshot_index_rows(feature_rows: Iterable[dict[str, Any]], pool_rows: Iterable[dict[str, Any]], window: str) -> list[dict[str, Any]]:
    features_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_pool_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        features_by_day[str(row.get("trade_date") or "")].append(dict(row))
    for row in pool_rows:
        if str(row.get("pool_variant") or "") == "structural_active_budget":
            active_pool_by_day[str(row.get("trade_date") or "")].append(dict(row))
    out: list[dict[str, Any]] = []
    for day in sorted(features_by_day):
        items = sorted(features_by_day[day], key=_structural_rank_key)
        pool = sorted(active_pool_by_day.get(day, ()), key=lambda item: int(item.get("pool_rank") or 0))
        active_symbols = [str(row.get("symbol") or "") for row in pool if bool(row.get("pool_active"))]
        frontier_symbols = [str(row.get("symbol") or "") for row in pool]
        out.append(
            {
                "window": window,
                "trade_date": day,
                "status": "research_snapshot_index",
                "source": "daily_selection_from_snapshot_structural_campaign_metadata",
                "item_symbols": [str(row.get("symbol") or "") for row in items],
                "active_symbols": active_symbols,
                "frontier_symbols": frontier_symbols,
                "overflow_symbols": [str(row.get("symbol") or "") for row in items if str(row.get("symbol") or "") not in set(active_symbols)],
                "active_budget_for_day": len(active_symbols),
                "item_count": len(items),
                "frontier_count": len(frontier_symbols),
                "overflow_count": max(len(items) - len(active_symbols), 0),
            }
        )
    return out


def summarize_structural_recall(
    feature_rows: Iterable[dict[str, Any]],
    pool_rows: Iterable[dict[str, Any]],
    *,
    oracle_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    features = [dict(row) for row in feature_rows]
    pools = [dict(row) for row in pool_rows]
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        by_day[str(row.get("trade_date") or "")].append(row)
    pools_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pools:
        pools_by_variant[str(row.get("pool_variant") or "structural_active_budget")].append(row)
    oracle = _best_oracle_by_key(oracle_rows or ())
    top_decile_oracle = _top_decile_oracle_keys(oracle.values())
    out: dict[str, Any] = {
        "feature_row_count": len(features),
        "pool_row_count": len(pools),
        "active_row_count": sum(1 for row in pools if bool(row.get("pool_active"))),
        "pool_variant_count": len(pools_by_variant),
        "avg_structural_campaign_score": _avg(row.get("structural_campaign_score") for row in features),
        "campaign_state_counts": _counts(row.get("campaign_state") for row in features),
        "score_uses_ex_post_labels": any(bool(row.get("score_uses_ex_post_labels")) for row in features),
        "oracle_label_available": bool(oracle),
        "oracle_row_count": len(oracle),
        "oracle_day_count": len({day for day, _symbol in oracle}),
        "structural_score_buckets": _score_bucket_summary(features, oracle),
    }
    if not oracle:
        out["recall_contract"] = "no_oracle_labels_provided_pool_coverage_only"
        out["variants"] = {
            name: _pool_coverage_summary(features, rows, by_day)
            for name, rows in sorted(pools_by_variant.items())
        }
        return out
    out["recall_contract"] = "oracle_rows_used_as_ex_post_labels_only"
    for k in RECALL_KS:
        hits = 0
        days = 0
        for day, oracle_row in _best_oracle_by_day(oracle.values()).items():
            rows = by_day.get(day, ())
            if not rows:
                continue
            days += 1
            top_symbols = {str(row.get("symbol") or "") for row in sorted(rows, key=_structural_rank_key)[:k]}
            hits += int(str(oracle_row.get("symbol") or "") in top_symbols)
        out[f"recall_at_{k}"] = hits / max(days, 1)
    variants = {
        name: _oracle_pool_recall_summary(features, rows, oracle, top_decile_oracle, by_day)
        for name, rows in sorted(pools_by_variant.items())
    }
    out["variants"] = variants
    out["best_variant_by_best_oracle_in_pool_share"] = _best_variant(variants, "best_oracle_in_pool_share")
    out["best_variant_by_top_decile_oracle_recall"] = _best_variant(variants, "top_decile_oracle_recall")
    return out


def _pool_coverage_summary(
    features: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    by_day: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pool_keys = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in pool_rows}
    active_keys = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in pool_rows if bool(row.get("pool_active"))}
    out: dict[str, Any] = {
        "oracle_label_available": False,
        "recall_contract": "pool_coverage_without_oracle_labels",
        "pool_row_count": len(pool_rows),
        "active_row_count": len(active_keys),
        "avg_pool_size": _avg_day_count(pool_rows),
        "avg_active_count": _avg_day_count([row for row in pool_rows if bool(row.get("pool_active"))]),
        "avg_structural_campaign_score_in_pool": _avg(row.get("structural_campaign_score") for row in pool_rows),
        "pool_feature_coverage": len(pool_keys) / max(len({(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in features}), 1),
    }
    for k in RECALL_KS:
        shares: list[float] = []
        for day, rows in by_day.items():
            ordered = sorted(rows, key=_structural_rank_key)[:k]
            if ordered:
                shares.append(
                    sum(1 for row in ordered if (day, str(row.get("symbol") or "")) in pool_keys)
                    / max(len(ordered), 1)
                )
        out[f"pool_top_{k}_coverage"] = mean(shares) if shares else 0.0
    return out


def _oracle_pool_recall_summary(
    features: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    oracle_by_key: dict[tuple[str, str], dict[str, Any]],
    top_decile_oracle: set[tuple[str, str]],
    by_day: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    del features
    pool_keys = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in pool_rows}
    active_keys = {(str(row.get("trade_date") or ""), str(row.get("symbol") or "")) for row in pool_rows if bool(row.get("pool_active"))}
    best_by_day = _best_oracle_by_day(oracle_by_key.values())
    oracle_days = sorted(best_by_day)
    in_pool_oracle_values = [
        oracle_by_key[key]
        for key in pool_keys
        if key in oracle_by_key
    ]
    out: dict[str, Any] = {
        "oracle_label_available": True,
        "pool_row_count": len(pool_rows),
        "active_row_count": len(active_keys),
        "avg_pool_size": _avg_day_count(pool_rows),
        "avg_active_count": _avg_day_count([row for row in pool_rows if bool(row.get("pool_active"))]),
        "best_oracle_in_pool_share": sum(1 for day in oracle_days if (day, str(best_by_day[day].get("symbol") or "")) in pool_keys) / max(len(oracle_days), 1),
        "best_oracle_in_active_share": sum(1 for day in oracle_days if (day, str(best_by_day[day].get("symbol") or "")) in active_keys) / max(len(oracle_days), 1),
        "top_decile_oracle_recall": len(top_decile_oracle & pool_keys) / max(len(top_decile_oracle), 1),
        "avg_in_pool_mfe_r": _avg(row.get("mfe_r") for row in in_pool_oracle_values),
        "avg_in_pool_net_r": _avg(row.get("net_r") for row in in_pool_oracle_values),
    }
    for k in RECALL_KS:
        hits = 0
        days = 0
        for day, oracle_row in best_by_day.items():
            rows = by_day.get(day, ())
            if not rows:
                continue
            days += 1
            top_symbols = {str(row.get("symbol") or "") for row in sorted(rows, key=_structural_rank_key)[:k]}
            hits += int(str(oracle_row.get("symbol") or "") in top_symbols)
        out[f"recall_at_{k}"] = hits / max(days, 1)
    out["recall_at_active_budget"] = out["best_oracle_in_active_share"]
    return out


def _best_oracle_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        day = str(row.get("trade_date") or row.get("entry_date") or "")[:10]
        symbol = str(row.get("symbol") or "")
        if not day or not symbol:
            continue
        row["trade_date"] = day
        row["symbol"] = symbol
        key = (day, symbol)
        if key not in out or _oracle_rank_tuple(row) > _oracle_rank_tuple(out[key]):
            out[key] = row
    return out


def _best_oracle_by_day(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        day = str(row.get("trade_date") or row.get("entry_date") or "")[:10]
        if not day:
            continue
        if day not in out or _oracle_rank_tuple(row) > _oracle_rank_tuple(out[day]):
            out[day] = row
    return out


def _top_decile_oracle_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        day = str(row.get("trade_date") or row.get("entry_date") or "")[:10]
        symbol = str(row.get("symbol") or "")
        if day and symbol:
            by_day[day].append(dict(row))
    out: set[tuple[str, str]] = set()
    for day, items in by_day.items():
        ordered = sorted(items, key=_oracle_rank_tuple, reverse=True)
        limit = max(1, math.ceil(0.10 * len(ordered)))
        out.update((day, str(row.get("symbol") or "")) for row in ordered[:limit])
    return out


def _oracle_rank_tuple(row: dict[str, Any]) -> tuple[float, float, float, str]:
    primary = 0.0
    for key in ORACLE_SCORE_KEYS:
        if _finite(row.get(key)):
            primary = float(row.get(key))
            break
    return (
        primary,
        _num(row.get("net_r")),
        _num(row.get("mfe_r")),
        str(row.get("symbol") or ""),
    )


def _score_bucket_summary(
    feature_rows: list[dict[str, Any]],
    oracle_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        score = max(0.0, min(10.0, _num(row.get("structural_campaign_score"))))
        lower = min(int(score // 2) * 2, 8)
        label = f"{lower:02d}_{lower + 2:02d}"
        buckets[label].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(buckets.items()):
        oracle_rows = [
            oracle_by_key[(str(row.get("trade_date") or ""), str(row.get("symbol") or ""))]
            for row in rows
            if (str(row.get("trade_date") or ""), str(row.get("symbol") or "")) in oracle_by_key
        ]
        summary[label] = {
            "count": len(rows),
            "avg_structural_campaign_score": _avg(row.get("structural_campaign_score") for row in rows),
            "avg_first30_confirmation_score": _avg(row.get("first30_confirmation_score") for row in rows),
            "oracle_labeled_count": len(oracle_rows),
            "avg_oracle_score": _avg(_oracle_rank_tuple(row)[0] for row in oracle_rows),
            "avg_mfe_r": _avg(row.get("mfe_r") for row in oracle_rows),
            "avg_net_r": _avg(row.get("net_r") for row in oracle_rows),
        }
    return summary


def _avg_day_count(rows: Iterable[dict[str, Any]]) -> float:
    counts: dict[str, int] = {}
    for row in rows:
        day = str(row.get("trade_date") or "")
        counts[day] = counts.get(day, 0) + 1
    return mean(counts.values()) if counts else 0.0


def _best_variant(variants: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    if not variants:
        return {}
    name, row = max(variants.items(), key=lambda item: (float(item[1].get(metric) or 0.0), str(item[0])))
    return {"name": name, "metric": metric, "value": float(row.get(metric) or 0.0), "metrics": row}


def render_structural_campaign_report(
    recall: dict[str, Any],
    manifest: dict[str, Any],
    optimizer: dict[str, Any] | None = None,
    alcb_delta: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# KALCB Stage 09 Structural Campaign Surfacing",
        "",
        f"- Version: `{STRUCTURAL_CAMPAIGN_SURFACING_VERSION}`",
        f"- Usage contract: `{STRUCTURAL_CAMPAIGN_USAGE_CONTRACT}`",
        f"- Variants: {', '.join(str(item.get('name')) for item in manifest.get('variants') or [])}",
        "",
        "## Recall Summary",
        "",
    ]
    for window in ("train", "holdout"):
        row = dict(recall.get(window) or {})
        lines.append(
            f"- {window}: features={int(row.get('feature_row_count') or 0)}, "
            f"pool={int(row.get('pool_row_count') or 0)}, "
            f"active={int(row.get('active_row_count') or 0)}, "
            f"avg_score={float(row.get('avg_structural_campaign_score') or 0.0):.2f}, "
            f"oracle={bool(row.get('oracle_label_available'))}, "
            f"contract={row.get('recall_contract') or 'unknown'}"
        )
    opt = dict(optimizer or {})
    if opt:
        best_train = dict(opt.get("best_train_variant") or {})
        best_holdout = dict(opt.get("best_holdout_frozen_variant") or {})
        lines.extend(
            [
                "",
                "## Train Optimizer",
                "",
                f"- Contract: `{opt.get('optimizer_contract') or 'unknown'}`",
                f"- Variants swept: {int(opt.get('train_variant_count') or 0)}",
                f"- Frozen shortlist: {int(opt.get('shortlist_size') or 0)}",
                f"- Best train: `{best_train.get('optimizer_variant_id') or ''}` "
                f"pool={best_train.get('pool_variant') or ''} "
                f"selector={best_train.get('selector_variant') or ''} "
                f"score={float(best_train.get('source_score') or 0.0):.3f} "
                f"active_recall={float(best_train.get('recall_at_active_budget') or 0.0):.3f} "
                f"recall32={float(best_train.get('recall_at_32') or 0.0):.3f}",
                f"- Frozen holdout rank1: `{best_holdout.get('optimizer_variant_id') or ''}` "
                f"pool={best_holdout.get('pool_variant') or ''} "
                f"selector={best_holdout.get('selector_variant') or ''} "
                f"score={float(best_holdout.get('source_score') or 0.0):.3f} "
                f"active_recall={float(best_holdout.get('recall_at_active_budget') or 0.0):.3f} "
                f"recall32={float(best_holdout.get('recall_at_32') or 0.0):.3f}",
                "",
                "| rank | variant | pool | selector | source_score | recall_active | recall32 | top1_oracle | route_share |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in opt.get("shortlist") or []:
            row = dict(row or {})
            lines.append(
                f"| {int(row.get('train_rank') or 0)} "
                f"| `{row.get('optimizer_variant_id') or ''}` "
                f"| {row.get('pool_variant') or ''} "
                f"| {row.get('selector_variant') or ''} "
                f"| {float(row.get('source_score') or 0.0):.3f} "
                f"| {float(row.get('recall_at_active_budget') or 0.0):.3f} "
                f"| {float(row.get('recall_at_32') or 0.0):.3f} "
                f"| {float(row.get('avg_top1_oracle_score') or 0.0):.2f} "
                f"| {float(row.get('route_route_eligible_share') or 0.0):.3f} |"
            )
    delta = dict(alcb_delta or {})
    if delta:
        lines.extend(["", "## ALCB Delta Diagnostics", ""])
        for window in ("train", "holdout"):
            row = dict(delta.get(window) or {})
            variants = dict(row.get("proxy_variants") or {})
            breakout = dict(variants.get("or_breakout_proxy") or {})
            strict = dict(variants.get("alcb_strict_box") or {})
            miss = dict(row.get("active_frontier_miss") or {})
            lines.append(
                f"- {window}: strict_box_rows={int(strict.get('row_count') or 0)}, "
                f"or_proxy_rows={int(breakout.get('row_count') or 0)}, "
                f"or_proxy_best_share={float(breakout.get('best_oracle_in_proxy_share') or 0.0):.3f}, "
                f"active_missed_best_share={float(miss.get('missed_best_in_frontier_share') or 0.0):.3f}"
            )
    return "\n".join(lines) + "\n"


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _structural_rank_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        -float(row.get("structural_campaign_score") or 0.0),
        -float(row.get("first30_confirmation_score") or 0.0),
        -float(row.get("causal_calibration_score") or 0.0),
        str(row.get("symbol") or ""),
    )


def _selector_spec(name: str) -> dict[str, Any]:
    for spec in SELECTOR_VARIANT_SPECS:
        if str(spec.get("name") or "") == str(name):
            return dict(spec)
    return dict(SELECTOR_VARIANT_SPECS[0])


def _selector_rank_key(row: dict[str, Any], selector: dict[str, Any] | None = None) -> tuple[Any, ...]:
    spec = dict(selector or _selector_spec("structural_first30_causal_tiebreak"))
    name = str(spec.get("name") or "")
    cache_key = f"_optimizer_selector_rank_key_{name}"
    cached = row.get(cache_key)
    if isinstance(cached, tuple):
        return cached
    structural = _structural_norm(row)
    first30 = _first30_norm(row)
    causal = _causal_norm(row)
    relvol = float(row.get("_optimizer_relvol_norm")) if row.get("_optimizer_relvol_norm") is not None else min(max(_feature_num(row, "first30_rel_volume") / 5.0, 0.0), 2.0)
    cpr = float(row.get("_optimizer_cpr_norm")) if row.get("_optimizer_cpr_norm") is not None else min(max(_feature_num(row, "first30_signal_cpr", "first30_signal_bar_cpr"), 0.0), 1.0)
    if name == "structural_first30":
        out = (-structural, -first30, -relvol, -cpr, str(row.get("symbol") or ""))
        row[cache_key] = out
        return out
    if name == "structural_first30_causal_tiebreak":
        out = (-structural, -first30, -causal, -relvol, -cpr, str(row.get("symbol") or ""))
        row[cache_key] = out
        return out
    if name == "first30_confirmation":
        out = (-first30, -relvol, -cpr, -structural, -causal, str(row.get("symbol") or ""))
        row[cache_key] = out
        return out
    out = (-_selector_score(row, spec), -structural, -first30, -causal, str(row.get("symbol") or ""))
    row[cache_key] = out
    return out


def _selector_score(row: dict[str, Any], selector: dict[str, Any] | None = None) -> float:
    spec = dict(selector or _selector_spec("structural_first30_causal_tiebreak"))
    name = str(spec.get("name") or "")
    cache_key = f"_optimizer_selector_score_{name}"
    cached = row.get(cache_key)
    if cached is not None:
        return float(cached)
    structural = _structural_norm(row)
    first30 = _first30_norm(row)
    causal = _causal_norm(row)
    if name == "structural_first30":
        score = 100.0 * structural + 10.0 * first30
        row[cache_key] = float(score)
        return float(score)
    if name == "structural_first30_causal_tiebreak":
        score = 100.0 * structural + 10.0 * first30 + 0.20 * causal
        row[cache_key] = float(score)
        return float(score)
    if name == "first30_confirmation":
        score = 100.0 * first30 + 10.0 * structural + 0.20 * causal
        row[cache_key] = float(score)
        return float(score)
    score = 100.0 * (
        float(spec.get("structural_weight", 0.0) or 0.0) * structural
        + float(spec.get("first30_weight", 0.0) or 0.0) * first30
        + float(spec.get("causal_weight", 0.0) or 0.0) * causal
    )
    row[cache_key] = float(score)
    return float(score)


def _structural_norm(row: dict[str, Any]) -> float:
    cached = row.get("_optimizer_structural_norm")
    if cached is not None:
        return float(cached)
    return max(0.0, min(_feature_num(row, "structural_campaign_score") / 10.0, 1.0))


def _first30_norm(row: dict[str, Any]) -> float:
    cached = row.get("_optimizer_first30_norm")
    if cached is not None:
        return float(cached)
    confirmation = max(_feature_num(row, "first30_confirmation_score"), 0.0)
    relvol = min(math.log1p(max(_feature_num(row, "first30_rel_volume"), 0.0)) / math.log1p(10.0), 1.0)
    cpr = max(0.0, min(_feature_num(row, "first30_signal_cpr", "first30_signal_bar_cpr"), 1.0))
    confirmation_norm = min(confirmation / 6.0, 1.0)
    return 0.60 * confirmation_norm + 0.25 * relvol + 0.15 * cpr


def _causal_norm(row: dict[str, Any]) -> float:
    cached = row.get("_optimizer_causal_norm")
    if cached is not None:
        return float(cached)
    return 0.5 + 0.5 * math.tanh(_feature_num(row, "causal_calibration_score") / 1.5)


def _causal_score_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = [dict(row) for row in rows]
    observed = [row for row in items if str(row.get("causal_score_source") or "")]
    return {
        "row_count": len(items),
        "scored_count": len(observed),
        "coverage": len(observed) / max(len(items), 1),
        "source_counts": _counts(row.get("causal_score_source") for row in observed),
    }


def _avg(values: Iterable[Any]) -> float:
    observed = [float(value) for value in values if _finite(value)]
    return mean(observed) if observed else 0.0


def _optional_num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite(value: Any) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _feature_num(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return _num(value, default)
    metadata = row.get("campaign_metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return _num(value, default)
        detail = metadata.get("campaign_selection_detail")
        if isinstance(detail, dict):
            for key in keys:
                value = detail.get(key)
                if value not in (None, ""):
                    return _num(value, default)
    return default


def _counts(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value or "missing")
        out[key] = out.get(key, 0) + 1
    return out


def _parse_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value)[:10])


def artifact_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    return stable_signature([dict(row) for row in rows])
