from __future__ import annotations

from pathlib import Path

import pytest

from backtests.strategies.kalcb.structural_campaign_surfacing import (
    _alcb_candidate_filter_reason,
    _alcb_breakout_replay_variants,
    _alcb_faithfulness_route_bundles,
    _alcb_faithfulness_variants,
    _alcb_replay_routes,
    optimize_structural_campaign_train,
    build_structural_campaign_surfacing_artifacts,
)
from strategy_kalcb.config import KALCBConfig


def test_structural_campaign_surfacing_writes_stage09_artifacts(tmp_path: Path) -> None:
    train = [_row("2026-01-05", "005930", score=7.0), _row("2026-01-05", "000660", score=5.0)]
    holdout = [_row("2026-01-06", "005930", score=6.0)]
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 1,
            "kalcb.frontier.rotation_slots": 0,
            "kalcb.research.structural_frontier_count": 2,
        }
    )

    summary = build_structural_campaign_surfacing_artifacts(train, holdout, output_dir=tmp_path, cfg=cfg, optimizer_grid=_tiny_grid())

    assert summary["usage_contract"] == "research_only_shared_daily_selection_source_no_oracle_live_features"
    assert summary["train"]["feature_row_count"] == 2
    assert summary["train"]["pool_row_count"] == 8
    assert summary["train"]["recall"]["score_uses_ex_post_labels"] is False
    assert summary["train"]["recall"]["oracle_label_available"] is False
    assert summary["train"]["recall"]["recall_contract"] == "no_oracle_labels_provided_pool_coverage_only"
    assert "structural_frontier32" in summary["train"]["recall"]["variants"]
    assert summary["train"]["recall"]["structural_score_buckets"]
    assert summary["optimizer"]["optimizer_contract"] == "train_sweep_selects_shortlist_holdout_scored_once_no_holdout_optimization_causal_tiebreaker_only"
    assert summary["optimizer"]["shortlist"]
    assert summary["alcb_delta_diagnostics"]["train"]["proxy_variants"]["or_breakout_proxy"]["row_count"] == 2
    for path in summary["artifact_paths"].values():
        assert Path(path).exists()


def test_structural_campaign_surfacing_uses_oracle_rows_only_as_labels(tmp_path: Path) -> None:
    train = [_row("2026-01-05", "005930", score=7.0), _row("2026-01-05", "000660", score=5.0)]
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 1,
            "kalcb.frontier.rotation_slots": 0,
            "kalcb.research.structural_frontier_count": 2,
        }
    )

    summary = build_structural_campaign_surfacing_artifacts(
        train,
        [],
        output_dir=tmp_path,
        cfg=cfg,
        optimizer_grid=_tiny_grid(),
        train_oracle_rows=[
            {"trade_date": "2026-01-05", "symbol": "000660", "oracle_score": 9.0, "mfe_r": 5.0, "net_r": 1.0}
        ],
    )

    recall = summary["train"]["recall"]
    assert recall["oracle_label_available"] is True
    assert recall["recall_at_8"] == pytest.approx(1.0)
    assert recall["variants"]["structural_active_budget"]["best_oracle_in_active_share"] == pytest.approx(0.0)
    assert recall["variants"]["structural_active_budget"]["best_oracle_in_pool_share"] == pytest.approx(1.0)
    assert recall["structural_score_buckets"]["04_06"]["oracle_labeled_count"] == 1


def test_structural_campaign_surfacing_preserves_zero_incumbent_active_budget(tmp_path: Path) -> None:
    train = [_row("2026-01-05", "005930", score=7.0), _row("2026-01-05", "000660", score=5.0)]

    summary = build_structural_campaign_surfacing_artifacts(
        train,
        [],
        output_dir=tmp_path,
        cfg=KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2, "kalcb.research.structural_frontier_count": 2}),
        optimizer_grid=_tiny_grid(),
        train_active_budget_by_day={"2026-01-05": 0},
    )

    active_variant = summary["train"]["recall"]["variants"]["structural_active_budget"]
    assert active_variant["active_row_count"] == 0
    assert active_variant["avg_active_count"] == pytest.approx(0.0)


def test_structural_campaign_optimizer_freezes_train_shortlist_before_holdout() -> None:
    train = [
        _row("2026-01-05", "005930", score=7.0),
        _row("2026-01-05", "000660", score=5.0),
        _row("2026-01-06", "035420", score=6.0),
    ]
    holdout = [_row("2026-01-07", "005930", score=7.0), _row("2026-01-07", "000660", score=5.0)]

    result = optimize_structural_campaign_train(
        train,
        holdout,
        cfg=KALCBConfig.from_mapping({"kalcb.session.ws_budget": 1, "kalcb.frontier.rotation_slots": 0}),
        train_active_budget_by_day={"2026-01-05": 1, "2026-01-06": 1},
        holdout_active_budget_by_day={"2026-01-07": 1},
        train_oracle_rows=[
            {"trade_date": "2026-01-05", "symbol": "000660", "oracle_score": 9.0, "mfe_r": 5.0, "net_r": 1.0},
            {"trade_date": "2026-01-06", "symbol": "035420", "oracle_score": 8.0, "mfe_r": 4.0, "net_r": 1.0},
        ],
        holdout_oracle_rows=[
            {"trade_date": "2026-01-07", "symbol": "005930", "oracle_score": 7.0, "mfe_r": 3.0, "net_r": 1.0}
        ],
        shortlist_size=3,
        grid=_tiny_grid(),
    )

    assert result["train_variant_count"] > len(result["shortlist"])
    assert len(result["shortlist"]) == 3
    assert {row["selection_basis"] for row in result["holdout_scored_once_rows"]} == {"frozen_train_shortlist_holdout_scored_once"}
    assert [row["frozen_train_rank"] for row in result["holdout_scored_once_rows"]] == [1, 2, 3]
    assert result["holdout_selection_basis"] == "reported_in_frozen_train_rank_order_no_holdout_sort"
    assert result["best_holdout_frozen_variant"] == result["holdout_scored_once_rows"][0]
    assert all(row["selector_variant"] for row in result["shortlist"])


def test_alcb_breakout_replay_variants_are_small_and_disciplined() -> None:
    variants = _alcb_breakout_replay_variants()

    assert len(variants) == 14
    assert {row["pool_variant"] for row in variants} == {"active_only", "frontier40_branch"}
    assert {row["min_first30_rel_volume"] for row in variants} == {2.0, 3.0}
    assert all(row["require_first30_campaign_breakout_acceptance"] is True for row in variants)
    assert all(row["max_session_trades_per_route"] == 1 for row in variants)

    breakout = next(row for row in variants if row["route_family"] == "breakout_family")
    routes = _alcb_replay_routes(breakout)
    assert [route["mode"] for route in routes] == ["combined_breakout", "or_breakout", "pdh_breakout"]
    assert all(route["context_min"]["first30_breakout_acceptance"] == 1.0 for route in routes)

    first30 = next(row for row in variants if row["route_family"] == "first30_open_proxy")
    assert first30["selector_mode"]
    first30_routes = _alcb_replay_routes(first30)
    assert first30_routes[0]["mode"] == "first30_open"
    assert first30_routes[0]["route_max_session_trades"] == 1
    assert first30_routes[0]["context_min"]["first30_breakout_acceptance"] == 1.0

    reclaim = next(row for row in variants if row["route_family"] == "campaign_or_high_reclaim")
    reclaim_routes = _alcb_replay_routes(reclaim)
    assert len(reclaim_routes) == 1
    assert reclaim_routes[0]["mode"] == "or_high_reclaim"
    assert reclaim_routes[0]["level_source"] == "campaign_breakout_level"
    assert reclaim_routes[0]["min_reclaim_closes"] == 2
    assert reclaim_routes[0]["route_max_session_trades"] == 1
    assert reclaim_routes[0]["context_min"]["first30_breakout_acceptance"] == 1.0


def test_alcb_faithfulness_funnel_variants_cover_topn_routes_and_filters() -> None:
    variants = _alcb_faithfulness_variants()

    assert len(variants) == 560
    assert {row["top_n"] for row in variants} == {8, 12, 16, 24, 32, 40, 64}
    assert {row["selector_mode"] for row in variants} == {
        "structural_first30",
        "structural_first30_causal_tiebreak",
        "first30_confirmation",
        "blend_struct60_first3030_causal10",
        "blend_struct70_causal30",
    }
    assert {row["route_bundle"] for row in variants} == {
        "raw_or_pdh_combined_18",
        "raw_or_pdh_combined_36",
        "campaign_levels_36",
        "full_alcb_stack_36",
    }

    route_bundles = {bundle["name"]: bundle for bundle in _alcb_faithfulness_route_bundles()}
    campaign_levels = {
        route["level_source"]
        for route in route_bundles["campaign_levels_36"]["routes"]
        if route.get("level_source")
    }
    assert campaign_levels == {"campaign_avwap", "campaign_box_high", "campaign_breakout_level"}

    strict = next(
        row
        for row in variants
        if row["min_first30_rel_volume"] == 2.0
        and row["min_first30_signal_cpr"] == 0.6
        and row["require_first30_campaign_breakout_acceptance"] is True
    )
    valid = _row("2026-01-05", "005930", score=7.0)
    assert _alcb_candidate_filter_reason(valid, strict) == ""

    low_relvol = {**valid, "first30_rel_volume": 1.9}
    assert _alcb_candidate_filter_reason(low_relvol, strict) == "first30_rel_volume_below_min"

    low_cpr = {**valid, "first30_signal_cpr": 0.5}
    assert _alcb_candidate_filter_reason(low_cpr, strict) == "first30_signal_cpr_below_min"

    no_acceptance = {**valid, "first30_breakout_acceptance": False, "first30_breakout_confirmation": False}
    assert _alcb_candidate_filter_reason(no_acceptance, strict) == "missing_first30_campaign_breakout_acceptance"


def _tiny_grid() -> dict[str, tuple[float | int, ...]]:
    return {
        "research_min_structural_campaign_score": (3.0, 5.0),
        "research_structural_frontier_count": (16, 32),
        "research_min_rs_percentile": (0.0,),
        "research_min_sector_daily_score_pct": (0.0,),
    }


def _row(day: str, symbol: str, *, score: float) -> dict[str, object]:
    return {
        "trade_date": day,
        "symbol": symbol,
        "sector": "TECH",
        "relative_strength_pct": 80.0,
        "stock_vs_universe_strength": score,
        "sector_daily_score_pct": 70.0,
        "sector_participation": 0.75,
        "structural_campaign_score": score,
        "first30_confirmation_score": 1.0,
        "campaign_state": "breakout_watch",
        "campaign_box_high": 101.0,
        "campaign_box_low": 95.0,
        "campaign_box_mid": 98.0,
        "campaign_box_range_pct": 0.06,
        "campaign_box_containment": 0.70,
        "campaign_box_atr_ratio": 1.0,
        "campaign_box_squeeze_pct": 0.2,
        "campaign_box_tier": "tight",
        "campaign_avwap": 100.0,
        "campaign_avwap_anchor_available": True,
        "first30_rel_volume": 2.5,
        "first30_signal_cpr": 0.7,
        "first30_vwap_ret": 0.01,
        "first30_gap": 0.02,
        "first30_low_vs_prev_close": 0.015,
        "first30_above_campaign_avwap": True,
        "first30_breakout_acceptance": True,
        "first30_breakout_acceptance_closes": 2,
        "campaign_breakout_level": 101.0,
        "campaign_avwap_distance_pct": 0.01,
        "campaign_metadata": {
            "structural_campaign_score": score,
            "campaign_box_high": 101.0,
            "campaign_avwap": 100.0,
            "campaign_breakout_level": 101.0,
            "campaign_box_atr_ratio": 1.0,
            "campaign_box_squeeze_pct": 0.2,
            "campaign_box_tier": "tight",
            "first30_rel_volume": 2.5,
            "first30_signal_cpr": 0.7,
            "relative_strength_pct": 80.0,
            "sector_daily_score_pct": 70.0,
            "score_uses_ex_post_labels": False,
        },
        "score_uses_ex_post_labels": False,
    }
