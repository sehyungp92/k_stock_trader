from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from backtests.strategies.olr.shadow_ledger_reranker import (
    actual_snapshots_from_ledger_rows,
    build_shadow_opportunity_ledger_for_day,
    evaluate_same_day_reranker_with_replay,
    fit_same_day_reranker_profile,
    score_shadow_ledger_rows,
    snapshots_from_reranked_rows,
)
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.research import afternoon_selection_from_contexts, build_afternoon_contexts


def test_olr_shadow_ledger_keeps_sector_blocked_shadow_and_route_labels() -> None:
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = _snapshot(trade_date)
    bars_by_key = _bars_by_key(trade_date, next_date)
    cfg = _hard_sector_cfg()
    contexts = build_afternoon_contexts(snapshot, bars_by_key, cfg)

    rows = build_shadow_opportunity_ledger_for_day(
        snapshot,
        contexts,
        bars_by_key,
        {trade_date: next_date},
        cfg,
        window="train",
        source_label="unit",
    )
    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["000001"]["actual_trade_slot"] is True
    assert by_symbol["000002"]["actual_trade_slot"] is False
    assert "afternoon_score_band_rule_miss" in by_symbol["000002"]["hard_filter_reject_reasons"]
    assert by_symbol["000002"]["fill_feasible"] is True
    assert by_symbol["000002"]["route_net_r"] > by_symbol["000001"]["route_net_r"]
    assert by_symbol["000002"]["same_day_replacement_value_r"] > 0.0
    assert "sector_daily_score_pct" in by_symbol["000002"]
    assert "sector_intraday_score_pct" in by_symbol["000002"]
    assert "stock_intraday_leadership_score" in by_symbol["000002"]


def test_train_only_profile_can_rerank_without_static_allowed_sector_filter() -> None:
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = _snapshot(trade_date)
    bars_by_key = _bars_by_key(trade_date, next_date)
    cfg = _hard_sector_cfg()
    contexts = build_afternoon_contexts(snapshot, bars_by_key, cfg)
    rows = build_shadow_opportunity_ledger_for_day(snapshot, contexts, bars_by_key, {trade_date: next_date}, cfg)

    profile = fit_same_day_reranker_profile(rows, min_feature_observations=2)
    scored = score_shadow_ledger_rows(rows, profile)
    reranked = snapshots_from_reranked_rows(scored, profile, top_n=2)[trade_date]
    actual = actual_snapshots_from_ledger_rows(rows)[trade_date]

    assert profile["source_window"] == "train"
    assert "BIO" in profile["sector_priors"]
    assert actual.candidates[0].symbol == "000001"
    assert reranked.candidates[0].symbol == "000002"

    live_cfg = OLRConfig.from_mapping(
        {
            **_hard_sector_mutations(),
            "olr.shadow_reranker.enabled": True,
            "olr.shadow_reranker.profile": profile,
            "olr.shadow_reranker.replace_score_band_rules": True,
        }
    )
    live_selected = afternoon_selection_from_contexts(snapshot, contexts, live_cfg)

    assert live_selected.candidates[0].symbol == "000002"
    assert live_selected.generated_at == datetime.combine(trade_date, time(14, 30), tzinfo=KST)
    assert live_selected.metadata["shadow_reranker_replaced_score_band_rules"] is True


def test_shadow_reranker_preserves_baseline_trade_day_gate() -> None:
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = _snapshot(trade_date)
    bars_by_key = _bars_by_key(trade_date, next_date)
    cfg = OLRConfig.from_mapping(
        {
            **_hard_sector_mutations(),
            "olr.afternoon.score_band_rules": [
                {"name": "defense_only", "min_score": -999.0, "max_score": 99999.0, "allowed_sectors": ["DEFENSE"]}
            ],
        }
    )
    contexts = build_afternoon_contexts(snapshot, bars_by_key, cfg)
    rows = build_shadow_opportunity_ledger_for_day(snapshot, contexts, bars_by_key, {trade_date: next_date}, cfg)

    profile = fit_same_day_reranker_profile(rows, min_feature_observations=2)
    reranked = snapshots_from_reranked_rows(rows, profile, top_n=2)
    live_cfg = OLRConfig.from_mapping(
        {
            **_hard_sector_mutations(),
            "olr.afternoon.score_band_rules": [
                {"name": "defense_only", "min_score": -999.0, "max_score": 99999.0, "allowed_sectors": ["DEFENSE"]}
            ],
            "olr.shadow_reranker.enabled": True,
            "olr.shadow_reranker.profile": profile,
            "olr.shadow_reranker.replace_score_band_rules": True,
        }
    )
    live_selected = afternoon_selection_from_contexts(snapshot, contexts, live_cfg)

    assert rows[0]["same_day_selected_count"] == 0
    assert reranked == {}
    assert live_selected.candidates == ()
    assert live_selected.metadata["shadow_reranker_baseline_selected_count"] == 0


def test_shadow_validation_only_replaces_score_band_rejects() -> None:
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = _snapshot(trade_date)
    bars_by_key = _bars_by_key(trade_date, next_date)
    cfg = _hard_sector_cfg()
    rows = build_shadow_opportunity_ledger_for_day(
        snapshot,
        build_afternoon_contexts(snapshot, bars_by_key, cfg),
        bars_by_key,
        {trade_date: next_date},
        cfg,
    )
    selected = dict(next(row for row in rows if row["symbol"] == "000001"))
    blocked = dict(next(row for row in rows if row["symbol"] == "000002"))
    blocked["hard_filter_reject_reasons"] = ["afternoon_ret_below_floor"]
    profile = {
        "profile_hash": "unit",
        "weights": {},
        "feature_stats": {},
        "sector_priors": {"BIO": 10.0, "SEMIS": 0.0},
        "sector_prior_weight": 1.0,
        "allow_slot_expansion": False,
    }

    reranked = snapshots_from_reranked_rows([selected, blocked], profile, top_n=2)[trade_date]

    assert [candidate.symbol for candidate in reranked.candidates] == ["000001"]


def test_shadow_reranker_config_is_inert_by_default() -> None:
    trade_date = date(2026, 1, 5)
    snapshot = _snapshot(trade_date)
    bars_by_key = _bars_by_key(trade_date, trade_date + timedelta(days=1))
    cfg = _hard_sector_cfg()
    contexts = build_afternoon_contexts(snapshot, bars_by_key, cfg)

    baseline = afternoon_selection_from_contexts(snapshot, contexts, cfg)
    inert = afternoon_selection_from_contexts(snapshot, contexts, OLRConfig.from_mapping(_hard_sector_mutations()))

    assert [candidate.symbol for candidate in inert.candidates] == [candidate.symbol for candidate in baseline.candidates]
    assert inert.artifact_hash == baseline.artifact_hash


def test_reranker_validation_uses_actual_replay_metrics() -> None:
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = _snapshot(trade_date)
    bars_by_key = _bars_by_key(trade_date, next_date)
    cfg = _hard_sector_cfg()
    rows = build_shadow_opportunity_ledger_for_day(
        snapshot,
        build_afternoon_contexts(snapshot, bars_by_key, cfg),
        bars_by_key,
        {trade_date: next_date},
        cfg,
    )

    summary = evaluate_same_day_reranker_with_replay(rows, rows, bars_by_key, cfg, _hard_sector_mutations())

    assert summary["train"]["actual"]["metrics"]["entry_fill_count"] == pytest.approx(1.0)
    assert summary["train"]["reranked"]["metrics"]["entry_fill_count"] == pytest.approx(1.0)
    assert summary["train"]["reranked"]["metrics"]["official_mtm_net_return_pct"] > summary["train"]["actual"]["metrics"]["official_mtm_net_return_pct"]
    assert summary["candidate_mutation"]["olr.shadow_reranker.replace_score_band_rules"] is True


def test_reranker_validation_handles_empty_windows() -> None:
    cfg = _hard_sector_cfg()

    summary = evaluate_same_day_reranker_with_replay([], [], {}, cfg, _hard_sector_mutations())

    assert summary["train"]["actual"]["metrics"]["entry_fill_count"] == pytest.approx(0.0)
    assert summary["train"]["reranked"]["metrics"]["entry_fill_count"] == pytest.approx(0.0)
    assert summary["oos"]["actual"]["metrics"]["official_mtm_net_return_pct"] == pytest.approx(0.0)
    assert summary["promotion_pass"] is False


def _hard_sector_cfg() -> OLRConfig:
    return OLRConfig.from_mapping(_hard_sector_mutations())


def _hard_sector_mutations() -> dict[str, object]:
    return {
        "olr.afternoon.top_n": 2,
        "olr.overnight.slot_count": 1,
        "olr.afternoon.score_mode": "momentum",
        "olr.execution.entry_mode": "close_auction",
        "olr.execution.exit_mode": "next_close",
        "olr.cost.slippage_bps": 0.0,
        "olr.cost.commission_bps": 0.0,
        "olr.cost.tax_bps_on_sell": 0.0,
        "olr.execution.auction_adverse_bps": 0.0,
        "olr.execution.auction_limit_offset_bps": 500.0,
        "olr.afternoon.score_band_rules": [
            {"name": "semis_only", "min_score": -999.0, "max_score": 99999.0, "allowed_sectors": ["SEMIS"]}
        ],
    }


def _snapshot(trade_date: date) -> OLRDailySnapshot:
    return OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit-shadow-ledger",
        generated_at=datetime.combine(trade_date, time(8, 50), tzinfo=KST),
        candidates=(
            _candidate("000001", trade_date, "SEMIS", rank=1, sector_score=55.0),
            _candidate("000002", trade_date, "BIO", rank=2, sector_score=80.0),
        ),
    )


def _candidate(symbol: str, trade_date: date, sector: str, *, rank: int, sector_score: float) -> OLRDailyCandidate:
    return OLRDailyCandidate(
        symbol=symbol,
        trade_date=trade_date,
        prior_day_high=103.0,
        prior_day_low=97.0,
        prior_day_close=100.0,
        daily_atr=5.0,
        expected_5m_volume=100_000.0,
        average_30m_volume=600_000.0,
        sector=sector,
        selection_score=100.0 - rank,
        daily_signal_score=70.0,
        rank=rank,
        rank_pct=50.0 * rank,
        rs_percentile=80.0 - rank,
        accumulation_score=0.5,
        flow_score=0.2,
        foreign_flow_5d=0.1,
        institutional_flow_5d=0.1,
        flow_agreement_5d=0.1,
        metadata={
            "sector_daily_score_pct": sector_score,
            "sector_daily_ret_5d": 0.03 if sector == "BIO" else 0.01,
            "sector_daily_ret_20d": 0.08 if sector == "BIO" else 0.02,
            "sector_daily_breadth_20d": 0.7 if sector == "BIO" else 0.55,
            "sector_daily_participation": 0.8 if sector == "BIO" else 0.55,
            "sector_daily_rel_volume": 1.4 if sector == "BIO" else 1.0,
            "sector_strength_pct": sector_score,
            "sector_participation": 0.7,
            "market_heat_score": 65.0,
        },
    )


def _bars_by_key(trade_date: date, next_date: date) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    return {
        (trade_date, "000001"): (
            _bar("000001", trade_date, time(9, 0), 100.0, 101.0, 99.0, 100.5),
            _bar("000001", trade_date, time(14, 25), 100.5, 102.0, 100.0, 101.5),
            _bar("000001", trade_date, time(14, 30), 101.5, 101.8, 101.0, 101.3),
            _bar("000001", trade_date, time(15, 30), 101.5, 102.0, 101.0, 101.0),
        ),
        (next_date, "000001"): (
            _bar("000001", next_date, time(15, 30), 98.0, 99.0, 97.0, 98.0),
        ),
        (trade_date, "000002"): (
            _bar("000002", trade_date, time(9, 0), 100.0, 103.0, 99.0, 102.5),
            _bar("000002", trade_date, time(14, 25), 102.5, 107.0, 102.0, 106.0),
            _bar("000002", trade_date, time(14, 30), 106.0, 106.5, 105.8, 106.2),
            _bar("000002", trade_date, time(15, 30), 106.0, 106.5, 105.5, 106.0),
        ),
        (next_date, "000002"): (
            _bar("000002", next_date, time(15, 30), 113.0, 116.0, 112.0, 115.0),
        ),
    }


def _bar(symbol: str, day: date, clock: time, open_px: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(day, clock, tzinfo=KST),
        timeframe="5m",
        open=open_px,
        high=high,
        low=low,
        close=close,
        volume=100_000.0,
        source="unit",
        source_fingerprint="unit",
    )
