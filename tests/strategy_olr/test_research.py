from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE, OLRArtifactStore
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.research import (
    afternoon_selection_from_contexts,
    afternoon_selection_from_snapshot,
    build_afternoon_contexts,
    build_research_snapshot,
    daily_selection_from_snapshot,
    run_daily_selection,
)
from strategy_olr.research_generator import generate_afternoon_candidate_snapshot, generate_candidate_snapshot, generate_research_snapshot


def test_olr_research_excludes_same_day_daily_and_flow_rows() -> None:
    trade_date = date(2026, 2, 2)
    cfg = OLRConfig.from_mapping(
        {
            "olr.research.top_long_count": 1,
            "olr.research.min_adv20_krw": 1_000_000,
            "olr.signal.daily_min_score": 0.0,
        }
    )
    daily = {
        "005930": _daily_rows(trade_date, start=5_000, drift=45),
        "000660": _daily_rows(trade_date, start=5_000, drift=5),
    }
    flow = {
        "005930": _flow_rows(trade_date, value=10_000_000),
        "000660": _flow_rows(trade_date, value=-10_000_000),
    }
    foreign = {
        "005930": _foreign_rows(trade_date, value=8_000_000),
        "000660": _foreign_rows(trade_date, value=-8_000_000),
    }
    institutional = {
        "005930": _institutional_rows(trade_date, value=4_000_000),
        "000660": _institutional_rows(trade_date, value=-4_000_000),
    }
    contaminated_daily = {
        **daily,
        "000660": daily["000660"]
        + [{"date": trade_date.isoformat(), "open": 1, "high": 999_999, "low": 1, "close": 999_999, "volume": 999_999_999}],
    }
    contaminated_flow = {
        **flow,
        "000660": flow["000660"] + [{"date": trade_date.isoformat(), "foreign_net": 99_000_000_000, "inst_net": 99_000_000_000}],
    }
    contaminated_foreign = {
        **foreign,
        "000660": foreign["000660"] + [{"date": trade_date.isoformat(), "foreign_net": 99_000_000_000}],
    }
    contaminated_institutional = {
        **institutional,
        "000660": institutional["000660"] + [{"date": trade_date.isoformat(), "institutional_net": 99_000_000_000}],
    }

    clean = build_research_snapshot(
        daily,
        trade_date,
        cfg,
        flow_by_symbol=flow,
        foreign_flow_by_symbol=foreign,
        institutional_flow_by_symbol=institutional,
    )
    contaminated = build_research_snapshot(
        contaminated_daily,
        trade_date,
        cfg,
        flow_by_symbol=contaminated_flow,
        foreign_flow_by_symbol=contaminated_foreign,
        institutional_flow_by_symbol=contaminated_institutional,
    )
    clean_selected = daily_selection_from_snapshot(clean, cfg)
    contaminated_selected = daily_selection_from_snapshot(contaminated, cfg)

    assert contaminated.source_fingerprint == clean.source_fingerprint
    assert clean_selected.metadata["daily_row_cutoff"] == "row_date < trade_date"
    assert clean_selected.metadata["flow_row_cutoff"] == "row_date < trade_date"
    assert clean_selected.metadata["official_performance"] is False
    assert clean_selected.candidates[0].symbol == "005930"
    assert contaminated_selected.candidates[0].symbol == "005930"
    assert contaminated_selected.artifact_hash == clean_selected.artifact_hash
    assert contaminated_selected.candidates[0].selection_score == clean_selected.candidates[0].selection_score
    assert contaminated_selected.candidates[0].metadata["same_day_flow_used"] is False
    assert contaminated_selected.candidates[0].metadata["lagged_foreign_flow_5d"] < 100.0
    assert contaminated_selected.candidates[0].metadata["lagged_institutional_flow_5d"] < 100.0


def test_olr_stage1_selection_preserves_presession_generated_at() -> None:
    trade_date = date(2026, 2, 2)
    generated_at = datetime.combine(trade_date, datetime.min.time(), tzinfo=KST)
    cfg = OLRConfig.from_mapping(
        {
            "olr.research.top_long_count": 1,
            "olr.research.min_adv20_krw": 1_000_000,
            "olr.signal.daily_min_score": 0.0,
        }
    )
    research = build_research_snapshot(
        {"005930": _daily_rows(trade_date, start=5_000, drift=45)},
        trade_date,
        cfg,
        generated_at=generated_at,
        source_fingerprint="unit-stage1",
    )

    selected = daily_selection_from_snapshot(research, cfg)

    assert selected.generated_at == generated_at
    assert selected.metadata["artifact_stage"] == OLR_STAGE1_ARTIFACT_STAGE
    assert selected.metadata["selection_time_basis"] == "pre_session_from_prior_completed_daily_rows"


def test_olr_afternoon_selection_excludes_1430_bar() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, prior_close=100.0, rank=1),
            _candidate("000660", trade_date, prior_close=100.0, rank=2),
        ),
    )
    bars = {
        (trade_date, "005930"): tuple(
            [
                _bar("005930", trade_date, 9, 0, 100.0, 103.0, 99.0, 102.0),
                _bar("005930", trade_date, 14, 25, 102.0, 105.0, 101.0, 104.0),
            ]
        ),
        (trade_date, "000660"): tuple(
            [
                _bar("000660", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
                _bar("000660", trade_date, 14, 25, 100.5, 101.0, 100.0, 100.7),
                _bar("000660", trade_date, 14, 30, 100.7, 150.0, 100.7, 150.0),
            ]
        ),
    }
    cfg = OLRConfig.from_mapping({"olr.afternoon.top_n": 1, "olr.afternoon.score_mode": "momentum"})

    selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)
    context_selected = afternoon_selection_from_contexts(snapshot, build_afternoon_contexts(snapshot, bars, cfg), cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["005930"]
    assert context_selected.artifact_hash == selected.artifact_hash
    assert selected.metadata["intraday_selection_cutoff"] == "timestamp < 14:30 KST"
    assert selected.metadata["official_performance"] is False
    assert selected.candidates[0].metadata["afternoon_features"]["bar_count"] == 2


def test_olr_live_daily_generator_builds_snapshot_then_shared_run_daily_selection(tmp_path) -> None:
    trade_date = date(2026, 2, 2)
    cfg = OLRConfig.from_mapping(
        {
            "olr.research.top_long_count": 1,
            "olr.research.min_adv20_krw": 1_000_000,
            "olr.signal.daily_min_score": 0.0,
        }
    )
    daily = {
        "005930": _daily_rows(trade_date, start=5_000, drift=45),
        "000660": _daily_rows(trade_date, start=5_000, drift=5),
    }
    flow = {
        "005930": _flow_rows(trade_date, value=10_000_000),
        "000660": _flow_rows(trade_date, value=-10_000_000),
    }
    source = "unit-live-research-builder"

    research_snapshot = generate_research_snapshot(daily, trade_date, config=cfg, flow_by_symbol=flow, source_fingerprint=source)
    direct = run_daily_selection(research_snapshot, config=cfg, artifact_root=tmp_path / "direct")
    generated = generate_candidate_snapshot(daily, trade_date, config=cfg, flow_by_symbol=flow, artifact_root=tmp_path / "generated", source_fingerprint=source)

    assert generated.artifact_hash == direct.artifact_hash
    assert generated.metadata["source"] == "olr_research_selection"
    assert generated.metadata["daily_row_cutoff"] == "row_date < trade_date"
    assert OLRArtifactStore(tmp_path / "generated").load_snapshot(trade_date).artifact_hash == direct.artifact_hash


def test_olr_research_keeps_daily_sector_metadata_separate_from_legacy_scoring() -> None:
    trade_date = date(2026, 2, 2)
    cfg = OLRConfig.from_mapping(
        {
            "olr.research.top_long_count": 2,
            "olr.research.min_adv20_krw": 1_000_000,
            "olr.signal.daily_min_score": 0.0,
        }
    )
    daily = {
        "005930": _daily_rows(trade_date, start=5_000, drift=45),
        "000660": _daily_rows(trade_date, start=5_000, drift=20),
        "012450": _daily_rows(trade_date, start=5_000, drift=5),
    }

    selected = daily_selection_from_snapshot(
        build_research_snapshot(daily, trade_date, cfg, sector_map={"005930": "SEMIS", "000660": "SEMIS", "012450": "DEFENSE"}),
        cfg,
    )

    assert selected.candidates
    metadata = selected.candidates[0].metadata
    assert metadata["sector_strength_pct"] == metadata["research_score_components"]["sector_strength_pct"]
    assert metadata["sector_participation"] == metadata["research_score_components"]["sector_participation"]
    assert "sector_daily_score_pct" in metadata
    assert "sector_daily_ret_20d" in metadata
    assert metadata["sector_daily_version"]


def test_olr_afternoon_selection_uses_only_lagged_flow_metadata() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate(
                "005930",
                trade_date,
                prior_close=100.0,
                rank=1,
                flow_score=0.3,
                metadata={
                    "lagged_flow_5d": 0.04,
                    "lagged_flow_z": 1.2,
                    "lagged_foreign_flow_5d": 0.03,
                    "lagged_institutional_flow_5d": 0.01,
                    "lagged_flow_agreement_5d": 0.01,
                    "sector_flow_5d": 0.02,
                    "sector_foreign_flow_5d": 0.015,
                    "sector_institutional_flow_5d": 0.005,
                    "market_heat_score": 70.0,
                },
            ),
            _candidate(
                "000660",
                trade_date,
                prior_close=100.0,
                rank=2,
                flow_score=-0.3,
                metadata={
                    "lagged_flow_5d": -0.04,
                    "lagged_flow_z": -1.2,
                    "sector_flow_5d": -0.02,
                    "market_heat_score": 70.0,
                },
            ),
        ),
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 101.0, 99.5, 100.8),
            _bar("005930", trade_date, 14, 25, 100.8, 103.0, 100.7, 102.5),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 101.0, 99.5, 100.8),
            _bar("000660", trade_date, 14, 25, 100.8, 103.0, 100.7, 102.5),
        ),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 1,
            "olr.afternoon.score_mode": "flow_confirmed",
            "olr.afternoon.min_flow_5d": 0.0,
            "olr.afternoon.min_sector_flow": 0.0,
        }
    )

    selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["005930"]
    features = selected.candidates[0].metadata["afternoon_features"]
    assert features["lagged_flow_5d"] == 0.04
    assert features["lagged_sector_foreign_flow_5d"] == 0.015
    assert selected.candidates[0].metadata["same_day_flow_used"] is False


def test_olr_afternoon_selection_can_reject_exhausted_high_score_candidates() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, prior_close=100.0, rank=1),
            _candidate("000660", trade_date, prior_close=100.0, rank=2),
        ),
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 100.5, 99.8, 100.0),
            _bar("005930", trade_date, 14, 25, 100.0, 116.0, 99.8, 115.5),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 100.4, 99.8, 100.0),
            _bar("000660", trade_date, 14, 25, 100.0, 102.0, 99.9, 101.8),
        ),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 2,
            "olr.afternoon.score_mode": "momentum",
            "olr.afternoon.max_exhaustion_score": 1.5,
            "olr.afternoon.score_calibration_mode": "exhaustion_adjusted",
            "olr.afternoon.exhaustion_penalty": 25.0,
        }
    )

    selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["000660"]
    assert "afternoon_exhaustion_above_cap" in selected.metadata["afternoon_rejected_symbols"]["005930"]
    assert selected.candidates[0].metadata["afternoon_score_calibration_mode"] == "exhaustion_adjusted"
    assert selected.candidates[0].metadata["afternoon_exhaustion_score"] >= 0.0


def test_olr_afternoon_selection_can_block_sectors_and_score_bands() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, prior_close=100.0, rank=1, sector="DEFENSE"),
            _candidate("000660", trade_date, prior_close=100.0, rank=2, sector="SEMICONDUCTORS"),
            _candidate("035420", trade_date, prior_close=100.0, rank=3, sector="INTERNET"),
        ),
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 102.0, 99.0, 101.0),
            _bar("005930", trade_date, 14, 25, 101.0, 104.0, 100.5, 103.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 102.0, 99.0, 101.0),
            _bar("000660", trade_date, 14, 25, 101.0, 116.0, 100.5, 115.0),
        ),
        (trade_date, "035420"): (
            _bar("035420", trade_date, 9, 0, 100.0, 101.5, 99.0, 100.8),
            _bar("035420", trade_date, 14, 25, 100.8, 103.0, 100.0, 102.0),
        ),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 3,
            "olr.afternoon.score_mode": "momentum",
            "olr.afternoon.blocked_sectors": ["DEFENSE"],
            "olr.afternoon.max_score": 600.0,
        }
    )

    selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["035420"]
    assert "afternoon_sector_blocked" in selected.metadata["afternoon_rejected_symbols"]["005930"]
    assert "afternoon_score_above_cap" in selected.metadata["afternoon_rejected_symbols"]["000660"]


def test_olr_afternoon_selection_can_reject_score_notch_without_blocking_tail() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, prior_close=100.0, rank=1, sector="STEEL"),
            _candidate("000660", trade_date, prior_close=100.0, rank=2, sector="SEMICONDUCTORS"),
            _candidate("035420", trade_date, prior_close=100.0, rank=3, sector="INTERNET"),
        ),
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 102.0, 99.0, 101.0),
            _bar("005930", trade_date, 14, 25, 101.0, 116.0, 100.5, 115.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 102.0, 99.0, 101.0),
            _bar("000660", trade_date, 14, 25, 101.0, 104.0, 100.5, 103.0),
        ),
        (trade_date, "035420"): (
            _bar("035420", trade_date, 9, 0, 100.0, 101.5, 99.0, 100.8),
            _bar("035420", trade_date, 14, 25, 100.8, 102.5, 100.0, 101.4),
        ),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 3,
            "olr.afternoon.score_mode": "momentum",
            "olr.afternoon.reject_score_min": 350.0,
            "olr.afternoon.reject_score_max": 650.0,
        }
    )

    selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["005930", "035420"]
    assert "afternoon_score_in_reject_band" in selected.metadata["afternoon_rejected_symbols"]["000660"]


def test_olr_afternoon_selection_supports_conditional_score_band_rules() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, prior_close=100.0, rank=1, sector="STEEL"),
            _candidate("000660", trade_date, prior_close=100.0, rank=2, sector="SEMICONDUCTORS"),
            _candidate("012450", trade_date, prior_close=100.0, rank=4, sector="DEFENSE"),
        ),
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 102.0, 99.0, 101.0),
            _bar("005930", trade_date, 14, 25, 101.0, 116.0, 100.5, 115.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 102.0, 99.0, 101.0),
            _bar("000660", trade_date, 14, 25, 101.0, 104.0, 100.5, 103.0),
        ),
        (trade_date, "012450"): (
            _bar("012450", trade_date, 9, 0, 100.0, 102.0, 99.0, 101.0),
            _bar("012450", trade_date, 14, 25, 101.0, 104.0, 100.5, 103.0),
        ),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 3,
            "olr.afternoon.score_mode": "momentum",
            "olr.afternoon.score_band_rules": [
                {"name": "tail_leaders", "min_score": 650.0},
                {
                    "name": "semi_mid_rank2",
                    "min_score": 350.0,
                    "max_score": 650.0,
                    "allowed_sectors": ["SEMICONDUCTORS"],
                    "max_rank": 2,
                },
            ],
        }
    )

    selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["005930", "000660"]
    assert selected.candidates[1].metadata["afternoon_score_band_rule"] == "semi_mid_rank2"
    assert "afternoon_score_band_rule_miss" in selected.metadata["afternoon_rejected_symbols"]["012450"]


def test_olr_afternoon_score_band_rules_support_derived_sector_features() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate(
                "005930",
                trade_date,
                prior_close=100.0,
                rank=1,
                sector="SEMIS",
                metadata={"return_20d_pct": 13.0, "sector_daily_score_pct": 70.0, "sector_daily_ret_20d": 0.04},
            ),
            _candidate(
                "000660",
                trade_date,
                prior_close=100.0,
                rank=2,
                sector="SEMIS",
                metadata={"return_20d_pct": 5.0, "sector_daily_score_pct": 70.0, "sector_daily_ret_20d": 0.04},
            ),
            _candidate(
                "012450",
                trade_date,
                prior_close=100.0,
                rank=3,
                sector="DEFENSE",
                metadata={"return_20d_pct": 12.0, "sector_daily_score_pct": 35.0, "sector_daily_ret_20d": 0.02},
            ),
        ),
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar("005930", trade_date, 14, 25, 100.5, 107.0, 100.0, 106.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.0),
            _bar("000660", trade_date, 14, 25, 100.0, 105.0, 99.0, 104.0),
        ),
        (trade_date, "012450"): (
            _bar("012450", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar("012450", trade_date, 14, 25, 100.5, 105.0, 100.0, 104.0),
        ),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 3,
            "olr.afternoon.score_mode": "momentum",
            "olr.afternoon.score_band_rules": [
                {
                    "name": "daily_intraday_combo",
                    "min_score": 0.0,
                    "min_sector_confirm_mean_score_pct": 55.0,
                    "min_features": {
                        "stock_sector_daily_ret20_gap_pct": 5.0,
                        "stock_intraday_sector_ret_gap_pct": 1.0,
                    },
                }
            ],
        }
    )

    selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["005930"]
    features = selected.candidates[0].metadata["afternoon_features"]
    assert features["sector_confirm_mean_score_pct"] >= 55.0
    assert features["stock_sector_daily_ret20_gap_pct"] >= 5.0
    assert features["stock_intraday_sector_ret_gap_pct"] >= 1.0
    assert "afternoon_score_band_rule_miss" in selected.metadata["afternoon_rejected_symbols"]["000660"]
    assert "afternoon_score_band_rule_miss" in selected.metadata["afternoon_rejected_symbols"]["012450"]


def test_olr_afternoon_score_band_rules_support_dynamic_sector_admission() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate(
                "005930",
                trade_date,
                prior_close=100.0,
                rank=1,
                sector="SEMIS",
                metadata={"return_20d_pct": 13.0, "sector_daily_score_pct": 80.0, "sector_daily_ret_20d": 0.04},
            ),
            _candidate(
                "000660",
                trade_date,
                prior_close=100.0,
                rank=2,
                sector="SEMIS",
                metadata={"return_20d_pct": 5.0, "sector_daily_score_pct": 80.0, "sector_daily_ret_20d": 0.04},
            ),
            _candidate(
                "012450",
                trade_date,
                prior_close=100.0,
                rank=3,
                sector="DEFENSE",
                metadata={"return_20d_pct": 13.0, "sector_daily_score_pct": 35.0, "sector_daily_ret_20d": 0.04},
            ),
        ),
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar("005930", trade_date, 14, 25, 100.5, 107.0, 100.0, 106.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar("000660", trade_date, 14, 25, 100.5, 107.0, 100.0, 106.0),
        ),
        (trade_date, "012450"): (
            _bar("012450", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar("012450", trade_date, 14, 25, 100.5, 107.0, 100.0, 106.0),
        ),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 3,
            "olr.afternoon.score_mode": "momentum",
            "olr.afternoon.score_band_rules": [
                {
                    "name": "dynamic_sector_admission",
                    "min_score": 0.0,
                    "sector_admission": {
                        "mode": "dynamic_confirmed_rotation",
                        "min_sector_daily_score_pct": 75.0,
                        "min_stock_sector_daily_ret20_gap_pct": 5.0,
                    },
                }
            ],
        }
    )

    selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["005930"]
    assert selected.candidates[0].metadata["afternoon_score_band_rule"] == "dynamic_sector_admission"
    assert "afternoon_score_band_rule_miss" in selected.metadata["afternoon_rejected_symbols"]["000660"]
    assert "afternoon_score_band_rule_miss" in selected.metadata["afternoon_rejected_symbols"]["012450"]


def test_olr_live_afternoon_generator_is_thin_over_shared_selector(tmp_path) -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, prior_close=100.0, rank=1),
            _candidate("000660", trade_date, prior_close=100.0, rank=2),
        ),
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 103.0, 99.0, 102.0),
            _bar("005930", trade_date, 14, 25, 102.0, 105.0, 101.0, 104.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar("000660", trade_date, 14, 25, 100.5, 101.0, 100.0, 100.7),
        ),
    }
    cfg = OLRConfig.from_mapping({"olr.afternoon.top_n": 1, "olr.afternoon.score_mode": "momentum"})

    replay_selected = afternoon_selection_from_snapshot(snapshot, bars, cfg)
    live_selected = generate_afternoon_candidate_snapshot(snapshot, bars, config=cfg, artifact_root=tmp_path)
    persisted = OLRArtifactStore(tmp_path).load_snapshot(trade_date)

    assert live_selected.artifact_hash == replay_selected.artifact_hash
    assert persisted is not None
    assert persisted.artifact_hash == replay_selected.artifact_hash
    assert live_selected.metadata["source"] == "olr_afternoon_selection"


def test_olr_afternoon_selection_adds_intraday_sector_features_and_aliases() -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, prior_close=100.0, rank=1, sector="SEMIS"),
            _candidate("000660", trade_date, prior_close=100.0, rank=2, sector="SEMIS"),
            _candidate("012450", trade_date, prior_close=100.0, rank=3, sector="DEFENSE"),
            _candidate("047810", trade_date, prior_close=100.0, rank=4, sector="DEFENSE"),
        ),
    )
    bars = {
        (trade_date, "005930"): (_bar("005930", trade_date, 9, 0, 100.0, 104.0, 99.0, 103.0), _bar("005930", trade_date, 14, 25, 103.0, 106.0, 102.0, 105.0)),
        (trade_date, "000660"): (_bar("000660", trade_date, 9, 0, 100.0, 103.0, 99.0, 102.0), _bar("000660", trade_date, 14, 25, 102.0, 105.0, 101.0, 104.0)),
        (trade_date, "012450"): (_bar("012450", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.0), _bar("012450", trade_date, 14, 25, 100.0, 101.0, 99.0, 100.0)),
        (trade_date, "047810"): (_bar("047810", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.0), _bar("047810", trade_date, 14, 25, 100.0, 101.0, 99.0, 100.0)),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 1,
            "olr.afternoon.score_mode": "momentum",
            "olr.afternoon.weight_intraday_sector": 0.001,
            "olr.afternoon.min_intraday_sector_score_pct": 50.0,
        }
    )

    selected = afternoon_selection_from_snapshot(
        snapshot,
        bars,
        cfg,
        sector_map={"005930": "SEMIS", "000660": "SEMIS", "012450": "DEFENSE", "047810": "DEFENSE"},
    )

    assert cfg.afternoon_weight_intraday_sector == 0.001
    assert [candidate.symbol for candidate in selected.candidates] == ["005930"]
    features = selected.candidates[0].metadata["afternoon_features"]
    assert features["sector_intraday_score_pct"] >= 50.0
    assert features["sector_intraday_effective_count"] == 1
    assert "sector_intraday_breadth" in features


def test_olr_artifact_store_round_trip(tmp_path) -> None:
    trade_date = date(2026, 2, 2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(_candidate("005930", trade_date, prior_close=100.0, rank=1),),
    )
    store = OLRArtifactStore(tmp_path)

    path = store.save_snapshot(snapshot)
    loaded = store.load_snapshot(trade_date)

    assert path.exists()
    assert loaded is not None
    assert loaded.artifact_hash == snapshot.artifact_hash
    assert loaded.candidates[0].symbol == "005930"


def test_olr_stage1_and_final_artifacts_do_not_overwrite(tmp_path) -> None:
    trade_date = date(2026, 2, 2)
    daily = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, prior_close=100.0, rank=1),
            _candidate("000660", trade_date, prior_close=100.0, rank=2),
        ),
        metadata={"source": "olr_research_selection", "artifact_stage": OLR_STAGE1_ARTIFACT_STAGE},
    )
    bars = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 103.0, 99.0, 102.0),
            _bar("005930", trade_date, 14, 25, 102.0, 105.0, 101.0, 104.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar("000660", trade_date, 14, 25, 100.5, 101.0, 100.0, 100.7),
        ),
    }
    store = OLRArtifactStore(tmp_path)

    store.save_snapshot(daily)
    final = generate_afternoon_candidate_snapshot(
        daily,
        bars,
        config=OLRConfig.from_mapping({"olr.afternoon.top_n": 1, "olr.afternoon.score_mode": "momentum"}),
        artifact_root=tmp_path,
    )

    assert store.path_for(trade_date, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE).exists()
    assert store.path_for(trade_date, artifact_stage=OLR_FINAL_ARTIFACT_STAGE).exists()
    assert store.load_snapshot(trade_date, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE).artifact_hash == daily.artifact_hash
    assert store.load_snapshot(trade_date, artifact_stage=OLR_FINAL_ARTIFACT_STAGE).artifact_hash == final.artifact_hash
    assert final.metadata["artifact_stage"] == OLR_FINAL_ARTIFACT_STAGE


def test_olr_artifact_store_rejects_unknown_stage(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported OLR artifact_stage"):
        OLRArtifactStore(tmp_path).path_for(date(2026, 2, 2), artifact_stage="bad-stage")


def _candidate(
    symbol: str,
    trade_date: date,
    *,
    prior_close: float,
    rank: int,
    flow_score: float = 0.1,
    sector: str = "UNKNOWN",
    metadata: dict | None = None,
) -> OLRDailyCandidate:
    return OLRDailyCandidate(
        symbol=symbol,
        trade_date=trade_date,
        prior_day_high=prior_close * 1.02,
        prior_day_low=prior_close * 0.98,
        prior_day_close=prior_close,
        daily_atr=2.0,
        expected_5m_volume=100.0,
        average_30m_volume=600.0,
        sector=sector,
        selection_score=100.0 - rank,
        rank=rank,
        flow_score=flow_score,
        tradable=True,
        source_fingerprint="unit",
        metadata=dict(metadata or {}),
    )


def _daily_rows(trade_date: date, *, start: float, drift: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    rows = []
    for index in range(days):
        day = first + timedelta(days=index)
        close = start + drift * index
        rows.append(
            {
                "date": day.isoformat(),
                "open": close - 10,
                "high": close + 20,
                "low": close - 20,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows


def _flow_rows(trade_date: date, *, value: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    return [
        {"date": (first + timedelta(days=index)).isoformat(), "foreign_net": value, "inst_net": value}
        for index in range(days)
    ]


def _foreign_rows(trade_date: date, *, value: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    return [{"date": (first + timedelta(days=index)).isoformat(), "foreign_net": value} for index in range(days)]


def _institutional_rows(trade_date: date, *, value: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    return [{"date": (first + timedelta(days=index)).isoformat(), "institutional_net": value} for index in range(days)]


def _bar(symbol: str, day: date, hour: int, minute: int, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(day, datetime.min.time(), tzinfo=KST).replace(hour=hour, minute=minute),
        timeframe="5m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
        is_completed=True,
        source="unit",
    )
