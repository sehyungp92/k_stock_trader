from __future__ import annotations

from datetime import date, timedelta

import pytest

from strategy_common.sector_daily import SectorDailyMember, build_sector_daily_panel, score_sector_daily_members


def test_sector_daily_panel_is_causal_and_excludes_trade_date_rows() -> None:
    trade_date = date(2026, 2, 2)
    rows = _rows(trade_date, start=100.0, drift=1.0)
    contaminated = rows + [{"date": trade_date.isoformat(), "open": 1.0, "high": 1_000.0, "low": 1.0, "close": 1_000.0, "volume": 9_999_999.0}]

    clean = build_sector_daily_panel({"005930": rows}, {"005930": "SEMIS"}, trade_dates=(trade_date,))
    dirty = build_sector_daily_panel({"005930": contaminated}, {"005930": "SEMIS"}, trade_dates=(trade_date,))

    assert dirty.sectors_by_key[(trade_date, "SEMIS")].ret_5d == pytest.approx(clean.sectors_by_key[(trade_date, "SEMIS")].ret_5d)
    assert dirty.feature_for(trade_date, "005930", sector="SEMIS").ret_20d == pytest.approx(clean.feature_for(trade_date, "005930", sector="SEMIS").ret_20d)


def test_sector_daily_features_are_leave_one_out() -> None:
    panel = score_sector_daily_members(
        (
            SectorDailyMember("005930", "SEMIS", ret_5d=0.08, ret_20d=0.20, ret_60d=0.30, above_sma20=True, rel_volume=2.0),
            SectorDailyMember("000660", "SEMIS", ret_5d=0.00, ret_20d=0.00, ret_60d=0.00, above_sma20=False, rel_volume=1.0),
            SectorDailyMember("042700", "SEMIS", ret_5d=0.00, ret_20d=0.00, ret_60d=0.00, above_sma20=False, rel_volume=1.0),
            SectorDailyMember("012450", "DEFENSE", ret_5d=0.03, ret_20d=0.12, ret_60d=0.20, above_sma20=True, rel_volume=1.2),
            SectorDailyMember("047810", "DEFENSE", ret_5d=0.03, ret_20d=0.12, ret_60d=0.20, above_sma20=True, rel_volume=1.2),
            SectorDailyMember("064350", "DEFENSE", ret_5d=0.03, ret_20d=0.12, ret_60d=0.20, above_sma20=True, rel_volume=1.2),
        ),
        min_effective_members=2,
    )

    strong = panel.feature_for(None, "005930", sector="SEMIS")
    peer = panel.feature_for(None, "000660", sector="SEMIS")

    assert strong.effective_count == 2
    assert strong.ret_20d == pytest.approx(0.0)
    assert peer.ret_20d > strong.ret_20d


def test_sector_daily_small_sector_shrinks_to_market() -> None:
    panel = score_sector_daily_members(
        (
            SectorDailyMember("005930", "SMALL", ret_5d=0.10, ret_20d=0.30, ret_60d=0.40, above_sma20=True, rel_volume=2.0),
            SectorDailyMember("035420", "LARGE", ret_5d=-0.02, ret_20d=-0.08, ret_60d=-0.10, above_sma20=False, rel_volume=0.8),
            SectorDailyMember("035720", "LARGE", ret_5d=-0.02, ret_20d=-0.08, ret_60d=-0.10, above_sma20=False, rel_volume=0.8),
            SectorDailyMember("012450", "LARGE", ret_5d=-0.02, ret_20d=-0.08, ret_60d=-0.10, above_sma20=False, rel_volume=0.8),
        )
    )

    feature = panel.feature_for(None, "005930", sector="SMALL")

    assert feature.effective_count == 0
    assert feature.shrinkage_weight == pytest.approx(0.0)
    assert feature.ret_20d == pytest.approx(-0.08)


def test_sector_daily_equal_sector_scores_rank_neutral() -> None:
    panel = score_sector_daily_members(
        (
            SectorDailyMember("005930", "SEMIS", ret_5d=0.03, ret_20d=0.10, ret_60d=0.20, above_sma20=True, rel_volume=1.2),
            SectorDailyMember("000660", "SEMIS", ret_5d=0.03, ret_20d=0.10, ret_60d=0.20, above_sma20=True, rel_volume=1.2),
            SectorDailyMember("012450", "DEFENSE", ret_5d=0.03, ret_20d=0.10, ret_60d=0.20, above_sma20=True, rel_volume=1.2),
            SectorDailyMember("047810", "DEFENSE", ret_5d=0.03, ret_20d=0.10, ret_60d=0.20, above_sma20=True, rel_volume=1.2),
        )
    )

    assert panel.sectors_by_key[(None, "SEMIS")].score_pct == pytest.approx(50.0)
    assert panel.sectors_by_key[(None, "DEFENSE")].score_pct == pytest.approx(50.0)
    assert panel.feature_for(None, "005930", sector="SEMIS").score_pct == pytest.approx(50.0)


def test_sector_daily_unknown_or_short_history_is_neutral() -> None:
    panel = build_sector_daily_panel(
        {"005930": _rows(date(2026, 2, 2), start=100.0, drift=1.0, days=10)},
        {"005930": "SEMIS"},
        trade_dates=(date(2026, 2, 2),),
    )

    feature = panel.feature_for(date(2026, 2, 2), "005930", sector="SEMIS")

    assert feature.score_pct == 50.0
    assert feature.ret_20d == 0.0
    assert feature.effective_count == 0


def test_sector_daily_single_symbol_universe_is_neutral_leave_one_out() -> None:
    trade_date = date(2026, 2, 2)
    panel = build_sector_daily_panel(
        {"005930": _rows(trade_date, start=100.0, drift=1.0)},
        {"005930": "SEMIS"},
        trade_dates=(trade_date,),
    )

    feature = panel.feature_for(trade_date, "005930", sector="SEMIS")

    assert feature.score_pct == 50.0
    assert feature.regime == "UNKNOWN"
    assert feature.effective_count == 0


def test_sector_daily_flow_is_optional_and_can_change_score() -> None:
    trade_date = date(2026, 2, 2)
    daily = {
        "005930": _rows(trade_date, start=100.0, drift=1.0),
        "000660": _rows(trade_date, start=100.0, drift=1.0),
        "012450": _rows(trade_date, start=100.0, drift=0.2),
        "047810": _rows(trade_date, start=100.0, drift=0.2),
    }
    sector_map = {"005930": "SEMIS", "000660": "SEMIS", "012450": "DEFENSE", "047810": "DEFENSE"}
    no_flow = build_sector_daily_panel(daily, sector_map, trade_dates=(trade_date,))
    with_flow = build_sector_daily_panel(
        daily,
        sector_map,
        trade_dates=(trade_date,),
        flow_by_symbol={"000660": _flow_rows(trade_date, foreign=200_000.0, inst=200_000.0), "005930": _flow_rows(trade_date, foreign=200_000.0, inst=200_000.0)},
    )

    assert with_flow.feature_for(trade_date, "005930", sector="SEMIS").raw_score > no_flow.feature_for(trade_date, "005930", sector="SEMIS").raw_score


def test_sector_daily_metadata_keys_and_decimal_units_are_stable() -> None:
    trade_date = date(2026, 2, 2)
    panel = build_sector_daily_panel(
        {"005930": _rows(trade_date, start=100.0, drift=1.0), "000660": _rows(trade_date, start=100.0, drift=0.5)},
        {"005930": "SEMIS", "000660": "SEMIS"},
        trade_dates=(trade_date,),
    )

    metadata = panel.feature_for(trade_date, "005930", sector="SEMIS").metadata()

    assert "sector_daily_score_pct" in metadata
    assert "sector_daily_version" in metadata
    assert abs(metadata["sector_daily_ret_5d"]) < 1.0


def _rows(trade_date: date, *, start: float, drift: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    out = []
    for index in range(days):
        day = first + timedelta(days=index)
        close = start + drift * index
        out.append(
            {
                "date": day.isoformat(),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    return out


def _flow_rows(trade_date: date, *, foreign: float, inst: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    return [
        {"date": (first + timedelta(days=index)).isoformat(), "foreign_net": foreign, "inst_net": inst}
        for index in range(days)
    ]
