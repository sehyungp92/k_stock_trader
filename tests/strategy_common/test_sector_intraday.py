from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_common.sector_intraday import (
    FIRST30_CUTOFF,
    SectorIntradayMember,
    build_sector_intraday_panel,
    score_sector_members,
)


def test_sector_intraday_panel_is_causal_at_cutoff() -> None:
    trade_date = date(2026, 1, 5)
    bars_by_key = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.0),
            _bar("005930", trade_date, 9, 25, 100.0, 101.0, 99.0, 101.0),
            _bar("005930", trade_date, 9, 30, 101.0, 180.0, 101.0, 180.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 100.5, 99.5, 100.0),
            _bar("000660", trade_date, 9, 25, 100.0, 100.5, 99.5, 100.0),
        ),
        (trade_date, "035420"): (
            _bar("035420", trade_date, 9, 0, 100.0, 100.5, 99.5, 100.0),
            _bar("035420", trade_date, 9, 25, 100.0, 100.5, 99.5, 100.0),
        ),
    }

    panel = build_sector_intraday_panel(
        bars_by_key,
        {"005930": "SEMIS", "000660": "SEMIS", "035420": "SEMIS"},
        trade_dates=(trade_date,),
        cutoff=FIRST30_CUTOFF,
    )

    sector = panel.sectors_by_key[(trade_date, "SEMIS")]
    assert sector.member_count == 3
    assert sector.ret == pytest.approx((0.01 + 0.0 + 0.0) / 3.0)


def test_sector_intraday_features_are_leave_one_out() -> None:
    panel = score_sector_members(
        (
            SectorIntradayMember("005930", "SEMIS", ret=0.06, vwap_ret=0.03, close_location=0.9, rel_volume=2.0),
            SectorIntradayMember("000660", "SEMIS", ret=0.00, vwap_ret=0.00, close_location=0.5, rel_volume=1.0),
            SectorIntradayMember("035420", "SEMIS", ret=0.00, vwap_ret=0.00, close_location=0.5, rel_volume=1.0),
            SectorIntradayMember("012450", "DEFENSE", ret=0.02, vwap_ret=0.01, close_location=0.7, rel_volume=1.0),
            SectorIntradayMember("047810", "DEFENSE", ret=0.02, vwap_ret=0.01, close_location=0.7, rel_volume=1.0),
            SectorIntradayMember("064350", "DEFENSE", ret=0.02, vwap_ret=0.01, close_location=0.7, rel_volume=1.0),
        ),
        min_effective_members=2,
    )

    strong = panel.feature_for(None, "005930", sector="SEMIS")
    peer = panel.feature_for(None, "000660", sector="SEMIS")

    assert strong.effective_count == 2
    assert peer.effective_count == 2
    assert strong.ret == pytest.approx(0.0)
    assert peer.ret > strong.ret


def test_sector_intraday_small_sector_shrinks_to_market() -> None:
    panel = score_sector_members(
        (
            SectorIntradayMember("005930", "SMALL", ret=0.10, vwap_ret=0.02, close_location=0.9, rel_volume=2.0),
            SectorIntradayMember("035420", "LARGE", ret=-0.02, vwap_ret=-0.01, close_location=0.3, rel_volume=0.8),
            SectorIntradayMember("035720", "LARGE", ret=-0.02, vwap_ret=-0.01, close_location=0.3, rel_volume=0.8),
            SectorIntradayMember("012450", "LARGE", ret=-0.02, vwap_ret=-0.01, close_location=0.3, rel_volume=0.8),
        ),
        min_effective_members=3,
        shrinkage_k=3.0,
    )

    feature = panel.feature_for(None, "005930", sector="SMALL")

    assert feature.effective_count == 0
    assert feature.shrinkage_weight == pytest.approx(0.0)
    assert feature.ret == pytest.approx(-0.02)


def test_sector_intraday_unknown_or_missing_symbol_is_neutral() -> None:
    panel = score_sector_members(())

    feature = panel.feature_for(None, "091990", sector="UNKNOWN")

    assert feature.score_pct == 50.0
    assert feature.ret == 0.0
    assert feature.breadth == 0.5
    assert feature.rel_volume == 1.0
    assert feature.effective_count == 0


def test_sector_intraday_single_symbol_universe_is_neutral_leave_one_out() -> None:
    panel = score_sector_members(
        (SectorIntradayMember("005930", "SEMIS", ret=0.10, vwap_ret=0.04, close_location=0.9, rel_volume=2.0),)
    )

    feature = panel.feature_for(None, "005930", sector="SEMIS")
    sector = panel.sectors_by_key[(None, "SEMIS")]

    assert feature.score_pct == 50.0
    assert feature.ret == 0.0
    assert feature.effective_count == 0
    assert sector.score_pct == 50.0


def _bar(symbol: str, trade_date: date, hour: int, minute: int, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST) + timedelta(hours=hour, minutes=minute),
        timeframe="5m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1_000.0,
    )
