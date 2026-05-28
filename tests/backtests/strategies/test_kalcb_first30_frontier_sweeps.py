from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.first30 import build_first30_features, completed_first30_bars

from backtests.strategies.kalcb.first30_signal_sweep import (
    First30Spec,
    KALCBFirst30Dataset,
    Selection,
    build_contexts,
    daily_feature,
    evaluate_selections,
    flow_feature,
    passes,
)
from backtests.strategies.kalcb.premarket_frontier_sweep import (
    FrontierSpec,
    PairResult,
    PairSpec,
    PremarketFeature,
    _assign_pareto_scores,
    name_frontier,
    score_pair_metrics,
    select_frontier,
)


TRADE_DATE = date(2026, 1, 5)


def test_first30_context_is_causal_and_entry_uses_0930_open() -> None:
    dataset, expected = _dataset(symbols=("000001",))
    contexts = build_contexts(dataset)
    ctx = contexts[TRADE_DATE][0]
    shared = build_first30_features(
        dataset.bars_by_key[(TRADE_DATE, "000001")],
        prior_close=ctx.daily.prev_close,
        daily_atr=ctx.daily.atr14,
        expected_30m_volume=ctx.intraday.expected_30m_volume,
    )

    assert shared is not None
    assert ctx.intraday.open == pytest.approx(expected["first30_open"])
    assert ctx.intraday.close == pytest.approx(expected["first30_close"])
    assert ctx.first30_ret == pytest.approx(shared.first30_ret)
    assert ctx.vwap_ret == pytest.approx(shared.vwap_ret)
    assert ctx.gap == pytest.approx(shared.gap)
    assert ctx.rel_volume == pytest.approx(shared.rel_volume)
    assert ctx.close_location == pytest.approx(shared.range_close_location)
    assert shared.signal_bar_timestamp.endswith("09:25:00+09:00")
    assert ctx.post_bars[0].timestamp.astimezone(KST).time() == time(9, 30)

    rows = evaluate_selections(
        dataset,
        [Selection(TRADE_DATE, "000001", 1.0, "unit")],
        KALCBConfig(flatten_time=time(15, 15)),
    )

    assert len(rows) == 1
    assert rows[0].gross_eod_pct == pytest.approx(expected["eod_close"] / expected["entry_open"] - 1.0)


def test_shared_first30_helper_uses_exact_completed_0900_to_0925_window() -> None:
    dataset, _ = _dataset(symbols=("000001",))
    bars = dataset.bars_by_key[(TRADE_DATE, "000001")]
    first = completed_first30_bars(bars)
    with_late_poison = build_first30_features(
        (*bars, _bar("000001", TRADE_DATE, 9, 30, 1.0, 9999.0, 1.0, 9999.0, 1.0)),
        prior_close=100.0,
        daily_atr=5.0,
        expected_30m_volume=10_000.0,
    )
    incomplete = MarketBar(
        symbol="000001",
        timestamp=datetime(TRADE_DATE.year, TRADE_DATE.month, TRADE_DATE.day, 9, 25, tzinfo=KST),
        timeframe="5m",
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
        is_completed=False,
    )

    assert len(first) == 6
    assert [bar.timestamp.astimezone(KST).time() for bar in first] == [time(9, minute) for minute in (0, 5, 10, 15, 20, 25)]
    assert with_late_poison is not None
    assert with_late_poison.high < 9999.0
    with pytest.raises(ValueError, match="requires completed bars"):
        completed_first30_bars((*bars[:5], incomplete))
    misaligned = MarketBar(
        symbol="000001",
        timestamp=datetime(TRADE_DATE.year, TRADE_DATE.month, TRADE_DATE.day, 9, 2, tzinfo=KST),
        timeframe="5m",
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
    )
    with pytest.raises(ValueError, match="aligned 5m"):
        completed_first30_bars((*bars, misaligned))


def test_daily_flow_and_index_features_exclude_trade_date_rows() -> None:
    dataset, expected = _dataset(symbols=("000001",))

    daily = daily_feature(dataset, "000001", TRADE_DATE)
    flow = flow_feature(dataset, "000001", TRADE_DATE)
    ctx = build_contexts(dataset)[TRADE_DATE][0]

    assert daily is not None
    assert daily.prev_close == pytest.approx(expected["prev_close"])
    assert daily.prev_close != pytest.approx(9999.0)
    assert flow.combined_1d == pytest.approx(expected["prior_combined_flow"] / expected["prior_volume"])
    assert flow.combined_1d < 1.0
    assert ctx.market.kospi_ret_1d < 1.0
    assert ctx.flow.sector_participation == pytest.approx(1.0)


def test_separate_foreign_and_institutional_flow_streams_override_combined_fallback() -> None:
    dataset, expected = _dataset(symbols=("000001",))
    combined_rows = [
        {
            **row,
            "foreign_net": -999_000_000.0,
            "inst_net": -999_000_000.0,
        }
        for row in dataset.flow_by_symbol["000001"]
    ]
    dataset = replace(
        dataset,
        flow_by_symbol={"000001": combined_rows},
        foreign_flow_by_symbol={"000001": _foreign_flow_rows(TRADE_DATE, "000001")},
        institutional_flow_by_symbol={"000001": _institutional_flow_rows(TRADE_DATE, "000001")},
    )

    flow = flow_feature(dataset, "000001", TRADE_DATE)

    assert flow.foreign_1d == pytest.approx(2000.0 / expected["prior_volume"])
    assert flow.inst_1d == pytest.approx(700.0 / expected["prior_volume"])
    assert flow.combined_1d == pytest.approx((2000.0 + 700.0) / expected["prior_volume"])
    assert flow.combined_1d > 0.0
    assert flow.agreement_5d > 0.0


def test_delta_like_first30_profile_passes_and_rejects_causally() -> None:
    dataset, _ = _dataset(symbols=("000001",))
    ctx = build_contexts(dataset)[TRADE_DATE][0]
    spec = First30Spec(
        name="",
        score_mode="gap_hold",
        top_n=1,
        min_gap=0.002,
        max_gap=0.10,
        min_first30_ret=0.0,
        min_vwap_ret=0.0,
        min_prior_ret5=0.03,
        min_low_vs_prev_close=-0.02,
    )

    assert passes(spec, ctx)
    assert not passes(First30Spec(name="", score_mode="gap_hold", top_n=1, min_gap=0.20), ctx)


def test_frontier_ordering_is_deterministic() -> None:
    spec = name_frontier(FrontierSpec(name="", mode="rs_trend", frontier_size=2))
    features = {
        TRADE_DATE: (
            _feature("000002", ret5=0.05),
            _feature("000001", ret5=0.05),
            _feature("000003", ret5=0.02),
        )
    }

    assert select_frontier(spec, features)[TRADE_DATE] == ("000001", "000002")


def test_smaller_frontier_is_not_preferred_when_performance_is_materially_worse() -> None:
    large = _pair_result(frontier_size=40, gross_slot=0.60, avg_mfe=1.20)
    small = _pair_result(frontier_size=4, gross_slot=0.20, avg_mfe=0.50)
    rows = [large, small]

    _assign_pareto_scores(rows)

    ranked = sorted(rows, key=lambda row: -row.pareto_score)
    assert ranked[0].metrics["frontier_avg_size"] == pytest.approx(40.0)


def test_score_pair_metrics_reports_rejects_for_sparse_selectors() -> None:
    return_score, mfe_score, reject = score_pair_metrics({"candidate_days": 10, "active_day_share": 0.5})

    assert return_score == 0.0
    assert mfe_score == 0.0
    assert "too_few_candidate_days" in reject


def _dataset(*, symbols: tuple[str, ...]) -> tuple[KALCBFirst30Dataset, dict[str, float]]:
    daily_by_symbol = {}
    flow_by_symbol = {}
    bars_by_key = {}
    sector_map = {}
    expected: dict[str, float] = {}
    for offset, symbol in enumerate(symbols):
        daily_rows = _daily_rows(TRADE_DATE, symbol, price_offset=offset)
        flow_rows = _flow_rows(TRADE_DATE, symbol)
        prev_close = float(daily_rows[-2]["close"])
        bars, bar_expected = _bars(symbol, TRADE_DATE, prev_close)
        daily_by_symbol[symbol] = daily_rows
        flow_by_symbol[symbol] = flow_rows
        bars_by_key[(TRADE_DATE, symbol)] = bars
        sector_map[symbol] = "TECH"
        if not expected:
            expected = {
                **bar_expected,
                "prev_close": prev_close,
                "prior_combined_flow": float(flow_rows[-2]["foreign_net"]) + float(flow_rows[-2]["inst_net"]),
                "prior_volume": float(daily_rows[-2]["volume"]),
            }
    index_by_code = {
        "KOSPI": _index_rows(TRADE_DATE),
        "KOSDAQ": _index_rows(TRADE_DATE),
    }
    return (
        KALCBFirst30Dataset(
            config={"kalcb": {"session": {"flatten_time": "15:15"}}},
            source_fingerprint="intraday-test",
            daily_source_fingerprint="daily-test",
            data_root=Path("unused"),
            daily_data_root=Path("unused"),
            timeframe="5m",
            symbols=symbols,
            data_available_symbols=symbols,
            daily_available_symbols=symbols,
            unavailable_symbols=(),
            daily_by_symbol=daily_by_symbol,
            flow_by_symbol=flow_by_symbol,
            index_by_code=index_by_code,
            trading_dates=(TRADE_DATE,),
            bars_by_key=bars_by_key,
            sector_map=sector_map,
        ),
        expected,
    )


def _daily_rows(trade_date: date, symbol: str, *, price_offset: int) -> list[dict[str, object]]:
    rows = []
    for index in range(80):
        day = trade_date - timedelta(days=80 - index)
        close = (100.0 + price_offset) * (1.006 ** index)
        rows.append(
            {
                "ticker": symbol,
                "date": day.isoformat(),
                "open": close * 0.995,
                "high": close * 1.015,
                "low": close * 0.985,
                "close": close,
                "volume": 100_000.0 + index * 1000.0,
            }
        )
    rows.append(
        {
            "ticker": symbol,
            "date": trade_date.isoformat(),
            "open": 9999.0,
            "high": 9999.0,
            "low": 9999.0,
            "close": 9999.0,
            "volume": 9999.0,
        }
    )
    return rows


def _flow_rows(trade_date: date, symbol: str) -> list[dict[str, object]]:
    rows = []
    for index in range(80):
        day = trade_date - timedelta(days=80 - index)
        rows.append(
            {
                "ticker": symbol,
                "date": day.isoformat(),
                "foreign_net": 1000.0 + index * 10.0,
                "inst_net": 500.0 + index * 5.0,
            }
        )
    rows.append({"ticker": symbol, "date": trade_date.isoformat(), "foreign_net": 1_000_000_000.0, "inst_net": 1_000_000_000.0})
    return rows


def _foreign_flow_rows(trade_date: date, symbol: str) -> list[dict[str, object]]:
    rows = []
    for index in range(80):
        day = trade_date - timedelta(days=80 - index)
        rows.append({"ticker": symbol, "date": day.isoformat(), "foreign_net": 1210.0 + index * 10.0})
    rows.append({"ticker": symbol, "date": trade_date.isoformat(), "foreign_net": 1_000_000_000.0})
    return rows


def _institutional_flow_rows(trade_date: date, symbol: str) -> list[dict[str, object]]:
    rows = []
    for index in range(80):
        day = trade_date - timedelta(days=80 - index)
        rows.append({"ticker": symbol, "date": day.isoformat(), "institutional_net": 305.0 + index * 5.0})
    rows.append({"ticker": symbol, "date": trade_date.isoformat(), "institutional_net": 1_000_000_000.0})
    return rows


def _index_rows(trade_date: date) -> list[dict[str, object]]:
    rows = []
    for index in range(80):
        day = trade_date - timedelta(days=80 - index)
        close = 100.0 * (1.002 ** index)
        rows.append(
            {
                "index_code": "KOSPI",
                "date": day.isoformat(),
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    rows.append({"index_code": "KOSPI", "date": trade_date.isoformat(), "open": 9999.0, "high": 9999.0, "low": 9999.0, "close": 9999.0, "volume": 1.0})
    return rows


def _bars(symbol: str, trade_date: date, prev_close: float) -> tuple[tuple[MarketBar, ...], dict[str, float]]:
    first_open = prev_close * 1.005
    first_closes = [first_open * value for value in (1.002, 1.004, 1.006, 1.008, 1.010, 1.012)]
    bars = []
    for index, close in enumerate(first_closes):
        open_ = first_open if index == 0 else first_closes[index - 1]
        bars.append(_bar(symbol, trade_date, 9, index * 5, open_, max(open_, close) * 1.002, min(open_, close) * 0.998, close, 10_000.0 + index * 1000.0))
    entry_open = first_closes[-1] * 1.002
    bars.extend(
        [
            _bar(symbol, trade_date, 9, 30, entry_open, entry_open * 1.04, entry_open * 0.99, entry_open * 1.02, 30_000.0),
            _bar(symbol, trade_date, 15, 15, entry_open * 1.02, entry_open * 1.05, entry_open * 1.01, entry_open * 1.03, 20_000.0),
        ]
    )
    return (
        tuple(bars),
        {
            "first30_open": first_open,
            "first30_close": first_closes[-1],
            "entry_open": entry_open,
            "eod_close": entry_open * 1.03,
        },
    )


def _bar(symbol: str, day: date, hour: int, minute: int, open_: float, high: float, low: float, close: float, volume: float) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime(day.year, day.month, day.day, hour, minute, tzinfo=KST),
        timeframe="5m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="unit",
        source_fingerprint="unit",
    )


def _feature(symbol: str, *, ret5: float) -> PremarketFeature:
    return PremarketFeature(
        day=TRADE_DATE,
        symbol=symbol,
        sector="TECH",
        ret5=ret5,
        ret20=ret5,
        ret60=ret5,
        atr_pct=0.03,
        adv20_krw=5_000_000_000.0,
        close20_loc=0.7,
        close60_loc=0.7,
        volume_surge=1.2,
        above_sma20=True,
        above_sma60=True,
        flow_1d=0.01,
        flow_3d=0.01,
        flow_5d=0.01,
        flow_20d=0.01,
        flow_notional_5d=0.01,
        flow_positive_days_5d=5.0,
        flow_acceleration=0.001,
        flow_z=0.5,
        sector_flow_5d=0.01,
        sector_participation=1.0,
        market_score=0.2,
        market_ret5=0.01,
        market_ret20=0.02,
        market_above_sma20=True,
    )


def _pair_result(*, frontier_size: int, gross_slot: float, avg_mfe: float) -> PairResult:
    frontier = name_frontier(FrontierSpec(name="", mode="hybrid", frontier_size=frontier_size))
    first30 = First30Spec(name="first30", score_mode="hybrid", top_n=8)
    metrics = {
        "candidate_days": 100.0,
        "active_day_share": 0.8,
        "slot_cumulative_gross_return_pct": gross_slot,
        "avg_mfe_r": avg_mfe,
        "frontier_avg_size": float(frontier_size),
        "full_first30_candidate_recall": 0.8,
    }
    return PairResult(
        spec=PairSpec(name=f"{frontier.name}__first30", frontier=frontier, first30=first30),
        return_score=1.0,
        mfe_score=1.0,
        combined_score=1.0,
        pareto_score=0.0,
        rejected=False,
        reject_reason="",
        metrics=metrics,
    )
