from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from backtests.strategies.kalcb import stage2_calibration
from backtests.strategies.kalcb.first30_signal_sweep import (
    DailyFeature,
    First30Context,
    First30Intraday,
    First30Spec,
    FlowFeature,
    KALCBFirst30Dataset,
    MarketFeature,
    Selection,
    score_candidate,
)
from backtests.strategies.kalcb.gap_retention_path_quality import (
    build_candidate_path_rows,
    build_gap_retention_report,
)
from backtests.strategies.kalcb.kalcb_path_quality_v1 import (
    PathObservation,
    build_path_quality_observations,
    fit_interaction_regime_model,
    fit_path_quality_model,
    score_interaction_regime,
    score_path_calibrated_row,
    summarize_path_risk,
)
from backtests.strategies.kalcb.premarket_frontier_sweep import FrontierSpec, name_frontier
from backtests.strategies.kalcb.stage2_calibration import select_calibrated_stage2_rows
from backtests.strategies.kalcb.trade_plan_sweep import (
    TradeOutcome,
    compile_core_replay,
    load_fixed_candidate_source,
)
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_common.sector_daily import SectorDailyFeature
from strategy_kalcb.config import KALCBConfig


TRADE_DATE = date(2026, 1, 5)


def test_path_quality_features_do_not_read_beyond_completed_horizon() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.0, 101.0, 99.5, 100.4),
            (100.4, 102.0, 100.2, 101.4),
            (101.4, 103.0, 101.0, 102.8),
            (102.8, 150.0, 102.0, 149.0),
        ]
    )
    outcome = _outcome(entry_time=bars[6].timestamp, entry=100.0, risk=1.0, net_pct=0.49, mfe_r=50.0, mae_r=-0.5)

    observations = build_path_quality_observations(
        [outcome],
        SimpleNamespace(snapshots={}),
        {(TRADE_DATE, "005930"): _ctx(bars)},
        {(TRADE_DATE, "005930"): tuple(bars)},
        horizons=(1, 3),
    )

    features = observations[0].features
    assert features["h1_mfe_r"] == pytest.approx(1.0)
    assert features["h3_mfe_r"] == pytest.approx(3.0)
    assert "h12_mfe_r" not in features


def test_path_quality_base_features_include_gap_retention_terms() -> None:
    bars = _bars([(100.0, 101.0, 100.0, 101.0)] * 7)
    outcome = _outcome(entry_time=bars[6].timestamp, entry=101.0, risk=1.0, net_pct=0.01, mfe_r=2.0, mae_r=-0.5)

    observation = build_path_quality_observations(
        [outcome],
        SimpleNamespace(snapshots={}),
        {(TRADE_DATE, "005930"): _ctx(bars)},
        {(TRADE_DATE, "005930"): tuple(bars)},
        horizons=(1,),
    )[0]

    assert "first30_gap_retention_ratio" in observation.features
    assert "first30_gap_relvol" in observation.features
    assert "first30_low_vs_prev_relvol" in observation.features


def test_path_quality_base_features_include_joint_daily_sector_terms() -> None:
    bars = _bars([(100.0, 101.0, 100.0, 101.0)] * 7)
    outcome = _outcome(entry_time=bars[6].timestamp, entry=101.0, risk=1.0, net_pct=0.01, mfe_r=2.0, mae_r=-0.5)

    observation = build_path_quality_observations(
        [outcome],
        SimpleNamespace(snapshots={}),
        {(TRADE_DATE, "005930"): _ctx(bars)},
        {(TRADE_DATE, "005930"): tuple(bars)},
        horizons=(1,),
    )[0]

    assert observation.features["daily_acceleration_5v20"] == pytest.approx(0.10)
    assert "daily_momentum_pct" in observation.features
    assert "first30_sector_leadership_pct" in observation.features
    assert "continuation_joint_quality_pct" in observation.features


def test_daily_sector_source_score_rewards_stock_leadership_with_sector_participation() -> None:
    bars = _bars([(100.0, 101.4, 100.8, 101.2)] * 6 + [(101.2, 101.8, 101.0, 101.4)])
    base = _ctx(bars)
    strong = replace(
        base,
        rel_volume=6.0,
        low_vs_prev_close=0.018,
        sector_daily=SectorDailyFeature(
            symbol=base.symbol,
            sector=base.sector,
            trade_date=TRADE_DATE,
            score_pct=88.0,
            ret_5d=0.025,
            ret_20d=0.040,
            participation=0.72,
        ),
    )
    weak = replace(
        base,
        rel_volume=1.0,
        low_vs_prev_close=-0.010,
        sector_daily=SectorDailyFeature(
            symbol=base.symbol,
            sector=base.sector,
            trade_date=TRADE_DATE,
            score_pct=28.0,
            ret_5d=0.120,
            ret_20d=0.260,
            participation=0.12,
        ),
    )
    spec = First30Spec(name="unit", score_mode="daily_sector_leadership", top_n=1)

    assert score_candidate(spec, strong) > score_candidate(spec, weak)


def test_path_quality_model_rejects_inconsistent_fold_lift() -> None:
    folds = [(date(2026, 1, 1), date(2026, 1, 2)), (date(2026, 1, 3), date(2026, 1, 4))]
    stable = [
        _observation(date(2026, 1, 1), 1.0, 3.0),
        _observation(date(2026, 1, 2), 0.0, -1.0),
        _observation(date(2026, 1, 3), 1.0, 3.0),
        _observation(date(2026, 1, 4), 0.0, -1.0),
    ]
    unstable = [
        _observation(date(2026, 1, 1), 1.0, 3.0),
        _observation(date(2026, 1, 2), 0.0, -1.0),
        _observation(date(2026, 1, 3), 1.0, -2.0),
        _observation(date(2026, 1, 4), 0.0, 2.0),
    ]

    assert fit_path_quality_model(stable, folds)["accepted"] is True
    rejected = fit_path_quality_model(unstable, folds)
    assert rejected["accepted"] is False
    assert rejected["reject_reason"] == "no_fold_stable_positive_lift_rule"


def test_interaction_regime_model_accepts_fold_stable_leadership_acceleration() -> None:
    folds = [(date(2026, 1, 1), date(2026, 1, 4)), (date(2026, 1, 5), date(2026, 1, 8))]
    observations = []
    for index in range(8):
        day = date(2026, 1, 1 + index)
        active = index in {0, 2, 4, 6}
        observations.append(
            _interaction_observation(
                day,
                leadership=92.0 if active else 48.0,
                acceleration=0.12 if active else -0.02,
                final_r=4.0 if active else -1.0,
            )
        )

    model = fit_interaction_regime_model(observations, folds)
    score = score_interaction_regime(
        {"first30_sector_leadership_pct": 95.0, "daily_acceleration_5v20": 0.14},
        model,
    )

    assert model["accepted"] is True
    assert model["rule"]["conditions"][0]["feature"] == "first30_sector_leadership_pct"
    assert score["active"] is True
    assert score["score"] > 0.0


def test_interaction_regime_model_rejects_unstable_leadership_acceleration() -> None:
    folds = [(date(2026, 1, 1), date(2026, 1, 4)), (date(2026, 1, 5), date(2026, 1, 8))]
    observations = []
    for index in range(8):
        day = date(2026, 1, 1 + index)
        active = index in {0, 2, 4, 6}
        final_r = 4.0 if active and day <= date(2026, 1, 4) else (-3.0 if active else 1.0)
        observations.append(
            _interaction_observation(
                day,
                leadership=92.0 if active else 48.0,
                acceleration=0.12 if active else -0.02,
                final_r=final_r,
            )
        )

    model = fit_interaction_regime_model(observations, folds)

    assert model["accepted"] is False
    assert model["reject_reason"] == "no_fold_stable_stock_leadership_daily_acceleration_rule"


def test_interaction_regime_score_requires_both_terms() -> None:
    model = {
        "accepted": True,
        "median_fold_lift_r": 2.5,
        "rule": {
            "conditions": (
                {"feature": "first30_sector_leadership_pct", "direction": "gte", "threshold": 80.0},
                {"feature": "daily_acceleration_5v20", "direction": "gte", "threshold": 0.05},
            )
        },
    }

    assert score_interaction_regime({"first30_sector_leadership_pct": 90.0, "daily_acceleration_5v20": 0.08}, model)["active"] is True
    assert score_interaction_regime({"first30_sector_leadership_pct": 90.0, "daily_acceleration_5v20": 0.01}, model)["active"] is False


def test_path_risk_conversion_requires_matching_selected_denominator() -> None:
    observation = _observation(TRADE_DATE, 1.0, 3.0)

    assert summarize_path_risk([observation])["conversion"] == 0.0
    assert summarize_path_risk([observation], selected_count=4.0)["conversion"] == pytest.approx(0.25)


def test_path_risk_score_can_outrank_higher_proxy_return() -> None:
    strong_path_score, strong_components = score_path_calibrated_row(
        {"broker_net_return_pct": 0.08, "broker_max_drawdown_pct": 0.03, "trade_count": 90.0},
        {"avg_mfe_capture": 0.45, "mae_le_neg_1_share": 0.30, "avg_giveback_r": 4.0},
        [{"metrics": {"portfolio_equivalent_net_return_pct": 0.03}}],
    )
    weak_path_score, weak_components = score_path_calibrated_row(
        {"broker_net_return_pct": 0.06, "broker_max_drawdown_pct": 0.06, "trade_count": 90.0},
        {"avg_mfe_capture": 0.20, "mae_le_neg_1_share": 0.90, "avg_giveback_r": 12.0},
        [{"metrics": {"portfolio_equivalent_net_return_pct": -0.01}}],
    )
    high_proxy_bad_path = _calibrated_row("high_proxy_bad_path", proxy=0.90, score=weak_path_score, components=weak_components)
    lower_proxy_good_path = _calibrated_row("lower_proxy_good_path", proxy=0.20, score=strong_path_score, components=strong_components)

    selected = select_calibrated_stage2_rows([high_proxy_bad_path, lower_proxy_good_path], finalist_count=2, require_audit_pass=True)

    assert [row["name"] for row in selected] == ["lower_proxy_good_path", "high_proxy_bad_path"]


def test_calibrated_source_metadata_flows_into_candidate_snapshot_hash(tmp_path: Path) -> None:
    frontier = name_frontier(FrontierSpec("", "rs_trend", 1))
    first30 = First30Spec("hybrid_top1", "hybrid", 1)
    source_path = tmp_path / "calibrated_source.json"
    source_path.write_text(
        json.dumps(
            {
                "sweep_hash": "unit-sweep",
                "top_path_calibrated_stage2": [
                    {
                        "name": "calibrated",
                        "frontier": asdict(frontier),
                        "first30": asdict(first30),
                        "calibration_version": "unit-v2",
                        "path_calibrated_score": 123.0,
                        "path_risk_metrics": {"avg_giveback_r": 4.0},
                        "path_quality_model": {"accepted": True},
                        "interaction_regime_model": {"accepted": True, "activation_label": "stock_leadership_plus_daily_acceleration"},
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    candidate_source = load_fixed_candidate_source(source_path, section="top_path_calibrated_stage2", rank=0, strict_expected=False)
    bars = _bars([(100.0, 100.2, 99.8, 100.0)] * 7)
    dataset = _dataset(bars)
    ctx = _ctx(bars)
    selection = Selection(TRADE_DATE, "005930", 1.0, "unit")

    calibrated = compile_core_replay(
        [selection],
        dataset,
        {(TRADE_DATE, "005930"): ctx},
        dataset.trading_dates,
        {TRADE_DATE: 1},
        KALCBConfig(),
        source_calibration_metadata=candidate_source.calibration_metadata,
    )
    plain = compile_core_replay(
        [selection],
        dataset,
        {(TRADE_DATE, "005930"): ctx},
        dataset.trading_dates,
        {TRADE_DATE: 1},
        KALCBConfig(),
    )
    candidate = calibrated.snapshots[TRADE_DATE].candidates[0]

    assert candidate.metadata["source_calibration"]["path_calibrated_score"] == 123.0
    assert candidate.metadata["source_calibration"]["interaction_regime_model"]["accepted"] is True
    assert calibrated.candidate_artifact_hash != plain.candidate_artifact_hash


def test_compiled_replay_candidate_pool_includes_frontier_shadows() -> None:
    bars_a = _bars([(100.0, 100.2, 99.8, 100.0)] * 7, symbol="005930")
    bars_b = _bars([(50.0, 50.2, 49.8, 50.0)] * 7, symbol="000660")
    dataset = _dataset(bars_a, extra_bars={"000660": bars_b})
    context_by_key = {
        (TRADE_DATE, "005930"): _ctx(bars_a, symbol="005930"),
        (TRADE_DATE, "000660"): _ctx(bars_b, symbol="000660"),
    }

    compiled = compile_core_replay(
        [Selection(TRADE_DATE, "005930", 1.0, "unit")],
        dataset,
        context_by_key,
        dataset.trading_dates,
        {TRADE_DATE: 1},
        KALCBConfig(),
        frontier_by_day={TRADE_DATE: ("005930", "000660")},
        frontier_scores_by_day={TRADE_DATE: {"005930": 2.0, "000660": 1.0}},
    )

    snapshot = compiled.snapshots[TRADE_DATE]
    assert snapshot.metadata["active_symbol_count"] == 1
    assert snapshot.metadata["candidate_pool_count"] == 2
    assert [candidate.metadata["frontier_role"] for candidate in snapshot.candidates] == ["initial_active", "frontier_shadow"]
    assert snapshot.candidates[1].metadata["frontier_rank"] == 2


def test_gap_retention_report_uses_train_thresholds_on_candidate_pool() -> None:
    strong_bars = _bars(
        [(100.0, 101.0, 100.0, 101.0)] * 6
        + [(101.0, 104.0, 100.5, 103.0), (103.0, 105.0, 102.0, 104.0)],
        symbol="005930",
    )
    weak_bars = _bars(
        [(100.0, 100.2, 98.0, 98.5)] * 6
        + [(98.5, 99.0, 96.0, 96.5), (96.5, 97.0, 95.0, 95.5)],
        symbol="000660",
    )
    dataset = _dataset(strong_bars, extra_bars={"000660": weak_bars})
    context_by_key = {
        (TRADE_DATE, "005930"): _ctx(strong_bars, symbol="005930"),
        (TRADE_DATE, "000660"): _ctx(weak_bars, symbol="000660"),
    }
    compiled = compile_core_replay(
        [Selection(TRADE_DATE, "005930", 1.0, "unit")],
        dataset,
        context_by_key,
        dataset.trading_dates,
        {TRADE_DATE: 1},
        KALCBConfig(),
        frontier_by_day={TRADE_DATE: ("005930", "000660")},
        frontier_scores_by_day={TRADE_DATE: {"005930": 2.0, "000660": 1.0}},
    )
    context = SimpleNamespace(
        train_dates=(TRADE_DATE,),
        compiled_replay=compiled,
        dataset=dataset,
        context_by_key=context_by_key,
    )

    train_rows = build_candidate_path_rows(context, window="train", stop_pct=0.003)
    report = build_gap_retention_report(train_rows, train_rows, min_holdout_rows=1)

    assert len(train_rows) == 2
    assert report["threshold_source"] == "train_only"
    assert report["summary"]["train"][0]["count"] == 2.0
    assert any(row["rule"] == "gap_q75" for row in report["rules"])


def test_stage2_calibration_passes_frontier_pool_to_compiled_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frontier = name_frontier(FrontierSpec("", "rs_trend", 2))
    first30 = First30Spec("hybrid_top1", "hybrid", 1)
    captured: dict[str, object] = {}
    base = stage2_calibration._CalibrationBase(
        training_config={},
        cfg=KALCBConfig(),
        dataset=SimpleNamespace(bars_by_key={}, trading_dates=(TRADE_DATE,)),
        contexts={TRADE_DATE: ()},
        context_by_key={},
        train_dates=(TRADE_DATE,),
    )

    monkeypatch.setattr(
        stage2_calibration,
        "build_fixed_candidate_selections",
        lambda *_args, **_kwargs: ([Selection(TRADE_DATE, "005930", 1.0, "unit")], {TRADE_DATE: ("005930", "000660")}),
    )
    monkeypatch.setattr(
        stage2_calibration,
        "_frontier_scores_by_day",
        lambda *_args, **_kwargs: {TRADE_DATE: {"005930": 2.0, "000660": 1.0}},
    )

    def fake_compile(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(initial_equity=100_000_000.0, snapshots={}, bars=())

    monkeypatch.setattr(stage2_calibration, "compile_core_replay", fake_compile)
    monkeypatch.setattr(
        stage2_calibration,
        "_core_outcomes_metrics_digest",
        lambda *_args, **_kwargs: (
            [],
            {"selected_count": 1.0, "trade_count": 0.0, "broker_net_return_pct": 0.0, "broker_max_drawdown_pct": 0.0},
            {},
        ),
    )
    monkeypatch.setattr(stage2_calibration, "_fold_metrics_from_outcomes_for_dates", lambda *_args, **_kwargs: ())

    stage2_calibration._calibrate_row(
        base,
        stage2_path=tmp_path / "stage2.json",
        stage2_payload={"sweep_hash": "unit"},
        stage2_source_hash="hash",
        source_row={"name": "unit", "frontier": asdict(frontier), "first30": asdict(first30)},
        candidate_section="top_portfolio_proxy",
        rank=0,
        cache_key="unit-cache",
        cache_path=tmp_path / "cache.json",
        baseline_spec=SimpleNamespace(name="baseline", entry=SimpleNamespace(), exit=SimpleNamespace()),
    )

    assert captured["frontier_by_day"] == {TRADE_DATE: ("005930", "000660")}
    assert captured["frontier_scores_by_day"] == {TRADE_DATE: {"005930": 2.0, "000660": 1.0}}


def _observation(day: date, feature: float, final_r: float) -> PathObservation:
    return PathObservation(
        trade_date=day,
        symbol=f"{day.day:06d}",
        features={"h3_current_r": feature},
        labels={
            "final_net_r": final_r,
            "mfe_capture": 0.5 if final_r > 0 else 0.1,
            "tail_loser": 0.0 if final_r > 0 else 1.0,
            "max_mfe_r": max(final_r, 0.0),
            "max_mae_r": -0.5 if final_r > 0 else -2.0,
            "giveback_r": 1.0,
            "loser": 0.0 if final_r > 0 else 1.0,
        },
    )


def _interaction_observation(day: date, *, leadership: float, acceleration: float, final_r: float) -> PathObservation:
    return PathObservation(
        trade_date=day,
        symbol=f"{day.day:06d}",
        features={
            "first30_sector_leadership_pct": leadership,
            "daily_acceleration_5v20": acceleration,
            "first30_quality_pct": leadership - 5.0,
            "daily_momentum_pct": 70.0 + acceleration * 100.0,
        },
        labels={
            "final_net_r": final_r,
            "mfe_capture": 0.55 if final_r > 0 else 0.20,
            "tail_loser": 0.0 if final_r > 0 else 1.0,
            "max_mfe_r": max(final_r, 0.0),
            "max_mae_r": -0.5 if final_r > 0 else -2.0,
            "giveback_r": 1.0,
            "loser": 0.0 if final_r > 0 else 1.0,
        },
    )


def _calibrated_row(name: str, *, proxy: float, score: float, components: dict[str, float]) -> dict[str, object]:
    return {
        "name": name,
        "source_rank": 0,
        "source_section": "top_portfolio_proxy",
        "proxy_net_return_pct": proxy,
        "calibrated_broker_net_return_pct": 0.05,
        "calibrated_official_mtm_net_return_pct": 0.05,
        "calibrated_broker_max_drawdown_pct": 0.03,
        "trade_count": 90.0,
        "selected_count": 100.0,
        "filled_selected_rate": 0.9,
        "path_calibrated_score": score,
        "path_score_components": components,
        "path_risk_metrics": {"avg_mfe_capture": 0.4, "avg_giveback_r": 4.0},
        "proxy_metrics": {"avg_mfe_r": 8.0},
        "audit_pass": True,
        "audit_status": "pass",
        "reject_reason": "",
    }


def _outcome(*, entry_time: datetime, entry: float, risk: float, net_pct: float, mfe_r: float, mae_r: float) -> TradeOutcome:
    return TradeOutcome(
        trade_date=TRADE_DATE,
        symbol="005930",
        entry_time=entry_time,
        entry_price=entry,
        stop_price=entry - risk,
        risk_per_share=risk,
        gross_return_pct=net_pct,
        net_return_pct=net_pct,
        mfe_r=mfe_r,
        mae_r=mae_r,
        mfe_capture=0.5,
        bars_held=4,
        exit_reason="eod_flatten",
        ambiguous_bar_count=0,
        stopped=False,
        target_hit=False,
        partial_hit=False,
        entry_type="KRX_FIRST30_OPEN",
        frontier_role="initial_active",
        candidate_rank=1,
        frontier_rank=1,
    )


def _ctx(bars: list[MarketBar], *, symbol: str = "005930") -> First30Context:
    pre = bars[:6]
    high = max(bar.high for bar in pre)
    low = min(bar.low for bar in pre)
    volume = sum(bar.volume for bar in pre)
    vwap = sum(((bar.high + bar.low + bar.close) / 3.0) * bar.volume for bar in pre) / volume
    daily = DailyFeature(
        symbol=symbol,
        trade_date=TRADE_DATE,
        prev_close=99.0,
        atr14=2.0,
        return_5d=0.10,
        return_20d=0.0,
        return_60d=0.0,
        adv20_krw=5_000_000_000.0,
        volume_ratio_20d=1.0,
        close20_loc=0.8,
        close60_loc=0.8,
        above_sma20=True,
        above_sma60=True,
    )
    intraday = First30Intraday(
        open=pre[0].open,
        high=high,
        low=low,
        close=pre[-1].close,
        vwap=vwap,
        volume=volume,
        expected_30m_volume=volume,
    )
    return First30Context(
        day=TRADE_DATE,
        symbol=symbol,
        sector="TECH",
        daily=daily,
        flow=FlowFeature(available=True),
        market=MarketFeature(score=1.0),
        intraday=intraday,
        bars=tuple(bars),
        post_bars=tuple(bar for bar in bars if bar.timestamp.astimezone(KST).time() >= time(9, 30)),
        first30_ret=intraday.close / intraday.open - 1.0,
        vwap_ret=intraday.close / intraday.vwap - 1.0,
        gap=intraday.open / daily.prev_close - 1.0,
        rel_volume=1.0,
        close_location=(intraday.close - intraday.low) / max(intraday.high - intraday.low, 1e-9),
        open_drawdown=intraday.low / intraday.open - 1.0,
        low_vs_prev_close=intraday.low / daily.prev_close - 1.0,
        range_atr=(intraday.high - intraday.low) / daily.atr14,
    )


def _dataset(bars: list[MarketBar], *, extra_bars: dict[str, list[MarketBar]] | None = None) -> KALCBFirst30Dataset:
    bars_by_symbol = {"005930": bars, **(extra_bars or {})}
    symbols = tuple(bars_by_symbol)
    return KALCBFirst30Dataset(
        config={"kalcb": {"session": {"flatten_time": "15:20"}}},
        source_fingerprint="intraday-test",
        daily_source_fingerprint="daily-test",
        data_root=Path("unused"),
        daily_data_root=Path("unused"),
        timeframe="5m",
        symbols=symbols,
        data_available_symbols=symbols,
        daily_available_symbols=symbols,
        unavailable_symbols=(),
        daily_by_symbol={
            symbol: [{"ticker": symbol, "date": (TRADE_DATE - timedelta(days=1)).isoformat(), "high": 112.0, "low": 99.0, "close": 100.0}]
            for symbol in symbols
        },
        flow_by_symbol={symbol: [] for symbol in symbols},
        index_by_code={},
        trading_dates=(TRADE_DATE,),
        bars_by_key={(TRADE_DATE, symbol): tuple(symbol_bars) for symbol, symbol_bars in bars_by_symbol.items()},
        sector_map={symbol: "TECH" for symbol in symbols},
    )


def _bars(rows: list[tuple[float, float, float, float]], *, symbol: str = "005930") -> list[MarketBar]:
    start = datetime(2026, 1, 5, 9, 0, tzinfo=KST)
    return [
        MarketBar(
            symbol=symbol,
            timestamp=start + timedelta(minutes=5 * index),
            timeframe="5m",
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=10_000.0,
        )
        for index, (open_, high, low, close) in enumerate(rows)
    ]
