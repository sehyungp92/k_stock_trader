from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from backtests.auto.cli import main as auto_cli_main
from backtests.auto.shared.types import Experiment
from backtests.strategies.olr import research_sweep
from backtests.strategies.olr.trade_plan_sweep import load_candidate_sources
from strategy_common.clock import KST
from strategy_common.market import MarketBar


def test_olr_complete_universe_requires_daily_flow_sector_and_intraday(monkeypatch, tmp_path) -> None:
    config = {"symbols": ["000001", "000002"]}
    non_empty = pd.DataFrame([{"date": "2026-01-01", "value": 1.0}])
    empty = pd.DataFrame()

    def symbol_frame(_root, symbol, *args, **kwargs):
        return non_empty if symbol == "000001" else empty

    monkeypatch.setattr(research_sweep, "load_daily_ohlcv", symbol_frame)
    monkeypatch.setattr(research_sweep, "load_daily_flow", symbol_frame)
    monkeypatch.setattr(research_sweep, "load_daily_foreign_flow", symbol_frame)
    monkeypatch.setattr(research_sweep, "load_daily_institutional_flow", symbol_frame)
    monkeypatch.setattr(research_sweep, "load_sector_map", lambda _root: {"000001": "TECH"})
    monkeypatch.setattr(research_sweep, "_has_intraday_data", lambda _root, symbol, _timeframe: symbol == "000001")

    selected, excluded = research_sweep.resolve_complete_daily_flow_intraday_universe(
        config,
        tmp_path,
        tmp_path,
        date(2026, 1, 31),
        expected_universe_size=1,
    )

    assert selected == ("000001",)
    assert "000002" in excluded
    with pytest.raises(ValueError, match="requires exactly 2"):
        research_sweep.resolve_complete_daily_flow_intraday_universe(
            config,
            tmp_path,
            tmp_path,
            date(2026, 1, 31),
            expected_universe_size=2,
        )


def test_olr_overnight_score_rewards_net_return_and_mfe() -> None:
    good = {
        "snapshot_count": 2.0,
        "valid_candidate_days": 4.0,
        "active_day_share": 1.0,
        "avg_active_net_return_pct": 0.008,
        "avg_active_mfe_r": 1.2,
        "active_mfe_ge_0_75_share": 0.60,
        "active_bad_mae_le_neg_1_share": 0.0,
        "active_low_mfe_lt_0_1_share": 0.0,
        "overnight_score": 80.0,
    }
    bad = {
        **good,
        "avg_active_net_return_pct": -0.004,
        "avg_active_mfe_r": 0.10,
        "active_mfe_ge_0_75_share": 0.05,
        "overnight_score": 10.0,
    }
    low_quality = {**bad, "avg_active_mfe_r": 0.01}

    good_score, good_reject = research_sweep.score_overnight_metrics(good)
    bad_score, bad_reject = research_sweep.score_overnight_metrics(bad)
    low_score, low_reject = research_sweep.score_overnight_metrics(low_quality)

    assert good_reject == ""
    assert bad_reject == ""
    assert good_score > bad_score
    assert low_score == 0.0
    assert low_reject == "too_low_candidate_mfe"


def test_olr_stage2_portfolio_proxy_rewards_deployable_return() -> None:
    base = {
        "snapshot_count": 60.0,
        "valid_candidate_days": 120.0,
        "active_day_share": 0.75,
        "avg_active_net_return_pct": 0.004,
        "avg_active_mfe_r": 1.0,
        "active_mfe_ge_0_75_share": 0.60,
        "active_bad_mae_le_neg_1_share": 0.20,
        "active_low_mfe_lt_0_1_share": 0.10,
        "overnight_score": 70.0,
        "portfolio_proxy_active_day_net_pct": 0.004,
        "portfolio_proxy_avg_active_gross_exposure_pct": 0.90,
        "portfolio_proxy_max_drawdown_pct": -0.08,
        "stage2_too_many_names_penalty": 0.0,
    }
    good_score, good_reject = research_sweep.score_stage2_portfolio_metrics(
        {**base, "portfolio_proxy_net_return_pct": 0.60}
    )
    weak_score, weak_reject = research_sweep.score_stage2_portfolio_metrics(
        {**base, "portfolio_proxy_net_return_pct": 0.05}
    )
    underdeployed_score, underdeployed_reject = research_sweep.score_stage2_portfolio_metrics(
        {**base, "portfolio_proxy_net_return_pct": 0.60, "portfolio_proxy_avg_active_gross_exposure_pct": 0.20}
    )

    assert good_reject == ""
    assert weak_reject == ""
    assert good_score > weak_score
    assert underdeployed_score == 0.0
    assert underdeployed_reject == "underdeployed_stage2_portfolio_proxy"


def test_olr_stage2_portfolio_proxy_rejects_negative_top_score_band() -> None:
    metrics = {
        "snapshot_count": 120.0,
        "valid_candidate_days": 160.0,
        "active_day_share": 0.80,
        "avg_active_net_return_pct": 0.004,
        "avg_active_mfe_r": 1.0,
        "active_mfe_ge_0_75_share": 0.60,
        "active_bad_mae_le_neg_1_share": 0.20,
        "active_low_mfe_lt_0_1_share": 0.10,
        "overnight_score": 70.0,
        "portfolio_proxy_active_day_net_pct": 0.004,
        "portfolio_proxy_avg_active_gross_exposure_pct": 0.90,
        "portfolio_proxy_max_drawdown_pct": -0.08,
        "portfolio_proxy_net_return_pct": 0.45,
        "stage2_too_many_names_penalty": 0.0,
        "stage2_score_band_sample_count": 120.0,
        "stage2_score_monotonicity": 0.33,
        "stage2_score_mid_half_net_pct": 0.006,
        "stage2_score_top_quartile_net_pct": -0.002,
        "stage2_score_top_loss_share": 0.70,
        "stage2_negative_selected_share": 0.55,
        "stage2_alpha_capture_pct": 0.25,
    }

    score, reject = research_sweep.score_stage2_portfolio_metrics(metrics)

    assert score == 0.0
    assert reject == "stage2_top_score_band_negative"


def test_olr_trade_plan_sources_use_stage2_parent_stage1_seed() -> None:
    payload = {
        "base_mutations": {"olr.overnight.slot_count": 4},
        "selected_stage1_seed": {"name": "fallback", "mutations": {"olr.research.min_trend_score": 10.0}},
        "stage2_frontier": [
            {
                "name": "s1a__stage2a",
                "score": 90.0,
                "mutations": {"olr.afternoon.top_n": 2},
                "stage1_seed": {"name": "s1a", "mutations": {"olr.research.min_trend_score": 75.0}},
            },
            {
                "name": "s1b__stage2b",
                "score": 80.0,
                "mutations": {"olr.afternoon.top_n": 4},
                "stage1_seed": {"name": "s1b", "mutations": {"olr.research.min_rs_percentile": 60.0}},
            },
        ],
    }

    sources = load_candidate_sources(payload, top_n=2)

    assert sources[0].stage1_name == "s1a"
    assert sources[0].stage1_mutations == {"olr.research.min_trend_score": 75.0}
    assert sources[0].mutations["olr.research.min_trend_score"] == 75.0
    assert sources[1].stage1_name == "s1b"
    assert sources[1].stage1_mutations == {"olr.research.min_rs_percentile": 60.0}
    assert sources[1].mutations["olr.research.min_rs_percentile"] == 60.0


def test_olr_stage2_resume_loads_stage1_seed_bundle(tmp_path) -> None:
    artifact = tmp_path / "stage1.json"
    artifact.write_text(
        json.dumps(
            {
                "strategy": "olr",
                "sweep_hash": "prior-hash",
                "sweep_version": "prior-version",
                "strategy_core_version": "olr-research-v1",
                "stage1_candidate_count": 99,
                "stage1_coarse_candidate_count": 10,
                "stage1_refinement_candidate_count": 89,
                "fast_replay_policy": {"stage2_portfolio_proxy_version": "old-stage2-version"},
                "stage1_frontier": [
                    {"name": "frontier", "score": 3.0, "mutations": {"olr.research.min_rs_percentile": 55.0}},
                ],
                "stage1_refinement_seeds": [
                    {"name": "refine", "score": 2.0, "mutations": {"olr.research.min_trend_score": 70.0}},
                ],
                "stage1_stage2_seeds": [
                    {"name": "seed-a", "score": 9.0, "mutations": {"olr.research.min_trend_score": 75.0}},
                    {"name": "seed-b", "score": 8.0, "mutations": {"olr.research.min_rs_percentile": 60.0}},
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = research_sweep._load_stage1_resume_bundle(artifact, stage1_stage2_seed_count=1)

    assert bundle.source_sweep_hash == "prior-hash"
    assert bundle.source_stage2_portfolio_proxy_version == "old-stage2-version"
    assert bundle.stage1_candidate_count == 99
    assert [row.experiment.name for row in bundle.stage1_stage2_seeds] == ["seed-a"]
    assert bundle.stage1_stage2_seeds[0].experiment.mutations == {"olr.research.min_trend_score": 75.0}
    assert bundle.stage1_rows[0].experiment.name == "seed-a"
    assert bundle.stage1_refinement_seeds[0].experiment.name == "refine"


def test_olr_research_sweep_resumes_stage1_without_rebuilding_stage1(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "stage1.json"
    artifact.write_text(
        json.dumps(
            {
                "strategy": "olr",
                "sweep_hash": "prior-hash",
                "stage1_candidate_count": 7,
                "stage1_stage2_seeds": [
                    {"name": "seed-a", "score": 9.0, "mutations": {"olr.research.top_long_count": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(research_sweep, "prepare_research_sweep_dataset", lambda *args, **kwargs: _mini_dataset())
    monkeypatch.setattr(
        research_sweep,
        "build_research_sweep_candidates",
        lambda: (_ for _ in ()).throw(AssertionError("stage1 builder should not run")),
    )

    payload = research_sweep.run_research_sweep(
        {},
        output_dir=tmp_path / "out",
        fold_count=1,
        max_refinement_candidates=1,
        stage1_stage2_seed_count=1,
        audit_finalist_count=1,
        resume_stage1_artifact=artifact,
        expected_universe_size=2,
    )

    assert payload["stage1_resume"]["enabled"] is True
    assert payload["stage1_resume"]["source_sweep_hash"] == "prior-hash"
    assert payload["stage1_candidate_count"] == 7
    assert payload["stage1_stage2_seed_count"] == 1
    assert payload["stage2_candidate_count"] == 2


def test_olr_research_sweep_cli_dry_run(capsys, tmp_path) -> None:
    assert auto_cli_main(
        [
            "research-sweep",
            "--strategy",
            "olr",
            "--config",
            "config/optimization/olr.yaml",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "olr"
    assert payload["dry_run"] is True
    assert payload["sweep_type"] == "overnight_leader_rotation_research_training_only"
    assert payload["expected_stock_universe_symbols"] == 103
    assert payload["official_performance"] is False
    assert payload["metric_contract"]["basis"] == "research_selection_label_only"
    assert payload["metric_contract"]["headline_allowed"] is False
    assert payload["implementation_lessons_contract"]["status"] == "research_only_thin_selector"
    assert "strategy_olr.research.afternoon_selection_from_snapshot" in payload["implementation_lessons_contract"]["shared_selection_api"]
    assert payload["implementation_lessons_contract"]["reference_pattern"]["live_data_builder"].endswith("artifact_generator.py")
    assert payload["fast_replay_policy"]["mode"] == "compiled_causal_research_replay"


def test_olr_research_sweep_cli_dry_run_reports_stage1_resume(capsys, tmp_path) -> None:
    assert auto_cli_main(
        [
            "research-sweep",
            "--strategy",
            "olr",
            "--config",
            "config/optimization/olr.yaml",
            "--output-root",
            str(tmp_path),
            "--resume-stage1-artifact",
            str(tmp_path / "prior.json"),
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["resume_stage1_artifact"].endswith("prior.json")
    assert payload["stage1_resume_enabled"] is True


def test_olr_generated_sweep_payload_contracts_are_lesson_aligned() -> None:
    lessons = research_sweep._implementation_lessons_contract()
    metrics = research_sweep._metric_contract()

    assert lessons["shared_selection_api"]["stage1"] == "strategy_olr.research.daily_selection_from_snapshot"
    assert lessons["shared_selection_api"]["stage2"] == "strategy_olr.research.afternoon_selection_from_snapshot"
    assert "research_replay.py" in lessons["reference_pattern"]["replay_data_builder"]
    assert "row_date < trade_date" in lessons["completed_bar_policy"]["daily"]
    assert "timestamp < 14:30 KST" in lessons["completed_bar_policy"]["stage2_intraday"]
    assert metrics["official_performance"] is False
    assert metrics["headline_allowed"] is False
    assert "MTM" in metrics["risk_metrics"]


def test_olr_fast_research_cache_matches_full_rebuild() -> None:
    dataset = _mini_dataset()
    mutations = _mini_mutations()

    cache = research_sweep.research_snapshots_for_dataset(dataset, mutations)
    fast = research_sweep.snapshots_for_experiment(dataset, mutations, research_snapshots=cache)
    full = research_sweep.snapshots_for_experiment(dataset, mutations)

    assert research_sweep._snapshot_set_hash(fast) == research_sweep._snapshot_set_hash(full)
    assert [candidate.symbol for candidate in fast[dataset.trading_dates[0]].candidates] == [
        candidate.symbol for candidate in full[dataset.trading_dates[0]].candidates
    ]


def test_olr_full_audit_replays_match_fast_stage_results() -> None:
    dataset = _mini_dataset()
    mutations = _mini_mutations()
    folds = research_sweep._resolve_folds(list(dataset.trading_dates), fold_days=None, fold_count=1)
    cache = research_sweep.research_snapshots_for_dataset(dataset, mutations)
    stage1_row = research_sweep.evaluate_stage1_experiment(
        Experiment("__baseline__", {}),
        dataset,
        mutations,
        folds,
        research_snapshots=cache,
    )
    stage1_snapshots = research_sweep.snapshots_for_experiment(dataset, mutations, research_snapshots=cache)
    stage2_row = research_sweep.evaluate_stage2_experiment(
        Experiment("__stage2_baseline__", {}),
        dataset,
        mutations,
        folds,
        stage1_snapshots=stage1_snapshots,
    )

    stage1_audit = research_sweep._audit_stage1_result(stage1_row, dataset, mutations, folds, tolerance=1e-12)
    stage2_audit = research_sweep._audit_stage2_results([stage2_row], dataset, mutations, folds, tolerance=1e-12)[0]

    assert stage1_audit["audit_pass"] is True
    assert stage1_audit["max_abs_metric_delta"] == pytest.approx(0.0)
    assert stage2_audit["audit_pass"] is True
    assert stage2_audit["artifact_hash_match"] is True


def _mini_mutations() -> dict:
    return {
        "olr.research.top_long_count": 1,
        "olr.research.min_adv20_krw": 1_000_000,
        "olr.signal.daily_min_score": 0.0,
        "olr.overnight.slot_count": 1,
        "olr.afternoon.top_n": 1,
    }


def _mini_dataset() -> research_sweep.OLRResearchSweepDataset:
    trade_date = date(2026, 2, 2)
    next_date = trade_date + timedelta(days=1)
    symbols = ("005930", "000660")
    daily = {
        "005930": _daily_rows(trade_date, start=50_000, drift=120) + [_label_row(trade_date, 60_000), _label_row(next_date, 61_200)],
        "000660": _daily_rows(trade_date, start=80_000, drift=20) + [_label_row(trade_date, 81_000), _label_row(next_date, 80_500)],
    }
    flow = {
        "005930": _flow_rows(trade_date, value=10_000_000),
        "000660": _flow_rows(trade_date, value=-2_000_000),
    }
    bars_by_key = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 60_000, 60_800, 59_800, 60_500),
            _bar("005930", trade_date, 14, 25, 60_500, 61_100, 60_400, 60_900),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 81_000, 81_100, 80_500, 80_800),
            _bar("000660", trade_date, 14, 25, 80_800, 80_900, 80_100, 80_200),
        ),
    }
    return research_sweep.OLRResearchSweepDataset(
        config={},
        source_fingerprint="unit-source",
        daily_source_fingerprint="unit-daily",
        intraday_source_fingerprint="unit-intraday",
        data_root=research_sweep.Path("unused"),
        daily_data_root=research_sweep.Path("unused"),
        timeframe="5m",
        symbols=symbols,
        requested_symbols=symbols,
        excluded_symbols={},
        intraday_available_symbols=symbols,
        intraday_unavailable_symbols=(),
        daily_by_symbol=daily,
        flow_by_symbol=flow,
        foreign_flow_by_symbol=flow,
        institutional_flow_by_symbol=flow,
        index_by_code={},
        sector_map={"005930": "TECH", "000660": "TECH"},
        trading_dates=(trade_date,),
        bars_by_key=bars_by_key,
        train_start=trade_date,
        train_end=trade_date,
        holdout_start=next_date,
        coverage_report={"complete_symbols": len(symbols), "training_sessions": 1},
    )


def _daily_rows(trade_date: date, *, start: float, drift: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    return [
        {
            "date": (first + timedelta(days=index)).isoformat(),
            "open": start + drift * index - 10,
            "high": start + drift * index + 50,
            "low": start + drift * index - 50,
            "close": start + drift * index,
            "volume": 1_000_000,
        }
        for index in range(days)
    ]


def _label_row(day: date, close: float) -> dict:
    return {"date": day.isoformat(), "open": close - 100, "high": close + 500, "low": close - 300, "close": close, "volume": 1_000_000}


def _flow_rows(trade_date: date, *, value: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    return [
        {"date": (first + timedelta(days=index)).isoformat(), "foreign_net": value, "inst_net": value, "institutional_net": value}
        for index in range(days)
    ]


def _bar(symbol: str, day: date, hour: int, minute: int, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(day, datetime.min.time(), tzinfo=KST).replace(hour=hour, minute=minute),
        timeframe="5m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100_000.0,
        is_completed=True,
        source="unit",
    )
