from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from backtests.auto.cli import main as auto_cli_main
from backtests.cli import main as backtest_cli_main
from backtests.core.replay_bundle import EventReplayBundle
from backtests.core.replay_events import ReplayEvent
from backtests.engine.sim_broker import BrokerCosts, SimBroker, SimPosition
from backtests.strategies.common.synthetic import make_kalcb_synthetic_bars, make_synthetic_replay_bundle
from backtests.strategies.kalcb.replay_cache import _resolve_symbols, _select_active_seed
from backtests.strategies.kalcb import replay_cache as kalcb_replay_cache
from backtests.strategies.kalcb.plugin import KALCBOptimizationPlugin
from backtests.strategies.kalcb.research_sweep import (
    build_research_refinement_candidates,
    build_research_sweep_candidates,
    run_research_sweep,
    score_research_metrics,
)
from backtests.strategies.kalcb.runner import KALCBReplayAdapter, run_kalcb_backtest
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.config import KALCB_CORE_VERSION
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.state import KALCBPositionState, SymbolStage
from strategy_kalcb.models import EntryType, KALCBDailyCandidate, KALCBDailySnapshot


def test_kalcb_synthetic_backtest_runs_through_shared_core():
    result = run_kalcb_backtest({"capability_level": "synthetic"}, {})

    assert result.metrics["total_trades"] >= 1
    assert result.metrics["same_bar_fill_count"] == 0
    assert result.metrics["strategy_core_version"] == KALCB_CORE_VERSION
    assert result.metrics["live_parity_fill_timing"] == "next_5m_open"
    assert result.metrics["auction_mode"] == "non_auction_continuous"
    assert "official_mtm_net_return_pct" in result.metrics
    assert result.metrics["final_equity"] == pytest.approx(result.replay_result.equity_curve[-1])
    assert result.metrics["end_open_position_count"] == len(result.replay_result.broker.positions)
    assert result.metrics["net_return_pct_basis"] == "closed_trade_net_pnl_over_initial_equity"
    assert result.metrics["official_metric_basis"] == "SimBroker.equity_curve_bar_level_mtm"
    assert result.source_fingerprint


def test_kalcb_official_replay_plumbs_intraday_leverage_to_sim_broker():
    result = run_kalcb_backtest(
        {"capability_level": "synthetic"},
        {"kalcb.risk.intraday_leverage": 2.0, "kalcb.entry.entry_score_blocklist": []},
    )

    assert result.replay_result.broker.buying_power_leverage == pytest.approx(2.0)


def test_kalcb_frontier_shadows_nonselected_symbols_without_breaking_ws_cap():
    bars = sorted(
        make_kalcb_synthetic_bars("000660") + make_kalcb_synthetic_bars("005930"),
        key=lambda bar: (bar.timestamp, bar.symbol),
    )
    bundle = EventReplayBundle(
        events=tuple(ReplayEvent.from_bar(bar) for bar in bars),
        source_fingerprint="unit-frontier",
        data_root=None,
        metadata={"capability_level": "synthetic", "strategy": "kalcb"},
    )

    result = run_kalcb_backtest(
        {"capability_level": "synthetic"},
        {
            "kalcb.session.ws_budget": 1,
            "kalcb.frontier.size": 2,
            "kalcb.frontier.shadow_enabled": True,
            "kalcb.frontier.rotation_enabled": True,
            "kalcb.frontier.rotation_slots": 1,
            "kalcb.entry.entry_score_blocklist": [],
        },
        replay_bundle=bundle,
    )

    assert result.metrics["active_symbol_max"] == 1
    assert result.metrics["frontier_symbol_max"] == 2
    assert result.metrics["selected_universe_fraction"] <= result.metrics["frontier_universe_fraction"]
    assert result.metrics["frontier_shadow_nonselected_trade_count"] >= 1
    assert result.metrics["frontier_rotation_promotion_count"] == 0


def test_campaign_active_seed_prefers_opportunity_over_raw_liquidity():
    trade_date = make_kalcb_synthetic_bars("005930")[0].timestamp.date()
    liquid_laggard = _daily_candidate(
        "005930",
        trade_date,
        selection_score=7.0,
        prior_notional=900_000_000_000.0,
        campaign_setup_score=2.0,
        rs_20d=-0.02,
        close_location_20d=0.40,
    )
    campaign_leader = _daily_candidate(
        "373220",
        trade_date,
        selection_score=12.0,
        prior_notional=25_000_000_000.0,
        campaign_setup_score=9.0,
        rs_20d=0.14,
        close_location_20d=0.90,
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 1,
            "kalcb.frontier.size": 2,
            "kalcb.frontier.rotation_slots": 0,
            "kalcb.frontier.active_selection_mode": "campaign",
        }
    )

    selected = _select_active_seed([liquid_laggard, campaign_leader], cfg)

    assert [candidate.symbol for candidate in selected] == ["373220"]


def test_kalcb_explicit_universe_file_does_not_fall_back_to_all_parquet(tmp_path):
    data_root = tmp_path / "kis"
    for symbol in ("005930", "000660"):
        symbol_dir = data_root / symbol
        symbol_dir.mkdir(parents=True)
        (symbol_dir / f"{symbol}_5m_unit.parquet").write_text("", encoding="utf-8")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text('symbols:\n  - "005930"\n', encoding="utf-8")

    assert _resolve_symbols({"universe": str(universe_path)}, data_root, "5m") == ["005930"]

    with pytest.raises(FileNotFoundError, match="explicit universe file"):
        _resolve_symbols({"universe": str(tmp_path / "missing.yaml")}, data_root, "5m")


def test_kalcb_sector_map_resolves_config_relative_path():
    from_sector_map = kalcb_replay_cache._resolve_sector_map({"sector_map": "olr/sector_map.yaml"})
    from_sector_map_path = kalcb_replay_cache._resolve_sector_map({"sector_map_path": "olr/sector_map.yaml"})

    assert from_sector_map["005930"] == "SEMICONDUCTORS"
    assert from_sector_map_path["005930"] == "SEMICONDUCTORS"


def test_kalcb_replay_snapshot_uses_research_selection_path(tmp_path, monkeypatch):
    trade_date = make_kalcb_synthetic_bars("005930")[0].timestamp.date()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 2,
            "kalcb.frontier.size": 4,
            "kalcb.research.top_long_count": 3,
            "kalcb.research.min_adv20_krw": 1_000_000,
        }
    )
    calls = {"count": 0}
    original = kalcb_replay_cache.daily_selection_from_snapshot

    def wrapped(snapshot, config):
        calls["count"] += 1
        return original(snapshot, config)

    monkeypatch.setattr(kalcb_replay_cache, "daily_selection_from_snapshot", wrapped)

    snapshot = kalcb_replay_cache._load_or_build_snapshot(
        trade_date,
        {
            "005930": _research_rows(trade_date, start=5_000, drift=40, volume=1_000_000),
            "000660": _research_rows(trade_date, start=4_800, drift=35, volume=900_000),
            "035420": _research_rows(trade_date, start=4_500, drift=25, volume=850_000),
            "035720": _research_rows(trade_date, start=4_400, drift=15, volume=800_000),
        },
        cfg,
        source_fingerprint="unit-research-replay",
        candidate_config_hash="unit-config",
        requested_universe_count=4,
        data_available_symbols=["000660", "005930", "035420", "035720"],
        unavailable_symbols=[],
        sector_map={"005930": "semis", "000660": "semis", "035420": "internet", "035720": "internet"},
        store=KALCBArtifactStore(tmp_path / "snapshots"),
    )

    assert calls["count"] == 1
    assert len(snapshot.candidates) <= 4
    assert snapshot.metadata["active_symbol_count"] == 2
    assert snapshot.metadata["research_model_version"] == "kalcb-research-selection-v3-structural-campaign"
    assert all(candidate.metadata["source"] == "real_kis_krx_parquet" for candidate in snapshot.candidates)
    assert all(candidate.metadata["candidate_config_hash"] == "unit-config" for candidate in snapshot.candidates)
    assert all(candidate.metadata["sector_map_hash"] for candidate in snapshot.candidates)


def test_kalcb_replay_snapshot_research_excludes_trade_date_daily_rows(tmp_path):
    trade_date = make_kalcb_synthetic_bars("005930")[0].timestamp.date()
    rows = _research_rows(trade_date, start=5_000, drift=20, volume=1_000_000)
    rows.append(
        {
            "date": trade_date.isoformat(),
            "open": 50_000,
            "high": 55_000,
            "low": 49_000,
            "close": 54_000,
            "volume": 20_000_000,
        }
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 1,
            "kalcb.frontier.size": 1,
            "kalcb.frontier.rotation_slots": 0,
            "kalcb.research.top_long_count": 1,
            "kalcb.research.min_adv20_krw": 1_000_000,
        }
    )

    snapshot = kalcb_replay_cache._load_or_build_snapshot(
        trade_date,
        {"005930": rows},
        cfg,
        source_fingerprint="unit-full-replay-source",
        candidate_config_hash="unit-causal-config",
        requested_universe_count=1,
        data_available_symbols=["005930"],
        unavailable_symbols=[],
        sector_map={"005930": "semis"},
        store=KALCBArtifactStore(tmp_path / "causal_snapshots"),
    )

    candidate = snapshot.candidates[0]
    assert candidate.metadata["prior_day_date"] == (trade_date - timedelta(days=1)).isoformat()
    assert candidate.metadata["research_as_of_date"] == (trade_date - timedelta(days=1)).isoformat()
    assert candidate.metadata["research_lookahead_policy"] == "strict_prior_daily_rows_only"
    assert candidate.prior_day_close != 54_000
    assert snapshot.source_fingerprint == "unit-full-replay-source"
    assert snapshot.metadata["research_causal_source_fingerprint"]


def test_official_kalcb_requires_candidate_feature_bundle():
    with pytest.raises(ValueError, match="candidate_artifact"):
        run_kalcb_backtest({"capability_level": "official", "available_features": ["completed_5m_signal_bars"]}, {})


def _daily_candidate(
    symbol: str,
    trade_date,
    *,
    selection_score: float,
    prior_notional: float,
    campaign_setup_score: float,
    rs_20d: float,
    close_location_20d: float,
) -> KALCBDailyCandidate:
    return KALCBDailyCandidate(
        symbol=symbol,
        trade_date=trade_date,
        prior_day_high=100.0,
        prior_day_low=95.0,
        prior_day_close=99.0,
        daily_atr=2.0,
        expected_5m_volume=1000.0,
        average_30m_volume=6000.0,
        selection_score=selection_score,
        metadata={
            "frontier_score_components": {
                "prior_notional": prior_notional,
                "campaign_setup_score": campaign_setup_score,
                "rs_20d": rs_20d,
                "close_location_20d": close_location_20d,
                "range_pct": 0.04,
                "volume_ratio_20d": 1.2,
            }
        },
    )


def _research_rows(trade_date, *, start: float, drift: float, volume: float, days: int = 70) -> list[dict]:
    from datetime import timedelta

    first = trade_date - timedelta(days=days)
    rows = []
    for index in range(days):
        close = start + drift * index
        open_ = close - max(drift * 0.5, 1.0)
        rows.append(
            {
                "date": (first + timedelta(days=index)).isoformat(),
                "open": open_,
                "high": close * 1.01,
                "low": open_ * 0.99,
                "close": close,
                "volume": volume,
            }
        )
    return rows


def test_last_bar_signal_has_no_end_of_data_entry_fill():
    full = make_synthetic_replay_bundle("kalcb", {"symbol": "005930"})
    truncated = EventReplayBundle(
        events=tuple(full.events[:7]),
        source_fingerprint=full.source_fingerprint,
        data_root=full.data_root,
        metadata=dict(full.metadata),
    )

    result = run_kalcb_backtest({"capability_level": "synthetic"}, {"kalcb.entry.entry_score_blocklist": []}, replay_bundle=truncated)

    assert result.metrics["entry_count"] == 1
    assert result.metrics["total_trades"] == 0
    assert not result.replay_result.broker.fills


def test_replay_timer_uses_shared_core_and_next_bar_fill():
    trade_date = make_kalcb_synthetic_bars("005930")[0].timestamp.date()
    cfg = KALCBConfig.from_mapping({"kalcb.session.flatten_time": "15:20", "kalcb.carry.mode": "off"})
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        candidates=(_daily_candidate(
            "005930",
            trade_date,
            selection_score=10.0,
            prior_notional=10_000_000_000.0,
            campaign_setup_score=4.0,
            rs_20d=0.05,
            close_location_20d=0.8,
        ),),
        source_fingerprint="unit-timer",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
    )
    adapter = KALCBReplayAdapter(cfg, {trade_date: snapshot}, initial_equity=10_000_000.0, costs=BrokerCosts())
    broker = SimBroker(initial_equity=10_000_000.0, costs=BrokerCosts())
    broker.positions[("KALCB", "005930")] = SimPosition(
        strategy_id="KALCB",
        symbol="005930",
        qty=10,
        avg_price=100.0,
        entry_decision_time=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=9, minute=25),
        entry_fill_time=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=9, minute=30),
        route_metadata={"risk_per_share": 5.0, "entry_type": EntryType.FIRST30_OPEN.value},
        max_price=101.0,
        min_price=99.0,
    )
    symbol_state = adapter.state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=9, minute=30),
        initial_stop=95.0,
        current_stop=95.0,
        risk_per_share=5.0,
        entry_type=EntryType.FIRST30_OPEN.value,
        momentum_score=3,
    )
    submit_bar = MarketBar("005930", datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=15, minute=15), "5m", 101.0, 101.2, 100.0, 100.8, 1000)
    symbol_state.add_bar(submit_bar)

    decisions = adapter.on_timestamp_end(submit_bar.timestamp, (submit_bar,), broker)
    fill_bar = MarketBar("005930", datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=15, minute=20), "5m", 100.7, 100.8, 100.0, 100.2, 1000)
    fills = broker.process_bar(fill_bar)

    assert decisions[0].decision_code == "timer_exit"
    assert decisions[0].timestamp.time() == cfg.flatten_time
    assert broker.orders == []
    assert fills
    assert fills[0].timestamp == fill_bar.timestamp
    assert broker.trades[0].exit_reason == "eod_flatten"


def test_kalcb_backtest_cli_dry_run_reports_core_version(capsys):
    assert backtest_cli_main(["run", "--strategy", "kalcb", "--config", "config/backtests/kalcb.yaml", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy_core_version"] == KALCB_CORE_VERSION
    assert payload["live_parity_fill_timing"] == "next_5m_open"
    assert payload["auction_mode"] == "non_auction_continuous"


def test_kalcb_optimizer_cli_dry_run_reports_phase_count(capsys, tmp_path):
    assert auto_cli_main([
        "optimize",
        "--strategy",
        "kalcb",
        "--config",
        "config/backtests/kalcb.yaml",
        "--output-root",
        str(tmp_path),
        "--num-phases",
        "3",
        "--dry-run",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy_core_version"] == KALCB_CORE_VERSION
    assert payload["num_phases"] == 3


def test_kalcb_research_sweep_cli_dry_run_reports_holdout_contract(capsys, tmp_path):
    assert auto_cli_main([
        "research-sweep",
        "--strategy",
        "kalcb",
        "--config",
        "config/backtests/kalcb.yaml",
        "--output-root",
        str(tmp_path),
        "--holdout-days",
        "42",
        "--max-candidates",
        "3",
        "--dry-run",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sweep_type"] == "research_candidate_opportunity_training_only"
    assert payload["holdout_days"] == 42
    assert payload["coarse_candidate_count"] == 4
    assert payload["candidate_count_estimate"] > payload["coarse_candidate_count"]


def test_kalcb_research_sweep_runs_synthetic_and_writes_seed(tmp_path):
    payload = run_research_sweep(
        {"capability_level": "synthetic"},
        output_dir=tmp_path,
        holdout_days=42,
        fold_days=42,
        max_candidates=2,
        refine_top_n=0,
        top_n=1,
    )

    assert payload["holdout_contract"]["selection_uses_holdout"] is False
    assert payload["sweep_type"] == "research_candidate_opportunity_training_only"
    assert payload["candidate_count"] == 3
    assert payload["refinement_candidate_count"] == 0
    assert payload["rows"][0]["metrics"]["valid_candidate_days"] > 0
    assert payload["rows"][0]["metrics"]["raw_signal_count"] > 0
    assert payload["phase_auto_seed"]["initial_mutations"]
    assert (tmp_path / Path(payload["artifact_paths"]["phase_auto_seed"]).name).exists()


def test_kalcb_research_sweep_candidates_are_research_only():
    for candidate in build_research_sweep_candidates():
        assert all(key.startswith("kalcb.research.") or key == "kalcb.frontier.active_selection_mode" for key in candidate.mutations)


def test_kalcb_research_refinement_expands_top_seed_smartly():
    candidates = build_research_refinement_candidates(
        [
            {
                "name": "sector_participation67",
                "score": 44.0,
                "mutations": {"kalcb.research.min_sector_participation": 0.67},
                "merged_mutations": {"kalcb.frontier.active_selection_mode": "liquidity"},
            }
        ],
        max_candidates=12,
    )

    assert len(candidates) == 12
    assert all(key.startswith("kalcb.research.") or key == "kalcb.frontier.active_selection_mode" for item in candidates for key in item.mutations)
    assert any(item.mutations.get("kalcb.frontier.active_selection_mode") == "score" for item in candidates)
    assert any(item.mutations.get("kalcb.research.top_long_count") for item in candidates)


def test_kalcb_research_score_uses_candidate_opportunity_not_trade_pnl():
    score, reject = score_research_metrics(
        {
            "snapshot_count": 10,
            "valid_candidate_days": 100,
            "active_valid_candidate_days": 40,
            "raw_signal_count": 35,
            "active_raw_signal_count": 18,
            "raw_signal_rate": 0.35,
            "active_raw_signal_rate": 0.45,
            "mfe_ge_0_5_per_valid": 0.20,
            "mfe_ge_1_0_per_valid": 0.08,
            "active_mfe_ge_0_5_per_valid": 0.25,
            "active_mfe_ge_1_0_per_valid": 0.10,
            "avg_signal_mfe_r": 0.8,
            "active_avg_signal_mfe_r": 0.9,
            "active_median_signal_mfe_r": 0.55,
            "active_days_with_good_signal_share": 0.70,
            "active_good_signal_capture_share": 0.5,
            "avg_good_signals_per_day": 2.0,
            "active_low_mfe_lt_0_3_signal_share": 0.10,
            "active_bad_mae_le_neg_1_0_signal_share": 0.05,
            "bad_mae_le_neg_1_0_signal_share": 0.05,
        }
    )

    assert reject == ""
    assert score > 0


def test_kalcb_plugin_uses_live_parity_execution_context(tmp_path):
    plugin = KALCBOptimizationPlugin({"capability_level": "synthetic"}, output_dir=tmp_path)
    assert plugin.execution_context["shared_decision_core"] == "live_shared_core"
    assert plugin.execution_context["strategy_core_version"] == KALCB_CORE_VERSION
    assert plugin.execution_context["live_parity_fill_timing"] == "next_5m_open"
    assert plugin.execution_context["auction_mode"] == "non_auction_continuous"


def test_kalcb_plugin_cache_keys_change_with_execution_contract_identity(tmp_path):
    baseline = KALCBOptimizationPlugin({"capability_level": "synthetic"}, output_dir=tmp_path / "baseline")
    cost_variant = KALCBOptimizationPlugin(
        {"capability_level": "synthetic", "slippage_bps": 9.0},
        output_dir=tmp_path / "cost",
    )
    equity_variant = KALCBOptimizationPlugin(
        {"capability_level": "synthetic", "initial_equity": 2_000_000.0},
        output_dir=tmp_path / "equity",
    )

    assert baseline.execution_context["raw_metric_cache_key"] != cost_variant.execution_context["raw_metric_cache_key"]
    assert baseline.execution_context["phase_score_cache_key"] != cost_variant.execution_context["phase_score_cache_key"]
    assert baseline.execution_context["raw_metric_cache_key"] != equity_variant.execution_context["raw_metric_cache_key"]
