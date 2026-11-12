import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, time
from types import SimpleNamespace

import pytest

from backtests.analysis.metrics import session_equity
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.phase_state import PhaseState
from backtests.auto.shared.round_manager import RoundManager
from backtests.strategies.olr.final_diagnostics import export_case, require_bound_result
from backtests.strategies.olr.phase_scoring import gate_criteria, score_olr_phase
from backtests.strategies.olr.plugin import OLROptimizationPlugin
from backtests.strategies.olr.runner import compile_olr_replay_bundle, run_olr_backtest, snapshots_from_bundle
from backtests.strategies.olr import research_sweep
from strategy_common.actions import action_to_json_dict
from strategy_common.clock import KST
from strategy_olr.config import OLRConfig
from strategy_olr.core.core_models import OLRPortfolioView
from strategy_olr.core.state import OLRState
from strategy_olr.core.logic import _maybe_submit_entry
from strategy_olr.engine import OLROMSLiveAdapter


def stamp(day):
    return datetime(2026, 1, day, 15, 30, tzinfo=KST)


def test_sizing_cap_reuses_candidates_but_changes_evaluation_identity():
    from backtests.strategies.olr.replay_cache import _candidate_config_hash, _stage1_config_hash
    from backtests.engine.evaluation_identity import request_identity

    config = {"initial_equity": 10000000}
    capped = {"olr.allocation.equity_cap_krw": config["initial_equity"]}
    baseline_cfg = OLRConfig.from_mapping(config, {})
    capped_cfg = OLRConfig.from_mapping(config, capped)
    assert _stage1_config_hash(baseline_cfg, {}) == _stage1_config_hash(capped_cfg, capped)
    assert _candidate_config_hash(baseline_cfg, {}) == _candidate_config_hash(capped_cfg, capped)
    assert request_identity(config, {}) != request_identity(config, capped)
    changed_selection = replace(baseline_cfg, afternoon_top_n=baseline_cfg.afternoon_top_n + 1)
    assert _candidate_config_hash(changed_selection, {}) != _candidate_config_hash(baseline_cfg, {})


def test_declared_window_pads_only_known_cash_sessions():
    values, coverage = session_equity([100, 110], [stamp(6)], [{"cash_only": True}],
        declared_window={"start": "2026-01-05", "end": "2026-01-08"}, initial_cash_only=True)
    assert values == [100, 100, 110, 110, 110]
    assert coverage["status"] == "complete" and coverage["cash_only_padded_sessions"] == 3
    values, coverage = session_equity([100, 110], [stamp(6)], [{"closing_marks_complete": True}],
        declared_window={"start": "2026-01-05", "end": "2026-01-08"}, initial_cash_only=True)
    assert values == [] and coverage["missing_sessions"] == ["2026-01-07", "2026-01-08"]
    values, coverage = session_equity([100], [], [],
        declared_window={"start": "2026-01-05", "end": "2026-01-06"}, initial_cash_only=True)
    assert values == [100, 100, 100] and coverage["status"] == "complete"
    with pytest.raises(ValueError, match="outside"):
        session_equity([100, 110], [stamp(8)], [{"cash_only": True}],
            declared_window={"start": "2026-01-05", "end": "2026-01-06"}, initial_cash_only=True)


def test_explicit_split_is_preserved_independently_of_cached_available_window(monkeypatch, tmp_path):
    cached = research_sweep.OLRReplayWindow(date(2025, 5, 12), date(2026, 7, 10),
        date(2025, 5, 12), date(2026, 5, 28), date(2026, 5, 29))
    class Cache(dict):
        def get(self, key):
            return cached
    monkeypatch.setattr(research_sweep, "_WINDOW_CACHE", Cache())
    config = {"date_range": {"start": "2025-05-12", "end": "2026-03-31"}, "holdout_start": "2026-04-01"}
    training = research_sweep._training_config(config, 42)
    assert training["date_range"] == config["date_range"]
    first = research_sweep._resolve_replay_window(training, tmp_path, "5m", [])
    assert first.train_end == date(2026, 3, 31) and first.holdout_start == date(2026, 4, 1)
    second = research_sweep._resolve_replay_window({**training, "holdout_start": "2026-03-01"}, tmp_path, "5m", [])
    assert second.train_end == date(2026, 2, 28)


def test_future_labels_cannot_change_entry_actions_or_nonfill_keys():
    bundle = compile_olr_replay_bundle()
    snapshot = next(iter(snapshots_from_bundle(bundle).values()))
    candidate = snapshot.candidates[0]
    from tests.strategy_olr.test_core_runner import _bar
    bar = _bar(candidate.symbol, snapshot.trade_date, time(14, 30), 100, 100, 100, 100)
    portfolio = OLRPortfolioView(cash=100000, equity=100000)
    cfg = OLRConfig(auction_nonfill_rate=.5)
    outputs = []
    for outcome in (-.99, 99):
        altered = replace(candidate, metadata={**candidate.metadata,
            "close_to_close_label_pct": outcome, "next_session_mfe_label_pct": outcome})
        altered_snapshot = replace(snapshot, candidates=(altered,))
        result = _maybe_submit_entry(OLRState(), bar, cfg, altered_snapshot, altered, portfolio)
        assert result.actions
        outputs.append([action_to_json_dict(a) for a in result.actions])
    assert outputs[0] == outputs[1]


def test_fixed_capital_changes_sizing_only_above_cap():
    from tests.strategy_olr.test_core_runner import _bar
    snapshot = next(iter(snapshots_from_bundle(compile_olr_replay_bundle()).values()))
    candidate = snapshot.candidates[0]
    bar = _bar(candidate.symbol, snapshot.trade_date, time(14, 30), 100, 100, 100, 100)
    def quantity(equity, cap):
        result = _maybe_submit_entry(OLRState(), bar, OLRConfig(allocation_equity_cap_krw=cap),
            snapshot, candidate, OLRPortfolioView(cash=equity, equity=equity))
        return result.actions[0].qty
    assert quantity(200000, 100000) == quantity(100000, 0)
    assert quantity(50000, 100000) == quantity(50000, 0)
    assert quantity(200000, 0) > quantity(200000, 100000)


def test_future_source_revisions_cannot_change_auction_nonfill_scenarios():
    from tests.strategy_olr.test_core_runner import _bar
    snapshot = next(iter(snapshots_from_bundle(compile_olr_replay_bundle()).values()))
    candidate = snapshot.candidates[0]
    bar = _bar(candidate.symbol, snapshot.trade_date, time(14, 30), 100, 100, 100, 100)
    keys = []
    for fingerprint in ("original-data", "revised-future-data"):
        result = _maybe_submit_entry(OLRState(), bar, OLRConfig(auction_nonfill_rate=.5),
            replace(snapshot, source_fingerprint=fingerprint), candidate,
            OLRPortfolioView(cash=100000, equity=100000))
        keys.append(result.actions[0].metadata["auction_nonfill_key"])
    assert keys[0] == keys[1]


@pytest.mark.parametrize("current", [None, SimpleNamespace(equity=0, buyable_cash=0)])
def test_live_missing_or_zero_account_does_not_reuse_old_buying_power(current):
    class OMS:
        async def get_account_state(self): return current
        async def get_all_positions(self): return {}
    adapter = OLROMSLiveAdapter(OMS(), last_account_state=SimpleNamespace(equity=100000, buyable_cash=100000))
    portfolio = asyncio.run(adapter.refresh_portfolio())
    assert portfolio.cash == portfolio.equity == 0


def test_labels_and_gross_mfe_cannot_improve_score_or_promotion_gates():
    base = {"official_mtm_net_return_pct": .5, "expected_total_r": 30, "total_trades": 200,
            "profit_factor": 2, "max_drawdown_pct": .1}
    altered = {**base, "olr_alpha_capture": 999, "olr_discrimination_quality": 999,
               "selected_negative_label_share": -999}
    assert score_olr_phase(1, base) == score_olr_phase(1, altered)
    assert gate_criteria(1, base) == gate_criteria(1, altered)


def test_full_diagnostics_are_observer_only_and_failed_receipts_prevent_publication(tmp_path, monkeypatch):
    config = {"capability_level": "synthetic"}
    plugin = OLROptimizationPlugin(config, max_workers=1)
    mutations = plugin.initial_mutations
    result = plugin._evaluate_result(mutations)
    before = {key: result.metrics[key] for key in ("decision_hash", "fill_hash", "trade_hash", "final_state_hash")}
    config = plugin.config
    # The small synthetic fixture intentionally lacks source qualification.
    # Isolate observer purity here; exercise the real fail-closed gate below.
    with monkeypatch.context() as patch:
        patch.setattr("backtests.strategies.olr.final_diagnostics.require_bound_result", lambda *args: None)
        export_case(result, config, mutations, tmp_path / "export")
    assert before == {key: result.metrics[key] for key in before}
    repeat = run_olr_backtest(config, mutations)
    assert before == {key: repeat.metrics[key] for key in before}
    result.metrics["audit_pass"] = False
    with pytest.raises(ValueError, match="request-bound"):
        require_bound_result(result, config, mutations)
    # Exercise the actual shared final-round lifecycle, not only a helper.
    plugin._evaluate_result = lambda _: result
    manager = RoundManager("stock", "olr", base_dir=tmp_path / "output")
    directory = manager.get_round_dir(1)
    runner = PhaseRunner(plugin, directory, round_manager=manager, round_num=1)
    with pytest.raises(ValueError, match="request-bound"):
        runner.run_end_of_round(PhaseState(cumulative_mutations=mutations))
    assert not (directory / "optimized_config.json").exists()
    assert not manager.manifest_path.exists()


def test_diagnostic_memory_release_preserves_active_bundle_and_execution_contract():
    import weakref
    from backtests.strategies.olr import replay_cache, research_sweep
    from kis_core import input_manifest
    from backtests.strategies.olr.final_diagnostics import release_diagnostic_replay_memory
    from backtests.strategies.common.plugin_base import build_execution_contract
    plugin = OLROptimizationPlugin({"capability_level": "synthetic"}, max_workers=1)
    baseline = plugin._last_result
    metrics = dict(baseline.metrics)
    contract = build_execution_contract(plugin, metrics)
    reference = weakref.ref(baseline.replay_result.broker)
    active_bundle = compile_olr_replay_bundle()
    original_events = active_bundle.events
    replay_cache._BUNDLE_CACHE['diagnostic-test'] = active_bundle
    research_sweep._FRAME_CACHE['diagnostic-test'] = object()
    input_manifest._VERIFIED_BYTES['diagnostic-test'] = b'cached source bytes'
    del baseline
    release_diagnostic_replay_memory(plugin)
    assert reference() is None
    assert active_bundle.events is original_events
    assert build_execution_contract(plugin, metrics) == contract
    assert not replay_cache._BUNDLE_CACHE
    assert not research_sweep._FRAME_CACHE
    assert not input_manifest._VERIFIED_BYTES


def test_final_report_preserves_selected_recipe_and_qualification_status(tmp_path, monkeypatch):
    import json
    from backtests.strategies.olr.final_diagnostics import write_full_diagnostics

    plugin = OLROptimizationPlugin({
        "capability_level": "synthetic", "selected_candidate": "low325",
        "promotion_status": "development_qualified_independent_validation_pending",
        "development_qualification": {"independent_oos_pass": False, "live_promotion": False},
    }, max_workers=1)
    # Fixture has no qualified source bars; production receipt rejection is
    # exercised by the publication test above. Here isolate report semantics.
    monkeypatch.setattr("backtests.strategies.olr.final_diagnostics.require_bound_result", lambda *args: None)
    state = PhaseState(cumulative_mutations=plugin.initial_mutations)
    payload = write_full_diagnostics(plugin, state, tmp_path, round_num=2, round_name="Low-325")
    saved = json.loads((tmp_path / "optimized_results.json").read_text())["selected"]
    assert saved == payload
    assert saved["selected_candidate"] == "low325"
    assert saved["promotion_status"] == plugin.config["promotion_status"]
    assert saved["development_qualification"]["independent_oos_pass"] is False
    assert saved["mutations"] == state.cumulative_mutations
