from dataclasses import replace
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pytest

from backtests.auto.shared.cache_keys import stable_signature
from backtests.engine.qualification_evidence import _replay_scope, _verify_feature_fallback
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_common.source_quality import POLICY, missing_prefix


DAY = date(2026, 1, 5)


def bars(symbol="005930", count=76):
    return [MarketBar(symbol, datetime.combine(DAY, time(9), KST) + timedelta(minutes=5*i),
        "5m", 100, 101, 99, 100, 1000) for i in range(count)]


def test_prefix_uses_only_completed_intervals_and_known_session_schedule():
    stamps = [bar.timestamp for bar in bars(count=3)]
    assert not missing_prefix("005930", DAY, stamps[:1], stamps[1])
    assert missing_prefix("005930", DAY, stamps[:1], stamps[2]) == [stamps[1]]
    # A future observation cannot repair an earlier missing interval.
    assert missing_prefix("005930", DAY, [stamps[0], stamps[2]], stamps[2]) == [stamps[1]]
    suspended = date(2025, 5, 22)
    assert not missing_prefix("207940", suspended, (), datetime.combine(suspended, time(9, 30), KST))


def test_olr_incomplete_peer_cannot_change_accepted_candidate_features():
    from strategy_olr.config import OLRConfig
    from strategy_olr.models import OLRDailySnapshot
    from strategy_olr.research import build_afternoon_contexts
    from tests.strategy_olr.test_research import _candidate
    candidate = _candidate("005930", DAY, prior_close=100, rank=1, sector="SEMICONDUCTOR")
    snapshot = OLRDailySnapshot(DAY, (candidate,), "source", datetime.combine(DAY, time(14, 30), KST))
    good, sparse = bars(count=66), bars("000660", count=66)[1:]
    config = OLRConfig(require_complete_feature_prefix=True)
    sectors = {"005930": candidate.sector, "000660": candidate.sector}
    source = {(DAY, "005930"): good, (DAY, "000660"): sparse}
    first = build_afternoon_contexts(snapshot, source, config, sectors)["005930"]
    source[(DAY, "000660")] = [replace(bar, close=200, high=210, volume=10_000_000) for bar in sparse]
    second = build_afternoon_contexts(snapshot, source, config, sectors)["005930"]
    assert first == second
    assert not build_afternoon_contexts(snapshot, {(DAY, "005930"): good[1:]}, config, sectors)


def test_kalcb_gap_blocks_entries_cancels_flat_pending_and_keeps_management():
    from strategy_common.actions import CancelOrders
    from strategy_kalcb.config import KALCBConfig
    from strategy_kalcb.core.logic import step_kalcb_core
    from strategy_kalcb.core.state import KALCBState, KALCBPositionState, SymbolStage
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from tests.strategy_kalcb.test_core_logic import _snapshot
    state = KALCBState()
    config = KALCBConfig(entry_require_complete_prefix=True)
    snapshot, portfolio = _snapshot(DAY), KALCBPortfolioView(cash=1_000_000)
    source = bars(count=8)
    step_kalcb_core(state, source[0], config, snapshot, portfolio)
    symbol = state.symbol_state("005930")
    symbol.pending_entry_order_id = "pending"
    symbol.stage = SymbolStage.ENTRY_QUEUED
    result = step_kalcb_core(state, source[2], config, snapshot, portfolio)
    assert isinstance(result.actions[0], CancelOrders)
    assert not symbol.pending_entry_order_id
    assert symbol.rejected_reason == "incomplete_observed_prefix"
    symbol.position = KALCBPositionState("005930", 1, 1, 100, source[0].timestamp, 90, 90, 10, "entry", 5)
    symbol.stage = SymbolStage.IN_POSITION
    count = symbol.position.hold_bars
    step_kalcb_core(state, source[3], config, snapshot, portfolio)
    assert symbol.position.hold_bars > count


def scope_fixture():
    rows = bars()
    rows = [bar for bar in rows if bar.timestamp.time().replace(tzinfo=None) != time(10, 40)]
    snapshots = [{"trade_date": DAY.isoformat(), "candidates": [{"symbol": "005930"}]}]
    broker = SimpleNamespace(qualification_snapshots=snapshots, qualification_snapshot_scopes={"KALCB": snapshots}, journal=[])
    identity = {"qualification_snapshot_hash": stable_signature(snapshots), "configuration": {
        "entry_require_complete_prefix": True, "source_dependency_scope": {"version": "causal-feature-and-management-v2",
            "strategy": "kalcb", "candidate_intraday_before": "15:20"}}}
    retained = {("005930", DAY, "5m"): [bar.timestamp for bar in rows]}
    return rows, broker, identity, retained


def test_unused_kalcb_gap_is_allowed_but_dropped_events_or_late_entries_fail():
    rows, broker, identity, retained = scope_fixture()
    run = lambda events: _replay_scope(broker, identity, events, set(retained), source_stamps=retained)
    events = [bar.to_json_dict() for bar in rows]
    run(events)
    with pytest.raises(ValueError, match="candidate_observations_dropped"):
        run(events[:-1])
    broker.journal = [{"event": "submit", "order": {"symbol": "005930", "side": "BUY", "order_id": "late",
        "submitted_at": "2026-01-05T11:00:00+09:00", "metadata": {"decision_available_at": "2026-01-05T11:05:00+09:00"}}}]
    with pytest.raises(ValueError, match="entry_after_missing_prefix"):
        run(events)


def test_gap_during_working_order_remains_unverified():
    rows, broker, identity, retained = scope_fixture()
    broker.journal = [{"event": "submit", "order": {"symbol": "005930", "side": "BUY", "order_id": "early",
        "submitted_at": "2026-01-05T10:30:00+09:00", "metadata": {"decision_available_at": "2026-01-05T10:35:00+09:00"}}}]
    with pytest.raises(ValueError, match="management_feed_incomplete"):
        _replay_scope(broker, identity, [bar.to_json_dict() for bar in rows], set(retained), source_stamps=retained)


def test_olr_fallback_receipt_is_checked_against_independent_source():
    rows = bars(count=66)[1:]
    grouped = {("005930", DAY, "5m"): [bar.timestamp for bar in rows]}
    snapshot = {"trade_date": DAY.isoformat(), "candidates": [], "metadata": {
        "feature_prefix_policy": POLICY, "feature_prefix_exclusions": ["005930"]}}
    broker = SimpleNamespace(qualification_snapshots=[snapshot])
    identity = {"configuration": {"require_complete_feature_prefix": True}}
    _verify_feature_fallback(broker, identity, grouped, set(grouped))
    snapshot["candidates"] = [{"symbol": "005930"}]
    with pytest.raises(ValueError, match="feature_fallback_evidence_mismatch"):
        _verify_feature_fallback(broker, identity, grouped, set(grouped))


def test_olr_fallback_does_not_waive_calendar_contradictions(monkeypatch):
    import backtests.engine.qualification_evidence as evidence
    monkeypatch.setattr(evidence, "session_bar_coverage", lambda *a, **k: {"observations_during_halt": ["observed"]})
    broker = SimpleNamespace(qualification_snapshots=[])
    with pytest.raises(ValueError, match="calendar_contradiction"):
        _verify_feature_fallback(broker, {"configuration": {"require_complete_feature_prefix": True}},
            {}, {("005930", DAY, "5m")})


def test_overlay_recipe_records_kalcb_evaluator_normalization():
    from backtests.strategies.common.corrective_selection import CorrectiveSelectionPlugin
    plugin = object.__new__(CorrectiveSelectionPlugin)
    plugin.name, plugin.config = "kalcb", {}
    row = plugin._overlay_recipe({"id": "original", "mutations": {}}, {"kalcb.source.require_complete_prefix": True})
    assert row["effective_config"]["config"]["fast_replay_suppress_rejections"] is True
    assert row["effective_config"]["config"]["entry_require_complete_prefix"] is True


def test_study_removes_count_floors_and_real_sizing_control_is_registered(monkeypatch, tmp_path):
    from backtests.strategies.common import corrective_selection as study
    from strategy_kalcb.config import KALCBConfig
    monkeypatch.setattr(study, "recover_recipes", lambda *a: {})
    protocol = study.register_study(tmp_path, tmp_path / "study.json")
    assert "min_total_trades" not in protocol["limits"]["olr"]
    assert "min_trades" not in protocol["limits"]["kalcb"]
    assert protocol["limits"]["kalcb"]["max_dd_pct"] == .08
    configs = [KALCBConfig.from_mapping({}, policy) for policy in protocol["kalcb_repair_policies"]]
    assert [cfg.max_participation_30m for cfg in configs] == [.01, .01, .005, .0025]
    assert [cfg.flatten_time for cfg in configs] == [time(15), time(14, 45), time(14, 45), time(14, 45)]


def test_olr_prepared_cache_preserves_current_recipe_config_and_rejects_changed_bytes(monkeypatch, tmp_path):
    from dataclasses import fields
    from kis_core.input_manifest import InputManifest
    from backtests.strategies.olr import replay_cache as cache
    from backtests.strategies.olr.research_sweep import OLRResearchSweepDataset
    from strategy_olr.config import OLRConfig
    generation = InputManifest.create([], source_root=tmp_path, destination=tmp_path / "generation")
    calls = []
    def prepare(config, **kwargs):
        calls.append(config)
        values = dict.fromkeys(field.name for field in fields(OLRResearchSweepDataset))
        values.update(config=dict(config), source_fingerprint="same-source")
        return OLRResearchSweepDataset(**values)
    monkeypatch.setattr(cache, "prepare_research_sweep_dataset", prepare)
    config = {"data_root": str(tmp_path / "intraday"), "daily_data_root": str(tmp_path / "daily"),
        "corrective_study_hash": "study", "initial_mutations": {"recipe": "first"}}
    cache._DATASET_CACHE.clear()
    with generation.activate():
        first = cache._load_dataset(config, OLRConfig())
        cache._DATASET_CACHE.clear()
        changed_config = {**config, "initial_mutations": {"recipe": "second"}}
        second = cache._load_dataset(changed_config, OLRConfig())
        assert len(calls) == 1
        assert first.source_fingerprint == second.source_fingerprint
        assert second.config == changed_config
        disk = next((generation.path.parent / "derived_artifacts" / "olr_datasets").glob("*.pkl"))
        disk.write_bytes(disk.read_bytes() + b"changed")
        cache._DATASET_CACHE.clear()
        with pytest.raises(ValueError, match="dataset bytes changed"):
            cache._load_dataset(config, OLRConfig())
