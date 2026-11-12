from datetime import date, datetime, time
from types import SimpleNamespace

from backtests.engine.integrity import audit_broker
from backtests.engine.sim_broker import SimBroker, BrokerCosts
from backtests.auto.shared.phase_runner import _metric_contract_criteria
from backtests.engine.qualification_evidence import _covered, _management_cutoffs
from strategy_common.clock import KST
from strategy_common.actions import SubmitEntry
from strategy_common.market import MarketBar


def test_real_buying_power_rejection_is_audited_and_tampering_fails():
    at = datetime(2026, 1, 5, 9, 5, tzinfo=KST)
    broker = SimBroker(100, BrokerCosts(0, 0, 0))
    broker.submit(SubmitEntry("KALCB", "005930", 2, "MARKET", None, None, "entry"), at.replace(minute=0))
    bar = MarketBar("005930", at, "5m", 100, 100, 100, 100, 1000)
    broker.expected_batches = [(bar.to_json_dict(),)]
    broker.process_batch([bar])
    result = audit_broker(broker)
    assert result["execution_validity_pass"], result["audit_receipt"]["errors"]
    assert result["invalid_order_rejection_count"] == 0
    assert result["rejected_order_reason_counts"] == {"buying_power": 1}
    rejection = next(r for r in broker.journal if r["event"] == "reject")
    rejection["cash"] = 10000
    assert audit_broker(broker)["invalid_order_rejection_count"] == 1


def test_shared_gate_uses_invalid_rejections_but_preserves_legacy_fail_closed():
    common = {"official_metric_basis": "mtm", "rejected_order_count": 2}
    criteria = {r.name: r.passed for r in _metric_contract_criteria(common)}
    assert not criteria["hard_rejected_order_count"]
    criteria = {r.name: r.passed for r in _metric_contract_criteria({**common, "invalid_order_rejection_count": 0})}
    assert criteria["hard_invalid_order_rejection_count"]


def test_sparse_acceptance_does_not_hide_dropped_replay_events():
    day = date(2026, 1, 5)
    stamps = [datetime.combine(day, t, KST) for t in (time(9), time(9, 10), time(15, 30))]
    proof = {("005930", day): frozenset(stamps)}
    assert _covered(stamps, day, 5, "005930", proof)
    assert not _covered(stamps[:-1], day, 5, "005930", proof)


def test_management_scope_stops_after_actual_exit_but_covers_overnight_positions():
    order = {"order_id": "1", "symbol": "005930", "strategy_id": "KALCB", "qty": 1,
             "submitted_at": "2026-01-05T10:00:00+09:00"}
    fill = {"symbol": "005930", "strategy_id": "KALCB", "qty": 1, "side": "BUY", "timestamp": "2026-01-05T10:05:00+09:00"}
    broker = SimpleNamespace(journal=[{"event": "submit", "order": order}, {"event": "fill", "order": order, "fill": fill}])
    days = {date(2026, 1, 5), date(2026, 1, 6)}
    assert _management_cutoffs(broker, days)[("005930", date(2026, 1, 5))] == "16:00"
    broker.journal.append({"event": "fill", "order": {**order, "order_id": "2"},
                           "fill": {**fill, "side": "SELL", "timestamp": "2026-01-05T11:00:00+09:00"}})
    scope = _management_cutoffs(broker, days)
    assert scope == {("005930", date(2026, 1, 5)): "11:05"}


def test_inventory_at_intraday_deadline_is_distinct_from_terminal_flatness():
    from strategy_common.actions import SubmitExit
    at = datetime(2026, 1, 5, 15, 10, tzinfo=KST)
    broker = SimBroker(1000, BrokerCosts(0, 0, 0))
    broker.submit(SubmitEntry("KALCB", "005930", 2, "MARKET", None, None, "entry"), at.replace(hour=14))
    bars = [MarketBar("005930", at, "5m", 100, 100, 100, 100, 1000),
            MarketBar("005930", at.replace(minute=15), "5m", 100, 100, 100, 100, 1000),
            MarketBar("005930", at.replace(day=6, hour=9, minute=5), "5m", 100, 100, 100, 100, 1000)]
    broker.expected_batches = [(bar.to_json_dict(),) for bar in bars]
    broker.process_batch([bars[0]])
    broker.process_batch([bars[1]])
    broker.submit(SubmitExit("KALCB", "005930", 2, "MARKET", None, "late_exit"), at.replace(minute=20))
    broker.process_batch([bars[2]])
    metrics = audit_broker(broker)
    assert not broker.positions
    assert metrics["intraday_residual_session_count"] == 1
    assert metrics["audit_receipt"]["continuous_session_end_inventory"]["2026-01-05"] == {"KALCB:005930": 2}


def test_prefix_coverage_does_not_require_later_unused_intervals():
    from datetime import timedelta
    day = date(2026, 1, 5)
    stamps = [datetime.combine(day, time(9), KST) + timedelta(minutes=5*i) for i in range(6)]
    assert _covered(stamps, day, 5, "005930", {}, before="09:30")
    assert not _covered(stamps[1:], day, 5, "005930", {}, before="09:30")
    assert not _covered(stamps, day, 5, "005930", {})


def test_documented_individual_halt_explains_opening_absence_only_for_that_issue():
    day = date(2025, 5, 22)
    assert _covered([], day, 5, "207940", {}, before="09:30")
    assert not _covered([], day, 5, "005930", {}, before="09:30")
    stamp = datetime.combine(day, time(9, 20), KST)
    assert not _covered([stamp], day, 5, "207940", {}, before="09:30")


def test_finalist_failure_selects_only_frozen_qualified_recipe_and_restores_it(tmp_path):
    from backtests.strategies.common.corrective_selection import CorrectiveSelectionPlugin
    plugin = object.__new__(CorrectiveSelectionPlugin)
    plugin.name, plugin.output, plugin.baseline = "olr", tmp_path, "base"
    plugin.study = {"study_hash": "study", "scope": "registered", "strategies": {"olr": {"unrecoverable_lineage": []}}}
    plugin.generation = SimpleNamespace(identity="manifest")
    plugin.rows = {key: {"id": key, "mutations": {"complete_recipe": key}} for key in ("best", "second", "base")}
    plugin.metrics = {"best": {"score": 3}, "second": {"score": 2}, "base": {"score": 1}}
    plugin.config, plugin.execution_context = {}, {}
    plugin._score = lambda m: (m["score"], None)
    plugin._qualify = lambda finalists: {key: {"qualified": key == "second"} for key in finalists}
    state = SimpleNamespace(cumulative_mutations={"_corrective_recipe": "best"})
    plugin.build_end_of_round_artifacts(state)
    assert state.cumulative_mutations == {"complete_recipe": "second"}
    assert plugin.final_report["selected_recipe"] == "second"
    assert plugin.final_report["process_status"] == "completed"


def test_no_eligible_candidate_is_not_promoted_as_baseline(tmp_path):
    from backtests.strategies.common.corrective_selection import CorrectiveSelectionPlugin
    plugin = object.__new__(CorrectiveSelectionPlugin)
    plugin.name, plugin.output, plugin.baseline = "olr", tmp_path, "base"
    plugin.study = {"study_hash": "study", "scope": "registered", "strategies": {"olr": {"unrecoverable_lineage": []}}}
    plugin.generation = SimpleNamespace(identity="manifest")
    plugin.rows = {"base": {"id": "base", "mutations": {"recipe": "baseline"}}}
    plugin.metrics = {"base": {}}
    plugin.config, plugin.execution_context = {}, {}
    plugin._score = lambda m: (None, "failed")
    plugin._qualify = lambda finalists: {} if not finalists else (_ for _ in ()).throw(AssertionError())
    state = SimpleNamespace(cumulative_mutations={"_corrective_recipe": "base"})
    plugin.build_end_of_round_artifacts(state)
    assert plugin.final_report["selected_recipe"] is None
    assert plugin.config["promotion_status"] == "no_qualifying_candidate"
    assert not plugin.final_report["historical_qualification"]


def test_shared_source_failure_rejects_without_fabricating_performance(tmp_path):
    from backtests.strategies.common.corrective_selection import CorrectiveSelectionPlugin
    plugin = object.__new__(CorrectiveSelectionPlugin)
    plugin.name, plugin.output, plugin.baseline = "olr", tmp_path, "base"
    plugin.study = {"study_hash": "study"}
    plugin.config = {"source_dependency_scope": {"strategy": "olr"}}
    plugin.generation = SimpleNamespace(identity="manifest")
    plugin.metrics = {"base": {"audit_receipt": {"receipt_hash": "control", "coverage": {
        "input_provenance": "pass", "source_completeness": "unverified"},
        "qualification": {"source_completeness": ["005930:2025-05-12:incomplete"]}}}}
    plugin._case = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Redundant replay"))
    result = plugin._evaluate({"id": "alternative", "mutations": {}})
    assert not result["replay_performed"]
    assert "official_mtm_net_return_pct" not in result
    assert plugin._score(result) == (None, "shared_feature_inputs_unverified")


def test_olr_frequency_score_and_gate_use_trades_not_fill_fragments():
    from backtests.strategies.olr.phase_scoring import _entry_frequency, olr_reject_reason
    metrics = {"entry_fill_count": 900, "entry_level_trade_count": 211, "total_trades": 211}
    assert _entry_frequency(metrics) == 211
    assert "hard_total_trades" in olr_reject_reason(6, metrics, {"min_total_trades": 230})
    assert _entry_frequency({"entry_fill_count": 900, "total_trades": 211}) == 211


def test_recovered_sparse_session_requires_actual_daily_conservation():
    from backtests.engine.retained_source import sparse_dispositions
    values = {"open": 100, "high": 102, "low": 98, "close": 101, "volume": 20}
    row = {"symbol": "005930", "date": "2026-01-05", "disposition": "recovered_for_shared_input",
        "coverage_basis": "daily_ohlcv_conservation", "recovered_daily_comparison": {
            "status": "exact_daily_ohlcv_match", "intraday": values, "daily": dict(values), "mismatched_fields": []}}
    report = {"version": "session-recovery-dispositions-v1", "all_original_flags_accounted": True, "sessions": [row]}
    read = lambda path: report
    assert sparse_dispositions(read, {"session_resolution_report": "report"})
    row["recovered_daily_comparison"]["daily"]["volume"] += 1
    assert not sparse_dispositions(read, {"session_resolution_report": "report"})


def test_kalcb_preflight_hashes_nested_daily_file_content(monkeypatch, tmp_path):
    from dataclasses import dataclass
    from backtests.strategies.kalcb import trade_plan_sweep as sweep
    @dataclass
    class Parameters:
        value: int = 0
    daily = tmp_path / "daily"
    (daily / "tables").mkdir(parents=True)
    prices = daily / "tables" / "prices.parquet"
    prices.write_bytes(b"old")
    source = SimpleNamespace(source_path="source", source_file_hash="file", source_sweep_hash="sweep",
        source_section="section", source_rank=0, frontier=Parameters(), first30=Parameters(), calibration_metadata={})
    monkeypatch.setattr(sweep, "_resolve_symbols", lambda *a, **k: ())
    config = {"daily_data_root": str(daily), "data_root": str(tmp_path / "intraday")}
    before = sweep._preflight_context_fingerprint(config, source, train_only=True, fold_count=2)
    prices.write_bytes(b"new")
    assert sweep._preflight_context_fingerprint(config, source, train_only=True, fold_count=2) != before


def test_candidate_feed_keeps_partial_exit_alive_on_non_candidate_days():
    from datetime import timedelta
    from backtests.engine.replay import candidate_management_bars, run_replay
    from backtests.engine.capacity import BarParticipationScenario
    from strategy_common.actions import SubmitExit
    first = date(2026, 1, 5)
    snapshot = SimpleNamespace(candidates=[SimpleNamespace(symbol="005930")],
        to_json_dict=lambda: {"trade_date": first.isoformat(), "candidates": [{"symbol": "005930"}]})
    by_key = {}
    for offset, times in ((0, ((9, 0, 100), (9, 5, 100), (15, 10, 10), (15, 15, 100))),
                          (1, ((9, 0, 100), (9, 5, 100), (9, 10, 100))), (2, ((9, 0, 100),))):
        day = first + timedelta(days=offset)
        by_key[(day, "005930")] = [MarketBar("005930", datetime.combine(day, time(h, m), KST),
            "5m", 100, 100, 100, 100, v) for h, m, v in times]
    class Adapter:
        strategy_id = "KALCB"
        snapshots = {first: snapshot}
        def on_bar(self, bar, broker):
            if bar.timestamp.date() == first and bar.timestamp.time().replace(tzinfo=None) == time(9):
                broker.submit(SubmitEntry("KALCB", "005930", 2, "MARKET", None, None, "entry"), bar.timestamp)
            if bar.timestamp.date() == first and bar.timestamp.time().replace(tzinfo=None) == time(15, 10):
                broker.submit(SubmitExit("KALCB", "005930", 2, "MARKET", None, "exit"), bar.timestamp)
            return []
    bars = candidate_management_bars(by_key, Adapter.snapshots, [key[0] for key in by_key])
    runs = [run_replay(bars, Adapter(), initial_equity=10000, costs=BrokerCosts(0, 0, 0),
        capacity_policy=BarParticipationScenario(.1, "test"), close_open_positions=False,
        candidate_scoped=scoped) for scoped in (False, True)]
    full, scoped = runs
    assert [(f.side, f.qty, f.timestamp) for f in scoped.broker.fills] == [(f.side, f.qty, f.timestamp) for f in full.broker.fills]
    assert [f.qty for f in scoped.broker.fills if f.side == "SELL"] == [1, 1]
    assert scoped.broker.fills[-1].timestamp.date() == first + timedelta(days=1)
    assert not scoped.broker.positions and not scoped.broker.orders
    assert scoped.broker.cash == full.broker.cash
    assert scoped.broker.source_event_count == 6 < full.broker.source_event_count
    assert audit_broker(scoped.broker)["execution_validity_pass"]


def test_management_only_day_does_not_require_bars_after_exit(monkeypatch):
    from backtests.engine import qualification_evidence as evidence
    from backtests.auto.shared.cache_keys import stable_signature
    first, second = date(2026, 1, 5), date(2026, 1, 6)
    snapshots = [{"trade_date": first.isoformat(), "candidates": [{"symbol": "005930"}]}]
    dispositions = {"KALCB": {second.isoformat(): "builder_verified_empty"}}
    broker = SimpleNamespace(qualification_snapshots=snapshots, qualification_snapshot_scopes={"KALCB": snapshots},
        qualification_day_dispositions=dispositions)
    identity = {"qualification_snapshot_hash": stable_signature(snapshots),
        "qualification_day_dispositions_hash": stable_signature(dispositions), "configuration": {
            "source_dependency_scope": {"version": "causal-feature-and-management-v1", "candidate_intraday_before": "15:20"}}}
    monkeypatch.setattr(evidence, "_management_cutoffs", lambda *a: {("005930", second): "09:10"})
    observed = []
    monkeypatch.setattr(evidence, "_covered", lambda stamps, day, *a, before=None: observed.append((day, before)) or True)
    evidence._replay_scope(broker, identity, [], {("005930", first, "5m"), ("005930", second, "5m")})
    assert observed == [(first, "15:20"), (second, "09:10")]


def test_shared_candidate_prerequisite_requires_exact_pool_scope_and_real_source_gap(monkeypatch, tmp_path):
    import json
    from contextlib import nullcontext
    from backtests.auto.shared.cache_keys import stable_signature
    from backtests.strategies.common.corrective_selection import shared_candidate_input_rejection
    from backtests.engine import qualification_evidence as evidence
    from backtests.engine import retained_source
    scope = {"strategy": "kalcb", "candidate_intraday_before": "15:20"}
    window = {"start": "2026-01-05", "end": "2026-01-05"}
    mutations = {"_kalcb.pool_source.path": "pool", "_kalcb.source.path": "seed", "_kalcb.pool_source.active_count": 16}
    baseline = {"mutations": dict(mutations)}
    row = {"mutations": {**mutations, "kalcb.exit.target_r": 5}}
    proof = {"market_files": [{"path": "bars", "symbol": "005930", "timeframe": "5m"}]}
    contents = {"pool": json.dumps({"symbol": "005930", "trade_date": "2026-01-05", "pool_rank": 7}).encode(),
                "qualification": json.dumps(proof).encode()}
    generation = SimpleNamespace(identity="manifest", root=tmp_path, payload={"source_generation_hash": "code",
        "evidence": {"qualification_path": "qualification"}}, activate=nullcontext,
        read=lambda p: contents[str(p).replace(str(tmp_path) + "/", "").replace(str(tmp_path) + "\\", "")],
        digest=lambda p: str(p))
    audit = {"source_generation_hash": "code", "identity": {"input_manifest": {"manifest_hash": "manifest"},
        "configuration": {"source_dependency_scope": scope, "date_range": window}},
        "coverage": {"input_provenance": "pass", "replay_completeness": "unverified"},
        "qualification": {"replay_completeness": "management_feed_incomplete:005930:2026-01-05",
                          "candidate_sessions": ["005930:2026-01-05"]}}
    def seal():
        audit.pop("receipt_hash", None)
        audit["receipt_hash"] = stable_signature(audit)
    seal()
    monkeypatch.setattr(evidence, "source_rows", lambda *a: iter(()))
    monkeypatch.setattr(retained_source, "sparse_dispositions", lambda *a: {})
    source_is_complete = False
    monkeypatch.setattr(evidence, "_covered", lambda *a, **k: source_is_complete)
    check = lambda: shared_candidate_input_rejection(generation, scope, row, baseline, audit, window=window)
    assert check()["reason"] == "shared_candidate_inputs_unverified"
    source_is_complete = True
    assert check() is None  # Never turn a replay omission into a source rejection.
    source_is_complete = False
    row["mutations"]["_kalcb.pool_source.path"] = "different_pool"
    assert check() is None
    row["mutations"]["_kalcb.pool_source.path"] = "pool"
    audit["qualification"]["candidate_sessions"] = []
    seal()
    assert check() is None  # A control's position lifetime is not common to every recipe.
    audit["qualification"]["candidate_sessions"] = ["005930:2026-01-05"]
    seal()
    audit["identity"]["input_manifest"]["manifest_hash"] = "changed"
    assert check() is None
