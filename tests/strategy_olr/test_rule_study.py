from copy import deepcopy

import pytest

from scripts.olr_rule_study import RULES, WEIGHTS, digest, load_study, read, recipes, register, signature, stability_recipes, write
from scripts.olr_rule_analysis import bootstrap_difference, candidate_difference, load_case, select_finalist, trade_statistics
from scripts.olr_rule_alpha_attribution import decompose_filled_trade, fill_lifecycles
from scripts.olr_rule_finish import qualification_decision
from scripts.olr_prospective_validation import check_contract


@pytest.fixture
def control():
    return {
        RULES: [
            {"name": "low", "max_score": 300.0},
            {"name": "middle", "min_score": 400.0, "max_score": 500.0, "allowed_sectors": ["IT"]},
            {"name": "dynamic", "min_score": 400.0, "max_score": 500.0, "sector_admission": {"min_sector_intraday_ret_pct": 2}},
            {"name": "high", "min_score": 650.0},
        ],
        WEIGHTS[0]: 0.0005, WEIGHTS[1]: 0.001,
        "olr.afternoon.reject_score_min": 0.0,
        "olr.afternoon.reject_score_max": 0.0,
        "olr.trade_plan.entry": {"mode": "confirm_next_bar"},
    }


def test_ablations_rebuild_admission_without_changing_execution(control):
    before = deepcopy(control)
    cases = recipes(control)
    assert control == before and cases["control"]["mutations"] == before
    assert len(cases) == 8
    assert all(c["mutations"]["olr.trade_plan.entry"] == before["olr.trade_plan.entry"] for c in cases.values())
    middle = cases["all_middle_sectors_soft"]["mutations"][RULES]
    assert len(middle) == 3 and all("allowed_sectors" not in r and "sector_admission" not in r for r in middle)
    assert len(cases["tails_only_soft"]["mutations"][RULES]) == 2
    assert cases["no_bands_soft"]["mutations"][RULES] == []
    assert all(cases["current_zero"]["mutations"][k] == 0 for k in WEIGHTS)
    cases["current_zero"]["mutations"][RULES][0]["max_score"] = 1
    assert control == before and cases["control"]["mutations"] == before


def test_no_band_ablation_cannot_leave_legacy_notch_active(control):
    control["olr.afternoon.reject_score_min"] = 300
    control["olr.afternoon.reject_score_max"] = 650
    result = recipes(control)["no_bands_soft"]["mutations"]
    assert result["olr.afternoon.reject_score_min"] == result["olr.afternoon.reject_score_max"] == 0


def test_stability_moves_both_middle_routes_together(control):
    neighbours = stability_recipes(control)
    assert len(neighbours) == 8
    for r in neighbours["middle_+25"][RULES][1:3]:
        assert r["min_score"] == 425 and r["max_score"] == 525
    assert neighbours["weights_x0.5"][WEIGHTS[1]] == 0.0005
    assert neighbours["low_-25"][RULES][-1] == control[RULES][-1]
    assert len({signature(m) for m in neighbours.values()}) == 8
    assert stability_recipes(recipes(control)["no_bands_zero"]["mutations"]) == {}


def test_registered_study_rejects_manifest_harness_or_policy_drift(tmp_path):
    harness, manifest = tmp_path / "worker.py", tmp_path / "manifest.json"
    harness.write_text("pass\n")
    manifest.write_text("{}\n")
    spec = {"harness": str(harness), "harness_sha256": digest(harness),
            "input_manifest": str(manifest), "input_manifest_file_sha256": digest(manifest)}
    spec["study_hash"] = signature(spec)
    path = tmp_path / "study.json"
    write(path, spec)
    assert load_study(path) == spec
    harness.write_text("print('changed')\n")
    with pytest.raises(ValueError, match="evaluator changed"):
        load_study(path)
    harness.write_text("pass\n")
    manifest.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="manifest changed"):
        load_study(path)
    spec["extra_rule"] = "change selection after seeing outcomes"
    write(path, spec)
    with pytest.raises(ValueError, match="study bytes changed"):
        load_study(path)


def selection_fixture():
    policy = {"maximum_drawdown": .12, "maximum_drawdown_increase": .01,
              "minimum_profit_factor": 1.1, "minimum_mean_net_r_ratio": .95}
    result = {"metrics": {"official_mtm_net_return_pct": 1.6, "total_trades": 234,
                          "max_drawdown_pct": .07, "profit_factor": 3, "audit_pass": True},
              "trade_statistics": {"mean_net_r": .3}}
    baseline = {mode: deepcopy(result) for mode in ("nominal", "sizing_cap")}
    candidate = deepcopy(baseline)
    for mode in candidate:
        candidate[mode]["metrics"]["official_mtm_net_return_pct"] += .05
        candidate[mode]["metrics"]["total_trades"] += 10
    return policy, {"control": baseline, "challenger": candidate}


def test_selection_cannot_use_regression_and_requires_capped_improvement():
    policy, cases = selection_fixture()
    assert select_finalist(cases, policy)["selected_challenger"] == "challenger"
    cases["challenger"]["regression"] = {"return": -999999}
    assert select_finalist(cases, policy)["selected_challenger"] == "challenger"
    cases["challenger"]["sizing_cap"]["metrics"]["official_mtm_net_return_pct"] = 1.59
    cases["challenger"]["regression"]["return"] = 999999
    outcome = select_finalist(cases, policy)
    assert outcome["selected_challenger"] is None
    assert "sizing_cap:lower_net_return" in outcome["assessment"]["challenger"]["reasons"]


def test_more_trades_do_not_win_with_poorer_net_quality():
    policy, cases = selection_fixture()
    cases["challenger"]["nominal"]["trade_statistics"]["mean_net_r"] = .2
    assert select_finalist(cases, policy)["selected_challenger"] is None
    cases["challenger"]["nominal"]["trade_statistics"]["mean_net_r"] = .3
    cases["challenger"]["nominal"]["metrics"]["audit_pass"] = False
    assert select_finalist(cases, policy)["selected_challenger"] is None


def test_paired_uncertainty_uses_matching_calendars_and_time_blocks():
    base = [{"date": f"2026-01-{i:02d}", "net_return": .01} for i in range(1, 21)]
    same = bootstrap_difference(base, base, samples=100)
    assert same["conditional_95pct_interval"] == [0, 0]
    shifted = [{**r, "net_return": .012} for r in base]
    better = bootstrap_difference(shifted, base, samples=100)
    assert better["conditional_95pct_interval"] == pytest.approx([.002, .002])
    with pytest.raises(ValueError, match="calendars differ"):
        bootstrap_difference(shifted[:-1], base)


def test_net_r_uses_actual_fees_and_risk_not_gross_r_label():
    trade = {"route_metadata": {"risk_per_share": 2, "sector": "IT"}, "qty": 10,
             "entry_price": 100, "net_pnl": 10, "r_multiple": 999,
             "entry_fill_time": "2026-01-05T14:35:00+09:00", "symbol": "005930"}
    stats = trade_statistics([trade])
    assert stats["mean_net_r"] == .5 and stats["mean_trade_net_return"] == .01


def test_candidate_comparison_separates_membership_and_rank():
    control = [{"date": "2026-01-05", "selected": [{"symbol": "A", "rank": 1, "score": 200}], "rejected": {}}]
    other = [{"date": "2026-01-05", "selected": [{"symbol": "A", "rank": 2, "score": 210},
                                                {"symbol": "B", "rank": 1, "score": 400}], "rejected": {}}]
    result = candidate_difference(other, control)
    assert result["added_selected_pairs"] == 1 and result["common_rank_changes"] == 1


def test_ranking_change_needs_multiple_independent_opportunities():
    policy, cases = selection_fixture()
    control_rows = [{"date": "2026-01-05", "selected": [{"symbol": "A", "rank": 1, "score": 100}], "rejected": {}}]
    changed = [{"date": "2026-01-05", "selected": [{"symbol": "A", "rank": 2, "score": 101}], "rejected": {}}]
    cases["control"]["nominal"]["candidates"] = control_rows
    cases["challenger"]["nominal"]["candidates"] = changed
    support = {"minimum_changed_selected_issuer_days": 10, "minimum_changed_calendar_months": 3}
    result = select_finalist(cases, policy, support)
    assert result["selected_challenger"] is None
    assert "insufficient_changed_issuer_days" in result["assessment"]["challenger"]["reasons"]


def test_price_path_components_reconcile_without_future_mfe():
    trade = {"qty": 10, "entry_price": 100, "gross_pnl": 100, "net_pnl": 90}
    fills = [{"symbol": "A", "timestamp": "2026-01-05T14:35", "side": "BUY", "qty": 10, "price": 100},
             {"symbol": "A", "timestamp": "2026-01-06T15:30", "side": "SELL", "qty": 10, "price": 110}]
    parts = decompose_filled_trade(trade, [{"date": "2026-01-05", "open": 98, "close": 102},
                                          {"date": "2026-01-06", "open": 105, "close": 110}], fills)
    assert parts == {"entry_session_pnl_krw": 20, "overnight_gap_pnl_krw": 30,
                     "subsequent_intraday_pnl_krw": 50, "explicit_fees_krw": 10, "net_pnl_krw": 90}
    fills[-1]["timestamp"] = "2026-01-05T15:30"
    same_day = decompose_filled_trade(trade, [{"date": "2026-01-05", "open": 98, "close": 102}], fills)
    assert same_day["overnight_gap_pnl_krw"] == 0 and same_day["subsequent_intraday_pnl_krw"] == 0


def test_partial_exits_reduce_quantity_earning_later_overnight_gaps():
    trade = {"qty": 10, "gross_pnl": 164, "net_pnl": 150}
    fills = [{"symbol": "A", "timestamp": "2026-01-05T14:35", "side": "BUY", "qty": 5, "price": 98},
             {"symbol": "A", "timestamp": "2026-01-05T14:40", "side": "BUY", "qty": 5, "price": 102},
             {"symbol": "A", "timestamp": "2026-01-06T15:30", "side": "SELL", "qty": 4, "price": 111},
             {"symbol": "A", "timestamp": "2026-01-07T15:30", "side": "SELL", "qty": 6, "price": 120}]
    sessions = [{"date": "2026-01-05", "open": 90, "close": 100},
                {"date": "2026-01-06", "open": 110, "close": 112},
                {"date": "2026-01-07", "open": 120, "close": 120}]
    groups = fill_lifecycles(fills)
    assert groups == {("A", "2026-01-05T14:35"): fills}
    parts = decompose_filled_trade(trade, sessions, fills)
    assert parts["overnight_gap_pnl_krw"] == 148
    assert parts["subsequent_intraday_pnl_krw"] == 16
    assert parts["explicit_fees_krw"] == 14


def test_winning_selection_failing_qualification_restores_whole_control(control):
    cases = recipes(control)
    spec = {"cases": cases, "stress_policy": {"scenarios": ["cost-stress"],
            "minimum_return_ratio_vs_control": .9, "minimum_trade_count_ratio_vs_control": .95,
            "maximum_drawdown": .15, "minimum_profit_factor": 1.1}}
    valid = {"audit_pass": True, "official_mtm_net_return_pct": 1.5, "total_trades": 230,
             "max_drawdown_pct": .1, "profit_factor": 2}
    stress = {"cost-stress": {**valid, "official_mtm_net_return_pct": .5}}
    result = qualification_decision("no_bands_zero", stress, {"cost-stress": valid}, {}, valid, spec)
    assert result["selected"] == "control" and result["mutations"] == control
    assert result["rejection_reasons"] == ["cost-stress:return"]
    assert result["challenger_development_qualified"] is False
    stress["cost-stress"] = valid
    unstable = {"bad_neighbour": {**valid, "official_mtm_net_return_pct": .2}}
    assert qualification_decision("current_zero", stress, {"cost-stress": valid}, unstable, valid, spec)["selected"] == "control"


def test_registration_freezes_reusable_pipeline_and_support_before_results(control, tmp_path):
    manifest = tmp_path / "inputs/manifest.json"
    write(manifest, {"source_generation_hash": "test-frozen-source"})
    round_dir = tmp_path / "round_1"
    write(round_dir / "diagnostics/training/request.json", {
        "config": {"input_manifest_path": str(manifest)}, "mutations": control})
    write(round_dir / "diagnostics/training/metrics.json", {"total_trades": 0})
    destination = tmp_path / "study"
    register(round_dir, destination)
    spec = load_study(destination / "study.json")
    support = read(destination / "support_policy.json")
    assert support["study_hash"] == spec["study_hash"]
    assert support["minimum_changed_selected_issuer_days"] == 10
    pipeline = read(destination / "analysis_pipeline_manifest.json")
    assert len(pipeline["files"]) == 4
    assert all(digest(destination / "harness" / name) == value for name, value in pipeline["files"].items())
    future = read(destination / "prospective_validator_manifest.json")
    assert "evaluation_sessions" not in future
    assert not future["completion_requires_future_sessions"]
    assert all(digest(destination / "prospective_harness" / name) == value for name, value in future["files"].items())
    with pytest.raises(FileExistsError):
        register(round_dir, destination)


def test_prospective_evaluation_rejects_consumed_dates_unbound_or_short_inputs(control):
    contract = {"study_hash": "study", "source_generation_hash": "source", "mutations": control,
                "mutations_hash": signature(control), "earliest_eligible_kst_date": "2026-10-08"}
    contract["contract_hash"] = signature(contract)
    manifest = {"source_generation_hash": "source", "evidence": {
        "validation_contract_hash": contract["contract_hash"], "shared_market_inputs": {"data_end": "2027-02-01"}}}
    window = {"start": "2026-10-08", "end": "2027-01-08"}
    check_contract(contract, {"study_hash": "study"}, manifest, window)
    with pytest.raises(ValueError, match="predates"):
        check_contract(contract, {"study_hash": "study"}, manifest, {"start": "2026-04-01", "end": "2026-05-12"})
    manifest["evidence"]["shared_market_inputs"]["data_end"] = "2026-05-12"
    with pytest.raises(ValueError, match="coverage"):
        check_contract(contract, {"study_hash": "study"}, manifest, window)
    manifest["evidence"].pop("validation_contract_hash")
    with pytest.raises(ValueError, match="not bound"):
        check_contract(contract, {"study_hash": "study"}, manifest, window)


def test_completed_outputs_cannot_be_relabelled_as_another_candidate(tmp_path):
    spec = {"output": str(tmp_path), "study_hash": "study"}
    directory = tmp_path / "cases/challenger/nominal"
    job = {"study_hash": "study", "case": "control", "mode": "nominal", "config": {}, "mutations": {}}
    write(directory / "job.json", job)
    write(directory / "completed.json", {"study_hash": "study", "job_hash": signature(job),
                                         "outputs": {"job.json": digest(directory / "job.json")}})
    with pytest.raises(ValueError, match="directory and request"):
        load_case(spec, "challenger", "nominal")


def test_final_acceptance_requires_all_neighbours_and_recomputes_qualification(control):
    from scripts.olr_rule_verify import verify_qualification_evidence

    spec = {"cases": recipes(control), "stress_policy": {"scenarios": ["cost-stress"],
            "minimum_return_ratio_vs_control": .9, "minimum_trade_count_ratio_vs_control": .95,
            "maximum_drawdown": .15, "minimum_profit_factor": 1.1}}
    valid = {"audit_pass": True, "official_mtm_net_return_pct": 1.5, "total_trades": 240,
             "max_drawdown_pct": .1, "profit_factor": 2}
    parents = ["control", "current_zero"]
    evidence = {"stability": {}, "stress": {parent: {"cost-stress": valid} for parent in parents}}
    actual = {(parent, mode): valid for parent in parents for mode in ("nominal", "cost-stress")}
    for parent in parents:
        labels = stability_recipes(spec["cases"][parent]["mutations"])
        evidence["stability"][parent] = {label: valid for label in labels}
        for label in labels:
            name = "stability__" + parent + "__" + label.replace("+", "plus").replace("-", "minus")
            actual[(name, "nominal")] = valid
    decision = qualification_decision("current_zero", evidence["stress"]["current_zero"],
        evidence["stress"]["control"], evidence["stability"]["current_zero"], valid, spec)
    verify_qualification_evidence(spec, "current_zero", evidence, actual, decision)
    missing = deepcopy(evidence)
    missing["stability"]["current_zero"].pop(next(iter(missing["stability"]["current_zero"])))
    with pytest.raises(ValueError, match="stability evidence is incomplete"):
        verify_qualification_evidence(spec, "current_zero", missing, actual, decision)
    forged = {**decision, "selected": "control", "mutations": control}
    with pytest.raises(ValueError, match="differs from registered gates"):
        verify_qualification_evidence(spec, "current_zero", evidence, actual, forged)


def test_rejected_native_result_is_preserved_without_acceptance(tmp_path):
    from types import SimpleNamespace
    from scripts.olr_rule_study import export_or_preserve_failure

    result = SimpleNamespace(metrics={"audit_pass": False, "audit_errors": ["missing_execution_bar"]},
        trades=[], decisions=[], replay_result=SimpleNamespace(broker=SimpleNamespace(fills=[], journal=[])))

    def rejected_export(*_args):
        raise ValueError("Invalid execution receipt")

    saved = {}
    with pytest.raises(ValueError, match="Invalid execution receipt"):
        export_or_preserve_failure(result, {}, {}, tmp_path, rejected_export,
                                   lambda path, records: saved.update({path.name: records}), lambda value: value)
    failure = read(tmp_path / "failure.json")
    assert failure["status"] == "rejected_not_qualified"
    assert failure["metrics"]["audit_errors"] == ["missing_execution_bar"]
    assert not (tmp_path / "completed.json").exists()
    assert set(saved) == {"failed_trades.jsonl.gz", "failed_fills.jsonl.gz", "failed_ledger.jsonl.gz", "failed_decisions.jsonl.gz"}


def test_unqualified_case_cannot_be_ranked_even_with_attractive_other_mode():
    policy, cases = selection_fixture()
    cases["challenger"]["nominal"] = {"qualification_rejected": True}
    cases["challenger"]["sizing_cap"]["metrics"]["official_mtm_net_return_pct"] = 999
    result = select_finalist(cases, policy)
    assert result["selected_challenger"] is None
    assert result["assessment"]["challenger"]["minimum_relative_wealth_gain"] is None
    assert result["assessment"]["challenger"]["reasons"] == ["nominal:unqualified_native_result"]


def test_unqualified_stress_or_neighbour_forces_complete_fallback(control):
    spec = {"cases": recipes(control), "stress_policy": {"scenarios": ["cost-stress"],
            "minimum_return_ratio_vs_control": .9, "minimum_trade_count_ratio_vs_control": .95,
            "maximum_drawdown": .15, "minimum_profit_factor": 1.1}}
    valid = {"audit_pass": True, "official_mtm_net_return_pct": 1.5, "total_trades": 240,
             "max_drawdown_pct": .1, "profit_factor": 2}
    unqualified = {"audit_pass": False, "qualification_rejected": True}
    decision = qualification_decision("current_zero", {"cost-stress": unqualified},
                                     {"cost-stress": valid}, {}, valid, spec)
    assert decision["selected"] == "control" and decision["mutations"] == control
    decision = qualification_decision("current_zero", {"cost-stress": valid},
                                     {"cost-stress": valid}, {"missing_feed": unqualified}, valid, spec)
    assert decision["selected"] == "control"
    assert "local_stability:drawdown_or_audit" in decision["rejection_reasons"]


def test_rejection_receipt_binds_native_evidence_and_omits_economics(tmp_path):
    from scripts.olr_rule_resolved_pipeline import reject_receipt
    manifest = tmp_path / "manifest.json"
    write(manifest, {"manifest_hash": "inputs"})
    spec = {"output": str(tmp_path), "study_hash": "study", "source_generation_hash": "source",
            "input_manifest": str(manifest)}
    directory = tmp_path / "cases/unsupported/nominal"
    job = {"case": "unsupported", "mode": "nominal", "study_hash": "study", "config": {}, "mutations": {}}
    request = signature({"config": {}, "mutations": {}, "input_generation": "inputs"})
    audit = {"source_generation_hash": "source", "identity": {"request": request, "mutations": signature({})},
             "errors": [], "coverage": {"daily_marks": "unverified"}}
    audit["receipt_hash"] = signature(audit)
    metrics = {"audit_pass": False, "daily_equity_coverage": {"status": "unverified"}, "audit_receipt": audit,
               "evaluation_request": request, "evaluation_mutations": signature({}), "official_mtm_net_return_pct": 999}
    write(directory / "job.json", job)
    write(directory / "forensic_request.json", {"config": {}, "mutations": {}})
    write(directory / "forensic_metrics.json", metrics)
    reject_receipt(spec, directory)
    loaded = load_case(spec, "unsupported", "nominal")
    assert loaded["qualification_rejected"] is True
    assert "official_mtm_net_return_pct" not in loaded["metrics"]
    assert not (directory / "completed.json").exists()
    metrics["audit_receipt"]["coverage"]["daily_marks"] = "pass"
    write(directory / "forensic_metrics.json", metrics)
    with pytest.raises(ValueError, match="Changed result"):
        load_case(spec, "unsupported", "nominal")


def test_bounded_verified_cache_evicts_storage_without_weakening_hash_checks(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from backtests.core import input_manifest
    from scripts.olr_rule_memory import BoundedVerifiedBytes
    cache = BoundedVerifiedBytes(4)
    monkeypatch.setattr(input_manifest, "_VERIFIED_BYTES", cache)
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"abc")
    second.write_bytes(b"def")
    expected = {first: hashlib.sha256(b"abc").hexdigest(), second: hashlib.sha256(b"def").hexdigest()}
    generation = SimpleNamespace(resolve=lambda path: path, digest=expected.get,
                                 identity="fixture", loaded=set(), root=tmp_path)
    read_native = lambda path: input_manifest.InputManifest.read(generation, path)
    assert read_native(first) == b"abc" and read_native(second) == b"def"
    assert cache.bytes == 3 and cache.peak_bytes <= 4 and cache.evictions == 1
    first.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Frozen input bytes changed"):
        read_native(first)
    cache["oversized"] = b"12345"
    assert "oversized" not in cache and cache.bytes == 3
    cache.clear()
    assert cache.bytes == 0 and not cache


def test_memory_observer_calls_the_original_audit_and_preserves_its_result(monkeypatch):
    from types import SimpleNamespace
    import pyarrow as pa
    from backtests.core import input_manifest
    from backtests.engine import integrity
    from scripts.olr_rule_memory import install_memory_controls
    calls, stages = [], []
    result, broker, identity = {"audit_pass": False, "reason": "still rejected"}, object(), {"request": "same"}
    def native(broker_arg, *, identity):
        calls.append((broker_arg, identity))
        return result
    monkeypatch.setattr(integrity, "audit_broker", native)
    monkeypatch.setattr(input_manifest, "_VERIFIED_BYTES", {})
    monkeypatch.setattr(pa, "set_cpu_count", lambda count: None)
    monkeypatch.setattr(pa, "set_io_thread_count", lambda count: None)
    monkeypatch.setattr(pa, "default_memory_pool", lambda: SimpleNamespace(release_unused=lambda: None))
    cache = install_memory_controls(stages.append, maximum_cache_bytes=4)
    cache["cache"] = b"abc"
    assert integrity.audit_broker(broker, identity=identity) is result
    assert calls == [(broker, identity)] and cache.bytes == 0
    assert stages == ["before_audit_trim", "after_audit_trim", "after_native_audit", "after_audit_cleanup"]


def test_split_runtime_preserves_native_bundle_values_aliases_and_immutable_payloads(tmp_path):
    from datetime import datetime
    from types import MappingProxyType
    from backtests.core.replay_bundle import EventReplayBundle
    from backtests.core.replay_events import ReplayEvent
    from strategy_common.market import MarketBar
    from scripts.olr_rule_split_worker import save_bundle, load_bundle
    stamp = datetime.fromisoformat('2025-05-12T14:35:00+09:00')
    bar = MarketBar(symbol='005930', timestamp=stamp, timeframe='5m',
                    open=100, high=102, low=99, close=101, volume=300, is_completed=True)
    event = ReplayEvent.from_bar(bar)
    bundle = EventReplayBundle(events=(event,), source_fingerprint='native-source',
                              metadata={'shared_bar': bar, 'signed_zero': -0.0})
    path = tmp_path / 'bundle.pkl'
    digest = save_bundle(bundle, path)
    loaded = load_bundle(path, digest)
    assert loaded == bundle
    assert loaded.events[0].bar is loaded.metadata['shared_bar']
    assert isinstance(loaded.events[0].payload, MappingProxyType)
    with pytest.raises(TypeError):
        loaded.events[0].payload['bar'] = None
    import math
    assert math.copysign(1, loaded.metadata['signed_zero']) == -1
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='bundle changed'):
        load_bundle(path, digest)


def test_single_pass_manifest_index_matches_native_and_reduces_path_lookups(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from backtests.core.input_manifest import InputManifest, signature as manifest_signature
    from scripts.olr_rule_manifest_index import initialize_manifest
    payload = {'files': {'a': 'hash-a', 'nested/b': 'hash-b'}, 'source_root': str(tmp_path),
               'source_generation_hash': 'source'}
    path = tmp_path / 'manifest.json'
    write(path, {**payload, 'manifest_hash': manifest_signature(payload)})
    calls = []
    original = Path.resolve
    def counted(self, *args, **kwargs):
        calls.append(str(self))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', counted)
    native = InputManifest(path)
    native_calls = len(calls)
    calls.clear()
    revised = SimpleNamespace()
    initialize_manifest(revised, path, manifest_signature)
    assert vars(revised) == vars(native)
    assert len(calls) == len(payload['files']) + 3
    assert native_calls == 3 * len(payload['files']) + 1
    assert revised.loaded is not native.loaded


def test_single_pass_manifest_keeps_identity_containment_and_native_byte_checks(tmp_path, monkeypatch):
    import hashlib
    from backtests.core import input_manifest
    from scripts.olr_rule_manifest_index import install_manifest_index_controls
    monkeypatch.setattr(input_manifest.InputManifest, '__init__', input_manifest.InputManifest.__init__)
    counters = install_manifest_index_controls()
    files = tmp_path / 'files'
    files.mkdir()
    data = files / 'a'
    data.write_bytes(b'original')
    path = tmp_path / 'manifest.json'
    payload = {'files': {'a': hashlib.sha256(b'original').hexdigest()}, 'source_root': str(tmp_path),
               'source_generation_hash': 'source'}
    write(path, {**payload, 'manifest_hash': input_manifest.signature(payload)})
    generation = input_manifest.InputManifest(path)
    assert generation.read(data) == b'original'
    data.write_bytes(b'altered')
    input_manifest._VERIFIED_BYTES.clear()
    with pytest.raises(ValueError, match='Frozen input bytes changed'):
        generation.read(data)
    escaped = {**payload, 'files': {'../outside': 'hash'}}
    write(path, {**escaped, 'manifest_hash': input_manifest.signature(escaped)})
    with pytest.raises(ValueError, match='escapes its generation'):
        input_manifest.InputManifest(path)
    write(path, {**payload, 'manifest_hash': 'wrong'})
    with pytest.raises(ValueError, match='identity mismatch'):
        input_manifest.InputManifest(path)
    assert counters == {'constructions': 1, 'indexed_files': 1}
