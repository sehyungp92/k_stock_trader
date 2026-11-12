import json
from pathlib import Path

import pytest

from scripts.olr_rule_study import signature, write, digest
from scripts.olr_prospective_validation import (
    evaluation_days, prospective_uncertainty, record_session, revise_protocol, setup_status, verified_contract,
)
from scripts.generate_olr_kalcb_artifacts import _load_strategy_configs


def contract_file(tmp_path):
    contract = {"study_hash": "study", "mutations": {"olr.afternoon.top_n": 4},
                "source_generation_hash": "engine", "earliest_eligible_kst_date": "2026-10-01",
                "completion_requires_future_sessions": False}
    contract["mutations_hash"] = signature(contract["mutations"])
    contract["contract_hash"] = signature(contract)
    path = tmp_path / "contract.json"
    write(path, contract)
    return path, contract


def test_olr_only_config_ignores_missing_kalcb_and_rejects_changed_bytes(tmp_path):
    config = tmp_path / "optimized_config.json"
    write(config, {"mutations": {"olr.afternoon.top_n": 4}})
    baseline = tmp_path / "baseline.json"
    write(baseline, {"artifacts": [{"label": "olr_optimized_config", "path": str(config), "sha256": digest(config)}]})
    assert set(_load_strategy_configs(baseline, ("OLR",))) == {"OLR"}
    with pytest.raises(ValueError, match="KALCB"):
        _load_strategy_configs(baseline)
    write(config, {"mutations": {"olr.afternoon.top_n": 5}})
    with pytest.raises(ValueError, match="SHA mismatch"):
        _load_strategy_configs(baseline, ("OLR",))


def test_setup_does_not_require_final_horizon_but_reports_pending_coverage(tmp_path, monkeypatch):
    from scripts import olr_prospective_validation as validator
    path, contract = contract_file(tmp_path)
    engine = tmp_path / "engine"
    schedule = engine / "kis_core/data/krx_sessions.yaml"
    schedule.parent.mkdir(parents=True)
    schedule.write_text("coverage_end: '2026-05-12'\n")
    monkeypatch.setattr(validator, "load_study", lambda _: {"study_hash": "study", "engine_root": str(engine)})
    result = setup_status(path, tmp_path / "status")
    assert result["setup_pass"]
    assert len(result["pending_data_requirements"]) == 2
    assert not result["final_performance_evaluated"]
    assert not result["completion_requires_future_sessions"]
    assert "evaluation_sessions" not in result


def test_available_window_completes_without_fixed_session_count():
    from datetime import date
    from types import SimpleNamespace
    calendar = SimpleNamespace(has_coverage=lambda _: True, has_session_coverage=lambda _: True,
                               is_trading_day=lambda day: day.weekday() < 5)
    contract = {"earliest_eligible_kst_date": "2026-10-01"}
    manifest = {"evidence": {"shared_market_inputs": {"data_end": "2026-10-03"}}}
    assert evaluation_days(contract, manifest, calendar) == [date(2026, 10, 1), date(2026, 10, 2)]
    calendar.has_session_coverage = lambda _: False
    with pytest.raises(ValueError, match="session coverage"):
        evaluation_days(contract, manifest, calendar)
    calendar.has_coverage = lambda _: False
    with pytest.raises(ValueError, match="calendar coverage"):
        evaluation_days(contract, manifest, calendar)
    manifest["evidence"]["shared_market_inputs"]["data_end"] = "2026-09-30"
    with pytest.raises(ValueError, match="No eligible observations"):
        evaluation_days(contract, manifest, calendar)
    with pytest.raises(ValueError, match="Declare evaluation data_end"):
        evaluation_days(contract, {}, calendar)


def test_short_report_does_not_mistake_degenerate_bootstrap_for_evidence():
    left = [{"date": "2026-10-01", "net_return": .01}]
    right = [{"date": "2026-10-01", "net_return": 0}]
    result = prospective_uncertainty(left, right)
    assert result["conditional_95pct_interval"][0] > 0
    assert result["support_status"] == "inconclusive"


def test_policy_revision_preserves_original_freeze_and_recipe(tmp_path, monkeypatch):
    from scripts import olr_prospective_validation as validator
    path, original = contract_file(tmp_path)
    # Migrate a legacy contract without rewriting its historical record.
    original.pop("contract_hash")
    original["evaluation_sessions"] = 60
    original["frozen_at_utc"] = "2026-09-30T00:00:00+00:00"
    original["contract_hash"] = signature(original)
    write(path, original)
    before = path.read_bytes()
    monkeypatch.setattr(validator, "load_study", lambda _: {"study_hash": "study"})
    destination = tmp_path / "revision"
    revised = revise_protocol(path, destination)
    assert path.read_bytes() == before
    assert revised["parent_contract_hash"] == original["contract_hash"]
    assert "evaluation_sessions" not in revised
    assert not revised["completion_requires_future_sessions"]
    for key in ("frozen_at_utc", "earliest_eligible_kst_date", "mutations", "source_generation_hash"):
        assert revised[key] == original[key]
    assert verified_contract(destination / "prospective_validation_contract.json") == revised
    implementation = json.loads((destination / "prospective_validator_manifest.json").read_text())
    assert all(digest(destination / "prospective_harness" / name) == sha for name, sha in implementation["files"].items())
    with pytest.raises(FileExistsError):
        revise_protocol(path, destination)


def test_capture_never_accepts_a_consumed_date_or_wrong_recipe(tmp_path):
    path, contract = contract_file(tmp_path)
    root = tmp_path / "capture"
    write(root / "session_manifest.json", {"trade_date": "2026-03-31"})
    with pytest.raises(ValueError, match="prospective dates"):
        record_session(path, root, tmp_path / "journal")
    write(root / "session_manifest.json", {"trade_date": "2026-10-01", "hash_contract_status": "sealed",
                                           "expected_hashes_complete": True, "strategy_configs": {"OLR": {"payload": {}}}})
    with pytest.raises(ValueError, match="recipe"):
        record_session(path, root, tmp_path / "journal")


def test_capture_archives_zero_trade_session_and_refuses_overwrite(tmp_path, monkeypatch):
    from deployment.olr_kalcb import session_capture
    path, contract = contract_file(tmp_path)
    root = tmp_path / "capture"
    hashes = {key: key for key in session_capture.REQUIRED_EXPECTED_HASH_GROUPS}
    write(root / "session_manifest.json", {"trade_date": "2026-10-01", "hash_contract_status": "sealed",
        "expected_hashes_complete": True, "expected_hashes": hashes,
        "strategy_configs": {"OLR": {"payload": contract["mutations"]}}, "mode": "dry_run"})
    (root / "fill_events.jsonl").write_text("")
    monkeypatch.setattr(session_capture, "session_hashes", lambda _: hashes)
    result = record_session(path, root, tmp_path / "journal")
    assert not result["performance_evaluated"]
    assert (Path(result["capture_root"]) / "fill_events.jsonl").exists()
    with pytest.raises(FileExistsError):
        record_session(path, root, tmp_path / "journal")
    incomplete = json.loads((root / "session_manifest.json").read_text())
    incomplete["expected_hashes"] = {}
    write(root / "session_manifest.json", incomplete)
    with pytest.raises(ValueError, match="hash groups"):
        record_session(path, root, tmp_path / "other_journal")


def test_explicit_execution_tree_hash_does_not_use_mutable_workspace(tmp_path):
    from backtests.auto.shared.cache_keys import source_generation_hash
    from kis_core.input_manifest import InputManifest
    source = tmp_path / "source"
    code = source / "strategy_olr/config.py"
    code.parent.mkdir(parents=True)
    code.write_text("value = 1\n")
    expected = source_generation_hash(root=source)
    generation = InputManifest.create([code], source_root=source, execution_source_root=source, destination=tmp_path / "generation")
    assert generation.payload["source_generation_hash"] == expected
    assert source_generation_hash(root=generation.root) == expected
    code.write_text("value = 2\n")
    assert source_generation_hash(root=source) != expected
    assert source_generation_hash(root=generation.root) == expected


def test_ambiguous_intraday_generations_require_explicit_selection(tmp_path):
    from datetime import date
    from scripts.generate_olr_kalcb_artifacts import _intraday_5m_files_for_trade_date
    first = tmp_path / "005930_5m_20250101_20260512.parquet"
    second = tmp_path / "005930_5m_20250710_20260710.parquet"
    first.touch()
    second.touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        _intraday_5m_files_for_trade_date(tmp_path, "005930", date(2026, 2, 27))
    assert _intraday_5m_files_for_trade_date(tmp_path, "005930", date(2026, 2, 27), snapshot_end="20260512") == (first,)


def test_sourced_halts_remove_only_fully_halted_intervals():
    from datetime import date, datetime, timedelta
    from zoneinfo import ZoneInfo
    from strategy_common.session_coverage import session_bar_coverage
    day = date(2026, 6, 8)
    opening = datetime(2026, 6, 8, 9, tzinfo=ZoneInfo("Asia/Seoul"))
    bars = [opening + timedelta(minutes=5 * i) for i in range(76)]
    bars += [opening.replace(hour=15, minute=30)]
    bars = [b for b in bars if b.strftime("%H:%M") not in {"09:05", "09:10", "09:15", "09:20", "09:25"}]
    assert session_bar_coverage(bars, day, 5, symbol="005930")["status"] == "pass"
    # 09:00 contains trading before the 09:03:42 halt and cannot be waived.
    assert session_bar_coverage(bars[1:], day, 5, symbol="005930")["status"] == "unverified"


def test_artifact_prefix_uses_sourced_halts_and_rejects_equal_count_wrong_timestamps():
    from datetime import date, datetime, timedelta
    from types import SimpleNamespace
    from strategy_common.clock import KST
    from scripts.generate_olr_kalcb_artifacts import _validate_afternoon_coverage
    day = date(2026, 6, 8)
    start = datetime(2026, 6, 8, 9, tzinfo=KST)
    stamps = [start + timedelta(minutes=5 * i) for i in range(66) if i not in range(1, 6)]
    bars = {(day, "005930"): tuple(SimpleNamespace(timestamp=stamp) for stamp in stamps)}
    result = _validate_afternoon_coverage(bars, ("005930",), min_bars_per_symbol=66,
                                         allow_partial=False, trade_date=day)
    assert not result.underfilled_symbols
    bars[(day, "005930")] = bars[(day, "005930")][1:] + (SimpleNamespace(timestamp=start.replace(hour=8)),)
    with pytest.raises(ValueError, match="completed 5m bar coverage"):
        _validate_afternoon_coverage(bars, ("005930",), min_bars_per_symbol=66,
                                     allow_partial=False, trade_date=day)


def test_forward_combined_stress_is_bound_and_preserves_parent_recipe():
    from scripts.olr_prospective_validation import scenario_configuration
    config = {"capacity_scenario": {"name": "research-baseline", "participation": .01}}
    recipe = {"olr.cost.slippage_bps": 5.0, "olr.robustness.slippage_bps": 5.0, "olr.afternoon.top_n": 4}
    scenario = {"capacity_scenario": {"name": "combined-stress", "participation": .005}, "additional_slippage_bps": 10.0}
    stressed, mutations = scenario_configuration(config, recipe, scenario)
    assert stressed["capacity_scenario"]["participation"] == .005
    assert mutations["olr.cost.slippage_bps"] == mutations["olr.robustness.slippage_bps"] == 15.0
    assert mutations["olr.afternoon.top_n"] == 4
    assert config["capacity_scenario"]["participation"] == .01
    assert recipe["olr.cost.slippage_bps"] == 5.0
