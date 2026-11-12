import json

import pytest

from backtests.strategies.olr import exported_diagnostics as reports


def fixture_export(tmp_path):
    directory = tmp_path / "training"
    directory.mkdir()
    expected = {"config": {"initial_equity": 10000000}, "mutations": {"olr.cost.slippage_bps": 5}}
    plan = {"output": str(tmp_path), "cases": {"training": expected}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    for name in reports.REQUIRED_EXPORTS:
        (directory / name).write_text("[]")
    (directory / "request.json").write_text(json.dumps(expected))
    metrics = {"audit_pass": False, "audit_receipt": {"status": "failed"}}
    (directory / "metrics.json").write_text(json.dumps(metrics))
    (directory / "audit_receipt.json").write_text(json.dumps(metrics["audit_receipt"]))
    reseal(directory, plan_path)
    return directory, plan_path, plan, expected


def reseal(directory, plan_path):
    seal = {"case": "training", "plan_sha256": reports.digest(plan_path),
        "outputs": {name: reports.digest(directory / name) for name in reports.REQUIRED_EXPORTS}}
    (directory / "export_receipt.json").write_text(json.dumps(seal))


def test_changed_ledger_bytes_are_rejected_before_reporting(tmp_path):
    directory, path, plan, expected = fixture_export(tmp_path)
    (directory / "ledger.jsonl.gz").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        reports.load_sealed_case(path, plan, "training", expected)


def test_resealed_wrong_scenario_cannot_supply_diagnostics(tmp_path):
    directory, path, plan, expected = fixture_export(tmp_path)
    wrong = {**expected, "mutations": {"olr.cost.slippage_bps": 0}}
    (directory / "request.json").write_text(json.dumps(wrong))
    reseal(directory, path)
    with pytest.raises(ValueError, match="recipe or scenario"):
        reports.load_sealed_case(path, plan, "training", expected)


def test_missing_evidence_group_cannot_be_declared_complete(tmp_path):
    directory, path, plan, expected = fixture_export(tmp_path)
    seal = reports.read(directory / "export_receipt.json")
    del seal["outputs"]["equity.jsonl.gz"]
    (directory / "export_receipt.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="incomplete"):
        reports.load_sealed_case(path, plan, "training", expected)


def test_sealed_export_still_requires_the_native_passing_receipt(tmp_path):
    _, path, plan, expected = fixture_export(tmp_path)
    with pytest.raises(ValueError, match="request-bound"):
        reports.load_sealed_case(path, plan, "training", expected)


def test_export_loading_is_observational_and_uses_exact_request(tmp_path, monkeypatch):
    directory, path, plan, expected = fixture_export(tmp_path)
    calls = []
    # Isolate export loading from audit validation; the preceding test exercises
    # the real failed-receipt path without replacing the validator.
    monkeypatch.setattr(reports, "require_bound_result", lambda result, config, mutations:
        calls.append((config, mutations)))
    before = {p.name: reports.digest(p) for p in directory.iterdir()}
    case = reports.load_sealed_case(path, plan, "training", expected)
    assert calls == [(expected["config"], expected["mutations"])]
    assert case["metrics"]["audit_pass"] is False
    assert before == {p.name: reports.digest(p) for p in directory.iterdir()}
