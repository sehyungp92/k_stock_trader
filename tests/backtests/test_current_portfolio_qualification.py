import json

import pytest

from backtests.strategies.kalcb.executed_value import digest
from scripts.qualify_portfolio_synergy_study import validate_case


@pytest.fixture
def qualified_case(tmp_path):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    study = tmp_path / "study.json"
    recipe = {"policy": {"funding_mode": "native_buying_power"}}
    save(study, {"source_generation_hash": "frozen", "candidates": {"selected": recipe}})
    output = tmp_path / "qualified"
    identity = {"study_hash": digest(study), "recipe": recipe, "scenario": "baseline",
        "no_upward_compounding": False, "absolute_caps": False, "standalone": None}
    result = {"source_generation_hash": "frozen", "recipe": "selected", "scenario": "baseline",
        "execution_validity_pass": True, "audit_status": "passed", "intraday_residual_sessions": 0,
        "terminal_inventory": {}, "terminal_pending": 0, "return_pct": 12., "drawdown_pct": 3.,
        "trades": 50, "active_days": 25, "trades_per_21_sessions": 21., "strategy_counts": {},
        "unique_blocked_orders": 2, "unique_resized_orders": 3}
    save(output / "result.json", result)
    save(tmp_path / "cases/selected/baseline/result.json", result)
    save(output / "execution_audit.json", {"audit_pass": True, "audit_status": "passed",
        "audit_receipt": {"identity": identity, "source_generation_hash": "frozen"}})
    save(output / "request.json", {"study_hash": digest(study), "recipe": recipe,
        "arguments": {"recipe": "selected", "scenario": "baseline", "full_audit": True,
            "no_upward_compounding": False, "absolute_caps": False, "standalone": None,
            "canonical_costs": False}})
    save(output / "completion.json", {"complete": True, "source_generation_hash": "frozen",
        "result_sha256": digest(output / "result.json"), "audit_sha256": digest(output / "execution_audit.json")})
    return output, study, save


def test_qualification_binds_execution_and_screening(qualified_case):
    output, study, _ = qualified_case
    assert validate_case(output, study, "selected", "baseline", False)["audit_status"] == "passed"


@pytest.mark.parametrize("change", ["unverified", "screening_drift", "wrong_recipe", "capped", "tampered"])
def test_qualification_rejects_incomplete_or_mismatched_evidence(qualified_case, change):
    output, study, save = qualified_case
    if change in {"wrong_recipe", "capped"}:
        path = output / "request.json"
        value = json.loads(path.read_text())
        value["arguments"]["recipe" if change == "wrong_recipe" else "no_upward_compounding"] = "other" if change == "wrong_recipe" else True
        save(path, value)
    elif change == "screening_drift":
        path = study.parent / "cases/selected/baseline/result.json"
        value = json.loads(path.read_text())
        value["return_pct"] += .01
        save(path, value)
    else:
        path = output / "execution_audit.json"
        value = json.loads(path.read_text())
        value["audit_status"] = "unverified"
        save(path, value)
        if change == "unverified":
            receipt = json.loads((output / "completion.json").read_text())
            receipt["audit_sha256"] = digest(path)
            save(output / "completion.json", receipt)
    with pytest.raises(ValueError):
        validate_case(output, study, "selected", "baseline", False)
