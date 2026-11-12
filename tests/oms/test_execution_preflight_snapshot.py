from datetime import datetime, timedelta, timezone
from copy import deepcopy

import pytest

from oms.execution_preflight import snapshot_requirements


def fixture():
    now = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    snapshot = {"oms_id": "paper", "account_alias": "fixture", "as_of": now.isoformat(),
                "positions": {"005930": 10}, "cost_basis": {"005930": 106}}
    risk = {"oms_id": "paper", "account_alias": "fixture", "required_stop_strategies": ["KALCB"],
            "enabled_strategies": ["KALCB"]}
    kwargs = {"oms_id": "paper", "now": now, "broker_positions": {"005930": 10},
        "allocations": [{"strategy_id": "KALCB", "symbol": "005930", "qty": 10, "cost_basis": 106}],
        "stops": [{"strategy_id": "KALCB", "symbol": "005930", "qty": 10, "status": "ACTIVE"}],
        "watermarks": {"KALCB": 7}, "checkpoints": [{"strategy_id": "KALCB", "meta": {"oms_feedback_cursor": 7}}]}
    return snapshot, risk, kwargs


def test_fresh_snapshot_reconciles_basis_protection_and_cursors():
    snapshot, risk, kwargs = fixture()
    assert snapshot_requirements(snapshot, risk, **kwargs) == []


@pytest.mark.parametrize("failure", ["stale", "future", "naive", "account", "basis", "missing_stop", "checkpoint", "tolerance"])
def test_cutover_rejects_unresolved_boundary(failure):
    snapshot, risk, kwargs = fixture()
    if failure in {"stale", "future"}:
        snapshot["as_of"] = (kwargs["now"] + timedelta(seconds=-301 if failure == "stale" else 1)).isoformat()
    elif failure == "naive":
        snapshot["as_of"] = "2026-10-03T12:00:00"
    elif failure == "account":
        snapshot["account_alias"] = "another-account"
    elif failure == "basis":
        snapshot["cost_basis"]["005930"] = 999
    elif failure == "missing_stop":
        kwargs["stops"] = []
    elif failure == "checkpoint":
        kwargs["checkpoints"] = []
    else:
        risk["cost_basis_tolerance_krw"] = 10000
    assert snapshot_requirements(snapshot, risk, **kwargs)


def test_disabled_stop_policy_is_explicit_and_does_not_require_protection():
    snapshot, risk, kwargs = fixture()
    risk["required_stop_strategies"] = []
    kwargs["stops"] = []
    assert snapshot_requirements(snapshot, risk, **kwargs) == []


def test_collected_snapshot_requires_matching_actual_account_fingerprint():
    snapshot, risk, kwargs = fixture()
    snapshot.update(source="KISExecutionAdapter", queries_complete=True, mode="paper", account_fingerprint="account-a")
    assert "broker_snapshot_route_or_account_fingerprint_mismatch" in snapshot_requirements(snapshot, risk, **kwargs)
    risk["account_fingerprint"] = "account-a"
    assert snapshot_requirements(snapshot, risk, **kwargs) == []
    snapshot["queries_complete"] = False
    assert "broker_snapshot_route_or_account_fingerprint_mismatch" in snapshot_requirements(snapshot, risk, **kwargs)
