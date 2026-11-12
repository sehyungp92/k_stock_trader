import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from backtests.auto.shared.cache_keys import stable_signature
from backtests.engine.qualification_evidence import verify_qualification, _basis, _capacity, execution_policy
from backtests.engine.sim_broker import SimBroker, BrokerCosts
from tests.backtests.test_qualification_evidence import evidence_fixture


def inputs(fixture):
    generation, bundle, prefix = fixture
    read = lambda path: json.loads(generation.read(generation.root / path))
    proof = read(f"{prefix}/qualification.json")
    events = read(proof["market_events"])
    broker = SimBroker(100_000_000, BrokerCosts(commission_bps=2, slippage_bps=3))
    broker.expected_batches = [events]
    broker.qualification_snapshots = [read(p) for p in proof["references"]["snapshots"]]
    identity = {"input_manifest": generation.receipt(),
                "qualification_snapshot_hash": stable_signature(broker.qualification_snapshots),
                "configuration": {"start": "2026-01-05", "end": "2026-01-05", "symbols": ["005930"]}}
    return read, proof, events, broker, identity


def test_independent_requested_scope_detects_whole_missing_days_and_symbols(evidence_fixture):
    _, _, _, broker, identity = inputs(evidence_fixture)
    identity["configuration"].update(end="2026-01-06", symbols=["005930", "000660"])
    coverage, details = verify_qualification(broker, identity)
    assert coverage["input_provenance"] == "pass"
    assert coverage["source_completeness"] == "unverified"
    assert len(details["source_completeness"]) == 3
    assert details["required_sessions"] == 4


def test_one_common_source_accepts_distinct_consumed_scopes_with_distinct_receipts(evidence_fixture):
    _, _, events, broker, identity = inputs(evidence_fixture)
    first, a = verify_qualification(broker, identity)
    broker.expected_batches = [events[::2]]
    second, b = verify_qualification(broker, identity)
    assert set(first.values()) == {"pass"}
    assert second["input_provenance"] == second["source_completeness"] == "pass"
    assert second["replay_completeness"] == "unverified"
    assert a["manifest_hash"] == b["manifest_hash"] and a["scope_hash"] != b["scope_hash"]


@pytest.mark.parametrize("check,section", [(_basis, "basis"), (_capacity, "capacity")])
def test_unrelated_correctly_read_source_document_does_not_qualify(evidence_fixture, check, section):
    read, proof, events, broker, _ = inputs(evidence_fixture)
    def wrong(path):
        return {"unrelated": "document"} if path in proof[section]["source_documents"] else read(path)
    with pytest.raises(ValueError, match="unrelated_source_document"):
        check(wrong, proof, events if section == "basis" else broker)


def test_participation_and_phase_rows_are_bound_to_evaluated_contract(evidence_fixture):
    read, proof, events, broker, _ = inputs(evidence_fixture)
    proof["capacity"]["registered_execution_policies"][0]["participation"] = .5
    with pytest.raises(ValueError, match="participation_policy_mismatch"):
        _capacity(read, proof, broker)
    proof["capacity"]["registered_execution_policies"][0]["participation"] = 1.0
    row = events[0]
    broker.fills = [SimpleNamespace(symbol=row["symbol"], timestamp=datetime.fromisoformat(row["timestamp"]), qty=1,
        metadata={"execution_phase": "open", "base_price": row["open"]})]
    def changed(path):
        rows = read(path)
        if path == proof["capacity"]["observations"]:
            rows[0]["volume"] += 1
        return rows
    with pytest.raises(ValueError, match="phase_source_row_mismatch"):
        _capacity(changed, proof, broker)


def test_missing_request_and_wrong_candidates_never_pass(evidence_fixture):
    _, _, _, broker, identity = inputs(evidence_fixture)
    identity["configuration"] = {}
    identity["qualification_snapshot_hash"] = "wrong"
    coverage, details = verify_qualification(broker, identity)
    assert coverage["source_completeness"] != "pass"
    assert coverage["dated_references"] != "pass"
