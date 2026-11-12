"""Acceptance cases for cross-source bases and supported capacity models."""
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest

from backtests.auto.shared.cache_keys import stable_signature
from backtests.engine.capacity import BarParticipationScenario, CAPACITY_VERSION
from backtests.engine.qualification_evidence import _references, _basis, _capacity, execution_policy
from kis_core.trading_calendar import get_trading_calendar
from tests.backtests.test_qualification_evidence import evidence_fixture
from tests.backtests.test_qualification_scope import inputs


def changed_reader(read, overrides):
    return lambda path: deepcopy(overrides[path]) if path in overrides else read(path)


def test_daily_and_intraday_must_share_verified_normalization(evidence_fixture):
    read, proof, events, broker, identity = inputs(evidence_fixture)
    path = proof["references"]["source_documents"][0]
    daily = read(path)
    daily["normalization"]["target_basis"]["id"] = "different-share-basis"
    daily["normalization"]["source_basis_id"] = "different-share-basis"
    with pytest.raises(ValueError, match="incompatible_daily_intraday_basis"):
        _references(changed_reader(read, {path: daily}), proof, get_trading_calendar(), identity, broker)
    del daily["normalization"]
    with pytest.raises(ValueError, match="missing_normalized_basis_contract"):
        _references(changed_reader(read, {path: daily}), proof, get_trading_calendar(), identity, broker)


def test_sourced_price_adjustment_rounding_and_clock_shift(evidence_fixture):
    read, proof, events, broker, identity = inputs(evidence_fixture)
    path = proof["basis"]["source_documents"][0]
    contract = read(path)
    source_path = contract["raw_paths"][0]
    raw = read(source_path)
    basis_rows = read(proof["basis"]["observations"])
    contract["timestamp_convention"] = "source_label_with_verified_shift"
    contract["normalization"]["source_basis_id"] = "unconverted-provider-basis"
    contract["normalization"]["transforms"] = [{"kind": "price_adjustment", "symbol": "005930",
        "start": "2026-01-05", "end": "2026-01-05", "price_factor": .5, "share_factor": 1,
        "timestamp_shift_seconds": 44, "price_rounding": {"mode": "floor", "quantum": 1},
        "evidence_paths": ["conversion-proof.json"]}]
    from datetime import timedelta
    for row, observation in zip(raw, basis_rows):
        for field in ("open", "high", "low", "close"):
            row[field] = row[field] * 2 + .5
        row["timestamp"] = (datetime.fromisoformat(row["timestamp"]) - timedelta(seconds=44)).isoformat()
        observation["price_factor"] = .5
    overrides = {path: contract, source_path: raw, proof["basis"]["observations"]: basis_rows,
                 "conversion-proof.json": {"notice": "test-only transformation evidence"}}
    _basis(changed_reader(read, overrides), proof, events)
    contract["normalization"]["transforms"][0]["kind"] = "split"
    with pytest.raises(ValueError, match="inconsistent_split_factors"):
        _basis(changed_reader(read, overrides), proof, events)


def test_conversion_needs_source_evidence_and_exact_arithmetic(evidence_fixture):
    read, proof, events, broker, identity = inputs(evidence_fixture)
    path = proof["basis"]["source_documents"][0]
    contract = read(path)
    contract["normalization"]["source_basis_id"] = "different"
    with pytest.raises(ValueError, match="missing_basis_conversion"):
        _basis(changed_reader(read, {path: contract}), proof, events)


def model_inputs(fixture):
    read, proof, events, broker, identity = inputs(fixture)
    broker.capacity_policy = BarParticipationScenario(.01, "predeclared-model")
    first = events[0]
    broker.fills = [SimpleNamespace(symbol=first["symbol"], timestamp=datetime.fromisoformat(first["timestamp"]),
        qty=1, metadata={"execution_phase": "open", "base_price": first["open"]})]
    capacity = proof["capacity"]
    capacity["kind"] = "calibrated_participation_model"
    capacity["registered_execution_policies"] = [{"policy": execution_policy(broker), "participation": .01}]
    capacity["model"] = {"algorithm": CAPACITY_VERSION, "capital_range": [1, 100_000_000],
        "max_participation_of_phase_prints": .05, "max_budget_exceedance_fraction": 0,
        "min_observations_per_phase": 2, "fit_end": "2025-12-29T15:30:00+09:00", "supported_phases": ["open"]}
    path = capacity["source_documents"][0]
    contract = read(path)
    contract["predictor"] = "previous_contiguous_continuous_5m_else_current_auction"
    raw_path = contract["raw_paths"][0]
    rows = [{"symbol": "005930", "timestamp": f"2025-12-{day}T10:{minute}:00+09:00", "phase": "open",
             "volume_kind": "phase_traded_shares", "volume": 1000, "predictor_volume": 1000}
            for day in (29, 30) for minute in (10, 15)]
    observed = [{**row, "source_path": raw_path, "source_index": i} for i, row in enumerate(rows)]
    overrides = {path: contract, raw_path: rows, capacity["observations"]: observed}
    return changed_reader(read, overrides), proof, broker, overrides, raw_path


def test_capacity_model_passes_with_separate_earlier_validation(evidence_fixture):
    read, proof, broker, _, _ = model_inputs(evidence_fixture)
    result = _capacity(read, proof, broker)
    assert result["basis"] == "model_conditioned"
    assert not result["exact_executable_quantity_observed"]
    assert result["validation"]["005930:open:validation"]["observations"] == 2


@pytest.mark.parametrize("defect", ["observed_shortfall", "leak", "capital", "unbound", "unsupported_phase", "unsupported_symbol", "basis"])
def test_capacity_model_cannot_self_certify(evidence_fixture, defect):
    read, proof, broker, overrides, raw_path = model_inputs(evidence_fixture)
    if defect == "observed_shortfall":
        for row in overrides[raw_path]:
            row["volume"] = 1
        for row in overrides[proof["capacity"]["observations"]]:
            row["volume"] = 1
    elif defect == "leak":
        proof["capacity"]["model"]["fit_end"] = "2026-01-06T15:30:00+09:00"
    elif defect == "capital":
        proof["capacity"]["model"]["capital_range"] = [1, 100]
    elif defect == "unbound":
        overrides[proof["capacity"]["observations"]][0]["predictor_volume"] = 2
    elif defect == "unsupported_phase":
        broker.fills[0].metadata["execution_phase"] = "auction"
    elif defect == "unsupported_symbol":
        broker.fills[0].symbol = "000660"
    else:
        path = proof["capacity"]["source_documents"][0]
        overrides[path]["normalization"]["target_basis"]["id"] = "different-share-basis"
    with pytest.raises(ValueError):
        _capacity(read, proof, broker)


def test_unrealizable_opening_proxy_fails_audit_without_rewriting_opening_fills():
    from backtests.engine.sim_broker import SimBroker
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.integrity import audit_broker
    from tests.backtests.test_strategy_integrity_repairs import ZERO, bar, buy, ts
    broker = SimBroker(10000, ZERO, capacity_policy=BarParticipationScenario(1, "adverse-volume-change"))
    drive_timestamp_batch(broker, [bar(at=ts(), volume=100)], [])
    buy(broker, qty=10)
    drive_timestamp_batch(broker, [bar(at=ts(5), volume=1)], [])
    assert [f.qty for f in broker.fills] == [10]
    assert not broker.orders
    metrics = audit_broker(broker)
    assert not metrics["execution_validity_pass"]
    assert "shared_bar_volume_exceeded" in metrics["audit_receipt"]["errors"]
