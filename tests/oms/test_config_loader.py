from __future__ import annotations

import json
from types import SimpleNamespace

from instrumentation.src.deployment_logger import DeploymentLogger
from instrumentation.src.lineage import LineageContext, context_from_env, stable_hash
from instrumentation.src.oms_exporter import OMSEventEmitter
from instrumentation.src.runtime_lineage import write_runtime_deployment_lineage
from oms.config_loader import (
    configured_active_strategy_ids,
    effective_risk_config_payload,
    load_oms_config,
    load_oms_sector_map,
    missing_strategy_budgets,
)
from oms.risk import RiskConfig
from oms.server import IntentRequest, _apply_config_snapshot_lineage, _runtime_deployment_lineage


def test_effective_risk_config_payload_uses_oms_yaml_limits():
    payload = effective_risk_config_payload(load_oms_config())

    assert payload["daily_loss_warn_pct"] == 0.03
    assert payload["daily_loss_halt_pct"] == 0.05
    assert payload["max_positions_count"] == 15
    assert configured_active_strategy_ids(load_oms_config()) == ("KALCB", "OLR")
    assert payload["strategy_budgets"]["KALCB"]["max_positions"] == 4
    assert payload["strategy_budgets"]["OLR"]["max_positions"] == 4
    assert payload["strategy_budgets"]["PCIM"]["max_positions"] == 8
    assert payload["strategy_budgets"]["PCIM"]["max_risk_pct"] == 0.025
    assert payload["unknown_sector_policy"] == "block"
    assert payload["require_durable_stops"] is True
    assert payload["default_stop_protection_mode"] == "oms_watcher"
    assert payload["allow_synthetic_stop_only"] is False


def test_oms_config_has_budgets_for_active_strategies():
    config = load_oms_config()
    active = configured_active_strategy_ids(config)

    assert missing_strategy_budgets(RiskConfig(strategy_budgets=config["strategy_budgets"]), active) == ()


def test_oms_sector_map_loads_approved_default():
    sector_map, source = load_oms_sector_map(load_oms_config())

    assert source is not None
    assert source.name == "sector_map.yaml"
    assert sector_map["005930"] == "SEMICONDUCTORS"


def test_intent_request_preserves_caller_idempotency_identifiers():
    request = IntentRequest(
        intent_id="intent-client",
        idempotency_key="idem-client",
        intent_type="ENTER",
        strategy_id="KALCB",
        symbol="005930",
        desired_qty=10,
    )

    assert request.intent_id == "intent-client"
    assert request.idempotency_key == "idem-client"


def test_oms_server_applies_config_snapshot_lineage_to_emitter(tmp_path):
    emitter = OMSEventEmitter(tmp_path, lineage=LineageContext(code_sha="abc123"))

    _apply_config_snapshot_lineage(
        emitter,
        {
            "payload": {
                "deployment_id": "deploy-oms",
                "strategy_version": "strategy-oms",
                "config_version": "cfg-oms",
                "portfolio_config_version": "portfolio-oms",
                "risk_config_version": "risk-oms",
                "allocation_version": "allocation-oms",
                "strategy_registry_version": "registry-oms",
                "kis_resource_plan_hash": "plan-oms",
            }
        },
    )

    assert emitter.lineage.deployment_id == "deploy-oms"
    assert emitter.lineage.strategy_version == "strategy-oms"
    assert emitter.lineage.config_version == "cfg-oms"
    assert emitter.lineage.portfolio_config_version == "portfolio-oms"
    assert emitter.lineage.risk_config_version == "risk-oms"
    assert emitter.lineage.allocation_version == "allocation-oms"
    assert emitter.lineage.strategy_registry_version == "registry-oms"
    assert emitter.lineage.kis_resource_plan_hash == "plan-oms"

    emitter.emit_risk_decision(
        SimpleNamespace(
            intent_id="intent-oms",
            idempotency_key="idem-oms",
            strategy_id="KALCB",
            symbol="005930",
            metadata={},
        ),
        SimpleNamespace(decision="REJECT", reason="unit", trace=[{"rule": "unit", "decision": "REJECT"}]),
    )
    risk_path = next((tmp_path / "risk_decisions").glob("*.jsonl"))
    row = json.loads(risk_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["deployment_id"] == "deploy-oms"
    assert row["strategy_version"] == "strategy-oms"
    assert row["config_version"] == "cfg-oms"
    assert row["portfolio_config_version"] == "portfolio-oms"
    assert row["risk_config_version"] == "risk-oms"
    assert row["allocation_version"] == "allocation-oms"
    assert row["lineage_gap"] is False


def test_oms_runtime_lineage_handoff_takes_precedence_over_env_and_local_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_ID", "env-deploy")
    monkeypatch.setenv("STRATEGY_VERSION", "env-strategy")
    monkeypatch.setenv("CONFIG_VERSION", "env-config")
    monkeypatch.setenv("PORTFOLIO_CONFIG_VERSION", "env-portfolio")
    monkeypatch.setenv("RISK_CONFIG_VERSION", "env-risk")
    monkeypatch.setenv("ALLOCATION_VERSION", "env-allocation")
    monkeypatch.setenv("STRATEGY_REGISTRY_VERSION", "env-registry")
    monkeypatch.setenv("CODE_SHA", "env-code")
    write_runtime_deployment_lineage(
        tmp_path,
        {
            "deployment_id": "deploy-runtime",
            "strategy_version": "strategy-runtime",
            "config_version": "cfg-runtime",
            "portfolio_config_version": "portfolio-runtime",
            "risk_config_version": "risk-runtime",
            "allocation_version": "allocation-runtime",
            "strategy_registry_version": "registry-runtime",
            "kis_resource_plan_hash": "plan-runtime",
            "code_sha": "abc123",
            "portfolio_id": "portfolio-runtime",
            "account_alias": "account-runtime",
        },
    )
    lineage = _runtime_deployment_lineage(context_from_env(data_source_id="postgres_oms"), tmp_path)
    emitter = OMSEventEmitter(tmp_path / "events", lineage=lineage)

    _apply_config_snapshot_lineage(
        emitter,
        {
            "payload": {
                "deployment_id": "deploy-oms-local",
                "config_version": "cfg-oms-local",
                "portfolio_config_version": "portfolio-oms-local",
                "risk_config_version": "risk-oms-local",
                "allocation_version": "allocation-oms-local",
                "strategy_registry_version": "registry-oms-local",
                "kis_resource_plan_hash": "plan-oms-local",
            }
        },
    )
    emitter.emit_risk_decision(
        SimpleNamespace(
            intent_id="intent-runtime",
            idempotency_key="idem-runtime",
            strategy_id="KALCB",
            symbol="005930",
            metadata={},
        ),
        SimpleNamespace(decision="REJECT", reason="unit", trace=[{"rule": "unit", "decision": "REJECT"}]),
    )

    row = json.loads(next((tmp_path / "events" / "risk_decisions").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert row["deployment_id"] == "deploy-runtime"
    assert row["strategy_version"] == "strategy-runtime"
    assert row["config_version"] == "cfg-runtime"
    assert row["portfolio_config_version"] == "portfolio-runtime"
    assert row["risk_config_version"] == "risk-runtime"
    assert row["allocation_version"] == "allocation-runtime"
    assert row["strategy_registry_version"] == "registry-runtime"
    assert row["kis_resource_plan_hash"] == "plan-runtime"
    assert row["portfolio_id"] == "portfolio-runtime"
    assert row["account_alias"] == "account-runtime"
    assert row["code_sha"] == "abc123"
    assert row["lineage_gap"] is False


def test_oms_config_snapshot_preserves_runtime_lineage_versions(tmp_path):
    lineage = LineageContext(
        deployment_id="deploy-runtime",
        code_sha="abc123",
        strategy_version="strategy-runtime",
        config_version="cfg-runtime",
        portfolio_config_version="portfolio-runtime",
        risk_config_version="risk-runtime",
        allocation_version="allocation-runtime",
        strategy_registry_version="registry-runtime",
        kis_resource_plan_hash="plan-runtime",
    )

    snapshot_event = DeploymentLogger(tmp_path, lineage=lineage).emit_config_snapshot(
        risk_config={},
        strategy_registry={"strategy_ids": ["KALCB", "OLR"], "producer": "oms_server"},
    )

    assert snapshot_event is not None
    row = json.loads(next((tmp_path / "config_snapshots").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert row["deployment_id"] == "deploy-runtime"
    assert row["config_version"] == "cfg-runtime"
    assert row["portfolio_config_version"] == "portfolio-runtime"
    assert row["risk_config_version"] == "risk-runtime"
    assert row["allocation_version"] == "allocation-runtime"
    assert row["strategy_registry_version"] == "registry-runtime"
    assert row["kis_resource_plan_hash"] == "plan-runtime"
    assert row["payload"]["deployment_id"] == "deploy-runtime"
    assert row["payload"]["config_version"] == "cfg-runtime"
    assert row["payload"]["risk_config_version"] == "risk-runtime"
    assert row["payload"]["computed_versions"]["config_version"] == stable_hash({})
    assert row["payload"]["computed_versions"]["risk_config_version"] == stable_hash({})
    assert row["lineage_gap"] is False
