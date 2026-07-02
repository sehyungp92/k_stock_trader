from __future__ import annotations

import json
from pathlib import Path

from deployment.olr_kalcb.bridge_contract import (
    CONTRACT_SCHEMA_VERSION,
    REQUIRED_JOIN_FIELDS,
    REQUIRED_PAYLOAD_IDENTITY_FIELDS,
    SNAPSHOT_EVENT_REQUIREMENTS,
    build_strategy_plugin_contract,
)
from deployment.olr_kalcb.deployment_metadata import DEFAULT_CONTRACT_PATH
from oms.config_loader import effective_risk_config_payload, load_oms_config


ROOT = Path(__file__).resolve().parents[3]


def test_checked_in_strategy_plugin_contract_matches_generator():
    path = ROOT / DEFAULT_CONTRACT_PATH

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == build_strategy_plugin_contract(ROOT)


def test_strategy_plugin_contract_covers_olr_kalcb_bridge_surface():
    contract = build_strategy_plugin_contract(ROOT)

    assert contract["schema_version"] == CONTRACT_SCHEMA_VERSION
    assert contract["bot_id"] == "k_stock_trader"
    assert contract["portfolio_id"] == "olr_kalcb"
    assert contract["strategy_ids"] == ["KALCB", "OLR"]
    assert set(contract["strategies"]) == {"KALCB", "OLR"}

    kalcb = contract["strategies"]["KALCB"]
    olr = contract["strategies"]["OLR"]
    assert kalcb["config_source"] == "config/kalcb.yaml"
    assert olr["config_source"] == "strategy_olr/config.py"
    assert _parameter(kalcb, "entry_plan_mode")["editable_paths"]
    assert "kalcb.entry.plan_mode" in _parameter(kalcb, "entry_plan_mode")["editable_paths"]
    assert "olr.afternoon.top_n" in _parameter(olr, "afternoon_top_n")["editable_paths"]

    resources = contract["shared_editable_resources"]
    assert set(resources) == {
        "deployment_universe",
        "oms_risk_policy",
        "portfolio_policy",
        "sector_map",
    }
    assert resources["deployment_universe"]["symbol_count"] == 103
    assert resources["deployment_universe"]["symbols_sha256"] == resources["deployment_universe"]["computed_symbols_sha256"]
    assert resources["portfolio_policy"]["effective_policy"]["strategy_priority"] == ["KALCB", "OLR"]
    assert "risk.max_gross_exposure_pct" in resources["oms_risk_policy"]["editable_paths"]
    assert resources["oms_risk_policy"]["effective_risk_config"] == effective_risk_config_payload(
        load_oms_config()
    )

    bridge = contract["assistant_bridge"]
    assert set(REQUIRED_PAYLOAD_IDENTITY_FIELDS).issubset(bridge["event_envelope"]["required_payload_identity_fields"])
    assert "assistant_strategy_id" in bridge["event_envelope"]["conditional_payload_identity_fields"]
    assert set(REQUIRED_JOIN_FIELDS).issubset(bridge["event_envelope"]["required_join_fields"])
    for event_type, fields in SNAPSHOT_EVENT_REQUIREMENTS.items():
        assert set(fields).issubset(bridge["required_snapshots"][event_type]["required_payload_fields"])
    bridge_event_types = (
        "decision_event",
        "strategy_action",
        "portfolio_rule",
        "risk_decision",
        "oms_intent",
        "order",
        "fill",
        "deployment",
        "config_snapshot",
        "resource_plan",
    )
    for event_type in bridge_event_types:
        assert event_type in bridge["event_streams"]


def _parameter(strategy_contract: dict, field: str) -> dict:
    for row in strategy_contract["editable_parameters"]:
        if row["canonical_field"] == field:
            return row
    raise AssertionError(f"missing parameter {field}")
