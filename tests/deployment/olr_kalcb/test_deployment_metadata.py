from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import pytest

from deployment.olr_kalcb.deployment_metadata import (
    DeploymentMetadataError,
    build_deployment_metadata,
    emit_deployment_metadata,
)
from deployment.olr_kalcb.hashing import file_sha256
from instrumentation.src.runtime_lineage import runtime_versions_from_manifest


def test_build_deployment_metadata_matches_approval_contract(tmp_path):
    contract = tmp_path / "trading_assistant_backtest" / "contracts" / "k_stock_olr_kalcb" / "strategy_plugin_contract.json"
    contract.parent.mkdir(parents=True)
    contract.write_text('{"schema": "unit"}\n', encoding="utf-8")

    strategy_configs = {"KALCB": {"threshold": 1.2}, "OLR": {"enabled": True}}
    portfolio_policy_config = {"max_gross_exposure_pct": 0.5}
    strategy_artifacts = {
        "KALCB": {
            "artifact_hash": "kalcb-artifact",
            "artifact_stage": "daily_finalized_candidate",
            "source_fingerprint": "kalcb-source",
        },
        "OLR": {
            "artifact_hash": "olr-artifact",
            "artifact_stage": "final_afternoon_1430",
            "source_fingerprint": "olr-source",
        },
    }
    initial_positions = {
        "005930": {
            "real_qty": 2,
            "avg_price": 100.0,
            "allocations": {"KALCB": {"qty": 2, "cost_basis": 100.0}},
        }
    }
    metadata = build_deployment_metadata(
        repo_root=tmp_path,
        contract_path=contract,
        mode="live",
        strategy_ids=("kalcb", " OLR "),
        strategy_configs=strategy_configs,
        portfolio_policy_config=portfolio_policy_config,
        strategy_artifacts=strategy_artifacts,
        initial_positions=initial_positions,
        kis_resource_plan_hash="plan-unit",
        deployment_id="deploy-runtime-unit",
        runtime_started_at_utc=datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc),
        runtime_entrypoint="unit:entrypoint",
        emission_environment=" production_vps ",
        source_control={
            "repo_url": "https://github.com/sehyungp92/k_stock_trader.git",
            "commit_sha": "a" * 40,
            "worktree_clean": True,
        },
    )
    runtime_versions = runtime_versions_from_manifest(
        {
            "mode": "live",
            "strategy_ids": ["KALCB", "OLR"],
            "strategy_configs": strategy_configs,
            "portfolio_policy_config": portfolio_policy_config,
            "strategy_artifacts": strategy_artifacts,
            "initial_positions": initial_positions,
            "kis_resource_plan_hash": "plan-unit",
        }
    )
    staged_runtime_versions = runtime_versions_from_manifest(
        {
            "mode": "live",
            "strategy_ids": ["KALCB", "OLR"],
            "strategy_configs": strategy_configs,
            "portfolio_policy_config": portfolio_policy_config,
            "staged_artifacts": [
                {"strategy_id": strategy_id, **artifact}
                for strategy_id, artifact in strategy_artifacts.items()
            ],
            "initial_positions": initial_positions,
            "kis_resource_plan_hash": "plan-unit",
        }
    )

    assert metadata["metadata_source"] == "live_bot_runtime_deployment_metadata_v1"
    assert metadata["emission_environment"] == "production_vps"
    assert metadata["strategy_ids"] == ["KALCB", "OLR"]
    assert metadata["repo_url"] == metadata["source_control_origin"]
    assert metadata["deployed_commit_sha"] == metadata["source_control_commit_sha"] == "a" * 40
    assert metadata["source_control_worktree_clean"] is True
    assert metadata["bot_id"] == "k_stock_trader"
    assert metadata["portfolio_id"] == "olr_kalcb"
    assert metadata["strategy_id"] == "OLR_KALCB"
    assert metadata["telemetry_schema_version"] == "olr_kalcb_decision_stream_v1"
    assert metadata["strategy_plugin_contract_path"].endswith("strategy_plugin_contract.json")
    assert metadata["strategy_plugin_contract_hash"] == file_sha256(contract)
    assert metadata["deployment_id"] == "deploy-runtime-unit"
    assert metadata["strategy_version"] == runtime_versions["strategy_version"]
    assert metadata["config_version"] == runtime_versions["config_version"]
    assert metadata["config_hash"] == runtime_versions["config_version"]
    assert metadata["portfolio_config_version"] == runtime_versions["portfolio_config_version"]
    assert metadata["risk_config_version"] == runtime_versions["risk_config_version"]
    assert metadata["allocation_version"] == runtime_versions["allocation_version"]
    assert metadata["strategy_registry_version"] == runtime_versions["strategy_registry_version"]
    assert metadata["strategy_registry_version"] == staged_runtime_versions["strategy_registry_version"]
    assert metadata["live_runtime_started_at_utc"] == "2026-02-02T00:00:00Z"
    assert metadata["runtime_entrypoint"] == "unit:entrypoint"
    assert metadata["dry_run"] is False
    assert metadata["strategy_artifacts"]["KALCB"]["artifact_hash"] == "kalcb-artifact"
    assert metadata["kis_resource_plan_hash"] == "plan-unit"


def test_build_deployment_metadata_rejects_dirty_or_local_source(tmp_path):
    contract = tmp_path / "strategy_plugin_contract.json"
    contract.write_text("{}", encoding="utf-8")

    base = {
        "repo_root": tmp_path,
        "contract_path": contract,
        "mode": "paper",
        "strategy_ids": ("KALCB", "OLR"),
        "source_control": {
            "repo_url": "https://github.com/sehyungp92/k_stock_trader.git",
            "commit_sha": "b" * 40,
            "worktree_clean": True,
        },
    }

    with pytest.raises(DeploymentMetadataError, match="worktree must be clean"):
        build_deployment_metadata(
            **{
                **base,
                "source_control": {**base["source_control"], "worktree_clean": False},
            }
        )

    with pytest.raises(DeploymentMetadataError, match="real remote"):
        build_deployment_metadata(
            **{
                **base,
                "source_control": {**base["source_control"], "repo_url": "local://checkout"},
            }
        )

    with pytest.raises(DeploymentMetadataError, match="full git object id"):
        build_deployment_metadata(
            **{
                **base,
                "source_control": {**base["source_control"], "commit_sha": "b" * 12},
            }
        )


def test_emit_deployment_metadata_can_refresh_its_own_output_inside_repo(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "unit@example.com")
    _git(tmp_path, "config", "user.name", "Unit Test")
    _git(tmp_path, "remote", "add", "origin", "https://github.com/sehyungp92/k_stock_trader.git")
    contract = tmp_path / "trading_assistant_backtest" / "contracts" / "k_stock_olr_kalcb" / "strategy_plugin_contract.json"
    contract.parent.mkdir(parents=True)
    contract.write_text('{"schema": "unit"}\n', encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "seed contract")

    output = contract.with_name("deployment_metadata.json")
    first = emit_deployment_metadata(
        output,
        repo_root=tmp_path,
        contract_path=contract,
        mode="paper",
        strategy_ids=("KALCB", "OLR"),
        runtime_started_at_utc="2026-02-02T00:00:00Z",
    )
    second = emit_deployment_metadata(
        output,
        repo_root=tmp_path,
        contract_path=contract,
        mode="paper",
        strategy_ids=("KALCB", "OLR"),
        runtime_started_at_utc="2026-02-02T00:00:00Z",
    )

    assert first["source_control_worktree_clean"] is True
    assert second["source_control_worktree_clean"] is True
    assert second["runtime_instance_id"] == first["runtime_instance_id"]
    assert output.is_file()


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
