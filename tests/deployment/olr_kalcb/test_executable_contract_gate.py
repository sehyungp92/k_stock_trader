from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

REQUIREMENT_TO_TESTS = {
    "artifact/config hash binding, including empty/no-candidate artifacts": (
        ("tests/backtests/strategies/test_olr_kalcb_live_replay_artifact_parity.py", "test_kalcb_live_generator_uses_replay_config_mutation_hash"),
        ("tests/backtests/strategies/test_olr_kalcb_live_replay_artifact_parity.py", "test_olr_afternoon_artifact_hash_changes_when_final_config_fingerprint_changes"),
        ("tests/strategy_kalcb/test_research.py", "test_empty_finalized_kalcb_hash_binds_finalization_metadata"),
        ("tests/deployment/olr_kalcb/test_readiness_and_coordinator.py", "test_runtime_execution_blocks_artifact_generated_from_different_config"),
        ("tests/deployment/olr_kalcb/test_readiness_and_coordinator.py", "test_runtime_session_api_has_no_default_config_escape_hatch"),
        ("tests/deployment/olr_kalcb/test_readiness_and_coordinator.py", "test_coordinator_default_configs_are_artifact_only_and_explicit"),
        ("tests/deployment/olr_kalcb/test_readiness_and_coordinator.py", "test_runtime_kalcb_config_binding_rejects_sector_map_metadata_drift"),
    ),
    "volatile broker/path fields are normalized out of canonical hashes": (
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_session_hashes_ignore_paper_vs_dry_run_mode_fields"),
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_session_hashes_ignore_broker_only_order_ids"),
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_session_hashes_ignore_original_order_ids_and_artifact_paths"),
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_session_hashes_ignore_local_strategy_config_paths"),
    ),
    "equal-timestamp runtime events replay in captured order": (
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_replay_input_loader_preserves_runtime_event_input_order_for_equal_timestamps"),
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_runtime_event_inputs_are_sequenced_and_reference_market_bar_rows"),
    ),
    "promotional replay rejects missing/default config, state, account, and event-source inputs": (
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_replay_input_loader_requires_explicit_config_account_and_positions"),
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_offline_replay_requires_driver_runtime_event_inputs"),
        ("tests/deployment/olr_kalcb/test_session_driver.py", "test_driver_blocks_unmapped_replay_fill"),
    ),
    "self-contained session evidence requires staged snapshots and matching artifact-generation rows": (
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_offline_replay_requires_complete_artifact_evidence"),
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_offline_replay_requires_artifact_generation_to_match_staged_snapshot"),
    ),
    "copied or incomplete offline streams cannot pass paper gate": (
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_replay_paper_session_blocks_copied_offline_streams_without_engine_manifest"),
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_replay_paper_session_rejects_incomplete_offline_rebuild"),
    ),
    "behavior-critical stream perturbations fail the gate": (
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_replay_paper_session_flags_behavior_stream_perturbation"),
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_offline_replay_uses_market_bars_parquet_as_bar_authority"),
    ),
    "incomplete bars are rejected in every promotional mode before engines run": (
        ("tests/deployment/olr_kalcb/test_session_driver.py", "test_driver_rejects_incomplete_bar_for_paper_mode_before_engine"),
        ("tests/deployment/olr_kalcb/test_session_driver.py", "test_driver_rejects_incomplete_bar_for_replay_mode"),
    ),
    "automatic closeout seals expected hashes without manual manifest rewriting": (
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_session_recorder_close_session_seals_expected_hashes"),
        ("tests/deployment/olr_kalcb/test_readiness_and_coordinator.py", "test_runtime_plan_close_session_uses_session_recorder"),
    ),
    "paper gate requires sealed closeout and complete expected hash groups": (
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_replay_paper_session_rejects_manual_hashes_without_sealed_closeout"),
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_paper_gate_rejects_sealed_manifest_with_missing_expected_hash_group"),
    ),
    "hash-contract-only CLI mode is non-promotional and non-successful without explicit debug opt-in": (
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_replay_cli_hash_contract_only_is_non_success_without_debug_opt_in"),
    ),
    "portfolio priority is applied to the full routed action batch": (
        ("tests/deployment/olr_kalcb/test_portfolio_and_replay.py", "test_router_applies_portfolio_priority_before_batch_arrival_order"),
        ("tests/deployment/olr_kalcb/test_session_driver.py", "test_runtime_plan_routes_market_bar_as_one_combined_priority_batch"),
    ),
    "startup OMS working-order snapshots replay through the full paper gate": (
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_offline_replay_preserves_startup_working_order_snapshot_parity"),
    ),
    "KIS resource plan binds candidate surfaces, phased leases, mode, routing, and session evidence": (
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_resource_plan_models_candidate_surfaces_by_phase_without_false_ws_conflict"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_resource_plan_fails_mode_mismatch_and_oversized_olr_acquisition"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_resource_plan_hash_changes_when_candidate_counts_change"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_resource_plan_router_restricts_kalcb_to_active_and_routes_olr_final_only_after_final_ready"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_prepare_runtime_session_writes_hash_bound_resource_plan"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_combined_session_can_start_kalcb_when_olr_final_is_not_ready_yet"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_combined_session_enables_olr_final_without_recreating_runtime"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_operator_enables_olr_final_and_refreshes_coordinator_plan"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_market_data_coordinator_owns_ws_leases_and_subscription_evidence"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_market_data_coordinator_subscribes_dynamic_kalcb_management_symbols"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_market_data_coordinator_derives_dynamic_symbols_from_runtime_plan"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_market_data_coordinator_subscribes_olr_final_orderable_symbols_only"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_market_data_coordinator_keeps_shared_symbol_subscription_until_last_lease_releases"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_ws_limit_constants_have_single_authority"),
        ("tests/deployment/olr_kalcb/test_kis_resource_plan.py", "test_paper_gate_blocks_missing_required_resource_plan"),
    ),
    "one golden driver-owned KALCB+OLR paper session covers entry, exit, blocked action, fill, expiry, timer, and no-action": (
        ("tests/deployment/olr_kalcb/test_offline_replay_engine.py", "test_golden_runtime_replay_contract_covers_event_source_paths"),
    ),
}


def test_executable_contract_gate_requirement_to_test_mapping_is_current():
    missing: list[str] = []
    for requirement, tests in REQUIREMENT_TO_TESTS.items():
        assert tests, requirement
        for relative_path, test_name in tests:
            if test_name not in _test_names(ROOT / relative_path):
                missing.append(f"{requirement}: {relative_path}::{test_name}")

    assert missing == []


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    }
