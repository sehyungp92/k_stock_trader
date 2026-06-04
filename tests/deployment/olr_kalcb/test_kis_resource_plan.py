from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import date, datetime, time
from types import SimpleNamespace

import pytest

from deployment.olr_kalcb.hashing import canonical_json_hash
from deployment.olr_kalcb.kis_limits import KISLimitProfile, limit_profile_for_runtime
from deployment.olr_kalcb.market_data_coordinator import KISMarketDataCoordinator
from deployment.olr_kalcb.kis_resource_plan import (
    candidate_surface_for,
    build_kis_resource_plan,
    resource_plan_hash,
    target_strategy_ids_for_bar,
)
from deployment.olr_kalcb.runtime import default_session_schedule, prepare_runtime_session
from deployment.olr_kalcb.replay import replay_paper_session
from deployment.olr_kalcb.session_capture import PaperSessionRecorder
from oms_client.client import AccountState
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot
from strategy_kalcb.research import KALCB_FINAL_ARTIFACT_STAGE, candidate_config_fingerprint as kalcb_candidate_config_fingerprint
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE, OLRArtifactStore
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.research import FINAL_CANDIDATE_CONFIG_HASH_VERSION, final_candidate_config_fingerprint
from scripts import run_olr_kalcb_runtime_session as operator


def test_resource_plan_models_candidate_surfaces_by_phase_without_false_ws_conflict():
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 8})
    olr_cfg = OLRConfig()
    kalcb = _kalcb_snapshot(trade_date, kalcb_cfg, count=104, active_count=8)
    olr_stage1 = _olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, olr_cfg, count=30)
    olr_final = _olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, olr_cfg, count=8)

    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="paper",
        kalcb_config=kalcb_cfg,
        olr_config=olr_cfg,
        kalcb_snapshot=kalcb,
        olr_stage1_snapshot=olr_stage1,
        olr_final_snapshot=olr_final,
        kis_is_paper=True,
    )

    assert plan.passed is True
    assert plan.plan_hash == resource_plan_hash(plan)
    assert candidate_surface_for(plan, "KALCB").active_symbols == _symbols(8)
    assert len(candidate_surface_for(plan, "KALCB").frontier_symbols) == 104
    assert len(candidate_surface_for(plan, "OLR", OLR_STAGE1_ARTIFACT_STAGE).frontier_symbols) == 30
    assert len(candidate_surface_for(plan, "OLR", OLR_FINAL_ARTIFACT_STAGE).final_symbols) == 8
    assert len(candidate_surface_for(plan, "OLR", OLR_FINAL_ARTIFACT_STAGE).orderable_symbols) == 4
    windows = {window.name: window for window in plan.lease_windows}
    assert windows["kalcb_entry_discovery"].ws_reg_count == 8
    assert windows["olr_stage1_bar_acquisition"].ws_reg_count == 0
    assert windows["olr_stage1_bar_acquisition"].rest_call_count == 30


def test_resource_plan_fails_mode_mismatch_and_oversized_olr_acquisition():
    trade_date = date(2026, 2, 2)
    olr_cfg = OLRConfig.from_mapping({"olr.research.top_long_count": 30})
    slow_profile = KISLimitProfile(
        mode="paper",
        kis_is_paper=False,
        rest_min_interval_s=30.0,
        rest_calls_per_5m=10,
        ws_max_registrations=40,
        ws_reserved_execution_regs=1,
        order_rest_reserve_per_5m=4,
        oms_reconcile_reserve_per_5m=2,
        source="unit",
    )

    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="paper",
        olr_config=olr_cfg,
        olr_stage1_snapshot=_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, olr_cfg, count=30),
        limit_profile=slow_profile,
    )

    assert plan.passed is False
    assert any(item.startswith("kis_mode_mismatch") for item in plan.failures)
    assert any(item.startswith("olr_stage1_rest_acquisition_window_exceeded") for item in plan.failures)


def test_resource_plan_hash_changes_when_candidate_counts_change():
    trade_date = date(2026, 2, 2)
    cfg = OLRConfig()

    left = build_kis_resource_plan(
        trade_date=trade_date,
        mode="dry_run",
        olr_config=cfg,
        olr_stage1_snapshot=_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, cfg, count=3),
    )
    right = build_kis_resource_plan(
        trade_date=trade_date,
        mode="dry_run",
        olr_config=cfg,
        olr_stage1_snapshot=_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, cfg, count=4),
    )

    assert left.plan_hash != right.plan_hash


def test_resource_plan_router_restricts_kalcb_to_active_and_routes_olr_final_only_after_final_ready():
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2})
    olr_cfg = OLRConfig.from_mapping({"olr.afternoon.top_n": 2, "olr.overnight.slot_count": 1})
    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="dry_run",
        kalcb_config=kalcb_cfg,
        olr_config=olr_cfg,
        kalcb_snapshot=_kalcb_snapshot(trade_date, kalcb_cfg, count=4, active_count=2),
        olr_final_snapshot=_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, olr_cfg, count=2),
    )

    assert target_strategy_ids_for_bar(
        plan,
        symbol="000001",
        timestamp=datetime.combine(trade_date, time(9, 35), tzinfo=KST),
        available_strategy_ids=("KALCB", "OLR"),
    ) == ("KALCB",)
    assert target_strategy_ids_for_bar(
        plan,
        symbol="000003",
        timestamp=datetime.combine(trade_date, time(9, 35), tzinfo=KST),
        available_strategy_ids=("KALCB", "OLR"),
    ) == ()
    assert target_strategy_ids_for_bar(
        plan,
        symbol="100001",
        timestamp=datetime.combine(trade_date, time(14, 30), tzinfo=KST),
        available_strategy_ids=("KALCB", "OLR"),
    ) == ()
    assert target_strategy_ids_for_bar(
        plan,
        symbol="100001",
        timestamp=datetime.combine(trade_date, time(14, 36), tzinfo=KST),
        available_strategy_ids=("KALCB", "OLR"),
    ) == ("OLR",)
    assert target_strategy_ids_for_bar(
        plan,
        symbol="100002",
        timestamp=datetime.combine(trade_date, time(14, 36), tzinfo=KST),
        available_strategy_ids=("KALCB", "OLR"),
    ) == ()


def test_kalcb_frontier_branch_can_replay_from_external_completed_bars_without_kis_ws_budget_failure():
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 2,
            "kalcb.frontier.size": 4,
            "kalcb.entry.frontier_branch_universe": True,
        }
    )
    snapshot = _kalcb_snapshot(trade_date, kalcb_cfg, count=4, active_count=2)

    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="dry_run",
        kalcb_config=kalcb_cfg,
        kalcb_snapshot=snapshot,
        completed_bar_source="external_completed_bars",
    )

    assert plan.passed is True
    assert "kalcb_frontier_branch_universe_requires_explicit_ws_budget_for_all_orderable_symbols" not in plan.failures
    assert candidate_surface_for(plan, "KALCB").orderable_symbols == _symbols(4)
    assert target_strategy_ids_for_bar(
        plan,
        symbol="000003",
        timestamp=datetime.combine(trade_date, time(10, 30), tzinfo=KST),
        available_strategy_ids=("KALCB",),
    ) == ("KALCB",)


def test_kalcb_frontier_branch_still_blocks_kis_websocket_when_orderable_exceeds_ws_budget():
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 2,
            "kalcb.frontier.size": 4,
            "kalcb.entry.frontier_branch_universe": True,
        }
    )

    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="paper",
        kalcb_config=kalcb_cfg,
        kalcb_snapshot=_kalcb_snapshot(trade_date, kalcb_cfg, count=4, active_count=2),
        completed_bar_source="kis_websocket",
        kis_is_paper=True,
    )

    assert plan.passed is False
    assert "kalcb_frontier_branch_universe_requires_explicit_ws_budget_for_all_orderable_symbols" in plan.failures


def test_kalcb_resource_plan_fails_inconsistent_active_frontier_metadata():
    trade_date = date(2026, 2, 2)
    config = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2, "kalcb.selection.frontier_size": 4})
    snapshot = _kalcb_snapshot(trade_date, config, count=2, active_count=1)
    snapshot = KALCBDailySnapshot(
        trade_date=snapshot.trade_date,
        source_fingerprint=snapshot.source_fingerprint,
        generated_at=snapshot.generated_at,
        candidates=snapshot.candidates,
        metadata={**snapshot.metadata, "active_symbols": ["000001", "999999"]},
    )

    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="dry_run",
        kalcb_config=config,
        kalcb_snapshot=snapshot,
    )

    assert "kalcb_active_symbols_not_subset_of_frontier_symbols" in plan.failures


def test_kalcb_resource_plan_fails_artifact_without_active_budget_metadata():
    trade_date = date(2026, 2, 2)
    config = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2, "kalcb.selection.frontier_size": 4})
    snapshot = _kalcb_snapshot(trade_date, config, count=4, active_count=2)
    metadata = dict(snapshot.metadata)
    metadata.pop("active_symbols")
    metadata.pop("active_budget_source")
    metadata.pop("frontier_rest_budget_symbols_per_5m")
    snapshot = KALCBDailySnapshot(
        trade_date=snapshot.trade_date,
        source_fingerprint=snapshot.source_fingerprint,
        generated_at=snapshot.generated_at,
        candidates=snapshot.candidates,
        metadata=metadata,
    )

    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="dry_run",
        kalcb_config=config,
        kalcb_snapshot=snapshot,
    )

    assert "kalcb_active_symbols_missing_from_artifact_metadata" in plan.failures
    assert "kalcb_active_budget_source_missing_or_invalid" in plan.failures
    assert "kalcb_frontier_rest_budget_symbols_per_5m_missing" in plan.failures


def test_execution_schedule_is_phase_chronological():
    steps = default_session_schedule("dry_run", ("KALCB", "OLR"))
    names = [step.name for step in steps]

    assert [step.run_at_kst for step in steps] == sorted(step.run_at_kst for step in steps)
    assert names.index("kalcb_runtime_start") < names.index("olr_final_artifact")


def test_prepare_runtime_session_writes_hash_bound_resource_plan(tmp_path):
    trade_date = date(2026, 2, 2)
    sector_map = {"000001": "UNKNOWN"}
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    kalcb_cfg = KALCBConfig.from_mapping(kalcb_payload)
    config_manifest = _write_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload={})
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(trade_date, kalcb_cfg, count=1, active_count=1, payload=kalcb_payload, sector_map=sector_map)
    )
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        sector_map=sector_map,
        initial_account_state=asdict(AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)),
        initial_positions={},
    )

    manifest = json.loads((tmp_path / "session" / "session_manifest.json").read_text(encoding="utf-8"))
    resource_plan = json.loads((tmp_path / "session" / "kis_resource_plan.json").read_text(encoding="utf-8"))
    assert plan.ready_to_start is True
    assert "kis_resource_plan_loaded" not in {check.name for check in plan.preflight.failures}
    assert manifest["kis_resource_plan_hash"] == resource_plan["plan_hash"]
    assert plan.kis_resource_plan.plan_hash == resource_plan["plan_hash"]


def test_combined_session_can_start_kalcb_when_olr_final_is_not_ready_yet(tmp_path):
    trade_date = date(2026, 2, 2)
    sector_map = {"000001": "UNKNOWN"}
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    olr_payload = {"olr.research.top_long_count": 1}
    kalcb_cfg = KALCBConfig.from_mapping(kalcb_payload)
    olr_cfg = OLRConfig.from_mapping(olr_payload)
    config_manifest = _write_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload=olr_payload)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(trade_date, kalcb_cfg, count=1, active_count=1, payload=kalcb_payload, sector_map=sector_map)
    )
    OLRArtifactStore(tmp_path / "olr").save_snapshot(
        _olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, olr_cfg, count=1),
        artifact_stage=OLR_STAGE1_ARTIFACT_STAGE,
    )

    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=PaperSessionRecorder(tmp_path / "session", trade_date),
        strategy_config_source=config_manifest,
        sector_map=sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    assert plan.ready_to_start is True
    assert plan.ready_for_kalcb_start is True
    assert plan.olr_stage1_ready is True
    assert plan.ready_for_olr_start is False
    assert set(plan.drivers) == {"KALCB"}
    result = asyncio.run(plan.handle_bar(_bar("000001", trade_date, time(9, 35))))
    assert [item.strategy_id for item in result] == ["KALCB"]


def test_combined_session_enables_olr_final_without_recreating_runtime(tmp_path):
    trade_date = date(2026, 2, 2)
    sector_map = {"000001": "UNKNOWN"}
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    olr_payload = {"olr.research.top_long_count": 1, "olr.afternoon.top_n": 1, "olr.overnight.slot_count": 1}
    kalcb_cfg = KALCBConfig.from_mapping(kalcb_payload)
    olr_cfg = OLRConfig.from_mapping(olr_payload)
    config_manifest = _write_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload=olr_payload)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(trade_date, kalcb_cfg, count=1, active_count=1, payload=kalcb_payload, sector_map=sector_map)
    )
    olr_store = OLRArtifactStore(tmp_path / "olr")
    olr_store.save_snapshot(_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, olr_cfg, count=1), artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date, assistant_event_dir=tmp_path / "assistant")
    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        sector_map=sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )
    original_router = plan.action_router
    original_context = next(iter(plan.drivers.values())).portfolio_context

    olr_store.save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, olr_cfg, count=1), artifact_stage=OLR_FINAL_ARTIFACT_STAGE)
    driver = plan.enable_olr_final(artifact_root=tmp_path / "olr")

    manifest = json.loads((tmp_path / "session" / "session_manifest.json").read_text(encoding="utf-8"))
    assert driver is plan.drivers["OLR"]
    assert plan.action_router is original_router
    assert plan.drivers["OLR"].portfolio_context is original_context
    assert plan.ready_for_olr_start is True
    assert manifest["kis_resource_plan_hash"] == plan.kis_resource_plan.plan_hash
    assert any(row["stage"] == OLR_FINAL_ARTIFACT_STAGE for row in manifest["staged_artifacts"])
    resource_rows = [
        json.loads(line)
        for path in (tmp_path / "assistant" / "resource_plans").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    deployment_rows = [
        json.loads(line)
        for path in (tmp_path / "assistant" / "deployments").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final_resource = next(row for row in resource_rows if row["payload"]["plan_hash"] == plan.kis_resource_plan.plan_hash)
    final_deployment = next(row for row in deployment_rows if row["payload"]["kis_resource_plan_hash"] == plan.kis_resource_plan.plan_hash)
    assert final_resource["deployment_id"] == final_deployment["deployment_id"]
    assert final_resource["strategy_registry_version"] == final_deployment["strategy_registry_version"]
    assert final_resource["lineage_gap"] is False
    result = asyncio.run(plan.handle_bar(_bar("100001", trade_date, time(14, 36))))
    assert [item.strategy_id for item in result] == ["OLR"]


def test_combined_session_enable_olr_final_uses_readiness_cutoff(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_payload = {"kalcb.session.ws_budget": 2}
    olr_payload = {"olr.afternoon.top_n": 1, "olr.overnight.slot_count": 1}
    kalcb_cfg = KALCBConfig.from_mapping(kalcb_payload)
    olr_cfg = OLRConfig.from_mapping(olr_payload)
    sector_map = {"000001": "UNKNOWN", "000002": "UNKNOWN"}
    config_manifest = _write_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload=olr_payload)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(trade_date, kalcb_cfg, count=2, active_count=2, payload=kalcb_payload, sector_map=sector_map)
    )
    olr_store = OLRArtifactStore(tmp_path / "olr")
    olr_store.save_snapshot(_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, olr_cfg, count=1), artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)
    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        sector_map=sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    olr_store.save_snapshot(
        _olr_snapshot(
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            olr_cfg,
            count=1,
            generated_at=datetime.combine(trade_date, time(14, 29), tzinfo=KST),
        ),
        artifact_stage=OLR_FINAL_ARTIFACT_STAGE,
    )

    with pytest.raises(ValueError, match="14:30"):
        plan.enable_olr_final(artifact_root=tmp_path / "olr")
    assert "OLR" not in plan.drivers


def test_operator_enables_olr_final_and_refreshes_coordinator_plan(tmp_path):
    trade_date = date(2026, 2, 2)
    sector_map = {"000001": "UNKNOWN"}
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    olr_payload = {"olr.research.top_long_count": 1, "olr.afternoon.top_n": 1, "olr.overnight.slot_count": 1}
    kalcb_cfg = KALCBConfig.from_mapping(kalcb_payload)
    olr_cfg = OLRConfig.from_mapping(olr_payload)
    config_manifest = _write_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload=olr_payload)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(trade_date, kalcb_cfg, count=1, active_count=1, payload=kalcb_payload, sector_map=sector_map)
    )
    olr_store = OLRArtifactStore(tmp_path / "olr")
    olr_store.save_snapshot(_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, olr_cfg, count=1), artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=PaperSessionRecorder(tmp_path / "session", trade_date),
        strategy_config_source=config_manifest,
        sector_map=sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )
    assert plan.ready_to_start is True
    assert set(plan.drivers) == {"KALCB"}
    assert plan.strategy_config_summaries["OLR"]["payload"] == olr_payload
    manager = _FakeSubscriptionManager()
    coordinator = KISMarketDataCoordinator(
        resource_plan=plan.kis_resource_plan,
        subscription_manager=manager,
        ledger_path=tmp_path / "ws_ledger.json",
        market_data_source="kis_websocket",
    )
    args = SimpleNamespace(olr_artifact_root=str(tmp_path / "olr"))

    assert operator._maybe_enable_olr_final(args, plan, coordinator=coordinator) is False
    olr_store.save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, olr_cfg, count=1), artifact_stage=OLR_FINAL_ARTIFACT_STAGE)
    assert operator._maybe_enable_olr_final(args, plan, coordinator=coordinator) is True
    rows = asyncio.run(coordinator.activate_due_windows(datetime.combine(trade_date, time(14, 36), tzinfo=KST), runtime_plan=plan))

    assert plan.ready_for_olr_start is True
    assert final_candidate_config_fingerprint(plan.strategy_configs["OLR"]) == final_candidate_config_fingerprint(olr_cfg)
    assert coordinator.resource_plan.plan_hash == plan.kis_resource_plan.plan_hash
    assert ("subscribe", "olr_final_runtime", "100001") in {
        (row["action"], row["lease_name"], row["symbol"]) for row in rows
    }
    assert manager.tick_subs == {"100001"}
    assert operator._maybe_enable_olr_final(args, plan, coordinator=coordinator) is False


def test_market_data_coordinator_owns_ws_leases_and_subscription_evidence(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2})
    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="dry_run",
        kalcb_config=kalcb_cfg,
        kalcb_snapshot=_kalcb_snapshot(trade_date, kalcb_cfg, count=2, active_count=2),
    )
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)
    manager = _FakeSubscriptionManager()
    coordinator = KISMarketDataCoordinator(
        resource_plan=plan,
        recorder=recorder,
        subscription_manager=manager,
        ledger_path=tmp_path / "ws_ledger.json",
    )

    activated = asyncio.run(coordinator.activate_window("kalcb_entry_discovery"))
    released = asyncio.run(coordinator.release_window("kalcb_entry_discovery"))

    assert [row["action"] for row in activated] == ["subscribe", "subscribe"]
    assert manager.tick_subs == set()
    assert len(released) == 2
    rows = (tmp_path / "session" / "subscription_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert all(json.loads(row)["kis_resource_plan_hash"] == plan.plan_hash for row in rows)


def test_external_completed_bars_do_not_emit_fake_subscribe_evidence(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2})
    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="dry_run",
        kalcb_config=kalcb_cfg,
        kalcb_snapshot=_kalcb_snapshot(trade_date, kalcb_cfg, count=2, active_count=2),
    )
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)
    coordinator = KISMarketDataCoordinator(
        resource_plan=plan,
        recorder=recorder,
        market_data_source="external_completed_bars",
    )

    rows = asyncio.run(coordinator.activate_due_windows(datetime.combine(trade_date, time(9, 35), tzinfo=KST)))

    assert [row["action"] for row in rows] == ["external_source_declared"]
    assert rows[0]["market_data_source"] == "external_completed_bars"
    assert rows[0]["reason_code"] == "external_completed_bars_no_kis_subscription"
    recorded = [
        json.loads(row)
        for row in (tmp_path / "session" / "subscription_events.jsonl").read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    assert [row["action"] for row in recorded] == ["external_source_declared"]
    assert "subscribe" not in {row["action"] for row in recorded}


def test_market_data_coordinator_subscribes_dynamic_kalcb_management_symbols(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2})
    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="paper",
        kalcb_config=kalcb_cfg,
        kalcb_snapshot=_kalcb_snapshot(trade_date, kalcb_cfg, count=2, active_count=2),
    )
    manager = _FakeSubscriptionManager()
    coordinator = KISMarketDataCoordinator(
        resource_plan=plan,
        subscription_manager=manager,
        ledger_path=tmp_path / "ws_ledger.json",
        market_data_source="kis_websocket",
    )

    activated = asyncio.run(
        coordinator.activate_due_windows(
            datetime.combine(trade_date, time(12, 5), tzinfo=KST),
            held_or_pending_symbols={"KALCB": ("000002",)},
        )
    )
    released = asyncio.run(
        coordinator.activate_due_windows(
            datetime.combine(trade_date, time(12, 10), tzinfo=KST),
            held_or_pending_symbols={"KALCB": ()},
        )
    )

    assert [(row["action"], row["lease_name"], row["symbol"]) for row in activated] == [
        ("subscribe", "kalcb_position_management", "000002")
    ]
    assert [(row["action"], row["lease_name"], row["symbol"]) for row in released] == [
        ("unsubscribe", "kalcb_position_management", "000002")
    ]
    assert manager.tick_subs == set()


def test_market_data_coordinator_derives_dynamic_symbols_from_runtime_plan(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2})
    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="paper",
        kalcb_config=kalcb_cfg,
        kalcb_snapshot=_kalcb_snapshot(trade_date, kalcb_cfg, count=2, active_count=2),
    )
    manager = _FakeSubscriptionManager()
    coordinator = KISMarketDataCoordinator(
        resource_plan=plan,
        subscription_manager=manager,
        ledger_path=tmp_path / "ws_ledger.json",
        market_data_source="kis_websocket",
    )
    runtime_plan = SimpleNamespace(
        drivers={
            "KALCB": SimpleNamespace(
                descriptor=SimpleNamespace(
                    engine=SimpleNamespace(
                        state=SimpleNamespace(
                            symbols={
                                "000002": SimpleNamespace(
                                    position=SimpleNamespace(qty_open=1),
                                    pending_entry_order_id="",
                                    pending_exit_order_id="",
                                )
                            }
                        )
                    )
                )
            )
        }
    )

    activated = asyncio.run(
        coordinator.activate_due_windows(
            datetime.combine(trade_date, time(12, 5), tzinfo=KST),
            runtime_plan=runtime_plan,
        )
    )

    assert [(row["action"], row["lease_name"], row["symbol"]) for row in activated] == [
        ("subscribe", "kalcb_position_management", "000002")
    ]
    assert manager.tick_subs == {"000002"}


def test_market_data_coordinator_subscribes_olr_final_orderable_symbols_only(tmp_path):
    trade_date = date(2026, 2, 2)
    olr_cfg = OLRConfig.from_mapping({"olr.afternoon.top_n": 2, "olr.overnight.slot_count": 1})
    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="paper",
        olr_config=olr_cfg,
        olr_final_snapshot=_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, olr_cfg, count=2),
    )
    manager = _FakeSubscriptionManager()
    coordinator = KISMarketDataCoordinator(
        resource_plan=plan,
        subscription_manager=manager,
        ledger_path=tmp_path / "ws_ledger.json",
        market_data_source="kis_websocket",
    )

    activated = asyncio.run(coordinator.activate_due_windows(datetime.combine(trade_date, time(14, 36), tzinfo=KST)))

    assert [(row["action"], row["lease_name"], row["symbol"]) for row in activated] == [
        ("subscribe", "olr_final_runtime", "100001")
    ]
    assert manager.tick_subs == {"100001"}


def test_market_data_coordinator_keeps_shared_symbol_subscription_until_last_lease_releases(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_cfg = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 2})
    olr_cfg = OLRConfig.from_mapping({"olr.afternoon.top_n": 1, "olr.overnight.slot_count": 1})
    plan = build_kis_resource_plan(
        trade_date=trade_date,
        mode="paper",
        kalcb_config=kalcb_cfg,
        olr_config=olr_cfg,
        kalcb_snapshot=_kalcb_snapshot(trade_date, kalcb_cfg, count=1, active_count=1),
        olr_final_snapshot=_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, olr_cfg, count=1),
    )
    manager = _FakeSubscriptionManager()
    coordinator = KISMarketDataCoordinator(
        resource_plan=plan,
        subscription_manager=manager,
        ledger_path=tmp_path / "ws_ledger.json",
        market_data_source="kis_websocket",
    )

    activated = asyncio.run(
        coordinator.activate_due_windows(
            datetime.combine(trade_date, time(14, 36), tzinfo=KST),
            held_or_pending_symbols={"KALCB": ("100001",)},
        )
    )
    kalcb_release = asyncio.run(coordinator.release_window("kalcb_position_management"))

    assert [(row["action"], row["lease_name"], row["symbol"]) for row in activated] == [
        ("subscribe", "kalcb_position_management", "100001"),
        ("subscribe", "olr_final_runtime", "100001"),
    ]
    assert activated[1]["reason_code"] == "shared_subscription_reused"
    assert activated[1]["ws_used_after"] == 1
    assert [(row["action"], row["symbol"]) for row in kalcb_release] == [("unsubscribe", "100001")]
    assert kalcb_release[0]["ws_used_after"] == 1
    assert manager.tick_subs == {"100001"}

    olr_release = asyncio.run(coordinator.release_window("olr_final_runtime"))

    assert [(row["action"], row["symbol"]) for row in olr_release] == [("unsubscribe", "100001")]
    assert olr_release[0]["ws_used_after"] == 0
    assert manager.tick_subs == set()


def test_ws_limit_constants_have_single_authority():
    from kis_core.ws_client import KIS_WS_EXECUTION_NOTIFICATION_RESERVE, KIS_WS_TOTAL_REGISTRATION_LIMIT, WS_MAX_REGS_DEFAULT

    assert WS_MAX_REGS_DEFAULT == KIS_WS_TOTAL_REGISTRATION_LIMIT - KIS_WS_EXECUTION_NOTIFICATION_RESERVE


def test_paper_gate_blocks_missing_required_resource_plan(tmp_path):
    session = tmp_path / "session"
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))
    (session / "market_bars_5m.parquet").write_bytes(b"fixture-bars")
    recorder.write_end_of_day_positions({"positions": []})
    recorder.write_manifest({"mode": "dry_run", "strategy_ids": ["KALCB"], "kis_resource_plan_required": True})

    report = replay_paper_session(session)

    assert report["session_bundle_complete"] is False
    assert "kis_resource_plan_missing" in report["promotion_blockers"]


def test_paper_gate_blocks_tampered_resource_plan_hash(tmp_path):
    session = tmp_path / "session"
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))
    recorder.write_resource_plan(
        {
            "trade_date": "2026-02-02",
            "mode": "paper",
            "limit_profile": {},
            "candidate_surfaces": [],
            "lease_windows": [],
            "passed": True,
            "failures": [],
            "warnings": [],
            "version": "unit",
            "plan_hash": "tampered",
        }
    )
    (session / "market_bars_5m.parquet").write_bytes(b"fixture-bars")
    recorder.write_end_of_day_positions({"positions": []})
    recorder.write_manifest({"mode": "paper", "strategy_ids": [], "kis_resource_plan_hash": "tampered"})

    report = replay_paper_session(session)

    assert report["session_bundle_complete"] is False
    assert "kis_resource_plan_hash_mismatch" in report["resource_plan_failures"]
    assert "kis_resource_plan_invalid" in report["promotion_blockers"]


def _kalcb_snapshot(
    trade_date: date,
    config: KALCBConfig,
    *,
    count: int,
    active_count: int,
    payload: dict | None = None,
    sector_map: dict[str, str] | None = None,
) -> KALCBDailySnapshot:
    symbols = _symbols(count)
    active = symbols[:active_count]
    config_payload = dict(payload or {})
    sectors = dict(sector_map or {symbol: "UNKNOWN" for symbol in symbols})
    return KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit-kalcb-source",
        generated_at=datetime.combine(trade_date, time(8, 40), tzinfo=KST),
        candidates=tuple(
            KALCBDailyCandidate(
                symbol=symbol,
                trade_date=trade_date,
                prior_day_high=102.0,
                prior_day_low=98.0,
                prior_day_close=100.0,
                daily_atr=2.0,
                expected_5m_volume=100.0,
                average_30m_volume=600.0,
                sector=sectors.get(symbol, "UNKNOWN"),
                source_fingerprint="unit-kalcb-source",
                metadata={"frontier_initial_active": symbol in active},
            )
            for symbol in symbols
        ),
        metadata={
            "artifact_stage": KALCB_FINAL_ARTIFACT_STAGE,
            "source": "unit",
            "candidate_config_hash": kalcb_candidate_config_fingerprint(config, config_payload, sectors),
            "sector_map_hash": canonical_json_hash(sectors),
            "sector_map_size": len(sectors),
            "active_symbols": list(active),
            "active_symbol_count": len(active),
            "active_budget_source": "ws_budget",
            "frontier_symbols": list(symbols),
            "frontier_symbol_count": len(symbols),
            "overflow_symbols": list(symbols[active_count:]),
            "overflow_symbol_count": max(0, count - active_count),
            "frontier_rest_budget_symbols_per_5m": _kalcb_frontier_rest_budget(config),
        },
    )


def _olr_snapshot(
    trade_date: date,
    stage: str,
    config: OLRConfig,
    *,
    count: int,
    generated_at: datetime | None = None,
) -> OLRDailySnapshot:
    symbols = tuple(f"1{index:05d}" for index in range(1, count + 1))
    metadata = {
        "artifact_stage": stage,
        "source": "olr_afternoon_selection" if stage == OLR_FINAL_ARTIFACT_STAGE else "olr_research_selection",
        "selection_time_basis": "14:30_decision_from_completed_5m_bars"
        if stage == OLR_FINAL_ARTIFACT_STAGE
        else "pre_session_from_prior_completed_daily_rows",
    }
    if stage == OLR_FINAL_ARTIFACT_STAGE:
        config_hash = final_candidate_config_fingerprint(config)
        metadata.update(
            {
                "candidate_config_hash": config_hash,
                "final_candidate_config_hash": config_hash,
                "final_candidate_config_hash_version": FINAL_CANDIDATE_CONFIG_HASH_VERSION,
            }
        )
    return OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit-olr-source",
        generated_at=generated_at or datetime.combine(
            trade_date,
            time(14, 31) if stage == OLR_FINAL_ARTIFACT_STAGE else time(8, 40),
            tzinfo=KST,
        ),
        candidates=tuple(
            OLRDailyCandidate(
                symbol=symbol,
                trade_date=trade_date,
                prior_day_high=102.0,
                prior_day_low=98.0,
                prior_day_close=100.0,
                daily_atr=2.0,
                expected_5m_volume=100.0,
                average_30m_volume=600.0,
                source_fingerprint="unit-olr-source",
            )
            for symbol in symbols
        ),
        metadata=metadata,
    )


def _write_config_manifest(tmp_path, *, kalcb_payload: dict, olr_payload: dict) -> dict:
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    kalcb_path = config_root / "kalcb_optimized_config.json"
    olr_path = config_root / "olr_optimized_config.json"
    kalcb_path.write_text(json.dumps({"mutations": kalcb_payload}, sort_keys=True), encoding="utf-8")
    olr_path.write_text(json.dumps({"mutations": olr_payload}, sort_keys=True), encoding="utf-8")
    return {
        "artifacts": [
            {"label": "kalcb optimized_config", "path": str(kalcb_path)},
            {"label": "olr optimized_config", "path": str(olr_path)},
        ]
    }


def _bar(symbol: str, trade_date: date, timestamp: time) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(trade_date, timestamp, tzinfo=KST),
        timeframe="5m",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
        is_completed=True,
    )


def _symbols(count: int) -> tuple[str, ...]:
    return tuple(f"{index:06d}" for index in range(1, count + 1))


def _kalcb_frontier_rest_budget(config: KALCBConfig) -> int:
    raw = int((5 * 60 / max(float(config.rest_min_interval_paper_s), 1e-9)) * max(min(float(config.frontier_rest_safety_fraction), 1.0), 0.01))
    return max(1, raw)


class _FakeSubscriptionManager:
    def __init__(self) -> None:
        self.tick_subs: set[str] = set()
        self.askbid_subs: set[str] = set()

    def total_regs(self) -> int:
        return len(self.tick_subs) + len(self.askbid_subs)

    async def ensure_tick(self, ticker: str) -> bool:
        self.tick_subs.add(str(ticker).zfill(6))
        return True

    async def ensure_askbid(self, ticker: str) -> bool:
        self.askbid_subs.add(str(ticker).zfill(6))
        return True

    async def drop_tick(self, ticker: str) -> None:
        self.tick_subs.discard(str(ticker).zfill(6))

    async def drop_askbid(self, ticker: str) -> None:
        self.askbid_subs.discard(str(ticker).zfill(6))
