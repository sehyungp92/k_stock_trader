from __future__ import annotations

import asyncio
import inspect
import json
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import deployment.olr_kalcb.runtime as runtime_module
from deployment.olr_kalcb.coordinator import create_strategy_descriptor, create_strategy_descriptors
from deployment.olr_kalcb.hashing import canonical_json_hash
from deployment.olr_kalcb.readiness import load_strategy_artifact
from deployment.olr_kalcb.runtime import prepare_runtime_session
from deployment.olr_kalcb.session_capture import PaperSessionRecorder
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot
from strategy_kalcb.research import KALCB_FINAL_ARTIFACT_STAGE, candidate_config_fingerprint as kalcb_candidate_config_fingerprint
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE, OLRArtifactStore
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.research import (
    FINAL_CANDIDATE_CONFIG_HASH_VERSION,
    afternoon_selection_from_contexts,
    build_afternoon_contexts,
    build_research_snapshot,
    daily_selection_from_snapshot,
    final_candidate_config_fingerprint,
)


def _paper_health_checks(**overrides):
    checks = {
        "dry_run_gate_passed": True,
        "market_session_open": True,
        "kis_auth_ok": True,
        "market_data_ok": True,
        "account_ok": True,
        "order_route_enabled": True,
        "risk_limits_loaded": True,
        "kill_switch_ready": True,
        "oms_health_ok": True,
        "durable_stops_ok": True,
        "idempotency_reservation_ok": True,
        "portfolio_context_fresh": True,
        "paper_trading_approved": True,
        "oms_health_payload": {
            "status": "ok",
            "stop_protection_status": "ok",
            "unprotected_positions_count": 0,
            "active_stop_count": 0,
            "triggered_stop_count": 0,
            "stop_watcher_price_stale_count": 0,
            "idempotency_status": "ok",
        },
    }
    checks.update(overrides)
    return checks


def test_readiness_accepts_valid_kalcb_and_rejects_stale(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))

    loaded = load_strategy_artifact(
        "KALCB",
        trade_date,
        KALCB_FINAL_ARTIFACT_STAGE,
        artifact_roots={"KALCB": tmp_path / "kalcb"},
    )

    assert loaded.artifact_hash
    with pytest.raises(FileNotFoundError):
        load_strategy_artifact(
            "KALCB",
            date(2026, 2, 3),
            KALCB_FINAL_ARTIFACT_STAGE,
            artifact_roots={"KALCB": tmp_path / "kalcb"},
        )


def test_readiness_requires_olr_final_for_executable_mode(tmp_path):
    trade_date = date(2026, 2, 2)
    store = OLRArtifactStore(tmp_path / "olr")
    store.save_snapshot(_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE))

    loaded_stage1 = load_strategy_artifact(
        "OLR",
        trade_date,
        OLR_STAGE1_ARTIFACT_STAGE,
        mode="artifact_only_stage1",
        artifact_roots={"OLR": tmp_path / "olr"},
    )

    assert loaded_stage1.metadata["artifact_stage"] == OLR_STAGE1_ARTIFACT_STAGE
    with pytest.raises(ValueError, match="not valid for mode"):
        load_strategy_artifact(
            "OLR",
            trade_date,
            OLR_STAGE1_ARTIFACT_STAGE,
            mode="paper",
            artifact_roots={"OLR": tmp_path / "olr"},
        )
    with pytest.raises(FileNotFoundError):
        load_strategy_artifact(
            "OLR",
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            mode="paper",
            artifact_roots={"OLR": tmp_path / "olr"},
        )

    store.save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE))
    loaded_final = load_strategy_artifact(
        "OLR",
        trade_date,
        OLR_FINAL_ARTIFACT_STAGE,
        mode="paper",
        artifact_roots={"OLR": tmp_path / "olr"},
    )
    assert loaded_final.metadata["artifact_stage"] == OLR_FINAL_ARTIFACT_STAGE


def test_readiness_rejects_wrong_stage_metadata(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date, metadata={"candidate_config_hash": ""}))
    OLRArtifactStore(tmp_path / "olr").save_snapshot(
        _olr_snapshot(
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            metadata={"source": "olr_research_selection"},
        )
    )

    with pytest.raises(ValueError, match="candidate_config_hash"):
        load_strategy_artifact(
            "KALCB",
            trade_date,
            KALCB_FINAL_ARTIFACT_STAGE,
            artifact_roots={"KALCB": tmp_path / "kalcb"},
        )
    with pytest.raises(ValueError, match="OLR final source"):
        load_strategy_artifact(
            "OLR",
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            artifact_roots={"OLR": tmp_path / "olr"},
        )


def test_readiness_requires_olr_final_config_fingerprint(tmp_path):
    trade_date = date(2026, 2, 2)
    OLRArtifactStore(tmp_path / "olr").save_snapshot(
        _olr_snapshot(
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            metadata={"candidate_config_hash": "", "final_candidate_config_hash": ""},
        )
    )

    with pytest.raises(ValueError, match="candidate_config_hash"):
        load_strategy_artifact(
            "OLR",
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            artifact_roots={"OLR": tmp_path / "olr"},
        )


def test_readiness_validates_generated_at_timezone_and_cutoff(tmp_path):
    trade_date = date(2026, 2, 2)
    store = OLRArtifactStore(tmp_path / "olr")
    store.save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, generated_at=datetime(2026, 2, 2, 5, 31, tzinfo=timezone.utc)))

    loaded = load_strategy_artifact(
        "OLR",
        trade_date,
        OLR_FINAL_ARTIFACT_STAGE,
        mode="paper",
        artifact_roots={"OLR": tmp_path / "olr"},
    )
    assert loaded.generated_at.astimezone(KST).time() >= time(14, 30)

    store.save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, generated_at=datetime(2026, 2, 2, 14, 31)))
    with pytest.raises(ValueError, match="timezone-aware"):
        load_strategy_artifact(
            "OLR",
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            mode="paper",
            artifact_roots={"OLR": tmp_path / "olr"},
        )

    store.save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, generated_at=datetime(2026, 2, 3, 5, 31, tzinfo=timezone.utc)))
    with pytest.raises(ValueError, match="date mismatch"):
        load_strategy_artifact(
            "OLR",
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            mode="paper",
            artifact_roots={"OLR": tmp_path / "olr"},
        )

    store.save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, generated_at=datetime(2026, 2, 2, 14, 29, tzinfo=KST)))
    with pytest.raises(ValueError, match="14:30"):
        load_strategy_artifact(
            "OLR",
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            mode="paper",
            artifact_roots={"OLR": tmp_path / "olr"},
        )

    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date, generated_at=datetime(2026, 2, 2, 9, 1, tzinfo=KST)))
    with pytest.raises(ValueError, match="09:00"):
        load_strategy_artifact(
            "KALCB",
            trade_date,
            KALCB_FINAL_ARTIFACT_STAGE,
            artifact_roots={"KALCB": tmp_path / "kalcb"},
        )

    store.save_snapshot(_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, generated_at=datetime(2026, 2, 2, 9, 1, tzinfo=KST)))
    with pytest.raises(ValueError, match="09:00"):
        load_strategy_artifact(
            "OLR",
            trade_date,
            OLR_STAGE1_ARTIFACT_STAGE,
            mode="artifact_only_stage1",
            artifact_roots={"OLR": tmp_path / "olr"},
        )


def test_readiness_accepts_olr_stage1_generated_by_daily_selector(tmp_path):
    trade_date = date(2026, 2, 2)
    generated_at = datetime.combine(trade_date, time(8, 50), tzinfo=KST)
    cfg = OLRConfig.from_mapping(
        {
            "olr.research.top_long_count": 1,
            "olr.research.min_adv20_krw": 1_000_000,
            "olr.signal.daily_min_score": 0.0,
        }
    )
    research = build_research_snapshot(
        {"005930": _olr_daily_rows(trade_date)},
        trade_date,
        cfg,
        generated_at=generated_at,
        source_fingerprint="unit-stage1-readiness",
    )
    snapshot = daily_selection_from_snapshot(research, cfg)

    assert snapshot.generated_at == generated_at

    store = OLRArtifactStore(tmp_path / "olr")
    store.save_snapshot(snapshot, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
    loaded = load_strategy_artifact(
        "OLR",
        trade_date,
        OLR_STAGE1_ARTIFACT_STAGE,
        mode="artifact_only_stage1",
        artifact_roots={"OLR": tmp_path / "olr"},
    )

    assert loaded.artifact_hash == snapshot.artifact_hash


def test_readiness_accepts_shadow_reranker_final_artifact_timestamp(tmp_path):
    trade_date = date(2026, 2, 2)
    stage1 = _olr_snapshot(
        trade_date,
        OLR_STAGE1_ARTIFACT_STAGE,
        generated_at=datetime.combine(trade_date, time(8, 50), tzinfo=KST),
    )
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 1,
            "olr.afternoon.min_bar_count": 1,
            "olr.afternoon.min_score": -999.0,
            "olr.afternoon.max_score": 99999.0,
            "olr.shadow_reranker.enabled": True,
            "olr.shadow_reranker.profile": {"profile_hash": "unit-shadow-readiness"},
        }
    )
    bars_by_key = {
        (trade_date, "005930"): (
            MarketBar(
                "005930",
                datetime.combine(trade_date, time(9, 0), tzinfo=KST),
                "5m",
                100.0,
                101.0,
                99.0,
                100.5,
                1000.0,
            ),
            MarketBar(
                "005930",
                datetime.combine(trade_date, time(14, 25), tzinfo=KST),
                "5m",
                100.5,
                102.0,
                100.0,
                101.5,
                1000.0,
            ),
        )
    }
    final = afternoon_selection_from_contexts(stage1, build_afternoon_contexts(stage1, bars_by_key, cfg), cfg)

    assert final.metadata["source"] == "olr_shadow_same_day_reranker"
    assert final.generated_at == datetime.combine(trade_date, time(14, 30), tzinfo=KST)

    OLRArtifactStore(tmp_path / "olr").save_snapshot(final, artifact_stage=OLR_FINAL_ARTIFACT_STAGE)
    loaded = load_strategy_artifact(
        "OLR",
        trade_date,
        OLR_FINAL_ARTIFACT_STAGE,
        mode="paper",
        artifact_roots={"OLR": tmp_path / "olr"},
    )

    assert loaded.metadata["source"] == "olr_shadow_same_day_reranker"
    assert loaded.generated_at == final.generated_at


def test_coordinator_creates_kalcb_and_olr_descriptors(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    OLRArtifactStore(tmp_path / "olr").save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE))

    descriptors = create_strategy_descriptors(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
        kalcb_config=KALCBConfig(),
        olr_config=OLRConfig(),
    )

    assert descriptors["KALCB"].engine.candidate_snapshot.strategy_id == "KALCB"
    assert descriptors["OLR"].engine.candidate_snapshot.strategy_id == "OLR"
    assert descriptors["OLR"].artifact_stage == OLR_FINAL_ARTIFACT_STAGE


def test_runtime_artifact_only_plan_validates_artifacts_without_starting_engines(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    OLRArtifactStore(tmp_path / "olr").save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE))

    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="artifact_only",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
    )

    assert plan.ready_to_start is True
    assert plan.descriptors == {}
    assert plan.portfolio_policy_hash
    assert plan.preflight.passed is True


def test_runtime_combined_plan_can_disable_portfolio(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    OLRArtifactStore(tmp_path / "olr").save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE))

    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="artifact_only",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
        portfolio_enabled=False,
    )

    assert plan.ready_to_start is True
    assert plan.portfolio_enabled is False
    assert plan.portfolio_policy_hash is None


def test_runtime_artifact_only_schedule_matches_enabled_strategy(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="artifact_only",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
    )

    step_names = [step.name for step in plan.schedule]
    assert plan.ready_to_start is True
    assert step_names == ["kalcb_daily_artifact", "artifact_readiness"]
    assert plan.portfolio_policy_hash is None


def test_runtime_paper_plan_blocks_without_required_operational_gates(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    OLRArtifactStore(tmp_path / "olr").save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE))

    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
    )

    failure_names = {check.name for check in plan.preflight.failures}
    assert plan.ready_to_start is False
    assert plan.descriptors == {}
    assert {"oms_client_available", "dry_run_gate_passed", "paper_trading_approved"} <= failure_names


def test_runtime_paper_plan_requires_live_hardening_health_gates(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks=_paper_health_checks(
            oms_health_ok=False,
            durable_stops_ok=False,
            idempotency_reservation_ok=False,
            portfolio_context_fresh=False,
        ),
        oms_client=_SubmitOnlyOMS(),
        session_recorder=PaperSessionRecorder(tmp_path / "session", trade_date, assistant_event_dir=tmp_path / "assistant"),
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    failure_names = {check.name for check in plan.preflight.failures}
    assert {
        "oms_health_ok",
        "durable_stops_ok",
        "idempotency_reservation_ok",
        "portfolio_context_fresh",
    } <= failure_names
    assert plan.ready_to_start is False


def test_runtime_paper_plan_rejects_manual_hardening_booleans_without_raw_oms_health(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    health_checks = _paper_health_checks()
    health_checks.pop("oms_health_payload")

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks=health_checks,
        oms_client=_SubmitOnlyOMS(),
        session_recorder=PaperSessionRecorder(tmp_path / "session", trade_date, assistant_event_dir=tmp_path / "assistant"),
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    failures = {check.name: check for check in plan.preflight.failures}
    assert {"oms_health_ok", "durable_stops_ok", "idempotency_reservation_ok"} <= set(failures)
    assert "raw OMS /health" in failures["oms_health_ok"].detail
    assert plan.ready_to_start is False


def test_runtime_paper_plan_rejects_incomplete_stop_health_payload(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks=_paper_health_checks(
            oms_health_payload={
                "status": "ok",
                "stop_protection_status": "ok",
                "idempotency_status": "ok",
            }
        ),
        oms_client=_SubmitOnlyOMS(),
        session_recorder=PaperSessionRecorder(tmp_path / "session", trade_date, assistant_event_dir=tmp_path / "assistant"),
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    failures = {check.name: check for check in plan.preflight.failures}
    assert "durable_stops_ok" in failures
    assert "missing" in failures["durable_stops_ok"].detail
    assert plan.ready_to_start is False


def test_runtime_paper_plan_requires_recent_stop_watcher_check_when_active_stops_exist(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks=_paper_health_checks(
            oms_health_payload={
                "status": "ok",
                "stop_protection_status": "ok",
                "unprotected_positions_count": 0,
                "active_stop_count": 1,
                "triggered_stop_count": 0,
                "stop_watcher_price_stale_count": 0,
                "idempotency_status": "ok",
            }
        ),
        oms_client=_SubmitOnlyOMS(),
        session_recorder=PaperSessionRecorder(tmp_path / "session", trade_date, assistant_event_dir=tmp_path / "assistant"),
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    failures = {check.name: check for check in plan.preflight.failures}
    assert "durable_stops_ok" in failures
    assert "stop_watcher_last_check_age_sec" in failures["durable_stops_ok"].detail
    assert plan.ready_to_start is False


def test_runtime_paper_plan_requires_and_records_initial_replay_state(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    sector_map = {"005930": "UNKNOWN"}
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload={})
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(
            trade_date,
            metadata={
                "candidate_config_hash": kalcb_candidate_config_fingerprint(
                    KALCBConfig.from_mapping(kalcb_payload),
                    kalcb_payload,
                    sector_map,
                )
            },
            sector_map=sector_map,
        )
    )
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date, assistant_event_dir=tmp_path / "assistant")
    health_checks = _paper_health_checks()

    missing_state = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks=health_checks,
        oms_client=_SubmitOnlyOMS(),
        session_recorder=recorder,
    )
    ready = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks=health_checks,
        oms_client=_SubmitOnlyOMS(),
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        sector_map=sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    assert "runtime_initial_state_capture" in {check.name for check in missing_state.preflight.failures}
    assert ready.ready_to_start is True
    manifest = json.loads((tmp_path / "session" / "session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["initial_account_state"]["equity"] == 1_000_000.0
    assert manifest["initial_positions"] == {}


def test_runtime_paper_plan_seeds_portfolio_context_from_initial_state(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    sector_map = {"005930": "UNKNOWN"}
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload={})
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(
            trade_date,
            metadata={
                "candidate_config_hash": kalcb_candidate_config_fingerprint(
                    KALCBConfig.from_mapping(kalcb_payload),
                    kalcb_payload,
                    sector_map,
                )
            },
            sector_map=sector_map,
        )
    )
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date, assistant_event_dir=tmp_path / "assistant")
    health_checks = _paper_health_checks()

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks=health_checks,
        oms_client=_SubmitOnlyOMS(),
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        sector_map=sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 900_000.0},
        initial_positions={
            "005930": {
                "real_qty": 2,
                "avg_price": 100.0,
                "allocations": {"KALCB": {"qty": 2, "cost_basis": 100.0}},
            }
        },
    )

    context = plan.drivers["KALCB"].portfolio_context
    assert context.cash_equity().cash == 900_000.0
    assert context.strategy_exposure("KALCB", "005930").qty == 2


def test_runtime_paper_plan_emits_approval_metadata_when_requested(tmp_path, monkeypatch):
    calls: list[tuple[object, dict]] = []

    def fake_emit(output_path, **kwargs):
        calls.append((output_path, kwargs))
        return {"deployment_id": "deploy-unit"}

    monkeypatch.setattr(runtime_module, "emit_deployment_metadata", fake_emit)
    trade_date = date(2026, 2, 2)
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    sector_map = {"005930": "UNKNOWN"}
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload={})
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(
            trade_date,
            metadata={
                "candidate_config_hash": kalcb_candidate_config_fingerprint(
                    KALCBConfig.from_mapping(kalcb_payload),
                    kalcb_payload,
                    sector_map,
                )
            },
            sector_map=sector_map,
        )
    )
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date, assistant_event_dir=tmp_path / "assistant")
    health_checks = _paper_health_checks()

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="paper",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks=health_checks,
        oms_client=_SubmitOnlyOMS(),
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        sector_map=sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={
            "005930": {
                "real_qty": 2,
                "avg_price": 100.0,
                "allocations": {"KALCB": {"qty": 2, "cost_basis": 100.0}},
            }
        },
        deployment_metadata_path=tmp_path / "deployment_metadata.json",
        deployment_metadata_contract_path=tmp_path / "strategy_plugin_contract.json",
        deployment_metadata_environment="paper_vps",
        runtime_entrypoint="unit:paper",
    )

    assert plan.ready_to_start is True
    assert len(calls) == 1
    output_path, kwargs = calls[0]
    assert Path(output_path) == tmp_path / "deployment_metadata.json"
    assert Path(kwargs["contract_path"]) == tmp_path / "strategy_plugin_contract.json"
    assert kwargs["mode"] == "paper"
    assert kwargs["strategy_ids"] == ("KALCB",)
    assert kwargs["strategy_configs"]["KALCB"]["uses_defaults"] is False
    assert kwargs["strategy_artifacts"]["KALCB"].artifact_hash
    assert kwargs["initial_positions"]["005930"]["allocations"]["KALCB"]["qty"] == 2
    assert kwargs["kis_resource_plan_hash"] == plan.kis_resource_plan.plan_hash
    assert kwargs["deployment_id"].startswith("deploy:")
    assert kwargs["runtime_instance_id"].startswith("runtime:")
    assert kwargs["runtime_entrypoint"] == "unit:paper"
    assert kwargs["emission_environment"] == "paper_vps"
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    assert Path(manifest["risk_config_source"]).name == "oms_config.yaml"
    assert manifest["risk_config"]["max_positions_count"] == 15
    assert manifest["risk_config"]["strategy_budgets"]["PCIM"]["max_risk_pct"] == 0.025


def test_runtime_deployment_metadata_uses_active_executable_descriptors(monkeypatch):
    calls: list[dict] = []

    def fake_emit(_output_path, **kwargs):
        calls.append(kwargs)
        return {"deployment_id": "deploy-unit"}

    monkeypatch.setattr(runtime_module, "emit_deployment_metadata", fake_emit)
    plan = SimpleNamespace(
        deployment_metadata_path="deployment_metadata.json",
        deployment_metadata_contract_path="strategy_plugin_contract.json",
        deployment_metadata_environment="paper_vps",
        mode="paper",
        ready_to_start=True,
        descriptors={"KALCB": object()},
        strategy_config_summaries={"KALCB": {"enabled": True}, "OLR": {"stage": "stage1"}},
        portfolio_policy_config={"max_gross_exposure_pct": 0.5},
        artifacts={"KALCB": object(), "OLR": object()},
        kis_resource_plan=SimpleNamespace(plan_hash="plan-unit"),
        session_recorder=None,
        runtime_entrypoint="unit:entrypoint",
    )

    runtime_module._emit_runtime_deployment_metadata(plan)

    assert calls[0]["strategy_ids"] == ("KALCB",)
    assert calls[0]["strategy_configs"].keys() == {"KALCB"}
    assert calls[0]["strategy_artifacts"].keys() == {"KALCB"}


def test_runtime_deployment_metadata_fail_open(monkeypatch):
    calls = 0

    def fake_emit(_output_path, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(runtime_module, "emit_deployment_metadata", fake_emit)
    plan = SimpleNamespace(
        deployment_metadata_path="deployment_metadata.json",
        deployment_metadata_contract_path="missing_contract.json",
        deployment_metadata_environment="paper_vps",
        mode="paper",
        ready_to_start=True,
        descriptors={"KALCB": object()},
        strategy_config_summaries={"KALCB": {"enabled": True}},
        portfolio_policy_config={},
        artifacts={"KALCB": object()},
        kis_resource_plan=SimpleNamespace(plan_hash="plan-unit"),
        session_recorder=None,
        runtime_entrypoint="unit:entrypoint",
    )

    runtime_module._emit_runtime_deployment_metadata(plan)

    assert calls == 1


def test_runtime_dry_run_plan_starts_descriptors_after_preflight(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    olr_payload = {"olr.afternoon.top_n": 1}
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload=olr_payload)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(
            trade_date,
            metadata={
                "candidate_config_hash": kalcb_candidate_config_fingerprint(
                    KALCBConfig.from_mapping(kalcb_payload),
                    kalcb_payload,
                    {"005930": "UNKNOWN"},
                )
            },
        )
    )
    OLRArtifactStore(tmp_path / "olr").save_snapshot(_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE))
    OLRArtifactStore(tmp_path / "olr").save_snapshot(
        _olr_snapshot(
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            metadata={
                "candidate_config_hash": final_candidate_config_fingerprint(OLRConfig.from_mapping(olr_payload)),
                "final_candidate_config_hash": final_candidate_config_fingerprint(OLRConfig.from_mapping(olr_payload)),
            },
        )
    )
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)

    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb", "OLR": tmp_path / "olr"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        sector_map={"005930": "UNKNOWN"},
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    assert plan.ready_to_start is True
    assert plan.action_router is not None
    assert set(plan.drivers) == {"KALCB", "OLR"}
    assert set(plan.descriptors) == {"KALCB", "OLR"}
    assert plan.strategy_config_summaries is not None
    assert plan.strategy_config_summaries["KALCB"]["uses_defaults"] is False
    assert plan.strategy_config_summaries["OLR"]["uses_defaults"] is False
    assert plan.descriptors["KALCB"].engine.config != KALCBConfig()
    assert plan.descriptors["OLR"].engine.config != OLRConfig()
    assert all(descriptor.oms_adapter is not None and descriptor.oms_adapter.dry_run for descriptor in plan.descriptors.values())
    assert list((tmp_path / "session" / "daily_snapshots").glob("*.json"))
    assert list((tmp_path / "session" / "olr_stage1_snapshots").glob("*.json"))
    assert list((tmp_path / "session" / "olr_final_snapshots").glob("*.json"))
    artifact_rows = [json.loads(line) for line in (tmp_path / "session" / "artifact_generation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {(row["strategy_id"], row["stage"]) for row in artifact_rows} == {
        ("KALCB", KALCB_FINAL_ARTIFACT_STAGE),
        ("OLR", OLR_STAGE1_ARTIFACT_STAGE),
        ("OLR", OLR_FINAL_ARTIFACT_STAGE),
    }
    with pytest.raises(RuntimeError, match="RuntimeSessionDriver"):
        asyncio.run(
            plan.descriptors["KALCB"].oms_adapter.submit(
                SubmitEntry(
                    strategy_id="KALCB",
                    symbol="005930",
                    qty=1,
                    order_type="LIMIT",
                    limit_price=100.0,
                    stop_price=None,
                    reason="unit_dry_run",
                )
            )
        )
    assert "runtime_session_pre_start" in (tmp_path / "session" / "state_snapshots.jsonl").read_text(encoding="utf-8")


def test_runtime_plan_close_session_uses_session_recorder(tmp_path):
    trade_date = date(2026, 2, 2)
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    sector_map = {"005930": "UNKNOWN"}
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload={})
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(
            trade_date,
            metadata={
                "candidate_config_hash": kalcb_candidate_config_fingerprint(
                    KALCBConfig.from_mapping(kalcb_payload),
                    kalcb_payload,
                    sector_map,
                )
            },
            sector_map=sector_map,
        )
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
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    manifest_path = plan.close_session({"positions": []}, {"strategy_trade_attempts": 0}, closeout_reason="unit_closeout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["closeout_reason"] == "unit_closeout"
    assert manifest["hash_contract_status"] == "unsealed_failure"
    assert "market_bars_5m.parquet" in manifest["closeout_missing_required_files"]


def test_runtime_execution_blocks_artifact_generated_from_different_config(tmp_path):
    trade_date = date(2026, 2, 2)
    config_payload = {"kalcb.session.ws_budget": 3}
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload=config_payload, olr_payload={})
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    assert plan.ready_to_start is False
    assert any("candidate_config_hash does not match approved runtime config" in check.detail for check in plan.preflight.failures)


def test_runtime_kalcb_config_binding_uses_full_approved_sector_map(tmp_path):
    trade_date = date(2026, 2, 2)
    full_sector_map = {"005930": "SEMIS", "000660": "SEMIS"}
    candidate_only_map = {"005930": "SEMIS"}
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload={})
    config_hash = kalcb_candidate_config_fingerprint(KALCBConfig.from_mapping(kalcb_payload), kalcb_payload, full_sector_map)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(
            trade_date,
            metadata={
                "candidate_config_hash": config_hash,
                "sector_map_hash": canonical_json_hash(full_sector_map),
                "sector_map_size": len(full_sector_map),
            },
        )
    )

    ready = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=PaperSessionRecorder(tmp_path / "ready_session", trade_date),
        strategy_config_source=config_manifest,
        sector_map=full_sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )
    blocked = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=PaperSessionRecorder(tmp_path / "blocked_session", trade_date),
        strategy_config_source=config_manifest,
        sector_map=candidate_only_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    assert ready.ready_to_start is True
    assert ready.strategy_config_summaries["KALCB"]["approved_config_fingerprint"]["sector_map_size"] == 2
    assert blocked.ready_to_start is False
    assert any("candidate_config_hash does not match approved runtime config" in check.detail for check in blocked.preflight.failures)


def test_runtime_kalcb_config_binding_rejects_sector_map_metadata_drift(tmp_path):
    trade_date = date(2026, 2, 2)
    full_sector_map = {"005930": "SEMIS", "000660": "SEMIS"}
    candidate_only_map = {"005930": "SEMIS"}
    kalcb_payload = {"kalcb.session.ws_budget": 3}
    cfg = KALCBConfig.from_mapping(kalcb_payload)
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload=kalcb_payload, olr_payload={})
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(
        _kalcb_snapshot(
            trade_date,
            metadata={
                "candidate_config_hash": kalcb_candidate_config_fingerprint(cfg, kalcb_payload, full_sector_map),
                "sector_map_hash": canonical_json_hash(candidate_only_map),
                "sector_map_size": len(candidate_only_map),
            },
            sector_map=full_sector_map,
        )
    )

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=PaperSessionRecorder(tmp_path / "session", trade_date),
        strategy_config_source=config_manifest,
        sector_map=full_sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    assert plan.ready_to_start is False
    assert any("sector_map_hash does not match approved runtime sector map" in check.detail for check in plan.preflight.failures)


def test_runtime_session_api_has_no_default_config_escape_hatch(tmp_path):
    assert "allow_default_strategy_config" not in inspect.signature(prepare_runtime_session).parameters

    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        strategy_config_source=tmp_path / "missing_manifest.json",
        sector_map={"005930": "UNKNOWN"},
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )

    assert plan.ready_to_start is False
    assert plan.descriptors == {}
    assert any("approved_runtime_config" in check.detail for check in plan.preflight.failures)


def test_runtime_dry_run_plan_blocks_without_positive_initial_account_state(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        initial_account_state={"equity": 0.0, "buyable_cash": 0.0},
        initial_positions={},
    )

    assert plan.ready_to_start is False
    assert "runtime_initial_state_capture" in {check.name for check in plan.preflight.failures}


def test_runtime_dry_run_plan_blocks_without_recording_oms_sink(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
    )

    assert plan.ready_to_start is False
    assert "dry_run_oms_available" in {check.name for check in plan.preflight.failures}


def test_runtime_dry_run_plan_rejects_non_recording_oms_placeholder(tmp_path):
    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        dry_run_oms_client=object(),
    )

    assert plan.ready_to_start is False
    assert "dry_run_oms_available" in {check.name for check in plan.preflight.failures}


def test_runtime_dry_run_plan_rejects_submit_only_non_recording_sink(tmp_path):
    class SubmitOnly:
        async def submit_intent(self, intent):
            return None

    trade_date = date(2026, 2, 2)
    KALCBArtifactStore(tmp_path / "kalcb").save_snapshot(_kalcb_snapshot(trade_date))
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)

    plan = prepare_runtime_session(
        ("KALCB",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": tmp_path / "kalcb"},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        dry_run_oms_client=SubmitOnly(),
    )

    assert plan.ready_to_start is False
    assert "dry_run_oms_available" in {check.name for check in plan.preflight.failures}


def test_coordinator_live_mode_does_not_force_dry_run_adapter():
    with pytest.raises(ValueError, match="explicit approved config"):
        create_strategy_descriptor("KALCB", _kalcb_snapshot(date(2026, 2, 2)), mode="live", oms_client=object())

    descriptor = create_strategy_descriptor(
        "KALCB",
        _kalcb_snapshot(date(2026, 2, 2)),
        mode="live",
        oms_client=object(),
        kalcb_config=KALCBConfig(),
    )

    assert descriptor.oms_adapter is None

    descriptor = create_strategy_descriptor(
        "KALCB",
        _kalcb_snapshot(date(2026, 2, 2)),
        mode="live",
        oms_client=object(),
        kalcb_config=KALCBConfig(),
        allow_unrouted_execution=True,
    )
    assert descriptor.oms_adapter is not None
    assert descriptor.oms_adapter.dry_run is False


def test_coordinator_default_configs_are_artifact_only_and_explicit():
    snapshot = _kalcb_snapshot(date(2026, 2, 2))

    with pytest.raises(ValueError, match="unsupported runtime descriptor mode"):
        create_strategy_descriptor(
            "KALCB",
            snapshot,
            mode="paper_live",
            allow_unoptimized_defaults_for_artifact_only=True,
        )

    with pytest.raises(ValueError, match="artifact-only defaults are explicitly allowed"):
        create_strategy_descriptor("KALCB", snapshot, mode="artifact_only")

    descriptor = create_strategy_descriptor(
        "KALCB",
        snapshot,
        mode="artifact_only",
        allow_unoptimized_defaults_for_artifact_only=True,
    )
    assert descriptor.engine.config == KALCBConfig()

    with pytest.raises(ValueError, match="execution descriptor requires an explicit approved config"):
        create_strategy_descriptor(
            "KALCB",
            snapshot,
            mode="paper",
            allow_unoptimized_defaults_for_artifact_only=True,
        )


def test_coordinator_dry_run_descriptor_is_non_promotional_without_opt_in():
    descriptor = create_strategy_descriptor(
        "KALCB",
        _kalcb_snapshot(date(2026, 2, 2)),
        mode="dry_run",
        oms_client=object(),
        kalcb_config=KALCBConfig(),
    )

    assert descriptor.oms_adapter is None

    descriptor = create_strategy_descriptor(
        "KALCB",
        _kalcb_snapshot(date(2026, 2, 2)),
        mode="dry_run",
        oms_client=object(),
        kalcb_config=KALCBConfig(),
        allow_unrouted_execution=True,
    )
    assert descriptor.oms_adapter is not None
    assert descriptor.oms_adapter.dry_run is True


class _SubmitOnlyOMS:
    async def submit_intent(self, intent):
        return None


def _write_runtime_config_manifest(tmp_path, *, kalcb_payload: dict, olr_payload: dict) -> dict:
    kalcb_path = tmp_path / "configs" / "kalcb_optimized_config.json"
    olr_path = tmp_path / "configs" / "olr_optimized_config.json"
    kalcb_path.parent.mkdir(parents=True, exist_ok=True)
    kalcb_path.write_text(json.dumps({"mutations": kalcb_payload}, sort_keys=True), encoding="utf-8")
    olr_path.write_text(json.dumps({"mutations": olr_payload}, sort_keys=True), encoding="utf-8")
    return {
        "artifacts": [
            {"label": "kalcb optimized_config", "path": str(kalcb_path)},
            {"label": "olr optimized_config", "path": str(olr_path)},
        ]
    }


def _kalcb_snapshot(
    trade_date: date,
    *,
    metadata: dict | None = None,
    generated_at: datetime | None = None,
    sector_map: dict[str, str] | None = None,
) -> KALCBDailySnapshot:
    config_sector_map = dict(sector_map or {"005930": "UNKNOWN"})
    base_metadata = {
        "artifact_stage": KALCB_FINAL_ARTIFACT_STAGE,
        "source": "unit",
        "candidate_config_hash": kalcb_candidate_config_fingerprint(KALCBConfig(), {}, config_sector_map),
        "sector_map_hash": canonical_json_hash(config_sector_map),
        "sector_map_size": len(config_sector_map),
        "active_symbols": ["005930"],
        "active_symbol_count": 1,
        "active_budget_source": "ws_budget",
        "frontier_symbols": ["005930"],
        "frontier_symbol_count": 1,
        "overflow_symbols": [],
        "overflow_symbol_count": 0,
        "frontier_rest_budget_symbols_per_5m": 300,
    }
    base_metadata.update(metadata or {})
    return KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit-source",
        generated_at=generated_at or datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        candidates=(
            KALCBDailyCandidate(
                symbol="005930",
                trade_date=trade_date,
                prior_day_high=102.0,
                prior_day_low=98.0,
                prior_day_close=100.0,
                daily_atr=2.0,
                expected_5m_volume=100.0,
                average_30m_volume=600.0,
                source_fingerprint="unit-source",
            ),
        ),
        metadata=base_metadata,
    )


def _olr_snapshot(
    trade_date: date,
    stage: str,
    *,
    metadata: dict | None = None,
    generated_at: datetime | None = None,
) -> OLRDailySnapshot:
    source = "olr_afternoon_selection" if stage == OLR_FINAL_ARTIFACT_STAGE else "olr_research_selection"
    basis = "14:30_decision_from_completed_5m_bars" if stage == OLR_FINAL_ARTIFACT_STAGE else "pre_session_from_prior_completed_daily_rows"
    base_metadata = {"artifact_stage": stage, "source": source, "selection_time_basis": basis}
    if stage == OLR_FINAL_ARTIFACT_STAGE:
        final_config_hash = final_candidate_config_fingerprint(OLRConfig())
        base_metadata.update(
            {
                "candidate_config_hash": final_config_hash,
                "final_candidate_config_hash": final_config_hash,
                "final_candidate_config_hash_version": FINAL_CANDIDATE_CONFIG_HASH_VERSION,
            }
        )
    base_metadata.update(metadata or {})
    return OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit-source",
        generated_at=generated_at
        or datetime.combine(
            trade_date,
            time(14, 31) if stage == OLR_FINAL_ARTIFACT_STAGE else datetime.min.time(),
            tzinfo=KST,
        ),
        candidates=(
            OLRDailyCandidate(
                symbol="005930",
                trade_date=trade_date,
                prior_day_high=102.0,
                prior_day_low=98.0,
                prior_day_close=100.0,
                daily_atr=2.0,
                expected_5m_volume=100.0,
                average_30m_volume=600.0,
                source_fingerprint="unit-source",
            ),
        ),
        metadata=base_metadata,
    )


def _olr_daily_rows(trade_date: date, *, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    rows = []
    for index in range(days):
        day = first + timedelta(days=index)
        close = 5_000.0 + 45.0 * index
        rows.append(
            {
                "date": day.isoformat(),
                "open": close - 10.0,
                "high": close + 20.0,
                "low": close - 20.0,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows
