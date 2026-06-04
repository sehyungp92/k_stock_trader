from __future__ import annotations

import json
import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from deployment.olr_kalcb.session_capture import PaperSessionRecorder
from deployment.olr_kalcb.action_router import RuntimeActionRouter
from deployment.olr_kalcb.coordinator import StrategyRuntimeDescriptor
from deployment.olr_kalcb.dry_run_oms import RecordingOMSClient
from deployment.olr_kalcb.portfolio import PortfolioArbitrationPolicy
from deployment.olr_kalcb.portfolio_context import PortfolioContextProvider
from deployment.olr_kalcb.session_driver import RuntimeSessionDriver
from instrumentation.src.config_snapshot import effective_config_snapshot
from instrumentation.src.event_envelope import wrap_for_relay
from instrumentation.src.event_writer import JSONLEventWriter
from instrumentation.src.lineage import LineageContext, stable_hash
from instrumentation.src.missed_opportunity import MissedOpportunityLogger
from instrumentation.src.oms_exporter import OMSEventEmitter
from instrumentation.src.risk_decision import build_risk_decision_payload
from instrumentation.src.sidecar import Sidecar
from oms.config_loader import effective_risk_config_payload, load_oms_config
from oms.intent import Intent, IntentResult, IntentStatus, IntentType, RiskPayload
from oms.risk import RiskConfig, RiskDecision, RiskGateway, RiskResult
from oms.state import StateStore, StrategyAllocation
from oms_client.client import AccountState
from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_common.clock import KST
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar


class _SnapshotService:
    def capture_now(self, symbol: str):
        return SimpleNamespace(
            snapshot_id="snap",
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            bid=0.0,
            ask=0.0,
            mid=70000.0,
            spread_bps=0.0,
            last_trade_price=70000.0,
            atr_14=1200.0,
            volume_24h=100000.0,
            to_dict=lambda: {
                "snapshot_id": "snap",
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mid": 70000.0,
                "last_trade_price": 70000.0,
                "atr_14": 1200.0,
            },
        )


def test_canonical_writer_and_relay_envelope_duplicate_lineage(tmp_path):
    lineage = LineageContext(
        strategy_id="KALCB",
        deployment_id="deploy-unit",
        code_sha="abc123",
        strategy_version="s1",
        config_version="cfg1",
        portfolio_config_version="pcfg1",
        risk_config_version="rcfg1",
        allocation_version="alloc1",
    )
    writer = JSONLEventWriter(tmp_path, lineage=lineage)

    event = writer.write(
        "decision_event",
        {"decision_ref": "decision-1", "timestamp": "2026-02-02T09:30:00+09:00"},
        payload_key="decision-1",
    )

    assert event is not None
    assert event["schema_version"] == "decision_event_v1"
    assert event["strategy_id"] == "KALCB"
    assert event["lineage_gap"] is False
    rows = _jsonl(next((tmp_path / "decisions").glob("*.jsonl")))
    assert rows[0]["event_id"] == event["event_id"]

    wrapped = wrap_for_relay(event, "decision_event", bot_id="k_stock_trader")
    assert isinstance(wrapped["payload"], str)
    assert wrapped["strategy_id"] == "KALCB"
    assert wrapped["deployment_id"] == "deploy-unit"

    payload_wrapped = wrap_for_relay(
        {
            "event_id": "evt-payload-lineage",
            "event_type": "decision_event",
            "exchange_timestamp": "2026-02-02T09:30:00+09:00",
            "payload": {
                "strategy_id": "OLR",
                "deployment_id": "deploy-from-payload",
                "risk_config_version": "risk-from-payload",
                "decision_ref": "payload-decision",
            },
        },
        "decision_event",
        bot_id="k_stock_trader",
        serialize_payload=False,
    )
    assert payload_wrapped["strategy_id"] == "OLR"
    assert payload_wrapped["deployment_id"] == "deploy-from-payload"
    assert payload_wrapped["risk_config_version"] == "risk-from-payload"


def test_config_snapshot_redacts_secrets_and_marks_missing_kalcb_olr_budgets():
    snapshot = effective_config_snapshot(
        strategy_configs={"KALCB": {"threshold": 1.2}},
        risk_config={"strategy_budgets": {"PCIM": {"max_positions": 8}}, "APP_SECRET": "secret"},
        strategy_registry={"strategy_ids": ["KALCB", "OLR"]},
    )

    assert snapshot["effective_configs"]["risk"]["APP_SECRET"] == "***REDACTED***"
    assert "risk.APP_SECRET" in snapshot["redacted_keys"]
    assert snapshot["active_strategy_budget_status"]["KALCB"] == "missing_uses_global_limits"
    assert snapshot["active_strategy_budget_status"]["OLR"] == "missing_uses_global_limits"


def test_missed_opportunity_backfill_appends_revision(tmp_path):
    logger = MissedOpportunityLogger(
        {"bot_id": "k_stock_trader", "data_dir": str(tmp_path), "data_source_id": "kis_rest"},
        _SnapshotService(),
    )
    event = logger.log_missed(
        pair="005930",
        side="LONG",
        signal="unit",
        signal_id="sig",
        signal_strength=0.5,
        blocked_by="portfolio_cap",
        strategy_type="KALCB",
        exchange_timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    file_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger._update_event(event.event_metadata["event_id"], file_date, {"outcome_1h": 70100.0}, "partial")

    rows = _jsonl(next((tmp_path / "missed").glob("*.jsonl")))
    assert len(rows) == 2
    assert [row["revision"] for row in rows] == [0, 1]
    assert rows[0]["logical_event_id"] == rows[1]["logical_event_id"]
    assert rows[0]["event_metadata"]["event_id"] != rows[1]["event_metadata"]["event_id"]
    assert rows[1]["outcome_1h"] == 70100.0


def test_runtime_recorder_exports_session_streams(tmp_path):
    recorder = PaperSessionRecorder(
        tmp_path / "session",
        date(2026, 2, 2),
        assistant_event_dir=tmp_path / "assistant",
        lineage=LineageContext(strategy_id="KALCB", deployment_id="deploy-unit", code_sha="abc123"),
    )

    recorder.write_resource_plan({"plan_hash": "plan-unit", "generated_at": "2026-02-02T00:00:00+00:00"})
    recorder.write_manifest(
        {
            "mode": "dry_run",
            "strategy_ids": ["KALCB"],
            "strategy_configs": {"KALCB": {"threshold": 1.2}},
            "portfolio_policy_config": {"max_gross_exposure_pct": 0.5},
            "kis_resource_plan_hash": "plan-unit",
            "initial_account_state": {"equity": 1000000, "buyable_cash": 500000},
            "initial_positions": {"005930": {"real_qty": 3, "avg_price": 100.0, "allocations": {"KALCB": {"qty": 3}}}},
        }
    )
    recorder.append_jsonl(
        "decision_stream.jsonl",
        {
            "record_type": "runtime_no_action",
            "strategy_id": "KALCB",
            "event_ref": "event-1",
            "timestamp": "2026-02-02T09:30:00+09:00",
            "reason_code": "no_signal",
            "decision_ref": "decision-1",
        },
    )

    decisions = _jsonl(next((tmp_path / "assistant" / "decisions").glob("*.jsonl")))
    deployments = _jsonl(next((tmp_path / "assistant" / "deployments").glob("*.jsonl")))
    config_snapshots = _jsonl(next((tmp_path / "assistant" / "config_snapshots").glob("*.jsonl")))
    resources = _jsonl(next((tmp_path / "assistant" / "resource_plans").glob("*.jsonl")))
    expected_risk_config = effective_risk_config_payload(load_oms_config())
    assert decisions[0]["event_type"] == "decision_event"
    assert decisions[0]["deployment_id"] == "deploy-unit"
    assert decisions[0]["strategy_version"]
    assert decisions[0]["lineage_gap"] is False
    assert decisions[0]["config_version"]
    assert deployments[0]["strategy_version"]
    assert deployments[0]["risk_config_version"]
    assert deployments[0]["risk_config_version"] == stable_hash(expected_risk_config)
    assert deployments[0]["lineage_gap"] is False
    assert config_snapshots[0]["payload"]["risk_config_version"]
    assert config_snapshots[0]["payload"]["risk_config"] == expected_risk_config
    assert config_snapshots[0]["payload"]["risk_config"]["max_positions_count"] == 15
    assert config_snapshots[0]["payload"]["risk_config"]["strategy_budgets"]["PCIM"]["max_risk_pct"] == 0.025
    assert config_snapshots[0]["lineage_gap"] is False
    assert decisions[0]["kis_resource_plan_hash"] == "plan-unit"
    assert decisions[0]["payload"]["record_type"] == "runtime_no_action"
    assert decisions[0]["payload"]["decision_code"] == "no_action"
    assert resources[0]["payload"]["plan_hash"] == "plan-unit"
    assert resources[0]["deployment_id"] == "deploy-unit"
    assert resources[0]["portfolio_config_version"]
    assert resources[0]["risk_config_version"] == stable_hash(expected_risk_config)
    assert resources[0]["allocation_version"]
    assert resources[0]["lineage_gap"] is False
    portfolio_payload = _jsonl(next((tmp_path / "assistant" / "portfolio").glob("*.jsonl")))[0]["payload"]
    assert portfolio_payload["reason"] == "runtime_session_start"
    assert portfolio_payload["gross_exposure_krw"] == 300.0
    assert portfolio_payload["strategy_exposures"]["KALCB"]["notional_krw"] == 300.0


def test_runtime_resource_plan_waits_for_matching_manifest_lineage(tmp_path):
    recorder = PaperSessionRecorder(
        tmp_path / "session",
        date(2026, 2, 2),
        assistant_event_dir=tmp_path / "assistant",
        lineage=LineageContext(code_sha="abc123"),
    )

    recorder.write_resource_plan({"plan_hash": "plan-stage1", "generated_at": "2026-02-02T00:00:00+00:00"})
    recorder.write_manifest(
        {
            "mode": "dry_run",
            "strategy_ids": ["KALCB"],
            "strategy_configs": {"KALCB": {"threshold": 1.2}},
            "portfolio_policy_config": {"max_gross_exposure_pct": 0.5},
            "staged_artifacts": [
                {"strategy_id": "KALCB", "artifact_hash": "kalcb-stage1", "artifact_stage": "daily"}
            ],
            "kis_resource_plan_hash": "plan-stage1",
            "initial_positions": {},
        }
    )
    resources = _jsonl(next((tmp_path / "assistant" / "resource_plans").glob("*.jsonl")))
    first_resource = next(row for row in resources if row["payload"]["plan_hash"] == "plan-stage1")

    recorder.write_resource_plan({"plan_hash": "plan-final", "generated_at": "2026-02-02T05:30:00+00:00"})
    resources_after_refresh = _jsonl(next((tmp_path / "assistant" / "resource_plans").glob("*.jsonl")))
    assert [row["payload"]["plan_hash"] for row in resources_after_refresh] == ["plan-stage1"]

    recorder.write_manifest(
        {
            "mode": "dry_run",
            "strategy_ids": ["KALCB", "OLR"],
            "strategy_configs": {"KALCB": {"threshold": 1.2}, "OLR": {"enabled": True}},
            "portfolio_policy_config": {"max_gross_exposure_pct": 0.5},
            "staged_artifacts": [
                {"strategy_id": "KALCB", "artifact_hash": "kalcb-stage1", "artifact_stage": "daily"},
                {"strategy_id": "OLR", "artifact_hash": "olr-final", "artifact_stage": "final_afternoon_1430"},
            ],
            "kis_resource_plan_hash": "plan-final",
            "initial_positions": {},
        }
    )

    resources = _jsonl(next((tmp_path / "assistant" / "resource_plans").glob("*.jsonl")))
    deployments = _jsonl(next((tmp_path / "assistant" / "deployments").glob("*.jsonl")))
    final_resource = next(row for row in resources if row["payload"]["plan_hash"] == "plan-final")
    final_deployment = next(row for row in deployments if row["payload"]["kis_resource_plan_hash"] == "plan-final")
    assert final_resource["deployment_id"] == final_deployment["deployment_id"]
    assert final_resource["deployment_id"] != first_resource["deployment_id"]
    assert final_resource["strategy_registry_version"] == final_deployment["strategy_registry_version"]
    assert final_resource["lineage_gap"] is False


def test_runtime_exporter_emits_market_snapshot_and_strategy_gate_miss(tmp_path):
    recorder = PaperSessionRecorder(
        tmp_path / "session",
        date(2026, 2, 2),
        assistant_event_dir=tmp_path / "assistant",
        lineage=LineageContext(strategy_id="KALCB", deployment_id="deploy-unit", code_sha="abc123"),
    )
    recorder.write_manifest(
        {
            "mode": "dry_run",
            "strategy_ids": ["KALCB"],
            "strategy_configs": {"KALCB": {"threshold": 1.2}},
            "portfolio_policy_config": {},
            "initial_positions": {},
        }
    )
    bar = _synthetic_bar(datetime(2026, 2, 2, 9, 30, tzinfo=KST))
    bar_payload = bar.to_json_dict()
    bar_hash = "bar-unit"
    recorder.append_jsonl(
        "decision_stream.jsonl",
        {
            "record_type": "runtime_event_input",
            "strategy_id": "KALCB",
            "event_ref": "event-bar",
            "event_type": "bar",
            "timestamp": bar.timestamp.isoformat(),
            "payload": bar_payload,
            "bar_hash": bar_hash,
        },
    )
    recorder.append_jsonl(
        "decision_stream.jsonl",
        {
            "record_type": "decision_event",
            "strategy_id": "KALCB",
            "symbol": "005930",
            "timestamp": bar.timestamp.isoformat(),
            "event_ref": "event-bar",
            "decision_ref": "decision-rejected",
            "decision_code": "entry_rejected",
            "reason": "spread_gate",
            "metadata": {
                "candidate_hash": "candidate-unit",
                "source_artifact_hash": "artifact-unit",
                "source_fingerprint": "source-unit",
                "bar_rvol": 3.2,
                "avwap": 70000.0,
                "daily_atr": 1200.0,
                "momentum_score": 4,
                "entry_type": "or_breakout",
                "gate_decisions": {
                    "spread_gate": {
                        "threshold": 50,
                        "threshold_operator": "<=",
                        "actual_value": 75,
                        "margin": 25,
                        "input_refs": ["spread_bps"],
                        "passed": False,
                    }
                },
            },
        },
    )

    snapshots = _jsonl(next((tmp_path / "assistant" / "snapshots").glob("*.jsonl")))
    missed = _jsonl(next((tmp_path / "assistant" / "missed").glob("*.jsonl")))
    filters = _jsonl(next((tmp_path / "assistant" / "filter_decisions").glob("*.jsonl")))
    indicators = _jsonl(next((tmp_path / "assistant" / "indicators").glob("*.jsonl")))

    assert snapshots[0]["event_type"] == "market_snapshot"
    assert snapshots[0]["payload"]["bar_hash"] == bar_hash
    assert missed[0]["payload"]["blocked_scope"] == "strategy_filter"
    assert missed[0]["payload"]["filter_decisions"][0]["filter_name"] == "spread_gate"
    assert missed[0]["payload"]["gate_decisions"][0]["threshold_operator"] == "<="
    assert missed[0]["payload"]["gate_decisions"][0]["input_refs"] == ["spread_bps"]
    assert filters[0]["payload"]["filter_name"] == "spread_gate"
    assert filters[0]["payload"]["filter_source"] == "gate_decisions"
    assert filters[0]["payload"]["passed"] is False
    assert filters[0]["payload"]["threshold_operator"] == "<="
    assert filters[0]["payload"]["input_refs"] == ["spread_bps"]
    assert indicators[0]["event_type"] == "indicator_snapshot"
    assert indicators[0]["payload"]["indicators"]["bar_rvol"] == 3.2
    assert indicators[0]["payload"]["indicators"]["avwap"] == 70000.0
    assert indicators[0]["payload"]["context"]["entry_type"] == "or_breakout"
    assert indicators[0]["lineage_gap"] is False


def test_runtime_closeout_exports_daily_and_family_bundle(tmp_path):
    recorder = PaperSessionRecorder(
        tmp_path / "session",
        date(2026, 2, 2),
        assistant_event_dir=tmp_path / "assistant",
        lineage=LineageContext(deployment_id="deploy-unit", code_sha="abc123"),
    )
    recorder.write_manifest({"mode": "dry_run", "strategy_ids": ["KALCB", "OLR"], "kis_resource_plan_hash": "plan-unit"})

    recorder.close_session(
        {
            "positions": [
                {
                    "strategy_id": "KALCB",
                    "symbol": "005930",
                    "qty_open": 2,
                    "entry_price": 100.0,
                    "sector": "TECH",
                }
            ]
        },
        session_metrics={"total_trades": 1},
        closeout_reason="unit",
    )

    daily_rows = _jsonl(next((tmp_path / "assistant" / "daily").glob("*.jsonl")))
    family_rows = _jsonl(next((tmp_path / "assistant" / "family").glob("*.jsonl")))
    portfolio_rows = _jsonl(next((tmp_path / "assistant" / "portfolio").glob("*.jsonl")))
    closeout_manifest = json.loads((tmp_path / "session" / "session_manifest.json").read_text(encoding="utf-8"))
    assert {row["event_type"] for row in daily_rows} >= {"session_closeout", "daily_snapshot"}
    assert family_rows[0]["event_type"] == "family_daily_snapshot"
    assert family_rows[0]["payload"]["strategy_summaries"].keys() == {"KALCB", "OLR"}
    assert family_rows[0]["lineage_gap"] is False
    assert "strategy_id" not in family_rows[0]["lineage_gaps"]
    closeout_portfolio = next(row for row in portfolio_rows if row["payload"].get("reason") == "runtime_session_closeout")
    assert closeout_portfolio["payload"]["timestamp"] == closeout_manifest["closeout_generated_at"]
    assert closeout_portfolio["payload"]["gross_exposure_krw"] == 200.0
    assert closeout_portfolio["payload"]["sector_exposures"]["TECH"]["notional_krw"] == 200.0


def test_risk_trace_and_payload_builder():
    state = StateStore()
    state.equity = 100_000_000
    config = RiskConfig(strategy_budgets={"ALPHA": {"max_positions": 4, "max_risk_pct": 0.015}})
    gateway = RiskGateway(state, config, price_getter=lambda _symbol: 72000.0)
    intent = Intent(
        intent_type=IntentType.ENTER,
        strategy_id="ALPHA",
        symbol="005930",
        desired_qty=100,
        risk_payload=RiskPayload(entry_px=72000.0, stop_px=71000.0),
    )

    result = gateway.check(intent)
    payload = build_risk_decision_payload(intent, result)

    assert result.decision == RiskDecision.APPROVE
    assert [row["rule"] for row in result.trace] == [
        "global_blocks",
        "daily_limits",
        "exposure_limits",
        "sector_limits",
        "strategy_budget",
        "microstructure",
    ]
    assert payload["trace"][0]["decision"] == "APPROVE"


def test_oms_event_emitter_writes_intent_and_risk_rows(tmp_path):
    emitter = OMSEventEmitter(tmp_path, lineage=LineageContext(deployment_id="deploy-unit", code_sha="abc123"))
    emitter.update_lineage(
        strategy_version="strategy-unit",
        config_version="cfg-unit",
        portfolio_config_version="portfolio-unit",
        risk_config_version="risk-unit",
        allocation_version="allocation-unit",
        strategy_registry_version="registry-unit",
        kis_resource_plan_hash="plan-unit",
    )
    intent = Intent(
        intent_type=IntentType.ENTER,
        strategy_id="KALCB",
        symbol="005930",
        desired_qty=1,
        risk_payload=RiskPayload(entry_px=70000.0, stop_px=69000.0),
        metadata={"event_ref": "event-1", "action_ref": "action-1"},
    )
    result = IntentResult(intent_id=intent.intent_id, status=IntentStatus.REJECTED, message="unit")
    risk = RiskResult(RiskDecision.REJECT, "unit", trace=[{"rule": "unit", "decision": "REJECT"}])
    state = StateStore()
    state.equity = 1_000_000.0
    state.buyable_cash = 500_000.0
    state.update_position("005930", real_qty=2, avg_price=70000.0)
    state.update_allocation("005930", "KALCB", 2, cost_basis=70000.0)
    oms = SimpleNamespace(
        state=state,
        risk=SimpleNamespace(
            safe_mode=False,
            halt_new_entries=False,
            flatten_in_progress=False,
            config=RiskConfig(),
        ),
    )

    emitter.emit_intent(intent, result, phase="finalized")
    emitter.emit_risk_decision(intent, risk, oms=oms)

    intents = _jsonl(next((tmp_path / "oms_intents").glob("*.jsonl")))
    risks = _jsonl(next((tmp_path / "risk_decisions").glob("*.jsonl")))
    assert intents[0]["payload"]["intent_id"] == intent.intent_id
    assert intents[0]["action_ref"] == "action-1"
    assert risks[0]["payload"]["trace"][0]["rule"] == "unit"
    assert risks[0]["payload"]["current_state_summary"]["equity_krw"] == 1_000_000.0
    assert risks[0]["payload"]["current_state_summary"]["symbol_exposures"]["005930"]["real_qty"] == 2
    assert risks[0]["payload"]["current_state_summary"]["intent_symbol_position"]["symbol"] == "005930"
    assert risks[0]["deployment_id"] == "deploy-unit"
    assert risks[0]["strategy_version"] == "strategy-unit"
    assert risks[0]["portfolio_config_version"] == "portfolio-unit"
    assert risks[0]["risk_config_version"] == "risk-unit"
    assert risks[0]["allocation_version"] == "allocation-unit"
    assert risks[0]["kis_resource_plan_hash"] == "plan-unit"
    assert risks[0]["lineage_gap"] is False


def test_oms_strategy_attributed_rows_require_strategy_versions(tmp_path):
    emitter = OMSEventEmitter(
        tmp_path,
        lineage=LineageContext(
            deployment_id="deploy-unit",
            code_sha="abc123",
            portfolio_config_version="portfolio-unit",
            risk_config_version="risk-unit",
            allocation_version="allocation-unit",
        ),
    )
    intent = Intent(
        intent_type=IntentType.ENTER,
        strategy_id="KALCB",
        symbol="005930",
        desired_qty=1,
        risk_payload=RiskPayload(entry_px=70000.0, stop_px=69000.0),
        metadata={"event_ref": "event-1", "action_ref": "action-1"},
    )
    risk = RiskResult(RiskDecision.REJECT, "unit", trace=[{"rule": "unit", "decision": "REJECT"}])

    emitter.emit_risk_decision(intent, risk)

    risks = _jsonl(next((tmp_path / "risk_decisions").glob("*.jsonl")))
    assert risks[0]["strategy_id"] == "KALCB"
    assert risks[0]["lineage_gap"] is True
    assert set(risks[0]["lineage_gaps"]) == {"strategy_version", "config_version"}


def test_oms_event_emitter_marks_inferred_fills(tmp_path):
    emitter = OMSEventEmitter(tmp_path, lineage=LineageContext(deployment_id="deploy-unit", code_sha="abc123"))
    order = SimpleNamespace(
        order_id="order-1",
        oms_order_id="",
        symbol="005930",
        side="BUY",
        qty=10,
        filled_qty=0,
        price=70000.0,
        strategy_id="KALCB",
        intent_id="intent-1",
        idempotency_key="idem-1",
    )

    emitter.emit_fill(order, 3, inferred=True)

    fills = _jsonl(next((tmp_path / "fills").glob("*.jsonl")))
    assert fills[0]["payload"]["inferred"] is True


def test_oms_snapshots_include_exposure_drift_and_unknown_allocations(tmp_path):
    state = StateStore()
    state.equity = 1_000_000.0
    state.buyable_cash = 400_000.0
    state.strategy_realized_pnl["KALCB"] = 1250.0
    pos = state.get_position("005930")
    pos.real_qty = 5
    pos.avg_price = 100.0
    pos.allocations["KALCB"] = StrategyAllocation(strategy_id="KALCB", qty=3, cost_basis=100.0)
    pos.allocations["_UNKNOWN_"] = StrategyAllocation(strategy_id="_UNKNOWN_", qty=2, cost_basis=100.0)

    risk = SimpleNamespace(
        safe_mode=False,
        halt_new_entries=False,
        flatten_in_progress=False,
        config=SimpleNamespace(current_regime="NORMAL", regime_exposure_caps={"NORMAL": 0.8}),
        _sector_exposure=SimpleNamespace(
            sym_to_sector={"005930": "TECH"},
            sector_working_count={"TECH": 1},
            sector_working_notional={"TECH": 250.0},
        ),
    )
    oms = SimpleNamespace(state=state, risk=risk)
    emitter = OMSEventEmitter(tmp_path, lineage=LineageContext(deployment_id="deploy-unit", code_sha="abc123"))

    emitter.emit_position_snapshot(state, reason="unit")
    emitter.emit_allocation_snapshot(state, reason="unit")
    emitter.emit_portfolio_snapshot(oms, reason="unit")

    position_payload = _jsonl(next((tmp_path / "positions").glob("*.jsonl")))[0]["payload"]["positions"][0]
    allocation_payloads = _jsonl(next((tmp_path / "allocations").glob("*.jsonl")))[0]["payload"]["allocations"]
    portfolio_payload = _jsonl(next((tmp_path / "portfolio").glob("*.jsonl")))[0]["payload"]

    assert position_payload["total_allocated_qty"] == 5
    assert position_payload["allocation_drift"] == 0
    assert position_payload["unknown_allocation"]["qty"] == 2
    assert next(row for row in allocation_payloads if row["strategy_id"] == "KALCB")["realized_pnl_krw"] == 1250.0
    assert portfolio_payload["gross_exposure_krw"] == 500.0
    assert portfolio_payload["symbol_exposures"]["005930"]["allocation_drift"] == 0
    assert portfolio_payload["strategy_exposures"]["KALCB"]["notional_krw"] == 300.0
    assert portfolio_payload["sector_exposures"]["TECH"]["notional_krw"] == 500.0
    assert portfolio_payload["pending_reservations"]["sector_working_notional"]["TECH"] == 250.0


def test_writer_accepts_epoch_seconds(tmp_path):
    writer = JSONLEventWriter(tmp_path, lineage=LineageContext(deployment_id="deploy-unit", code_sha="abc123"))

    event = writer.write("heartbeat", {"timestamp": 1770000000.0}, payload_key="epoch")

    assert event is not None
    assert event["exchange_timestamp"].startswith("2026-")


def test_synthetic_day_e2e_assistant_bundle_and_sidecar_contract(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAY_URL", raising=False)
    trade_date = date(2026, 2, 2)
    assistant_dir = tmp_path / "assistant"
    recorder = PaperSessionRecorder(
        tmp_path / "session",
        trade_date,
        assistant_event_dir=assistant_dir,
        lineage=LineageContext(
            strategy_id="KALCB",
            deployment_id="deploy-synth",
            code_sha="abc123",
            strategy_version="unit-strategy",
            risk_config_version="risk-unit",
            allocation_version="alloc-unit",
        ),
    )
    recorder.write_resource_plan({"plan_hash": "plan-unit", "generated_at": "2026-02-02T00:00:00+00:00"})
    recorder.write_manifest(
        {
            "mode": "dry_run",
            "strategy_ids": ["KALCB", "OLR"],
            "strategy_configs": {"KALCB": {"entry_threshold": 1.2}, "OLR": {"enabled": True}},
            "portfolio_policy_config": {"max_gross_exposure_pct": 0.5},
            "kis_resource_plan_hash": "plan-unit",
            "initial_account_state": {"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
            "initial_positions": {},
        }
    )
    oms = RecordingOMSClient(recorder, account_state=AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0))
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    engine = _SyntheticLifecycleEngine()
    descriptor = StrategyRuntimeDescriptor("KALCB", "unit_stage", "artifact-unit", engine, SimpleNamespace(source_fingerprint="source-unit", candidates=()))
    driver = RuntimeSessionDriver(descriptor, router, recorder, context, "dry_run")

    asyncio.run(driver.handle_bar(_synthetic_bar(datetime(2026, 2, 2, 9, 30, tzinfo=KST))))
    entry_order_id = _jsonl(tmp_path / "session" / "order_events.jsonl")[-1]["order_id"]
    asyncio.run(
        driver.handle_fill(
            SimpleNamespace(
                order_id=entry_order_id,
                symbol="005930",
                side="BUY",
                qty=2,
                price=100.0,
                timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
                reason="entry_fill",
                metadata={},
            )
        )
    )
    asyncio.run(driver.handle_timer(datetime(2026, 2, 2, 10, 0, tzinfo=KST)))
    asyncio.run(driver.handle_bar(_synthetic_bar(datetime(2026, 2, 2, 10, 5, tzinfo=KST))))
    exit_order_id = _jsonl(tmp_path / "session" / "order_events.jsonl")[-1]["order_id"]
    asyncio.run(
        driver.handle_fill(
            SimpleNamespace(
                order_id=exit_order_id,
                symbol="005930",
                side="SELL",
                qty=2,
                price=105.0,
                timestamp=datetime(2026, 2, 2, 10, 10, tzinfo=KST),
                reason="exit_fill",
                metadata={},
            )
        )
    )
    recorder.close_session({"positions": []}, session_metrics={"replay_parity_status": "synthetic_pass"}, closeout_reason="synthetic_eod")

    rows = _assistant_rows(assistant_dir)
    event_types = {row["event_type"] for row in rows}
    assert {
        "deployment",
        "config_snapshot",
        "resource_plan",
        "decision_event",
        "strategy_action",
        "portfolio_rule",
        "oms_intent",
        "order",
        "fill",
        "trade",
        "position_snapshot",
        "allocation_snapshot",
        "portfolio_snapshot",
        "session_closeout",
        "daily_snapshot",
        "family_daily_snapshot",
    }.issubset(event_types)

    trade = next(row for row in rows if row["event_type"] == "trade")
    trade_payload = trade["payload"]
    for key in (
        "trade_id",
        "decision_ref",
        "action_ref",
        "portfolio_decision_ref",
        "intent_id",
        "order_id",
        "broker_order_id",
        "artifact_hash",
        "config_version",
        "deployment_id",
        "kis_resource_plan_hash",
    ):
        assert trade_payload.get(key), key
    assert all(trade_payload["join_completeness"].values())

    assert any(row["event_type"] == "portfolio_snapshot" and row["payload"].get("reason") == "fill_applied" for row in rows)
    assert any(row["event_type"] == "allocation_snapshot" and row["payload"].get("reason") == "fill_applied" for row in rows)
    assert any(
        row["event_type"] == "portfolio_snapshot"
        and row["payload"].get("reason") == "fill_applied"
        and row["payload"].get("symbol_exposures", {}).get("005930", {}).get("notional_krw") == 200.0
        for row in rows
    )
    family = next(row for row in rows if row["event_type"] == "family_daily_snapshot")["payload"]
    assert family["strategy_summaries"]["KALCB"]["total_trades"] == 1
    assert family["strategy_summaries"]["KALCB"]["fills"] == 2
    assert family["portfolio_summary"]["resource_plan_suppressions"] == 0

    sidecar = Sidecar({"bot_id": "k_stock_trader", "data_dir": str(assistant_dir), "sidecar": {"buffer_dir": str(tmp_path / "buffer")}})
    wrapped = [event for path, event_type in sidecar._get_event_files() for event in sidecar._read_unsent_events(path, event_type)]
    wrapped_types = {event["event_type"] for event in wrapped}
    assert {"trade", "fill", "session_closeout", "family_daily_snapshot"}.issubset(wrapped_types)
    assert all(event.get("event_id") and event.get("schema_version") and event.get("exchange_timestamp") for event in wrapped)


class _SyntheticLifecycleEngine:
    def __init__(self):
        self.state = SimpleNamespace(symbols={"005930": SimpleNamespace(position=None)})

    def on_bar(self, bar, portfolio, submit):
        symbol_state = self.state.symbols["005930"]
        if symbol_state.position is None:
            action = SubmitEntry(
                "KALCB",
                bar.symbol,
                2,
                "LIMIT",
                100.0,
                None,
                "synthetic_entry",
                metadata={"candidate_hash": "candidate-unit", "source_artifact_hash": "artifact-unit", "source_fingerprint": "source-unit"},
            )
            submit(action)
            return [DecisionEvent(bar.timestamp, "KALCB", bar.symbol, "entry", "synthetic_entry", actions=(action,))]
        action = SubmitExit("KALCB", bar.symbol, 2, "LIMIT", 105.0, "synthetic_exit", metadata={"order_role": "EXIT"})
        submit(action)
        return [DecisionEvent(bar.timestamp, "KALCB", bar.symbol, "exit", "synthetic_exit", actions=(action,))]

    def on_timer(self, timestamp, submit):
        return []

    def on_fill(self, fill, submit):
        metadata = dict(getattr(fill, "metadata", {}) or {})
        if str(fill.side).upper() == "BUY":
            self.state.symbols["005930"].position = {
                "symbol": "005930",
                "qty_open": int(fill.qty),
                "entry_price": float(fill.price),
                "entry_time": fill.timestamp,
                "entry_order_id": fill.order_id,
                "source_artifact_hash": metadata.get("source_artifact_hash", "artifact-unit"),
                "source_fingerprint": metadata.get("source_fingerprint", "source-unit"),
                "candidate_hash": metadata.get("candidate_hash", "candidate-unit"),
                "metadata": metadata,
            }
        else:
            self.state.symbols["005930"].position = None
        return []


def _synthetic_bar(timestamp: datetime) -> MarketBar:
    return MarketBar("005930", timestamp, "5m", 100.0, 106.0, 99.0, 105.0, 1000.0, True)


def _assistant_rows(data_dir):
    rows = []
    for path in sorted(data_dir.glob("*/*.jsonl")):
        rows.extend(_jsonl(path))
    return rows


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
