from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from oms_client.client import AccountState, AllocationInfo, PositionInfo, WorkingOrderInfo

from deployment.olr_kalcb.action_router import RuntimeActionRouter
from deployment.olr_kalcb.coordinator import StrategyRuntimeDescriptor
from deployment.olr_kalcb.dry_run_oms import RecordingOMSClient
from deployment.olr_kalcb.hashing import canonical_json_hash
from deployment.olr_kalcb.offline_replay import ReplayInputLoader, rebuild_offline_replay_from_session
from deployment.olr_kalcb.portfolio import PortfolioArbitrationPolicy, PortfolioPolicyConfig
from deployment.olr_kalcb.portfolio_context import PortfolioContextProvider
from deployment.olr_kalcb.replay import replay_paper_session
from deployment.olr_kalcb.runtime import prepare_runtime_session
from deployment.olr_kalcb.session_capture import PaperSessionRecorder, REQUIRED_EXPECTED_HASH_GROUPS, session_hashes
from deployment.olr_kalcb.session_driver import ActionCollector, RuntimeSessionDriver
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.state import KALCBPositionState, KALCBState, SymbolStage
from strategy_kalcb.engine import KALCBEngine
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot
from strategy_kalcb.research import KALCB_FINAL_ARTIFACT_STAGE, candidate_config_fingerprint as kalcb_candidate_config_fingerprint
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE, OLRArtifactStore
from strategy_olr.config import OLRConfig
from strategy_olr.engine import OLREngine
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.research import FINAL_CANDIDATE_CONFIG_HASH_VERSION, final_candidate_config_fingerprint


def test_offline_replay_rebuilds_positive_no_trade_session(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 25), include_fill=False)

    offline_root = rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    assert offline_root.is_dir()
    assert _jsonl(session / "strategy_actions.jsonl") == []
    assert report["offline_rebuild_implemented"] is True
    assert report["paper_gate_passed"] is True


def test_offline_replay_rebuilds_positive_action_session(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 30), include_fill=False)

    rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    assert len(_jsonl(session / "strategy_actions.jsonl")) == 1
    assert _jsonl(session / "portfolio_arbitration.jsonl")[0]["decision"] == "accepted"
    assert report["paper_gate_passed"] is True


def test_offline_replay_preserves_startup_working_order_snapshot_parity(tmp_path):
    session = _captured_olr_session(
        tmp_path,
        decision_time=time(14, 30),
        include_fill=False,
        startup_working_order=True,
    )

    rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    live_portfolio = _jsonl(session / "portfolio_arbitration.jsonl")
    offline_portfolio = _jsonl(session / "offline_replay" / "portfolio_arbitration.jsonl")
    replay_manifest = json.loads((session / "offline_replay" / "replay_manifest.json").read_text(encoding="utf-8"))

    assert live_portfolio[0]["record_type"] == "pending_reservations_rehydrated"
    assert live_portfolio[0]["working_orders"][0]["remaining_qty"] == 6
    assert live_portfolio[-1]["decision"] == "blocked"
    assert live_portfolio[-1]["reason_code"] == "duplicate_symbol_conflict"
    assert offline_portfolio == live_portfolio
    assert replay_manifest["startup_working_order_count"] == 1
    assert replay_manifest["startup_working_order_source"] == "oms_positions"
    assert report["paper_gate_passed"] is True


def test_paper_gate_rejects_sealed_manifest_with_missing_expected_hash_group(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 30), include_fill=False)
    rebuild_offline_replay_from_session(session)
    manifest_path = session / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    removed_group = REQUIRED_EXPECTED_HASH_GROUPS[0]
    manifest["expected_hashes"].pop(removed_group)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    report = replay_paper_session(session)

    assert report["hash_contract_status"] == "sealed"
    assert report["hash_contract_available"] is False
    assert report["paper_gate_passed"] is False
    assert removed_group in report["hash_contract_missing_expected_groups"]
    assert "hash_contract_incomplete" in report["promotion_blockers"]


def test_offline_replay_rebuilds_positive_fill_session(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 30), include_fill=True)

    rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    fill_rows = _jsonl(session / "fill_events.jsonl")
    eod = json.loads((session / "end_of_day_positions.json").read_text(encoding="utf-8"))
    assert fill_rows[0]["event"]["order_id"].startswith("OLR:")
    assert fill_rows[0]["event"]["metadata"]["broker_order_id"].startswith("dry-run:OLR:")
    assert eod["positions"][0]["entry_order_id"].startswith("OLR:")
    assert eod["positions"][0]["candidate_rank"] == 1
    assert eod["positions"][0]["sector"] == "SEMIS"
    assert eod["positions"][0]["source_artifact_hash"]
    assert report["paper_gate_passed"] is True


def test_offline_replay_requires_complete_artifact_evidence(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 25), include_fill=False)
    for path in (session / "olr_stage1_snapshots").glob("*.json"):
        path.unlink()

    report = replay_paper_session(session)
    with pytest.raises(FileNotFoundError, match="olr_stage1_snapshots"):
        rebuild_offline_replay_from_session(session)

    assert report["paper_gate_passed"] is False
    assert any("olr_stage1_snapshots" in item for item in report["missing_artifact_evidence"])


def test_offline_replay_requires_artifact_generation_to_match_staged_snapshot(tmp_path):
    session = _captured_olr_session(tmp_path / "mismatch", decision_time=time(14, 25), include_fill=False)
    rows = _jsonl(session / "artifact_generation.jsonl")
    rows[0]["candidate_count"] = int(rows[0]["candidate_count"]) + 1
    (session / "artifact_generation.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = replay_paper_session(session)
    with pytest.raises(FileNotFoundError, match="artifact_generation:OLR"):
        rebuild_offline_replay_from_session(session)

    assert report["paper_gate_passed"] is False
    assert any(item.startswith("artifact_generation:OLR") for item in report["missing_artifact_evidence"])


def test_offline_replay_regenerates_non_candidate_no_action_rows(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 25), include_fill=False, include_non_candidate_bar=True)

    rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    live_rows = _jsonl(session / "decision_stream.jsonl")
    offline_rows = _jsonl(session / "offline_replay" / "decision_stream.jsonl")
    assert any(row.get("reason_code") == "symbol_not_in_snapshot" for row in live_rows)
    assert any(row.get("reason_code") == "symbol_not_in_snapshot" for row in offline_rows)
    assert report["paper_gate_passed"] is True


def test_runtime_event_inputs_are_sequenced_and_reference_market_bar_rows(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 25), include_fill=False, include_non_candidate_bar=True)

    events = [row for row in _jsonl(session / "decision_stream.jsonl") if row.get("record_type") == "runtime_event_input"]
    hashes = session_hashes(session)

    assert [row["event_sequence"] for row in events] == list(range(1, len(events) + 1))
    assert all(row.get("bar_hash") and row.get("bar_row_key") == row.get("bar_hash") for row in events if row["event_type"] == "bar")

    rows = _jsonl(session / "decision_stream.jsonl")
    rows[0]["event_sequence"] = 99
    (session / "decision_stream.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert session_hashes(session)["runtime_events"] != hashes["runtime_events"]


def test_offline_replay_uses_market_bars_parquet_as_bar_authority(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 25), include_fill=False)

    import pandas as pd

    path = session / "market_bars_5m.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "close"] = float(frame.loc[0, "close"]) + 1.0
    frame.to_parquet(path, index=False)

    with pytest.raises(FileNotFoundError, match="market_bars_5m row hash"):
        rebuild_offline_replay_from_session(session)


def test_session_recorder_close_session_seals_expected_hashes(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 25), include_fill=False)
    eod = json.loads((session / "end_of_day_positions.json").read_text(encoding="utf-8"))
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))

    recorder.close_session(eod, {"strategy_trade_attempts": 0})
    manifest = json.loads((session / "session_manifest.json").read_text(encoding="utf-8"))
    hashes = session_hashes(session)

    assert manifest["hash_contract_status"] == "sealed"
    assert manifest["expected_hashes"] == {key: hashes[key] for key in REQUIRED_EXPECTED_HASH_GROUPS}
    assert manifest["expected_hash_groups"] == list(REQUIRED_EXPECTED_HASH_GROUPS)
    assert manifest["expected_hashes_complete"] is True
    assert manifest["session_metrics"]["strategy_trade_attempts"] == 0


def test_replay_input_loader_preserves_runtime_event_input_order_for_equal_timestamps(tmp_path):
    trade_date = date(2026, 1, 5)
    session = tmp_path / "session"
    timestamp = datetime.combine(trade_date, time(14, 30), tzinfo=KST)
    _write_json(session / "session_manifest.json", {"trade_date": trade_date.isoformat(), "strategy_ids": ["KALCB", "OLR"]})
    bar_payload = MarketBar("005930", timestamp, "5m", 100.0, 101.0, 99.0, 100.0, 1_000.0).to_json_dict()
    fill_payload = {
        "order_id": "OLR:entry:1",
        "symbol": "005930",
        "side": "BUY",
        "qty": 1,
        "price": 100.0,
        "timestamp": timestamp.isoformat(),
        "metadata": {"provisional_order_ref": "OLR:entry:1"},
    }
    fill_ref = _runtime_event_ref("OLR", "fill", fill_payload)
    decision_rows = [
        {
            "record_type": "runtime_event_input",
            "event_sequence": 1,
            "strategy_id": "KALCB",
            "event_ref": _runtime_event_ref("KALCB", "bar", bar_payload),
            "event_type": "bar",
            "timestamp": timestamp.isoformat(),
            "payload": bar_payload,
        },
        {
            "record_type": "runtime_event_input",
            "event_sequence": 2,
            "strategy_id": "OLR",
            "event_ref": fill_ref,
            "event_type": "fill",
            "timestamp": timestamp.isoformat(),
            "payload": fill_payload,
        },
        {
            "record_type": "runtime_event_input",
            "event_sequence": 3,
            "strategy_id": "OLR",
            "event_ref": _runtime_event_ref("OLR", "bar", bar_payload),
            "event_type": "bar",
            "timestamp": timestamp.isoformat(),
            "payload": bar_payload,
        },
    ]
    (session / "decision_stream.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in decision_rows) + "\n",
        encoding="utf-8",
    )
    (session / "fill_events.jsonl").write_text(
        json.dumps(
            {
                "record_type": "runtime_fill_event",
                "strategy_id": "OLR",
                "event_ref": fill_ref,
                "timestamp": timestamp.isoformat(),
                "event": fill_payload,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    events = ReplayInputLoader(session).load_runtime_events()

    assert [(kind, payload[0]) for _timestamp, kind, payload in events] == [
        ("strategy_bar", "KALCB"),
        ("fill", "OLR"),
        ("strategy_bar", "OLR"),
    ]


def test_replay_input_loader_groups_adjacent_same_bar_rows_for_combined_routing(tmp_path):
    trade_date = date(2026, 1, 5)
    session = tmp_path / "session"
    timestamp = datetime.combine(trade_date, time(14, 30), tzinfo=KST)
    _write_json(session / "session_manifest.json", {"trade_date": trade_date.isoformat(), "strategy_ids": ["KALCB", "OLR"]})
    bar_payload = MarketBar("005930", timestamp, "5m", 100.0, 101.0, 99.0, 100.0, 1_000.0).to_json_dict()
    bar_hash = "unit-same-bar-hash"
    decision_rows = [
        {
            "record_type": "runtime_event_input",
            "event_sequence": 1,
            "strategy_id": "KALCB",
            "event_ref": _runtime_event_ref("KALCB", "bar", bar_payload),
            "event_type": "bar",
            "timestamp": timestamp.isoformat(),
            "bar_hash": bar_hash,
            "bar_row_key": bar_hash,
            "payload": bar_payload,
        },
        {
            "record_type": "runtime_event_input",
            "event_sequence": 2,
            "strategy_id": "OLR",
            "event_ref": _runtime_event_ref("OLR", "bar", bar_payload),
            "event_type": "bar",
            "timestamp": timestamp.isoformat(),
            "bar_hash": bar_hash,
            "bar_row_key": bar_hash,
            "payload": bar_payload,
        },
    ]
    (session / "decision_stream.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in decision_rows) + "\n",
        encoding="utf-8",
    )

    events = ReplayInputLoader(session).load_runtime_events()

    assert len(events) == 1
    assert events[0][1] == "combined_bar"
    strategy_ids, bar = events[0][2]
    assert strategy_ids == ("KALCB", "OLR")
    assert bar.symbol == "005930"


def test_offline_replay_requires_driver_runtime_event_inputs(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 25), include_fill=False)
    rows = [
        row
        for row in _jsonl(session / "decision_stream.jsonl")
        if row.get("record_type") != "runtime_event_input"
    ]
    (session / "decision_stream.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime_event_input"):
        rebuild_offline_replay_from_session(session)


def test_offline_replay_regenerates_closed_trade_outcomes(tmp_path):
    session = _captured_olr_session(tmp_path, decision_time=time(14, 30), include_fill=True, include_exit_fill=True)

    rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    live_outcomes = _jsonl(session / "trade_outcomes.jsonl")
    offline_outcomes = _jsonl(session / "offline_replay" / "trade_outcomes.jsonl")
    assert len(live_outcomes) == 1
    assert live_outcomes == offline_outcomes
    assert live_outcomes[0]["record_type"] == "runtime_trade_outcome"
    assert live_outcomes[0]["position_closed"] is True
    assert report["paper_gate_passed"] is True


def test_runtime_engines_defer_order_memory_for_runtime_collector():
    action = SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "unit_entry")
    collector = ActionCollector("KALCB", "event", "bar", datetime(2026, 2, 2, 9, 30, tzinfo=KST))
    kalcb = KALCBEngine()

    kalcb._submit_actions([action], collector.submit)

    assert kalcb.state.symbol_state("005930").pending_entry_order_id == ""
    kalcb.reconcile_submitted_order(collector.actions[0].provisional_order_ref, collector.actions[0].action)
    assert kalcb.state.symbol_state("005930").pending_entry_order_id.startswith("KALCB:")

    olr_action = SubmitEntry("OLR", "005930", 1, "LIMIT", 100.0, None, "unit_entry")
    olr_collector = ActionCollector("OLR", "event", "bar", datetime(2026, 2, 2, 14, 30, tzinfo=KST))
    olr = OLREngine()

    olr._submit_actions([olr_action], olr_collector.submit)

    assert olr.state.symbol_state("005930").pending_entry_order_id == ""
    olr.reconcile_submitted_order(olr_collector.actions[0].provisional_order_ref, olr_collector.actions[0].action)
    assert olr.state.symbol_state("005930").pending_entry_order_id.startswith("OLR:")


def test_golden_runtime_replay_contract_covers_event_source_paths(tmp_path):
    trade_date = date(2026, 2, 2)
    exit_date = trade_date + timedelta(days=1)
    session = tmp_path / "session"
    kalcb_root = tmp_path / "kalcb"
    olr_root = tmp_path / "olr"
    sector_map = {"035420": "INTERNET", "005930": "SEMIS", "000660": "SEMIS"}
    KALCBArtifactStore(kalcb_root).save_snapshot(
        _kalcb_snapshot(trade_date, symbol="035420", sector="INTERNET", sector_map=sector_map)
    )
    olr_store = OLRArtifactStore(olr_root)
    olr_store.save_snapshot(_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE, symbols=("005930", "000660")))
    olr_store.save_snapshot(_olr_snapshot(trade_date, OLR_FINAL_ARTIFACT_STAGE, symbols=("005930", "000660")))
    config_manifest = _write_runtime_config_manifest(
        tmp_path,
        kalcb_payload={"kalcb.session.flatten_time": "15:20", "kalcb.carry.mode": "off"},
        olr_payload={"olr.execution.auction_limit_offset_bps": 100.0},
    )
    initial_positions = {
        "035420": PositionInfo(
            symbol="035420",
            real_qty=10,
            avg_price=100.0,
            allocations={"KALCB": AllocationInfo("KALCB", qty=10, cost_basis=100.0)},
        )
    }
    recorder = PaperSessionRecorder(session, trade_date)
    plan = prepare_runtime_session(
        ("KALCB", "OLR"),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"KALCB": kalcb_root, "OLR": olr_root},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        portfolio_config=PortfolioPolicyConfig(max_gross_notional=300_000.0, max_symbol_notional=300_000.0, max_sector_notional=500_000.0),
        sector_map=sector_map,
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions=initial_positions,
        initial_strategy_states={"KALCB": _kalcb_position_state(trade_date, "035420")},
    )
    assert plan.ready_to_start is True
    assert plan.strategy_config_summaries["KALCB"]["uses_defaults"] is False
    assert plan.strategy_config_summaries["OLR"]["uses_defaults"] is False

    _handle_bar_all(plan, MarketBar("123456", datetime.combine(trade_date, time(14, 20), tzinfo=KST), "5m", 100.0, 101.0, 99.0, 100.0, 1000.0))
    _handle_bar_all(plan, _bar(trade_date, time(14, 36), close=100.0))
    _handle_bar_all(plan, MarketBar("000660", datetime.combine(trade_date, time(14, 37), tzinfo=KST), "5m", 100.0, 101.0, 99.0, 100.0, 1000.0))
    _handle_bar_all(plan, MarketBar("035420", datetime.combine(trade_date, time(15, 15), tzinfo=KST), "5m", 100.0, 102.0, 99.0, 101.0, 1000.0))
    asyncio.run(plan.drivers["KALCB"].handle_timer(datetime.combine(trade_date, time(15, 20), tzinfo=KST)))

    entry_action = _latest_action(session, strategy_id="OLR", action_type="SubmitEntry", symbol="005930")
    entry_order = _order_for_action(session, entry_action)
    asyncio.run(plan.drivers["OLR"].handle_fill(_fill(entry_order["order_id"], "BUY", entry_action["action"]["qty"], 100.0, trade_date, time(15, 30))))

    _handle_bar_all(plan, _bar(exit_date, time(14, 36), close=105.0))
    exit_action = _latest_action(session, strategy_id="OLR", action_type="SubmitExit", symbol="005930", reason="next_close_exit")
    exit_order = _order_for_action(session, exit_action)
    asyncio.run(
        plan.drivers["OLR"].handle_order_event(
            SimpleNamespace(
                order_id=exit_order["order_id"],
                symbol="005930",
                status="EXPIRED",
                side="SELL",
                order_type="CLOSE_AUCTION",
                qty=exit_action["action"]["qty"],
                timestamp=datetime.combine(exit_date, time(15, 31), tzinfo=KST),
                reason="auction_expired",
                metadata={},
            )
        )
    )
    fallback_action = _latest_action(session, strategy_id="OLR", action_type="SubmitExit", symbol="005930", reason="auction_exit_nonfill_market_fallback")
    fallback_order = _order_for_action(session, fallback_action)
    asyncio.run(plan.drivers["OLR"].handle_fill(_fill(fallback_order["order_id"], "SELL", fallback_action["action"]["qty"], 105.0, exit_date, time(15, 32))))

    recorder.close_session(_end_positions({sid: descriptor.engine for sid, descriptor in plan.descriptors.items()}))

    rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    live_decisions = _jsonl(session / "decision_stream.jsonl")
    live_actions = _jsonl(session / "strategy_actions.jsonl")
    live_portfolio = _jsonl(session / "portfolio_arbitration.jsonl")
    replay_manifest = json.loads((session / "offline_replay" / "replay_manifest.json").read_text(encoding="utf-8"))
    assert any(row.get("record_type") == "runtime_no_action" for row in live_decisions)
    assert any(row.get("decision_code") == "EXIT_FALLBACK_SUBMITTED" for row in live_decisions)
    assert any(row["action_type"] == "FlattenPosition" for row in live_actions)
    assert any(row["decision"] == "accepted" and row["strategy_id"] == "OLR" and row["symbol"] == "005930" for row in live_portfolio)
    assert any(row["decision"] in {"blocked", "resized"} and row["strategy_id"] == "OLR" and row["symbol"] == "000660" for row in live_portfolio)
    assert len(_jsonl(session / "trade_outcomes.jsonl")) == 1
    assert replay_manifest["timer_replay_status"] == "replayed"
    assert report["paper_gate_passed"] is True


def test_offline_replay_replays_kalcb_timer_flatten(tmp_path):
    trade_date = date(2026, 2, 2)
    session = tmp_path / "session"
    recorder = PaperSessionRecorder(session, trade_date)
    snapshot = _kalcb_snapshot(trade_date)
    config = KALCBConfig.from_mapping({"kalcb.session.flatten_time": "15:20", "kalcb.carry.mode": "off"})
    account = AccountState(equity=1_000_000.0, buyable_cash=900_000.0)
    positions = {
        "005930": PositionInfo(
            symbol="005930",
            real_qty=10,
            avg_price=100.0,
            allocations={"KALCB": AllocationInfo("KALCB", qty=10, cost_basis=100.0)},
        )
    }
    _stage_snapshot(session, recorder, snapshot, "daily_snapshots", f"kalcb_{trade_date.isoformat()}.json")
    engine = KALCBEngine(config=config, candidate_snapshot=snapshot)
    symbol_state = engine.state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime.combine(trade_date, time(9, 35), tzinfo=KST),
        initial_stop=95.0,
        current_stop=95.0,
        risk_per_share=5.0,
        entry_type="unit",
        momentum_score=3,
        sector="SEMIS",
    )
    oms = RecordingOMSClient(recorder, account_state=account, positions=positions)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms_client=oms, sector_map={"005930": "SEMIS"})
    context.account_state = account
    context.positions = positions
    descriptor = StrategyRuntimeDescriptor("KALCB", KALCB_FINAL_ARTIFACT_STAGE, snapshot.artifact_hash, engine, snapshot)
    driver = RuntimeSessionDriver(descriptor, router, recorder, context, mode="dry_run")

    router.record_state_snapshot(
        "KALCB",
        engine.state,
        metadata={
            "record_reason": "runtime_session_pre_start",
            "mode": "dry_run",
            "trade_date": trade_date.isoformat(),
            "artifact_stage": descriptor.artifact_stage,
            "artifact_hash": descriptor.artifact_hash,
        },
    )
    asyncio.run(driver.handle_bar(MarketBar("000660", datetime.combine(trade_date, time(15, 15), tzinfo=KST), "5m", 100.0, 101.0, 99.0, 100.0, 1000.0)))
    asyncio.run(driver.handle_timer(datetime.combine(trade_date, time(15, 20), tzinfo=KST)))
    manifest = _session_manifest(
        strategy_id="KALCB",
        config=config,
        config_payload={"kalcb.session.flatten_time": "15:20", "kalcb.carry.mode": "off"},
        initial_account_state=asdict(account),
        initial_positions={
            "005930": {
                "real_qty": 10,
                "avg_price": 100.0,
                "allocations": {"KALCB": {"qty": 10, "cost_basis": 100.0}},
            }
        },
        policy_config=PortfolioPolicyConfig(),
        policy=PortfolioArbitrationPolicy(),
        sector_map={"005930": "SEMIS"},
    )
    recorder.write_manifest(manifest)
    recorder.close_session(_end_positions(engine, "KALCB"))

    rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    live_actions = _jsonl(session / "strategy_actions.jsonl")
    offline_actions = _jsonl(session / "offline_replay" / "strategy_actions.jsonl")
    replay_manifest = json.loads((session / "offline_replay" / "replay_manifest.json").read_text(encoding="utf-8"))
    assert any(row["action_type"] == "FlattenPosition" for row in live_actions)
    assert any(row["action_type"] == "FlattenPosition" for row in offline_actions)
    assert replay_manifest["timer_replay_status"] == "replayed"
    assert report["paper_gate_passed"] is True


def test_prepare_runtime_session_captures_bars_for_closed_trade_replay(tmp_path):
    trade_date = date(2026, 2, 2)
    session = tmp_path / "session"
    artifact_root = tmp_path / "olr"
    store = OLRArtifactStore(artifact_root)
    olr_payload = {"olr.afternoon.top_n": 1}
    final_config_hash = final_candidate_config_fingerprint(OLRConfig.from_mapping(olr_payload))
    config_manifest = _write_runtime_config_manifest(tmp_path, kalcb_payload={}, olr_payload=olr_payload)
    store.save_snapshot(_olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE))
    store.save_snapshot(
        _olr_snapshot(
            trade_date,
            OLR_FINAL_ARTIFACT_STAGE,
            metadata_override={
                "candidate_config_hash": final_config_hash,
                "final_candidate_config_hash": final_config_hash,
            },
        )
    )
    recorder = PaperSessionRecorder(session, trade_date)

    plan = prepare_runtime_session(
        ("OLR",),
        trade_date=trade_date,
        mode="dry_run",
        artifact_roots={"OLR": artifact_root},
        health_checks={"artifact_only_gate_passed": True, "market_data_ok": True, "risk_limits_loaded": True},
        session_recorder=recorder,
        strategy_config_source=config_manifest,
        sector_map={"005930": "SEMIS"},
        initial_account_state={"equity": 1_000_000.0, "buyable_cash": 1_000_000.0},
        initial_positions={},
    )
    assert plan.ready_to_start is True
    assert not (session / "market_bars_5m.parquet").exists()

    driver = plan.drivers["OLR"]
    entry_bar = _bar(trade_date, time(14, 30), close=100.0)
    asyncio.run(driver.handle_bar(entry_bar))
    action_row = _jsonl(session / "strategy_actions.jsonl")[0]
    order_row = _jsonl(session / "order_events.jsonl")[0]
    asyncio.run(driver.handle_fill(_fill(order_row["order_id"], "BUY", action_row["action"]["qty"], 100.0, trade_date, time(15, 30))))

    exit_date = trade_date.replace(day=trade_date.day + 1)
    asyncio.run(driver.handle_bar(_bar(exit_date, time(14, 30), close=105.0)))
    exit_action = [row for row in _jsonl(session / "strategy_actions.jsonl") if row["action_type"] == "SubmitExit"][-1]
    exit_order = [row for row in _jsonl(session / "order_events.jsonl") if row.get("action_ref") == exit_action["action_ref"]][-1]
    asyncio.run(driver.handle_fill(_fill(exit_order["order_id"], "SELL", exit_action["action"]["qty"], 105.0, exit_date, time(15, 30))))

    recorder.close_session(_end_positions(driver.descriptor.engine))

    rebuild_offline_replay_from_session(session)
    report = replay_paper_session(session)

    assert (session / "market_bars_5m.parquet").is_file()
    assert len(_jsonl(session / "trade_outcomes.jsonl")) == 1
    assert report["paper_gate_passed"] is True


def _captured_olr_session(
    tmp_path: Path,
    *,
    decision_time: time,
    include_fill: bool,
    include_non_candidate_bar: bool = False,
    include_exit_fill: bool = False,
    startup_working_order: bool = False,
) -> Path:
    trade_date = date(2026, 2, 2)
    session = tmp_path / "session"
    recorder = PaperSessionRecorder(session, trade_date)
    stage1_snapshot = _olr_snapshot(trade_date, OLR_STAGE1_ARTIFACT_STAGE)
    snapshot = _olr_snapshot(trade_date)
    config = OLRConfig()
    account = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    initial_account_state = asdict(account)
    initial_positions = {}
    if startup_working_order:
        initial_positions = {
            "005930": PositionInfo(
                symbol="005930",
                real_qty=0,
                avg_price=100.0,
                allocations={},
                working_orders=[
                    WorkingOrderInfo(
                        order_id="ORD-WORKING",
                        symbol="005930",
                        side="BUY",
                        qty=10,
                        filled_qty=4,
                        remaining_qty=6,
                        price=100.0,
                        status="WORKING",
                        strategy_id="KALCB",
                        intent_id="intent-working",
                        idempotency_key="idem-working",
                    )
                ],
            )
        }
    policy_config = PortfolioPolicyConfig()
    policy = PortfolioArbitrationPolicy(policy_config)
    sector_map = {"005930": "SEMIS"}

    _stage_snapshot(session, recorder, stage1_snapshot, "olr_stage1_snapshots", f"olr_stage1_{trade_date.isoformat()}.json")
    _stage_snapshot(session, recorder, snapshot, "olr_final_snapshots", f"olr_final_{trade_date.isoformat()}.json")
    entry_bar = MarketBar(
        symbol="005930",
        timestamp=datetime.combine(trade_date, decision_time, tzinfo=KST),
        timeframe="5m",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
        is_completed=True,
    )
    pre_fill_bars = [entry_bar]
    if include_non_candidate_bar:
        pre_fill_bars.append(
            MarketBar(
                symbol="000660",
                timestamp=datetime.combine(trade_date, decision_time, tzinfo=KST),
                timeframe="5m",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000.0,
                is_completed=True,
            )
        )
    exit_bar = MarketBar(
        symbol="005930",
        timestamp=datetime.combine(trade_date.replace(day=trade_date.day + 1), time(14, 30), tzinfo=KST),
        timeframe="5m",
        open=104.0,
        high=106.0,
        low=103.0,
        close=105.0,
        volume=1000.0,
        is_completed=True,
    )
    oms = RecordingOMSClient(recorder, account_state=account, positions=dict(initial_positions))
    router = RuntimeActionRouter(
        recorder=recorder,
        oms_client=oms,
        portfolio_policy=policy,
        portfolio_enabled=True,
        dry_run=True,
    )
    context = PortfolioContextProvider(oms_client=oms, sector_map=sector_map)
    engine = OLREngine(config=config, candidate_snapshot=snapshot)
    descriptor = StrategyRuntimeDescriptor("OLR", OLR_FINAL_ARTIFACT_STAGE, snapshot.artifact_hash, engine, snapshot)
    driver = RuntimeSessionDriver(descriptor, router, recorder, context, mode="dry_run")

    router.record_state_snapshot(
        "OLR",
        engine.state,
        metadata={
            "record_reason": "runtime_session_pre_start",
            "mode": "dry_run",
            "trade_date": trade_date.isoformat(),
            "artifact_stage": descriptor.artifact_stage,
            "artifact_hash": descriptor.artifact_hash,
        },
    )
    for bar in sorted(pre_fill_bars, key=lambda item: (item.timestamp, item.symbol)):
        asyncio.run(driver.handle_bar(bar))

    if include_fill:
        action_row = _jsonl(session / "strategy_actions.jsonl")[0]
        order_row = _jsonl(session / "order_events.jsonl")[0]
        asyncio.run(
            driver.handle_fill(
                SimpleNamespace(
                    order_id=order_row["order_id"],
                    symbol="005930",
                    side="BUY",
                    qty=action_row["action"]["qty"],
                    price=100.0,
                    timestamp=datetime.combine(trade_date, time(15, 30), tzinfo=KST),
                    reason="unit_fill",
                    metadata={},
                )
            )
        )
    if include_exit_fill:
        asyncio.run(driver.handle_bar(exit_bar))
        action_rows = _jsonl(session / "strategy_actions.jsonl")
        exit_action = [row for row in action_rows if row["action_type"] == "SubmitExit"][-1]
        order_rows = _jsonl(session / "order_events.jsonl")
        exit_order = [row for row in order_rows if row.get("action_ref") == exit_action["action_ref"]][-1]
        asyncio.run(
            driver.handle_fill(
                SimpleNamespace(
                    order_id=exit_order["order_id"],
                    symbol="005930",
                    side="SELL",
                    qty=exit_action["action"]["qty"],
                    price=105.0,
                    timestamp=datetime.combine(trade_date.replace(day=trade_date.day + 1), time(15, 30), tzinfo=KST),
                    reason="unit_exit_fill",
                    metadata={},
                )
            )
        )

    manifest = _session_manifest(
        config=config,
        initial_account_state=initial_account_state,
        initial_positions={symbol: asdict(position) for symbol, position in initial_positions.items()},
        policy_config=policy_config,
        policy=policy,
        sector_map=sector_map,
    )
    recorder.write_manifest(manifest)
    recorder.close_session(_end_positions(engine))
    return session


def _session_manifest(
    *,
    strategy_id: str = "OLR",
    config: object,
    config_payload: dict | None = None,
    initial_account_state: dict,
    initial_positions: dict | None = None,
    policy_config: PortfolioPolicyConfig,
    policy: PortfolioArbitrationPolicy,
    sector_map: dict[str, str],
) -> dict:
    sid = strategy_id.upper()
    config_payload = dict(config_payload or asdict(config))
    return {
        "mode": "dry_run",
        "strategy_ids": [sid],
        "strategy_configs": {
            sid: {
                "uses_defaults": not bool(config_payload),
                "payload": config_payload,
                "payload_hash": canonical_json_hash(config_payload),
            }
        },
        "portfolio_enabled": True,
        "portfolio_policy_config": asdict(policy_config),
        "portfolio_policy_hash": policy.policy_hash,
        "sector_map": sector_map,
        "initial_account_state": initial_account_state,
        "initial_positions": dict(initial_positions or {}),
    }


def _write_runtime_config_manifest(tmp_path: Path, *, kalcb_payload: dict, olr_payload: dict) -> dict:
    config_root = tmp_path / "configs"
    kalcb_path = config_root / "kalcb_optimized_config.json"
    olr_path = config_root / "olr_optimized_config.json"
    _write_json(kalcb_path, {"mutations": kalcb_payload})
    _write_json(olr_path, {"mutations": olr_payload})
    return {
        "artifacts": [
            {"label": "kalcb optimized_config", "path": str(kalcb_path)},
            {"label": "olr optimized_config", "path": str(olr_path)},
        ]
    }


def _kalcb_position_state(trade_date: date, symbol: str) -> KALCBState:
    state = KALCBState(session_date=trade_date, snapshot_hash="unit-kalcb-snapshot", source_fingerprint="unit-source")
    symbol_state = state.symbol_state(symbol)
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol=symbol,
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime.combine(trade_date, time(9, 35), tzinfo=KST),
        initial_stop=95.0,
        current_stop=95.0,
        risk_per_share=5.0,
        entry_type="unit",
        momentum_score=3,
        sector="INTERNET",
    )
    return state


def _handle_bar_all(plan, bar: MarketBar) -> None:
    asyncio.run(plan.handle_bar(bar))


def _latest_action(session: Path, *, strategy_id: str, action_type: str, symbol: str, reason: str | None = None) -> dict:
    matches = [
        row
        for row in _jsonl(session / "strategy_actions.jsonl")
        if row["strategy_id"] == strategy_id
        and row["action_type"] == action_type
        and row["symbol"] == symbol
        and (reason is None or row["action"].get("reason") == reason)
    ]
    assert matches
    return matches[-1]


def _order_for_action(session: Path, action_row: dict) -> dict:
    matches = [row for row in _jsonl(session / "order_events.jsonl") if row.get("action_ref") == action_row["action_ref"]]
    assert matches
    return matches[-1]


def _olr_snapshot(
    trade_date: date,
    artifact_stage: str = OLR_FINAL_ARTIFACT_STAGE,
    *,
    symbols: tuple[str, ...] = ("005930",),
    metadata_override: dict | None = None,
) -> OLRDailySnapshot:
    is_final = artifact_stage == OLR_FINAL_ARTIFACT_STAGE
    candidates = tuple(
        OLRDailyCandidate(
            symbol=symbol,
            trade_date=trade_date,
            prior_day_high=102.0,
            prior_day_low=98.0,
            prior_day_close=100.0,
            daily_atr=2.0,
            expected_5m_volume=100.0,
            average_30m_volume=600.0,
            sector="SEMIS",
            rank=index,
            selection_score=1.0 / index,
            source_fingerprint="unit-source",
        )
        for index, symbol in enumerate(symbols, start=1)
    )
    metadata = {
        "artifact_stage": artifact_stage,
        "source": "olr_afternoon_selection" if is_final else "olr_research_selection",
        "selection_time_basis": "14:30_decision_from_completed_5m_bars" if is_final else "pre_session_from_prior_completed_daily_rows",
    }
    if is_final:
        final_config_hash = final_candidate_config_fingerprint(OLRConfig())
        metadata.update(
            {
                "candidate_config_hash": final_config_hash,
                "final_candidate_config_hash": final_config_hash,
                "final_candidate_config_hash_version": FINAL_CANDIDATE_CONFIG_HASH_VERSION,
            }
        )
    metadata.update(metadata_override or {})
    return OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit-source",
        generated_at=datetime.combine(trade_date, time(14, 31) if is_final else time(0, 0), tzinfo=KST),
        candidates=candidates,
        metadata=metadata,
    )


def _kalcb_snapshot(
    trade_date: date,
    *,
    symbol: str = "005930",
    sector: str = "SEMIS",
    sector_map: dict[str, str] | None = None,
) -> KALCBDailySnapshot:
    config_sector_map = dict(sector_map or {str(symbol).zfill(6): sector})
    return KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit-source",
        generated_at=datetime.combine(trade_date, time(8, 40), tzinfo=KST),
        candidates=(
            KALCBDailyCandidate(
                symbol=symbol,
                trade_date=trade_date,
                prior_day_high=102.0,
                prior_day_low=98.0,
                prior_day_close=100.0,
                daily_atr=2.0,
                expected_5m_volume=100.0,
                average_30m_volume=600.0,
                sector=sector,
                source_fingerprint="unit-source",
            ),
        ),
        metadata={
            "artifact_stage": KALCB_FINAL_ARTIFACT_STAGE,
            "source": "real_kis_krx_parquet",
            "candidate_config_hash": kalcb_candidate_config_fingerprint(KALCBConfig(), {}, config_sector_map),
            "sector_map_hash": canonical_json_hash(config_sector_map),
            "sector_map_size": len(config_sector_map),
            "active_symbols": [str(symbol).zfill(6)],
            "active_symbol_count": 1,
            "active_budget_source": "ws_budget",
            "frontier_symbols": [str(symbol).zfill(6)],
            "frontier_symbol_count": 1,
            "overflow_symbols": [],
            "overflow_symbol_count": 0,
            "frontier_rest_budget_symbols_per_5m": 300,
        },
    )


def _bar(trade_date: date, at: time, *, close: float) -> MarketBar:
    return MarketBar(
        symbol="005930",
        timestamp=datetime.combine(trade_date, at, tzinfo=KST),
        timeframe="5m",
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1000.0,
        is_completed=True,
    )


def _fill(order_id: str, side: str, qty: int, price: float, trade_date: date, at: time) -> SimpleNamespace:
    return SimpleNamespace(
        order_id=order_id,
        symbol="005930",
        side=side,
        qty=qty,
        price=price,
        timestamp=datetime.combine(trade_date, at, tzinfo=KST),
        reason="unit_fill",
        metadata={},
    )


def _end_positions(engine: OLREngine | KALCBEngine | dict[str, OLREngine | KALCBEngine], strategy_id: str = "OLR") -> dict:
    positions = []
    engines = engine.items() if isinstance(engine, dict) else ((strategy_id, engine),)
    for sid, item in engines:
        for symbol, state in item.state.symbols.items():
            position = state.position
            if position is not None and int(position.qty_open) > 0:
                positions.append({"strategy_id": str(sid).upper(), "symbol": symbol, **asdict(position)})
    return {"positions": positions}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _runtime_event_ref(strategy_id: str, event_type: str, payload: dict) -> str:
    return canonical_json_hash(
        {
            "strategy_id": strategy_id.upper(),
            "event_type": event_type,
            "payload": payload,
        }
    )[:24]


def _stage_snapshot(session: Path, recorder: PaperSessionRecorder, snapshot, bucket: str, filename: str) -> Path:
    path = session / bucket / filename
    _write_json(path, snapshot.to_json_dict())
    recorder.append_jsonl(
        "artifact_generation.jsonl",
        {
            "record_type": "artifact_generation",
            "strategy_id": snapshot.strategy_id,
            "trade_date": snapshot.trade_date.isoformat(),
            "stage": str(snapshot.metadata.get("artifact_stage") or ""),
            "artifact_hash": snapshot.artifact_hash,
            "source_fingerprint": snapshot.source_fingerprint,
            "candidate_count": len(snapshot.candidates),
            "bucket": bucket,
            "source_path": str(path),
            "session_path": str(path),
        },
    )
    return path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
