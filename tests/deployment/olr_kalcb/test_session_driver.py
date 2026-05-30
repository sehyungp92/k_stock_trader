from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from oms.intent import IntentResult, IntentStatus
from oms_client.client import AccountState

from deployment.olr_kalcb.action_router import RuntimeActionRouter
from deployment.olr_kalcb.coordinator import StrategyRuntimeDescriptor
from deployment.olr_kalcb.dry_run_oms import RecordingOMSClient
from deployment.olr_kalcb.portfolio import PortfolioArbitrationPolicy, PortfolioPolicyConfig
from deployment.olr_kalcb.portfolio_context import PortfolioContextProvider
from deployment.olr_kalcb.runtime import RuntimePreflightCheck, RuntimePreflightResult, RuntimeSessionPlan
from deployment.olr_kalcb.session_capture import PaperSessionRecorder
from deployment.olr_kalcb.session_driver import RuntimeSessionDriver
from strategy_common.actions import SubmitEntry, SubmitExit, SubmitPartialExit, SubmitProtectiveStop
from strategy_common.clock import KST
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar
from strategy_kalcb.core.state import KALCBPositionState, SymbolStage
from strategy_kalcb.engine import KALCBEngine
from strategy_olr.engine import OLREngine


def test_driver_records_decision_action_portfolio_intent_and_state_for_entry(tmp_path):
    driver = _driver(tmp_path, _EntryEngine())

    result = asyncio.run(driver.handle_bar(_bar("005930")))

    assert result.decision_count == 1
    assert result.action_count == 1
    assert result.accepted_intent_count == 1
    assert result.blocked_action_count == 0
    decision_rows = _jsonl(tmp_path / "session" / "decision_stream.jsonl")
    action_rows = _jsonl(tmp_path / "session" / "strategy_actions.jsonl")
    portfolio_rows = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")
    intent_rows = _jsonl(tmp_path / "session" / "oms_intents.jsonl")
    order_rows = _jsonl(tmp_path / "session" / "order_events.jsonl")
    state_rows = _jsonl(tmp_path / "session" / "state_snapshots.jsonl")

    assert [row["record_type"] for row in decision_rows] == ["runtime_event_input", "decision_event"]
    assert action_rows[0]["event_ref"] == result.event_ref
    assert action_rows[0]["provisional_order_ref"].startswith("KALCB:")
    assert portfolio_rows[0]["decision"] == "accepted"
    assert intent_rows[0]["metadata"]["provisional_order_ref"] == action_rows[0]["provisional_order_ref"]
    assert order_rows[0]["order_id"].startswith("dry-run:KALCB:005930:")
    assert state_rows[-1]["metadata"]["event_ref"] == result.event_ref
    assert result.state_hash == state_rows[-1]["state_hash"]


def test_driver_records_no_action_row_when_engine_returns_no_decisions(tmp_path):
    driver = _driver(tmp_path, _NoActionEngine())

    result = asyncio.run(driver.handle_bar(_bar("005930")))

    assert result.decision_count == 0
    assert result.action_count == 0
    rows = _jsonl(tmp_path / "session" / "decision_stream.jsonl")
    assert rows[-1]["record_type"] == "runtime_no_action"
    assert rows[-1]["reason_code"] == "no_signal"
    assert rows[-1]["event_ref"] == result.event_ref


def test_driver_records_post_fill_state_transition(tmp_path):
    engine = _EntryEngine()
    driver = _driver(tmp_path, engine)
    before = driver.action_router.record_state_snapshot("KALCB", engine.state, metadata={"record_reason": "before_fill"})

    result = asyncio.run(
        driver.handle_fill(
            SimpleNamespace(
                order_id="order-a",
                symbol="005930",
                side="BUY",
                qty=1,
                price=100.0,
                timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
                reason="unit_fill",
                metadata={},
            )
        )
    )

    assert result.state_hash != before
    assert _jsonl(tmp_path / "session" / "fill_events.jsonl")[0]["record_type"] == "runtime_fill_event"
    assert _jsonl(tmp_path / "session" / "decision_stream.jsonl")[-1]["record_type"] == "runtime_no_action"


def test_router_records_compact_strategy_state_snapshot(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder, account_state=AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0))
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    engine = KALCBEngine()
    symbol_state = engine.state.symbol_state("005930")
    symbol_state.candidate = SimpleNamespace(
        symbol="005930",
        trade_date=date(2026, 2, 2),
        source_fingerprint="unit-source",
        tradable=True,
        metadata={"oversized": "x" * 100_000},
    )
    symbol_state.add_bar(_bar("005930"))

    state_hash = router.record_state_snapshot("KALCB", engine.state, metadata={"record_reason": "unit"})

    row = _jsonl(tmp_path / "session" / "state_snapshots.jsonl")[0]
    assert row["state_encoding"] == "kalcb-state-compact-v1"
    assert row["state_hash"] == state_hash
    assert row["state"]["symbol_count"] == 1
    assert row["state"]["symbols_hash"]
    assert row["state"]["stage_counts"] == {"WATCHING": 1}
    assert row["state"]["position_symbols"] == []
    assert row["state"]["pending_symbols"] == []
    assert "symbols" not in row["state"]
    assert "oversized" not in json.dumps(row)


def test_driver_applies_paper_fill_to_seeded_portfolio_context_when_refresh_empty(tmp_path):
    driver = _driver(tmp_path, _NoActionEngine(), mode="paper", oms=_NoRefreshOMS())
    driver.portfolio_context.account_state = AccountState(equity=1_000.0, buyable_cash=1_000.0)

    asyncio.run(
        driver.handle_fill(
            SimpleNamespace(
                order_id="broker-fill",
                symbol="005930",
                side="BUY",
                qty=2,
                price=100.0,
                timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
                reason="paper_fill",
                metadata={},
            )
        )
    )

    assert driver.portfolio_context.strategy_exposure("KALCB", "005930").qty == 2
    assert driver.portfolio_context.cash_equity().cash == 800.0


def test_driver_records_timer_no_action(tmp_path):
    driver = _driver(tmp_path, _NoActionEngine())

    result = asyncio.run(driver.handle_timer(datetime(2026, 2, 2, 10, 0, tzinfo=KST)))

    rows = _jsonl(tmp_path / "session" / "decision_stream.jsonl")
    assert result.event_type == "timer"
    assert rows[-1]["record_type"] == "runtime_no_action"
    assert rows[-1]["reason_code"] == "timer_no_action"


def test_driver_routes_expired_olr_update_to_expiry_handler(tmp_path):
    engine = _ExpiryRoutingEngine()
    driver = _driver(tmp_path, engine, strategy_id="OLR")

    asyncio.run(
        driver.handle_order_event(
            SimpleNamespace(
                order_id="broker-exit",
                symbol="005930",
                status="EXPIRED",
                side="SELL",
                order_type="CLOSE_AUCTION",
                qty=1,
                timestamp=datetime(2026, 2, 2, 15, 31, tzinfo=KST),
                reason="auction_expired",
                metadata={},
            )
        )
    )

    assert engine.expired_calls == 1
    assert engine.update_calls == 0
    assert any(row.get("decision_code") == "expired_path" for row in _jsonl(tmp_path / "session" / "decision_stream.jsonl"))


def test_driver_links_actions_to_unique_decision_in_multi_decision_event(tmp_path):
    driver = _driver(tmp_path, _MultiDecisionEngine())

    asyncio.run(driver.handle_bar(_bar("005930")))

    decisions = _jsonl(tmp_path / "session" / "decision_stream.jsonl")
    action = _jsonl(tmp_path / "session" / "strategy_actions.jsonl")[0]
    entry_ref = [row["decision_ref"] for row in decisions if row.get("decision_code") == "entry"][-1]

    assert action["decision_ref"] == entry_ref
    assert len(action["event_decision_refs"]) == 2


def test_recorder_buffers_market_bars_until_flush(tmp_path, monkeypatch):
    from deployment.olr_kalcb import session_capture

    writes = []

    def fake_write(path, rows):
        writes.append((path, list(rows)))
        path.write_bytes(b"unit-parquet-placeholder")

    monkeypatch.setattr(session_capture, "_write_market_bar_rows", fake_write)
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))

    recorder.record_market_bar(_bar("005930"))
    recorder.record_market_bar(_bar("005930"))
    assert writes == []

    recorder.flush_market_bars()
    recorder.flush_market_bars()
    assert len(writes) == 1


def test_driver_rejects_incomplete_bar_for_replay_mode(tmp_path):
    driver = _driver(tmp_path, _NoActionEngine(), mode="offline_replay")
    bar = _bar("005930", is_completed=False)

    try:
        asyncio.run(driver.handle_bar(bar))
    except ValueError as exc:
        assert "incomplete bar" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("incomplete replay bar was accepted")


def test_driver_rejects_incomplete_bar_for_paper_mode_before_engine(tmp_path):
    engine = _NoActionEngine()
    driver = _driver(tmp_path, engine, mode="paper")
    bar = _bar("005930", is_completed=False)

    with pytest.raises(ValueError, match="incomplete bar"):
        asyncio.run(driver.handle_bar(bar))

    assert engine.state["seen"] == 0
    assert _jsonl(tmp_path / "session" / "decision_stream.jsonl") == []


def test_driver_clears_kalcb_pending_state_when_portfolio_blocks_entry(tmp_path):
    engine = _SubmittingKALCBEngine(qty=1, price=100.0)
    driver = _driver(
        tmp_path,
        engine,
        policy=PortfolioArbitrationPolicy(PortfolioPolicyConfig(max_gross_notional=50.0, max_symbol_notional=50.0)),
    )

    result = asyncio.run(driver.handle_bar(_bar("005930")))

    symbol_state = engine.state.symbol_state("005930")
    assert result.blocked_action_count == 1
    assert symbol_state.pending_entry_order_id == ""
    assert "KALCB:" not in "".join(engine.state.order_roles)


def test_driver_clears_olr_pending_state_when_oms_rejects_entry(tmp_path):
    engine = _SubmittingOLREngine(qty=1, price=100.0)
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = _RejectingOMSClient(recorder, account_state=AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0))
    driver = _driver(tmp_path, engine, strategy_id="OLR", oms=oms, recorder=recorder)

    result = asyncio.run(driver.handle_bar(_bar("005930")))

    symbol_state = engine.state.symbol_state("005930")
    assert result.accepted_intent_count == 0
    assert symbol_state.pending_entry_order_id == ""
    assert engine.state.order_roles == {}
    assert engine.decisions[-1].reason == "rejected"


def test_driver_reconciles_resized_olr_submission_metadata(tmp_path):
    engine = _SubmittingOLREngine(qty=10, price=100.0)
    driver = _driver(
        tmp_path,
        engine,
        strategy_id="OLR",
        policy=PortfolioArbitrationPolicy(PortfolioPolicyConfig(max_gross_notional=150.0, max_symbol_notional=150.0)),
    )

    result = asyncio.run(driver.handle_bar(_bar("005930")))

    symbol_state = engine.state.symbol_state("005930")
    assert result.accepted_intent_count == 1
    assert symbol_state.pending_entry_order_id.startswith("OLR:")
    assert symbol_state.pending_entry_metadata["submitted_qty"] == 1


def test_driver_clears_kalcb_exit_state_when_route_rejects_position_actions(tmp_path):
    exit_engine = _SubmittingKALCBPositionActionEngine("exit")
    partial_engine = _SubmittingKALCBPositionActionEngine("partial")
    stop_engine = _SubmittingKALCBPositionActionEngine("stop")

    asyncio.run(_driver(tmp_path / "exit", exit_engine).handle_bar(_bar("005930")))
    asyncio.run(_driver(tmp_path / "partial", partial_engine).handle_bar(_bar("005930")))
    asyncio.run(
        _driver(
            tmp_path / "stop",
            stop_engine,
            oms=_RejectingOMSClient(PaperSessionRecorder(tmp_path / "stop" / "session", date(2026, 2, 2))),
        ).handle_bar(_bar("005930"))
    )

    assert exit_engine.state.symbol_state("005930").position.exit_in_flight is False
    assert partial_engine.state.symbol_state("005930").position.partial_order_id == ""
    assert stop_engine.state.symbol_state("005930").position.stop_order_id == ""


def test_driver_resolves_broker_fill_to_provisional_order_ref(tmp_path):
    engine = _SubmittingKALCBEngine(qty=1, price=100.0)
    driver = _driver(tmp_path, engine)
    asyncio.run(driver.handle_bar(_bar("005930")))
    order_row = _jsonl(tmp_path / "session" / "order_events.jsonl")[0]

    asyncio.run(
        driver.handle_fill(
            SimpleNamespace(
                order_id=order_row["order_id"],
                symbol="005930",
                side="BUY",
                qty=1,
                price=100.0,
                timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
                reason="unit_fill",
                metadata={},
            )
        )
    )

    fill_row = _jsonl(tmp_path / "session" / "fill_events.jsonl")[0]
    position = engine.state.symbol_state("005930").position
    assert fill_row["event"]["order_id"].startswith("KALCB:")
    assert fill_row["event"]["metadata"]["broker_order_id"] == order_row["order_id"]
    assert position is not None
    assert position.entry_order_id.startswith("KALCB:")
    assert position.entry_order_id not in engine.state.order_roles


def test_driver_partial_fill_releases_only_filled_router_reservation(tmp_path):
    engine = _SubmittingKALCBEngine(qty=10, price=100.0)
    driver = _driver(tmp_path, engine)
    asyncio.run(driver.handle_bar(_bar("005930")))
    order_row = _jsonl(tmp_path / "session" / "order_events.jsonl")[0]

    asyncio.run(
        driver.handle_fill(
            SimpleNamespace(
                order_id=order_row["order_id"],
                symbol="005930",
                side="BUY",
                qty=4,
                price=100.0,
                timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
                reason="partial_fill",
                metadata={},
            )
        )
    )

    reservation = next(iter(driver.action_router.pending_reservations.values()))
    assert reservation.qty == 6
    assert reservation.notional == 600.0


def test_driver_blocks_unmapped_replay_fill(tmp_path):
    driver = _driver(tmp_path, _NoActionEngine(), mode="offline_replay")

    try:
        asyncio.run(
            driver.handle_fill(
                SimpleNamespace(
                    order_id="broker-only",
                    symbol="005930",
                    side="BUY",
                    qty=1,
                    price=100.0,
                    timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
                    reason="unit_fill",
                    metadata={},
                )
            )
        )
    except ValueError as exc:
        assert "unmapped fill order identity" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("unmapped replay fill was accepted")


def test_driver_blocks_unknown_provisional_replay_fill(tmp_path):
    driver = _driver(tmp_path, _NoActionEngine(), mode="offline_replay")

    try:
        asyncio.run(
            driver.handle_fill(
                SimpleNamespace(
                    order_id="KALCB:unknown-event:action:0",
                    symbol="005930",
                    side="BUY",
                    qty=1,
                    price=100.0,
                    timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
                    reason="unit_fill",
                    metadata={"provisional_order_ref": "KALCB:unknown-event:action:0"},
                )
            )
        )
    except ValueError as exc:
        assert "unmapped fill order identity" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("unknown provisional replay fill was accepted")


def test_runtime_plan_routes_market_bar_as_one_combined_priority_batch(tmp_path):
    trade_date = date(2026, 2, 2)
    recorder = PaperSessionRecorder(tmp_path / "session", trade_date)
    oms = RecordingOMSClient(recorder, account_state=AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0))
    router = RuntimeActionRouter(
        recorder=recorder,
        oms_client=oms,
        portfolio_policy=PortfolioArbitrationPolicy(PortfolioPolicyConfig(strategy_priority=("OLR", "KALCB"))),
        portfolio_enabled=True,
        dry_run=True,
    )
    context = PortfolioContextProvider(oms)
    snapshot = SimpleNamespace(source_fingerprint="unit-source", candidates=(SimpleNamespace(symbol="005930"),))
    descriptors = {
        "KALCB": StrategyRuntimeDescriptor("KALCB", "unit_stage", "kalcb-hash", _StrategyEntryEngine("KALCB"), snapshot),
        "OLR": StrategyRuntimeDescriptor("OLR", "unit_stage", "olr-hash", _StrategyEntryEngine("OLR"), snapshot),
    }
    drivers = {
        sid: RuntimeSessionDriver(descriptor, router, recorder, context, "dry_run")
        for sid, descriptor in descriptors.items()
    }
    plan = RuntimeSessionPlan(
        mode="dry_run",
        trade_date=trade_date,
        artifacts={},
        artifact_failures=(),
        preflight=RuntimePreflightResult("dry_run", trade_date, (RuntimePreflightCheck("unit", True),)),
        descriptors=descriptors,
        drivers=drivers,
        schedule=(),
        action_router=router,
        session_recorder=recorder,
    )

    results = asyncio.run(plan.handle_bar(_bar("005930")))

    portfolio_rows = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")
    assert [row["strategy_id"] for row in portfolio_rows] == ["OLR", "KALCB"]
    assert [result.strategy_id for result in results] == ["KALCB", "OLR"]
    assert results[0].blocked_action_count == 1
    assert results[1].accepted_intent_count == 1


class _EntryEngine:
    def __init__(self):
        self.state = {"submitted": []}

    def on_bar(self, bar, portfolio, submit):
        action = SubmitEntry(
            strategy_id="KALCB",
            symbol=bar.symbol,
            qty=1,
            order_type="LIMIT",
            limit_price=100.0,
            stop_price=None,
            reason="unit_entry",
        )
        order_ref = submit(action)
        self.state["submitted"].append(order_ref)
        return [
            DecisionEvent(
                timestamp=bar.timestamp,
                strategy_id="KALCB",
                symbol=bar.symbol,
                decision_code="entry",
                reason="unit",
                actions=(action,),
            )
        ]

    def on_timer(self, timestamp, submit):
        return []

    def on_fill(self, fill, submit):
        self.state["fill"] = fill.order_id
        return []


class _StrategyEntryEngine:
    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        self.state = {"submitted": []}

    def on_bar(self, bar, portfolio, submit):
        action = SubmitEntry(self.strategy_id, bar.symbol, 1, "LIMIT", 100.0, None, "unit_entry")
        self.state["submitted"].append(submit(action))
        return [
            DecisionEvent(
                timestamp=bar.timestamp,
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                decision_code="entry",
                reason="unit",
                actions=(action,),
            )
        ]


class _NoActionEngine:
    def __init__(self):
        self.state = {"seen": 0}

    def on_bar(self, bar, portfolio, submit):
        self.state["seen"] += 1
        return []

    def on_timer(self, timestamp, submit):
        return []

    def on_fill(self, fill, submit):
        return []


class _ExpiryRoutingEngine:
    def __init__(self):
        self.state = {"expired": 0, "updated": 0}
        self.expired_calls = 0
        self.update_calls = 0

    def on_bar(self, bar, portfolio, submit):
        return []

    def on_timer(self, timestamp, submit):
        return []

    def on_fill(self, fill, submit):
        return []

    def on_order_expired(self, expired, submit):
        self.expired_calls += 1
        self.state["expired"] += 1
        return [
            DecisionEvent(
                timestamp=expired.timestamp,
                strategy_id="OLR",
                symbol=expired.symbol,
                decision_code="expired_path",
                reason=expired.reason,
            )
        ]

    def on_order_update(self, update, submit):
        self.update_calls += 1
        self.state["updated"] += 1
        return [
            DecisionEvent(
                timestamp=update.timestamp,
                strategy_id="OLR",
                symbol=update.symbol,
                decision_code="update_path",
                reason=update.reason,
            )
        ]


class _MultiDecisionEngine:
    def __init__(self):
        self.state = {"seen": 0}

    def on_bar(self, bar, portfolio, submit):
        self.state["seen"] += 1
        action = SubmitEntry("KALCB", bar.symbol, 1, "LIMIT", 100.0, None, "unit_entry")
        submit(action)
        return [
            DecisionEvent(
                timestamp=bar.timestamp,
                strategy_id="KALCB",
                symbol=bar.symbol,
                decision_code="context",
                reason="context_only",
            ),
            DecisionEvent(
                timestamp=bar.timestamp,
                strategy_id="KALCB",
                symbol=bar.symbol,
                decision_code="entry",
                reason="unit",
                actions=(action,),
            ),
        ]

    def on_timer(self, timestamp, submit):
        return []

    def on_fill(self, fill, submit):
        return []


class _SubmittingKALCBEngine(KALCBEngine):
    def __init__(self, *, qty: int, price: float):
        super().__init__()
        self.qty = qty
        self.price = price

    def on_bar(self, bar, portfolio, submit):
        action = SubmitEntry("KALCB", bar.symbol, self.qty, "LIMIT", self.price, None, "unit_entry")
        order_ref = submit(action)
        self.reconcile_submitted_order(order_ref, action)
        return [
            DecisionEvent(
                timestamp=bar.timestamp,
                strategy_id="KALCB",
                symbol=bar.symbol,
                decision_code="entry",
                reason="unit",
                actions=(action,),
            )
        ]


class _SubmittingOLREngine(OLREngine):
    def __init__(self, *, qty: int, price: float):
        super().__init__()
        self.qty = qty
        self.price = price

    def on_bar(self, bar, portfolio, submit):
        action = SubmitEntry("OLR", bar.symbol, self.qty, "LIMIT", self.price, None, "unit_entry")
        order_ref = submit(action)
        self.reconcile_submitted_order(order_ref, action)
        return [
            DecisionEvent(
                timestamp=bar.timestamp,
                strategy_id="OLR",
                symbol=bar.symbol,
                decision_code="entry",
                reason="unit",
                actions=(action,),
            )
        ]


class _SubmittingKALCBPositionActionEngine(KALCBEngine):
    def __init__(self, action_kind: str):
        super().__init__()
        self.action_kind = action_kind
        symbol_state = self.state.symbol_state("005930")
        symbol_state.stage = SymbolStage.IN_POSITION
        symbol_state.position = KALCBPositionState(
            symbol="005930",
            qty_entry=10,
            qty_open=10,
            entry_price=100.0,
            entry_time=datetime(2026, 2, 2, 9, 30, tzinfo=KST),
            initial_stop=95.0,
            current_stop=95.0,
            risk_per_share=5.0,
            entry_type="unit",
            momentum_score=1,
        )

    def on_bar(self, bar, portfolio, submit):
        position = self.state.symbol_state("005930").position
        if self.action_kind == "exit":
            position.exit_in_flight = True
            action = SubmitExit("KALCB", bar.symbol, 10, "LIMIT", 100.0, "unit_exit", metadata={"order_role": "EXIT"})
        elif self.action_kind == "partial":
            position.partial_order_id = "__pending__"
            action = SubmitPartialExit("KALCB", bar.symbol, 5, "LIMIT", 100.0, "unit_partial", metadata={"order_role": "TP"})
        else:
            action = SubmitProtectiveStop("KALCB", bar.symbol, 10, 95.0, "unit_stop", metadata={"order_role": "STOP"})
        order_ref = submit(action)
        self.reconcile_submitted_order(order_ref, action)
        return [
            DecisionEvent(
                timestamp=bar.timestamp,
                strategy_id="KALCB",
                symbol=bar.symbol,
                decision_code=self.action_kind,
                reason="unit",
                actions=(action,),
            )
        ]


class _RejectingOMSClient(RecordingOMSClient):
    async def submit_intent(self, intent):
        return IntentResult(intent_id=intent.intent_id, status=IntentStatus.REJECTED, message="unit_rejected")


class _NoRefreshOMS:
    async def submit_intent(self, intent):
        return IntentResult(intent_id=intent.intent_id, status=IntentStatus.ACCEPTED, order_id="paper-order")

    async def get_account_state(self):
        return None

    async def get_all_positions(self):
        return None


def _driver(
    tmp_path,
    engine,
    *,
    mode="dry_run",
    strategy_id="KALCB",
    policy=None,
    oms=None,
    recorder=None,
) -> RuntimeSessionDriver:
    recorder = recorder or PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = oms or RecordingOMSClient(recorder, account_state=AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0))
    router = RuntimeActionRouter(
        recorder=recorder,
        oms_client=oms,
        portfolio_policy=policy or PortfolioArbitrationPolicy(),
        portfolio_enabled=True,
        dry_run=True,
    )
    context = PortfolioContextProvider(oms)
    snapshot = SimpleNamespace(source_fingerprint="unit-source", candidates=(SimpleNamespace(symbol="005930"),))
    descriptor = StrategyRuntimeDescriptor(strategy_id, "unit_stage", "artifact-hash", engine, snapshot)
    return RuntimeSessionDriver(descriptor, router, recorder, context, mode)


def _bar(symbol: str, *, is_completed: bool = True) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime(2026, 2, 2, 9, 30, tzinfo=KST),
        timeframe="5m",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
        is_completed=is_completed,
    )


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
