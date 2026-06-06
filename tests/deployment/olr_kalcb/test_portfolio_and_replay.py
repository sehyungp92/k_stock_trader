from __future__ import annotations

import json
import shutil
import asyncio
import importlib.util
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pytest

from oms.intent import IntentResult, IntentStatus
from oms_client.client import AccountState, AllocationInfo, PositionInfo, WorkingOrderInfo
from deployment.olr_kalcb.action_router import RuntimeActionRouter
from deployment.olr_kalcb.dry_run_oms import RecordingOMSClient
from deployment.olr_kalcb.hashing import canonical_json_hash
from deployment.olr_kalcb.offline_replay import ReplayInputLoader, write_offline_replay_manifest
from deployment.olr_kalcb.portfolio_context import PortfolioContextProvider
from deployment.olr_kalcb.portfolio import PortfolioArbitrationInput, PortfolioArbitrationPolicy, PortfolioPolicyConfig
from deployment.olr_kalcb.replay import replay_paper_session, summarize_paper_parity
from deployment.olr_kalcb.session_capture import PaperSessionRecorder, session_hashes
from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_common.clock import KST


def test_portfolio_policy_accepts_blocks_and_resizes_deterministically():
    ts = datetime(2026, 2, 2, 9, 30, tzinfo=KST)
    policy = PortfolioArbitrationPolicy(
        PortfolioPolicyConfig(max_gross_notional=1_500_000, max_symbol_notional=1_000_000, max_sector_notional=1_200_000)
    )
    rows = [
        PortfolioArbitrationInput("a2", "OLR", "005930", "BUY", 10, 1_000_000, ts, sector="SEMIS", cash=2_000_000, equity=2_000_000),
        PortfolioArbitrationInput("a1", "KALCB", "000660", "BUY", 10, 1_000_000, ts, sector="SEMIS", cash=2_000_000, equity=2_000_000),
        PortfolioArbitrationInput("a3", "KALCB", "035420", "BUY", 10, 1_000_000, ts, sector="INTERNET", cash=2_000_000, equity=2_000_000),
    ]

    decisions = policy.decide_many(rows)
    again = policy.decide_many(rows)

    assert [item.to_json_dict() for item in decisions] == [item.to_json_dict() for item in again]
    assert decisions[0].action_ref == "a1"
    assert decisions[0].decision == "accepted"
    assert decisions[1].decision in {"blocked", "resized"}
    assert policy.metrics(decisions)["accepted_count"] >= 1


def test_portfolio_policy_validates_side_and_exit_exposure():
    ts = datetime(2026, 2, 2, 9, 30, tzinfo=KST)
    policy = PortfolioArbitrationPolicy(PortfolioPolicyConfig(max_gross_notional=1_500_000))

    with pytest.raises(ValueError, match="unsupported side"):
        PortfolioArbitrationInput("bad", "KALCB", "005930", "SHORT", 1, 100_000, ts)

    duplicate = policy.decide_many(
        [
            PortfolioArbitrationInput("buy1", "KALCB", "005930", "BUY", 1, 100_000, ts, cash=1_000_000, equity=1_000_000),
            PortfolioArbitrationInput("buy2", "OLR", "005930", "BUY", 1, 100_000, ts, cash=1_000_000, equity=1_000_000),
        ]
    )
    assert duplicate[0].decision == "accepted"
    assert duplicate[1].reason_code == "duplicate_symbol_conflict"

    reducing = policy.decide_one(
        PortfolioArbitrationInput(
            "sell1",
            "KALCB",
            "005930",
            "SELL",
            1,
            100_000,
            ts,
            current_strategy_exposure=100_000,
            current_symbol_exposure=100_000,
            current_strategy_symbol_qty=1,
            current_symbol_qty=1,
        )
    )
    full_exit_above_cost = policy.decide_one(
        PortfolioArbitrationInput(
            "sell_gain",
            "KALCB",
            "005930",
            "SELL",
            10,
            1_500,
            ts,
            current_strategy_exposure=1_000,
            current_symbol_exposure=1_000,
            current_strategy_symbol_qty=10,
            current_symbol_qty=10,
        )
    )
    unmatched = policy.decide_one(PortfolioArbitrationInput("sell2", "KALCB", "005930", "SELL", 1, 100_000, ts))
    wrong_symbol = policy.decide_one(
        PortfolioArbitrationInput(
            "sell3",
            "KALCB",
            "005930",
            "SELL",
            1,
            100_000,
            ts,
            current_strategy_exposure=500_000,
        )
    )

    assert reducing.decision == "accepted"
    assert reducing.reason_code == "accepted_exit_reduces_exposure"
    assert full_exit_above_cost.decision == "accepted"
    assert full_exit_above_cost.final_qty == 10
    assert full_exit_above_cost.final_notional == 1_500
    assert unmatched.decision == "blocked"
    assert unmatched.reason_code == "unsupported_short_or_unmatched_exit"
    assert wrong_symbol.decision == "blocked"
    assert wrong_symbol.reason_code == "unsupported_short_or_unmatched_exit"
    symbol_owned_by_other_strategy = policy.decide_one(
        PortfolioArbitrationInput(
            "sell_other",
            "KALCB",
            "005930",
            "SELL",
            1,
            100_000,
            ts,
            current_symbol_exposure=100_000,
            current_strategy_exposure=0,
        )
    )
    assert symbol_owned_by_other_strategy.decision == "blocked"
    assert symbol_owned_by_other_strategy.reason_code == "unsupported_short_or_unmatched_exit"

    duplicate_exit = policy.decide_many(
        [
            PortfolioArbitrationInput(
                "sell4",
                "KALCB",
                "005930",
                "SELL",
                6,
                60_000,
                ts,
                current_symbol_exposure=100_000,
                current_strategy_exposure=100_000,
                current_strategy_symbol_qty=10,
                current_symbol_qty=10,
            ),
            PortfolioArbitrationInput(
                "sell5",
                "KALCB",
                "005930",
                "SELL",
                6,
                60_000,
                ts,
                current_symbol_exposure=100_000,
                current_strategy_exposure=100_000,
                current_strategy_symbol_qty=10,
                current_symbol_qty=10,
            ),
        ]
    )
    assert duplicate_exit[0].decision == "accepted"
    assert duplicate_exit[1].decision == "resized"
    assert duplicate_exit[1].final_qty == 4
    assert duplicate_exit[1].final_notional == 40_000


def test_router_sell_uses_strategy_owned_context_exposure(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    context.positions = {
        "005930": PositionInfo(
            symbol="005930",
            real_qty=10,
            avg_price=100.0,
            allocations={"KALCB": AllocationInfo("KALCB", qty=10, cost_basis=100.0)},
        )
    }

    results = asyncio.run(
        router.route_actions(
            (
                SubmitExit(
                    strategy_id="KALCB",
                    symbol="005930",
                    qty=5,
                    order_type="LIMIT",
                    limit_price=100.0,
                    reason="unit_exit",
                ),
            ),
            portfolio_context=context,
            event_ref="event-a",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    rows = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")
    assert results[0].accepted is True
    assert rows[0]["decision"] == "accepted"
    assert rows[0]["reason_code"] == "accepted_exit_reduces_exposure"


def test_router_blocks_sell_owned_by_other_strategy_and_resizes_oversized_exit(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    context.positions = {
        "005930": PositionInfo(
            symbol="005930",
            real_qty=20,
            avg_price=100.0,
            allocations={
                "OLR": AllocationInfo("OLR", qty=10, cost_basis=100.0),
                "KALCB": AllocationInfo("KALCB", qty=10, cost_basis=100.0),
            },
        ),
        "000660": PositionInfo(
            symbol="000660",
            real_qty=10,
            avg_price=100.0,
            allocations={"OLR": AllocationInfo("OLR", qty=10, cost_basis=100.0)},
        ),
    }

    wrong_owner = asyncio.run(
        router.route_actions(
            (
                SubmitExit("KALCB", "000660", 1, "LIMIT", 100.0, "other_owned"),
                SubmitExit("KALCB", "005930", 20, "LIMIT", 100.0, "oversized"),
            ),
            portfolio_context=context,
            event_ref="event-b",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    rows = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")
    assert wrong_owner[0].blocked is True
    assert rows[0]["reason_code"] == "unsupported_short_or_unmatched_exit"
    assert wrong_owner[1].resized is True
    assert rows[1]["decision"] == "resized"
    assert rows[1]["final_qty"] == 10
    assert rows[1]["original_action"]["qty"] == 20
    assert rows[1]["routed_action"]["qty"] == 10


def test_router_requires_context_for_portfolio_arbitration(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)

    with pytest.raises(ValueError, match="portfolio_context"):
        asyncio.run(
            router.route_actions(
                (
                    SubmitEntry(
                        strategy_id="KALCB",
                        symbol="005930",
                        qty=1,
                        order_type="LIMIT",
                        limit_price=100.0,
                        stop_price=None,
                        reason="missing_context",
                    ),
                ),
                event_ref="event-missing-context",
                event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
            )
        )


def test_router_buy_uses_provider_cash_instead_of_action_metadata(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=50.0, buyable_cash=50.0)

    results = asyncio.run(
        router.route_actions(
            (
                SubmitEntry(
                    strategy_id="KALCB",
                    symbol="005930",
                    qty=10,
                    order_type="LIMIT",
                    limit_price=100.0,
                    stop_price=None,
                    reason="metadata_cash_should_not_win",
                    metadata={"cash": 1_000_000.0},
                ),
            ),
            portfolio_context=context,
            event_ref="event-c",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    rows = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")
    assert results[0].blocked is True
    assert rows[0]["decision"] == "blocked"
    assert rows[0]["reason_code"] in {"capital_or_exposure_limit", "capacity_below_min_quantity"}


def test_router_buy_blocks_missing_or_zero_account_state(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=0.0, buyable_cash=0.0)

    results = asyncio.run(
        router.route_actions(
            (SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "zero_account"),),
            portfolio_context=context,
            event_ref="event-zero-account",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    rows = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")
    assert results[0].blocked is True
    assert rows[0]["reason_code"] == "missing_or_zero_account_state"


def test_router_pre_submit_intent_does_not_claim_broker_submission(tmp_path):
    class RejectingOMS(RecordingOMSClient):
        async def submit_intent(self, intent):
            return IntentResult(intent.intent_id, IntentStatus.REJECTED, message="unit reject")

    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RejectingOMS(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=False)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)

    asyncio.run(
        router.route_actions(
            (SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "reject"),),
            portfolio_context=context,
            event_ref="event-reject",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    intent_row = _jsonl(tmp_path / "session" / "oms_intents.jsonl")[0]
    result_row = _jsonl(tmp_path / "session" / "order_events.jsonl")[0]
    assert intent_row["intended_broker_submit"] is True
    assert intent_row["actually_submitted_to_broker"] is False
    assert intent_row["submitted_to_broker"] is False
    assert result_row["status"] == "REJECTED"
    assert result_row["actually_submitted_to_broker"] is False


def test_router_uses_provider_sector_map_when_action_metadata_lacks_sector(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    policy = PortfolioArbitrationPolicy(PortfolioPolicyConfig(max_gross_notional=10_000.0, max_symbol_notional=10_000.0, max_sector_notional=1_000.0))
    router = RuntimeActionRouter(recorder, oms, policy, portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms, sector_map={"005930": "SEMIS", "000660": "SEMIS"})
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    context.positions = {"000660": PositionInfo(symbol="000660", real_qty=10, avg_price=100.0, allocations={})}

    results = asyncio.run(
        router.route_actions(
            (SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "sector_cap"),),
            portfolio_context=context,
            event_ref="event-sector-cap",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    row = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")[0]
    assert results[0].blocked is True
    assert row["sector"] == "SEMIS"
    assert row["reason_code"] == "capital_or_exposure_limit"


def test_portfolio_context_counts_oms_working_buy_exposure_after_restart(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms, sector_map={"005930": "SEMIS"})
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    context.positions = {
        "005930": PositionInfo(
            symbol="005930",
            real_qty=0,
            avg_price=0.0,
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
                )
            ],
        )
    }

    strategy_exposure = context.strategy_exposure("KALCB", "005930")
    symbol_exposure = context.symbol_exposure("005930")
    portfolio_exposure = context.portfolio_exposure()
    sector_exposure = context.sector_exposure("SEMIS")
    duplicate = asyncio.run(
        router.route_actions(
            (SubmitEntry("OLR", "005930", 1, "LIMIT", 100.0, None, "duplicate_after_restart"),),
            portfolio_context=context,
            event_ref="event-working",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    assert strategy_exposure.qty == 6
    assert strategy_exposure.notional == 600.0
    assert symbol_exposure.qty == 6
    assert portfolio_exposure.notional == 600.0
    assert sector_exposure.notional == 600.0
    assert duplicate[0].blocked is True
    assert duplicate[0].portfolio_reason_code == "duplicate_symbol_conflict"


def test_router_rehydrates_pending_buy_reservations_from_working_orders(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms, sector_map={"005930": "SEMIS"})
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    order = WorkingOrderInfo(
        order_id="ORD-WORKING",
        symbol="005930",
        side="BUY",
        qty=10,
        filled_qty=4,
        remaining_qty=6,
        price=100.0,
        status="WORKING",
        strategy_id="KALCB",
        intent_id="intent-1",
        idempotency_key="idem-1",
    )

    evidence = router.rehydrate_pending_reservations([order], source="oms_positions", portfolio_context=context)
    duplicate = asyncio.run(
        router.route_actions(
            (SubmitEntry("OLR", "005930", 1, "LIMIT", 100.0, None, "duplicate_after_rehydrate"),),
            portfolio_context=context,
            event_ref="event-rehydrate",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    reservation = next(iter(router.pending_reservations.values()))
    rows = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")
    assert evidence["record_type"] == "pending_reservations_rehydrated"
    assert evidence["working_orders"][0]["remaining_qty"] == 6
    assert reservation.qty == 6
    assert reservation.notional == 600.0
    assert reservation.provenance == "rehydrated:oms_positions"
    assert duplicate[0].blocked is True
    assert duplicate[0].portfolio_reason_code == "duplicate_symbol_conflict"
    assert rows[0]["rehydrated_pending_notional"] == 600.0


def test_router_counts_rehydrated_context_working_buy_exposure_exactly_once(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    policy = PortfolioArbitrationPolicy(
        PortfolioPolicyConfig(max_gross_notional=700.0, max_symbol_notional=700.0, max_sector_notional=700.0)
    )
    router = RuntimeActionRouter(recorder, oms, policy, portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms, sector_map={"005930": "SEMIS", "035420": "INTERNET"})
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    order = WorkingOrderInfo(
        order_id="ORD-WORKING",
        symbol="005930",
        side="BUY",
        qty=10,
        filled_qty=4,
        remaining_qty=6,
        price=100.0,
        status="WORKING",
        strategy_id="KALCB",
        intent_id="intent-1",
        idempotency_key="idem-1",
    )
    context.positions = {
        "005930": PositionInfo(
            symbol="005930",
            real_qty=0,
            avg_price=0.0,
            allocations={},
            working_orders=[order],
        )
    }

    router.rehydrate_pending_reservations([order], source="oms_positions", portfolio_context=context)
    result = asyncio.run(
        router.route_actions(
            (SubmitEntry("OLR", "035420", 1, "LIMIT", 100.0, None, "accepted_if_working_counted_once"),),
            portfolio_context=context,
            event_ref="event-exact-once",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    assert context.portfolio_exposure().notional == 600.0
    assert router.rehydrated_pending_notional == 600.0
    assert result[0].accepted is True
    assert result[0].portfolio_reason_code == "accepted"


def test_router_excludes_terminal_working_orders_from_rehydration(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)

    router.rehydrate_pending_reservations(
        [
            WorkingOrderInfo("ORD-CANCELLED", "005930", "BUY", qty=10, remaining_qty=10, price=100.0, status="CANCELLED", strategy_id="KALCB"),
            WorkingOrderInfo("ORD-FILLED", "000660", "BUY", qty=10, remaining_qty=0, price=100.0, status="FILLED", strategy_id="KALCB"),
        ],
        source="oms_positions",
        portfolio_context=context,
    )

    assert router.pending_reservations == {}


def test_router_missing_working_order_price_fails_closed(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)

    router.rehydrate_pending_reservations(
        [WorkingOrderInfo("ORD-NOPRICE", "005930", "BUY", qty=10, remaining_qty=10, price=0.0, status="WORKING", strategy_id="KALCB")],
        source="oms_positions",
        portfolio_context=context,
    )
    result = asyncio.run(
        router.route_actions(
            (SubmitEntry("OLR", "000660", 1, "LIMIT", 100.0, None, "blocked_by_degraded_context"),),
            portfolio_context=context,
            event_ref="event-missing-price",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    assert router.portfolio_context_degraded is True
    assert result[0].blocked is True
    assert result[0].portfolio_reason_code == "working_order_price_missing"


def test_router_missing_working_order_price_does_not_use_position_avg_price(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    missing_price_order = WorkingOrderInfo(
        "ORD-NOPRICE",
        "005930",
        "BUY",
        qty=10,
        remaining_qty=10,
        price=0.0,
        status="WORKING",
        strategy_id="KALCB",
    )
    context.positions = {
        "005930": PositionInfo(
            symbol="005930",
            real_qty=10,
            avg_price=100.0,
            allocations={},
            working_orders=[missing_price_order],
        )
    }

    evidence = router.rehydrate_pending_reservations(
        [missing_price_order],
        source="oms_positions",
        portfolio_context=context,
    )
    result = asyncio.run(
        router.route_actions(
            (SubmitEntry("OLR", "000660", 1, "LIMIT", 100.0, None, "blocked_by_missing_working_price"),),
            portfolio_context=context,
            event_ref="event-missing-price-with-cost-basis",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    assert evidence["degraded"] is True
    assert evidence["degraded_reason"] == "working_order_price_missing"
    assert evidence["rehydrated_pending_notional"] == 0.0
    assert router.pending_reservations == {}
    assert context.portfolio_exposure().notional == 1_000.0
    assert result[0].blocked is True
    assert result[0].portfolio_reason_code == "working_order_price_missing"


def test_router_preserves_rehydrated_reservations_when_oms_refresh_fails(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)
    order = WorkingOrderInfo(
        order_id="ORD-WORKING",
        symbol="005930",
        side="BUY",
        qty=10,
        remaining_qty=6,
        price=100.0,
        status="WORKING",
        strategy_id="KALCB",
    )

    router.rehydrate_pending_reservations([order], source="oms_positions", portfolio_context=context)
    context.last_refresh_ts = time.time()
    context.last_refresh_ok = False
    context.last_refresh_error = "oms_context_unavailable"

    evidence = router.rehydrate_pending_reservations([], source="oms_positions", portfolio_context=context)
    result = asyncio.run(
        router.route_actions(
            (SubmitEntry("OLR", "000660", 1, "LIMIT", 100.0, None, "blocked_by_stale_context"),),
            portfolio_context=context,
            event_ref="event-stale-context",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    reservation = next(iter(router.pending_reservations.values()))
    assert evidence["preserved_existing_reservations"] is True
    assert evidence["rehydrated_pending_notional"] == 600.0
    assert reservation.reservation_id.startswith("rehydrated:oms_positions:")
    assert router.portfolio_context_degraded is True
    assert result[0].blocked is True
    assert result[0].portfolio_reason_code == "oms_context_unavailable"


def test_router_rehydrated_reservation_releases_by_aliases(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    order = WorkingOrderInfo(
        order_id="ORD-WORKING",
        symbol="005930",
        side="BUY",
        qty=10,
        remaining_qty=10,
        price=100.0,
        status="WORKING",
        strategy_id="KALCB",
        intent_id="intent-1",
        idempotency_key="idem-1",
        submit_ref="submit-1",
    )

    router.rehydrate_pending_reservations([order], source="oms_positions", portfolio_context=context)

    assert router.release_order_ref("intent-1", qty=4) is True
    reservation = next(iter(router.pending_reservations.values()))
    assert reservation.qty == 6
    assert router.release_order_ref("idem-1") is True
    assert router.pending_reservations == {}


def test_router_admitted_exposure_persists_until_released(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    policy = PortfolioArbitrationPolicy(PortfolioPolicyConfig(max_gross_notional=1_000.0, max_symbol_notional=1_000.0, max_sector_notional=1_000.0))
    router = RuntimeActionRouter(recorder, oms, policy, portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)

    first = asyncio.run(
        router.route_actions(
            (SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "first"),),
            portfolio_context=context,
            event_ref="event-1",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )
    second = asyncio.run(
        router.route_actions(
            (SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "duplicate_before_fill"),),
            portfolio_context=context,
            event_ref="event-2",
            event_timestamp=datetime(2026, 2, 2, 9, 40, tzinfo=KST),
        )
    )
    router.release_order_ref(first[0].final_order_ref)
    after_release = asyncio.run(
        router.route_actions(
            (SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "after_release"),),
            portfolio_context=context,
            event_ref="event-3",
            event_timestamp=datetime(2026, 2, 2, 9, 45, tzinfo=KST),
        )
    )

    assert first[0].accepted is True
    assert second[0].blocked is True
    assert second[0].portfolio_reason_code == "duplicate_symbol_conflict"
    assert after_release[0].accepted is True


def test_router_applies_portfolio_priority_before_batch_arrival_order(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    policy = PortfolioArbitrationPolicy(PortfolioPolicyConfig(strategy_priority=("OLR", "KALCB")))
    router = RuntimeActionRouter(recorder, oms, policy, portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)

    results = asyncio.run(
        router.route_actions(
            (
                SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "kalcb_arrived_first"),
                SubmitEntry("OLR", "005930", 1, "LIMIT", 100.0, None, "olr_priority_wins"),
            ),
            portfolio_context=context,
            event_ref="event-priority",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )

    rows = _jsonl(tmp_path / "session" / "portfolio_arbitration.jsonl")
    assert [row["strategy_id"] for row in rows] == ["OLR", "KALCB"]
    assert results[0].routed_action.strategy_id == "OLR"
    assert results[0].accepted is True
    assert results[1].blocked is True
    assert results[1].portfolio_reason_code == "duplicate_symbol_conflict"


def test_router_partial_fill_releases_only_filled_reservation(tmp_path):
    recorder = PaperSessionRecorder(tmp_path / "session", date(2026, 2, 2))
    oms = RecordingOMSClient(recorder)
    router = RuntimeActionRouter(recorder, oms, PortfolioArbitrationPolicy(), portfolio_enabled=True, dry_run=True)
    context = PortfolioContextProvider(oms)
    context.account_state = AccountState(equity=1_000_000.0, buyable_cash=1_000_000.0)

    first = asyncio.run(
        router.route_actions(
            (SubmitEntry("KALCB", "005930", 10, "LIMIT", 100.0, None, "first"),),
            portfolio_context=context,
            event_ref="event-1",
            event_timestamp=datetime(2026, 2, 2, 9, 35, tzinfo=KST),
        )
    )
    assert first[0].accepted is True

    assert router.release_order_ref(first[0].final_order_ref, qty=4) is True
    reservation = next(iter(router.pending_reservations.values()))
    assert reservation.qty == 6
    assert reservation.notional == pytest.approx(600.0)

    duplicate = asyncio.run(
        router.route_actions(
            (SubmitEntry("KALCB", "005930", 1, "LIMIT", 100.0, None, "duplicate_while_remainder_pending"),),
            portfolio_context=context,
            event_ref="event-2",
            event_timestamp=datetime(2026, 2, 2, 9, 40, tzinfo=KST),
        )
    )
    assert duplicate[0].blocked is True
    assert duplicate[0].portfolio_reason_code == "duplicate_symbol_conflict"

    assert router.release_order_ref(first[0].final_order_ref, qty=6) is True
    assert router.pending_reservations == {}


def test_portfolio_context_sector_exposure_uses_sector_map():
    context = PortfolioContextProvider(sector_map={"005930": "SEMIS", "035420": "INTERNET"})
    context.positions = {
        "005930": PositionInfo(symbol="005930", real_qty=10, avg_price=100.0, allocations={}),
        "035420": PositionInfo(symbol="035420", real_qty=5, avg_price=200.0, allocations={}),
    }

    semis = context.sector_exposure("SEMIS")
    internet = context.sector_exposure("internet")

    assert semis.qty == 10
    assert semis.notional == 1_000.0
    assert internet.qty == 5
    assert internet.notional == 1_000.0


def test_replay_paper_session_writes_zero_mismatch_report(tmp_path):
    session = tmp_path / "2026-02-02"
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))
    recorder.append_jsonl("decision_stream.jsonl", {"strategy_id": "KALCB", "symbol": "005930", "decision_code": "noop"})
    recorder.append_jsonl("portfolio_arbitration.jsonl", {"decision": "accepted", "strategy_id": "KALCB"})
    (session / "market_bars_5m.parquet").write_bytes(b"fixture-bars")
    recorder.write_end_of_day_positions({"positions": []})
    hashes = session_hashes(session)
    recorder.write_manifest({"expected_hashes": {key: value for key, value in hashes.items() if key != "session_manifest"}})

    report = replay_paper_session(session)
    summary = summarize_paper_parity(tmp_path)

    assert report["hash_contract_passed"] is False
    assert report["hash_contract_available"] is False
    assert report["behavior_parity_passed"] is False
    assert report["paper_gate_passed"] is False
    assert report["paper_gate_status"] == "blocked_until_offline_rebuild"
    assert "offline_rebuild_not_implemented" in report["promotion_blockers"]
    assert "hash_contract_unsealed" in report["promotion_blockers"]
    assert "hash_contract_incomplete" in report["promotion_blockers"]
    assert report["replay_mode"] == "hash_contract_only"
    assert report["offline_rebuild_implemented"] is False
    assert json.loads((session / "parity_report.json").read_text(encoding="utf-8"))["mismatches"] == []
    assert summary["sessions_analyzed"] == 1
    assert summary["sessions_passing"] == 0
    assert summary["sessions_with_hash_contract"] == 0
    assert summary["sessions_hash_contract_passing"] == 0
    assert summary["sessions_paper_gate_passing"] == 0
    assert summary["portfolio_decisions"]["accepted"] == 1


def test_replay_paper_session_requires_expected_hash_contract(tmp_path):
    session = tmp_path / "2026-02-02"
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))
    (session / "market_bars_5m.parquet").write_bytes(b"fixture-bars")
    recorder.write_end_of_day_positions({"positions": []})
    recorder.write_manifest()

    report = replay_paper_session(session)

    assert report["hash_contract_available"] is False
    assert report["hash_contract_passed"] is False
    assert report["paper_gate_passed"] is False
    assert "hash_contract_missing" in report["promotion_blockers"]
    assert "hash_contract_mismatch" not in report["promotion_blockers"]


def test_replay_paper_session_rejects_manual_hashes_without_sealed_closeout(tmp_path):
    session = tmp_path / "2026-02-02"
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))
    (session / "market_bars_5m.parquet").write_bytes(b"fixture-bars")
    recorder.write_end_of_day_positions({"positions": []})
    hashes = session_hashes(session)
    recorder.write_manifest({"expected_hashes": {key: value for key, value in hashes.items() if key != "session_manifest"}})

    report = replay_paper_session(session)

    assert report["hash_contract_status"] == ""
    assert report["hash_contract_available"] is False
    assert report["hash_contract_passed"] is False
    assert report["paper_gate_passed"] is False
    assert "hash_contract_unsealed" in report["promotion_blockers"]
    assert "hash_contract_incomplete" in report["promotion_blockers"]


def test_replay_cli_hash_contract_only_is_non_success_without_debug_opt_in(monkeypatch, capsys):
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "replay_paper_session.py"
    spec = importlib.util.spec_from_file_location("unit_replay_paper_session", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = {
        "session": "unit-session",
        "replay_mode": "hash_contract_only",
        "behavior_parity_passed": False,
        "paper_gate_passed": False,
        "paper_gate_status": "blocked_until_offline_rebuild",
        "promotion_blockers": ["offline_rebuild_not_implemented"],
        "hash_contract_passed": True,
        "session_bundle_complete": True,
        "mismatches": [],
    }
    monkeypatch.setattr(module, "replay_paper_session", lambda _session: report)
    monkeypatch.setattr(sys, "argv", ["replay_paper_session.py", "--session", "unit", "--hash-contract-only"])

    assert module.main() == 1

    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay_paper_session.py", "--session", "unit", "--hash-contract-only", "--allow-debug-success"],
    )
    assert module.main() == 0


def test_replay_paper_session_flags_behavior_stream_perturbation(tmp_path):
    session = tmp_path / "2026-02-02"
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))
    recorder.append_jsonl("strategy_actions.jsonl", {"strategy_id": "KALCB", "symbol": "005930", "action_type": "SubmitEntry", "qty": 1})
    recorder.append_jsonl("portfolio_arbitration.jsonl", {"decision": "accepted", "strategy_id": "KALCB"})
    (session / "market_bars_5m.parquet").write_bytes(b"fixture-bars")
    recorder.write_end_of_day_positions({"positions": []})
    hashes = session_hashes(session)
    recorder.write_manifest({"expected_hashes": {key: value for key, value in hashes.items() if key != "session_manifest"}})
    (session / "strategy_actions.jsonl").write_text(
        json.dumps({"strategy_id": "KALCB", "symbol": "005930", "action_type": "SubmitEntry", "qty": 2}) + "\n",
        encoding="utf-8",
    )

    report = replay_paper_session(session)

    assert report["hash_contract_passed"] is False
    assert report["paper_gate_passed"] is False
    assert "hash_contract_mismatch" in report["promotion_blockers"]
    assert any(item["class"] == "action_serialization_mismatch" and item["key"] == "strategy_actions" for item in report["mismatches"])


def test_session_hashes_ignore_noncausal_dry_run_oms_fields(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    PaperSessionRecorder(left, date(2026, 2, 2)).append_jsonl(
        "oms_intents.jsonl",
        {
            "recorded_at": "2026-02-02T01:00:00Z",
            "intent_id": "random-a",
            "idempotency_key": "KALCB:005930:ENTER:20260202:a",
            "timestamp": 1.0,
            "intent_type": "ENTER",
            "strategy_id": "KALCB",
            "symbol": "005930",
            "desired_qty": 1,
        },
    )
    PaperSessionRecorder(right, date(2026, 2, 2)).append_jsonl(
        "oms_intents.jsonl",
        {
            "recorded_at": "2026-02-02T02:00:00Z",
            "intent_id": "random-b",
            "idempotency_key": "KALCB:005930:ENTER:20260203:b",
            "timestamp": 2.0,
            "intent_type": "ENTER",
            "strategy_id": "KALCB",
            "symbol": "005930",
            "desired_qty": 1,
        },
    )

    assert session_hashes(left)["oms_intents"] == session_hashes(right)["oms_intents"]


def test_session_hashes_ignore_paper_vs_dry_run_mode_fields(tmp_path):
    for name, record_type, dry_run, submitted in (
        ("dry", "dry_run_oms_intent", True, False),
        ("paper", "oms_intent", False, True),
    ):
        recorder = PaperSessionRecorder(tmp_path / name, date(2026, 2, 2))
        recorder.append_jsonl(
            "oms_intents.jsonl",
            {
                "record_type": record_type,
                "dry_run": dry_run,
                "submitted_to_broker": submitted,
                "intent_type": "ENTER",
                "strategy_id": "KALCB",
                "symbol": "005930",
                "desired_qty": 1,
            },
        )
        recorder.append_jsonl(
            "order_events.jsonl",
            {
                "record_type": "dry_run_order_result" if dry_run else "oms_order_result",
                "dry_run": dry_run,
                "submitted_to_broker": submitted,
                "action_ref": "action-a",
                "status": "ACCEPTED",
                "modified_qty": None,
            },
        )

    assert session_hashes(tmp_path / "dry")["oms_intents"] == session_hashes(tmp_path / "paper")["oms_intents"]
    assert session_hashes(tmp_path / "dry")["order_events"] == session_hashes(tmp_path / "paper")["order_events"]


def test_session_hashes_ignore_broker_only_order_ids(tmp_path):
    for name, order_id in (("left", "broker-a"), ("right", "broker-b")):
        recorder = PaperSessionRecorder(tmp_path / name, date(2026, 2, 2))
        recorder.append_jsonl(
            "order_events.jsonl",
            {
                "record_type": "oms_order_result",
                "action_ref": "action-a",
                "intent_id": f"intent-{order_id}",
                "order_id": order_id,
                "status": "ACCEPTED",
                "modified_qty": None,
            },
        )

    assert session_hashes(tmp_path / "left")["order_events"] == session_hashes(tmp_path / "right")["order_events"]


def test_session_hashes_ignore_original_order_ids_and_artifact_paths(tmp_path):
    variants = (
        ("left", "broker-a", "C:/tmp/live/olr.json", "C:/tmp/session-a/olr.json"),
        ("right", "broker-b", "D:/paper/live/olr.json", "D:/paper/session-b/olr.json"),
    )
    for name, original_order_id, source_path, session_path in variants:
        recorder = PaperSessionRecorder(tmp_path / name, date(2026, 2, 2))
        recorder.append_jsonl(
            "order_events.jsonl",
            {
                "record_type": "runtime_order_event",
                "action_ref": "action-a",
                "status": "ACCEPTED",
                "modified_qty": None,
                "event": {"metadata": {"original_order_id": original_order_id}},
            },
        )
        recorder.append_jsonl(
            "fill_events.jsonl",
            {
                "record_type": "runtime_fill_event",
                "event": {"metadata": {"original_order_id": original_order_id}},
            },
        )
        recorder.append_jsonl(
            "artifact_generation.jsonl",
            {
                "record_type": "artifact_generation",
                "strategy_id": "OLR",
                "trade_date": "2026-02-02",
                "stage": "final_afternoon_1430",
                "artifact_hash": "artifact-a",
                "source_fingerprint": "source-a",
                "candidate_count": 1,
                "bucket": "olr_final_snapshots",
                "source_path": source_path,
                "session_path": session_path,
            },
        )

    left_hashes = session_hashes(tmp_path / "left")
    right_hashes = session_hashes(tmp_path / "right")
    assert left_hashes["order_events"] == right_hashes["order_events"]
    assert left_hashes["fill_events"] == right_hashes["fill_events"]
    assert left_hashes["artifact_generation"] == right_hashes["artifact_generation"]


def test_session_hashes_ignore_local_strategy_config_paths(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_recorder = PaperSessionRecorder(left, date(2026, 2, 2))
    right_recorder = PaperSessionRecorder(right, date(2026, 2, 2))
    common = {
        "payload": {"kalcb.session.ws_budget": 3},
        "payload_hash": "payload-hash",
        "approved_config_fingerprint": {
            "artifact_config_hash": "artifact-config-hash",
            "optimized_config_sha256": "config-sha",
        },
    }
    left_recorder.write_manifest(
        {
            "strategy_configs": {
                "KALCB": {
                    **common,
                    "source_path": "C:/local/a/kalcb_optimized_config.json",
                    "source_manifest": "C:/local/a/baseline_manifest.json",
                    "approved_config_fingerprint": {
                        **common["approved_config_fingerprint"],
                        "optimized_config_path": "C:/local/a/kalcb_optimized_config.json",
                    },
                }
            }
        }
    )
    right_recorder.write_manifest(
        {
            "strategy_configs": {
                "KALCB": {
                    **common,
                    "source_path": "D:/ci/b/kalcb_optimized_config.json",
                    "source_manifest": "D:/ci/b/baseline_manifest.json",
                    "approved_config_fingerprint": {
                        **common["approved_config_fingerprint"],
                        "optimized_config_path": "D:/ci/b/kalcb_optimized_config.json",
                    },
                }
            }
        }
    )

    assert session_hashes(left)["strategy_configs_manifest"] == session_hashes(right)["strategy_configs_manifest"]


def test_replay_paper_session_blocks_copied_offline_streams_without_engine_manifest(tmp_path):
    session = tmp_path / "2026-02-02"
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))
    recorder.append_jsonl("decision_stream.jsonl", {"strategy_id": "KALCB", "symbol": "005930", "decision_code": "entry"})
    recorder.append_jsonl("strategy_actions.jsonl", {"strategy_id": "KALCB", "symbol": "005930", "action_type": "SubmitEntry"})
    recorder.append_jsonl("portfolio_arbitration.jsonl", {"decision": "accepted", "strategy_id": "KALCB"})
    recorder.append_jsonl("oms_intents.jsonl", {"intent_type": "ENTER", "strategy_id": "KALCB", "symbol": "005930"})
    recorder.append_jsonl("state_snapshots.jsonl", {"strategy_id": "KALCB", "state_hash": "state"})
    (session / "market_bars_5m.parquet").write_bytes(b"fixture-bars")
    recorder.write_end_of_day_positions({"positions": []})

    offline = session / "offline_replay"
    offline.mkdir()
    for name in (
        "decision_stream.jsonl",
        "strategy_actions.jsonl",
        "portfolio_arbitration.jsonl",
        "oms_intents.jsonl",
        "state_snapshots.jsonl",
        "end_of_day_positions.json",
    ):
        shutil.copy2(session / name, offline / name)
    hashes = session_hashes(session)
    recorder.write_manifest({"expected_hashes": {key: value for key, value in hashes.items() if key != "session_manifest"}})

    report = replay_paper_session(session)

    assert report["offline_rebuild_implemented"] is False
    assert report["offline_rebuild_status"] == "missing_engine_replay_manifest"
    assert report["replay_mode"] == "external_offline_stream_contract"
    assert report["paper_gate_passed"] is False
    assert report["behavior_parity_passed"] is False
    assert "offline_rebuild_not_implemented" in report["promotion_blockers"]


def test_replay_paper_session_rejects_incomplete_offline_rebuild(tmp_path):
    session = tmp_path / "2026-02-02"
    recorder = PaperSessionRecorder(session, date(2026, 2, 2))
    recorder.append_jsonl("decision_stream.jsonl", {"strategy_id": "KALCB", "symbol": "005930", "decision_code": "entry"})
    recorder.append_jsonl("strategy_actions.jsonl", {"strategy_id": "KALCB", "symbol": "005930", "action_type": "SubmitEntry"})
    recorder.append_jsonl("portfolio_arbitration.jsonl", {"decision": "accepted", "strategy_id": "KALCB"})
    recorder.append_jsonl("oms_intents.jsonl", {"intent_type": "ENTER", "strategy_id": "KALCB", "symbol": "005930"})
    recorder.append_jsonl("state_snapshots.jsonl", {"strategy_id": "KALCB", "state_hash": "state"})
    (session / "market_bars_5m.parquet").write_bytes(b"fixture-bars")
    recorder.write_end_of_day_positions({"positions": []})
    offline = session / "offline_replay"
    offline.mkdir()
    write_offline_replay_manifest(offline, source_session=session)
    shutil.copy2(session / "decision_stream.jsonl", offline / "decision_stream.jsonl")
    hashes = session_hashes(session)
    recorder.write_manifest({"expected_hashes": {key: value for key, value in hashes.items() if key != "session_manifest"}})

    report = replay_paper_session(session)

    assert report["offline_rebuild_implemented"] is True
    assert report["paper_gate_passed"] is False
    assert "offline_rebuild_mismatch" in report["promotion_blockers"]
    assert any(item["actual"] is None for item in report["mismatches"])
    assert any(item["key"] == "offline_replay.driver_replay" for item in report["mismatches"])


def test_replay_input_loader_requires_explicit_config_account_and_positions(tmp_path):
    session = tmp_path / "2026-02-02"
    session.mkdir()
    (session / "session_manifest.json").write_text(
        json.dumps({"trade_date": "2026-02-02", "strategy_configs": {"KALCB": {}}}),
        encoding="utf-8",
    )
    loader = ReplayInputLoader(session)

    with pytest.raises(ValueError, match="config payload"):
        loader.load_configs(("KALCB",))
    with pytest.raises(ValueError, match="initial account"):
        loader.initial_account_state()
    with pytest.raises(ValueError, match="initial positions"):
        loader.initial_positions()


def test_replay_input_loader_reads_rehydrated_working_order_snapshot(tmp_path):
    session = tmp_path / "2026-02-02"
    session.mkdir()
    (session / "session_manifest.json").write_text(json.dumps({"trade_date": "2026-02-02"}), encoding="utf-8")
    (session / "portfolio_arbitration.jsonl").write_text(
        json.dumps(
            {
                "record_type": "pending_reservations_rehydrated",
                "working_orders": [
                    {
                        "strategy_id": "KALCB",
                        "symbol": "005930",
                        "side": "BUY",
                        "remaining_qty": 6,
                        "price": 100.0,
                        "status": "WORKING",
                        "sector": "SEMIS",
                        "intent_id": "intent-1",
                        "idempotency_key": "idem-1",
                        "missing_price": False,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    orders = ReplayInputLoader(session).initial_working_orders()

    assert orders == [
        {
            "strategy_id": "KALCB",
            "symbol": "005930",
            "side": "BUY",
            "remaining_qty": 6,
            "price": 100.0,
            "status": "WORKING",
            "sector": "SEMIS",
            "intent_id": "intent-1",
            "idempotency_key": "idem-1",
            "missing_price": False,
        }
    ]


def test_replay_input_loader_unwraps_optimized_config_mutations(tmp_path):
    session = tmp_path / "2026-02-02"
    config_path = session / "strategy_configs" / "kalcb_optimized_config.json"
    mutations = {"kalcb.session.ws_budget": 3}
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"mutations": mutations}, sort_keys=True), encoding="utf-8")
    session.mkdir(exist_ok=True)
    (session / "session_manifest.json").write_text(
        json.dumps(
            {
                "trade_date": "2026-02-02",
                "strategy_configs": {
                    "KALCB": {
                        "path": "strategy_configs/kalcb_optimized_config.json",
                        "mutation_hash": canonical_json_hash(mutations),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    configs = ReplayInputLoader(session).load_configs(("KALCB",))

    assert configs["KALCB"].ws_budget == 3


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
