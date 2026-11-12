import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from deployment.olr_kalcb.execution_feedback import drain_execution_feedback
from oms.execution_feedback import ExecutionFeedback
from oms.oms_core import OMSCore
from oms.state import WorkingOrder
from oms_client.client import AccountState, AllocationInfo, PositionInfo
from strategy_kalcb.engine import KALCBEngine
from strategy_olr.engine import OLREngine
from tests.deployment.olr_kalcb.test_session_driver import _driver


def test_stream_rewind_or_account_change_blocks_before_actions():
    from deployment.olr_kalcb.execution_feedback import _validate_stream
    state = SimpleNamespace(meta={"oms_feedback_stream": "account-a"})
    with pytest.raises(ValueError, match="exceeds"):
        _validate_stream(state, 9, {"watermark": 0, "stream_id": "account-a"})
    with pytest.raises(ValueError, match="changed"):
        _validate_stream(state, 0, {"watermark": 9, "stream_id": "account-b"})
    assert state.meta["oms_feedback_stream"] == "account-a"


@pytest.mark.parametrize("strategy,engine_type", [("KALCB", KALCBEngine), ("OLR", OLREngine)])
def test_empty_canonical_stream_has_a_durable_checkpoint_before_first_trade(tmp_path, strategy, engine_type):
    from deployment.olr_kalcb.execution_feedback import checkpoint_path
    class Client:
        async def get_execution_feedback(self, *args):
            return []
        async def get_execution_snapshot(self, *args):
            return {"events": [], "watermark": 0, "has_more": False, "allocations": {},
                    "stream_id": f"postgres-execution-v2:paper:{strategy}:generation"}
        async def get_account_state(self):
            return AccountState(equity=1000, buyable_cash=1000)
        async def get_all_positions(self):
            return {}
    client = Client()
    driver = _driver(tmp_path, engine_type(), mode="paper", strategy_id=strategy, oms=client)
    assert not checkpoint_path(driver).exists()
    assert asyncio.run(drain_execution_feedback(SimpleNamespace(mode="paper", action_router=driver.action_router,
        drivers={strategy: driver}))) == 0
    packet = json.loads(checkpoint_path(driver).read_text(encoding="utf-8"))
    assert packet["state"]["meta"]["oms_feedback_cursor"] == 0
    restarted = _driver(tmp_path, engine_type(), mode="paper", strategy_id=strategy, oms=client)
    assert restarted.descriptor.engine.state.meta["oms_feedback_stream"].endswith(":generation")


@pytest.mark.parametrize("strategy,engine_type", [("KALCB", KALCBEngine), ("OLR", OLREngine)])
def test_actual_oms_producer_partial_restart_completion_and_exit(tmp_path, strategy, engine_type):
    feed = ExecutionFeedback(tmp_path / "feedback.sqlite3")
    core = OMSCore(MagicMock(), execution_feedback=feed)

    class Client:
        async def get_execution_feedback(self, strategy_id, after=0):
            return feed.read(strategy_id, after)

        async def get_account_state(self):
            return AccountState(equity=1_000_000, buyable_cash=900_000)

        async def get_all_positions(self):
            pos = core.state.get_position("005930")
            alloc = pos.allocations.get(strategy)
            if not alloc or not alloc.qty:
                return {}
            return {"005930": PositionInfo(symbol="005930", real_qty=alloc.qty, avg_price=alloc.cost_basis,
                allocations={strategy: AllocationInfo(strategy, alloc.qty, alloc.cost_basis)})}

        async def submit_intent(self, intent):
            from oms.intent import IntentResult, IntentStatus
            return IntentResult(intent.intent_id, IntentStatus.ACCEPTED, order_id=intent.intent_id)

    client = Client()
    def driver():
        return _driver(tmp_path, engine_type(), mode="paper", strategy_id=strategy, oms=client)

    def drain(d):
        return asyncio.run(drain_execution_feedback(SimpleNamespace(mode="paper", action_router=d.action_router,
            drivers={strategy: d})))

    order = WorkingOrder("entry", "005930", "BUY", 10, strategy_id=strategy)
    asyncio.run(core._apply_fill(order, 4, fill_price=100))
    order.filled_qty = 4
    first = driver()
    assert drain(first) == 1
    assert first.descriptor.engine.state.symbol_state("005930").position.qty_open == 4
    restarted = driver()
    assert drain(restarted) == 0
    asyncio.run(core._apply_fill(order, 6, fill_price=110))
    order.filled_qty = 10
    assert drain(restarted) == 1
    position = restarted.descriptor.engine.state.symbol_state("005930").position
    assert position.qty_open == 10 and position.entry_price == 106
    assert len(feed.read(strategy)) == 2
    sell = WorkingOrder("exit", "005930", "SELL", 10, strategy_id=strategy)
    asyncio.run(core._apply_fill(sell, 6, fill_price=120))
    sell.filled_qty = 6
    drain(restarted)
    assert position.qty_open == 4
    asyncio.run(core._apply_fill(sell, 4, fill_price=121))
    drain(restarted)
    assert restarted.descriptor.engine.state.symbol_state("005930").position is None
    assert drain(driver()) == 0


def test_outbox_collision_is_not_silent_and_survives_reopen(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    first = ExecutionFeedback(path)
    first.append("fill:a", "KALCB", {"qty": 4})
    first.append("fill:a", "KALCB", {"qty": 4})
    second = ExecutionFeedback(path)
    assert second.read("KALCB") == [{"sequence": 1, "qty": 4}]
    with pytest.raises(ValueError, match="collision"):
        second.append("fill:a", "KALCB", {"qty": 6})


def test_feedback_reconciliation_blocks_missing_execution(tmp_path):
    d = _driver(tmp_path, KALCBEngine())
    d.mode = "paper"
    async def empty(*args):
        return []
    class Client:
        get_execution_feedback = staticmethod(empty)
        async def get_account_state(self):
            return AccountState(equity=1_000_000, buyable_cash=900_000)
        async def get_all_positions(self):
            return {"005930": PositionInfo("005930", real_qty=4, avg_price=100,
                allocations={"KALCB": AllocationInfo("KALCB", 4, 100)})}
    client = Client()
    d.action_router.oms_client = client
    d.portfolio_context.oms_client = client
    with pytest.raises(RuntimeError, match="disagree"):
        asyncio.run(drain_execution_feedback(SimpleNamespace(mode="paper", action_router=d.action_router,
            drivers={"KALCB": d})))


def test_runtime_crash_after_real_oms_acceptance_recovers_exact_action(tmp_path):
    from oms.intent import IntentStatus
    from tests.deployment.olr_kalcb.test_session_driver import _SubmittingKALCBPositionActionEngine, _bar
    core = OMSCore(MagicMock())
    core.state.set_allocation("005930", "KALCB", 10, cost_basis=100)
    calls = []
    original = core._handle_modify_risk
    async def counted(intent):
        calls.append(intent.idempotency_key)
        return await original(intent)
    core._handle_modify_risk = counted

    class Client:
        fail = True
        async def get_account_state(self):
            return AccountState(equity=1_000_000, buyable_cash=900_000)
        async def get_all_positions(self):
            return {"005930": PositionInfo("005930", real_qty=10, avg_price=100,
                allocations={"KALCB": AllocationInfo("KALCB", 10, 100)})}
        async def submit_intent(self, intent):
            result = await core.submit_intent(intent)
            assert result.status not in {IntentStatus.REJECTED, IntentStatus.DEFERRED}
            if self.fail:
                self.fail = False
                raise RuntimeError("lost response after OMS acceptance")
            return result
    client = Client()
    def create():
        driver = _driver(tmp_path, _SubmittingKALCBPositionActionEngine("stop"), mode="paper", oms=client)
        driver.action_router.dry_run = False
        return driver
    first = create()
    with pytest.raises(RuntimeError, match="lost response"):
        asyncio.run(first.handle_bar(_bar("005930")))
    assert first.pending_event_payload is not None
    restarted = create()
    assert restarted.pending_event_payload is not None
    asyncio.run(restarted.resume_pending_event())
    assert restarted.pending_event_payload is None and len(calls) == 1
    assert not create().pending_event_payload
