"""Boundary tests; the optional integration target must be an isolated test DB."""
import copy
from datetime import datetime
import os
from pathlib import Path
from unittest.mock import MagicMock
import uuid
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
import pytest_asyncio

from oms.oms_core import OMSCore
from oms.persistence import OMSPersistence
from oms.state import WorkingOrder, OrderStatus


@pytest_asyncio.fixture
async def persistence():
    dsn = os.environ.get("INTEGRITY_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("INTEGRITY_TEST_DATABASE_URL must name an isolated test database")
    schema = "integrity_" + uuid.uuid4().hex
    admin = await asyncpg.connect(dsn)
    await admin.execute(f'CREATE DATABASE "{schema}"')
    parts = urlsplit(dsn)
    isolated_dsn = urlunsplit(parts._replace(path="/" + schema))
    store = OMSPersistence(dsn=isolated_dsn, oms_id="test-account")
    store.pool = await asyncpg.create_pool(isolated_dsn, min_size=1, max_size=3)
    await store.pool.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    root = Path(__file__).resolve().parents[2] / "infra/postgres/init"
    try:
        for filename in ("002_tables.sql", "005_oms_scoping.sql", "007_oms_scope_finalize.sql", "008_protective_stops.sql", "009_idempotency_hardening.sql", "010_execution_commits.sql", "011_execution_generation.sql"):
            await store.pool.execute((root / filename).read_text(encoding="utf-8"))
        yield store
    finally:
        await store.pool.close()
        await admin.execute(f'DROP DATABASE "{schema}"')
        await admin.close()


async def order(store, name="entry", side="BUY", qty=10, day="2026-01-05"):
    oid = uuid.uuid4()
    await store.pool.execute("""INSERT INTO orders(oms_order_id,oms_id,strategy_id,symbol,side,order_type,qty,status,kis_order_id,kis_order_date)
        VALUES($1,$2,'KALCB','005930',$3,'MARKET',$4,'WORKING',$5,$6)""",
        oid, store.oms_id, side, qty, name, day.replace("-", ""))
    return WorkingOrder(name, "005930", side, qty, strategy_id="KALCB", oms_order_id=str(oid),
                        created_at=datetime.fromisoformat(day), risk_hard_stop_px=95 if side == "BUY" else None)


@pytest.mark.asyncio
async def test_commit_lost_ack_retry_and_restart(persistence):
    core = OMSCore(MagicMock(), persistence=persistence)
    entry = await order(persistence)
    original = persistence.commit_execution
    async def lost_ack(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("lost commit acknowledgment")
    persistence.commit_execution = lost_ack
    with pytest.raises(RuntimeError, match="lost commit"):
        await core._apply_fill(entry, 4, fill_price=100, cumulative_qty=4)
    assert entry.filled_qty == 0
    assert core.state.get_position("005930").get_allocation("KALCB") == 0
    persistence.commit_execution = original
    await core._apply_fill(entry, 4, fill_price=100, cumulative_qty=4)
    await core._apply_fill(entry, 4, fill_price=100, cumulative_qty=4)
    assert core.state.get_position("005930").get_allocation("KALCB") == 4
    assert entry.filled_qty == 4
    assert len(await persistence.read_execution_feedback("KALCB")) == 1
    assert await persistence.pool.fetchval("SELECT qty FROM protective_stops") == 4
    await core._apply_fill(entry, 6, fill_price=110, cumulative_qty=10)
    allocation = core.state.get_position("005930").allocations["KALCB"]
    assert allocation.qty == 10 and allocation.cost_basis == 106
    restored = await persistence.load_allocations()
    assert restored["005930"]["KALCB"].qty == 10
    sell = await order(persistence, "exit", "SELL")
    await core._apply_fill(sell, 6, fill_price=120, cumulative_qty=6)
    assert core.state.get_position("005930").get_allocation("KALCB") == 4
    assert await persistence.pool.fetchval("SELECT qty FROM protective_stops") == 4
    assert await persistence.pool.fetchval("SELECT realized_pnl_krw FROM trades") == 84
    realized = await persistence.load_daily_realized_pnl(datetime.fromisoformat((await persistence.read_execution_feedback('KALCB'))[-1]['timestamp']).date())
    assert realized == {"KALCB": 84}
    await core._finalize_working_order(sell, OrderStatus.CANCELLED, OrderStatus.PARTIAL, "CANCELLED")
    events = await persistence.read_execution_feedback("KALCB")
    assert len(events) == 4 and events[-1]["status"] == "CANCELLED"


@pytest.mark.asyncio
async def test_transaction_rolls_back_every_effect_and_rejects_collision(persistence):
    core = OMSCore(MagicMock(), persistence=persistence)
    entry = await order(persistence)
    original = persistence._execution_trade
    async def fail(*args):
        raise RuntimeError("before commit")
    persistence._execution_trade = fail
    with pytest.raises(RuntimeError, match="before commit"):
        await core._apply_fill(entry, 4, fill_price=100, cumulative_qty=4, execution_id="broker-exec")
    for table in ("fills", "allocations", "execution_commits", "execution_feedback", "protective_stops"):
        assert await persistence.pool.fetchval(f"SELECT count(*) FROM {table}") == 0
    persistence._execution_trade = original
    await core._apply_fill(entry, 4, fill_price=100, cumulative_qty=4, execution_id="broker-exec")
    with pytest.raises(ValueError, match="collision"):
        await core._apply_fill(entry, 4, fill_price=101, cumulative_qty=4, execution_id="broker-exec")


@pytest.mark.asyncio
async def test_same_order_and_broker_execution_ids_are_scoped(persistence):
    for account in ("first", "second"):
        persistence.oms_id = account
        core = OMSCore(MagicMock(), persistence=persistence)
        entry = await order(persistence)
        await core._apply_fill(entry, 4, fill_price=100, cumulative_qty=4, execution_id="same-id")
    assert await persistence.pool.fetchval("SELECT count(*) FROM fills") == 2


@pytest.mark.asyncio
async def test_concurrent_delivery_snapshot_and_stale_action(persistence):
    import asyncio
    from oms.intent import Intent, IntentType, IntentStatus
    first, second = OMSCore(MagicMock(), persistence=persistence), OMSCore(MagicMock(), persistence=persistence)
    entry = await order(persistence)
    await asyncio.gather(first._apply_fill(entry, 4, fill_price=100, cumulative_qty=4, execution_id="same"),
                         second._apply_fill(copy.deepcopy(entry), 4, fill_price=100, cumulative_qty=4, execution_id="same"))
    assert await persistence.pool.fetchval("SELECT count(*) FROM fills") == 1
    packet = await persistence.execution_snapshot("KALCB")
    assert packet["allocations"] == {"005930": 4}
    assert packet["watermark"] == packet["events"][-1]["sequence"] and not packet["has_more"]
    intent = Intent(IntentType.MODIFY_RISK, "KALCB", "005930", metadata={
        "action_ref": "stale", "execution_feedback_sequence": 0})
    result = await first.submit_intent(intent)
    assert result.status == IntentStatus.REJECTED and "superseded" in result.message


@pytest.mark.asyncio
async def test_action_effect_conflict_survives_core_restart(persistence):
    from oms.intent import Intent, IntentType
    first = Intent(IntentType.EXIT, "KALCB", "005930", desired_qty=1, metadata={"action_ref": "target-1"})
    await persistence.verify_action_identity(first)
    await persistence.verify_action_identity(copy.deepcopy(first))
    changed = copy.deepcopy(first)
    changed.desired_qty = 2
    with pytest.raises(ValueError, match="different execution effects"):
        await persistence.verify_action_identity(changed)


@pytest.mark.asyncio
async def test_lost_ack_is_retried_before_larger_broker_cumulative(persistence):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    adapter = MagicMock()
    core = OMSCore(adapter, persistence=persistence)
    entry = await order(persistence)
    core.state.get_position(entry.symbol).working_orders.append(entry)
    original = persistence.commit_execution
    async def lost_ack(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("lost acknowledgment")
    persistence.commit_execution = lost_ack
    with pytest.raises(RuntimeError, match="lost acknowledgment"):
        await core._apply_fill(entry, 4, fill_price=100, cumulative_qty=4)
    persistence.commit_execution = original
    core.adapter.get_orders = AsyncMock(return_value=SimpleNamespace(ok=True, data=[
        SimpleNamespace(order_id=entry.order_id, branch="", filled_qty=10, price=106)]))
    await core._sync_working_orders()
    assert entry.filled_qty == 10 and not core._pending_execution_events
    assert core.state.get_position(entry.symbol).get_allocation("KALCB") == 10
    events = [e for e in await persistence.read_execution_feedback("KALCB") if e["kind"] == "fill"]
    assert [(e["qty"], e["price"]) for e in events] == [(4, 100), (6, 110)]
    assert await persistence.pool.fetchval("SELECT qty FROM protective_stops") == 10


@pytest.mark.asyncio
async def test_process_loss_before_commit_recovers_cumulative_broker_observation(persistence):
    import pandas as pd
    from types import SimpleNamespace
    first = OMSCore(MagicMock(), persistence=persistence)
    entry = await order(persistence)
    await persistence.pool.execute("UPDATE orders SET meta=$1::jsonb WHERE oms_order_id=$2::uuid",
        persistence._order_meta(entry), entry.oms_order_id)
    original = persistence.commit_execution
    async def unavailable(*args, **kwargs):
        raise RuntimeError("database unavailable before commit")
    persistence.commit_execution = unavailable
    with pytest.raises(RuntimeError, match="before commit"):
        await first._apply_fill(entry, 4, fill_price=100, cumulative_qty=4)
    assert first._pending_execution_events
    assert await persistence.pool.fetchval("SELECT COUNT(*) FROM fills") == 0
    persistence.commit_execution = original
    # New core has no process-local pending observations. The persisted order and
    # broker's cumulative quantity/VWAP recover aggregate economics, not lost
    # execution fragment times or broker execution IDs.
    api = SimpleNamespace(get_orders=MagicMock(), get_execution_orders=MagicMock())
    api.get_orders.return_value = pd.DataFrame()
    api.get_execution_orders.return_value = pd.DataFrame([{
        "ord_dt": "20260105", "odno": entry.order_id, "pdno": "005930", "sll_buy_dvsn_cd": "02",
        "ord_qty": "10", "tot_ccld_qty": "10", "tot_ccld_amt": "1060", "ord_unpr": "999",
        "ord_tmd": "090000", "cncl_yn": "N"}])
    restarted = OMSCore(api, persistence=persistence)
    await restarted._load_persisted_state()
    await restarted._sync_working_orders()
    await restarted._sync_working_orders()
    allocation = restarted.state.get_position(entry.symbol).allocations["KALCB"]
    assert allocation.qty == 10 and allocation.cost_basis == 106
    events = [e for e in await persistence.read_execution_feedback("KALCB") if e["kind"] == "fill"]
    assert len(events) == 1 and events[0]["qty"] == 10
    assert events[0]["metadata"]["execution_identity_basis"] == "cumulative_delta"
    assert events[0]["metadata"]["commission"] is None
    assert events[0]["metadata"]["fill_price_observed"] is True
    assert events[0]["metadata"]["timestamp_provenance"] == "oms_observation"
    assert events[0]["metadata"]["order_date"] == "2026-01-05"
    assert await persistence.pool.fetchval("SELECT qty FROM protective_stops") == 10


@pytest.mark.asyncio
async def test_legacy_open_trade_requires_matching_entry_identity(persistence):
    entry = await order(persistence)
    core = OMSCore(MagicMock(), persistence=persistence)
    legacy_id = uuid.uuid4()
    await persistence.pool.execute("""INSERT INTO trades(trade_id,oms_id,strategy_id,symbol,direction,
        entry_qty,entry_price,entry_ts,status) VALUES($1,$2,'KALCB','005930','LONG',4,100,NOW(),'OPEN')""",
        legacy_id, persistence.oms_id)
    with pytest.raises(ValueError, match="reconciled trade identity"):
        await core._apply_fill(entry, 6, fill_price=110, cumulative_qty=6)
    assert await persistence.pool.fetchval("SELECT count(*) FROM fills") == 0
    assert await persistence.pool.fetchval("SELECT count(*) FROM trades") == 1


@pytest.mark.asyncio
async def test_ledger_generation_survives_restart_and_rejects_rebuild(persistence):
    from types import SimpleNamespace
    from deployment.olr_kalcb.execution_feedback import _validate_stream
    first = await persistence.execution_snapshot("KALCB")
    restarted = OMSPersistence(dsn=persistence.dsn, oms_id=persistence.oms_id)
    restarted.pool = persistence.pool
    assert (await restarted.execution_snapshot("KALCB"))["stream_id"] == first["stream_id"]
    # An explicit ledger rebuild rotates identity even if its sequence advances.
    await persistence.pool.execute("UPDATE execution_generations SET generation=$1 WHERE oms_id=$2", uuid.uuid4(), persistence.oms_id)
    second = await restarted.execution_snapshot("KALCB")
    second["watermark"] = 10
    with pytest.raises(ValueError, match="changed"):
        _validate_stream(SimpleNamespace(meta={"oms_feedback_stream": first["stream_id"]}), 5, second)


@pytest.mark.asyncio
async def test_http_postgres_runtime_checkpoint_and_partial_exit_recovery(persistence, tmp_path, monkeypatch):
    import asyncio
    import socket
    from types import SimpleNamespace
    import uvicorn
    from oms import server
    from oms_client.client import OMSClient
    from deployment.olr_kalcb import execution_feedback as feedback
    from strategy_kalcb.engine import KALCBEngine
    from strategy_kalcb.config import KALCBConfig
    from tests.deployment.olr_kalcb.test_session_driver import _driver
    from instrumentation.src.trade_logger import TradeLogger, TradeEvent

    core = OMSCore(MagicMock(), persistence=persistence)
    core.state.equity = core.state.buyable_cash = 1_000_000
    monkeypatch.setattr(server, "_oms", core)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    http = uvicorn.Server(uvicorn.Config(server.app, lifespan="off", log_level="error"))
    task = asyncio.create_task(http.serve(sockets=[listener]))
    client = OMSClient(f"http://127.0.0.1:{listener.getsockname()[1]}")
    try:
        for _ in range(100):
            if http.started:
                break
            await asyncio.sleep(.02)
        assert http.started
        def driver():
            result = _driver(tmp_path, KALCBEngine(KALCBConfig(exit_hard_stop_enabled=False)), mode="paper", oms=client)
            result.action_router.dry_run = False
            return result
        async def drain(current):
            return await feedback.drain_execution_feedback(SimpleNamespace(mode="paper", action_router=current.action_router, drivers={"KALCB": current}))
        entry = await order(persistence)
        await core._apply_fill(entry, 4, fill_price=100, cumulative_qty=4)
        first = driver()
        assert await drain(first) == 1
        assert first.descriptor.engine.state.symbol_state("005930").position.qty_open == 4
        await core._apply_fill(entry, 6, fill_price=110, cumulative_qty=10)
        original_save = feedback.save_checkpoint
        def interrupted(_):
            raise OSError("checkpoint interrupted")
        monkeypatch.setattr(feedback, "save_checkpoint", interrupted)
        with pytest.raises(OSError, match="checkpoint"):
            await drain(first)
        monkeypatch.setattr(feedback, "save_checkpoint", original_save)
        restored = driver()
        assert await drain(restored) == 1
        position = restored.descriptor.engine.state.symbol_state("005930").position
        assert position.qty_open == 10 and position.entry_price == 106
        assert await drain(driver()) == 0
        # A new OMS object restores canonical quantities and the same stream.
        stream = (await client.get_execution_snapshot("KALCB"))["stream_id"]
        core = OMSCore(MagicMock(), persistence=persistence)
        await core._load_persisted_state()
        core.state.equity = core.state.buyable_cash = 1_000_000
        monkeypatch.setattr(server, "_oms", core)
        assert (await client.get_execution_snapshot("KALCB"))["stream_id"] == stream
        sell = await order(persistence, "exit", "SELL")
        await core._apply_fill(sell, 6, fill_price=120, cumulative_qty=6)
        assert await drain(restored) == 1
        assert position.qty_open == 4
        assert await persistence.pool.fetchval("SELECT qty FROM protective_stops") == 4
        fill = (await client.get_execution_snapshot("KALCB"))["events"][-1]
        logger = TradeLogger({"data_dir": str(tmp_path/"telemetry")}, None)
        logger._open_trades["entry"] = TradeEvent("entry", {}, pair="005930", entry_price=106, position_size=10)
        event = logger.log_exit("entry", fill["price"], "actual_exit", exited_qty=fill["qty"],
            execution_id=fill["metadata"]["execution_id"], consumer_id=stream, consumer_sequence=fill["sequence"])
        assert event.gross_pnl == 84 and event.remaining_qty == 4 and event.net_pnl is None
        reopened = TradeLogger({"data_dir": str(tmp_path/"telemetry")}, None)
        assert reopened.get_open_trades()["entry"].position_size == 4
        assert reopened.log_exit("entry", fill["price"], "actual_exit", exited_qty=fill["qty"], execution_id=fill["metadata"]["execution_id"]) is None
        from oms.execution_preflight import execution_preflight
        preflight = await execution_preflight(persistence, {"005930": 4})
        assert preflight["ledger_passed"] and not preflight["passed"]
        assert not (await execution_preflight(persistence, {"005930": 10}))["passed"]
        await core._apply_fill(sell, 4, fill_price=121, cumulative_qty=10)
        assert await drain(restored) == 1
        assert restored.descriptor.engine.state.symbol_state("005930").position is None
    finally:
        await client.close()
        http.should_exit = True
        await asyncio.wait_for(task, 10)
        listener.close()
