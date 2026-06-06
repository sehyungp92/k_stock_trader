from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from oms.persistence import OMSPersistence
from oms.stop_protection import (
    PriceObservation,
    ProtectiveStop,
    StopProtectionMode,
    StopStatus,
    TriggerPriceSource,
    deterministic_stop_id,
    evaluate_stop_trigger,
)


def test_long_stop_trigger_helper_matches_live_and_bar_low_semantics():
    now = 1_780_000_000.0

    live = evaluate_stop_trigger(
        stop_price=95.0,
        side="LONG",
        observation=PriceObservation("005930", price=94.5, timestamp=now, source=TriggerPriceSource.LAST.value),
        stale_after_sec=30.0,
        now=now,
    )
    bar_low = evaluate_stop_trigger(
        stop_price=95.0,
        side="LONG",
        observation=PriceObservation("005930", price=94.0, timestamp=now, source=TriggerPriceSource.BAR_LOW.value),
        stale_after_sec=30.0,
        now=now,
    )
    stale = evaluate_stop_trigger(
        stop_price=95.0,
        side="LONG",
        observation=PriceObservation("005930", price=90.0, timestamp=now - 120.0),
        stale_after_sec=30.0,
        now=now,
    )

    assert live.triggered is True
    assert bar_low.triggered is True
    assert stale.triggered is False
    assert stale.stale is True
    assert stale.reason == "stale_price"

    halted = evaluate_stop_trigger(
        stop_price=95.0,
        side="LONG",
        observation=PriceObservation("005930", price=94.0, timestamp=now, executable=False),
        stale_after_sec=30.0,
        now=now,
    )

    assert halted.triggered is True
    assert halted.degraded is True
    assert halted.reason == "long_stop_breached_not_executable"


def test_protective_stop_uses_deterministic_allocation_id():
    first = deterministic_stop_id("primary", "KALCB", "005930")
    second = ProtectiveStop.for_allocation(
        oms_id="primary",
        strategy_id="kalcb",
        symbol="5930",
        qty=10,
        stop_price=90.0,
    )

    assert second.stop_id == first
    assert second.strategy_id == "KALCB"
    assert second.symbol == "005930"


def test_same_symbol_multi_strategy_stops_have_distinct_ids():
    kalcb = deterministic_stop_id("primary", "KALCB", "005930")
    olr = deterministic_stop_id("primary", "OLR", "005930")

    assert kalcb != olr


@pytest.mark.asyncio
async def test_persistence_upserts_scoped_protective_stop_row():
    persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
    persistence.pool = MagicMock()
    persistence.pool.fetchrow = AsyncMock(
        return_value={
            "stop_id": uuid.uuid4(),
            "oms_id": "primary",
            "strategy_id": "KALCB",
            "symbol": "005930",
            "side": "LONG",
            "qty": 10,
            "stop_price": 95.0,
            "trigger_price_source": "LAST",
            "protection_mode": "OMS_WATCHER",
            "status": "ACTIVE",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "failure_count": 0,
            "source_metadata": {},
        }
    )

    stop = ProtectiveStop.for_allocation(
        oms_id="primary",
        strategy_id="KALCB",
        symbol="005930",
        qty=10,
        stop_price=95.0,
        protection_mode=StopProtectionMode.OMS_WATCHER.value,
        status=StopStatus.ACTIVE.value,
        source_metadata={"source": "unit"},
    )

    result = await persistence.upsert_stop(stop)

    sql = persistence.pool.fetchrow.await_args.args[0]
    assert result is not None
    assert "INSERT INTO protective_stops" in sql
    assert "ON CONFLICT (stop_id) DO UPDATE" in sql
    assert "protective_stops.status IN ('TRIGGERED', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED', 'FILLED', 'CANCELLED', 'FAILED')" in sql
    assert persistence.pool.fetchrow.await_args.args[2] == "primary"
    assert persistence.pool.fetchrow.await_args.args[3] == "KALCB"


@pytest.mark.asyncio
async def test_mark_triggered_is_compare_and_set_for_exactly_once_exit():
    persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
    persistence.pool = MagicMock()
    persistence.pool.fetchrow = AsyncMock(return_value=None)

    triggered = await persistence.mark_triggered(str(uuid.uuid4()), 94.0, datetime.now(timezone.utc))

    assert triggered is False
    sql = persistence.pool.fetchrow.await_args.args[0]
    assert "SET status = 'TRIGGERED_PENDING_EXECUTION'" in sql
    assert "status IN ('PENDING', 'ACTIVE')" in sql


@pytest.mark.asyncio
async def test_update_stop_quantity_preserves_triggered_and_exit_submitted_status():
    persistence = OMSPersistence(dsn="postgres://test", oms_id="primary")
    persistence.pool = MagicMock()
    persistence.pool.fetchrow = AsyncMock(
        return_value={
            "stop_id": uuid.uuid4(),
            "oms_id": "primary",
            "strategy_id": "KALCB",
            "symbol": "005930",
            "side": "LONG",
            "qty": 7,
            "stop_price": 95.0,
            "trigger_price_source": "LAST",
            "protection_mode": "OMS_WATCHER",
            "status": "EXIT_SUBMITTED",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "failure_count": 0,
            "source_metadata": {},
        }
    )

    stop = await persistence.update_stop_quantity("KALCB", "005930", 7)

    sql = persistence.pool.fetchrow.await_args.args[0]
    assert stop is not None
    assert stop.status == StopStatus.EXIT_SUBMITTED.value
    assert "WHEN status IN ('TRIGGERED', 'TRIGGERED_PENDING_EXECUTION', 'EXIT_SUBMITTED')" in sql
    assert "THEN status" in sql
    assert "WHEN $4 <= 0 THEN 'CANCELLED'" in sql
