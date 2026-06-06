from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from oms.intent import IntentResult, IntentStatus
from oms.stop_protection import PriceObservation, ProtectiveStop, StopStatus
from oms.stop_watcher import StopWatcher


class FakeStopStore:
    def __init__(self, stop: ProtectiveStop):
        self.stop = stop
        self.trigger_calls = 0
        self.exit_submitted = 0
        self.failed_calls = 0
        self.touches = []

    async def load_active_stops(self):
        if self.stop.status in {"PENDING", "ACTIVE", "TRIGGERED_PENDING_EXECUTION"}:
            return [self.stop]
        return []

    async def touch_stop_check(self, stop_id, *, checked_at, last_price, last_error=None):
        self.touches.append((stop_id, checked_at, last_price, last_error))

    async def mark_triggered(self, stop_id, trigger_price, triggered_at):
        self.trigger_calls += 1
        if self.stop.status != StopStatus.ACTIVE.value:
            return False
        self.stop.status = StopStatus.TRIGGERED_PENDING_EXECUTION.value
        self.stop.triggered_at = triggered_at
        self.stop.last_price = trigger_price
        return True

    async def mark_exit_submitted(self, stop_id, exit_intent_id, order_id, idempotency_key=None):
        self.exit_submitted += 1
        self.stop.status = StopStatus.EXIT_SUBMITTED.value
        self.stop.exit_intent_id = exit_intent_id
        self.stop.broker_order_id = order_id

    async def mark_failed(self, stop_id, reason):
        self.failed_calls += 1
        self.stop.status = StopStatus.FAILED.value
        self.stop.last_error = reason


@pytest.mark.asyncio
async def test_stop_watcher_breached_price_submits_exit_once():
    stop = ProtectiveStop.for_allocation(
        oms_id="primary",
        strategy_id="KALCB",
        symbol="005930",
        qty=10,
        stop_price=95.0,
        status=StopStatus.ACTIVE.value,
    )
    store = FakeStopStore(stop)
    submitted = []

    async def price_source(symbol):
        return PriceObservation(symbol, price=94.0, timestamp=1_780_000_000.0)

    async def submit_exit(stop, observation):
        submitted.append((stop.stop_id, observation.price))
        return SimpleNamespace(intent_id="exit-intent", order_id="ORD-STOP")
    triggered = []

    async def trigger_notifier(stop, observation):
        triggered.append((stop.stop_id, observation.price))

    watcher = StopWatcher(
        store=store,
        price_source=price_source,
        exit_submitter=submit_exit,
        trigger_notifier=trigger_notifier,
        stale_after_sec=30.0,
    )

    first = await watcher.check_once(now=1_780_000_000.0)
    second = await watcher.check_once(now=1_780_000_001.0)

    assert len(first) == 1
    assert first[0].decision.triggered is True
    assert second == []
    assert submitted == [(stop.stop_id, 94.0)]
    assert triggered == [(stop.stop_id, 94.0)]
    assert store.trigger_calls == 1
    assert store.exit_submitted == 1
    assert watcher.health.status == "ok"


@pytest.mark.asyncio
async def test_stop_watcher_stale_price_degrades_without_triggering():
    stop = ProtectiveStop.for_allocation(
        oms_id="primary",
        strategy_id="KALCB",
        symbol="005930",
        qty=10,
        stop_price=95.0,
        status=StopStatus.ACTIVE.value,
    )
    store = FakeStopStore(stop)

    async def price_source(symbol):
        return PriceObservation(symbol, price=90.0, timestamp=1_780_000_000.0)

    async def submit_exit(stop, observation):
        raise AssertionError("stale prices must not trigger stop exits")

    watcher = StopWatcher(store=store, price_source=price_source, exit_submitter=submit_exit, stale_after_sec=10.0)

    results = await watcher.check_once(now=1_780_000_100.0)

    assert results[0].decision.triggered is False
    assert results[0].decision.stale is True
    assert store.trigger_calls == 0
    assert watcher.health.status == "degraded"


@pytest.mark.asyncio
async def test_stop_watcher_raw_float_price_is_unverified_and_stale():
    stop = ProtectiveStop.for_allocation(
        oms_id="primary",
        strategy_id="KALCB",
        symbol="005930",
        qty=10,
        stop_price=95.0,
        status=StopStatus.ACTIVE.value,
    )
    store = FakeStopStore(stop)

    async def submit_exit(stop, observation):
        raise AssertionError("float-only price observations must not trigger stop exits")

    watcher = StopWatcher(store=store, price_source=lambda symbol: 94.0, exit_submitter=submit_exit, stale_after_sec=30.0)

    results = await watcher.check_once(now=1_780_000_000.0)

    assert results[0].observation.price == 94.0
    assert results[0].observation.timestamp == 0.0
    assert results[0].observation.source == "UNVERIFIED_LAST"
    assert results[0].observation.market_open is False
    assert results[0].observation.executable is False
    assert results[0].decision.triggered is False
    assert results[0].decision.stale is True
    assert store.trigger_calls == 0
    assert store.exit_submitted == 0
    assert watcher.health.status == "degraded"


@pytest.mark.asyncio
async def test_stop_watcher_marks_breached_non_executable_stop_pending_without_exit_submit():
    stop = ProtectiveStop.for_allocation(
        oms_id="primary",
        strategy_id="KALCB",
        symbol="005930",
        qty=10,
        stop_price=95.0,
        status=StopStatus.ACTIVE.value,
    )
    store = FakeStopStore(stop)

    async def price_source(symbol):
        return PriceObservation(symbol, price=94.0, timestamp=1_780_000_000.0, executable=False)

    async def submit_exit(stop, observation):
        raise AssertionError("non-executable breached quotes must wait for retry")

    watcher = StopWatcher(store=store, price_source=price_source, exit_submitter=submit_exit, stale_after_sec=30.0)

    results = await watcher.check_once(now=1_780_000_000.0)

    assert results[0].decision.triggered is True
    assert results[0].decision.degraded is True
    assert stop.status == StopStatus.TRIGGERED_PENDING_EXECUTION.value
    assert store.trigger_calls == 1
    assert store.exit_submitted == 0
    assert watcher.health.status == "error"
    assert any(item[3] == "long_stop_breached_not_executable" for item in store.touches)


@pytest.mark.asyncio
async def test_stop_watcher_retries_triggered_pending_execution_until_order_identity_exists():
    stop = ProtectiveStop.for_allocation(
        oms_id="primary",
        strategy_id="KALCB",
        symbol="005930",
        qty=10,
        stop_price=95.0,
        status=StopStatus.ACTIVE.value,
    )
    store = FakeStopStore(stop)
    submissions = []

    async def price_source(symbol):
        return PriceObservation(symbol, price=94.0, timestamp=1_780_000_000.0)

    async def submit_exit(stop, observation):
        submissions.append(stop.status)
        if len(submissions) == 1:
            return IntentResult(intent_id="exit-intent", status=IntentStatus.DEFERRED, message="broker temporarily unavailable")
        return SimpleNamespace(intent_id="exit-intent", order_id="ORD-STOP")

    watcher = StopWatcher(
        store=store,
        price_source=price_source,
        exit_submitter=submit_exit,
        stale_after_sec=30.0,
    )

    first = await watcher.check_once(now=1_780_000_000.0)

    assert watcher.health.status == "error"
    assert stop.status == StopStatus.TRIGGERED_PENDING_EXECUTION.value
    assert store.exit_submitted == 0

    second = await watcher.check_once(now=1_780_000_001.0)

    assert len(first) == 1
    assert first[0].exit_intent_id == "exit-intent"
    assert first[0].order_id is None
    assert stop.status == StopStatus.EXIT_SUBMITTED.value
    assert second[0].exit_intent_id == "exit-intent"
    assert submissions == [
        StopStatus.TRIGGERED_PENDING_EXECUTION.value,
        StopStatus.TRIGGERED_PENDING_EXECUTION.value,
    ]
    assert store.trigger_calls == 1
    assert store.exit_submitted == 1
    assert any(item[3] == "exit_not_submitted:DEFERRED" for item in store.touches)


@pytest.mark.asyncio
async def test_stop_watcher_transient_exception_keeps_stop_loadable_and_retries():
    stop = ProtectiveStop.for_allocation(
        oms_id="primary",
        strategy_id="KALCB",
        symbol="005930",
        qty=10,
        stop_price=95.0,
        status=StopStatus.ACTIVE.value,
    )
    store = FakeStopStore(stop)
    observations = 0
    submitted = []

    async def price_source(symbol):
        nonlocal observations
        observations += 1
        if observations == 1:
            raise RuntimeError("quote source temporarily unavailable")
        return PriceObservation(symbol, price=94.0, timestamp=1_780_000_001.0)

    async def submit_exit(stop, observation):
        submitted.append((stop.stop_id, observation.price))
        return SimpleNamespace(intent_id="exit-intent", order_id="ORD-STOP")

    watcher = StopWatcher(
        store=store,
        price_source=price_source,
        exit_submitter=submit_exit,
        stale_after_sec=30.0,
    )

    first = await watcher.check_once(now=1_780_000_000.0)

    assert first == []
    assert stop.status == StopStatus.ACTIVE.value
    assert store.failed_calls == 0
    assert watcher.health.status == "error"
    assert any(item[3] == "quote source temporarily unavailable" for item in store.touches)

    second = await watcher.check_once(now=1_780_000_001.0)

    assert len(second) == 1
    assert second[0].decision.triggered is True
    assert submitted == [(stop.stop_id, 94.0)]
    assert stop.status == StopStatus.EXIT_SUBMITTED.value
    assert store.failed_calls == 0
