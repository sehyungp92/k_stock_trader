"""OMS-owned durable protective stop watcher."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Optional

from loguru import logger

from .stop_protection import (
    PriceObservation,
    ProtectiveStop,
    StopProtectionMode,
    StopStatus,
    StopTriggerDecision,
    StopTriggerResult,
    TriggerPriceSource,
    evaluate_stop_trigger,
)


PriceSource = Callable[[str], PriceObservation | float | Awaitable[PriceObservation | float]]
ExitSubmitter = Callable[[ProtectiveStop, PriceObservation], Awaitable[Any]]
TriggerNotifier = Callable[[ProtectiveStop, PriceObservation], Awaitable[Any] | Any]


@dataclass(frozen=True, slots=True)
class StopWatcherHealth:
    status: str = "unknown"
    active_stop_count: int = 0
    triggered_stop_count: int = 0
    last_check_ts: float | None = None
    stale_price_count: int = 0
    last_error: str = ""


class StopWatcher:
    """Poll durable OMS-watcher stops and submit stop exits exactly once."""

    def __init__(
        self,
        *,
        store: Any,
        price_source: PriceSource,
        exit_submitter: ExitSubmitter,
        trigger_notifier: TriggerNotifier | None = None,
        stale_after_sec: float = 30.0,
        interval_sec: float = 5.0,
    ) -> None:
        self.store = store
        self.price_source = price_source
        self.exit_submitter = exit_submitter
        self.trigger_notifier = trigger_notifier
        self.stale_after_sec = float(stale_after_sec or 30.0)
        self.interval_sec = float(interval_sec or 5.0)
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        self.health = StopWatcherHealth()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._loop(), name="oms-stop-watcher")

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def reload(self) -> None:
        await self.check_once()

    async def check_once(self, *, now: float | None = None) -> list[StopTriggerResult]:
        current_time = time.time() if now is None else float(now)
        results: list[StopTriggerResult] = []
        active_count = 0
        triggered_count = 0
        stale_count = 0
        last_error = ""
        try:
            stops = await _maybe_await(self.store.load_active_stops())
        except Exception as exc:
            logger.error(f"Stop watcher could not load active stops: {exc}")
            self.health = StopWatcherHealth(status="error", last_check_ts=current_time, last_error=str(exc))
            return results

        for stop in _only_oms_watcher_stops(stops):
            active_count += 1
            try:
                observation = await self._observe(stop.symbol, now=current_time)
                pending_execution = str(stop.status).upper() == StopStatus.TRIGGERED_PENDING_EXECUTION.value
                decision = self._decision_for_stop(stop, observation, now=current_time)
                if decision.stale:
                    stale_count += 1
                await _maybe_await(
                    self.store.touch_stop_check(
                        stop.stop_id,
                        checked_at=datetime.fromtimestamp(current_time, tz=timezone.utc),
                        last_price=observation.price,
                        last_error=decision.reason if decision.degraded and not decision.triggered else None,
                    )
                )
                result = StopTriggerResult(stop=stop, observation=observation, decision=decision)
                results.append(result)
                if not decision.triggered:
                    continue
                if not pending_execution:
                    triggered_at = datetime.fromtimestamp(current_time, tz=timezone.utc)
                    triggered = await _maybe_await(
                        self.store.mark_triggered(
                            stop.stop_id,
                            observation.price,
                            triggered_at,
                        )
                    )
                    if triggered is False:
                        continue
                    stop.status = StopStatus.TRIGGERED_PENDING_EXECUTION.value
                    stop.triggered_at = triggered_at
                    stop.last_price = observation.price
                    if self.trigger_notifier is not None:
                        await _maybe_await(self.trigger_notifier(stop, observation))
                triggered_count += 1
                if decision.degraded:
                    last_error = decision.reason
                    await _maybe_await(
                        self.store.touch_stop_check(
                            stop.stop_id,
                            checked_at=datetime.fromtimestamp(current_time, tz=timezone.utc),
                            last_price=observation.price,
                            last_error=last_error,
                        )
                    )
                    continue
                exit_result = await self.exit_submitter(stop, observation)
                exit_intent_id = str(getattr(exit_result, "intent_id", "") or "") or None
                order_id = str(getattr(exit_result, "order_id", "") or "") or None
                status_name = str(getattr(getattr(exit_result, "status", None), "name", "") or "").upper()
                accepted_with_order = bool(order_id) and status_name not in {"REJECTED", "DEFERRED", "CANCELLED"}
                if accepted_with_order:
                    await _maybe_await(
                        self.store.mark_exit_submitted(
                            stop.stop_id,
                            exit_intent_id,
                            order_id,
                            getattr(stop, "idempotency_key", None),
                        )
                    )
                else:
                    reason = status_name or getattr(exit_result, "message", None) or "exit_not_submitted"
                    last_error = f"exit_not_submitted:{reason}"
                    await _maybe_await(
                        self.store.touch_stop_check(
                            stop.stop_id,
                            checked_at=datetime.fromtimestamp(current_time, tz=timezone.utc),
                            last_price=observation.price,
                            last_error=last_error,
                        )
                    )
                results[-1] = StopTriggerResult(
                    stop=stop,
                    observation=observation,
                    decision=decision,
                    exit_intent_id=exit_intent_id,
                    order_id=order_id,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.error(f"Stop watcher check failed for {stop.symbol}/{stop.strategy_id}: {exc}")
                try:
                    await _maybe_await(
                        self.store.touch_stop_check(
                            stop.stop_id,
                            checked_at=datetime.fromtimestamp(current_time, tz=timezone.utc),
                            last_price=getattr(stop, "last_price", None),
                            last_error=last_error,
                        )
                    )
                except Exception:
                    pass

        status = "ok"
        if last_error:
            status = "error"
        elif stale_count:
            status = "degraded"
        self.health = StopWatcherHealth(
            status=status,
            active_stop_count=active_count,
            triggered_stop_count=triggered_count,
            last_check_ts=current_time,
            stale_price_count=stale_count,
            last_error=last_error,
        )
        return results

    async def _loop(self) -> None:
        while not self._stopped.is_set():
            await self.check_once()
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=max(self.interval_sec, 0.1))
            except asyncio.TimeoutError:
                pass

    async def _observe(self, symbol: str, *, now: float) -> PriceObservation:
        raw = self.price_source(symbol)
        if inspect.isawaitable(raw):
            raw = await raw
        if isinstance(raw, PriceObservation):
            return raw
        return PriceObservation(
            symbol=str(symbol).zfill(6),
            price=float(raw or 0.0),
            timestamp=0.0,
            source="UNVERIFIED_LAST",
            market_open=False,
            executable=False,
        )

    def _decision_for_stop(
        self,
        stop: ProtectiveStop,
        observation: PriceObservation,
        *,
        now: float,
    ) -> StopTriggerDecision:
        decision = evaluate_stop_trigger(
            stop_price=stop.stop_price,
            side=stop.side,
            observation=observation,
            stale_after_sec=self.stale_after_sec,
            now=now,
        )
        if str(stop.status).upper() != StopStatus.TRIGGERED_PENDING_EXECUTION.value:
            return decision
        if decision.stale or decision.degraded:
            return decision
        return StopTriggerDecision(True, "trigger_retry_pending_execution")


def _only_oms_watcher_stops(stops: Iterable[ProtectiveStop]) -> list[ProtectiveStop]:
    return [
        stop
        for stop in stops
        if str(stop.protection_mode).upper() == StopProtectionMode.OMS_WATCHER.value
        and str(stop.status).upper()
        in {StopStatus.PENDING.value, StopStatus.ACTIVE.value, StopStatus.TRIGGERED_PENDING_EXECUTION.value}
        and int(stop.qty or 0) > 0
    ]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
