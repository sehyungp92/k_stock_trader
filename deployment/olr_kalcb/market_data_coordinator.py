from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from kis_core.bar_aggregator import BarAggregator
from strategy_common.market import MarketBar
from strategy_kalcb.data import WebSocketRegistrationBudget

from .kis_resource_plan import KISResourcePlan, ResourceLeaseWindow, target_strategy_ids_for_bar
from .session_capture import PaperSessionRecorder


SUBSCRIPTION_EVENTS_FILE = "subscription_events.jsonl"
KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True, slots=True)
class SubscriptionEvent:
    record_type: str
    event_time: str
    strategy_id: str
    lease_name: str
    symbol: str
    action: str
    registration_type: str
    ws_used_before: int
    ws_used_after: int
    ws_budget: int
    kis_resource_plan_hash: str
    market_data_source: str
    reason_code: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActiveLease:
    strategy_id: str
    lease_name: str
    symbol: str
    registration_type: str


class KISMarketDataCoordinator:
    """Deployment-owned KIS market-data lease, subscription, and routing coordinator."""

    def __init__(
        self,
        *,
        resource_plan: KISResourcePlan,
        recorder: PaperSessionRecorder | None = None,
        websocket_client: Any | None = None,
        subscription_manager: Any | None = None,
        registration_budgets: Mapping[str, WebSocketRegistrationBudget] | None = None,
        ledger_path: str | Path | None = None,
        market_data_source: str | None = None,
    ) -> None:
        self.resource_plan = resource_plan
        self.recorder = recorder
        self.websocket_client = websocket_client
        self.subscription_manager = subscription_manager or _subscription_manager_for(websocket_client, resource_plan)
        self.market_data_source = _normalize_market_data_source(
            market_data_source or ("kis_websocket" if self.subscription_manager is not None else "external_completed_bars")
        )
        self.registration_budgets = {str(key).upper().strip(): value for key, value in dict(registration_budgets or {}).items()}
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self._active_leases: dict[tuple[str, str, str, str], ActiveLease] = {}
        self._subscription_rows: list[dict[str, Any]] = []
        self._external_declared_windows: set[str] = set()

    @property
    def active_leases(self) -> tuple[ActiveLease, ...]:
        return tuple(self._active_leases.values())

    def sync_runtime_plan(self, runtime_plan: Any | None) -> bool:
        current = getattr(runtime_plan, "kis_resource_plan", None)
        if not isinstance(current, KISResourcePlan):
            return False
        if current.plan_hash == self.resource_plan.plan_hash:
            return False
        self.resource_plan = current
        return True

    async def activate_window(self, name: str, *, symbols: Sequence[str] | None = None) -> tuple[dict[str, Any], ...]:
        window = self._window(name)
        desired_symbols = _normalize_symbols(symbols if symbols is not None else self._window_ws_symbols(window, {}))
        symbol_budget = max(len(desired_symbols) * max(window.ws_regs_per_symbol, 1), 1)
        rows: list[dict[str, Any]] = []
        for symbol in desired_symbols:
            for registration_type in _registration_types(window.ws_regs_per_symbol):
                row = await self._activate_ws_symbol(
                    window,
                    symbol,
                    registration_type=registration_type,
                    symbol_budget=symbol_budget,
                )
                if row is not None:
                    rows.append(row)
        return tuple(rows)

    async def release_window(self, name: str) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for key, lease in list(self._active_leases.items()):
            if lease.lease_name != name:
                continue
            rows.append(await self._release_lease(key, lease, reason_code="lease_released"))
        return tuple(rows)

    async def release_all(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for window_name in tuple(dict.fromkeys(lease.lease_name for lease in self._active_leases.values())):
            rows.extend(await self.release_window(window_name))
        return tuple(rows)

    async def route_completed_bar(
        self,
        runtime_plan: Any,
        bar: MarketBar,
        *,
        held_or_pending_symbols: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[Any, ...]:
        if not bar.is_completed:
            raise ValueError(f"incomplete bar cannot be routed by market-data coordinator: {bar.symbol} {bar.timestamp}")
        resource_plan = self._current_resource_plan(runtime_plan)
        held_or_pending = held_or_pending_symbols
        if held_or_pending is None:
            held_or_pending = _held_or_pending_symbols_from_runtime_plan(runtime_plan)
        await self.activate_due_windows(bar.timestamp, runtime_plan=runtime_plan, held_or_pending_symbols=held_or_pending)
        if self.recorder is not None:
            self.recorder.record_market_bar(bar)
        targets = target_strategy_ids_for_bar(
            resource_plan,
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            available_strategy_ids=tuple(getattr(runtime_plan, "drivers", {}) or ()),
            held_or_pending_symbols=held_or_pending,
        )
        if not targets:
            self.record_subscription_event(
                strategy_id="NONE",
                lease_name="market_data_route",
                symbol=bar.symbol,
                action="suppressed",
                registration_type="completed_bar",
                ws_used_before=0,
                ws_used_after=0,
                ws_budget=self.resource_plan.limit_profile.ws_max_registrations,
                reason_code="no_resource_plan_target",
            )
            return ()
        return await runtime_plan.handle_bar(bar, target_strategy_ids=targets)

    async def activate_due_windows(
        self,
        timestamp: datetime,
        *,
        runtime_plan: Any | None = None,
        held_or_pending_symbols: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        resource_plan = self._current_resource_plan(runtime_plan)
        current = _hhmm(timestamp)
        rows: list[dict[str, Any]] = []
        held_or_pending = _normalize_held_or_pending_symbols(
            held_or_pending_symbols
            if held_or_pending_symbols is not None
            else _held_or_pending_symbols_from_runtime_plan(runtime_plan)
        )
        active_symbols_by_window = {
            window.name: symbols
            for window in resource_plan.lease_windows
            if _time_in_window(current, window.starts_at_kst, window.ends_at_kst)
            for symbols in (self._window_ws_symbols(window, held_or_pending),)
            if symbols
        }
        active_names = set(active_symbols_by_window)
        if self.market_data_source != "kis_websocket":
            for name in active_symbols_by_window:
                if name in self._external_declared_windows:
                    continue
                window = self._window(name)
                rows.append(
                    self.record_subscription_event(
                        strategy_id=window.strategy_id,
                        lease_name=name,
                        symbol="000000",
                        action="external_source_declared",
                        registration_type="completed_bar_source",
                        ws_used_before=0,
                        ws_used_after=0,
                        ws_budget=self.resource_plan.limit_profile.ws_max_registrations,
                        reason_code=f"{self.market_data_source}_no_kis_subscription",
                    )
                )
                self._external_declared_windows.add(name)
            return tuple(rows)
        for name, symbols in active_symbols_by_window.items():
            rows.extend(await self.activate_window(name, symbols=symbols))
        for key, lease in list(self._active_leases.items()):
            desired_symbols = set(active_symbols_by_window.get(lease.lease_name, ()))
            if lease.lease_name not in active_names or lease.symbol not in desired_symbols:
                rows.append(await self._release_lease(key, lease, reason_code="lease_reconciled"))
        return tuple(rows)

    def record_subscription_event(
        self,
        *,
        strategy_id: str,
        lease_name: str,
        symbol: str,
        action: str,
        registration_type: str = "tick",
        ws_used_before: int = 0,
        ws_used_after: int = 0,
        ws_budget: int | None = None,
        reason_code: str = "",
    ) -> dict[str, Any]:
        event = SubscriptionEvent(
            record_type="subscription_event",
            event_time=datetime.now(timezone.utc).isoformat(),
            strategy_id=str(strategy_id or "").upper().strip(),
            lease_name=str(lease_name or ""),
            symbol=str(symbol or "").zfill(6),
            action=str(action or ""),
            registration_type=str(registration_type or ""),
            ws_used_before=int(ws_used_before),
            ws_used_after=int(ws_used_after),
            ws_budget=int(ws_budget if ws_budget is not None else self.resource_plan.limit_profile.ws_max_registrations),
            kis_resource_plan_hash=self.resource_plan.plan_hash,
            market_data_source=self.market_data_source,
            reason_code=str(reason_code or ""),
        ).to_json_dict()
        self._subscription_rows.append(event)
        if self.recorder is not None:
            self.recorder.append_jsonl(SUBSCRIPTION_EVENTS_FILE, event)
        return event

    def write_subscription_events(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in self._subscription_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return target

    def _window(self, name: str) -> ResourceLeaseWindow:
        for window in self.resource_plan.lease_windows:
            if window.name == name:
                return window
        raise KeyError(f"unknown KIS resource lease window {name!r}")

    def _current_resource_plan(self, runtime_plan: Any | None = None) -> KISResourcePlan:
        self.sync_runtime_plan(runtime_plan)
        return self.resource_plan

    async def _activate_ws_symbol(
        self,
        window: ResourceLeaseWindow,
        symbol: str,
        *,
        registration_type: str,
        symbol_budget: int,
    ) -> dict[str, Any] | None:
        sid = str(window.strategy_id or "").upper().strip()
        normalized = str(symbol or "").zfill(6)
        key = (sid, window.name, normalized, registration_type)
        if key in self._active_leases:
            return None
        before = self._used_regs(sid)
        shared_subscription = self._has_other_lease(key, symbol=normalized, registration_type=registration_type)
        if shared_subscription:
            ok = await self._ensure_subscription(normalized, registration_type)
            reason = "shared_subscription_reused" if ok else "subscription_manager_rejected"
        else:
            budget = self._budget_for(sid, strategy_symbol_budget=symbol_budget)
            ok, reason = budget.allocate_hot(_budget_symbol(normalized, registration_type), reason=window.name)
            if ok:
                ok = await self._ensure_subscription(normalized, registration_type)
                if not ok:
                    budget.release_hot(_budget_symbol(normalized, registration_type))
                    reason = "subscription_manager_rejected"
        if ok:
            self._active_leases[key] = ActiveLease(sid, window.name, normalized, registration_type)
        return self.record_subscription_event(
            strategy_id=sid,
            lease_name=window.name,
            symbol=normalized,
            action="subscribe" if ok else "rejected",
            registration_type=registration_type,
            ws_used_before=before,
            ws_used_after=self._used_regs(sid),
            ws_budget=self.resource_plan.limit_profile.ws_max_registrations,
            reason_code=reason,
        )

    async def _release_lease(self, key: tuple[str, str, str, str], lease: ActiveLease, *, reason_code: str) -> dict[str, Any]:
        before = self._used_regs(lease.strategy_id)
        if not self._has_other_lease(key, symbol=lease.symbol, registration_type=lease.registration_type):
            await self._drop_subscription(lease.symbol, lease.registration_type)
        if not self._has_other_lease(
            key,
            strategy_id=lease.strategy_id,
            symbol=lease.symbol,
            registration_type=lease.registration_type,
        ):
            self._budget_for(lease.strategy_id, strategy_symbol_budget=0).release_hot(
                _budget_symbol(lease.symbol, lease.registration_type)
            )
        self._active_leases.pop(key, None)
        return self.record_subscription_event(
            strategy_id=lease.strategy_id,
            lease_name=lease.lease_name,
            symbol=lease.symbol,
            action="unsubscribe",
            registration_type=lease.registration_type,
            ws_used_before=before,
            ws_used_after=self._used_regs(lease.strategy_id),
            ws_budget=self.resource_plan.limit_profile.ws_max_registrations,
            reason_code=reason_code,
        )

    def _has_other_lease(
        self,
        current_key: tuple[str, str, str, str],
        *,
        symbol: str,
        registration_type: str,
        strategy_id: str | None = None,
    ) -> bool:
        sid = str(strategy_id or "").upper().strip()
        for key, lease in self._active_leases.items():
            if key == current_key:
                continue
            if lease.symbol != symbol or lease.registration_type != registration_type:
                continue
            if sid and lease.strategy_id != sid:
                continue
            return True
        return False

    def _window_ws_symbols(
        self,
        window: ResourceLeaseWindow,
        held_or_pending_symbols: Mapping[str, Sequence[str]],
    ) -> tuple[str, ...]:
        sid = str(window.strategy_id or "").upper().strip()
        symbols: list[str] = list(window.ws_symbols)
        if sid == "KALCB" and (
            window.name == "kalcb_position_management" or window.rest_endpoint_class == "held_position_dynamic"
        ):
            symbols.extend(held_or_pending_symbols.get("KALCB", ()))
        elif sid == "OLR" and window.name == "olr_final_runtime":
            metadata = dict(window.metadata or {})
            symbols.extend(_coerce_symbol_sequence(metadata.get("orderable_symbols")))
            symbols.extend(held_or_pending_symbols.get("OLR", ()))
        return _normalize_symbols(symbols)

    def _budget_for(self, strategy_id: str, *, strategy_symbol_budget: int) -> WebSocketRegistrationBudget:
        sid = str(strategy_id or "").upper().strip()
        budget = self.registration_budgets.get(sid)
        if budget is None:
            budget = WebSocketRegistrationBudget(
                max_registrations=self.resource_plan.limit_profile.ws_max_registrations,
                reserved_execution_regs=self.resource_plan.limit_profile.ws_reserved_execution_regs,
                strategy_symbol_budget=max(strategy_symbol_budget, 1),
                ledger_path=self.ledger_path,
                strategy_id=sid,
            )
            self.registration_budgets[sid] = budget
        else:
            budget.strategy_symbol_budget = max(budget.strategy_symbol_budget, max(strategy_symbol_budget, 1))
        return budget

    def _used_regs(self, strategy_id: str) -> int:
        budget = self.registration_budgets.get(str(strategy_id or "").upper().strip())
        budget_used = budget.used_regs if budget is not None else 0
        manager_total = int(self.subscription_manager.total_regs()) if callable(getattr(self.subscription_manager, "total_regs", None)) else 0
        return max(budget_used, manager_total)

    async def _ensure_subscription(self, symbol: str, registration_type: str) -> bool:
        if self.subscription_manager is None:
            return False
        if registration_type == "askbid":
            return bool(await self.subscription_manager.ensure_askbid(symbol))
        return bool(await self.subscription_manager.ensure_tick(symbol))

    async def _drop_subscription(self, symbol: str, registration_type: str) -> None:
        if self.subscription_manager is None:
            return
        if registration_type == "askbid":
            await self.subscription_manager.drop_askbid(symbol)
        else:
            await self.subscription_manager.drop_tick(symbol)


def _subscription_manager_for(websocket_client: Any | None, resource_plan: KISResourcePlan) -> Any | None:
    if websocket_client is None:
        return None
    from kis_core.ws_client import BaseSubscriptionManager

    return BaseSubscriptionManager(websocket_client, max_regs=resource_plan.limit_profile.ws_max_registrations)


class KISWebSocketCompletedBarSource:
    """Convert KIS tick callbacks into completed 5-minute MarketBar objects."""

    def __init__(
        self,
        websocket_client: Any,
        *,
        timeframe_minutes: int = 5,
        source: str = "kis_websocket",
        source_fingerprint: str = "kis_websocket_tick_stream",
        queue_maxsize: int = 10_000,
    ) -> None:
        self.websocket_client = websocket_client
        self.timeframe_minutes = int(timeframe_minutes)
        self.source = str(source or "kis_websocket")
        self.source_fingerprint = str(source_fingerprint or "kis_websocket_tick_stream")
        self._aggregators: dict[str, BarAggregator] = {}
        self._queue: asyncio.Queue[MarketBar] = asyncio.Queue(maxsize=max(1, int(queue_maxsize)))
        self._run_task: asyncio.Task | None = None
        self._started = False
        self.dropped_bar_count = 0

    async def start(self) -> None:
        if self._started:
            return
        on_tick = getattr(self.websocket_client, "on_tick", None)
        if not callable(on_tick):
            raise RuntimeError("KIS WebSocket completed-bar source requires websocket_client.on_tick")
        on_tick(self._on_tick)
        run = getattr(self.websocket_client, "run", None)
        if not callable(run):
            raise RuntimeError("KIS WebSocket completed-bar source requires websocket_client.run")
        self._run_task = asyncio.create_task(run(auto_reconnect=True))
        self._started = True

    async def stop(self) -> None:
        task = self._run_task
        self._run_task = None
        self._started = False
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def next_bar(self, *, timeout_s: float) -> MarketBar | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=max(float(timeout_s), 0.001))
        except asyncio.TimeoutError:
            return None

    def _on_tick(self, tick: Any) -> None:
        symbol = str(getattr(tick, "ticker", "") or "").zfill(6)
        timestamp = getattr(tick, "timestamp", None)
        price = float(getattr(tick, "price", 0.0) or 0.0)
        volume = float(getattr(tick, "volume", 0.0) or 0.0)
        if not symbol.strip("0") or not isinstance(timestamp, datetime) or price <= 0:
            return
        completed = self._aggregators.setdefault(symbol, BarAggregator(interval_minutes=self.timeframe_minutes)).update_tick(
            timestamp,
            price,
            max(volume, 0.0),
        )
        if completed is None:
            return
        bar = MarketBar(
            symbol=symbol,
            timestamp=completed.timestamp,
            timeframe=f"{self.timeframe_minutes}m",
            open=completed.open,
            high=completed.high,
            low=completed.low,
            close=completed.close,
            volume=completed.volume,
            is_completed=True,
            source=self.source,
            source_fingerprint=self.source_fingerprint,
            metadata={"source_event": "kis_tick_aggregator"},
        )
        try:
            self._queue.put_nowait(bar)
        except asyncio.QueueFull:
            self.dropped_bar_count += 1


def _registration_types(regs_per_symbol: int) -> tuple[str, ...]:
    if int(regs_per_symbol) <= 1:
        return ("tick",)
    return ("tick", "askbid", *(f"extra_{index}" for index in range(3, int(regs_per_symbol) + 1)))


def _budget_symbol(symbol: str, registration_type: str) -> str:
    return f"{str(symbol).zfill(6)}:{registration_type}"


def _normalize_symbols(symbols: Sequence[str] | None) -> tuple[str, ...]:
    result: list[str] = []
    for raw in symbols or ():
        symbol = str(raw or "").zfill(6)
        if symbol.strip("0"):
            result.append(symbol)
    return tuple(dict.fromkeys(result))


def _coerce_symbol_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _normalize_held_or_pending_symbols(symbols: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, ...]]:
    return {
        str(strategy_id or "").upper().strip(): _normalize_symbols(values)
        for strategy_id, values in dict(symbols or {}).items()
    }


def _held_or_pending_symbols_from_runtime_plan(runtime_plan: Any | None) -> dict[str, tuple[str, ...]]:
    drivers = getattr(runtime_plan, "drivers", {}) or {}
    result: dict[str, tuple[str, ...]] = {}
    for strategy_id, driver in dict(drivers).items():
        state = getattr(getattr(getattr(driver, "descriptor", None), "engine", None), "state", None)
        symbols: set[str] = set()
        for symbol, symbol_state in dict(getattr(state, "symbols", {}) or {}).items():
            position = getattr(symbol_state, "position", None)
            if position is not None and int(getattr(position, "qty_open", 1) or 0) > 0:
                symbols.add(str(symbol).zfill(6))
                continue
            for attr in ("pending_entry_order_id", "pending_exit_order_id"):
                if str(getattr(symbol_state, attr, "") or ""):
                    symbols.add(str(symbol).zfill(6))
                    break
        result[str(strategy_id or "").upper().strip()] = tuple(sorted(symbols))
    return result


def _hhmm(timestamp: datetime) -> str:
    ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=KST)
    return ts.astimezone(KST).strftime("%H:%M")


def _time_in_window(current: str, starts_at: str, ends_at: str) -> bool:
    return starts_at <= current < ends_at


def _normalize_market_data_source(source: str) -> str:
    value = str(source or "").strip().lower()
    return value if value in {"kis_websocket", "external_completed_bars"} else "external_completed_bars"
