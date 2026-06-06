"""OMS HTTP Client for strategies."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger

try:
    import aiohttp
except ImportError:
    aiohttp = None

from oms.intent import Intent, IntentResult, IntentStatus


_READY_HEALTH_STATES = {"ok", "ready", "healthy"}
_ACCEPTABLE_OMS_STATUS = _READY_HEALTH_STATES | {"warn"}
_REQUIRED_STOP_HEALTH_FIELDS = (
    "unprotected_positions_count",
    "active_stop_count",
    "triggered_stop_count",
    "stop_watcher_price_stale_count",
)
_MAX_STOP_WATCHER_AGE_SEC = 60.0


def _health_payload_ready(payload: Dict[str, Any]) -> bool:
    status = str(payload.get("status") or "").lower().strip()
    if not status:
        return False
    if status in {"error", "degraded"}:
        return False
    if status and status not in _ACCEPTABLE_OMS_STATUS:
        return False
    stop_status = str(payload.get("stop_protection_status") or "").lower().strip()
    if not stop_status:
        return False
    if stop_status in {"error", "degraded"}:
        return False
    if stop_status not in _READY_HEALTH_STATES:
        return False
    stop_counts = {
        field: _required_nonnegative_int(payload, field)
        for field in _REQUIRED_STOP_HEALTH_FIELDS
    }
    if any(value is None for value in stop_counts.values()):
        return False
    if stop_counts["unprotected_positions_count"] > 0:
        return False
    if stop_counts["stop_watcher_price_stale_count"] > 0:
        return False
    if stop_counts["active_stop_count"] > 0:
        watcher_age = _required_nonnegative_float(payload, "stop_watcher_last_check_age_sec")
        if watcher_age is None or watcher_age > _MAX_STOP_WATCHER_AGE_SEC:
            return False
    idempotency_status = str(
        payload.get("idempotency_status")
        or payload.get("idempotency_health")
        or payload.get("reservation_reconcile_status")
        or ""
    ).lower().strip()
    if not idempotency_status:
        return False
    if idempotency_status in {"error", "degraded", "ambiguous"}:
        return False
    return idempotency_status in _READY_HEALTH_STATES


def _required_nonnegative_int(payload: Dict[str, Any], field: str) -> Optional[int]:
    if field not in payload or payload.get(field) is None:
        return None
    try:
        value = int(payload[field])
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _required_nonnegative_float(payload: Dict[str, Any], field: str) -> Optional[float]:
    if field not in payload or payload.get(field) is None:
        return None
    try:
        value = float(payload[field])
    except (TypeError, ValueError):
        return None
    return value if value >= 0.0 else None


@dataclass
class AllocationInfo:
    """Per-strategy allocation info."""
    strategy_id: str
    qty: int
    cost_basis: float
    entry_ts: Optional[str] = None  # ISO format datetime string from API
    soft_stop_px: Optional[float] = None
    time_stop_ts: Optional[float] = None


@dataclass
class WorkingOrderInfo:
    """Working order info from OMS."""
    order_id: str
    symbol: str
    side: str
    qty: int
    filled_qty: int = 0
    remaining_qty: int = 0
    price: float = 0.0
    order_type: str = ""
    status: str = ""
    strategy_id: str = ""
    intent_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    submit_ref: Optional[str] = None
    risk_stop_px: Optional[float] = None
    risk_hard_stop_px: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    submit_ts: Optional[float] = None
    cancel_after_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if self.remaining_qty <= 0 and self.qty > self.filled_qty:
            self.remaining_qty = self.qty - self.filled_qty


@dataclass
class PositionInfo:
    """Position info from OMS."""
    symbol: str
    real_qty: int
    avg_price: float
    allocations: Dict[str, AllocationInfo] = field(default_factory=dict)
    hard_stop_px: Optional[float] = None
    entry_lock_owner: Optional[str] = None
    entry_lock_until: Optional[float] = None
    frozen: bool = False
    working_order_count: int = 0
    working_orders: List[WorkingOrderInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.working_orders and self.working_order_count <= 0:
            self.working_order_count = len(self.working_orders)

    def get_allocation(self, strategy_id: str) -> int:
        """Get allocation qty for strategy."""
        alloc = self.allocations.get(strategy_id)
        return alloc.qty if alloc else 0


@dataclass
class AccountState:
    """Account state from OMS."""
    equity: float = 0.0
    buyable_cash: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    safe_mode: bool = False
    halt_new_entries: bool = False
    flatten_in_progress: bool = False
    gross_exposure_pct: float = 0.0
    regime_exposure_cap: float = 1.0


class OMSClient:
    """
    Async HTTP client for OMS service.

    Usage:
        oms = OMSClient("http://localhost:8000", strategy_id="PCIM")
        await oms.wait_ready()
        result = await oms.submit_intent(intent)
        await oms.close()
    """

    def __init__(self, base_url: str = "http://localhost:8000", strategy_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.strategy_id = strategy_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if aiohttp is None:
            raise ImportError("aiohttp required: pip install aiohttp")
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(enable_cleanup_closed=True)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def wait_ready(self, timeout: float = 60.0):
        """Wait for OMS to be ready. Raises TimeoutError if not ready."""
        session = await self._get_session()
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                async with session.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        try:
                            payload = await resp.json(content_type=None)
                        except Exception:
                            payload = {}
                        if _health_payload_ready(dict(payload or {})):
                            logger.info("OMS ready")
                            return
            except Exception:
                pass
            await asyncio.sleep(1)
        raise TimeoutError("OMS not ready")

    _SUBMIT_MAX_RETRIES = 3
    _SUBMIT_BACKOFF_BASE = 0.5  # seconds; doubles each retry (0.5, 1.0, 2.0)
    _READ_MAX_RETRIES = 2
    _READ_BACKOFF_BASE = 0.3  # seconds; doubles each retry (0.3, 0.6)

    async def _get_with_retry(self, url, params=None, timeout=10):
        """GET request with retry on transient connection errors."""
        last_err = None
        for attempt in range(self._READ_MAX_RETRIES + 1):
            try:
                session = await self._get_session()
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json()
            except Exception as e:
                last_err = e
                if attempt < self._READ_MAX_RETRIES:
                    delay = self._READ_BACKOFF_BASE * (2 ** attempt)
                    logger.debug(f"OMS read retry {attempt + 1}: {e}, retrying in {delay:.1f}s")
                    if self._session and not self._session.closed:
                        await self._session.close()
                    self._session = None
                    await asyncio.sleep(delay)
        logger.warning(f"OMS read failed after retries: {last_err}")
        return None

    async def submit_intent(self, intent: Intent) -> IntentResult:
        """Submit intent to OMS with retry on transient connection errors."""
        payload = {
            "intent_id": intent.intent_id,
            "idempotency_key": intent.idempotency_key,
            "intent_type": intent.intent_type.name,
            "strategy_id": intent.strategy_id,
            "symbol": intent.symbol,
            "desired_qty": intent.desired_qty,
            "target_qty": intent.target_qty,
            "urgency": intent.urgency.name,
            "time_horizon": intent.time_horizon.name,
            "constraints": {
                "max_slippage_bps": intent.constraints.max_slippage_bps,
                "max_spread_bps": intent.constraints.max_spread_bps,
                "limit_price": intent.constraints.limit_price,
                "stop_price": intent.constraints.stop_price,
                "expiry_ts": intent.constraints.expiry_ts,
                "execution_style": intent.constraints.execution_style,
            },
            "risk_payload": {
                "entry_px": intent.risk_payload.entry_px,
                "stop_px": intent.risk_payload.stop_px,
                "hard_stop_px": intent.risk_payload.hard_stop_px,
                "rationale_code": intent.risk_payload.rationale_code,
                "confidence": intent.risk_payload.confidence,
            },
            "signal_hash": intent.signal_hash,
            "metadata": dict(intent.metadata or {}),
        }

        last_err = None
        for attempt in range(self._SUBMIT_MAX_RETRIES):
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.base_url}/api/v1/intents",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return IntentResult(
                            intent_id=intent.intent_id,
                            status=IntentStatus.REJECTED,
                            message=f"OMS error {resp.status}: {text}",
                        )
                    data = await resp.json()
                    return IntentResult(
                        intent_id=data["intent_id"],
                        status=IntentStatus[data["status"]],
                        message=data.get("message", ""),
                        modified_qty=data.get("modified_qty"),
                        order_id=data.get("order_id"),
                        cooldown_until=data.get("cooldown_until"),
                        blocking_positions=data.get("blocking_positions"),
                        resource_conflict_type=data.get("resource_conflict_type"),
                        oms_received_at=data.get("oms_received_at"),
                        order_submitted_at=data.get("order_submitted_at"),
                    )
            except Exception as e:
                last_err = e
                if attempt < self._SUBMIT_MAX_RETRIES - 1:
                    delay = self._SUBMIT_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"OMS unreachable (attempt {attempt + 1}/{self._SUBMIT_MAX_RETRIES}): {e}, retrying in {delay:.1f}s")
                    # Force session recreation on next attempt
                    if self._session and not self._session.closed:
                        await self._session.close()
                    self._session = None
                    await asyncio.sleep(delay)

        logger.error(f"OMS unreachable after {self._SUBMIT_MAX_RETRIES} attempts: {last_err}")
        return IntentResult(
            intent_id=intent.intent_id,
            status=IntentStatus.REJECTED,
            message=f"OMS unreachable: {last_err}",
        )

    async def get_account_state(self) -> Optional[AccountState]:
        """Get account state from OMS (with capital allocation applied).

        Returns None if OMS is unreachable (distinguishes from default/empty state).
        """
        url = f"{self.base_url}/api/v1/state/account"
        params = {"strategy_id": self.strategy_id} if self.strategy_id else {}
        data = await self._get_with_retry(url, params=params)
        if data is None:
            return None
        return AccountState(
            equity=data.get("equity", 0.0),
            buyable_cash=data.get("buyable_cash", 0.0),
            daily_pnl=data.get("daily_pnl", 0.0),
            daily_pnl_pct=data.get("daily_pnl_pct", 0.0),
            safe_mode=data.get("safe_mode", False),
            halt_new_entries=data.get("halt_new_entries", False),
            flatten_in_progress=data.get("flatten_in_progress", False),
            gross_exposure_pct=data.get("gross_exposure_pct", 0.0),
            regime_exposure_cap=data.get("regime_exposure_cap", 1.0),
        )

    async def get_all_positions(self) -> Optional[Dict[str, PositionInfo]]:
        """Get all positions from OMS.

        Returns None if OMS is unreachable; returns {} if OMS says no positions.
        """
        data = await self._get_with_retry(f"{self.base_url}/api/v1/positions")
        if data is None:
            return None
        return {symbol: self._parse_position(symbol, pos) for symbol, pos in data.items()}

    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get single position from OMS."""
        data = await self._get_with_retry(f"{self.base_url}/api/v1/positions/{symbol}")
        if data is None:
            return None
        return self._parse_position(symbol, data)

    async def get_working_orders(self) -> Optional[List[WorkingOrderInfo]]:
        """Get all OMS working orders."""
        data = await self._get_with_retry(f"{self.base_url}/api/v1/working-orders")
        if data is None:
            return None
        return [self._parse_working_order(row) for row in data]

    async def get_allocation(self, symbol: str, strategy_id: str) -> Optional[int]:
        """Get allocation qty for strategy on symbol.

        Returns None if OMS is unreachable; returns 0 if position exists but no allocation.
        """
        pos = await self.get_position(symbol)
        if pos is None:
            return None
        return pos.get_allocation(strategy_id)

    async def get_strategy_allocations(self, strategy_id: str) -> Optional[Dict[str, AllocationInfo]]:
        """Get all allocations for a strategy.

        Returns None if OMS is unreachable; returns {} if no allocations exist.
        """
        url = f"{self.base_url}/api/v1/allocations/{strategy_id}"
        data = await self._get_with_retry(url)
        if data is None:
            return None
        return {
            symbol: AllocationInfo(
                strategy_id=alloc["strategy_id"],
                qty=alloc["qty"],
                cost_basis=alloc["cost_basis"],
                entry_ts=alloc.get("entry_ts"),
                soft_stop_px=alloc.get("soft_stop_px"),
                time_stop_ts=alloc.get("time_stop_ts"),
            )
            for symbol, alloc in data.items()
        }

    async def set_vi_cooldown(self, symbol: str, duration_sec: int):
        """Notify OMS of VI cooldown."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/api/v1/risk/vi-cooldown",
                json={"symbol": symbol, "duration_sec": duration_sec},
            ) as resp:
                pass
        except Exception as e:
            logger.debug(f"set_vi_cooldown failed: {e}")

    async def set_regime(self, regime: str):
        """Set market regime."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/api/v1/risk/regime",
                json={"regime": regime},
            ) as resp:
                pass
        except Exception as e:
            logger.debug(f"set_regime failed: {e}")

    async def report_heartbeat(
        self,
        mode: str = "RUNNING",
        symbols_hot: int = 0,
        symbols_warm: int = 0,
        symbols_cold: int = 0,
        positions_count: int = 0,
        last_error: Optional[str] = None,
        version: Optional[str] = None,
        strategy_id: Optional[str] = None,
        pulse_snapshot: Optional[dict] = None,
    ) -> None:
        """Report strategy heartbeat to OMS."""
        strat_id = strategy_id or self.strategy_id
        if not strat_id:
            logger.debug("report_heartbeat: no strategy_id configured")
            return
        session = await self._get_session()
        try:
            payload = {
                "mode": mode,
                "symbols_hot": symbols_hot,
                "symbols_warm": symbols_warm,
                "symbols_cold": symbols_cold,
                "positions_count": positions_count,
            }
            if last_error is not None:
                payload["last_error"] = last_error
            if version is not None:
                payload["version"] = version
            if pulse_snapshot:
                payload["pulse_verdict"] = pulse_snapshot.get("verdict")
                payload["pulse_md_ok_pct"] = pulse_snapshot.get("md_ok_pct")
                payload["pulse_signals_eval"] = pulse_snapshot.get("signals_eval")
            async with session.post(
                f"{self.base_url}/api/v1/strategies/{strat_id}/heartbeat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                pass
        except Exception as e:
            logger.debug(f"report_heartbeat failed: {e}")

    def _parse_position(self, symbol: str, data: dict) -> PositionInfo:
        """Parse position data from OMS response."""
        allocations = {}
        for strat_id, alloc_data in data.get("allocations", {}).items():
            allocations[strat_id] = AllocationInfo(
                strategy_id=alloc_data["strategy_id"],
                qty=alloc_data["qty"],
                cost_basis=alloc_data["cost_basis"],
                entry_ts=alloc_data.get("entry_ts"),
                soft_stop_px=alloc_data.get("soft_stop_px"),
                time_stop_ts=alloc_data.get("time_stop_ts"),
            )
        return PositionInfo(
            symbol=symbol,
            real_qty=data.get("real_qty", 0),
            avg_price=data.get("avg_price", 0.0),
            allocations=allocations,
            hard_stop_px=data.get("hard_stop_px"),
            entry_lock_owner=data.get("entry_lock_owner"),
            entry_lock_until=data.get("entry_lock_until"),
            frozen=data.get("frozen", False),
            working_order_count=data.get("working_order_count", 0),
            working_orders=[self._parse_working_order(row) for row in data.get("working_orders", [])],
        )

    def _parse_working_order(self, data: dict) -> WorkingOrderInfo:
        """Parse working order data from OMS response."""
        qty = int(data.get("qty", 0) or 0)
        filled_qty = int(data.get("filled_qty", 0) or 0)
        return WorkingOrderInfo(
            order_id=str(data.get("order_id") or ""),
            symbol=str(data.get("symbol") or "").zfill(6),
            side=str(data.get("side") or "").upper(),
            qty=qty,
            filled_qty=filled_qty,
            remaining_qty=int(data.get("remaining_qty", max(qty - filled_qty, 0)) or 0),
            price=float(data.get("price", 0.0) or 0.0),
            order_type=str(data.get("order_type") or ""),
            status=str(data.get("status") or ""),
            strategy_id=str(data.get("strategy_id") or "").upper(),
            intent_id=data.get("intent_id"),
            idempotency_key=data.get("idempotency_key"),
            submit_ref=data.get("submit_ref"),
            risk_stop_px=data.get("risk_stop_px"),
            risk_hard_stop_px=data.get("risk_hard_stop_px"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            submit_ts=data.get("submit_ts"),
            cancel_after_sec=data.get("cancel_after_sec"),
        )

    # Convenience property for compatibility with current code patterns
    @property
    def state(self):
        """Returns self for attribute access compatibility."""
        return _OMSStateProxy(self)


class _OMSStateProxy:
    """Proxy for accessing state via client with auto-refresh.

    WARNING: This proxy caches async state for synchronous access.
    Callers MUST `await proxy.refresh()` before reading properties,
    otherwise they get stale or default (0.0) values.
    """

    def __init__(self, client: OMSClient):
        self._client = client
        self._cached_account: Optional[AccountState] = None
        self._cached_positions: Dict[str, PositionInfo] = {}
        self._last_refresh: float = 0.0
        self._refresh_interval: float = 5.0  # seconds

    @property
    def equity(self) -> float:
        if self._cached_account is None:
            logger.warning(
                "_OMSStateProxy.equity accessed before refresh() — returning 0.0. "
                "Call 'await proxy.refresh()' first."
            )
            return 0.0
        if self.stale:
            logger.debug("_OMSStateProxy.equity is stale; consider calling refresh()")
        return self._cached_account.equity

    @property
    def stale(self) -> bool:
        """Check if cache needs refresh."""
        import time
        return (time.time() - self._last_refresh) > self._refresh_interval

    async def refresh(self):
        """Refresh cached state. Must be awaited before reading properties."""
        import time
        acct = await self._client.get_account_state()
        if acct is not None:
            self._cached_account = acct
        positions = await self._client.get_all_positions()
        if positions is not None:
            self._cached_positions = positions
        self._last_refresh = time.time()

    def get_all_positions(self) -> Dict[str, PositionInfo]:
        if not self._cached_positions and self._last_refresh == 0.0:
            logger.warning(
                "_OMSStateProxy.get_all_positions() called before refresh(). "
                "Call 'await proxy.refresh()' first."
            )
        return self._cached_positions
