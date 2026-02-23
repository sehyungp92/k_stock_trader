"""OMS HTTP Client for strategies."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
from loguru import logger

try:
    import aiohttp
except ImportError:
    aiohttp = None

from oms.intent import Intent, IntentResult, IntentStatus


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
        oms = OMSClient("http://localhost:8000", strategy_id="KMP")
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
            self._session = aiohttp.ClientSession()
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
                        logger.info("OMS ready")
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
        raise TimeoutError("OMS not ready")

    _SUBMIT_MAX_RETRIES = 3
    _SUBMIT_BACKOFF_BASE = 0.5  # seconds; doubles each retry (0.5, 1.0, 2.0)

    async def submit_intent(self, intent: Intent) -> IntentResult:
        """Submit intent to OMS with retry on transient connection errors."""
        payload = {
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
            },
            "risk_payload": {
                "entry_px": intent.risk_payload.entry_px,
                "stop_px": intent.risk_payload.stop_px,
                "hard_stop_px": intent.risk_payload.hard_stop_px,
                "rationale_code": intent.risk_payload.rationale_code,
                "confidence": intent.risk_payload.confidence,
            },
            "signal_hash": intent.signal_hash,
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

    async def get_account_state(self) -> AccountState:
        """Get account state from OMS (with capital allocation applied)."""
        session = await self._get_session()
        try:
            # Pass strategy_id to get allocated equity
            url = f"{self.base_url}/api/v1/state/account"
            params = {"strategy_id": self.strategy_id} if self.strategy_id else {}
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return AccountState()
                data = await resp.json()
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
        except Exception as e:
            logger.debug(f"get_account_state failed: {e}")
            return AccountState()

    async def get_all_positions(self) -> Dict[str, PositionInfo]:
        """Get all positions from OMS."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/v1/positions", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                return {symbol: self._parse_position(symbol, pos) for symbol, pos in data.items()}
        except Exception as e:
            logger.debug(f"get_all_positions failed: {e}")
            return {}

    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get single position from OMS."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/v1/positions/{symbol}", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return self._parse_position(symbol, data)
        except Exception as e:
            logger.debug(f"get_position failed: {e}")
            return None

    async def get_allocation(self, symbol: str, strategy_id: str) -> int:
        """Get allocation qty for strategy on symbol."""
        pos = await self.get_position(symbol)
        return pos.get_allocation(strategy_id) if pos else 0

    async def get_strategy_allocations(self, strategy_id: str) -> Dict[str, AllocationInfo]:
        """Get all allocations for a strategy."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/v1/allocations/{strategy_id}", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
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
        except Exception as e:
            logger.debug(f"get_strategy_allocations failed: {e}")
            return {}

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
        self._cached_account = await self._client.get_account_state()
        self._cached_positions = await self._client.get_all_positions()
        self._last_refresh = time.time()

    def get_all_positions(self) -> Dict[str, PositionInfo]:
        if not self._cached_positions and self._last_refresh == 0.0:
            logger.warning(
                "_OMSStateProxy.get_all_positions() called before refresh(). "
                "Call 'await proxy.refresh()' first."
            )
        return self._cached_positions
