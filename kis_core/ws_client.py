"""
KIS WebSocket Client - Shared real-time data client.

Provides async WebSocket connection management for KIS real-time tick
and orderbook data streams (H0STCNT0, H0STASP0).

Features:
- Auto-reconnect with exponential backoff
- Callback-based message dispatch
- Subscription budget management (40 max registrations, KIS limit is 41)
- Message parsing for tick and bid/ask streams
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

# Default max WebSocket registrations (KIS limit)
# KIS allows 41 total registrations per session across all real-time data types:
# H0STCNT0 (tick) + H0STASP0 (bid/ask) + expected execution + execution notification
# Ref: https://hky035.github.io/web/refact-kis-websocket/
WS_MAX_REGS_DEFAULT = 40  # Use 40 to leave 1 slot for execution notification


@dataclass
class TickMessage:
    """Parsed H0STCNT0 tick message."""

    ticker: str
    price: float
    volume: float
    cum_vol: float
    cum_val: float
    vi_ref: float
    timestamp: datetime


@dataclass
class AskBidMessage:
    """Parsed H0STASP0 bid/ask message."""

    ticker: str
    bid: float
    ask: float


def get_kst_now() -> datetime:
    """Get current time in KST."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return datetime.now(tz=ZoneInfo("Asia/Seoul"))


def parse_ws_message(raw: str) -> Optional[tuple[str, str, str]]:
    """
    Parse KIS WebSocket message header.

    KIS WebSocket messages are pipe-delimited:
    header_field0^header_field1|data_type|...|data

    Returns:
        Tuple of (tr_id, data_type, data_body) or None if invalid.
    """
    if '|' not in raw:
        return None

    parts = raw.split('|')
    if len(parts) < 4:
        return None

    header_parts = parts[0].split('^')
    tr_id = header_parts[1] if len(header_parts) > 1 else ""
    data = parts[3] if len(parts) > 3 else ""

    return (tr_id, parts[1] if len(parts) > 1 else "", data)


def parse_tick_message(data: str, now_kst: Optional[datetime] = None) -> Optional[TickMessage]:
    """
    Parse H0STCNT0 tick data into TickMessage.

    KIS H0STCNT0 fields (caret-delimited):
    0: MKSC_SHRN_ISCD (ticker)
    1: STCK_CNTG_HOUR (timestamp HHMMSS)
    2: STCK_PRPR (price)
    12: CNTG_VOL (tick volume)
    13: ACML_VOL (cumulative volume)
    14: ACML_TR_PBMN (cumulative value)
    45: VI_STND_PRC (VI reference price)
    """
    fields = data.split('^')
    if len(fields) < 15:
        return None

    ticker = fields[0]
    if not ticker:
        return None

    try:
        price = float(fields[2]) if fields[2] else 0.0
        volume = float(fields[12]) if fields[12] else 0.0
        cum_vol = float(fields[13]) if fields[13] else 0.0
        cum_val = float(fields[14]) if fields[14] else 0.0
        vi_ref = float(fields[45]) if len(fields) > 45 and fields[45] else 0.0
    except (ValueError, IndexError):
        return None

    if price <= 0:
        return None

    # Parse timestamp (HHMMSS) into KST datetime
    if now_kst is None:
        now_kst = get_kst_now()

    ts_str = fields[1]
    if len(ts_str) >= 6:
        try:
            ts = now_kst.replace(
                hour=int(ts_str[:2]),
                minute=int(ts_str[2:4]),
                second=int(ts_str[4:6]),
                microsecond=0,
            )
        except ValueError:
            ts = now_kst
    else:
        ts = now_kst

    return TickMessage(
        ticker=ticker,
        price=price,
        volume=volume,
        cum_vol=cum_vol,
        cum_val=cum_val,
        vi_ref=vi_ref,
        timestamp=ts,
    )


def parse_askbid_message(data: str) -> Optional[AskBidMessage]:
    """
    Parse H0STASP0 bid/ask data into AskBidMessage.

    KIS H0STASP0 fields (caret-delimited):
    0: MKSC_SHRN_ISCD (ticker)
    3: ASKP1 (best ask)
    13: BIDP1 (best bid)
    """
    fields = data.split('^')
    if len(fields) < 4:
        return None

    ticker = fields[0]
    if not ticker:
        return None

    try:
        ask = float(fields[3]) if len(fields) > 3 and fields[3] else 0.0
        bid = float(fields[13]) if len(fields) > 13 and fields[13] else 0.0
    except (ValueError, IndexError):
        return None

    return AskBidMessage(ticker=ticker, bid=bid, ask=ask)


class KISWebSocketClient:
    """
    Async WebSocket client for KIS real-time data.

    Manages connection lifecycle, auto-reconnect, and message dispatch
    to registered callbacks.
    """

    def __init__(
        self,
        api: Any,
        reconnect_delay_base: float = 1.0,
        reconnect_delay_max: float = 30.0,
        connect_timeout: float = 30.0,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
    ):
        """
        Args:
            api: KoreaInvestAPI instance (for building subscription payloads).
            reconnect_delay_base: Initial reconnect delay in seconds.
            reconnect_delay_max: Maximum reconnect delay in seconds.
            connect_timeout: Timeout for WebSocket connection in seconds.
            ping_interval: Interval between ping frames in seconds.
            ping_timeout: Timeout waiting for pong response in seconds.
        """
        self.api = api
        self.ws: Any = None
        self._url: str = ""
        self._connected: bool = False
        self._running: bool = False
        self._reconnect_delay_base = reconnect_delay_base
        self._reconnect_delay_max = reconnect_delay_max
        self._reconnect_attempts = 0
        self._connect_timeout = connect_timeout
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._connected_since: float = 0.0  # Timestamp of last successful connect
        self._STABLE_CONNECTION_SEC: float = 30.0  # Min duration before resetting backoff

        # Callbacks for message types
        self._tick_callbacks: List[Callable[[TickMessage], None]] = []
        self._askbid_callbacks: List[Callable[[AskBidMessage], None]] = []

        # Subscription tracking (for reconnect replay)
        self._tick_subs: Set[str] = set()
        self._asp_subs: Set[str] = set()

    @property
    def connected(self) -> bool:
        """Whether the WebSocket is currently connected."""
        return self._connected and self.ws is not None

    async def connect(self, url: str) -> bool:
        """
        Connect to the WebSocket server.

        Args:
            url: WebSocket URL to connect to.

        Returns:
            True if connection successful, False otherwise.
        """
        self._url = url
        try:
            import websockets
            import time as _time
            self.ws = await websockets.connect(
                url,
                open_timeout=self._connect_timeout,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
            )
            self._connected = True
            self._connected_since = _time.monotonic()
            # Don't reset _reconnect_attempts here — wait until connection is stable
            # (see _STABLE_CONNECTION_SEC). This prevents rapid reconnect cycling
            # when the server accepts connections but immediately drops them.
            logger.info(f"WebSocket connected to {url}")
            return True
        except ImportError:
            logger.error("websockets package not installed")
            return False
        except Exception as e:
            logger.warning(f"WebSocket connect failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        self._running = False
        self._connected = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
        logger.info("WebSocket disconnected")

    async def _reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        if not self._url:
            return False

        delay = min(
            self._reconnect_delay_base * (2 ** self._reconnect_attempts),
            self._reconnect_delay_max,
        )
        self._reconnect_attempts += 1
        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts})")

        await asyncio.sleep(delay)

        if await self.connect(self._url):
            # Replay subscriptions
            await self._replay_subscriptions()
            return True
        return False

    async def _replay_subscriptions(self) -> None:
        """Re-subscribe to all previously subscribed tickers after reconnect."""
        failed_tick = []
        for ticker in list(self._tick_subs):
            try:
                payload = self.api.get_send_data(cmd=3, stockcode=ticker)
                await self.ws.send(payload)
                logger.debug(f"Replayed CNT subscription: {ticker}")
            except Exception as e:
                logger.warning(f"Failed to replay CNT sub for {ticker}: {e}")
                failed_tick.append(ticker)

        failed_asp = []
        for ticker in list(self._asp_subs):
            try:
                payload = self.api.get_send_data(cmd=1, stockcode=ticker)
                await self.ws.send(payload)
                logger.debug(f"Replayed ASP subscription: {ticker}")
            except Exception as e:
                logger.warning(f"Failed to replay ASP sub for {ticker}: {e}")
                failed_asp.append(ticker)

        # Remove failed subscriptions so strategies know they're not receiving data
        for ticker in failed_tick:
            self._tick_subs.discard(ticker)
            logger.warning(f"Removed failed CNT subscription: {ticker}")
        for ticker in failed_asp:
            self._asp_subs.discard(ticker)
            logger.warning(f"Removed failed ASP subscription: {ticker}")

    def on_tick(self, callback: Callable[[TickMessage], None]) -> None:
        """Register a callback for tick messages."""
        self._tick_callbacks.append(callback)

    def on_askbid(self, callback: Callable[[AskBidMessage], None]) -> None:
        """Register a callback for bid/ask messages."""
        self._askbid_callbacks.append(callback)

    async def subscribe_tick(self, ticker: str) -> bool:
        """Subscribe to tick stream (H0STCNT0) for a ticker."""
        if not self.connected:
            return False
        if ticker in self._tick_subs:
            return True
        try:
            payload = self.api.get_send_data(cmd=3, stockcode=ticker)
            await self.ws.send(payload)
            self._tick_subs.add(ticker)
            logger.debug(f"Subscribed CNT: {ticker}")
            return True
        except Exception as e:
            logger.error(f"Subscribe CNT error: {e}")
            return False

    async def subscribe_askbid(self, ticker: str) -> bool:
        """Subscribe to bid/ask stream (H0STASP0) for a ticker."""
        if not self.connected:
            return False
        if ticker in self._asp_subs:
            return True
        try:
            payload = self.api.get_send_data(cmd=1, stockcode=ticker)
            await self.ws.send(payload)
            self._asp_subs.add(ticker)
            logger.debug(f"Subscribed ASP: {ticker}")
            return True
        except Exception as e:
            logger.error(f"Subscribe ASP error: {e}")
            return False

    async def unsubscribe_tick(self, ticker: str) -> None:
        """Unsubscribe from tick stream (H0STCNT0) for a ticker."""
        if not self.connected or ticker not in self._tick_subs:
            self._tick_subs.discard(ticker)
            return
        try:
            payload = self.api.get_send_data(cmd=4, stockcode=ticker)
            await self.ws.send(payload)
            self._tick_subs.discard(ticker)
            logger.debug(f"Unsubscribed CNT: {ticker}")
        except Exception as e:
            logger.error(f"Unsubscribe CNT error: {e}")

    async def unsubscribe_askbid(self, ticker: str) -> None:
        """Unsubscribe from bid/ask stream (H0STASP0) for a ticker."""
        if not self.connected or ticker not in self._asp_subs:
            self._asp_subs.discard(ticker)
            return
        try:
            payload = self.api.get_send_data(cmd=2, stockcode=ticker)
            await self.ws.send(payload)
            self._asp_subs.discard(ticker)
            logger.debug(f"Unsubscribed ASP: {ticker}")
        except Exception as e:
            logger.error(f"Unsubscribe ASP error: {e}")

    def get_tick_subs(self) -> Set[str]:
        """Get current tick subscriptions."""
        return self._tick_subs.copy()

    def get_asp_subs(self) -> Set[str]:
        """Get current bid/ask subscriptions."""
        return self._asp_subs.copy()

    def total_subs(self) -> int:
        """Get total subscription count."""
        return len(self._tick_subs) + len(self._asp_subs)

    async def run(self, auto_reconnect: bool = True) -> None:
        """
        Run the message read loop.

        This is a blocking coroutine that reads messages and dispatches
        them to registered callbacks. Should be run as a task.

        Args:
            auto_reconnect: Whether to automatically reconnect on disconnect.
        """
        self._running = True

        while self._running:
            if not self.connected:
                if auto_reconnect:
                    if not await self._reconnect():
                        continue
                else:
                    break

            try:
                async for raw in self.ws:
                    if not self._running:
                        break

                    # Reset backoff once connection has been stable long enough
                    import time as _time
                    if (self._reconnect_attempts > 0
                            and self._connected_since > 0
                            and (_time.monotonic() - self._connected_since) >= self._STABLE_CONNECTION_SEC):
                        self._reconnect_attempts = 0

                    try:
                        self._dispatch_message(raw)
                    except Exception as e:
                        logger.debug(f"WS dispatch error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"WS read error: {e}")
                self._connected = False
                if not auto_reconnect:
                    break

        self._running = False

    def _dispatch_message(self, raw: str) -> None:
        """Parse and dispatch a single WebSocket message."""
        parsed = parse_ws_message(raw)
        if not parsed:
            return

        tr_id, _, data = parsed
        now_kst = get_kst_now()

        if tr_id == "H0STCNT0":
            msg = parse_tick_message(data, now_kst)
            if msg:
                for cb in self._tick_callbacks:
                    try:
                        cb(msg)
                    except Exception as e:
                        logger.debug(f"Tick callback error: {e}")

        elif tr_id == "H0STASP0":
            msg = parse_askbid_message(data)
            if msg:
                for cb in self._askbid_callbacks:
                    try:
                        cb(msg)
                    except Exception as e:
                        logger.debug(f"AskBid callback error: {e}")


class BaseSubscriptionManager:
    """
    Base subscription budget manager.

    Tracks tick (H0STCNT0) and bid/ask (H0STASP0) subscriptions
    within the KIS WebSocket registration limit.

    Subclasses can override eviction logic for strategy-specific
    priority management.

    WebSocket Slot Sharing:
    -----------------------
    KIS enforces a hard limit of 20 concurrent WebSocket registrations
    (WS_MAX_REGS_DEFAULT). This limit is shared across ALL strategies
    using the same KISWebSocketClient instance.

    When running multiple strategies concurrently:
    1. Use a single shared KISWebSocketClient for all strategies
    2. Each strategy's SubscriptionManager will compete for the 20 slots
    3. The eviction logic (_evict_for_tick, _evict_for_askbid) determines
       which subscriptions are dropped when budget is exceeded
    4. Strategies should set their max_regs to leave room for others:
       - KMP: 20 slots (runs in exclusive window 09:15-10:00)
       - KPR: 20 slots (HOT tier only, uses tiering to reduce needs)
       - NULRIMOK: 20 slots (active set rotation)
       - PCIM: 0 slots (no WebSocket usage)

    Recommended concurrent configurations:
    - KMP + PCIM: Safe (PCIM uses no WS slots)
    - KPR + PCIM: Safe (PCIM uses no WS slots)
    - KMP + KPR: Requires careful slot management or time-based arbitration
    - Any strategy alone: Full 20 slots available
    """

    def __init__(
        self,
        ws_client: KISWebSocketClient,
        max_regs: int = WS_MAX_REGS_DEFAULT,
    ):
        """
        Args:
            ws_client: KISWebSocketClient instance.
            max_regs: Maximum combined registrations (default 40, KIS limit is 41).
        """
        self.ws = ws_client
        self.max_regs = max_regs

        # Warn if max_regs exceeds KIS hard limit
        if max_regs > WS_MAX_REGS_DEFAULT:
            logger.warning(
                f"max_regs={max_regs} exceeds KIS WebSocket limit of {WS_MAX_REGS_DEFAULT}. "
                f"Subscriptions beyond {WS_MAX_REGS_DEFAULT} will be rejected by KIS."
            )

    @property
    def tick_subs(self) -> Set[str]:
        """Current tick subscriptions."""
        return self.ws.get_tick_subs()

    @property
    def asp_subs(self) -> Set[str]:
        """Current bid/ask subscriptions."""
        return self.ws.get_asp_subs()

    def total_regs(self) -> int:
        """Total registration count."""
        return self.ws.total_subs()

    async def ensure_tick(self, ticker: str) -> bool:
        """
        Ensure ticker has tick subscription.

        Returns True if subscribed (or already subscribed), False if budget exceeded.
        """
        if ticker in self.tick_subs:
            return True
        if self.total_regs() >= self.max_regs:
            await self._evict_for_tick(ticker)
        if self.total_regs() >= self.max_regs:
            return False
        return await self.ws.subscribe_tick(ticker)

    async def ensure_askbid(self, ticker: str) -> bool:
        """
        Ensure ticker has bid/ask subscription.

        Returns True if subscribed (or already subscribed), False if budget exceeded.
        """
        if ticker in self.asp_subs:
            return True
        if self.total_regs() >= self.max_regs:
            await self._evict_for_askbid(ticker)
        if self.total_regs() >= self.max_regs:
            return False
        return await self.ws.subscribe_askbid(ticker)

    async def drop_tick(self, ticker: str) -> None:
        """Drop tick subscription for ticker."""
        await self.ws.unsubscribe_tick(ticker)

    async def drop_askbid(self, ticker: str) -> None:
        """Drop bid/ask subscription for ticker."""
        await self.ws.unsubscribe_askbid(ticker)

    async def drop_all(self, ticker: str) -> None:
        """Drop all subscriptions for ticker."""
        await self.drop_askbid(ticker)
        await self.drop_tick(ticker)

    async def _evict_for_tick(self, incoming: str) -> None:
        """
        Evict a subscription to make room for a new tick subscription.

        Default: evict a tick-only subscription (no ASP).
        Override in subclasses for strategy-specific logic.
        """
        for t in list(self.tick_subs):
            if t not in self.asp_subs:
                await self.drop_tick(t)
                return

    async def _evict_for_askbid(self, incoming: str) -> None:
        """
        Evict a subscription to make room for a new bid/ask subscription.

        Default: evict an ASP subscription.
        Override in subclasses for strategy-specific logic.
        """
        for t in list(self.asp_subs):
            await self.drop_askbid(t)
            return
