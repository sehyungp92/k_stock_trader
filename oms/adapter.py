"""
KIS Execution Adapter: Bridge between OMS and KIS API.

This is the ONLY code that knows KIS endpoints.
"""

from __future__ import annotations
import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


class AdapterError(Enum):
    NONE = auto()
    RATE_LIMIT = auto()
    TEMP_ERROR = auto()
    REJECTED_INVALID = auto()
    REJECTED_RISK = auto()
    UNKNOWN = auto()


@dataclass
class BrokerQueryResult:
    """Result from broker query that distinguishes error from empty data.

    Callers MUST check `ok` before using `data`. When `ok` is False,
    `data` is empty and should NOT be treated as "no orders/positions exist".
    """
    ok: bool
    data: list = field(default_factory=list)
    error_message: str = ""


@dataclass
class AdapterResult:
    """Result from adapter operation."""
    success: bool
    order_id: Optional[str] = None
    error: AdapterError = AdapterError.NONE
    message: str = ""


@dataclass
class BrokerOrder:
    """Normalized broker order."""
    order_id: str
    symbol: str
    side: str
    qty: int
    filled_qty: int
    price: float
    status: str
    created_at: str
    branch: str = ""  # KRX_FWDG_ORD_ORGNO for cancel/revise


@dataclass
class BrokerPosition:
    """Normalized broker position."""
    symbol: str
    qty: int
    avg_price: float
    current_price: float
    pnl: float


@dataclass
class BrokerFill:
    """Normalized fill event."""
    order_id: str
    symbol: str
    side: str
    qty: int
    price: float
    timestamp: float


class KISExecutionAdapter:
    """
    KIS execution adapter.

    Wraps kis_core.KoreaInvestAPI and normalizes responses.
    """

    def __init__(self, kis_api: 'KoreaInvestAPI'):
        self.api = kis_api

    async def submit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        max_retries: int = 3,
    ) -> AdapterResult:
        """
        Submit order to KIS with retry on transient errors.

        Args:
            symbol: Stock code
            side: "BUY" or "SELL"
            qty: Order quantity
            order_type: "MARKET", "LIMIT", "STOP_LIMIT", "MARKETABLE_LIMIT"
            limit_price: Limit price (required for LIMIT/STOP_LIMIT)
            stop_price: Stop trigger price (required for STOP_LIMIT)
            max_retries: Max retry attempts for transient errors

        Returns:
            AdapterResult with order_id if successful
        """
        # Client-side order reference for deduplication across retries.
        # If first attempt succeeds but response times out, the retry
        # would create a duplicate order. This ref lets us detect that
        # by checking existing orders before retrying.
        client_ref = f"OMS-{uuid.uuid4().hex[:12]}"

        for attempt in range(max_retries):
            # On retry, check if the previous attempt actually succeeded
            if attempt > 0:
                try:
                    result = await self.get_orders()
                    if result.ok:
                        for bo in result.data:
                            if bo.symbol == symbol and bo.side == side and bo.qty == qty:
                                logger.warning(
                                    f"Detected likely duplicate order on retry: {bo.order_id} "
                                    f"(ref={client_ref})"
                                )
                                return AdapterResult(True, order_id=bo.order_id)
                except Exception:
                    pass  # Best-effort check; proceed with retry
            try:
                if order_type == "MARKET":
                    if side == "BUY":
                        order_id = await asyncio.to_thread(self.api.place_market_buy, symbol, qty)
                    else:
                        order_id = await asyncio.to_thread(self.api.place_market_sell, symbol, qty)

                elif order_type in ("LIMIT", "MARKETABLE_LIMIT"):
                    if side == "BUY":
                        order_id = await asyncio.to_thread(self.api.place_limit_buy, symbol, limit_price, qty)
                    else:
                        order_id = await asyncio.to_thread(self.api.place_limit_sell, symbol, limit_price, qty)

                elif order_type == "STOP_LIMIT":
                    logger.warning(f"STOP_LIMIT simulated as LIMIT at {stop_price}")
                    if side == "BUY":
                        order_id = await asyncio.to_thread(self.api.place_limit_buy, symbol, limit_price or stop_price, qty)
                    else:
                        order_id = await asyncio.to_thread(self.api.place_limit_sell, symbol, limit_price or stop_price, qty)
                else:
                    return AdapterResult(False, error=AdapterError.REJECTED_INVALID, message=f"Unknown order type: {order_type}")

                if order_id:
                    return AdapterResult(True, order_id=order_id)
                else:
                    logger.warning(
                        f"KIS order rejected: {symbol} {side} x{qty} "
                        f"type={order_type} limit={limit_price}"
                    )
                    return AdapterResult(
                        False, error=AdapterError.REJECTED_INVALID,
                        message=f"Order rejected by KIS: {symbol} {side} x{qty} type={order_type}",
                    )

            except Exception as e:
                err_str = str(e).lower()
                if attempt < max_retries - 1 and ("rate" in err_str or "timeout" in err_str or "temporary" in err_str):
                    logger.warning(f"Transient error (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(f"Order submit error: {e}")
                return AdapterResult(False, error=AdapterError.TEMP_ERROR, message=str(e))

        return AdapterResult(False, error=AdapterError.TEMP_ERROR, message="Max retries exhausted")

    async def cancel_order(self, order_id: str, symbol: str, qty: int, branch: str = "") -> AdapterResult:
        """Cancel order. Looks up branch from open orders if not provided."""
        try:
            # If branch not stored, look it up from get_orders()
            if not branch:
                try:
                    orders_df = await asyncio.to_thread(self.api.get_orders)
                    if orders_df is not None and order_id in orders_df.index:
                        branch = str(orders_df.loc[order_id, '주문점'])
                except Exception as e:
                    logger.debug(f"Branch lookup failed for {order_id}: {e}")

            kwargs = {}
            if branch:
                kwargs['order_branch'] = branch
            result = await asyncio.to_thread(self.api.cancel_order, order_id, qty, **kwargs)
            if result:
                return AdapterResult(True)
            return AdapterResult(False, error=AdapterError.REJECTED_INVALID)
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return AdapterResult(False, error=AdapterError.TEMP_ERROR, message=str(e))

    async def get_orders(self) -> BrokerQueryResult:
        """Get open orders. Returns BrokerQueryResult — check .ok before using .data."""
        try:
            df = await asyncio.to_thread(self.api.get_orders)
            if df is None:
                return BrokerQueryResult(ok=True, data=[])

            orders = []
            for odno, row in df.iterrows():
                raw_side = row.get('매도매수구분코드', row.get('매매구분코드', ''))
                side = "SELL" if str(raw_side) in ("01", "sell", "SELL") else "BUY"
                orders.append(BrokerOrder(
                    order_id=str(odno),
                    symbol=row['종목코드'],
                    side=side,
                    qty=int(row['주문수량']),
                    filled_qty=int(row['주문수량']) - int(row['주문가능수량']),
                    price=float(row['주문가격']),
                    status="WORKING",
                    created_at=row['시간'],
                    branch=str(row.get('주문점', '')),
                ))
            return BrokerQueryResult(ok=True, data=orders)
        except Exception as e:
            logger.error(f"Get orders error: {e}")
            return BrokerQueryResult(ok=False, error_message=str(e))

    async def get_positions(self) -> BrokerQueryResult:
        """Get current positions. Returns BrokerQueryResult — check .ok before using .data."""
        try:
            _, df = await asyncio.to_thread(self.api.get_acct_balance)
            if df.empty:
                return BrokerQueryResult(ok=True, data=[])

            positions = []
            for _, row in df.iterrows():
                positions.append(BrokerPosition(
                    symbol=row['종목코드'],
                    qty=int(row['보유수량']),
                    avg_price=float(row['매입단가']),
                    current_price=float(row['현재가']),
                    pnl=float(row['수익률']),
                ))
            return BrokerQueryResult(ok=True, data=positions)
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return BrokerQueryResult(ok=False, error_message=str(e))

    async def get_balance_snapshot(self) -> Tuple[BrokerQueryResult, Optional[int]]:
        """Get positions and equity from a single get_acct_balance() call.

        Returns (positions_result, equity) — equity is None on failure.
        Eliminates the duplicate get_acct_balance() call that previously
        occurred when get_positions() and get_account_info() were called
        separately during reconciliation.
        """
        try:
            total_amt, df = await asyncio.to_thread(self.api.get_acct_balance)
            if df.empty:
                return BrokerQueryResult(ok=True, data=[]), total_amt

            positions = []
            for _, row in df.iterrows():
                positions.append(BrokerPosition(
                    symbol=row['종목코드'],
                    qty=int(row['보유수량']),
                    avg_price=float(row['매입단가']),
                    current_price=float(row['현재가']),
                    pnl=float(row['수익률']),
                ))
            return BrokerQueryResult(ok=True, data=positions), total_amt
        except Exception as e:
            logger.error(f"Get balance snapshot error: {e}")
            return BrokerQueryResult(ok=False, error_message=str(e)), None

    async def get_buyable_cash(self) -> Optional[int]:
        """Get buyable cash from KIS API. Returns None on failure."""
        try:
            return await asyncio.to_thread(self.api.get_buyable_cash)
        except Exception as e:
            logger.error(f"Get buyable cash error: {e}")
            return None

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account balance info. Raises on failure to avoid false equity=0."""
        try:
            total_amt, df = await asyncio.to_thread(self.api.get_acct_balance)
            buyable = await asyncio.to_thread(self.api.get_buyable_cash)

            return {
                "equity": total_amt,
                "buyable_cash": buyable or 0,
                "positions_count": len(df),
            }
        except Exception as e:
            logger.error(f"Get account error: {e}")
            raise  # Let reconciliation handle the error and skip this cycle
