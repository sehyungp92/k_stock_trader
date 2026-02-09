"""
Mock KoreaInvestAPI for testing.

Simulates KIS API responses without network calls.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import uuid


@dataclass
class MockOrder:
    """Represents a mock order."""
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    order_type: str  # "MARKET" or "LIMIT"
    qty: int
    price: float
    filled_qty: int = 0
    status: str = "WORKING"
    created_at: str = "09:30:00"


@dataclass
class MockPosition:
    """Represents a mock position."""
    symbol: str
    qty: int
    avg_price: float
    current_price: float = 0.0
    pnl: float = 0.0


class MockKoreaInvestAPI:
    """
    Mock KoreaInvestAPI for testing.

    Features:
    - Tracks orders and positions in memory
    - Simulates order submission and fills
    - Configurable failure modes for testing error handling
    """

    def __init__(
        self,
        prices: Optional[Dict[str, float]] = None,
        positions: Optional[List[MockPosition]] = None,
        fail_orders: bool = False,
        fail_rate_limit: bool = False,
    ):
        self.prices = prices or {}
        self._positions: Dict[str, MockPosition] = {}
        self._orders: Dict[str, MockOrder] = {}
        self._order_counter = 0

        # Failure modes
        self.fail_orders = fail_orders
        self.fail_rate_limit = fail_rate_limit
        self._fail_count = 0

        # Account info
        self.equity = 100_000_000  # 100M KRW default
        self.buyable_cash = 50_000_000  # 50M KRW default

        # Initialize positions
        if positions:
            for pos in positions:
                self._positions[pos.symbol] = pos

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last price for symbol."""
        return self.prices.get(symbol)

    def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price data."""
        price = self.prices.get(symbol)
        if price is None:
            return None
        return {
            "stck_prpr": price,
            "stck_oprc": price * 0.99,
            "stck_hgpr": price * 1.01,
            "stck_lwpr": price * 0.98,
            "acml_vol": 1000000,
            "acml_tr_pbmn": price * 1000000,
        }

    def get_minute_bars(self, symbol: str, minutes: int = 30) -> Optional[pd.DataFrame]:
        """Get minute bars (returns empty DataFrame for mock)."""
        price = self.prices.get(symbol, 10000)
        # Generate simple mock bars
        data = {
            "open": [price * (1 - 0.001 * i) for i in range(minutes)],
            "high": [price * (1 + 0.002 * i) for i in range(minutes)],
            "low": [price * (1 - 0.002 * i) for i in range(minutes)],
            "close": [price * (1 + 0.001 * i) for i in range(minutes)],
            "volume": [10000 + i * 100 for i in range(minutes)],
        }
        return pd.DataFrame(data)

    def get_daily_bars(self, symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
        """Get daily bars (returns mock data)."""
        price = self.prices.get(symbol, 10000)
        data = {
            "open": [price * (1 - 0.01 * i / days) for i in range(days)],
            "high": [price * (1 + 0.02 * i / days) for i in range(days)],
            "low": [price * (1 - 0.02 * i / days) for i in range(days)],
            "close": [price * (1 + 0.01 * i / days) for i in range(days)],
            "volume": [100000 + i * 1000 for i in range(days)],
        }
        return pd.DataFrame(data)

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD{self._order_counter:08d}"

    def place_limit_buy(self, symbol: str, price: float, qty: int) -> Optional[str]:
        """Place limit buy order."""
        if self.fail_orders:
            return None
        if self.fail_rate_limit:
            self._fail_count += 1
            if self._fail_count <= 2:
                raise Exception("rate limit exceeded")
            self._fail_count = 0

        order_id = self._generate_order_id()
        self._orders[order_id] = MockOrder(
            order_id=order_id,
            symbol=symbol,
            side="BUY",
            order_type="LIMIT",
            qty=qty,
            price=price,
        )
        return order_id

    def place_limit_sell(self, symbol: str, price: float, qty: int) -> Optional[str]:
        """Place limit sell order."""
        if self.fail_orders:
            return None

        order_id = self._generate_order_id()
        self._orders[order_id] = MockOrder(
            order_id=order_id,
            symbol=symbol,
            side="SELL",
            order_type="LIMIT",
            qty=qty,
            price=price,
        )
        return order_id

    def place_market_buy(self, symbol: str, qty: int) -> Optional[str]:
        """Place market buy order."""
        if self.fail_orders:
            return None

        price = self.prices.get(symbol, 10000)
        order_id = self._generate_order_id()
        order = MockOrder(
            order_id=order_id,
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            qty=qty,
            price=price,
        )
        # Market orders fill immediately
        order.filled_qty = qty
        order.status = "FILLED"
        self._orders[order_id] = order

        # Update position
        self._update_position_on_fill(symbol, qty, price)
        return order_id

    def place_market_sell(self, symbol: str, qty: int) -> Optional[str]:
        """Place market sell order."""
        if self.fail_orders:
            return None

        price = self.prices.get(symbol, 10000)
        order_id = self._generate_order_id()
        order = MockOrder(
            order_id=order_id,
            symbol=symbol,
            side="SELL",
            order_type="MARKET",
            qty=qty,
            price=price,
        )
        # Market orders fill immediately
        order.filled_qty = qty
        order.status = "FILLED"
        self._orders[order_id] = order

        # Update position
        self._update_position_on_fill(symbol, -qty, price)
        return order_id

    def cancel_order(self, order_id: str, qty: int) -> bool:
        """Cancel order."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status == "FILLED":
            return False

        order.status = "CANCELLED"
        return True

    def modify_order(self, order_id: str, price: float, qty: int) -> bool:
        """Modify order price/qty."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status != "WORKING":
            return False

        order.price = price
        order.qty = qty
        return True

    def get_orders(self) -> Optional[pd.DataFrame]:
        """Get open orders."""
        working_orders = [o for o in self._orders.values() if o.status == "WORKING"]
        if not working_orders:
            return pd.DataFrame()

        data = {
            "종목코드": [o.symbol for o in working_orders],
            "주문수량": [o.qty for o in working_orders],
            "주문가능수량": [o.qty - o.filled_qty for o in working_orders],
            "주문가격": [o.price for o in working_orders],
            "매도매수구분코드": ["01" if o.side == "SELL" else "02" for o in working_orders],
            "시간": [o.created_at for o in working_orders],
        }
        df = pd.DataFrame(data)
        df.index = [o.order_id for o in working_orders]
        return df

    def get_acct_balance(self) -> tuple[float, pd.DataFrame]:
        """Get account balance."""
        positions = list(self._positions.values())
        if not positions:
            return self.equity, pd.DataFrame()

        data = {
            "종목코드": [p.symbol for p in positions],
            "보유수량": [p.qty for p in positions],
            "매입단가": [p.avg_price for p in positions],
            "현재가": [p.current_price or p.avg_price for p in positions],
            "수익률": [p.pnl for p in positions],
        }
        return self.equity, pd.DataFrame(data)

    def get_buyable_cash(self) -> float:
        """Get available cash."""
        return self.buyable_cash

    def _update_position_on_fill(self, symbol: str, qty_delta: int, price: float) -> None:
        """Update position after fill."""
        if symbol not in self._positions:
            if qty_delta > 0:
                self._positions[symbol] = MockPosition(
                    symbol=symbol,
                    qty=qty_delta,
                    avg_price=price,
                    current_price=price,
                )
        else:
            pos = self._positions[symbol]
            new_qty = pos.qty + qty_delta
            if new_qty <= 0:
                del self._positions[symbol]
            else:
                if qty_delta > 0:
                    # Average up/down
                    total_cost = pos.qty * pos.avg_price + qty_delta * price
                    pos.avg_price = total_cost / new_qty
                pos.qty = new_qty
                pos.current_price = price

    # --- Test helper methods ---

    def set_price(self, symbol: str, price: float) -> None:
        """Set price for symbol."""
        self.prices[symbol] = price
        if symbol in self._positions:
            self._positions[symbol].current_price = price

    def fill_order(self, order_id: str, fill_qty: Optional[int] = None) -> bool:
        """Simulate order fill."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status == "FILLED":
            return False

        qty = fill_qty or (order.qty - order.filled_qty)
        order.filled_qty += qty
        if order.filled_qty >= order.qty:
            order.status = "FILLED"
        else:
            order.status = "PARTIAL"

        # Update position
        delta = qty if order.side == "BUY" else -qty
        self._update_position_on_fill(order.symbol, delta, order.price)
        return True

    def get_order(self, order_id: str) -> Optional[MockOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_position(self, symbol: str) -> Optional[MockPosition]:
        """Get position for symbol."""
        return self._positions.get(symbol)

    def reset(self) -> None:
        """Reset all state."""
        self._orders.clear()
        self._positions.clear()
        self._order_counter = 0
        self._fail_count = 0
