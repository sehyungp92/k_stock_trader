"""Tests for KIS client methods."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd

from tests.mocks.mock_kis_api import MockKoreaInvestAPI, MockPosition


class TestGetLastPrice:
    """Tests for get_last_price method."""

    def test_returns_price_for_known_symbol(self):
        """Test returns price for known symbol."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})

        price = api.get_last_price("005930")

        assert price == 72000

    def test_returns_none_for_unknown_symbol(self):
        """Test returns None for unknown symbol."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})

        price = api.get_last_price("UNKNOWN")

        assert price is None

    def test_updates_with_set_price(self):
        """Test price updates with set_price."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})
        api.set_price("005930", 73000)

        price = api.get_last_price("005930")

        assert price == 73000


class TestGetCurrentPrice:
    """Tests for get_current_price method."""

    def test_returns_price_data(self):
        """Test returns price data dict."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})

        data = api.get_current_price("005930")

        assert data is not None
        assert data["stck_prpr"] == 72000

    def test_returns_none_for_unknown(self):
        """Test returns None for unknown symbol."""
        api = MockKoreaInvestAPI()

        data = api.get_current_price("UNKNOWN")

        assert data is None


class TestOrderMethods:
    """Tests for order placement methods."""

    def test_place_limit_buy(self):
        """Test placing limit buy order."""
        api = MockKoreaInvestAPI()

        order_id = api.place_limit_buy("005930", 72000, 100)

        assert order_id is not None
        assert order_id.startswith("ORD")

    def test_place_limit_sell(self):
        """Test placing limit sell order."""
        api = MockKoreaInvestAPI()

        order_id = api.place_limit_sell("005930", 72000, 100)

        assert order_id is not None

    def test_place_market_buy(self):
        """Test placing market buy order."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})

        order_id = api.place_market_buy("005930", 100)

        assert order_id is not None
        # Market orders fill immediately
        order = api.get_order(order_id)
        assert order.filled_qty == 100

    def test_place_market_sell(self):
        """Test placing market sell order."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})

        order_id = api.place_market_sell("005930", 100)

        assert order_id is not None

    def test_order_failure(self):
        """Test order failure returns None."""
        api = MockKoreaInvestAPI(fail_orders=True)

        order_id = api.place_limit_buy("005930", 72000, 100)

        assert order_id is None

    def test_rate_limit_retry(self):
        """Test rate limit causes exception."""
        api = MockKoreaInvestAPI(fail_rate_limit=True)

        with pytest.raises(Exception):
            api.place_limit_buy("005930", 72000, 100)


class TestCancelOrder:
    """Tests for cancel_order method."""

    def test_cancel_working_order(self):
        """Test cancelling working order."""
        api = MockKoreaInvestAPI()
        order_id = api.place_limit_buy("005930", 72000, 100)

        result = api.cancel_order(order_id, 100)

        assert result is True
        order = api.get_order(order_id)
        assert order.status == "CANCELLED"

    def test_cancel_filled_order_fails(self):
        """Test cancelling filled order fails."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})
        order_id = api.place_market_buy("005930", 100)

        result = api.cancel_order(order_id, 100)

        assert result is False

    def test_cancel_unknown_order_fails(self):
        """Test cancelling unknown order fails."""
        api = MockKoreaInvestAPI()

        result = api.cancel_order("UNKNOWN_ORDER", 100)

        assert result is False


class TestModifyOrder:
    """Tests for modify_order method."""

    def test_modify_working_order(self):
        """Test modifying working order."""
        api = MockKoreaInvestAPI()
        order_id = api.place_limit_buy("005930", 72000, 100)

        result = api.modify_order(order_id, 73000, 150)

        assert result is True
        order = api.get_order(order_id)
        assert order.price == 73000
        assert order.qty == 150


class TestGetOrders:
    """Tests for get_orders method."""

    def test_get_working_orders(self):
        """Test getting working orders."""
        api = MockKoreaInvestAPI()
        api.place_limit_buy("005930", 72000, 100)
        api.place_limit_buy("000660", 130000, 50)

        orders_df = api.get_orders()

        assert len(orders_df) == 2

    def test_get_orders_empty(self):
        """Test getting orders when none exist."""
        api = MockKoreaInvestAPI()

        orders_df = api.get_orders()

        assert orders_df.empty


class TestBalanceMethods:
    """Tests for balance methods."""

    def test_get_acct_balance(self):
        """Test getting account balance."""
        api = MockKoreaInvestAPI(
            positions=[MockPosition("005930", 100, 70000)]
        )

        equity, df = api.get_acct_balance()

        assert equity == 100_000_000  # Default equity
        assert len(df) == 1

    def test_get_buyable_cash(self):
        """Test getting buyable cash."""
        api = MockKoreaInvestAPI()

        cash = api.get_buyable_cash()

        assert cash == 50_000_000  # Default buyable cash


class TestFillSimulation:
    """Tests for fill simulation."""

    def test_fill_limit_order(self):
        """Test filling limit order."""
        api = MockKoreaInvestAPI()
        order_id = api.place_limit_buy("005930", 72000, 100)

        result = api.fill_order(order_id)

        assert result is True
        order = api.get_order(order_id)
        assert order.status == "FILLED"

    def test_partial_fill(self):
        """Test partial fill."""
        api = MockKoreaInvestAPI()
        order_id = api.place_limit_buy("005930", 72000, 100)

        result = api.fill_order(order_id, fill_qty=50)

        assert result is True
        order = api.get_order(order_id)
        assert order.filled_qty == 50
        assert order.status == "PARTIAL"

    def test_fill_updates_position(self):
        """Test fill updates position."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})
        order_id = api.place_limit_buy("005930", 72000, 100)

        api.fill_order(order_id)

        position = api.get_position("005930")
        assert position is not None
        assert position.qty == 100


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_orders(self):
        """Test reset clears orders."""
        api = MockKoreaInvestAPI()
        api.place_limit_buy("005930", 72000, 100)

        api.reset()

        orders_df = api.get_orders()
        assert orders_df.empty

    def test_reset_clears_positions(self):
        """Test reset clears positions."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})
        api.place_market_buy("005930", 100)

        api.reset()

        position = api.get_position("005930")
        assert position is None
