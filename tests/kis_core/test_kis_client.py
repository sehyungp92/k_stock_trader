"""Tests for KIS client methods."""

from datetime import date

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from tests.mocks.mock_kis_api import MockKoreaInvestAPI, MockPosition
from kis_core.kis_client import (
    KoreaInvestAPI,
    _get_circuit_breaker,
    _circuit_breaker_quote,
    _circuit_breaker_order,
    _circuit_breaker_investor,
)


class _FakeEnv:
    def get_full_config(self):
        return {
            "custtype": "P",
            "websocket_approval_key": "",
            "account_num": "12345678",
            "is_paper_trading": True,
            "htsid": "TEST",
            "using_url": "https://example.test",
        }


def _make_real_api() -> KoreaInvestAPI:
    return KoreaInvestAPI(_FakeEnv())


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

        result = api.place_limit_buy("005930", 72000, 100)

        assert result.success is True
        assert result.order_id is not None
        assert result.order_id.startswith("ORD")

    def test_place_limit_sell(self):
        """Test placing limit sell order."""
        api = MockKoreaInvestAPI()

        result = api.place_limit_sell("005930", 72000, 100)

        assert result.success is True
        assert result.order_id is not None

    def test_place_market_buy(self):
        """Test placing market buy order."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})

        result = api.place_market_buy("005930", 100)

        assert result.success is True
        assert result.order_id is not None
        # Market orders fill immediately
        order = api.get_order(result.order_id)
        assert order.filled_qty == 100

    def test_place_market_sell(self):
        """Test placing market sell order."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})

        result = api.place_market_sell("005930", 100)

        assert result.success is True
        assert result.order_id is not None

    def test_order_failure(self):
        """Test order failure returns OrderResult with success=False."""
        api = MockKoreaInvestAPI(fail_orders=True)

        result = api.place_limit_buy("005930", 72000, 100)

        assert result.success is False
        assert result.error_code == 'MOCK_REJECT'

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
        order_id = api.place_limit_buy("005930", 72000, 100).order_id

        result = api.cancel_order(order_id, 100)

        assert result is True
        order = api.get_order(order_id)
        assert order.status == "CANCELLED"

    def test_cancel_filled_order_fails(self):
        """Test cancelling filled order fails."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})
        order_id = api.place_market_buy("005930", 100).order_id

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
        order_id = api.place_limit_buy("005930", 72000, 100).order_id

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
        order_id = api.place_limit_buy("005930", 72000, 100).order_id

        result = api.fill_order(order_id)

        assert result is True
        order = api.get_order(order_id)
        assert order.status == "FILLED"

    def test_partial_fill(self):
        """Test partial fill."""
        api = MockKoreaInvestAPI()
        order_id = api.place_limit_buy("005930", 72000, 100).order_id

        result = api.fill_order(order_id, fill_qty=50)

        assert result is True
        order = api.get_order(order_id)
        assert order.filled_qty == 50
        assert order.status == "PARTIAL"

    def test_fill_updates_position(self):
        """Test fill updates position."""
        api = MockKoreaInvestAPI(prices={"005930": 72000})
        order_id = api.place_limit_buy("005930", 72000, 100).order_id

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


class TestCircuitBreakerRouting:
    """Tests for _get_circuit_breaker routing logic."""

    def test_investor_url_returns_investor_breaker(self):
        """inquire-investor should route to _circuit_breaker_investor, not quote."""
        cb = _get_circuit_breaker('/uapi/domestic-stock/v1/quotations/inquire-investor', False)
        assert cb is _circuit_breaker_investor

    def test_quote_url_returns_quote_breaker(self):
        """Regular quote endpoints should still use the quote breaker."""
        cb = _get_circuit_breaker('/uapi/domestic-stock/v1/quotations/inquire-price', False)
        assert cb is _circuit_breaker_quote

    def test_order_url_returns_order_breaker(self):
        """Trading endpoints should use the order breaker even for GET."""
        cb = _get_circuit_breaker('/uapi/domestic-stock/v1/trading/inquire-daily-ccld', False)
        assert cb is _circuit_breaker_order

    def test_post_request_returns_order_breaker(self):
        """Any POST request should use the order breaker."""
        cb = _get_circuit_breaker('/uapi/domestic-stock/v1/quotations/inquire-price', True)
        assert cb is _circuit_breaker_order


class TestResolveSymbol:
    def test_valid_6digit_code_passes_through(self):
        api = _make_real_api()
        api.get_current_price = MagicMock(return_value={"stck_prpr": 72000})

        with patch.object(api, "_get_symbol_lookup_cache", return_value={}):
            assert api.resolve_symbol("005930") == "005930"

    @patch("kis_core.kis_client.pykrx_stock")
    def test_valid_6digit_code_uses_listing_cache_before_quote(self, mock_pykrx_stock):
        api = _make_real_api()
        api.get_current_price = MagicMock(return_value={"stck_prpr": 72000})
        mock_pykrx_stock.get_market_ticker_list.return_value = ["005930"]
        mock_pykrx_stock.get_market_ticker_name.return_value = "\uc0bc\uc131\uc804\uc790"

        with patch.object(api, "_get_symbol_lookup_target_date", return_value=date(2026, 4, 24)):
            assert api.resolve_symbol("005930") == "005930"
        api.get_current_price.assert_not_called()

    @patch("kis_core.kis_client.pykrx_stock")
    def test_6digit_code_falls_back_to_listing_cache_when_quote_unavailable(self, mock_pykrx_stock):
        api = _make_real_api()
        api.get_current_price = MagicMock(return_value=None)
        mock_pykrx_stock.get_market_ticker_list.return_value = ["005930"]
        mock_pykrx_stock.get_market_ticker_name.return_value = "\uc0bc\uc131\uc804\uc790"

        with patch.object(api, "_get_symbol_lookup_target_date", return_value=date(2026, 4, 24)):
            assert api.resolve_symbol("005930") == "005930"

    @patch("kis_core.kis_client.pykrx_stock")
    def test_exact_korean_name_resolves(self, mock_pykrx_stock):
        api = _make_real_api()
        mock_pykrx_stock.get_market_ticker_list.return_value = ["005930", "000660"]
        mock_pykrx_stock.get_market_ticker_name.side_effect = lambda code: {
            "005930": "\uc0bc\uc131\uc804\uc790",
            "000660": "\uc8fc\uc2dd\ud68c\uc0ac SK\ud558\uc774\ub2c9\uc2a4",
        }[code]

        with patch.object(api, "_get_symbol_lookup_target_date", return_value=date(2026, 4, 24)):
            assert api.resolve_symbol("\uc0bc\uc131\uc804\uc790") == "005930"

    @patch("kis_core.kis_client.pykrx_stock")
    def test_normalized_company_name_variant_resolves(self, mock_pykrx_stock):
        api = _make_real_api()
        mock_pykrx_stock.get_market_ticker_list.return_value = ["000660"]
        mock_pykrx_stock.get_market_ticker_name.return_value = "\uc8fc\uc2dd\ud68c\uc0ac SK\ud558\uc774\ub2c9\uc2a4"

        with patch.object(api, "_get_symbol_lookup_target_date", return_value=date(2026, 4, 24)):
            assert api.resolve_symbol("(\uc8fc) SK \ud558\uc774\ub2c9\uc2a4 Co., Ltd.") == "000660"

    @patch("kis_core.kis_client.pykrx_stock")
    def test_market_specific_fallback_is_used_when_all_market_query_is_empty(self, mock_pykrx_stock):
        api = _make_real_api()
        mock_pykrx_stock.get_market_ticker_list.side_effect = lambda date, market: {
            "ALL": [],
            "KOSPI": ["005930"],
            "KOSDAQ": [],
            "KONEX": [],
        }[market]
        mock_pykrx_stock.get_market_ticker_name.return_value = "\uc0bc\uc131\uc804\uc790"

        with patch.object(api, "_get_symbol_lookup_target_date", return_value=date(2026, 4, 24)):
            assert api.resolve_symbol("\uc0bc\uc131\uc804\uc790") == "005930"

    @patch("kis_core.kis_client.pykrx_stock")
    def test_unknown_name_returns_none(self, mock_pykrx_stock):
        api = _make_real_api()
        mock_pykrx_stock.get_market_ticker_list.return_value = ["005930"]
        mock_pykrx_stock.get_market_ticker_name.return_value = "\uc0bc\uc131\uc804\uc790"

        with patch.object(api, "_get_symbol_lookup_target_date", return_value=date(2026, 4, 24)):
            assert api.resolve_symbol("\uc5c6\ub294\ud68c\uc0ac") is None

    @patch("kis_core.kis_client.pykrx_stock")
    def test_empty_listing_cache_fails_safely(self, mock_pykrx_stock):
        api = _make_real_api()
        mock_pykrx_stock.get_market_ticker_list.return_value = []

        with patch.object(api, "_get_symbol_lookup_target_date", return_value=date(2026, 4, 24)):
            assert api.resolve_symbol("\uc0bc\uc131\uc804\uc790") is None
