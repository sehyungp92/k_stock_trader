"""Tests for OMS adapter module."""

from datetime import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from zoneinfo import ZoneInfo

from kis_core.kis_client import OrderResult
from oms.adapter import (
    KISExecutionAdapter,
    AdapterResult,
    AdapterError,
    BrokerOrder,
    BrokerPosition,
    BrokerFill,
    BrokerQueryResult,
)


class TestAdapterError:
    """Tests for AdapterError enum."""

    def test_all_errors_defined(self):
        """Test all adapter errors are defined."""
        assert AdapterError.NONE
        assert AdapterError.RATE_LIMIT
        assert AdapterError.TEMP_ERROR
        assert AdapterError.REJECTED_INVALID
        assert AdapterError.REJECTED_RISK
        assert AdapterError.UNKNOWN


class TestAdapterResult:
    """Tests for AdapterResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = AdapterResult(success=True, order_id="ORD001")

        assert result.success is True
        assert result.order_id == "ORD001"
        assert result.error == AdapterError.NONE
        assert result.message == ""

    def test_failure_result(self):
        """Test failure result."""
        result = AdapterResult(
            success=False,
            error=AdapterError.RATE_LIMIT,
            message="Rate limit exceeded",
        )

        assert result.success is False
        assert result.order_id is None
        assert result.error == AdapterError.RATE_LIMIT


class TestBrokerOrder:
    """Tests for BrokerOrder dataclass."""

    def test_broker_order(self):
        """Test BrokerOrder creation."""
        order = BrokerOrder(
            order_id="ORD001",
            symbol="005930",
            side="BUY",
            qty=100,
            filled_qty=50,
            price=72000,
            status="WORKING",
            created_at="09:30:00",
        )

        assert order.order_id == "ORD001"
        assert order.symbol == "005930"
        assert order.filled_qty == 50


class TestBrokerPosition:
    """Tests for BrokerPosition dataclass."""

    def test_broker_position(self):
        """Test BrokerPosition creation."""
        position = BrokerPosition(
            symbol="005930",
            qty=100,
            avg_price=70000,
            current_price=72000,
            pnl=2.86,
        )

        assert position.symbol == "005930"
        assert position.qty == 100
        assert position.pnl == 2.86


class TestKISExecutionAdapterSubmitOrder:
    """Tests for KISExecutionAdapter.submit_order method."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        api = MagicMock()
        api.place_market_buy.return_value = OrderResult(success=True, order_id="ORD001")
        api.place_market_sell.return_value = OrderResult(success=True, order_id="ORD002")
        api.place_limit_buy.return_value = OrderResult(success=True, order_id="ORD003")
        api.place_limit_sell.return_value = OrderResult(success=True, order_id="ORD004")
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_market_buy(self, adapter, mock_api):
        """Test market buy order."""
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
        )

        assert result.success is True
        assert result.order_id == "ORD001"
        mock_api.place_market_buy.assert_called_once_with("005930", 100)

    @pytest.mark.asyncio
    async def test_market_sell(self, adapter, mock_api):
        """Test market sell order."""
        result = await adapter.submit_order(
            symbol="005930",
            side="SELL",
            qty=100,
            order_type="MARKET",
        )

        assert result.success is True
        assert result.order_id == "ORD002"
        mock_api.place_market_sell.assert_called_once_with("005930", 100)

    @pytest.mark.asyncio
    async def test_limit_buy(self, adapter, mock_api):
        """Test limit buy order."""
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
        )

        assert result.success is True
        assert result.order_id == "ORD003"
        mock_api.place_limit_buy.assert_called_once_with("005930", 72000, 100)

    @pytest.mark.asyncio
    async def test_limit_sell(self, adapter, mock_api):
        """Test limit sell order."""
        result = await adapter.submit_order(
            symbol="005930",
            side="SELL",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
        )

        assert result.success is True
        assert result.order_id == "ORD004"
        mock_api.place_limit_sell.assert_called_once_with("005930", 72000, 100)

    @pytest.mark.asyncio
    async def test_marketable_limit(self, adapter, mock_api):
        """Test marketable limit order (treated as LIMIT)."""
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKETABLE_LIMIT",
            limit_price=72100,
        )

        assert result.success is True
        mock_api.place_limit_buy.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_auction_maps_to_bounded_limit(self, adapter, mock_api, monkeypatch):
        """KRX close-auction style maps to a bounded limit order."""
        monkeypatch.setattr(adapter, "_is_order_session_open", lambda now=None: True)
        result = await adapter.submit_order(
            symbol="005930",
            side="SELL",
            qty=100,
            order_type="CLOSE_AUCTION",
            limit_price=72000,
        )

        assert result.success is True
        mock_api.place_limit_sell.assert_called_once_with("005930", 72000, 100)

    @pytest.mark.asyncio
    async def test_close_auction_requires_limit(self, adapter, monkeypatch):
        """Close auction is bounded; unsupported fake market-on-close is rejected."""
        monkeypatch.setattr(adapter, "_is_order_session_open", lambda now=None: True)
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="CLOSE_AUCTION",
        )

        assert result.success is False
        assert result.error == AdapterError.REJECTED_INVALID

    @pytest.mark.asyncio
    async def test_stop_limit_simulated(self, adapter, mock_api):
        """Test stop-limit order (simulated as limit)."""
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="STOP_LIMIT",
            limit_price=72100,
            stop_price=72000,
        )

        assert result.success is True
        mock_api.place_limit_buy.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_order_type_rejected(self, adapter):
        """Test unknown order type is rejected."""
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="UNKNOWN_TYPE",
        )

        assert result.success is False
        assert result.error == AdapterError.REJECTED_INVALID

    @pytest.mark.asyncio
    async def test_api_returns_none_rejected(self, adapter, mock_api):
        """Test API returning failure OrderResult is treated as rejection."""
        mock_api.place_market_buy.return_value = OrderResult(
            success=False, error_code='MOCK01', error_message='Test rejection'
        )

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
        )

        assert result.success is False
        assert result.error == AdapterError.REJECTED_INVALID
        assert "MOCK01" in result.message


class TestKISExecutionAdapterRetry:
    """Tests for KISExecutionAdapter retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry on rate limit error."""
        mock_api = MagicMock()
        call_count = 0

        def mock_buy(symbol, qty):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limit exceeded")
            return OrderResult(success=True, order_id="ORD001")

        mock_api.place_market_buy.side_effect = mock_buy
        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
            max_retries=3,
        )

        assert result.success is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Timeouts fail closed to avoid duplicate live orders."""
        mock_api = MagicMock()
        call_count = 0

        def mock_buy(symbol, qty):
            nonlocal call_count
            call_count += 1
            raise Exception("timeout error")

        mock_api.place_market_buy.side_effect = mock_buy
        adapter = KISExecutionAdapter(mock_api)
        adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(ok=True, data=[]))

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
            max_retries=3,
        )

        assert result.success is False
        assert result.error == AdapterError.TEMP_ERROR
        assert "ambiguous after timeout" in result.message.lower()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_binding_flag_fails_closed_on_multiple_unknown_matches(self):
        """Even when the experimental binding flag is enabled, ambiguous matches must not fresh-submit."""
        mock_api = MagicMock()
        mock_api.place_market_buy.side_effect = [
            Exception("temporary broker failure"),
            OrderResult(success=True, order_id="ORD-FRESH"),
        ]
        adapter = KISExecutionAdapter(mock_api)
        adapter.retry_bind_open_order_on_ambiguous_submit = True
        adapter._is_order_session_open = MagicMock(return_value=True)
        adapter.get_orders = AsyncMock(
            return_value=BrokerQueryResult(
                ok=True,
                data=[
                    BrokerOrder("ORD-1", "005930", "BUY", 100, 0, 100.0, "WORKING", "09:00:01"),
                    BrokerOrder("ORD-2", "005930", "BUY", 100, 0, 100.0, "WORKING", "09:00:02"),
                ],
            )
        )

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
            max_retries=2,
        )

        assert result.success is False
        assert result.error == AdapterError.TEMP_ERROR
        assert "Ambiguous retry match" in result.message
        assert mock_api.place_market_buy.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test failure when max retries exhausted."""
        mock_api = MagicMock()
        mock_api.place_market_buy.side_effect = Exception("rate limit")

        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
            max_retries=2,
        )

        assert result.success is False
        assert result.error == AdapterError.TEMP_ERROR
        assert mock_api.place_market_buy.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test non-retryable error fails immediately."""
        mock_api = MagicMock()
        mock_api.place_market_buy.side_effect = Exception("invalid symbol")

        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
            max_retries=3,
        )

        assert result.success is False
        # Should only try once for non-retryable error
        assert mock_api.place_market_buy.call_count == 1


class TestKISExecutionAdapterCancelOrder:
    """Tests for KISExecutionAdapter.cancel_order method."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        api = MagicMock()
        api.cancel_order.return_value = True
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_cancel_success(self, adapter, mock_api):
        """Test successful cancel."""
        result = await adapter.cancel_order("ORD001", "005930", 100)

        assert result.success is True
        mock_api.cancel_order.assert_called_once_with("ORD001", 100)

    @pytest.mark.asyncio
    async def test_cancel_failure(self, adapter, mock_api):
        """Test cancel failure."""
        mock_api.cancel_order.return_value = False

        result = await adapter.cancel_order("ORD001", "005930", 100)

        assert result.success is False
        assert result.error == AdapterError.REJECTED_INVALID

    @pytest.mark.asyncio
    async def test_cancel_exception(self, adapter, mock_api):
        """Test cancel with exception."""
        mock_api.cancel_order.side_effect = Exception("API error")

        result = await adapter.cancel_order("ORD001", "005930", 100)

        assert result.success is False
        assert result.error == AdapterError.TEMP_ERROR


class TestKISExecutionAdapterGetOrders:
    """Tests for KISExecutionAdapter.get_orders method."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        import pandas as pd

        api = MagicMock()
        df = pd.DataFrame({
            "종목코드": ["005930", "000660"],
            "주문수량": [100, 50],
            "주문가능수량": [50, 50],
            "주문가격": [72000, 130000],
            "매도매수구분코드": ["02", "01"],
            "시간": ["09:30:00", "09:31:00"],
        })
        df.index = ["ORD001", "ORD002"]
        api.get_orders.return_value = df
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_get_orders(self, adapter):
        """Test getting open orders."""
        result = await adapter.get_orders()

        assert result.ok is True
        assert len(result.data) == 2
        assert result.data[0].order_id == "ORD001"
        assert result.data[0].symbol == "005930"
        assert result.data[0].side == "BUY"
        assert result.data[0].filled_qty == 50
        assert result.data[1].side == "SELL"

    @pytest.mark.asyncio
    async def test_get_orders_normalizes_adapter_reconciliation_evidence(self, mock_api):
        """KIS dataframe rows must expose reconciliation evidence through BrokerOrder."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "pdno": ["005930"],
                "ord_qty": [10],
                "psbl_qty": [10],
                "ord_unpr": [72000],
                "sll_buy_dvsn_cd": ["02"],
                "ord_tmd": ["093001"],
                "ord_dt": ["20260605"],
                "ord_gno_brno": ["001"],
                "ord_dvsn": ["00"],
                "submit_ref": ["OMS-submit-ref"],
            },
            index=["ORD-KIS"],
        )
        mock_api.get_orders.return_value = df
        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.get_orders()

        assert result.ok is True
        assert len(result.data) == 1
        order = result.data[0]
        assert order.order_id == "ORD-KIS"
        assert order.symbol == "005930"
        assert order.side == "BUY"
        assert order.qty == 10
        assert order.filled_qty == 0
        assert order.price == 72000.0
        assert order.created_at == "2026-06-05T09:30:01+09:00"
        assert order.created_ts is not None
        assert order.order_type == "LIMIT"
        assert order.submit_ref == "OMS-submit-ref"

    @pytest.mark.asyncio
    async def test_get_orders_unknown_side_is_absent_evidence_not_buy(self, mock_api):
        import pandas as pd

        df = pd.DataFrame(
            {
                "pdno": ["005930", "000660"],
                "ord_qty": [10, 20],
                "psbl_qty": [10, 20],
                "ord_unpr": [72000, 130000],
                "sll_buy_dvsn_cd": ["", "99"],
                "ord_tmd": ["093001", "093002"],
                "ord_dt": ["20260605", "20260605"],
            },
            index=["ORD-MISSING-SIDE", "ORD-UNKNOWN-SIDE"],
        )
        mock_api.get_orders.return_value = df
        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.get_orders()

        assert result.ok is True
        assert [order.side for order in result.data] == ["", ""]

    @pytest.mark.asyncio
    async def test_get_orders_empty(self, adapter, mock_api):
        """Test getting orders when none exist."""
        mock_api.get_orders.return_value = None

        result = await adapter.get_orders()

        assert result.ok is True
        assert result.data == []

    @pytest.mark.asyncio
    async def test_get_orders_exception(self, adapter, mock_api):
        """Test getting orders with exception returns error result."""
        mock_api.get_orders.side_effect = Exception("API error")

        result = await adapter.get_orders()

        assert result.ok is False
        assert result.data == []
        assert "API error" in result.error_message


class TestKISExecutionAdapterGetPositions:
    """Tests for KISExecutionAdapter.get_positions method."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        import pandas as pd

        api = MagicMock()
        df = pd.DataFrame({
            "종목코드": ["005930", "000660"],
            "보유수량": [100, 50],
            "매입단가": [70000, 125000],
            "현재가": [72000, 130000],
            "수익률": [2.86, 4.0],
        })
        api.get_acct_balance.return_value = (100_000_000, df)
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_get_positions(self, adapter):
        """Test getting positions."""
        result = await adapter.get_positions()

        assert result.ok is True
        assert len(result.data) == 2
        assert result.data[0].symbol == "005930"
        assert result.data[0].qty == 100
        assert result.data[0].avg_price == 70000
        assert result.data[0].current_price == 72000
        assert result.data[0].pnl == 2.86

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, adapter, mock_api):
        """Test getting positions when none exist."""
        import pandas as pd
        mock_api.get_acct_balance.return_value = (100_000_000, pd.DataFrame())

        result = await adapter.get_positions()

        assert result.ok is True
        assert result.data == []


class TestKISExecutionAdapterGetAccountInfo:
    """Tests for KISExecutionAdapter.get_account_info method."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        import pandas as pd

        api = MagicMock()
        df = pd.DataFrame({
            "종목코드": ["005930"],
            "보유수량": [100],
        })
        api.get_acct_balance.return_value = (100_000_000, df)
        api.get_buyable_cash.return_value = 50_000_000
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_get_account_info(self, adapter):
        """Test getting account info."""
        info = await adapter.get_account_info()

        assert info["equity"] == 100_000_000
        assert info["buyable_cash"] == 50_000_000
        assert info["positions_count"] == 1

    @pytest.mark.asyncio
    async def test_get_account_info_exception(self, adapter, mock_api):
        """Test getting account info with exception raises to caller."""
        mock_api.get_acct_balance.side_effect = Exception("API error")

        import pytest
        with pytest.raises(Exception, match="API error"):
            await adapter.get_account_info()


class TestBrokerFillCreation:
    """Tests for BrokerFill dataclass instantiation."""

    def test_broker_fill_creation(self):
        """Test BrokerFill creation with all fields."""
        fill = BrokerFill(
            order_id="ORD001",
            symbol="005930",
            side="BUY",
            qty=100,
            price=72000,
            timestamp=1234567890.0,
        )
        assert fill.order_id == "ORD001"
        assert fill.symbol == "005930"
        assert fill.side == "BUY"
        assert fill.qty == 100
        assert fill.price == 72000
        assert fill.timestamp == 1234567890.0

    def test_broker_fill_sell_side(self):
        """Test BrokerFill with SELL side."""
        fill = BrokerFill(
            order_id="ORD002",
            symbol="000660",
            side="SELL",
            qty=50,
            price=130000,
            timestamp=1234567891.0,
        )
        assert fill.side == "SELL"
        assert fill.qty == 50


class TestKISExecutionAdapterStopLimitSell:
    """Tests for STOP_LIMIT with SELL side."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        api = MagicMock()
        api.place_limit_sell.return_value = OrderResult(success=True, order_id="ORD004")
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_stop_limit_sell(self, adapter, mock_api):
        """Test stop-limit sell order (simulated as limit sell)."""
        result = await adapter.submit_order(
            symbol="005930",
            side="SELL",
            qty=100,
            order_type="STOP_LIMIT",
            limit_price=72000,
            stop_price=72100,
        )
        assert result.success is True
        mock_api.place_limit_sell.assert_called_once()


class TestKISExecutionAdapterCancelWithBranch:
    """Tests for cancel_order with branch parameter."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        api = MagicMock()
        api.cancel_order.return_value = True
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_cancel_with_branch(self, adapter, mock_api):
        """Test cancel order passes branch as order_branch kwarg."""
        result = await adapter.cancel_order("ORD001", "005930", 100, branch="0001")
        assert result.success is True
        mock_api.cancel_order.assert_called_once_with("ORD001", 100, order_branch="0001")

    @pytest.mark.asyncio
    async def test_cancel_without_branch(self, adapter, mock_api):
        """Test cancel order without branch does not pass order_branch."""
        result = await adapter.cancel_order("ORD001", "005930", 100)
        assert result.success is True
        mock_api.cancel_order.assert_called_once_with("ORD001", 100)


class TestKISExecutionAdapterGetAccountInfoBuyableNone:
    """Tests for get_account_info when buyable_cash is None."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        import pandas as pd

        api = MagicMock()
        df = pd.DataFrame({
            "종목코드": ["005930"],
            "보유수량": [100],
        })
        api.get_acct_balance.return_value = (100_000_000, df)
        api.get_buyable_cash.return_value = None
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_get_account_info_buyable_none(self, adapter, mock_api):
        """Test buyable_cash defaults to 0 when API returns None."""
        info = await adapter.get_account_info()
        assert info["buyable_cash"] == 0
        assert info["equity"] == 100_000_000
        assert info["positions_count"] == 1


class TestKISExecutionAdapterGetBalanceSnapshot:
    """Tests for KISExecutionAdapter.get_balance_snapshot method."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        import pandas as pd

        api = MagicMock()
        df = pd.DataFrame({
            "종목코드": ["005930", "000660"],
            "보유수량": [100, 50],
            "매입단가": [70000, 125000],
            "현재가": [72000, 130000],
            "수익률": [2.86, 4.0],
        })
        api.get_acct_balance.return_value = (100_000_000, df)
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_returns_positions_and_equity(self, adapter):
        """Test get_balance_snapshot returns both positions and equity."""
        positions_result, equity = await adapter.get_balance_snapshot()

        assert positions_result.ok is True
        assert len(positions_result.data) == 2
        assert positions_result.data[0].symbol == "005930"
        assert positions_result.data[0].qty == 100
        assert equity == 100_000_000

    @pytest.mark.asyncio
    async def test_empty_positions(self, adapter, mock_api):
        """Test get_balance_snapshot with no positions."""
        import pandas as pd
        mock_api.get_acct_balance.return_value = (100_000_000, pd.DataFrame())

        positions_result, equity = await adapter.get_balance_snapshot()

        assert positions_result.ok is True
        assert positions_result.data == []
        assert equity == 100_000_000

    @pytest.mark.asyncio
    async def test_api_failure_returns_error(self, adapter, mock_api):
        """Test get_balance_snapshot returns error on API failure."""
        mock_api.get_acct_balance.side_effect = Exception("API error")

        positions_result, equity = await adapter.get_balance_snapshot()

        assert positions_result.ok is False
        assert equity is None
        assert "API error" in positions_result.error_message

    @pytest.mark.asyncio
    async def test_single_api_call(self, adapter, mock_api):
        """Test only one get_acct_balance call is made."""
        await adapter.get_balance_snapshot()

        mock_api.get_acct_balance.assert_called_once()


class TestKISExecutionAdapterGetBuyableCash:
    """Tests for KISExecutionAdapter.get_buyable_cash method."""

    @pytest.fixture
    def mock_api(self):
        """Create mock KIS API."""
        api = MagicMock()
        api.get_buyable_cash.return_value = 50_000_000
        return api

    @pytest.fixture
    def adapter(self, mock_api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(mock_api)

    @pytest.mark.asyncio
    async def test_returns_buyable_cash(self, adapter):
        """Test get_buyable_cash returns value."""
        result = await adapter.get_buyable_cash()
        assert result == 50_000_000

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self, adapter, mock_api):
        """Test get_buyable_cash returns None on API failure."""
        mock_api.get_buyable_cash.side_effect = Exception("API error")

        result = await adapter.get_buyable_cash()
        assert result is None


class TestMarketClosedGuard:
    """Tests for market closed (weekend/holiday) order rejection."""

    @pytest.mark.asyncio
    async def test_rejects_on_non_trading_day(self, mock_trading_calendar_for_adapter):
        """Test orders are rejected when market is closed."""
        mock_trading_calendar_for_adapter.return_value.is_trading_day.return_value = False
        mock_api = MagicMock()
        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order("005930", "BUY", 100, "MARKET")

        assert result.success is False
        assert result.error == AdapterError.REJECTED_INVALID
        assert "Market closed" in result.message

    @pytest.mark.asyncio
    async def test_allows_on_trading_day(self, mock_trading_calendar_for_adapter):
        """Test orders proceed when market is open."""
        mock_trading_calendar_for_adapter.return_value.is_trading_day.return_value = True
        mock_api = MagicMock()
        mock_api.place_market_buy.return_value = OrderResult(
            success=True, order_id="ORD001"
        )
        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order("005930", "BUY", 100, "MARKET")

        assert result.success is True
        assert result.order_id == "ORD001"
        mock_api.place_market_buy.assert_called_once_with("005930", 100)

    @pytest.mark.asyncio
    async def test_rejects_outside_trading_hours(self, mock_trading_calendar_for_adapter):
        """Test weekday after-hours orders are rejected before hitting KIS."""
        mock_trading_calendar_for_adapter.return_value.is_trading_day.return_value = True
        mock_api = MagicMock()
        adapter = KISExecutionAdapter(mock_api)
        adapter._now_kst = MagicMock(
            return_value=datetime(2026, 4, 24, 16, 0, tzinfo=ZoneInfo("Asia/Seoul"))
        )

        result = await adapter.submit_order("005930", "BUY", 100, "MARKET")

        assert result.success is False
        assert result.error == AdapterError.REJECTED_INVALID
        assert "Market closed" in result.message
        mock_api.place_market_buy.assert_not_called()


class TestStopCapabilities:
    def test_native_stop_support_is_unverified_by_default(self):
        adapter = KISExecutionAdapter(MagicMock())

        snapshot = adapter.stop_capabilities_snapshot()

        assert adapter.supports_native_stop("005930") is False
        assert snapshot["broker_native_stop_verified_at"] is None
        assert snapshot["broker_native_stop_status"] == "unverified"

    @pytest.mark.asyncio
    async def test_submit_stop_order_fails_closed_when_unverified(self):
        adapter = KISExecutionAdapter(MagicMock())

        result = await adapter.submit_stop_order(symbol="005930", side="SELL", qty=10, stop_price=95.0)

        assert result.success is False
        assert result.error == AdapterError.REJECTED_INVALID
        assert "not paper-verified" in result.message
