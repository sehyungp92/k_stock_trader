"""Integration tests for OMS-KIS API interaction."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from tests.mocks.mock_kis_api import MockKoreaInvestAPI
from oms.adapter import KISExecutionAdapter, AdapterResult, AdapterError


class TestAdapterRetryBehavior:
    """Tests for adapter retry on transient errors."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry on rate limit error."""
        mock_api = MagicMock()
        call_count = 0

        def side_effect(symbol, price, qty):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("rate limit")
            return "ORD001"

        mock_api.place_limit_buy = side_effect
        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
            max_retries=3,
        )

        assert result.success is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry on timeout error."""
        mock_api = MagicMock()
        call_count = 0

        def side_effect(symbol, price, qty):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("timeout")
            return "ORD001"

        mock_api.place_limit_buy = side_effect
        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
            max_retries=3,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        """Test no retry on permanent error."""
        mock_api = MagicMock()
        mock_api.place_limit_buy = MagicMock(side_effect=Exception("invalid symbol"))

        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order(
            symbol="INVALID",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
            max_retries=3,
        )

        assert result.success is False
        # Should only try once for permanent error
        assert mock_api.place_limit_buy.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test failure after max retries."""
        mock_api = MagicMock()
        mock_api.place_limit_buy = MagicMock(side_effect=Exception("rate limit"))

        adapter = KISExecutionAdapter(mock_api)

        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
            max_retries=2,
        )

        assert result.success is False
        assert result.error == AdapterError.TEMP_ERROR
        assert mock_api.place_limit_buy.call_count == 2


class TestOrderSubmission:
    """Tests for order submission via adapter."""

    @pytest.fixture
    def api(self):
        """Create mock API."""
        return MockKoreaInvestAPI(prices={"005930": 72000})

    @pytest.fixture
    def adapter(self, api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(api)

    @pytest.mark.asyncio
    async def test_market_buy_order(self, adapter, api):
        """Test market buy order submission."""
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
        )

        assert result.success is True
        assert result.order_id is not None

        # Verify order was placed
        order = api.get_order(result.order_id)
        assert order is not None
        assert order.side == "BUY"
        assert order.qty == 100

    @pytest.mark.asyncio
    async def test_market_sell_order(self, adapter, api):
        """Test market sell order submission."""
        result = await adapter.submit_order(
            symbol="005930",
            side="SELL",
            qty=100,
            order_type="MARKET",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_limit_buy_order(self, adapter, api):
        """Test limit buy order submission."""
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
        )

        assert result.success is True

        order = api.get_order(result.order_id)
        assert order.price == 72000

    @pytest.mark.asyncio
    async def test_stop_limit_order(self, adapter, api):
        """Test stop-limit order submission (simulated)."""
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="STOP_LIMIT",
            limit_price=72100,
            stop_price=72000,
        )

        assert result.success is True


class TestOrderCancellation:
    """Tests for order cancellation via adapter."""

    @pytest.fixture
    def api(self):
        """Create mock API."""
        return MockKoreaInvestAPI(prices={"005930": 72000})

    @pytest.fixture
    def adapter(self, api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(api)

    @pytest.mark.asyncio
    async def test_cancel_working_order(self, adapter, api):
        """Test cancelling working order."""
        # Place order
        submit_result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
        )

        # Cancel order
        cancel_result = await adapter.cancel_order(
            submit_result.order_id,
            "005930",
            100,
        )

        assert cancel_result.success is True

        # Verify order cancelled
        order = api.get_order(submit_result.order_id)
        assert order.status == "CANCELLED"

    @pytest.mark.asyncio
    async def test_cancel_filled_order_fails(self, adapter, api):
        """Test cancelling filled order fails."""
        # Place and fill market order
        submit_result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="MARKET",
        )

        # Try to cancel (should fail)
        cancel_result = await adapter.cancel_order(
            submit_result.order_id,
            "005930",
            100,
        )

        assert cancel_result.success is False


class TestPositionSync:
    """Tests for position synchronization via adapter."""

    @pytest.fixture
    def api(self):
        """Create mock API with positions."""
        from tests.mocks.mock_kis_api import MockPosition

        return MockKoreaInvestAPI(
            prices={"005930": 72000, "000660": 130000},
            positions=[
                MockPosition("005930", 100, 70000, 72000),
                MockPosition("000660", 50, 125000, 130000),
            ],
        )

    @pytest.fixture
    def adapter(self, api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(api)

    @pytest.mark.asyncio
    async def test_get_positions(self, adapter):
        """Test getting positions."""
        result = await adapter.get_positions()

        assert result.ok is True
        assert len(result.data) == 2

        # Find Samsung position
        samsung = next(p for p in result.data if p.symbol == "005930")
        assert samsung.qty == 100
        assert samsung.avg_price == 70000
        assert samsung.current_price == 72000

    @pytest.mark.asyncio
    async def test_get_orders(self, adapter, api):
        """Test getting open orders."""
        # Place some orders
        await adapter.submit_order("005930", "BUY", 100, "LIMIT", limit_price=72000)
        await adapter.submit_order("000660", "BUY", 50, "LIMIT", limit_price=130000)

        result = await adapter.get_orders()

        assert result.ok is True
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_get_account_info(self, adapter):
        """Test getting account info."""
        info = await adapter.get_account_info()

        assert "equity" in info
        assert "buyable_cash" in info
        assert info["equity"] > 0


class TestFillHandling:
    """Tests for fill event handling."""

    @pytest.fixture
    def api(self):
        """Create mock API."""
        return MockKoreaInvestAPI(prices={"005930": 72000})

    @pytest.fixture
    def adapter(self, api):
        """Create adapter with mock API."""
        return KISExecutionAdapter(api)

    @pytest.mark.asyncio
    async def test_detect_fill_via_get_orders(self, adapter, api):
        """Test detecting fill via order polling."""
        # Place limit order
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
        )

        # Simulate fill
        api.fill_order(result.order_id)

        # Order should no longer appear as working
        orders_result = await adapter.get_orders()
        assert orders_result.ok is True
        assert len([o for o in orders_result.data if o.order_id == result.order_id]) == 0

    @pytest.mark.asyncio
    async def test_partial_fill_detection(self, adapter, api):
        """Test detecting partial fill."""
        # Place limit order
        result = await adapter.submit_order(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type="LIMIT",
            limit_price=72000,
        )

        # Simulate partial fill
        api.fill_order(result.order_id, fill_qty=50)

        # Check order state
        order = api.get_order(result.order_id)
        assert order.filled_qty == 50
        assert order.status == "PARTIAL"
