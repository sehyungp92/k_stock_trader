"""Integration tests for OMS-Strategy interaction."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import time

from tests.mocks.mock_kis_api import MockKoreaInvestAPI
from tests.mocks.mock_oms_client import MockOMSClient, MockIntentResult, MockIntentStatus


class TestIntentSubmission:
    """Tests for intent submission via OMS."""

    @pytest.fixture
    def mock_oms(self):
        """Create mock OMS client."""
        return MockOMSClient(default_status="EXECUTED")

    @pytest.mark.asyncio
    async def test_enter_intent_submission(self, mock_oms):
        """Test ENTER intent is submitted correctly."""
        from oms.intent import Intent, IntentType, RiskPayload

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = await mock_oms.submit_intent(intent)

        assert result.status.name == "EXECUTED"
        assert len(mock_oms.submitted_intents) == 1
        assert mock_oms.submitted_intents[0].symbol == "005930"

    @pytest.mark.asyncio
    async def test_exit_intent_submission(self, mock_oms):
        """Test EXIT intent is submitted correctly."""
        from oms.intent import Intent, IntentType, RiskPayload

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
            risk_payload=RiskPayload(rationale_code="stop_hit"),
        )

        result = await mock_oms.submit_intent(intent)

        assert result.status.name == "EXECUTED"

    @pytest.mark.asyncio
    async def test_intent_rejection(self):
        """Test intent rejection handling."""
        mock_oms = MockOMSClient(fail_intents=True)

        from oms.intent import Intent, IntentType

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        result = await mock_oms.submit_intent(intent)

        assert result.status.name == "REJECTED"

    @pytest.mark.asyncio
    async def test_intent_deferral(self):
        """Test intent deferral handling."""
        mock_oms = MockOMSClient(defer_intents=True)

        from oms.intent import Intent, IntentType

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        result = await mock_oms.submit_intent(intent)

        assert result.status.name == "DEFERRED"


class TestIdempotency:
    """Tests for intent idempotency."""

    @pytest.fixture
    def mock_oms(self):
        """Create mock OMS client."""
        return MockOMSClient(default_status="EXECUTED")

    @pytest.mark.asyncio
    async def test_same_signal_hash_deduplicated(self, mock_oms):
        """Test same signal_hash produces same idempotency key."""
        from oms.intent import Intent, IntentType

        intent1 = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            signal_hash="break_1",
            idempotency_key="KMP:005930:ENTER:20240115:break_1",
        )

        intent2 = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            signal_hash="break_1",
            idempotency_key="KMP:005930:ENTER:20240115:break_1",
        )

        # Set response for first intent
        mock_oms.set_response(
            intent1.idempotency_key,
            status="EXECUTED",
            order_id="ORD001",
        )

        result1 = await mock_oms.submit_intent(intent1)
        result2 = await mock_oms.submit_intent(intent2)

        # Both should get same result
        assert result1.status.name == "EXECUTED"
        assert result2.status.name == "EXECUTED"

    @pytest.mark.asyncio
    async def test_different_signal_hash_not_deduplicated(self, mock_oms):
        """Test different signal_hash produces different keys."""
        from oms.intent import Intent, IntentType

        intent1 = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            signal_hash="break_1",
        )

        intent2 = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            signal_hash="break_2",  # Different hash
        )

        await mock_oms.submit_intent(intent1)
        await mock_oms.submit_intent(intent2)

        # Both should be submitted
        assert len(mock_oms.submitted_intents) == 2


class TestPositionVisibility:
    """Tests for position state visibility."""

    @pytest.fixture
    def mock_oms(self):
        """Create mock OMS client with positions."""
        oms = MockOMSClient()
        oms.set_position("005930", 100)
        oms.set_allocation("005930", "KMP", 100)
        return oms

    def test_get_position(self, mock_oms):
        """Test getting position from OMS."""
        qty = mock_oms.get_position("005930")
        assert qty == 100

    def test_get_allocation(self, mock_oms):
        """Test getting allocation from OMS."""
        qty = mock_oms.get_allocation("005930", "KMP")
        assert qty == 100

    def test_get_allocation_missing_strategy(self, mock_oms):
        """Test getting allocation for missing strategy."""
        qty = mock_oms.get_allocation("005930", "KPR")
        assert qty == 0

    def test_get_position_missing_symbol(self, mock_oms):
        """Test getting position for missing symbol."""
        qty = mock_oms.get_position("000660")
        assert qty == 0


class TestConcurrentStrategies:
    """Tests for concurrent strategy behavior."""

    @pytest.fixture
    def mock_oms(self):
        """Create mock OMS client."""
        return MockOMSClient(default_status="EXECUTED")

    @pytest.mark.asyncio
    async def test_multiple_strategies_same_symbol(self, mock_oms):
        """Test multiple strategies on same symbol."""
        from oms.intent import Intent, IntentType

        intent_kmp = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        intent_kpr = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KPR",
            symbol="005930",
            desired_qty=50,
        )

        result_kmp = await mock_oms.submit_intent(intent_kmp)
        result_kpr = await mock_oms.submit_intent(intent_kpr)

        # Both should succeed (in real OMS, one might be deferred)
        assert result_kmp.status.name == "EXECUTED"
        assert result_kpr.status.name == "EXECUTED"
        assert len(mock_oms.submitted_intents) == 2

    @pytest.mark.asyncio
    async def test_exit_while_entry_pending(self, mock_oms):
        """Test exit intent while entry pending."""
        from oms.intent import Intent, IntentType

        enter_intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        exit_intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        await mock_oms.submit_intent(enter_intent)
        result = await mock_oms.submit_intent(exit_intent)

        assert result.status.name == "EXECUTED"


class TestStrategySpecificBehavior:
    """Tests for strategy-specific OMS behavior."""

    @pytest.fixture
    def mock_oms(self):
        """Create mock OMS client."""
        return MockOMSClient(default_status="EXECUTED")

    @pytest.mark.asyncio
    async def test_kmp_high_urgency(self, mock_oms):
        """Test KMP submits with HIGH urgency."""
        from oms.intent import Intent, IntentType, Urgency

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            urgency=Urgency.HIGH,
        )

        await mock_oms.submit_intent(intent)

        submitted = mock_oms.get_last_intent()
        assert submitted.urgency.name == "HIGH"

    @pytest.mark.asyncio
    async def test_kpr_normal_urgency(self, mock_oms):
        """Test KPR submits with NORMAL urgency."""
        from oms.intent import Intent, IntentType, Urgency

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KPR",
            symbol="005930",
            desired_qty=100,
            urgency=Urgency.NORMAL,
        )

        await mock_oms.submit_intent(intent)

        submitted = mock_oms.get_last_intent()
        assert submitted.urgency.name == "NORMAL"

    @pytest.mark.asyncio
    async def test_intraday_time_horizon(self, mock_oms):
        """Test intraday time horizon."""
        from oms.intent import Intent, IntentType, TimeHorizon

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            time_horizon=TimeHorizon.INTRADAY,
        )

        await mock_oms.submit_intent(intent)

        submitted = mock_oms.get_last_intent()
        assert submitted.time_horizon.name == "INTRADAY"
