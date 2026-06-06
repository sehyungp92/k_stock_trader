"""Tests for OMS client retry logic on transient connection errors."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from oms_client.client import OMSClient, _health_payload_ready
from oms.intent import Intent, IntentType, IntentStatus, IntentResult, Urgency, TimeHorizon, IntentConstraints, RiskPayload


def _make_intent(**overrides):
    defaults = dict(
        intent_type=IntentType.ENTER,
        strategy_id="GAMMA",
        symbol="005930",
        desired_qty=100,
        urgency=Urgency.LOW,
        time_horizon=TimeHorizon.SWING,
        constraints=IntentConstraints(),
        risk_payload=RiskPayload(entry_px=50000, stop_px=49000),
    )
    defaults.update(overrides)
    return Intent(**defaults)


class TestSubmitIntentRetry:
    """Tests for submit_intent retry on connection errors."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        """No retry needed when first attempt succeeds."""
        client = OMSClient("http://localhost:8000", strategy_id="GAMMA")
        intent = _make_intent()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "intent_id": "test-id",
            "status": "EXECUTED",
            "message": "ok",
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = await client.submit_intent(intent)
        assert result.status == IntentStatus.EXECUTED
        assert mock_session.post.call_count == 1
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["intent_id"] == intent.intent_id
        assert payload["idempotency_key"] == intent.idempotency_key
        await client.close()

    def test_health_payload_ready_rejects_semantic_degraded_or_unprotected_states(self):
        healthy_no_active_stops = {
            "status": "ok",
            "stop_protection_status": "ok",
            "unprotected_positions_count": 0,
            "active_stop_count": 0,
            "triggered_stop_count": 0,
            "stop_watcher_price_stale_count": 0,
            "idempotency_status": "ok",
        }
        healthy_active_stops = {
            **healthy_no_active_stops,
            "active_stop_count": 1,
            "stop_watcher_last_check_age_sec": 5.0,
        }
        assert _health_payload_ready(healthy_no_active_stops) is True
        assert _health_payload_ready(healthy_active_stops) is True
        assert _health_payload_ready({}) is False
        assert _health_payload_ready({"status": "error"}) is False
        assert _health_payload_ready({"status": "ok"}) is False
        assert _health_payload_ready({"status": "ok", "stop_protection_status": "ok"}) is False
        assert _health_payload_ready({"status": "ok", "stop_protection_status": "ok", "idempotency_status": "ok"}) is False
        assert _health_payload_ready({**healthy_no_active_stops, "unprotected_positions_count": None}) is False
        assert _health_payload_ready({**healthy_no_active_stops, "active_stop_count": 1}) is False
        assert _health_payload_ready({**healthy_active_stops, "stop_watcher_last_check_age_sec": 120.0}) is False
        assert _health_payload_ready({"status": "ok", "stop_protection_status": "degraded"}) is False
        assert _health_payload_ready({**healthy_no_active_stops, "unprotected_positions_count": 1}) is False
        assert _health_payload_ready({**healthy_no_active_stops, "idempotency_status": "ambiguous"}) is False

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        """Retries on transient connection error, succeeds on 2nd attempt."""
        client = OMSClient("http://localhost:8000", strategy_id="GAMMA")
        intent = _make_intent()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "intent_id": "test-id",
            "status": "EXECUTED",
            "message": "ok",
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        call_count = 0

        def post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Server disconnected")
            return mock_resp

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=post_side_effect)
        mock_session.closed = False

        # Patch _get_session to always return our mock
        async def fake_get_session():
            return mock_session

        client._get_session = fake_get_session
        # Patch sleep to avoid waiting
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.submit_intent(intent)

        assert result.status == IntentStatus.EXECUTED
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        """Returns REJECTED after exhausting all retries."""
        client = OMSClient("http://localhost:8000", strategy_id="GAMMA")
        intent = _make_intent()

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=ConnectionError("Server disconnected"))
        mock_session.closed = False

        async def fake_get_session():
            return mock_session

        client._get_session = fake_get_session
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.submit_intent(intent)

        assert result.status == IntentStatus.REJECTED
        assert "OMS unreachable" in result.message
        assert mock_session.post.call_count == client._SUBMIT_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_no_retry_on_oms_rejection(self):
        """OMS HTTP 200 with REJECTED status is not retried (intentional risk decision)."""
        client = OMSClient("http://localhost:8000", strategy_id="GAMMA")
        intent = _make_intent()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "intent_id": "test-id",
            "status": "REJECTED",
            "message": "Gross exposure would exceed 80%",
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = await client.submit_intent(intent)
        assert result.status == IntentStatus.REJECTED
        assert "Gross exposure" in result.message
        # Only 1 call ??no retry for OMS-level rejections
        assert mock_session.post.call_count == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_http_error(self):
        """HTTP 4xx/5xx errors are not retried (server responded)."""
        client = OMSClient("http://localhost:8000", strategy_id="GAMMA")
        intent = _make_intent()

        mock_resp = AsyncMock()
        mock_resp.status = 422
        mock_resp.text = AsyncMock(return_value="Validation error")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = await client.submit_intent(intent)
        assert result.status == IntentStatus.REJECTED
        assert "OMS error 422" in result.message
        assert mock_session.post.call_count == 1
        await client.close()
