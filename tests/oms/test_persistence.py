"""Tests for OMS persistence order keying."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from oms.persistence import OMSPersistence
from oms.state import OrderStatus, WorkingOrder


class TestOMSPersistenceOrderKeying:
    """Tests for broker-order-ID to OMS-order-ID persistence mapping."""

    @pytest.mark.asyncio
    async def test_record_order_uses_broker_id_as_kis_order_id(self):
        """Broker order IDs should be stored in kis_order_id, not UUID columns."""
        persistence = OMSPersistence(dsn="postgres://test")
        persistence.pool = MagicMock()

        inserted_uuid = uuid.uuid4()
        persistence.pool.fetchval = AsyncMock(side_effect=[None, inserted_uuid])

        intent_id = str(uuid.uuid4())
        order = WorkingOrder(
            order_id="1234567890",
            symbol="005930",
            side="BUY",
            qty=100,
            price=72000,
            strategy_id="KMP",
            status=OrderStatus.WORKING,
        )

        oms_order_id = await persistence.record_order(order, intent_id=intent_id)

        assert oms_order_id == str(inserted_uuid)
        assert order.oms_order_id == str(inserted_uuid)

        insert_call = persistence.pool.fetchval.await_args_list[1]
        assert insert_call.args[10] == "1234567890"
        assert insert_call.args[12] == intent_id

    @pytest.mark.asyncio
    async def test_record_order_event_resolves_oms_order_id_from_broker_id(self):
        """Order events should resolve broker IDs back to the OMS UUID row."""
        persistence = OMSPersistence(dsn="postgres://test")
        persistence.pool = MagicMock()

        resolved_uuid = str(uuid.uuid4())
        persistence.pool.fetchval = AsyncMock(return_value=resolved_uuid)
        persistence.pool.execute = AsyncMock()

        await persistence.record_order_event(
            "ORDER_SUBMITTED",
            order_id="1234567890",
            strategy_id="KMP",
            symbol="005930",
        )

        execute_call = persistence.pool.execute.await_args
        assert execute_call.args[1] == resolved_uuid
