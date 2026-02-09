"""
Mock OMSClient for testing strategies.

Simulates OMS responses without real OMS connection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock
import uuid


@dataclass
class MockIntentResult:
    """Mock result from OMS intent submission."""
    intent_id: str
    status: 'IntentStatus'
    message: str = ""
    order_id: Optional[str] = None
    modified_qty: Optional[int] = None
    cooldown_until: Optional[float] = None


class MockIntentStatus:
    """Mock IntentStatus enum."""
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"

    def __init__(self, value: str):
        self.name = value
        self.value = value


class MockOMSClient:
    """
    Mock OMS client for strategy testing.

    Features:
    - Configurable responses for submit_intent
    - Tracks all submitted intents
    - Simulates position/allocation queries
    """

    def __init__(
        self,
        default_status: str = "EXECUTED",
        default_order_id: Optional[str] = None,
        fail_intents: bool = False,
        defer_intents: bool = False,
    ):
        self.default_status = default_status
        self.default_order_id = default_order_id or "ORD00000001"
        self.fail_intents = fail_intents
        self.defer_intents = defer_intents

        # Track submitted intents
        self.submitted_intents: List[Any] = []

        # Mock positions (symbol -> qty)
        self.positions: Dict[str, int] = {}

        # Mock allocations (symbol -> {strategy_id -> qty})
        self.allocations: Dict[str, Dict[str, int]] = {}

        # Per-intent response overrides
        self._response_overrides: Dict[str, MockIntentResult] = {}

    async def submit_intent(self, intent: Any) -> MockIntentResult:
        """Submit intent and return mock result."""
        self.submitted_intents.append(intent)

        # Check for override
        if intent.idempotency_key in self._response_overrides:
            return self._response_overrides[intent.idempotency_key]

        # Failure modes
        if self.fail_intents:
            return MockIntentResult(
                intent_id=intent.intent_id,
                status=MockIntentStatus("REJECTED"),
                message="Mock rejection",
            )

        if self.defer_intents:
            return MockIntentResult(
                intent_id=intent.intent_id,
                status=MockIntentStatus("DEFERRED"),
                message="Mock deferral",
            )

        # Default success
        order_id = self._generate_order_id()
        return MockIntentResult(
            intent_id=intent.intent_id,
            status=MockIntentStatus(self.default_status),
            order_id=order_id,
        )

    def get_position(self, symbol: str) -> int:
        """Get position for symbol."""
        return self.positions.get(symbol, 0)

    def get_allocation(self, symbol: str, strategy_id: str) -> int:
        """Get allocation for strategy on symbol."""
        symbol_allocs = self.allocations.get(symbol, {})
        return symbol_allocs.get(strategy_id, 0)

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"ORD{uuid.uuid4().hex[:8].upper()}"

    # --- Test helper methods ---

    def set_response(
        self,
        idempotency_key: str,
        status: str,
        message: str = "",
        order_id: Optional[str] = None,
        modified_qty: Optional[int] = None,
    ) -> None:
        """Set response for specific intent."""
        self._response_overrides[idempotency_key] = MockIntentResult(
            intent_id="",  # Will be overwritten
            status=MockIntentStatus(status),
            message=message,
            order_id=order_id,
            modified_qty=modified_qty,
        )

    def set_position(self, symbol: str, qty: int) -> None:
        """Set position for symbol."""
        self.positions[symbol] = qty

    def set_allocation(self, symbol: str, strategy_id: str, qty: int) -> None:
        """Set allocation for strategy on symbol."""
        if symbol not in self.allocations:
            self.allocations[symbol] = {}
        self.allocations[symbol][strategy_id] = qty

    def get_intent_count(self, intent_type: Optional[str] = None) -> int:
        """Count submitted intents, optionally filtered by type."""
        if intent_type is None:
            return len(self.submitted_intents)
        return sum(
            1 for i in self.submitted_intents
            if i.intent_type.name == intent_type
        )

    def get_last_intent(self) -> Optional[Any]:
        """Get most recently submitted intent."""
        if not self.submitted_intents:
            return None
        return self.submitted_intents[-1]

    def clear_intents(self) -> None:
        """Clear submitted intents."""
        self.submitted_intents.clear()

    def reset(self) -> None:
        """Reset all state."""
        self.submitted_intents.clear()
        self.positions.clear()
        self.allocations.clear()
        self._response_overrides.clear()


def create_mock_oms_client(**kwargs) -> MockOMSClient:
    """Factory for creating mock OMS clients."""
    return MockOMSClient(**kwargs)


def create_async_mock_oms() -> AsyncMock:
    """Create an AsyncMock-based OMS client for advanced mocking."""
    mock = AsyncMock()

    # Default successful response
    mock.submit_intent.return_value = MockIntentResult(
        intent_id="test-intent-id",
        status=MockIntentStatus("EXECUTED"),
        order_id="ORD00000001",
    )

    return mock
