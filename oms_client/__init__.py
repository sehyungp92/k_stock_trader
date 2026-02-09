"""OMS Client - HTTP client for strategies to connect to OMS service."""

from .client import OMSClient, PositionInfo, AllocationInfo, AccountState

# Re-export Intent domain types directly from oms.intent (not oms.__init__ which imports other modules)
from oms.intent import (
    Intent,
    IntentType,
    IntentStatus,
    IntentResult,
    Urgency,
    TimeHorizon,
    IntentConstraints,
    RiskPayload,
)

__all__ = [
    # Client
    "OMSClient",
    "PositionInfo",
    "AllocationInfo",
    "AccountState",
    # Intent types (re-exported from oms.intent)
    "Intent",
    "IntentType",
    "IntentStatus",
    "IntentResult",
    "Urgency",
    "TimeHorizon",
    "IntentConstraints",
    "RiskPayload",
]
