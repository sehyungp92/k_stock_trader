"""
Intent API: The ONLY interface strategies use to interact with OMS.
Strategies emit Intents; OMS handles execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional
import time
import uuid


class IntentType(Enum):
    ENTER = auto()
    REDUCE = auto()
    EXIT = auto()
    SET_TARGET = auto()
    CANCEL_ORDERS = auto()
    MODIFY_RISK = auto()
    FLATTEN = auto()


class Urgency(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()


class TimeHorizon(Enum):
    INTRADAY = auto()
    SWING = auto()


class IntentStatus(Enum):
    PENDING = auto()
    ACCEPTED = auto()
    APPROVED = auto()
    MODIFIED = auto()
    REJECTED = auto()
    DEFERRED = auto()
    EXECUTED = auto()
    CANCELLED = auto()


@dataclass
class IntentConstraints:
    """Execution constraints for an intent."""
    max_slippage_bps: Optional[float] = None
    max_spread_bps: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    expiry_ts: Optional[float] = None  # Unix epoch seconds


@dataclass
class RiskPayload:
    """Risk metadata for position management."""
    entry_px: Optional[float] = None
    stop_px: Optional[float] = None
    hard_stop_px: Optional[float] = None
    rationale_code: str = ""
    confidence: str = "YELLOW"


def _kst_trade_date() -> str:
    """Get current KST date string for idempotency keys."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")


@dataclass
class Intent:
    """
    Core intent object emitted by strategies.

    Immutable once created. OMS processes and returns IntentResult.
    """
    intent_type: IntentType
    strategy_id: str
    symbol: str

    # Quantity (one of these)
    desired_qty: Optional[int] = None
    target_qty: Optional[int] = None

    urgency: Urgency = Urgency.NORMAL
    time_horizon: TimeHorizon = TimeHorizon.INTRADAY

    constraints: IntentConstraints = field(default_factory=IntentConstraints)
    risk_payload: RiskPayload = field(default_factory=RiskPayload)

    # Deterministic dedup: strategies should set this for ENTER intents
    signal_hash: Optional[str] = None

    # Auto-generated
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: Optional[str] = None
    timestamp: float = field(default_factory=time.time)  # Epoch seconds

    def __post_init__(self):
        # Normalize strategy ID (case-sensitive match with RiskConfig budgets)
        self.strategy_id = self.strategy_id.upper().strip()

        if self.idempotency_key is None:
            trade_date = _kst_trade_date()
            qty_part = self.desired_qty or self.target_qty or 0
            if self.intent_type == IntentType.ENTER:
                # One entry per strategy:symbol:date:signal:qty — deterministic
                suffix = self.signal_hash or self.risk_payload.rationale_code or "default"
            elif self.intent_type in (IntentType.EXIT, IntentType.REDUCE, IntentType.FLATTEN):
                # One exit per strategy:symbol:date:reason:qty
                suffix = self.risk_payload.rationale_code or "manual"
            else:
                # Operational intents (CANCEL_ORDERS, MODIFY_RISK, SET_TARGET)
                # are not deduplicated — each call is unique
                suffix = self.intent_id[:8]
            self.idempotency_key = (
                f"{self.strategy_id}:{self.symbol}:"
                f"{self.intent_type.name}:{trade_date}:{suffix}:{qty_part}"
            )

    def validate(self) -> tuple[bool, str]:
        """Validate intent fields. Returns (valid, error_message)."""
        if not self.symbol:
            return False, "symbol required"
        if not self.strategy_id:
            return False, "strategy_id required"
        if self.intent_type in (IntentType.ENTER, IntentType.REDUCE):
            if self.desired_qty is None and self.target_qty is None:
                return False, "desired_qty or target_qty required"
        # Expiry enforcement
        if self.constraints.expiry_ts is not None:
            if time.time() > self.constraints.expiry_ts:
                return False, "intent expired"
        return True, ""


@dataclass
class IntentResult:
    """Result returned by OMS after processing an intent."""
    intent_id: str
    status: IntentStatus
    message: str = ""
    modified_qty: Optional[int] = None
    order_id: Optional[str] = None
    cooldown_until: Optional[float] = None
