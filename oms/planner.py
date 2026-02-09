"""
Order Planner: Convert approved intents to executable order plans.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional
import uuid


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP_LIMIT = auto()
    MARKETABLE_LIMIT = auto()


@dataclass
class OrderPlan:
    """Executable order plan."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""
    qty: int = 0
    order_type: OrderType = OrderType.LIMIT

    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Timeouts
    submit_by: Optional[float] = None
    cancel_after: Optional[float] = None

    # Attribution
    intent_ids: List[str] = field(default_factory=list)
    strategy_id: str = ""

    # Constraints
    max_chase_bps: float = 30.0

    created_at: datetime = field(default_factory=datetime.now)


class OrderPlanner:
    """
    Order planner: converts intents to order plans.

    Applies execution policies based on strategy and urgency.
    """

    def __init__(self):
        pass

    def create_plan(
        self,
        symbol: str,
        side: str,
        qty: int,
        intent: 'Intent',
        current_price: float,
    ) -> OrderPlan:
        """
        Create order plan from intent.

        Applies execution policy based on strategy/urgency.
        """
        from .intent import Urgency

        plan = OrderPlan(
            symbol=symbol,
            side=side,
            qty=qty,
            intent_ids=[intent.intent_id],
            strategy_id=intent.strategy_id,
        )

        if intent.constraints.stop_price and side == "BUY":
            # Stop-limit for breakout entries
            plan.order_type = OrderType.STOP_LIMIT
            plan.stop_price = intent.constraints.stop_price
            plan.limit_price = intent.constraints.limit_price or (
                intent.constraints.stop_price * 1.003
            )
            plan.cancel_after = 30.0

        elif intent.urgency == Urgency.HIGH:
            # Marketable limit for urgent orders
            plan.order_type = OrderType.MARKETABLE_LIMIT
            if side == "BUY":
                plan.limit_price = current_price * 1.002
            else:
                plan.limit_price = current_price * 0.998
            plan.cancel_after = 10.0

        else:
            # Standard limit
            plan.order_type = OrderType.LIMIT
            plan.limit_price = intent.constraints.limit_price or current_price
            plan.cancel_after = 120.0

        return plan

    def create_exit_plan(
        self,
        symbol: str,
        qty: int,
        strategy_id: str,
        intent_id: str,
        urgency: 'Urgency',
    ) -> OrderPlan:
        """Create market exit plan."""
        return OrderPlan(
            symbol=symbol,
            side="SELL",
            qty=qty,
            order_type=OrderType.MARKET,
            intent_ids=[intent_id],
            strategy_id=strategy_id,
            cancel_after=5.0,
        )
